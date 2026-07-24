# F3 KV Cache 独立验证工程

本目录实现 Qwen2.5-0.5B Attention 数据通路的 F3 KV Cache，不覆盖 F2 `rope_qk_layer0` 或更早阶段的任何工程和位流。

验证对象：

```text
模型层数：28
KV heads：2
head_dim：64
K：RoPE 后 [2,64] signed int64 Q28
V：F1 输出 [2,64] signed int64 Q28
DDR3：1 GiB，用户侧 256 bit AXI
硬件上下文上限：16384 token
```

## 1. 容量选择

每个 K 或 V 向量包含 `2×64=128` 个 signed int64：

```text
单个 K = 128 × 8 B = 1024 B
单个 V = 128 × 8 B = 1024 B
单 token K+V = 2048 B = 64 个 256 bit AXI beat
```

若完整支持模型标称的 32768 positions，仅 KV Cache 就需要：

```text
28 × 32768 × 2048 B = 1792 MiB
```

超过板载 1 GiB DDR3。因此 F3 第一版保留低端 128 MiB 给模型权重、激活和临时缓冲，将高端 896 MiB 全部分配给 KV Cache：

```text
28 × 16384 × 2048 B = 896 MiB
```

所以硬件上下文上限确定为 16384 token。模型 RoPE 仍可生成 32768 个位置，但当前 1 GiB 单板 KV Cache 只保存前 16384 个位置。

需要明确：剩余低端 128 MiB 无法同时常驻约 251.63 MiB 的完整 INT4 模型。后续完整模型集成必须采用权重流式加载/按层重载，或降低上下文长度后重新平衡权重区与 KV 区；F3 当前验证的是 KV Cache 地址与读写能力，不代表完整模型和 16384-token Cache 已经能同时常驻 DDR3。

## 2. DDR3 地址布局

### 字节地址

```text
KV_BASE_BYTES      = 0x08000000 = 128 MiB
LAYER_STRIDE_BYTES = 0x02000000 = 32 MiB
TOKEN_STRIDE_BYTES = 0x00000800 = 2048 B
V_OFFSET_BYTES     = 0x00000400 = 1024 B

K_byte(layer, position)
  = 0x08000000 + layer × 0x02000000 + position × 0x00000800

V_byte(layer, position)
  = K_byte(layer, position) + 0x00000400
```

### DDR3 Controller 地址

Controller 地址单位为 32 bit，因此全部字节地址除以 4：

```text
K_ctrl(layer, position)
  = 0x02000000 + layer × 0x00800000 + position × 0x00000200

V_ctrl(layer, position)
  = K_ctrl(layer, position) + 0x00000100
```

边界：

```text
首槽：layer=0,  position=0
K_byte = 0x08000000

末槽：layer=27, position=16383
K_byte = 0x3FFFF800
V_byte = 0x3FFFFC00
slot_end = 0x40000000 = 1 GiB
```

层内相邻 token 槽严格首尾相接；每层最后一个槽与下一层首槽严格首尾相接；最后一个槽恰好结束于 DDR3 容量边界，不发生层间或 token 间覆盖。

## 3. Token 槽格式

每个 token 固定占 2048 B：

```text
offset 0x000..0x3FF：K[2,64]，head-major，signed int64 little-endian Q28
offset 0x400..0x7FF：V[2,64]，head-major，signed int64 little-endian Q28
```

K 与 V 各 32 个 256 bit beat。硬件写入当前 token 后自动将 `current_position` 加 1。

历史读取支持一次返回连续 `1..16` 个 token。每 16 个 256 bit beat 发起一个 AXI burst，长读取自动分段，UART 返回顺序始终为：

```text
K(token0), V(token0), K(token1), V(token1), ...
```

## 4. UART 协议

串口：115200 8N1。

### 信息

```text
发送：I
返回：PANGU50K KV CACHE V1\r\n
```

### 状态

```text
发送：S
返回：'S' + flags + layer_u8 + start_u16_le + current_u16_le
      + written_u16_le + CRLF
```

`flags`：

```text
bit0 DDR3 ready
bit1 configured
bit2 write_valid
bit3 read_valid
bit5 context_full
bit6 protocol_error
```

### 配置

```text
发送：'C' + layer_u8 + start_position_u16_le
返回：K\r\n
```

有效范围：

```text
0 <= layer < 28
0 <= start_position < 16384
```

### 写入当前 token

```text
发送：'W' + 2048 B token payload
返回：'K' + layer_u8 + written_position_u16_le + CRLF
```

写入成功后 `current_position += 1`。当其达到 16384 时置 `context_full`，下一次 `W` 返回错误 `0x05`，不会接收或写入载荷。

### 历史顺序读取

```text
发送：'R' + start_position_u16_le + count_u8
返回：'D' + layer_u8 + start_position_u16_le + count_u8
      + count × 2048 B
```

`count` 有效范围为 `1..16`，且 `start_position + count <= 16384`。读取地址只由当前配置的 layer 和请求 position 决定，因此可重新配置 layer 后回读此前保存的数据，用于层间隔离验证和后续 Attention 调度。

### 复位写指针

```text
发送：Z
返回：K\r\n
```

将 `current_position` 恢复为最近一次配置的起点，并清零本次会话计数，不清除 DDR3 中已有 K/V 数据。

### 错误帧

```text
'E' + error_code + CRLF
```

主要错误码：

```text
0x01 未知命令
0x02 DDR3 未初始化
0x03 尚未配置
0x04 layer/position 配置越界
0x05 上下文已满
0x06 历史读取范围越界
0xFF 状态机异常
```

## 5. 关键文件

```text
model_tools/kv_cache_reference.py        地址、容量、真实 K/V 和软件金标准
model_tools/kv_cache_reference.json      四个真实固定槽清单和 SHA256
model_tools/test_kv_cache_reference.py   地址、边界、载荷和真实清单单元测试
kv_cache_f3/rtl/kv_cache_ctrl.v          UART、地址计算、DDR3 写入和历史读取控制器
kv_cache_f3/rtl/kv_cache_top.v           DDR3 Controller + PHY 独立顶层
kv_cache_f3/pnr/build_kv_cache.tcl       PDS 全流程构建脚本
kv_cache_f3/pnr/program_sram.tcl         仅 JTAG 下载易失性 SRAM
tools/pangu_kv_cache_host.py              软件自检、固定、隔离和随机上板工具
```

## 6. 软件验证

```bash
python -m unittest model_tools.test_kv_cache_reference
python -m unittest discover -s model_tools -p "test_*.py"
python tools/pangu_kv_cache_host.py selftest --rounds 1000 --seed 20260801
```

结果：

```text
F3 新增单元测试：9/9 PASS
完整 model_tools 回归：64/64 PASS
地址/载荷软件随机压力：1000/1000 PASS，seed=20260801
```

固定真实用例：

```text
layer=0,  position=0
layer=0,  position=1
layer=13, position=2026
layer=27, position=16383
```

其中 K 来自 F2 RoPE 后真实 layer0 K，V 来自 F1 真实 layer0 V。固定 payload SHA256：

```text
layer0/position0    11bbaa5adb6b3250404bd508bc46e10d80bedba04d1b2440f9744b882fa169ca
layer0/position1    86e518b08be537acf51e33b6cc75abaadf509f199a094ec743e8aa38b2ab1ec4
layer13/position2026 95aec9a82a8daf2cee5c872f47efc93b868dee01d335a15b1a35e36b36804c20
layer27/position16383 86c2fa83fa0ffd31cf943728f0a0e5fc1e92349f043b2db5ad7b7b27ea4dd9ac
```

## 7. PDS 结果

目标器件：`PGL50H-6IFBG484`，核心时钟 100 MHz，seed=`5/11`。

```text
编译：成功
综合：成功
Device Map：成功
布局布线：成功
未布线网络：0
时序：All Constraints Met
位流生成：成功
```

Device Map 资源：

```text
LUT：7572 / 42800（17.69%）
FF：9884 / 64200（15.40%）
Distributed RAM：70
DRM18K：0
APM：0
```

多角时序：

```text
慢角 100 MHz core setup WNS = +1.781 ns，TNS = 0
慢角 core hold WHS          = +0.171 ns，THS = 0
快角 core setup WNS         = +4.142 ns，TNS = 0
快角 core hold WHS          = +0.100 ns，THS = 0
恢复、移除、最小脉宽：无违例
```

位流：

```text
kv_cache_f3/pnr/generate_bitstream/kv_cache_top.sbit
大小：2101696 B
SHA256：11a0240a2ee42f0c92b6a5919f4a4b71ceb7bb806b55f1810b4ef3ff88d23216
```

## 8. 真实上板结果

JTAG 仅下载 FPGA 易失性 SRAM：100%，`done bit=1`，未操作 Flash。

```text
固件：PANGU50K KV CACHE V1
DDR3 初始化：成功
```

固定与边界：

```text
layer0 position 0..1 连续真实 K/V：逐位一致
current_position 自动推进：PASS
layer13 position 2026：逐位一致
layer27 position 16383：逐位一致
最后槽结束于 1 GiB：PASS
下一 token 写入返回 0x05：PASS
固定测试耗时：1.66 秒
```

层间隔离：

```text
layer 3 和 layer 17 在相同 position=4096 写入不同 2048 B 载荷
重新配置并分别回读：两层均逐位一致，互不覆盖
```

随机压力：

```text
随机层、随机 position、每批 1..16 个连续 token
随机 signed int64 全位型 K/V
每批写确认、自动位置推进、历史顺序读取逐字节比较
周期性跨配置回读旧层旧位置，验证后续写入未覆盖
结果：300/300 token PASS
seed：20260801
耗时：124.41 秒
```

F3 的五项任务均已形成软件参考、PDS、多角时序和真实上板闭环。下一阶段为 F4 Attention Score：Q·K 点积、`1/sqrt(head_dim)` 缩放、causal mask 和多头循环调度。
