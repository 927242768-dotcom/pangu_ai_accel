# PGL50H 真实 tied Embedding K=896 验证工程

## 1. 工程目标

本工程完成 Qwen2.5-0.5B 真实 tied Embedding 的独立 FPGA 闭环：

```text
Token ID
→ 计算 DDR3 行槽地址
→ 读取一行 packed INT4 权重和 14 个分组 scale
→ INT4 × UQ4.28
→ RNE 转 signed Q6.10
→ 896 个 int16 结果写回 DDR3
→ UART 返回
→ Python 逐元素比较
```

工程位于独立目录 `embedding_k896`，没有覆盖任何已经验证的 GEMV、Linear、RMSNorm 或元素级工程及位流。

## 2. 真实模型对象

- 张量：`model.embed_tokens.weight`
- shape：`[151936, 896]`
- storage：`int4_groupwise_symmetric`
- group size：64
- 每行 group 数：14
- tied embedding：是
- Token ID 有效范围：`0..151935`

P50 中每行按 row-major 保存 448 B packed signed INT4；每个 64 元素 group 配一个 FP16 正 scale。

## 3. 定点格式

- 权重：signed INT4，范围 `[-7, 7]`，每字节低半字节保存较小列号。
- scale：FP16 在主机端无损转换为 unsigned UQ4.28 `uint32`。
- 乘积：signed INT4 × unsigned UQ4.28。
- 输出：RNE 右移 18 位得到 signed Q6.10，再显式饱和到 int16。
- 真实 embedding 全部 FP16 scales 均可被 UQ4.28 精确表示。
- 硬件固定路径与 `round_to_nearest_even(INT4 * FP16_scale * 2^10)` 逐位一致。

软件 1000 个真实随机 Token 行的最大 Q6.10 量化误差为 `0.00048828125`，即不超过 0.5 个 Q10 LSB；未发生输出饱和。

## 4. DDR3 地址布局

DDR3 Controller 地址单位为 32 bit；一个 256 bit 数据拍占 8 个控制器地址单位。

每个 Token ID 使用固定 512 B 行槽：

```text
row_base_ctrl_addr = token_id * 128 = token_id << 7
```

| 行槽内容 | 字节 | 256-bit 拍 |
|---|---:|---:|
| packed INT4 权重 | 448 B | beat 0..13 |
| 14 个 UQ4.28 scale | 56 B | beat 14 和 beat 15 的前 24 B |
| padding | 8 B | beat 15 的后 8 B |

- Token 0 行槽：控制器地址 `0x0000000`
- Token 151935 行槽：控制器地址 `0x128bf80`
- 最大行槽末端仍位于完整 1 GiB DDR3 有效范围内。
- 输出固定写入控制器地址 `0x02000000`，共 56 拍、1792 B。

## 5. UART 协议 V1

串口参数：`115200, 8N1`。

| 命令 | 请求 | 回复 |
|---|---|---|
| `I` | 固件信息 | `PANGU50K EMBEDDING K896 V1\r\n` |
| `S` | 状态查询 | `S + flags + \r\n` |
| `C` | `token_id(uint32_le)` | `K\r\n` |
| `L` | 当前 Token 的 512 B 行槽 | `K\r\n` |
| `G` | 按 Token 地址读取、计算、写回和回读 | `R + 896个int16_le` |

状态 flags：

- bit0：DDR3 初始化完成。
- bit1：当前 Token 行已加载。
- bit2：结果有效。
- bit3：计算核心忙。
- bit4：Token ID 已配置。

错误码：

- `0x01`：未知命令。
- `0x02`：DDR3 尚未初始化。
- `0x03`：Token ID 越界。
- `0x04`：尚未配置 Token ID。
- `0x05`：尚未加载当前 Token 行。
- `0xff`：状态机异常。

## 6. 主要文件

| 文件 | 作用 |
|---|---|
| `rtl/embedding_k896_core.v` | 14 组权重/scale 缓存、INT4×UQ4.28、RNE、饱和和结果打包 |
| `rtl/embedding_k896_ctrl.v` | UART 协议、Token 地址映射、DDR3 行读写和结果回读调度 |
| `rtl/embedding_k896_top.v` | DDR3 IP、UART、LED 与计算控制器顶层 |
| `pnr/build_embedding_k896.tcl` | PDS 编译、综合、布局布线、时序和位流生成 |
| `pnr/program_sram.tcl` | 仅下载 FPGA 易失性 SRAM，不操作 Flash |
| `../model_tools/embedding_fixed_reference.py` | 真实 P50 Embedding 软件与硬件等价金标准 |
| `../model_tools/test_embedding_fixed_reference.py` | 地址、RNE、饱和、载荷和真实随机 Token 测试 |
| `../model_tools/embedding_k896_reference.json` | 四个固定 Token 的输出和载荷 SHA256 清单 |
| `../tools/pangu_embedding_k896_host.py` | 软件自检、固定 Token 和随机 Token 上板验证工具 |

## 7. 构建、下载与验证

在 `embedding_k896/pnr` 目录执行：

```bat
D:\Pango\PDS_2022.2-SP6.4\bin\pds_shell.exe -file build_embedding_k896.tcl -project_name embedding_k896
```

仅下载到 FPGA SRAM：

```bat
D:\Pango\PDS_2022.2-SP6.4\bin\cdt_cfg_shell.exe -file program_sram.tcl -work_dir .
```

软件与上板验证：

```bat
python -m unittest model_tools.test_embedding_fixed_reference -v
python tools\pangu_embedding_k896_host.py selftest --rounds 1000 --seed 20260728
python tools\pangu_embedding_k896_host.py --port COM20 info
python tools\pangu_embedding_k896_host.py --port COM20 status
python tools\pangu_embedding_k896_host.py --port COM20 fixed
python tools\pangu_embedding_k896_host.py --port COM20 stress --rounds 300 --seed 20260728
```

## 8. 2026-07-24 验证结果

### 软件参考

- E3 单元测试：11/11 PASS。
- 完整 `model_tools` 回归：45/45 PASS。
- 真实 P50 软件/载荷压力：1000/1000 PASS，seed=`20260728`。
- 四个固定 Token ID：`[0, 1, 2026, 151935]`。
- 固定路径与直接 Q6.10 量化逐位一致。

### PDS 实现与时序

- 编译、综合、Device Map、布局布线、时序分析和位流生成：全部成功。
- 最终未布线网络：0。
- `Design Summary : All Constraints Met.`
- 慢角 100 MHz 建立：WNS=`+0.679 ns`、TNS=`0`。
- 慢角保持：WHS=`+0.172 ns`、THS=`0`。
- 快角建立：WNS=`+2.964 ns`、TNS=`0`。
- 快角保持：WHS=`+0.101 ns`、THS=`0`。
- 恢复、移除和最小脉宽：所有角均 0 违例。
- 资源：LUT=`7637`、FF=`7380`、distributed RAM=`326`、APM=`2`、DRM=`0`。

最终位流：

```text
embedding_k896/pnr/generate_bitstream/embedding_k896_top.sbit
SHA256: cd0e138e494875035cf5c66d76eaf250729625c172bf51c935b831d31c45c0fa
```

### 真实上板

- JTAG 识别 `PANGO USB CABLE II` 和 `PGL50H`。
- SRAM 下载进度 100%，`done bit=1`，未操作 Flash。
- 固件：`PANGU50K EMBEDDING K896 V1`。
- DDR3 初始化成功。
- 四个固定 Token 的 896 个输出均与 Python 逐位一致，总耗时约 0.93 秒。
- 随机 Token ID 真实上板压力：300/300 PASS，seed=`20260728`，耗时约 75.53 秒。

固定输出 SHA256：

| Token ID | 输出 SHA256 |
|---:|---|
| 0 | `fd5e2ad5a2f7861324e911c48e0a7503d732c39a000fd96a500cf92410006a74` |
| 1 | `26d13ee91a84f894232f60a63d0deda878628d2dc3e2a0de409c5038630e6a1b` |
| 2026 | `d4348c1a5a4375a15c2325f07d8336efb2fc2c2efc39b79ba2997cdc6175cdc8` |
| 151935 | `b10edc1533867b2fa584fe03b2d0f782b15433afbbb85b62da1aa241c9c9be97` |

E3 Embedding/查表阶段已经形成完整软件、PDS、时序和真实上板闭环。
