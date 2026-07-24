# K=896 signed Q6.10 元素级算子工程

本目录是阶段 E2 的独立验证工程，不覆盖任何已有 GEMV、真实 Linear 或 RMSNorm 工程与位流。

工程目标是建立 Qwen2 Transformer 后续残差连接和 MLP 所需的四种 K=896 元素级操作：

```text
0. residual_add:      y = saturate_int16(a + b)
1. fixed_scale:       y = saturate_int16(rne((a * scale) / 2^10))
2. elementwise_mul:   y = saturate_int16(rne((a * b) / 2^10))
3. silu_pwl64:        y ≈ x * sigmoid(x)
```

输入 A、输入 B、标量 scale、SiLU 端点和输出均采用 signed Q6.10 int16。

## 1. 定点规则

### 1.1 数据格式

| 数据 | 格式 | 位宽 |
|---|---|---:|
| 输入 A/B | signed Q6.10 | 16 bit |
| 标量 scale | signed Q6.10 | 16 bit |
| 残差中间值 | signed integer | 18 bit |
| 乘法中间值 | signed Q12.20 | 32 bit |
| SiLU 端点 | signed Q6.10 | 16 bit |
| 输出 | signed Q6.10 | 16 bit |

### 1.2 舍入和饱和

所有定点右移使用 round-to-nearest-even（RNE）：

- 小于半个 LSB：向零方向保留；
- 大于半个 LSB：向远离零方向加一；
- 恰好半个 LSB：使保留结果最低位为偶数；
- 正负数使用对称规则。

最终结果显式饱和到 signed int16：

```text
value >  32767 ->  32767
value < -32768 -> -32768
```

## 2. SiLU 方案比较

软件参考在完整 65536 个 signed int16 输入上比较两种候选方案。两种方案均覆盖 `[-8, 8)`，区间外采用：

```text
x < -8  -> 0
x >= 8  -> x
```

| 方案 | 表项 | 表容量 | 最大误差 | 平均误差 | 计算代价 |
|---|---:|---:|---:|---:|---|
| 2048 项中点直接 LUT | 2048×16 bit | 32768 bit | 5 Q10 LSB | 0.352692 LSB | 边界判断 + ROM |
| 64 段端点 PWL | 65×16 bit | 1040 bit | 4 Q10 LSB | 0.232300 LSB | 小位宽乘法 + RNE + 加法 |

第一版选择 **64 段端点分段线性 PWL**。每段宽度为 `0.25`，即 Q6.10 中的 256：

```text
segment = (x_q10 + 8192) >> 8
fraction = (x_q10 + 8192) & 0xff
y = endpoint[segment]
  + rne((endpoint[segment+1] - endpoint[segment]) * fraction / 256)
```

RTL 将乘法、RNE 和端点加法拆成独立寄存级，避免长进位链影响 100 MHz 时序。

## 3. 硬件结构

```text
UART 上传 A/B/PWL 端点
→ DDR3 固定地址保存
→ AXI burst 读取并装入片上缓存
→ 逐拍读取 16 个元素
→ 逐 lane 执行所选操作
→ 16 个 Q6.10 结果打包为 256 bit
→ 写回 DDR3
→ DDR3 回读
→ UART 返回 896 个 int16
→ Python 逐元素比较
```

K=896 共 56 个 256-bit 数据拍。A/B 各使用 8 个 DRM18K 宏中的一半组合，共计 8 个 DRM；PWL 端点使用小型分布式存储。

## 4. DDR3 地址布局

控制器地址按 256-bit 数据拍编址，每拍地址增加 8。

| 区域 | 控制器地址 | 数据量 | 数据拍 |
|---|---:|---:|---:|
| 输入 A | `0x0000` | 1792 B | 56 |
| 输入 B | `0x1000` | 1792 B | 56 |
| SiLU PWL 端点 | `0x2000` | 160 B | 5 |
| 输出 | `0x3000` | 1792 B | 56 |

PWL 实际有 65 个 int16 端点，共 130 B；UART 载荷补零到 5 个完整数据拍，即 160 B。

## 5. UART 协议

串口参数：`115200 8N1`。

### 5.1 固件信息

主机发送：

```text
I
```

FPGA 返回：

```text
PANGU50K ELEMENTWISE K896 V1\r\n
```

### 5.2 状态

主机发送 `S`，FPGA 返回：

```text
'S' + flags + '\r\n'
```

flags：

| bit | 含义 |
|---:|---|
| 0 | DDR3 初始化完成 |
| 1 | 数据已加载 |
| 2 | 结果有效 |
| 3 | 计算核心忙 |
| 4 | 操作配置有效 |

### 5.3 操作配置

主机发送：

```text
'C' + op(uint8) + scale_q10(int16 little-endian)
```

操作编号：

| op | 操作 |
|---:|---|
| 0 | 残差加法 |
| 1 | 定点缩放 |
| 2 | 元素级乘法 |
| 3 | SiLU PWL64 |

配置成功返回 `K\r\n`。

### 5.4 载荷上传

主机发送：

```text
'L' + 3744 B
```

载荷顺序：

```text
input_a_q6_10[896]       1792 B
input_b_q6_10[896]       1792 B
silu_pwl_q6_10[80]        160 B
```

其中 PWL 只使用前 65 项。写入完成返回 `K\r\n`。

### 5.5 启动和结果

主机发送 `G`，FPGA 返回：

```text
'R' + result_q6_10[896]
```

结果为 1792 B little-endian signed int16。

错误帧格式：

```text
'E' + error_code + '\r\n'
```

## 6. 软件参考和固定向量

关键文件：

| 文件 | 作用 |
|---|---|
| `../model_tools/elementwise_fixed_reference.py` | 四种操作和 SiLU 候选方案的硬件等价参考 |
| `../model_tools/elementwise_k896_reference.json` | 固定向量、误差、饱和统计和 SHA256 |
| `../model_tools/test_elementwise_fixed_reference.py` | RNE、边界、完整输入域和 1000 轮软件压力测试 |
| `../tools/pangu_elementwise_k896_host.py` | 载荷自检、串口固定向量和随机上板测试 |

运行单元测试：

```bat
python -m unittest model_tools.test_elementwise_fixed_reference
```

运行 1000 轮软件载荷压力：

```bat
python tools\pangu_elementwise_k896_host.py selftest ^
  --rounds 1000 --seed 20260727
```

当前软件证据：

- E2 单元测试：11/11 PASS；
- 完整 `model_tools` 回归：34/34 PASS；
- 软件随机压力：1000/1000 PASS，seed=`20260727`；
- SiLU 完整 int16 输入域：PWL64 最大误差 4 Q10 LSB；
- 固定向量饱和元素数：残差 219、缩放 0、元素乘法 756；
- 固定输出 SHA256：
  - residual：`dd6cf26e917004e52973ee8506bfdc2e403dac2d31e64abba9c6cd4619196dca`
  - fixed scale：`8137acd3e9c983380ef1d024858e88ed54b675791cf416539ca3b03fa9c3455c`
  - elementwise multiply：`f07847b17449eb401324b413b4df7765d14377e9b20c340f48e6dc87112f25aa`
  - SiLU PWL64：`1933e7c436030c00285bffb2def77c70c979b32c041af3833f61fa25825fdbf8`

## 7. PDS 构建

标准完整构建使用已通过多角时序的布局 seed=17、布线 seed=29，并开启保持修复：

```bat
cd elementwise_k896\pnr
D:\Pango\PDS_2022.2-SP6.4\bin\pds_shell.exe ^
  -file build_elementwise_k896.tcl ^
  -project_name elementwise_k896
```

如需在不覆盖标准 PDS 输出的情况下复现同一组种子，可使用独立构建目录：

```bat
cd elementwise_k896\pnr_seed17
D:\Pango\PDS_2022.2-SP6.4\bin\pds_shell.exe ^
  -file run_seed17.tcl ^
  -project_name elementwise_k896_seed17
```

验收必须检查：

- 编译、综合、Device Map、布局布线、时序和位流生成成功；
- 0 条未布线网络；
- 快慢角建立、保持、恢复、移除全部 TNS=0；
- 位流 SHA256 已记录。

### 7.1 开发中发现并修复的问题

- 首版 SiLU 将小乘法、64 位 RNE 和 64 位端点加法放在同一长组合路径，慢角建立时间违例；改为 27 位乘积寄存、19 位 RNE 寄存和 20 位端点加法寄存后修复。
- 首次固定上板时残差、缩放和元素乘法全部逐位通过，SiLU 仅最高 `segment=63` 的 3 个固定元素错误。根因是 6 位 `pwl_index_reg + 1'b1` 在 `63+1` 时回绕到 0，误读端点 0；改为 7 位加法 `pwl_index_reg + 7'd1` 后消除回绕。

## 8. SRAM 下载和真实上板

仅允许通过 JTAG 下载易失性 SRAM，不执行 Flash 擦写或编程。

串口验证命令：

```bat
python tools\pangu_elementwise_k896_host.py --port COM20 info
python tools\pangu_elementwise_k896_host.py --port COM20 status
python tools\pangu_elementwise_k896_host.py --port COM20 fixed
python tools\pangu_elementwise_k896_host.py --port COM20 stress ^
  --rounds 300 --seed 20260727
```

最终验证结果（2026-07-24）：

- PDS 编译、综合、Device Map、布局布线、时序分析和位流生成全部成功；
- 最终未布线网络：0；
- 资源：7872 LUT、7778 FF、70 个 distributed RAM LUT、8 DRM、2 APM；
- 多角时序：`All Constraints Met`；
- 慢角 100 MHz：WNS=`+0.580 ns`、TNS=0，WHS=`+0.112 ns`、THS=0；
- 快角 100 MHz：WNS=`+2.951 ns`、TNS=0，WHS=`+0.051 ns`、THS=0；
- 恢复、移除和最小脉宽均无违例；
- 验证位流：`pnr_seed17/generate_bitstream/elementwise_k896_top.sbit`；
- 位流 SHA256：`809b436f1c369d66a20c5f2faaa8e684a15a3963d659b95d080e342c3a7d9d50`；
- JTAG 仅下载易失性 SRAM，进度 100%，`done bit=1`，未操作 Flash；
- 固件：`PANGU50K ELEMENTWISE K896 V1`，DDR3 初始化成功；
- 固定边界向量：四种操作、每种 896 个输出均与 Python 逐位一致，端到端约 1.01 秒；
- 真实随机上板：分三批完成累计 300/300 PASS，seed=`20260727..20261026`；
- 三批耗时分别为 104.40、104.39、103.70 秒，合计约 312.49 秒。
