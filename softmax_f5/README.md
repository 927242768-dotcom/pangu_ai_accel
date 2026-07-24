# F5 Softmax 独立验证工程

本目录只验证 `PROJECT_ROADMAP.md` 中的 F5：

- 直接消费 F4 已验证的 `[14,16]` signed int64 Q28 Attention Score；
- mask 感知的每 head 最大值归约与减最大值；
- 定点 exp 近似、求和、倒数和概率归一化；
- 全 mask、单有效 token、部分窗口、16 token 满窗口和极端 score 差值；
- 概率结果逐位回读。

本工程**不包含 V 加权和、Attention 输出或 Transformer Block 调度**，不会覆盖 F4 及更早阶段的工程和位流。

## 1. 定点定义

输入：

```text
scores[14][16]
布局：head-major
格式：signed int64 Q28
mask：INT64_MIN = 0x8000000000000000
```

每个 head 独立执行：

1. max reduction 忽略 mask 哨兵；
2. 有效 score 减去最大值，因此差值恒不大于 0；
3. exp 使用 `[-16,0]`、步长 `1/32` 的 513 点端点 LUT；
4. 区间内执行线性插值，差值小于 `-16` 时 exp 置 0；
5. exp 输出为 unsigned UQ1.31 uint32；
6. 最多 16 项 exp 使用 36 位无符号和；
7. 每个 head 计算一次 `reciprocal_q31 = RNE(2^62 / sum_exp_q31)`；
8. 概率为 `RNE(exp_q31 * reciprocal_q31 / 2^31)`；
9. 概率显式限制在 `[0, 1.0]`，其中 `1.0 = 0x80000000`。

边界规则：

- mask 槽概率严格为 0；
- 全 mask head 输出全 0；
- 单有效 token 精确输出 `0x80000000`；
- 相同的 16 个有效 score 精确输出 16 个 `0x08000000`。

## 2. DDR3 地址布局

DDR3 Controller 地址单位为 32 bit，一个 256 bit 数据拍占 8 个地址单位。

```text
scores        : ctrl 0x0000800，1792 B / 56 beats
probabilities : ctrl 0x0000A00， 896 B / 28 beats
exp LUT       : ctrl 0x0000B00，2080 B / 65 beats
```

exp LUT 原始数据为 513 个 little-endian uint32，共 2052 B；上传时补齐到 2080 B，最后 28 B 必须为 0。

## 3. 文件

```text
softmax_f5/
├── README.md
├── rtl/
│   ├── softmax_core.v
│   ├── softmax_ctrl.v
│   └── softmax_top.v
└── pnr/
    ├── build_softmax.tcl
    └── program_sram.tcl

model_tools/
├── softmax_fixed_reference.py
├── softmax_f5_reference.json
└── test_softmax_fixed_reference.py

tools/
└── pangu_softmax_host.py
```

## 4. 软件金标准

运行 F5 单元测试：

```bash
python -m unittest model_tools.test_softmax_fixed_reference -v
```

运行固定真实用例和 1000 轮软件随机压力：

```bash
python model_tools/softmax_fixed_reference.py verify \
  --rounds 1000 \
  --seed 20260803
```

统一上位机软件自检：

```bash
python tools/pangu_softmax_host.py selftest \
  --rounds 1000 \
  --seed 20260803
```

固定真实用例直接使用 F4 的四组 score：

- layer 0，query 0，单有效 token；
- layer 0，query 1，连续 2 token；
- layer 13，query 2026，窗口 2023..2028，含未来位置 mask；
- layer 27，query 16383，16 token 满窗口和硬件上下文末边界。

## 5. RTL 结构

`softmax_core.v` 的主要调度：

```text
DDR3 burst 读取 score 与 exp LUT
→ 每 head 读取 16 个 score
→ mask 感知 max reduction
→ PWL exp 与 36 位求和
→ 63 周期 restoring divider 计算 Q31 倒数
→ 逐 token 概率乘法、RNE 与饱和
→ 32 位局部 byte-enable 写回 DDR3
```

倒数使用顺序恢复除法器，避免综合出长组合除法器；score 和 LUT 缓存采用同步 RAM 读结构，目标是推断 DRM18K。

## 6. PDS 构建

在 `softmax_f5/pnr` 目录运行：

```bash
D:/Pango/PDS_2022.2-SP6.4/bin/pds_shell.exe \
  -file build_softmax.tcl \
  -project_name softmax_f5
```

验收必须同时满足：

- Compile、Synthesize、Device Map、Place & Route、Timing、Bitstream 全部完成；
- 0 unrouted nets；
- 所有分析角 TNS = 0、THS = 0；
- 位流独立生成于 `softmax_f5/pnr/generate_bitstream/softmax_top.sbit`。

## 7. SRAM 下载

只允许下载到易失性 SRAM：

```bash
D:/Pango/PDS_2022.2-SP6.4/bin/cdt_cfg_shell.exe \
  -file program_sram.tcl \
  -work_dir .
```

`program_sram.tcl` 不包含任何 Flash 擦除或编程命令。

## 8. UART 协议

串口参数：115200、8N1。

```text
I
  -> "PANGU50K SOFTMAX F5 V1\r\n"

S
  -> 'S' + flags + CRLF

L + 1792 B scores
  -> "K\r\n"

T + 2080 B exp LUT
  -> "K\r\n"

G
  -> 计算并写回固定 14x16 probabilities
  -> "K\r\n"

R
  -> 'D' + 896 B probabilities
```

状态 flags：

- bit0：DDR3 初始化完成；
- bit1：scores 已加载；
- bit2：exp LUT 已加载；
- bit3：结果有效；
- bit4：Softmax 核心忙；
- bit5：协议错误。

## 9. 真实板卡测试

```bash
python tools/pangu_softmax_host.py ports
python tools/pangu_softmax_host.py --port COM20 info
python tools/pangu_softmax_host.py --port COM20 status
python tools/pangu_softmax_host.py --port COM20 fixed
python tools/pangu_softmax_host.py --port COM20 stress \
  --rounds 100 \
  --seed 20260803
```

只有固定真实 F4 score、全部边界用例、随机逐位回归、PDS 全流程和所有角时序全部通过，才能在路线图中把 F5 标记为完成。

## 10. 最终验证结果（2026-07-24）

软件验证：

- F5 单元测试：10/10 PASS；
- 完整 `model_tools` 回归：83/83 PASS；
- 软件随机压力：1000/1000 PASS，seed=`20260803`；
- 最坏 float64 概率误差：`3.04973546883e-05`，低于 `2e-4` 阈值。

PDS 结果：

- Compile、Synthesize、Device Map、Place & Route、Timing、Bitstream 全部成功；
- 最终未布线网络：0；
- 资源：10515 LUT、12703 FF、70 个 distributed RAM、12 DRM18K、8 APM；
- 慢角：setup WNS=`+0.227 ns`、TNS=0；hold WHS=`+0.143 ns`、THS=0；
- 快角：setup WNS=`+2.958 ns`、TNS=0；hold WHS=`+0.067 ns`、THS=0；
- 恢复、移除和最小脉宽均无违例；
- 位流：`softmax_f5/pnr/generate_bitstream/softmax_top.sbit`；
- 位流大小：2101696 B；
- SHA256：`d6e505ea5495c6054a447608406db0f93855ef55dbfc357c8d113b00adba34fe`。

真实板结果：

- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；
- 固件：`PANGU50K SOFTMAX F5 V1`；
- DDR3 初始化成功，协议错误标志为 0；
- 四组真实 F4 score 固定窗口全部逐位一致；
- 固定概率 SHA256：
  - `768bd8912f9168473b8978963e805b52b5eeb40b26c517229dc6a4c8d96ce608`；
  - `021ac6fad9854aeede02829734c6afdc3dc9cb41ce79ce078b830a53b695ce81`；
  - `267a18e4d4fef9d1afb118d8f1a025cd9922f14963ae75fe672c30c816e5495f`；
  - `b1ae419016695bb6c2a62ffb1b92c7bcbb70c87b22c1ba6d7cb96e327b201f39`；
- 全 mask、单有效、部分窗口、16 token、全等 score、`-16` 截断边界和随机稀疏 mask 回归 100/100 PASS；
- seed=`20260803`，耗时约 29.05 秒；
- 真实板最坏 float64 概率误差：`2.96390625578e-05`；
- 每个 FPGA uint32 概率均与 Python 金标准逐位一致。

F5 已完成，工程严格停在 Softmax 概率输出；下一任务为 F6 Attention 输出。
