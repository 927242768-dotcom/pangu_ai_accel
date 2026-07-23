# 盘古 50K AI 大模型 FPGA 项目进展

> 本文件记录截至 2026-07-23 的历史验证证据。后续任务状态和当前下一步统一以 `PROJECT_ROADMAP.md` 为准。

更新时间：2026-07-23

## 一、`E:\50K` 文件夹概况

`E:\50K` 是盘古 50K 开发板的完整工作目录，主要由开发工具、开发板资料、扩展模块资料和 AI 加速项目组成。

| 目录/文件 | 大小约 | 内容 |
|---|---:|---|
| `PDS开发软件安装包` | 2.6 GiB | Pango Design Suite 安装介质 |
| `AI_LLM_FPGA` | 540 MiB | 当前 AI/LLM FPGA 研发目录 |
| `盘古50K开发板` | 260 MiB | MES50H/MES50HP 开发板手册、例程和相关资料 |
| `MES50HP-Ethernet` | 45 MiB | 以太网扩展资料 |
| `MES50H-HDMI` | 28 MiB | HDMI 扩展资料 |
| `PMOD音频模块组合` | 24 MiB | 音频 PMOD 资料 |
| `ADDA模块资料` | 18 MiB | AD/DA 模块资料 |
| `PCIE资料` | 14 MiB | PCIe 相关资料 |
| `OV5640资料` | 3.6 MiB | 摄像头模块资料 |
| 说明文件、联系方式、视频教程文档 | 数百 KiB | 板卡配套说明和支持信息 |

当前真正持续开发的项目位于：

```text
E:\50K\AI_LLM_FPGA\pangu_ai_accel
```

目标器件：

```text
Pango Logos PGL50H-6IFBG484
```

## 二、项目目录说明

| 目录/文件 | 作用 |
|---|---|
| `source` | 已上板验证的 UART、INT8 MAC16 和原基础顶层 RTL |
| `tools` | Python 上位机、串口验证和自动比较工具 |
| `model_tools` | 模型量化与权重转换脚本 |
| `model_output` | Qwen2.5-0.5B + LoRA 转换后的 INT4 模型文件和元数据 |
| `ipcore/pangu_ddr3_x32` | PGL50H、FBG484、32 位 DDR3 Controller + PHY 工程 |
| `ddr3_selftest` | 完整 1 GiB DDR3 全地址顺序写读与地址相关数据 BIST |
| `ddr_mac16_integration` | 本次新建的 DDR3 + MAC16 + INT4 解包集成验证工程 |
| 根目录 PDS 输出目录 | 早期 MAC16 工程的编译、综合、布局布线和位流结果 |

模型文件：

```text
model_output\yanbo_qwen25_0.5b_int4.p50
大小：263,857,920 字节，约 251.63 MiB
```

该文件已经完成转换，但尚未进行完整模型分层加载与文本推理。

## 三、此前已经完成并真实上板验证的能力

### 1. INT8 MAC16

- 固件信息读取。
- 自检。
- 16 维 INT8 向量点积。
- 多轮随机压力测试。
- Python 参考结果自动比较。

### 2. 完整 1 GiB DDR3

- 使用正确的 PGL50H、FBG484、32 位 DDR3 Controller + PHY。
- DDR3 初始化与训练成功。
- 完整 1 GiB 地址空间顺序写入、读回和地址相关数据校验。
- 编译、综合、布局布线和多角时序通过。
- 已验证位流：

```text
ipcore\pangu_ddr3_x32\pangu_ddr3_x32\pnr\generate_bitstream\test_ddr.sbit
```

- JTAG 下载到 FPGA SRAM 后，串口状态：

```text
test_main_state=5（PASS）
err_cnt=0
```

## 四、本次新完成：DDR3 + MAC16 + INT4 集成闭环

新工程：

```text
ddr_mac16_integration
```

### 数据通路

```text
Python上位机
  → UART发送激活与权重
  → FPGA写入DDR3
  → 一次2拍×256 bit AXI burst读取
  → 片上寄存缓冲与数据拆分
  → INT8直接输入或packed INT4解包/符号扩展
  → MAC16点积
  → 32位结果写回DDR3
  → UART返回
  → Python自动比较
```

### INT8 闭环

- 16 个 INT8 激活。
- 16 个 INT8 权重。
- DDR3 写入、2 拍 256 bit burst 读回、MAC16、结果回写与返回。
- 固定向量：FPGA 272，Python 272，PASS。
- 随机压力测试：1000/1000 PASS。

### INT4 × INT8 闭环

- 16 个 INT8 激活。
- 16 个有符号 INT4 权重压缩为 8 字节。
- 每字节低半字节为偶数下标权重，高半字节为奇数下标权重。
- FPGA 对 INT4 二补码进行符号扩展，转换成 INT8 后复用 MAC16。
- 固定向量：FPGA 272，Python 272，PASS。
- 随机压力测试：1000/1000 PASS。

### 流水与时序修复

首个 INT4 版本把“解包/选择”和“MAC16”放在同一周期，100 MHz 下出现约 `-1.31 ns` 建立时间违例。随后加入 MAC 输入流水寄存器，将其拆成两个周期。

最终结果：

```text
Design Summary : All Constraints Met.
```

关键时序：

- 100 MHz `ddrphy_clkin` 慢速角建立 WNS：`+0.841 ns`，TNS：0。
- 慢速角保持 WHS：`+0.171 ns`，THS：0。
- 快速角建立 WNS：`+3.210 ns`。
- 快速角保持 WHS：`+0.100 ns`。

### 最终位流

```text
ddr_mac16_integration\pnr\generate_bitstream\ddr_mac16_top.sbit
SHA256: e625e6dbe0e7f49915b41be805a970ea3977a72a6cb189f98c50497371b0af9f
```

JTAG 实测：

- 识别 `PANGO USB CABLE II`。
- 识别 `PGL50H`。
- 下载进度 100%。
- `done bit=1`。
- 当前只写入易失性 FPGA SRAM，没有擦写 Flash。

## 五、当前项目状态

现在已经不是“DDR3 和 MAC16 各自单独工作”，而是完成了第一条真实的计算闭环：

```text
DDR3取数 → INT4/INT8处理 → MAC16计算 → DDR3结果回写 → 上位机校验
```

这证明以下基础环节可以协同工作：

- DDR3 Controller + PHY。
- 256 bit AXI burst。
- 片上数据缓存。
- INT4 权重解包与符号扩展。
- INT8 MAC16。
- 状态机调度。
- UART 数据传输。
- Python 自动验证。

但目前仍然只是长度 16 的单次点积，不是完整大模型推理。

## 六、下一阶段路线

### 最近一步：矩阵向量乘法 `y = W × x`

1. 在 DDR3 中连续保存多行 packed INT4 权重。
2. 激活向量只加载和读取一次。
3. AXI 连续 burst 读取多行权重。
4. MAC16 对每一行循环调度。
5. 生成完整输出向量并批量写回 DDR3。
6. Python 对整个输出向量逐元素比较。

### 后续算子

按以下顺序逐步实现并逐层验证：

1. 分块 GEMV 与量化缩放。
2. RMSNorm。
3. Q/K/V 线性层。
4. RoPE。
5. Attention。
6. MLP。
7. Transformer Block。
8. 完整模型权重加载与分层调度。
9. tokenizer、采样与文本推理验证。

每一步都应保留“FPGA 结果与 Python 参考逐元素自动比较”的闭环，避免直接跳到完整模型后难以定位错误。
