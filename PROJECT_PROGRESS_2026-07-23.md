# 盘古 50K AI 大模型 FPGA 项目进展

> 本文件记录截至 2026-07-24 的历史验证证据。后续任务状态和当前下一步统一以 `PROJECT_ROADMAP.md` 为准。

更新时间：2026-07-24

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

## 五、本次新完成：固定 M=4、K=64 packed INT4 GEMV

新增独立工程：

```text
gemv_int4_m4k64
```

数据通路：

```text
Python生成 W[4,64] INT4 和 x[64] INT8
→ UART写入DDR3
→ 激活以2拍burst读取并缓存一次
→ 4行权重以4拍burst连续读取
→ 每行执行4次MAC16
→ 跨分块INT32累加
→ 生成4个INT32输出
→ 写回DDR3并通过UART返回
→ Python逐元素比较
```

验证结果：

- Python 金标准自检：1000/1000 PASS，seed=`20260725`。
- 固定向量：FPGA `[1376, -1344, 416, 256]`，Python 完全一致。
- 真实上板随机压力测试：1000/1000 PASS，耗时约 19.70 秒。
- PDS 编译、综合、Device Map、布局布线、时序分析和位流生成全部成功。
- 布局布线最终未布线网络：0。
- 多角时序：`Design Summary : All Constraints Met.`
- 100 MHz 慢速角建立 WNS：`+0.983 ns`，TNS：0。
- 慢速角保持 WHS：`+0.171 ns`，THS：0。
- 快速角建立 WNS：`+3.276 ns`；快速角保持 WHS：`+0.100 ns`。
- 位流：`gemv_int4_m4k64\pnr\generate_bitstream\gemv_m4k64_top.sbit`。
- SHA256：`349a26b45362778849868e68475c5b8f6620bc8edb8375ebb237efbab4d352ed`。
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash。

首版曾因“MAC16 乘加树 + 跨分块累加”同周期串联出现慢速角 WNS=`-1.961 ns`、TNS=`-164.261 ns`。增加 MAC 结果寄存级后，时序全部通过。

## 六、本次新完成：运行时参数化 packed INT4 GEMV

新增独立工程：

```text
gemv_int4_param
```

支持范围：

```text
1 <= M <= 64
1 <= K <= 896
```

实现能力：

- UART 在运行时配置 M 和 K。
- 激活按最长 16 拍 AXI burst 自动分段读取，并只缓存一次。
- packed INT4 权重按行 burst 读取，行地址自动递增。
- 每行执行 `ceil(K/16)` 次 MAC16 分块累加。
- K 不是 16 整数倍时，硬件屏蔽尾块无效激活字节和权重半字节。
- 输出每 8 个 INT32 一拍写回 DDR3，输出地址自动递增。
- Python 自动生成不同 M/K 数据并逐元素比较。

验证结果：

- Python 参数化金标准：1025 例全部通过，含标准尺寸、尾块尺寸和固定 M4K64 回归，seed=`20260728`。
- 标准和尾块共 24 种形状，每种 1 个固定例和 2 个随机例，共 72 例真实上板全部通过。
- 标准组合完整覆盖 `M={1,4,16,64}`、`K={16,64,256,896}`。
- 尾块覆盖 `K={1,15,17,63,65,255,257,895}`。
- 固定 M4K64：1000/1000 随机上板通过，seed=`20260730`，约 19.89 秒。
- 尾块 M16K65：1000/1000 随机上板通过，seed=`20260731`，约 105.27 秒。
- 近最大尾块 M4K895：100/100 随机上板通过，seed=`20260801`，约 23.90 秒。
- INT32 边界：FPGA `[917504, -802816, 57344, 57344]`，与 Python 完全一致。
- 当前 K 上限下理论绝对累加上界为 `917504`，远小于 INT32 最大值。
- PDS 编译、综合、Device Map、布局布线、时序分析和位流生成全部成功。
- 布局布线最终未布线网络：0。
- 多角时序：`Design Summary : All Constraints Met.`
- 100 MHz 慢速角建立 WNS：`+0.682 ns`，TNS：0；保持 WHS：`+0.086 ns`，THS：0。
- 快速角建立 WNS：`+3.137 ns`，TNS：0；保持 WHS：`+0.001 ns`，THS：0。
- 资源：10715 LUT、8136 Register、4 个 DRM18K、9 个 APM。
- 位流：`gemv_int4_param\pnr\generate_bitstream\gemv_param_top.sbit`。
- SHA256：`90c67a74841826b358f4a4de5e0783c587de01a296d7991c3b2a8d3fc1bcd2a3`。
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash。

片上激活和单行权重缓存使用同步 RAM 结构，PDS 推断为 4 个 DRM18K。相比首版大寄存向量方案，LUT 从约 23962 降至 10715。

## 七、本次新完成：D1.3 GEMV 性能基础设施

独立构建目录：

```text
gemv_int4_perf
```

新增能力：

- 固件协议升级为 `PANGU50K GEMV PARAM V2`；
- 新增 `P` 命令，返回激活读取、权重读取、MAC 计算和 GEMV 总周期；
- 状态字节 bit5 表示性能计数有效；
- Python 自动计算 DDR3 实测带宽、核心/端到端 GMAC/s、MAC16 利用率和瓶颈分类；
- 原有 `I/S/C/L/G` 协议与 GEMV 结果帧保持兼容。

代表性实测：

| 形状 | 激活读周期 | 权重读周期 | MAC周期 | 总周期 | 合并DDR3带宽 | 核心GMAC/s | 端到端GMAC/s | 端到端利用率 | 主瓶颈 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M4K64 | 32 | 116 | 64 | 244 | 129.73 MB/s | 0.4000 | 0.1049 | 6.56% | DDR3读取 |
| M16K65 | 33 | 480 | 320 | 919 | 218.32 MB/s | 0.3250 | 0.1132 | 7.07% | DDR3读取 |
| M64K896 | 86 | 3152 | 14336 | 17912 | 913.16 MB/s | 0.4000 | 0.3201 | 20.01% | MAC数量/计算 |

验证结果：

- Python 金标准和性能公式：1025 例 PASS，seed=`20260728`；
- 24 种形状、72 例真实上板全部 PASS；
- M4K64：1000/1000 PASS，seed=`20260730`，约 19.79 秒；
- M16K65：1000/1000 PASS，seed=`20260731`，约 105.26 秒；
- M4K895：100/100 PASS，seed=`20260801`，约 23.90 秒；
- INT32 边界 `[917504, -802816, 57344, 57344]` 与 Python 完全一致；
- PDS 全流程成功，0 条未布线网络；
- 资源：10906 LUT、8269 Register、4 个 DRM18K、9 个 APM；
- 多角时序：`All Constraints Met`；慢速角 100 MHz WNS=`+0.589 ns`、TNS=0，WHS=`+0.142 ns`、THS=0；快速角 WNS=`+3.074 ns`、WHS=`+0.065 ns`；
- 位流：`gemv_int4_perf\pnr\generate_bitstream\gemv_param_top.sbit`；
- SHA256：`a727f7427143b874da278ae83d7e8a2cdeff8b82bd7c0bb4361e7a2efed73c35`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash。

性能结论：小尺寸和短 K 主要被 DDR3 命令/返回延迟限制；随着 M、K 增大，DDR3 burst 效率明显提高，最大尺寸的主要瓶颈转为单套 MAC16 每个 16 元素分块需要 4 个核心周期，后续性能优化应优先增加 MAC 并行度或让读取与计算重叠。

## 八、本次新完成：D2 `.p50` 真实模型格式解析

新增文件：

```text
model_tools/p50_format.py
model_tools/p50_inspect.py
model_tools/test_p50_format.py
model_tools/README.md
```

完成能力：

- 解析 48 字节小端固定头和 4096 字节固定头区域；
- 解析镜像内嵌 JSON 张量目录；
- 全量校验张量名称、shape、storage、data/scale 偏移和长度；
- 按 shape 和 group size 重新推导 padded columns、group 数和实际字节数；
- 检查所有张量数据 4 KiB 对齐、所有 scale 64 字节对齐；
- 检查数据范围不越界且互不重叠；
- 将外部 JSON 与镜像内嵌 JSON 逐字段比较；
- 按张量名提取任意 INT4 行、跨 group 二维块或 FP16 行；
- INT4 提取同时返回有符号量化值、相关 FP16 scales 和 FP32 反量化值。

真实镜像解析结果：

```text
文件大小：263,857,920 字节
SHA256：f0c0a22886499715fe16832b88ac59bff48fea8f3069c247437726aca6f19e9d
magic：P50Q4V1\0
version：1
header_size：4096
metadata_size：63716
data_offset：528384
tensor_count：290
group_size：64
```

张量统计：

- 169 个二维分组 INT4 张量；
- 121 个连续 FP16 张量；
- 外部 JSON 与内嵌 JSON 完全一致；
- 全部目录、形状、偏移、长度、对齐和范围检查通过。

真实量化格式：

- 按输出行 row-major；
- 每行输入列按 64 个元素分组；
- 每字节低半字节保存较小列号，高半字节保存下一列；
- 4 位二补码，导出范围 `[-7, 7]`；
- 对称量化，zero point 固定为 0，不保存独立 zero point；
- 每个 `[row, group]` 保存一个 FP16 scale；
- 反量化公式为 `weight = int4_value * scale`。

验证结果：

- 独立微型镜像单元测试：5/5 PASS；
- 真实 q_proj 完整行提取通过；
- 真实 gate_proj 跨 group 二维块提取通过；
- 真实 RMSNorm FP16 行提取通过；
- 原 BF16 模型 + LoRA 软件参考的 4 组抽样反量化误差全部通过理论半 scale 上限检查；
- 本阶段未修改 FPGA RTL、PDS 工程或已验证位流。

## 九、本次新完成：D2 真实 Linear 量化软件参考

新增文件：

```text
model_tools/linear_quant_reference.py
model_tools/test_linear_quant_reference.py
model_tools/q_proj_m4k896_reference.json
```

统一格式：

- 激活采用逐向量对称 INT8，范围 `[-127,127]`，zero point=`0`；
- `activation_scale=max(abs(x))/127`，全零向量使用 scale=`1.0`；
- 浮点转整数统一采用 round-to-nearest-even（RNE），随后饱和；
- 主机预计算 `activation_scale * weight_scale[row,group]`，编码为 32 位无符号 `UQ4.28`；
- 每 64 元素 group 先产生 INT32 点积；
- 分组点积乘 UQ4.28 后，在带 28 位小数的有符号 INT64 中跨组累加；
- bias 同样转换为有符号 Q28 后加入；
- 理论定点误差上界为 `(sum(abs(group_acc)) + 1) * 0.5 / 2^28`。

真实固定向量：

```text
张量：model.layers.0.self_attn.q_proj.weight
bias：model.layers.0.self_attn.q_proj.bias
切片：M=4，K=896，14 个 group
激活生成：32 位 LCG，seed=20260723
激活 scale：0.0314826064222441
```

结果：

- P50 浮点基线：`[0.7752590203, -0.6386315781, 1.0810645018, -0.8347725510]`；
- 量化激活浮点参考：`[0.7720806824, -0.6458171611, 1.0714217223, -0.8315785984]`；
- 定点 Q28：`[207253689, -173360554, 287606739, -223225713]`；
- 定点反量化：`[0.7720801570, -0.6458183900, 1.0714185946, -0.8315805830]`；
- 激活量化最大绝对误差：`0.0096427795`；
- UQ4.28 最大绝对误差：`3.1277186e-6`；
- 理论最大误差上界：`3.8200990e-5`；
- 激活 INT8 饱和数：0；
- UQ4.28 scale 饱和数：0；
- 原有解析与新增量化测试：13/13 PASS；
- 随机软件压力测试：1000/1000 PASS，seed=`20260723`；
- 固定清单记录激活、packed 权重、scale、累加器和输出的 SHA256；
- 完整 NPZ 可由真实 `.p50` 镜像确定性重建；
- 本轮未修改 FPGA RTL、PDS 工程或任何已验证位流。

## 十、2026-07-24 新完成：D2 真实分组 UQ4.28 FPGA 小闭环

新增独立工程与工具：

```text
gemv_int4_group_q28/rtl/int8_dot16_pipe.v
gemv_int4_group_q28/rtl/gemv_group_q28_core.v
gemv_int4_group_q28/rtl/gemv_group_q28_ctrl.v
gemv_int4_group_q28/rtl/gemv_group_q28_top.v
gemv_int4_group_q28/pnr/build_gemv_group_q28.tcl
gemv_int4_group_q28/pnr/program_sram.tcl
gemv_int4_group_q28/README.md
tools/pangu_gemv_group_q28_host.py
```

固定验收对象仍为 layer0 `q_proj` 前 4 行、完整 K=896 输入，共 14 个 64 元素 group。固定 UART 载荷共 2976 B，包含激活、packed INT4 权重、逐行 UQ4.28 combined scale 和 signed int64 bias_q28。

硬件计算流程：

```text
每组 4 次流水 MAC16
→ signed INT32 group 点积
→ signed INT32 × unsigned UQ4.28
→ signed INT64 Q28 跨 14 组累加
→ 加 bias_q28
→ 4 个 signed int64 写回 DDR3 并经 UART 返回
```

验证结果：

- Python 载荷往返与精确定点参考：1000/1000 PASS，seed=`20260724`；
- 固定真实向量 FPGA 输出：`[207253689, -173360554, 287606739, -223225713]`，逐位一致；
- scale bit31 和 `0xFFFFFFFF` 边界向量：PASS；
- 随机分组 scale 真实上板压力测试：1000/1000 PASS，seed=`20260724`；
- PDS 编译、综合、Device Map、布局布线、时序、位流生成全部成功；
- 最终未布线网络：0；
- 慢角 100 MHz 建立 WNS=`+0.909 ns`、TNS=0，保持 WHS=`+0.111 ns`、THS=0；
- 快角建立 WNS=`+3.041 ns`、TNS=0，保持 WHS=`+0.051 ns`、THS=0；
- 资源：8379 LUT、7492 FF、4 DRM、12 APM；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；
- 位流 SHA256：`d8c7d194d4d8ce1e5d189df39fae5fc904030fe4be6e981a5876a4df73ea17bd`。

首版组合 MAC16 在慢角有 WNS=`-0.109 ns`、TNS=`-0.163 ns` 的 2 个违例端点。改为显式平衡流水归约后，最终报告为 `All Constraints Met`。

## 十一、2026-07-24 新完成：D2 layer0 q_proj 完整真实 Linear 层

新增独立工程、工具和固定清单：

```text
gemv_int4_qproj_full/rtl/int8_dot16_pipe.v
gemv_int4_qproj_full/rtl/gemv_qproj_full_core.v
gemv_int4_qproj_full/rtl/gemv_qproj_full_ctrl.v
gemv_int4_qproj_full/rtl/gemv_qproj_full_top.v
gemv_int4_qproj_full/pnr/build_gemv_qproj_full.tcl
gemv_int4_qproj_full/pnr/program_sram.tcl
gemv_int4_qproj_full/README.md
tools/pangu_gemv_qproj_full_host.py
model_tools/q_proj_full_reference.json
```

固定验收对象为 layer0 `q_proj` 全部 896 个输出行、完整 K=896 输入，每行 14 个 64 元素 group。

完整上传载荷共 `488320 B`：

- 激活：896 B；
- packed INT4 权重：401408 B；
- 每行补齐到 64 B 的 UQ4.28 scale：57344 B；
- 每行补齐到 32 B 的 signed int64 bias_q28：28672 B。

硬件计算和结果调度：

```text
激活读取并缓存一次
→ 每行读取14拍权重、2拍scale、1拍bias
→ 每组4次流水MAC16形成INT32点积
→ signed INT32 × unsigned UQ4.28
→ signed INT64 Q28跨14组累加并加入bias
→ 每4行组成一个256 bit拍立即写回DDR3
→ 完成后从DDR3逐拍读取并通过UART流式返回896个int64
```

验证结果：

- 固定载荷打包/解包、补齐区域和独立 Q28 重算：PASS；
- 固定完整层输出 SHA256：`ea1f04bf4ff313dad07025ff35e66a088f13afd28d817422b89bb135f63525a0`；
- 前 4 行与已验证的 M4K896 小闭环逐位完全一致；
- 软件随机激活压力测试：`1000/1000 PASS`，seed 起点=`20260725`，约 25.88 秒；
- PDS 编译、综合、Device Map、布局布线、时序分析和位流生成全部成功；
- 最终未布线网络：0；
- 资源：8510 LUT、7619 FF、4 DRM、12 APM；
- 多角时序：`All Constraints Met`；慢角 100 MHz WNS=`+0.670 ns`、TNS=0，WHS=`+0.171 ns`、THS=0；快角 WNS=`+3.034 ns`、TNS=0，WHS=`+0.100 ns`、THS=0；
- 恢复、移除和最小脉宽均无违例；
- 位流：`gemv_int4_qproj_full\pnr\generate_bitstream\gemv_qproj_full_top.sbit`；
- 位流 SHA256：`432454b80678c11f493856cb725d791e271d86eada1b5cabccefc0d7486f8894`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；
- 固件：`PANGU50K QPROJ FULL V1`，DDR3 初始化成功；
- 固定完整层真实上板：896 个 signed int64 与 Python 金标准逐位一致；
- 固定完整层上传、计算和回读约 43.03 秒；
- 随机激活真实上板回归：`3/3 PASS`，seed=`20260725..20260727`，约 130.13 秒。

## 十二、2026-07-24 新完成：E1 layer0 input_layernorm K=896 RMSNorm

新增独立工程、软件参考、固定清单和上位机：

```text
model_tools/rmsnorm_fixed_reference.py
model_tools/test_rmsnorm_fixed_reference.py
model_tools/rmsnorm_layer0_reference.json
rmsnorm_k896/rtl/rmsnorm_k896_core.v
rmsnorm_k896/rtl/rmsnorm_k896_ctrl.v
rmsnorm_k896/rtl/rmsnorm_k896_top.v
rmsnorm_k896/pnr/build_rmsnorm_k896.tcl
rmsnorm_k896/pnr/program_sram.tcl
rmsnorm_k896/README.md
tools/pangu_rmsnorm_k896_host.py
```

真实对象为 `model.layers.0.input_layernorm.weight`，连续 FP16、长度 K=896；模型 `rms_norm_eps=1e-6`。算子定义为：

```text
y_i = gamma_i * x_i * rsqrt(mean(x^2) + epsilon)
```

第一版定点格式：

- 输入、gamma 和输出：signed Q6.10 int16；
- 平方和：unsigned 40 位，保留 20 位小数；
- 均值和 epsilon：Q12.20，epsilon 量化为 `1`；
- rsqrt：unsigned UQ12.20 uint32；
- 浮点转整数、除法和右移统一采用 RNE；
- 输出显式饱和到 signed int16。

软件比较了 256 项中点 LUT 和 32 项种子 LUT + 一次 Newton-Raphson。固定向量中 LUT256 的 rsqrt 相对误差为约 `1.3878e-4`，最终输出相对精确定点路径最多相差 1 个 Q10 LSB；NR1 更精确但需要额外乘法流水，因此第一版选择 LUT256。

固定向量关键值：

```text
sum_squares     = 5176164753
mean_square_q20 = 5776970
variance_q20    = 5776971
exact_rsqrt_q20 = 446735
lut_rsqrt_q20   = 446797
output first16  = [20, -16, -38, -11, -71, 4, -65, -32,
                   140, -32, -36, 13, 43, -1, -71, 68]
```

DDR3 与 UART 闭环：

```text
UART 上传 4608 B
→ DDR3 保存 1792 B 输入、1792 B gamma、1024 B LUT
→ FPGA 分段读取并缓存
→ 平方和、RNE 均值、LUT rsqrt、gamma 乘法和输出饱和
→ 896 个 int16 结果写回 DDR3
→ UART 返回并由 Python 逐元素比较
```

验证结果：

- 相关软件单元测试：23/23 PASS；
- RMSNorm 软件随机压力：1000/1000 PASS，seed=`20260726`；
- 固定输出 SHA256：`1f52890780e0f4cc0f734d47a4e3bdb28c3c964b8734b442d7781d4ca155a4f0`；
- PDS 编译、综合、Device Map、布局布线、时序分析和位流生成全部成功；
- 最终未布线网络：0；
- 资源：8801 LUT、7051 FF、12 DRM、9 APM；
- 多角时序：`All Constraints Met`；慢角 100 MHz WNS=`+0.374 ns`、TNS=0，WHS=`+0.171 ns`、THS=0；快角 WNS=`+2.832 ns`、TNS=0，WHS=`+0.100 ns`、THS=0；
- 恢复、移除和最小脉宽均无违例；
- 位流：`rmsnorm_k896\pnr\generate_bitstream\rmsnorm_k896_top.sbit`；
- 位流 SHA256：`94c82d1ef6adf563043c6f90f5744ec258156d85c6db134389132ae4f2938b11`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；
- 固件：`PANGU50K RMSNORM K896 V1`，DDR3 初始化成功；
- 固定真实上板：896 个 signed Q6.10 输出与 Python LUT256 金标准逐位一致，端到端约 0.61 秒；
- 真实随机上板：300/300 PASS，seed=`20260726..20261025`，约 183.11 秒。

时序优化过程：首版 rsqrt 常数校正、动态移位和后级乘法串联导致慢角 WNS=`-4.219 ns`；拆分 rsqrt 流水后提升至 `-0.968 ns`。继续为输入平方与累加、输出 RNE/饱和/打包增加寄存边界后，最终慢角 WNS 收敛到 `+0.374 ns`，所有角落 TNS=0。

## 十三、2026-07-24 新完成：E2 K=896 元素级运算

新增独立工程、软件参考、固定清单和上位机：

```text
model_tools/elementwise_fixed_reference.py
model_tools/test_elementwise_fixed_reference.py
model_tools/elementwise_k896_reference.json
elementwise_k896/rtl/elementwise_k896_core.v
elementwise_k896/rtl/elementwise_k896_ctrl.v
elementwise_k896/rtl/elementwise_k896_top.v
elementwise_k896/pnr/build_elementwise_k896.tcl
elementwise_k896/pnr/program_sram.tcl
elementwise_k896/pnr_seed17/run_seed17.tcl
elementwise_k896/pnr_seed17/program_sram.tcl
elementwise_k896/README.md
tools/pangu_elementwise_k896_host.py
```

统一格式和操作：

- 输入 A/B、标量 scale、SiLU 端点和输出均为 signed Q6.10 int16；
- 残差加法使用扩展加法和显式 signed int16 饱和；
- 定点缩放与元素乘法使用 signed Q12.20 乘积、RNE 右移 10 位和显式饱和；
- SiLU 第一版采用覆盖 `[-8,8)` 的 64 段端点 PWL，区间外采用 `x<-8 -> 0`、`x>=8 -> x`。

完整 65536 个 int16 输入域上的 SiLU 比较：

- 2048 项中点直接 LUT：最大误差 5 Q10 LSB，平均误差 0.352692 LSB，表容量 32768 bit；
- 64 段端点 PWL：最大误差 4 Q10 LSB，平均误差 0.232300 LSB，端点表容量 1040 bit；
- 因误差更小且存储开销显著更低，第一版选择 PWL64，并用一个可流水复用的小乘法器完成插值。

DDR3 与 UART 闭环：

```text
UART 上传 A[896]、B[896] 和 65 个 PWL 端点
→ DDR3 固定地址保存
→ AXI burst 读取并装入片上缓存
→ 选择残差/缩放/元素乘法/SiLU 四种操作
→ 16 个 int16 结果打包为一个 256 bit 拍
→ 56 拍结果写回 DDR3
→ UART 返回 896 个 int16
→ Python 逐元素比较
```

验证结果：

- E2 单元测试：11/11 PASS；
- 完整 `model_tools` 回归：34/34 PASS；
- 软件和上传载荷随机压力：1000/1000 PASS，seed=`20260727`；
- 固定边界向量覆盖 RNE tie、正负溢出、饱和、SiLU 尾部和最高 `segment=63`；
- 固定四操作真实上板：每种 896 个 signed Q6.10 输出与 Python 逐位一致，端到端约 1.01 秒；
- 固定输出 SHA256：residual=`dd6cf26e917004e52973ee8506bfdc2e403dac2d31e64abba9c6cd4619196dca`，scale=`8137acd3e9c983380ef1d024858e88ed54b675791cf416539ca3b03fa9c3455c`，multiply=`f07847b17449eb401324b413b4df7765d14377e9b20c340f48e6dc87112f25aa`，SiLU=`1933e7c436030c00285bffb2def77c70c979b32c041af3833f61fa25825fdbf8`；
- 真实随机上板：分三批累计 300/300 PASS，seed=`20260727..20261026`，总耗时约 312.49 秒；
- PDS 编译、综合、Device Map、布局布线、时序分析和位流生成全部成功；
- 最终未布线网络：0；
- 资源：7872 LUT、7778 FF、70 个 distributed RAM LUT、8 DRM、2 APM；
- 多角时序：`All Constraints Met`；慢角 100 MHz WNS=`+0.580 ns`、TNS=0，WHS=`+0.112 ns`、THS=0；快角 WNS=`+2.951 ns`、TNS=0，WHS=`+0.051 ns`、THS=0；
- 恢复、移除和最小脉宽均无违例；
- 位流：`elementwise_k896\pnr_seed17\generate_bitstream\elementwise_k896_top.sbit`；
- 位流 SHA256：`809b436f1c369d66a20c5f2faaa8e684a15a3963d659b95d080e342c3a7d9d50`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；
- 固件：`PANGU50K ELEMENTWISE K896 V1`，DDR3 初始化成功。

开发中修复了两类问题：首版 SiLU 小乘法、64 位 RNE 和端点加法形成长组合路径；拆成窄位寄存流水后时序通过。首次固定上板仅最高 PWL 段 3 个元素错误，根因是 6 位 `63+1` 索引回绕到 0；改用 7 位端点索引后固定和随机测试全部通过。

## 十四、2026-07-24 新完成：E3 真实 tied Embedding K=896

新增独立工程、软件参考、固定清单和上位机：

```text
model_tools/embedding_fixed_reference.py
model_tools/test_embedding_fixed_reference.py
model_tools/embedding_k896_reference.json
embedding_k896/rtl/embedding_k896_core.v
embedding_k896/rtl/embedding_k896_ctrl.v
embedding_k896/rtl/embedding_k896_top.v
embedding_k896/pnr/build_embedding_k896.tcl
embedding_k896/pnr/program_sram.tcl
embedding_k896/README.md
tools/pangu_embedding_k896_host.py
```

真实对象为 tied `model.embed_tokens.weight`，shape=`[151936,896]`，storage=`int4_groupwise_symmetric`，group size=64，每行 14 groups，Token ID 有效范围为 `0..151935`。

DDR3 行槽和定点路径：

```text
row_base_ctrl_addr = token_id << 7
每个 Token 固定 512 B / 16 个 256 bit 拍
→ 448 B packed signed INT4
→ 56 B / 14 个 UQ4.28 scale
→ 8 B padding
→ signed INT4 × unsigned UQ4.28
→ RNE 右移 18 位
→ signed Q6.10 int16 显式饱和
→ 896 个输出写回 DDR3 并经 UART 返回
```

真实 embedding 的全部 FP16 scales 均可被 UQ4.28 精确表示，因此硬件固定路径与直接执行 `round_to_nearest_even(INT4 * FP16_scale * 2^10)` 逐位一致。

验证结果：

- E3 单元测试：11/11 PASS；
- 完整 `model_tools` 回归：45/45 PASS；
- 真实 P50 软件/载荷随机压力：1000/1000 PASS，seed=`20260728`；
- 最大 Q6.10 量化误差：`0.00048828125`，不超过 0.5 个 Q10 LSB；
- 四个固定 Token ID `[0,1,2026,151935]` 的 896 个输出真实上板逐位一致，总耗时约 0.93 秒；
- 真实随机 Token ID 上板压力：300/300 PASS，seed=`20260728`，约 75.53 秒；
- PDS 编译、综合、Device Map、布局布线、时序分析和位流生成全部成功；
- 最终未布线网络：0；
- 资源：7637 LUT、7380 FF、326 个 distributed RAM、2 APM、0 DRM；
- 多角时序：`All Constraints Met`；慢角 100 MHz WNS=`+0.679 ns`、TNS=0，WHS=`+0.172 ns`、THS=0；快角 WNS=`+2.964 ns`、TNS=0，WHS=`+0.101 ns`、THS=0；
- 恢复、移除和最小脉宽均无违例；
- 位流：`embedding_k896\pnr\generate_bitstream\embedding_k896_top.sbit`；
- 位流 SHA256：`cd0e138e494875035cf5c66d76eaf250729625c172bf51c935b831d31c45c0fa`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；
- 固件：`PANGU50K EMBEDDING K896 V1`，DDR3 初始化成功。

## 十五、2026-07-24 新完成：F1 layer0 真实 Q/K/V 线性层

新增统一软件参考、固定清单、独立硬件工程和上位机：

```text
model_tools/qkv_linear_reference.py
model_tools/test_qkv_linear_reference.py
model_tools/qkv_layer0_reference.json
qkv_linear_layer0/rtl/int8_dot16_pipe.v
qkv_linear_layer0/rtl/qkv_linear_core.v
qkv_linear_layer0/rtl/qkv_linear_ctrl.v
qkv_linear_layer0/rtl/qkv_linear_top.v
qkv_linear_layer0/pnr/build_qkv_linear.tcl
qkv_linear_layer0/pnr_seed5/run_seed5.tcl
qkv_linear_layer0/pnr_seed5/program_sram.tcl
qkv_linear_layer0/README.md
tools/pangu_qkv_linear_host.py
```

真实对象和 GQA 布局：

```text
q_proj.weight = [896,896] -> 14 Q heads × 64
k_proj.weight = [128,896] ->  2 K heads × 64
v_proj.weight = [128,896] ->  2 V heads × 64
```

三种投影均使用 group size 64 的真实 signed INT4 权重、同一逐向量对称 INT8 hidden state、UQ4.28 combined scale、signed Q28 bias 和 signed int64 Q28 输出。平坦输出按 head-major 连续排列，可无损还原为 Q=`[14,64]`、K/V=`[2,64]`。

载荷随投影动态切换：Q 为 488320 B，K/V 各为 70528 B；硬件命令 `Q/K/V` 选择投影，输出行数和结果回读长度分别为 896/128/128。DDR3 地址布局复用已验证完整 q_proj 工程，但新建独立目录和位流，不覆盖任何既有成果。

验证结果：

- 新增 F1 单元测试 3/3 PASS；完整 `model_tools` 回归 48/48 PASS；
- 固定清单、packed INT4、补齐 scale/bias、载荷往返、独立 Q28 重算和共享 hidden state 检查全部通过；
- QKV 软件随机 hidden state 压力 1000/1000 PASS，seed=`20260729`；
- 固定 Q 全 896 行真实上板逐位一致，输出 SHA256=`ea1f04bf4ff313dad07025ff35e66a088f13afd28d817422b89bb135f63525a0`；
- 固定 K 全 128 行真实上板逐位一致，输出 SHA256=`20728d329c32c722b0194032897bc3cf9a3a31323317e389d8fd7b6f78745474`；
- 固定 V 全 128 行真实上板逐位一致，输出 SHA256=`162622e05e0013ca342f28032cb280c264f428f93a197eb67dbfafd76e20a168`；
- 固定输出 head shape 分别为 `(14,64)`、`(2,64)`、`(2,64)`；
- 真实随机完整 Q+K+V 上板回归 3/3 PASS，seed=`20260729..20260731`，约 166.72 秒；
- 默认种子和 seed17/29 均只在 DDR3 IP 内部出现极小快角 hold 违例，未作为有效位流；最终 seed5/11 全约束通过；
- PDS 编译、综合、Device Map、布局布线、时序分析和位流生成全部成功，最终未布线网络为 0；
- 资源：8503 LUT、7641 FF、326 个 distributed RAM、4 DRM、12 APM；
- 多角时序：`All Constraints Met`；慢角 setup WNS=`+0.363 ns`、TNS=0，hold WHS=`+0.169 ns`、THS=0；快角 setup WNS=`+2.985 ns`、TNS=0，hold WHS=`+0.100 ns`、THS=0；
- 恢复、移除和最小脉宽均无违例；
- 位流：`qkv_linear_layer0\pnr_seed5\generate_bitstream\qkv_linear_top.sbit`，大小 2101696 B；
- 位流 SHA256：`e3a4b6849a5716f38d6bdd3fbd039d46f2d350a32a0417ee347462d1a8f96e26`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；
- 固件：`PANGU50K QKV LINEAR V1`，DDR3 初始化成功。

## 十六、F2 layer0 Q/K RoPE 已完成（2026-07-24）

新增文件：

```text
model_tools/rope_fixed_reference.py
model_tools/rope_layer0_reference.json
model_tools/test_rope_fixed_reference.py
rope_qk_layer0/rtl/rope_pair_q28_core.v
rope_qk_layer0/rtl/rope_qk_ctrl.v
rope_qk_layer0/rtl/rope_qk_top.v
rope_qk_layer0/pnr/build_rope_qk.tcl
rope_qk_layer0/pnr/program_sram.tcl
rope_qk_layer0/README.md
tools/pangu_rope_qk_host.py
```

模型配置和数学规则：

```text
head_dim = rotary_dim = 64
rope_theta = 1000000
max_position_embeddings = 32768
Q = [14,64] signed int64 Q28
K = [2,64] signed int64 Q28
sin/cos = signed int32 Q1.30
```

已核对 Qwen2 实际 `rotate_half`：每个 head 的前 32 维与后 32 维配对，即 `dim i <-> dim i+32`，而不是相邻 `(0,1)、(2,3)` 配对。硬件对四个 64×32 乘积进行精确计算，在 signed 97 bit 中完成两项加/减，最后执行一次 RNE 右移 30 位并饱和到 signed int64 Q28。

首版直接 64×32 组合乘法虽能生成位流，但慢角 setup 为 `WNS=-2.017 ns`、`TNS=-254.708 ns`，未作为有效成果。最终将乘法拆为 8 个 16×16 limb 部分积并顺序复用一个 APM，同时将 97 位 combine、绝对值、RNE 和饱和分级寄存，完成时序收敛。

验证结果：

- 固定位置 `[0,1,2026,32767]` 的真实 Q/K 软件参考与清单建立完成；
- 固定位置最大绝对误差为 `0`、`5.453485130147e-08`、`4.564708433463e-08`、`7.232674192892e-08`，均低于 `9.294017896955e-08` 保守界；
- F2 新增单元测试 7/7 PASS；完整 `model_tools` 回归 55/55 PASS；
- 软件随机 Q/K 与位置压力 1000/1000 PASS，seed=`20260730`；上位机软件自检 1000/1000 PASS，seed=`20260731`；
- 固定位置 `0、1、2026、32767` 真实上板 Q/K 全输出逐位一致；
- 连续位置 `2026..2033` 自动递增 8/8 PASS，结束状态正确，`Z` 复位后位置 2026 重放逐位一致；
- 真实随机位置上板回归 300/300 PASS，seed=`20260731`，约 235.59 秒；
- PDS 编译、综合、Device Map、布局布线、时序分析和位流生成全部成功，未布线网络为 0；
- 资源：8859 LUT、9886 FF、70 个 distributed RAM、1 APM、0 DRM；
- 多角时序：`All Constraints Met`；慢角 setup WNS=`+0.988 ns`、TNS=0，hold WHS=`+0.171 ns`、THS=0；快角 setup WNS=`+3.483 ns`、TNS=0，hold WHS=`+0.100 ns`、THS=0；
- 恢复、移除和最小脉宽均无违例；
- 位流：`rope_qk_layer0\pnr\generate_bitstream\rope_qk_top.sbit`；
- 位流 SHA256：`25396ffc894abc15b81ab99f62619f3694e7e662f620f3c6a89e28ae116d153a`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；
- 固件：`PANGU50K ROPE QK V1`，DDR3 初始化成功。

## 十七、F3 KV Cache 已完成（2026-07-24）

新增软件参考、固定清单、独立硬件工程和上位机：

```text
model_tools/kv_cache_reference.py
model_tools/kv_cache_reference.json
model_tools/test_kv_cache_reference.py
kv_cache_f3/rtl/kv_cache_ctrl.v
kv_cache_f3/rtl/kv_cache_top.v
kv_cache_f3/pnr/build_kv_cache.tcl
kv_cache_f3/pnr/program_sram.tcl
kv_cache_f3/README.md
tools/pangu_kv_cache_host.py
```

容量与布局结论：

```text
K = [2,64] signed int64 Q28 = 1024 B
V = [2,64] signed int64 Q28 = 1024 B
单 token 槽 = 2048 B
低端保留区 = 128 MiB
KV Cache = 896 MiB
每层 = 32 MiB
层数 = 28
硬件上下文 = 16384 token
```

完整支持模型标称 32768 positions 需要 1792 MiB，超过板载 1 GiB，因此 F3 第一版将硬件上下文确定为 16384。Controller 地址公式为：

```text
K = 0x02000000 + layer × 0x00800000 + position × 0x00000200
V = K + 0x00000100
```

首槽从字节地址 `0x08000000` 开始，layer27/position16383 的末槽严格结束于 `0x40000000`，即 1 GiB 边界。硬件支持当前 token 写入后自动推进位置，以及一次连续 1..16 token 的历史 K/V 分段 AXI burst 顺序读取。

验证结果：

- F3 新增单元测试 9/9 PASS；完整 `model_tools` 回归 64/64 PASS；
- 软件地址、容量、边界、载荷往返随机压力 1000/1000 PASS，seed=`20260801`；
- 固定真实 K/V 来自 F2 RoPE 后 K 和 F1 V，覆盖 layer0 position `0..1`、layer13 position `2026`、layer27 position `16383`；全部真实上板逐位一致；
- 连续 position 自动推进和 2 token 历史顺序读取通过；固定测试约 1.66 秒；
- 最后槽结束于 1 GiB，下一 token 写入被错误码 `0x05` 正确拒绝；
- layer3/layer17 在相同 position `4096` 写入不同 K/V，跨配置回读均逐位一致，层间无覆盖；
- 真实随机层、随机 position、每批 1..16 token 上板回归 300/300 token PASS，seed=`20260801`，约 124.41 秒；
- 随机回归周期性重新读取旧层旧位置，证明后续写入没有覆盖此前数据；
- PDS 编译、综合、Device Map、布局布线、时序和位流生成全部成功，最终未布线网络 0；
- 资源：7572 LUT、9884 FF、70 个 distributed RAM、0 DRM、0 APM；
- 多角时序 `All Constraints Met`；慢角 core setup WNS=`+1.781 ns`、TNS=0，hold WHS=`+0.171 ns`、THS=0；快角 setup WNS=`+4.142 ns`、TNS=0，hold WHS=`+0.100 ns`、THS=0；
- 恢复、移除和最小脉宽无违例；
- 位流：`kv_cache_f3\pnr\generate_bitstream\kv_cache_top.sbit`，大小 2101696 B；
- 位流 SHA256：`11a0240a2ee42f0c92b6a5919f4a4b71ceb7bb806b55f1810b4ef3ff88d23216`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；
- 固件：`PANGU50K KV CACHE V1`，DDR3 初始化成功。

## 十八、当前项目状态

当前已经完成十二级真实闭环：

```text
长度16单点积
→ 固定 M=4、K=64 packed INT4 GEMV
→ 运行时参数化 M/K、尾块屏蔽的通用 packed INT4 GEMV
→ GEMV 周期计数、带宽、GMAC/s、利用率和瓶颈分析
→ 真实 q_proj M4K896 分组 UQ4.28 signed INT64 Q28 小闭环
→ 真实 layer0 q_proj M896K896 完整 Linear 层闭环
→ 真实 layer0 input_layernorm K896 定点 RMSNorm 闭环
→ K896 残差、缩放、元素乘法和 PWL64 SiLU 闭环
→ 真实 tied Embedding Token 行查表与 Q6.10 格式转换闭环
→ 真实 layer0 Q/K/V 与 14Q/2KV GQA head-major 布局闭环
→ 真实 layer0 Q/K Qwen2 split-half RoPE、位置递增和 Q28/Q1.30 闭环
→ 28 层、16384 token K/V Cache 写入、历史顺序读取、边界和防覆盖闭环
```

这证明 DDR3 Controller + PHY、长 burst、片上缓存、INT4 解包、流水 MAC16、逐组 scale、64 位定点乘加、动态 896/128 行调度、GQA head 布局、Qwen2 split-half RoPE、位置表自动推进、28 层 KV 地址调度、当前 token 写入、历史分段 burst 读取、RMSNorm、元素级非线性、Embedding、结果流式写回和 Python 自动验证可以协同工作。

当前仍不是完整 Qwen 推理；真实 Q/K/V Linear、RoPE、KV Cache、RMSNorm、元素级基础算子和 Embedding 已完成，尚未完成 Attention Score、Softmax、Attention 输出、MLP、Transformer Block 和文本生成。

## 十九、下一阶段路线

### 当前唯一下一步：F4 Attention Score

1. 基于 F1 已验证的 Q、F2 RoPE 输出和 F3 历史 K Cache，建立 Q·K 点积软件金标准。
2. 明确 `head_dim=64` 对应的 `1/sqrt(head_dim)=1/8` 定点缩放、舍入和饱和规则。
3. 建立 14 个 Q heads 到 2 个 KV heads 的 GQA 映射，并定义 score 输出布局。
4. 加入 causal mask，覆盖当前位置、历史位置和越界边界。
5. 新建独立 Attention Score 工程，实现历史 K 顺序读取、多头循环调度和 score 输出。
6. 完成固定/随机序列、PDS 全流程、多角时序和真实上板验证；本阶段不得提前进入 Softmax。

### 后续算子

按以下顺序逐步实现并逐层验证：

1. Attention Score。
2. Softmax。
3. Attention 输出。
4. MLP。
5. Transformer Block。
6. 完整模型权重加载与分层调度。
7. tokenizer、采样与文本推理验证。

每一步都应保留“FPGA 结果与 Python 参考逐元素自动比较”的闭环，避免直接跳到完整模型后难以定位错误。
