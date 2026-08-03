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

## 十八、F4 Attention Score 已完成（2026-07-24）

新增软件参考、固定清单、独立硬件工程和上位机：

```text
model_tools/attention_score_reference.py
model_tools/attention_score_f4_reference.json
model_tools/test_attention_score_reference.py
attention_score_f4/rtl/attention_score_core.v
attention_score_f4/rtl/attention_score_ctrl.v
attention_score_f4/rtl/attention_score_top.v
attention_score_f4/pnr/build_attention_score.tcl
attention_score_f4/pnr/program_sram.tcl
attention_score_f4/README.md
tools/pangu_attention_score_host.py
```

定点和布局结论：

```text
Q = [14,64] signed int64 Q28
K = [2,64] signed int64 Q28
GQA = Q head 0..6 -> KV0，Q head 7..13 -> KV1
点积 = 64 项精确 signed Q56 累加
缩放 = 1/sqrt(64)=1/8
输出 = signed RNE 右移 31 位并饱和到 int64 Q28
mask = INT64_MIN
固定输出 = [14,16] head-major，1792 B
```

K 地址完全复用 F3：

```text
K = 0x02000000 + layer × 0x00800000 + position × 0x00000200
```

Q/K 缓存改为 256 bit beat 同步读结构，成功推断 8 个 DRM18K；64×64 有符号乘法由 16 个 16×16 部分积顺序精确重构，避免把大缓存和大乘法直接展开为触发器与长组合路径。score 区先统一初始化为 mask，再按 64 位 byte-enable 部分写回，避免额外 14 KiB 结果寄存器。

验证结果：

- F4 新增单元测试 9/9 PASS；完整 `model_tools` 回归 73/73 PASS；
- 软件随机 GQA、RNE、causal mask、载荷和窗口压力 1000/1000 PASS，seed=`20260802`；
- 固定真实窗口覆盖 layer0/query0、layer0/query1、layer13/query2026 的部分未来 mask、layer27/query16383 的最后 16 token 边界；四组完整 score 全部真实上板逐位一致；
- 四组 score SHA256：`0697d1457bbd91a13a86e06b7de87a9928258c51c2b2b23d31a054bbc99325c5`、`30deb88a395f65ebaa92810278a8954f4cb0c8999462eb7071449dbf957a515d`、`c91ad94ac9af6da06aa2143cd81c87c8d3aeb68cd93ee5e50ecc54271bd51096`、`466cea477112b15a43ee5d03529bc34c10880f93887e44ae078bfac3ef527948`；
- 未来位置和固定未使用槽均严格输出 `INT64_MIN`；
- 真实随机层、随机 query/start、1..16 token 窗口和随机 Q/K 上板回归 100/100 PASS，seed=`20260802`，约 170.16 秒；
- PDS 编译、综合、Device Map、布局布线、时序和位流生成全部成功，最终未布线网络 0；
- 资源：9594 LUT、11621 FF、70 个 distributed RAM、8 DRM、1 APM；
- 多角时序 `All Constraints Met`；慢角 core setup WNS=`+0.482 ns`、TNS=0，hold WHS=`+0.170 ns`、THS=0；快角 setup WNS=`+3.003 ns`、TNS=0，hold WHS=`+0.100 ns`、THS=0；
- 恢复、移除和最小脉宽无违例；
- 位流：`attention_score_f4\pnr\generate_bitstream\attention_score_top.sbit`，大小 2101696 B；
- 位流 SHA256：`669cb5b23cb6c5d33d0003f32452e57cda251751179c318c1b5d8f2ed8c0e0f8`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；
- 固件：`PANGU50K ATTN SCORE V1`，DDR3 初始化成功；
- 本阶段严格停在 Attention Score，没有提前实现 Softmax。

## 十九、F5 Softmax 已完成（2026-07-24）

新增软件参考、固定清单、独立硬件工程和上位机：

```text
model_tools/softmax_fixed_reference.py
model_tools/softmax_f5_reference.json
model_tools/test_softmax_fixed_reference.py
softmax_f5/rtl/softmax_core.v
softmax_f5/rtl/softmax_ctrl.v
softmax_f5/rtl/softmax_top.v
softmax_f5/pnr/build_softmax.tcl
softmax_f5/pnr/program_sram.tcl
softmax_f5/README.md
tools/pangu_softmax_host.py
```

定点和数值稳定性结论：

```text
输入 = [14,16] signed int64 Q28
mask = INT64_MIN，概率严格为 0
输出 = [14,16] unsigned UQ1.31 uint32
1.0 = 0x80000000
exp = [-16,0]、步长 1/32 的 513 点端点 LUT + 线性插值
sum = 最多 16 项 UQ1.31 的 36 位和
reciprocal = RNE(2^62 / sum_exp_q31)
probability = RNE(exp_q31 × reciprocal_q31 / 2^31)
```

每个 head 先忽略 mask 执行 max reduction，再减最大值，确保 exp 输入不大于 0。差值小于 `-16` 时 exp 尾部置 0；全 mask head 输出全 0；单有效 token 精确输出 1.0；16 个相同 score 精确输出均匀概率。硬件将 26×23 插值乘法拆成四个 13×12/11 部分积和两级加法流水，并在 32×32 概率乘法前增加 exp 选择寄存，最终满足 100 MHz 慢角时序。

验证结果：

- F5 新增单元测试 10/10 PASS；完整 `model_tools` 回归 83/83 PASS；
- 软件随机 mask、窗口、极端差值、LUT 和载荷压力 1000/1000 PASS，seed=`20260803`，最坏 float64 概率误差 `3.04973546883e-05`；
- 四组真实 F4 score 固定窗口全部真实上板逐位一致，概率 SHA256 为 `768bd8912f9168473b8978963e805b52b5eeb40b26c517229dc6a4c8d96ce608`、`021ac6fad9854aeede02829734c6afdc3dc9cb41ce79ce078b830a53b695ce81`、`267a18e4d4fef9d1afb118d8f1a025cd9922f14963ae75fe672c30c816e5495f`、`b1ae419016695bb6c2a62ffb1b92c7bcbb70c87b22c1ba6d7cb96e327b201f39`；
- 单有效 token 精确 1.0、未来位置和未使用槽严格为 0；
- 真实 FPGA 全 mask、单有效、部分窗口、满 16 token、全等 score、`-16` 截断边界和随机稀疏 mask 回归 100/100 PASS，seed=`20260803`，约 29.05 秒；最坏 float64 概率误差 `2.96390625578e-05`；
- PDS 编译、综合、Device Map、布局布线、时序和位流生成全部成功，最终未布线网络 0；
- 资源：10515 LUT、12703 FF、70 个 distributed RAM、12 DRM18K、8 APM；
- 多角时序 `All Constraints Met`；慢角 core setup WNS=`+0.227 ns`、TNS=0，hold WHS=`+0.143 ns`、THS=0；快角 setup WNS=`+2.958 ns`、TNS=0，hold WHS=`+0.067 ns`、THS=0；
- 恢复、移除和最小脉宽无违例；
- 位流：`softmax_f5\pnr\generate_bitstream\softmax_top.sbit`，大小 2101696 B；
- 位流 SHA256：`d6e505ea5495c6054a447608406db0f93855ef55dbfc357c8d113b00adba34fe`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；
- 固件：`PANGU50K SOFTMAX F5 V1`，DDR3 初始化成功；
- 本阶段严格停在 Softmax 概率输出，没有提前实现 V 加权和。

## 二十、F6 Attention 输出加权和与多头拼接已完成（2026-07-24）

新增软件参考、固定清单、独立硬件工程和上位机：

```text
model_tools/attention_output_reference.py
model_tools/attention_output_f6_reference.json
model_tools/test_attention_output_reference.py
attention_output_f6/rtl/attention_output_core.v
attention_output_f6/rtl/attention_output_ctrl.v
attention_output_f6/rtl/attention_output_top.v
attention_output_f6/pnr/build_attention_output.tcl
attention_output_f6/pnr/program_sram.tcl
attention_output_f6/README.md
tools/pangu_attention_output_host.py
```

定点和布局结论：

```text
probability = [14,16] unsigned UQ1.31 uint32
V history = [count,2,64] signed int64 Q28，count=1..16
GQA = Q head 0..6 -> KV0；Q head 7..13 -> KV1
product = signed Q59
accumulator = signed 100 bit，最多 16 项精确累加
output = single signed RNE >>31，再显式 int64 饱和
heads = [14,64] head-major
concat = [896] head-major contiguous，共 7168 B
```

硬件先将 28 个概率 beat 和最多 512 个 V beat 装入同步双口 DRM 缓存。概率拆成 2 个 16-bit limb，V 绝对值拆成 4 个 16-bit limb，8 个 16×16 部分积顺序精确重构 signed 96-bit Q59；896 个输出按 head/dimension 顺序流式计算并以 byte-enable 写回 DDR3。全 mask 输出严格全 0，单 token 1.0 概率精确复制对应 GQA V，全部 token 只在累加结束后执行一次 RNE。

验证结果：

- F6 新增单元测试 10/10 PASS；完整 `model_tools` 回归 93/93 PASS；
- 软件随机 GQA、Q59、signed RNE、int64 饱和、载荷和 `[14,64] <-> [896]` 拼接压力 1000/1000 PASS，seed=`20260804`；
- 四组真实 F5 概率与逐 position 真实 layer0 `v_proj` 固定 V 全部上板逐位一致，输出 SHA256 为 `c5107911c0e6b9f1d9c471d7dde2c26d1192282abc964eb5005051aa5a4c9f71`、`14a6ee14736ed6132ea1357d3af27d55074492a727b196c33cd6905d4e1c9b02`、`86bc89cc77d49ac9451cc2a707510df310566997faf4c76fdfa95599043c248b`、`73e6b464b8c27e6a4d1066df5b5dade1c11c6a085a712d58dc54f6b598a0e407`；
- 真实 FPGA 全 mask / 全零概率严格全 0；单 token 1.0、14Q/2KV GQA 和 `INT64_MIN/MAX` 极端 V 精确复制通过；16-token Q59 宽累加后的 INT64 正/负双向显式饱和通过；
- 真实 FPGA 随机层、随机 start、count `1..16`、随机概率/V 回归 100/100 PASS，seed=`20260804`，约 182.09 秒；每 10 轮包含一次全 64-bit V 随机模式；
- PDS 编译、综合、Device Map、布局布线、时序和位流生成全部成功，详细路由 155 轮后未布线网络 0；
- 资源：9184 LUT、11357 FF、70 个 distributed RAM、12 DRM、1 APM；
- 多角时序 `All Constraints Met`；慢角 setup WNS=`+0.825 ns`、TNS=0，hold WHS=`+0.112 ns`、THS=0；快角 setup WNS=`+3.349 ns`、TNS=0，hold WHS=`+0.032 ns`、THS=0；
- 慢/快角恢复和移除无违例；
- 位流：`attention_output_f6\pnr\generate_bitstream\attention_output_top.sbit`，大小 2101696 B；
- 位流 SHA256：`d7e64c58b73f8ca93f7a7dd981feabe5cc48f9b43e6b2ff0d8f60155886f36a3`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；
- 固件：`PANGU50K ATTN OUTPUT V1`，DDR3 初始化成功；
- 本阶段严格停在加权和与多头拼接，没有提前实现 `O_proj`、Attention 残差或 MLP。

## 二十一、F6 layer0 Attention O_proj 已完成（2026-07-24）

新增软件参考、固定清单、独立硬件工程和上位机：

```text
model_tools/attention_oproj_reference.py
model_tools/attention_oproj_f6_reference.json
model_tools/test_attention_oproj_reference.py
attention_oproj_f6/rtl/attention_oproj_top.v
attention_oproj_f6/pnr/build_attention_oproj.tcl
attention_oproj_f6/pnr/program_sram.tcl
attention_oproj_f6/README.md
tools/pangu_attention_oproj_host.py
```

真实模型与定点结论：

```text
input = F6 [14,64] head-major concat -> [896] signed int64 Q28
weight = model.layers.0.self_attn.o_proj.weight，shape [896,896]
storage = groupwise symmetric signed INT4，group_size=64，14 groups/row
bias = .p50 中不存在，bias_q28 固定全 0
activation = Q28 转 float32 后逐向量对称 INT8 [-127,127]
combined scale = activation_scale * FP16 weight_scale -> unsigned UQ4.28
group_acc = 64 项 INT8×INT4 的 signed INT32 点积
output = 14 个 group product 在 signed int64 Q28 中累加
```

为避免覆盖已有工程，本阶段仅复用已经真实上板通过的完整 Linear `controller/core`，新建独立顶层和 PDS 工作目录。顶层直接实例化 DDR3 IP，保持板级约束所需的 `I_ipsxb_ddr_top` 层级。底层 UART 固件标识继续为通用协议版本 `PANGU50K QPROJ FULL V1`，但实际上传权重、scale、零 bias 和金标准均为真实 O_proj。

验证结果：

- O_proj 新增单元测试 7/7 PASS；完整 `model_tools` 回归 100/100 PASS；
- 四组真实固定输入直接复用 F6 第一段的 1、2、6、16-token 窗口输出，真实参数、独立 Q28 重算、固定清单和 488320 B 载荷往返全部 PASS；
- 四组 O_proj 输出 SHA256 为 `19008a25a59cde0f8def0c938ada397b6866dc143774b74c6ff77a2a95a7fcd5`、`0e70753bea148c81d0bce79360d250710a1cc6ee817a40e4b6cbccf7d4f30279`、`c0ffeb8b5a1168b661d52a34f34a5f4f12f3d075805b05b4ace346683cb8b018`、`af63d1efc3913f597fdcd5dbe520ac782a943074301a60b249f4f25a3cf34a65`；
- 软件随机/边界 Attention Q28 输入和完整上传载荷压力 1000/1000 PASS，seed=`20260805`，约 35.26 秒；全零输入严格输出全零，固定用例 combined scale 饱和和 activation clipping 数均为 0；
- PDS 编译、综合、Device Map、布局布线、时序和位流生成全部成功，最终未布线网络 0；
- 资源：8510 LUT、7619 FF、326 个 distributed RAM、4 DRM、12 APM；
- 多角时序 `All Constraints Met`；慢角 core setup WNS=`+0.614 ns`、TNS=0，hold WHS=`+0.171 ns`、THS=0；快角 setup WNS=`+3.023 ns`、TNS=0，hold WHS=`+0.101 ns`、THS=0；
- 慢角 recovery WNS=`+2.919 ns`、removal WHS=`+0.537 ns`；快角 recovery WNS=`+4.898 ns`、removal WHS=`+0.337 ns`；最小脉宽无违例；
- 位流：`attention_oproj_f6\pnr\generate_bitstream\attention_oproj_top.sbit`，大小 2101696 B；
- 位流 SHA256：`017517f877f29e62d945ecd3ae4ba22c2d690b6e6b92778eb0502ba7ac115533`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；DDR3 初始化成功；
- 四组真实固定输入的 896 个 O_proj 输出全部真实上板逐位一致，单组上传、计算与回读约 43.03~43.04 秒；
- 真实 FPGA 全零、随机常量、稀疏极值和完整 896 维随机 Attention Q28 输入回归 4/4 PASS，seed=`20260805`，约 172.34 秒；
- 第二段严格停在 O_proj，当时尚未实现 Attention 残差、完整 Attention 子层或 MLP。

## 二十二、F6 Attention 残差与完整子层闭环（2026-07-24）

本阶段建立独立 `attention_residual_f6` 工程，完成 layer0 Attention 第一处残差，并首次把此前分别验证的算子串成同一 hidden state 来源的连贯软件参考：

```text
hidden state signed Q6.10
→ layer0 input_layernorm
→ Q/K/V
→ RoPE
→ K/V 历史
→ Attention Score
→ Softmax
→ probability × V
→ [14,64] 多头拼接为 [896]
→ 真实 layer0 O_proj signed Q28
→ Q28 到 Q6.10 重标定
→ 与原 hidden state 残差相加
→ 完整 layer0 Attention 子层 signed Q6.10 输出
```

统一定点规则：

```text
residual hidden = [896] signed int16 Q6.10
O_proj output = [896] signed int64 Q28
oproj_q10 = saturate_int16(signed_RNE(oproj_q28 / 2^18))
attention_output_q10 = saturate_int16(residual_hidden_q10 + oproj_q10)
```

重标定饱和和最终残差饱和分别执行。RTL 对 `INT64_MIN` 使用无符号幅值路径，保证正负 half-way tie 均采用 round-to-nearest-even。

验证结果：

- 新增单元测试 5/5 PASS，完整 `model_tools` 回归 105/105 PASS；
- 四组连贯固定 query/count 为 `0/1`、`1/2`、`5/6`、`15/16`，从各 token 原始 hidden state 开始生成真实 RMSNorm、Q/K/V、RoPE、Score、Softmax、V 加权、O_proj 和残差；
- 四组最终输出 SHA256 为 `36859690e421b96cb8db65a5760a364d165a73b63fd1121040a7d1b42c042eb7`、`2b4a2d9240e6e30c2afe2943fa30ac60decd47f8fc8d377ab7e530e516009378`、`c0c0776d71e3dc97aa1a4d4e0709f38441cc82717c0c3081a79c47c30a21af10`、`7e61dc1fd0eb43b231e25fe1d08b1c08342723f537b6e25924858608235fa61e`；
- 8960 B 固定上传载荷、signed RNE 正负 half-way tie、`INT64_MIN/MAX`、Q28→Q10 正负饱和和最终残差正负饱和全部通过；
- 软件随机/边界压力 1000/1000 PASS，seed=`20260806`；
- PDS Compile、Synthesize、Device Map、Place & Route、Timing 和 Bitstream 全部成功，最终未布线网络为 0；
- 资源：7695 LUT、6868 FF、70 个 distributed RAM、20 DRM、0 APM、79 IO；
- 多角时序 `All Constraints Met`：慢角 core setup WNS=`+1.493 ns`、TNS=0，hold WHS=`+0.112 ns`、THS=0；快角 setup WNS=`+3.841 ns`、TNS=0，hold WHS=`+0.051 ns`、THS=0；恢复和移除无违例；
- 位流：`attention_residual_f6\pnr\generate_bitstream\attention_residual_top.sbit`，大小 2101696 B；
- 位流 SHA256：`609e1f569aa1e4579cffb995b0d7d0bc89fa34529790b35e8b26d6778226bcbd`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；固件 `PANGU50K ATTN RESIDUAL V1`，DDR3 初始化成功；
- 四组连贯真实固定 Attention 子层输出全部 896/896 上板逐位一致，约 3.94 秒；
- 真实 FPGA 随机/边界累计 300/300 PASS，分三批 seed_start=`20260806`、`20260906`、`20261006`，每批约 98.67~98.68 秒。

F6 Attention 至此完整通过，允许进入 G1 MLP。

## 二十三、当前项目状态

当前已经完成十七级真实闭环：

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
→ 14Q/2KV GQA Attention Score、1/8 缩放、causal mask 和随机窗口闭环
→ 14 heads mask 感知 Softmax、PWL exp、Q31 倒数和概率归一化闭环
→ F5 概率 × F3 V、14Q/2KV GQA、Q59 累加、RNE 和 `[14,64]/[896]` Attention 输出闭环
→ 真实 layer0 self_attn.o_proj M896K896 分组 INT4、零 bias 和 signed int64 Q28 闭环
→ O_proj Q28 到 Q6.10 signed RNE、第一处残差和完整 layer0 Attention 子层闭环
```

这证明 DDR3 Controller + PHY、长 burst、片上 DRM 缓存、INT4 解包、流水 MAC16、逐组 scale、64 位定点乘加、精确部分积重构、动态 896/128 行调度、GQA head 布局、Qwen2 split-half RoPE、位置表自动推进、28 层 KV 地址调度、当前 token 写入、历史分段 burst 读取、Attention Score、causal mask、mask 感知 Softmax、PWL exp、Q31 倒数、概率归一化、概率×V Q59 加权和、多头拼接、真实 O_proj、Q28→Q6.10 重标定、第一处残差、RMSNorm、元素级非线性、Embedding、结果流式写回和 Python 自动验证可以协同工作。

当前仍不是完整 Qwen 推理；完整 layer0 Attention 子层、RMSNorm、元素级基础算子和 Embedding 已完成，尚未完成 MLP、完整 Transformer Block、模型分层调度、LM Head 和文本生成。

## 二十四、下一阶段路线

### 当前唯一下一步：G1 MLP 输入 post_attention_layernorm

1. 使用已经真实上板逐位通过的完整 Attention 子层 `[896]` signed Q6.10 输出作为输入。
2. 从真实 `.p50` 读取 `model.layers.0.post_attention_layernorm.weight`，确认 shape、FP16 gamma 和 epsilon。
3. 复用已验证 RMSNorm Q6.10、Q12.20、LUT256 rsqrt、RNE 和饱和能力，但建立独立软件清单、协议、顶层、PDS 工程和位流，不覆盖 E1 或 F6。
4. 建立连贯 MLP 输入软件参考，完成真实固定 Attention 输出、随机与饱和边界、PDS 全流程、多角时序、JTAG SRAM 和真实板卡逐位压力验证。
5. post_attention_layernorm 通过前不得进入 gate/up projection。

### 后续算子

按以下顺序逐步实现并逐层验证：

1. gate projection 与 up projection。
2. SiLU(gate) × up、down projection 和 MLP 残差。
3. 完整 Transformer Block。
4. 完整模型权重加载与分层调度。
5. tokenizer、采样与文本推理验证。

每一步都应保留“FPGA 结果与 Python 参考逐元素自动比较”的闭环，避免直接跳到完整模型后难以定位错误。


## 二十五、G1 layer0 post_attention_layernorm 独立闭环（2026-07-24）

本阶段完成 MLP 入口的真实 `post_attention_layernorm`，并保持与已验证 E1/F6 工程和位流隔离：

```text
完整 layer0 Attention 子层输出 [896] signed Q6.10
→ mean(x²) + epsilon
→ LUT256 UQ12.20 rsqrt
→ x × rsqrt × 真实 post_attention_layernorm gamma
→ RNE 与 signed int16 饱和
→ MLP 输入 [896] signed Q6.10
```

真实参数与格式：

- gamma：`model.layers.0.post_attention_layernorm.weight`，shape=`[896]`，源类型 BF16，`.p50` 中连续 FP16 1792 B；
- epsilon=`1e-6`，Q12.20 中为 `1`；
- input/gamma/output 为 signed Q6.10，平方和为 40 位，rsqrt 为 LUT256 midpoint unsigned UQ12.20；
- 所有除法和右移使用 round-to-nearest-even，输出显式饱和。

验证结果：

- 软件固定输入直接使用 F6 已上板通过的四组完整 Attention 输出，query/count=`0/1、1/2、5/6、15/16`；输入 SHA256 与 F6 清单完全一致；
- G1 四组输出 SHA256 分别为 `93d2d3ee866a7923e3ce9d450ae5d6e43a05c50daeaa952cae052c4584891f80`、`0ef1296dde8e999f6ac707725da227bd8f87b5da848a7a81113f422a03d0cbdf`、`40965e0cb4d96cf8de644d4b7081df5acef34d6c24ec8cd6d448fac4943b83aa`、`fa574c09c76580173c62d59bd5a682cd35bb97b70d25459dcf0ac6e3808e48b1`；
- 新增单元测试 5/5 PASS；完整 `model_tools` 回归 110/110 PASS；
- 软件固定清单、4608 B 载荷往返与随机/边界压力 1000/1000 PASS，seed=`20260807`；覆盖全零、交替 `INT16_MIN/MAX`、常量、稀疏、小幅值、一般和完整 int16 随机输入；
- 固定输入中 LUT256 相对精确 rsqrt 路径最大差值为 2 Q10 LSB；完整 int16 极端软件压力中最大差值为 8 Q10 LSB；
- 独立 `post_attention_layernorm_g1/pnr` 完成 Compile、Synthesize、Device Map、Place & Route、Timing 和 Bitstream，最终未布线网络为 0；
- 资源：8801 LUT、7051 FF、70 个 distributed RAM、12 DRM、9 APM、79 IO；
- 多角时序 `All Constraints Met`：慢角 core setup WNS=`+0.411 ns`、TNS=0，hold WHS=`+0.169 ns`、THS=0；快角 setup WNS=`+2.857 ns`、TNS=0，hold WHS=`+0.100 ns`、THS=0；恢复和移除无违例；
- 位流：`post_attention_layernorm_g1\pnr\generate_bitstream\rmsnorm_k896_top.sbit`，大小 2101696 B；
- 位流 SHA256：`b8c87ee10edf435617ab110cfdf0cf2a8d3c3ad3d3b91748c80ef04363305ec2`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；固件协议标识复用 `PANGU50K RMSNORM K896 V1`，DDR3 初始化成功；
- 四组连贯真实固定输入全部 896/896 上板逐位一致，合计约 2.44 秒；
- 真实 FPGA 随机/边界 300/300 PASS，seed=`20260807`，耗时 182.74 秒。

G1 MLP 输入 `post_attention_layernorm` 至此完整通过，允许进入 gate/up projection。

## 二十六、G1 gate_proj 与 up_proj 真实双投影完成记录

本轮新增独立 `mlp_gate_up_g1` 工程，未修改或覆盖已验证的 q_proj、QKV、Attention、RMSNorm 工程和位流。两路投影直接消费上一阶段已经真实上板逐位通过的 `post_attention_layernorm` `[896]` signed Q6.10 输出。

真实模型参数：

- `model.layers.0.mlp.gate_proj.weight`：shape=`[4864,896]`，group size 64，对称 signed INT4，每行 14 groups；
- `model.layers.0.mlp.up_proj.weight`：shape=`[4864,896]`，group size 64，对称 signed INT4，每行 14 groups；
- 两路 `.p50` 均不存在 bias；硬件通用 bias 槽全零；
- 两路共享完全相同的逐向量对称 INT8 激活和 activation scale；combined scale 为 UQ4.28；输出为 signed int64 Q28 `[4864]`。

软件与载荷：

- 新增 `model_tools/mlp_gate_up_reference.py`、固定清单 `model_tools/mlp_gate_up_g1_reference.json` 和 6 项单元测试；
- 四组 query/count=`0/1、1/2、5/6、15/16` 的 gate 输出 SHA256 分别为：
  - `4c1c79e14e8f788aeaaaea64924863f847ca276fd8b99b7406cfb6a50fbcea4e`
  - `42bbe0f30579275a6411abd9ad020639e0aedb9060efdccb544f5e5f4a3203c3`
  - `869b64d81d6c5f2cacc314fd869a2e20eceee7014571b0974595a81b8acf34dc`
  - `449c12f1f2904a1c4f56892a4c7049f7c785862b4c9ad9b7922b1990f161f7f6`
- 对应 up 输出 SHA256 分别为：
  - `9794e50eb90d560dfcfb55a2e54687ea3e3dcd06da368aeb557885c5e2a605a0`
  - `7eb40a12c870187737f47231342d02806f30a87076575cf4e00aecb361dfcc62`
  - `6b09b2ba30bba3ffb742cbd6ebf8a257322b78f779e5c1b86ffacdc2cb96d31d`
  - `03eca75bbbb7e9849f549124ddc1a0e4506f4bb64bb79daf769ec01d2d368041`
- 每路上传载荷 2646912 B：896 B activation、2179072 B packed weight、311296 B padded UQ4.28 scale、155648 B zero bias；结果 38912 B；
- 新增测试 6/6 PASS，完整 `model_tools` 回归 116/116 PASS；
- 软件随机/边界 1000/1000 PASS，seed=`20260808`，覆盖全零、交替 `INT16_MIN/MAX`、常量、稀疏、小幅值和一般/完整 int16 随机输入。

PDS 与位流：

- Compile、Synthesize、Device Map、Place & Route、Timing、Bitstream 全部成功，最终未布线网络为 0；
- 资源：8548 LUT、7628 FF、326 个 distributed RAM、4 DRM、12 APM、79 IO；
- 多角时序 `All Constraints Met`：慢角 core setup WNS=`+0.916 ns`、TNS=0，hold WHS=`+0.157 ns`、THS=0；快角 setup WNS=`+3.046 ns`、TNS=0，hold WHS=`+0.089 ns`、THS=0；恢复、移除和最小脉宽无违例；
- 位流：`mlp_gate_up_g1/pnr/generate_bitstream/mlp_gate_up_top.sbit`，大小 2101696 B；
- 位流 SHA256：`e72959d2968a543bf3a2bcfd31f2b2c7a0d31a9888daba9ceac2d7c50cd5db6b`。

真实上板：

- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；
- 固件 `PANGU50K MLP GATEUP V1`，DDR3 初始化成功；
- 四组连贯真实输入的 gate/up 共 8 个完整投影全部 `4864/4864` 逐位一致；
- 单路完整上传、计算和回读约 232.99~233.26 秒；
- gate 和 up 各通过全零、交替 INT16 极值、一般随机 3/3，双路合计 6/6 PASS，seed=`20260808`，mode index=`0/1/6`。

G1 gate/up 双投影至此完整通过，允许进入 `SiLU(gate)`。

## 二十七、最新项目状态与唯一下一任务

当前已完成十九级真实闭环，在完整 layer0 Attention 子层之后形成：

```text
完整 Attention 子层 signed Q6.10 输出
→ 真实 layer0 post_attention_layernorm
→ 共享 INT8 激活
├─ 真实 gate_proj [4864] signed int64 Q28
└─ 真实 up_proj   [4864] signed int64 Q28
```

当前仍不是完整 Qwen 推理；尚未完成 `SiLU(gate)`、`SiLU(gate) × up`、down projection、第二处残差，亦未完成完整 Transformer Block、全模型分层调度、LM Head 和文本生成。

### 当前唯一下一任务：G1 SiLU(gate)

1. 输入必须直接来自本阶段已真实上板逐位通过的 gate_proj `[4864]` signed int64 Q28 输出。
2. 首先固定 Q28 到 SiLU 输入格式的缩放、round-to-nearest-even 和正负饱和规则，禁止隐含截断。
3. 复用或扩展已验证 E2 SiLU 定点/PWL 定义，建立四组连贯真实输入、随机/边界软件参考和固定清单。
4. 建立独立硬件流式调度，完成 PDS 全流程、多角时序、JTAG SRAM 和真实上板逐位压力测试。
5. `SiLU(gate)` 单独全部通过前不得进入 `SiLU(gate) × up`。


## 二十八、G1 layer0 `SiLU(gate)` 独立闭环完成记录（2026-07-25）

本轮新增独立 `mlp_silu_g1` 工程，输入直接来自上一阶段已经真实上板逐位通过的 gate projection `[4864]` signed int64 Q28 输出。没有重算 gate/up，没有覆盖历史工程和位流，也没有提前执行 `SiLU(gate) × up`。

固定数据通路：

```text
gate_proj [4864] signed int64 Q28
→ 对称 signed RNE 右移 18 位
→ 显式饱和到 signed int16 Q6.10
→ E2 已验证 65 端点 / 64 段 PWL SiLU
→ SiLU(gate) [4864] signed int16 Q6.10
```

数值规则：

- Q28→Q6.10 使用正负对称 round-to-nearest-even；
- `INT64_MIN` 使用无符号二补码幅值路径，避免有符号取反溢出；
- PWL 主区间为 `[-8,8)`，步长 `0.25`；
- `x<-8 -> 0`，`x>=8 -> x`；
- 区间内插值乘积执行 signed RNE 右移 8 位，最终显式饱和到 signed int16；
- PWL64 相对精确 SiLU 的完整 int16 输入域最大误差为 4 Q10 LSB；
- 四组真实 gate 输出范围约为 `[-3.5440,2.8183]`，均未触发尾部规则或 Q6.10 饱和。

软件与固定清单：

- 新增 `model_tools/mlp_silu_reference.py`、`model_tools/mlp_silu_g1_reference.json` 和 7 项单元测试；
- 四组 query/count=`0/1、1/2、5/6、15/16` 的 `SiLU(gate)` 输出 SHA256 分别为：
  - `d3a50e88eba59160b61eccaf9a25c0d3f5dd8c5f799dbd28ede20acbd383cd18`
  - `4dc5e4f4d3240ce628ee7db071ed31faa570212a1a5dc56e5b01c69d9702d310`
  - `b807ad37514a9bd1702625666f2c13670bfa460c423a2fc53fe483a44900e9c9`
  - `4f16572a82b583edb041444edf7bdea5841ffcc3a5a7de71a28cae138f2e980e`
- 上传载荷为 39072 B：4864 个 Q28 gate 共 38912 B，65 个 PWL 端点补齐到 80 项共 160 B；结果为 4864 个 Q6.10，共 9728 B；
- 新增测试 7/7 PASS；完整 `model_tools` 回归 123/123 PASS；
- 软件随机/边界压力 1000/1000 PASS，seed=`20260809`；
- 覆盖全零、`INT64_MIN/MAX`、正负 RNE half-way tie、`±8` PWL 边界、int16 饱和边界、稀疏真实范围、一般真实范围和完整随机 int64 bit pattern；
- 上位机 `tools/pangu_mlp_silu_host.py` 的固定清单、上传载荷和软件自检 1000/1000 PASS。

RTL、DDR3 与协议：

- `mlp_silu_core.v` 使用 4 个 gate DRM bank，将 1216 个 256 bit 输入 beat 重排为 304 个输出 beat；每个输出 beat 包含 16 个 signed Q6.10；
- 每个 lane 流水完成幅值、Q28 RNE、输入饱和、PWL 读取、插值乘法、RNE、输出饱和和打包；
- `mlp_silu_ctrl.v` 实现 UART、39072 B 上传、DDR3 长 burst、片上缓存装载、结果流式写回和 9728 B 回读；
- DDR3 32 bit 地址基址：gate=`0x0000000`、PWL=`0x0003000`、result=`0x0004000`；
- 固件命令：`I/S/L/G`；固件标识：`PANGU50K MLP SILU V1`。

PDS 与位流：

- Compile、Synthesize、Device Map、Place & Route、Report Timing、Generate Bitstream 全部成功，最终未布线网络为 0；
- 资源：8024 LUT、7901 FF、70 个 distributed RAM、32 DRM、1 APM、79 IO；
- 显式 seed17/29 首版在 Fast Corner 仍有 1 条 hold `WHS/THS=-0.015 ns`，未作为验收版本；
- 改为与已验证 gate/up 工程一致的默认 PnR 策略后重新完成完整流程，最终 `Design Summary: All Constraints Met`；
- 慢角 core setup WNS=`+1.468 ns`、TNS=0，hold WHS=`+0.169 ns`、THS=0；
- 快角 core setup WNS=`+3.793 ns`、TNS=0，hold WHS=`+0.100 ns`、THS=0；
- Slow/Fast recovery、removal 和 minimum pulse width 无违例；
- 位流：`mlp_silu_g1\pnr\generate_bitstream\mlp_silu_top.sbit`，大小 2101696 B；
- 位流 SHA256：`87e643c65b70949297d54042921ac62e70454c018b6ff31f1386bbf2c8770550`。

真实上板：

- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；
- COM20 固件标识正确，DDR3 初始化成功；
- 四组连贯真实固定输入全部 4864/4864 上板逐位一致，四个输出 SHA256 与清单完全相同，合计约 17.16 秒；
- 真实 FPGA 随机/边界分六批各 50 组，累计 300/300 PASS；
- seeds=`20260809`、`20260810`、`20260811`、`20260812`、`20260813`、`20260814`；
- 每批均轮转覆盖 8 类模式，所有 4864 项结果均与 Python 金标准逐位一致。

G1 `SiLU(gate)` 至此完整通过，允许进入 `SiLU(gate) × up`。

## 二十九、最新项目状态与唯一下一任务

当前已完成二十级真实闭环，在完整 layer0 Attention 子层之后形成：

```text
完整 Attention 子层 signed Q6.10 输出
→ 真实 layer0 post_attention_layernorm
→ 共享 INT8 激活
├─ 真实 gate_proj [4864] signed int64 Q28
│  └─ Q28→Q6.10 signed RNE + PWL64 SiLU
│     └─ SiLU(gate) [4864] signed Q6.10
└─ 真实 up_proj [4864] signed int64 Q28
```

当前仍不是完整 Qwen 推理；尚未完成 `SiLU(gate) × up`、down projection、第二处残差，亦未完成完整 Transformer Block、全模型分层调度、LM Head 和文本生成。

### 当前唯一下一任务：G1 `SiLU(gate) × up`

1. 两路输入必须直接来自已经分别真实上板逐位通过的 `SiLU(gate)` `[4864]` signed Q6.10 和 `up_proj` `[4864]` signed int64 Q28。
2. 首先固定两路格式对齐、完整乘法位宽、输出定点格式、round-to-nearest-even 和正负饱和规则，禁止隐含截断。
3. 建立四组连贯真实输入、随机/边界软件参考、固定清单和上传载荷往返校验。
4. 建立独立硬件流式调度，完成 PDS 全流程、多角时序、JTAG SRAM 和真实板卡逐位压力验证。
5. `SiLU(gate) × up` 单独全部通过前不得进入 down projection。


## 三十、G1 `SiLU(gate) × up` 独立真实闭环已完成（2026-07-25）

本轮严格执行路线图的唯一下一任务，只完成 layer0 MLP `SiLU(gate) × up`，没有进入
`down_proj`、MLP 残差或完整 Transformer Block。

输入来源与数值规则：

- SiLU 输入直接复用 `mlp_silu_g1` 已真实上板逐位通过的 `[4864]` signed int16 Q6.10；
- up 输入直接复用 `mlp_gate_up_g1` 已真实上板逐位通过的 `[4864]` signed int64 Q28；
- 每项完整执行 signed 16×64 乘法，保留 signed 80 bit Q38；
- 对乘积绝对值执行 round-to-nearest-even 右移 10 位，再恢复符号；
- 最终显式饱和到 signed int64，输出保持 Q28，后续可直接作为 `down_proj` 输入；
- 不允许任何隐含截断；Python 使用任意精度整数路径独立确认 80 位边界。

软件与固定清单：

- 新增 `model_tools/mlp_silu_up_mul_reference.py`、`model_tools/mlp_silu_up_mul_g1_reference.json` 和 7 项单元测试；
- 四组 query/count=`0/1、1/2、5/6、15/16` 的完整输出 SHA256 分别为：
  - `278ceccc804b8f74266b6000745c1ae21d09cf47ba19041ff13cb5cbdaeac0ca`
  - `96e1191832febbb2bf246918e489725094567811869ab85bf8452ee8e6520fa9`
  - `9f01a9589fc9ee4f8b33acd9a64b8a767b37bee8f788697f0043ac395c7a28dc`
  - `297b982da2fb3ee7bd9202cd8d655dec200a9e19fee9a8c614e2e5412ae97802`
- 上传载荷为 48640 B：9728 B SiLU Q6.10 + 38912 B up Q28；结果为 38912 B signed Q28；
- 新增测试 7/7 PASS；完整 `model_tools` 回归 130/130 PASS；
- 软件固定清单、载荷往返和随机/边界压力 1000/1000 PASS，seed=`20260815`；
- 覆盖全零、正负 RNE half-way tie、真实范围、稀疏、完整 int16/int64 bit pattern、
  `INT64_MIN/MAX`、完整 80 位乘积和正负饱和；
- 上位机 `tools/pangu_mlp_silu_up_mul_host.py` 支持同一 seed 的 `--start-index` 分批连续回归。

RTL、DDR3 与协议：

- 新增独立 `mlp_silu_up_mul_g1` 工程，未修改任何已有验证工程或位流；
- `mlp_silu_up_mul_core.v` 使用一个 304×256 SiLU 缓存和四个 304×256 up bank；
- 单个 16×16 无符号乘法器分四个 16-bit limb 精确重构完整 80-bit 乘积；
- RNE、符号恢复和 int64 饱和均为显式硬件状态；每 4 个 int64 打包为一个 256-bit result beat；
- `mlp_silu_up_mul_ctrl.v` 实现 UART、48640 B 上传、DDR3 burst、片上缓存装载、
  1216 个结果 beat 流式写回和 38912 B 回读；
- DDR3 Controller 32-bit 地址基址：SiLU=`0x0000000`、up=`0x0001000`、result=`0x0004000`；
- 固件命令 `I/S/L/G`；固件标识 `PANGU50K MLP SILUUP V1`。

PDS 与位流：

- 默认 PnR 完成 Compile、Synthesize、Device Map、Place & Route、Timing 和 Bitstream，
  但 Fast Corner core hold 有一条 `-0.005 ns`，不作为验收版本；
- 独立 seed17/29 重新完成全流程，最终未布线网络为 0，`Design Summary: All Constraints Met`；
- 资源：7895 LUT、6910 FF、70 distributed RAM、40 DRM、1 APM、79 IO；
- 慢角 core setup WNS=`+0.511 ns`、TNS=0，hold WHS=`+0.141 ns`、THS=0；
- 快角 core setup WNS=`+3.050 ns`、TNS=0，hold WHS=`+0.065 ns`、THS=0；
- Slow/Fast recovery、removal 和 minimum pulse width 无违例；
- 验收位流：`mlp_silu_up_mul_g1\pnr_seed17\generate_bitstream\mlp_silu_up_mul_top.sbit`；
- 位流大小 2101696 B；SHA256=`a83797a8b2ec75d030fc01144e6bf51e7de0ec930fc135c1a0aba89ebf1c4336`。

真实上板：

- 仅通过 JTAG 下载到 FPGA 易失性 SRAM，识别器件 `PGL50H`；
- program 成功，`done bit=1`，未执行任何 Flash 擦除或写入；
- COM20 固件标识正确，DDR3 初始化成功；
- 四组连贯真实固定输入均为 4864/4864 上板逐位一致，四个 SHA256 与清单完全相同，
  合计耗时 30.55 秒；
- 同一固定 seed=`20260815` 的连续随机序列分四批执行，范围为 1..25、26..50、51..75、76..100；
- 真实 FPGA 随机/边界累计 100/100 PASS，每组 4864 项均与 Python 任意精度金标准逐位一致；
- 已达到 AGENTS.md 规定的至少 100 组随机/边界真实板卡压力门槛。

G1 `SiLU(gate) × up` 至此完整通过，允许进入独立 `down_proj`。

## 三十一、最新项目状态与唯一下一任务

当前已完成二十一级真实闭环，在完整 layer0 Attention 子层之后形成：

```text
完整 Attention 子层 signed Q6.10 输出
→ 真实 layer0 post_attention_layernorm
→ 共享 INT8 激活
├─ 真实 gate_proj [4864] signed int64 Q28
│  └─ Q28→Q6.10 signed RNE + PWL64 SiLU
│     └─ SiLU(gate) [4864] signed Q6.10
└─ 真实 up_proj [4864] signed int64 Q28

SiLU(gate) Q6.10 × up_proj Q28
→ 完整 signed 80-bit Q38 乘积
→ 对称 RNE >> 10 + int64 饱和
→ MLP 中间结果 [4864] signed int64 Q28
```

当前仍不是完整 Qwen 推理；尚未完成 down projection、第二处残差，亦未完成完整 MLP、
Transformer Block、全模型分层调度、LM Head 和文本生成。

### 当前唯一下一任务：G1 `down_proj`

1. 输入必须直接来自已经真实上板逐位通过的 `SiLU(gate) × up` `[4864]` signed int64 Q28。
2. 读取真实 `model.layers.0.mlp.down_proj.weight`，shape=`[896,4864]`、group size 64 的对称 signed INT4 参数，并确认 bias 是否存在。
3. 首先冻结 Q28 到逐向量 INT8 激活量化、UQ4.28 combined scale、每 64 元素点积、76 groups 跨组累加、输出 Q28/bias 和显式饱和规则。
4. 建立四组连贯真实输入、随机/边界软件参考、固定清单和完整上传载荷往返校验。
5. 建立独立硬件流式调度，完成 PDS 全流程、多角时序、JTAG SRAM 和真实板卡逐位压力验证。
6. `down_proj` 单独全部通过前不得进入 MLP 残差或完整 MLP。


## 三十二、G1 `down_proj` 独立真实闭环已完成（2026-07-25）

本轮严格执行路线图中的唯一下一任务，只完成 layer0 MLP `down_proj`，没有进入第二处残差、
完整 MLP 或完整 Transformer Block。

输入、参数与数值规则：

- 输入直接复用 `mlp_silu_up_mul_g1` 已真实上板逐位通过的 `[4864]` signed int64 Q28；
- 真实权重为 `model.layers.0.mlp.down_proj.weight`，shape=`[896,4864]`，group size 64，
  每行 76 groups 的对称 signed INT4；`.p50` 中不存在 down_proj bias；
- Q28 按实数转换为 float32，完整向量采用逐向量对称 INT8 `[-127,127]`、RNE、zero point=0；
- combined scale=`activation_scale × weight_scale`，以 unsigned UQ4.28 RNE 编码并显式饱和；
- 每 64 项执行 signed INT32 点积，76 组乘 scale 后在 signed int64 Q28 中精确累加；
- 理论最坏组累加为 `56896`，完整 76 组最坏绝对值
  `18571850900440320 < 2^63-1`，因此不允许任何隐含截断、回绕或额外饱和；
- 输出为 `[896]` signed int64 Q28；通用 bias 槽固定全零。

软件、清单与回归：

- 新增 `model_tools/mlp_down_proj_reference.py`、
  `model_tools/mlp_down_proj_g1_reference.json` 和 7 项单元测试；
- 四组 query/count=`0/1、1/2、5/6、15/16` 的完整输出 SHA256 分别为：
  - `20ada87fb91b6f3a286d554eed7ede0d369e417162683bb4828f4ba2d0a45da3`
  - `05daecd0467d77bd1cf4f48be22caaece068cd844d263b46f88e016775deacec`
  - `2e8933ddb0423cf7f7c43d7165f82ce62c128607b883f80ca942d919740a0ccf`
  - `2dcea63a160554e624edd6f1c42e28a15f17a59e4999badfabdc8a7db80a82ee`
- 完整载荷为 2499328 B：4864 B activation、2179072 B packed weight、
  286720 B padded scale、28672 B zero bias；结果为 7168 B；
- 新增单元测试 7/7 PASS，完整 `model_tools` 回归 137/137 PASS；
- 软件固定清单、载荷往返和随机/边界 1000/1000 PASS，seed=`20260816`；
- 覆盖全零、上游乘法极值/饱和、正负 RNE half-way tie、稀疏、一般范围和完整 int64 bit pattern；
- 上位机 `tools/pangu_mlp_down_proj_host.py` 支持固定 case、软件 selftest、状态、
  真实板卡压力和连续随机序列的 `--start-index`。

RTL、DDR3 与协议：

- 新增独立 `mlp_down_proj_g1` 工程，未修改或覆盖任何已有验证工程和位流；
- `mlp_down_proj_core.v` 缓存 152 拍 activation 和当前行 76 拍 packed weight，
  复用流水 MAC16，并对 304 个 16 元素 block 完成 76 组 Q28 累加；
- 80 个 scale 槽覆盖 76 个有效 combined scale 与 4 个 padding；
- `mlp_down_proj_ctrl.v` 实现 UART、2499328 B DDR3 上传、896 行调度、结果写回与回读；
- 152 拍 activation 自动拆分为最多 16 拍 burst；每行 76 拍权重拆分为
  `16+16+16+16+12`；每行 scale 固定 10 拍；每 4 行结果组成一个 256-bit beat；
- DDR3 Controller 32-bit 地址基址：activation=`0x0000000`、weight=`0x0001000`、
  scale=`0x0090000`、bias=`0x00a4000`、result=`0x00a8000`；
- 固件命令为 `I/S/L/G`，标识为 `PANGU50K MLP DOWN V1`。

PDS 与位流：

- PDS Compile、Synthesize、Device Map、Place & Route、Timing 和 Bitstream 全部成功；
- 详细布线 153 轮后未布线网络为 0；hold 修复 6 轮完成；
- 资源：8915 LUT、9426 FF、70 distributed RAM、8 DRM、12 APM、79 IO；
- `Design Summary : All Constraints Met`；
- 慢角 core setup WNS=`+0.872 ns`、TNS=0，hold WHS=`+0.110 ns`、THS=0；
- 快角 core setup WNS=`+3.026 ns`、TNS=0，hold WHS=`+0.015 ns`、THS=0；
- Slow/Fast recovery、removal 和 minimum pulse width 无违例；
- 位流：`mlp_down_proj_g1\pnr\generate_bitstream\mlp_down_proj_top.sbit`；
- 位流大小 2101696 B；SHA256=`f4d1013a287fc27003db88905f3c61e25620d213475039ddbb14900580c46757`。

真实上板：

- 仅通过 JTAG 下载到 FPGA 易失性 SRAM，识别 `PANGO USB CABLE II` 与 `PGL50H`；
- program 100%，`done bit=1`，未执行任何 Flash 擦除或写入；
- COM20 固件标识正确，DDR3 初始化成功；
- 四组连贯真实固定输入均为 896/896 上板逐位一致，四个输出 SHA256 与固定清单完全相同；
- 每组 2499328 B 上传约 216.78~216.82 秒，完整计算与 7168 B 回读约 0.65 秒；
- 真实 FPGA 随机/边界 3/3 PASS，seed=`20260816`，global index=`0..2`；
- 三组分别覆盖全零、INT16/INT64 极值经上游饱和链形成的输入、正负 RNE half-way tie；
- 每组 896 项均与 Python 金标准逐位一致。

G1 `down_proj` 至此独立完整通过，允许进入第二处残差。

## 三十三、最新项目状态与唯一下一任务

当前已完成二十二级真实闭环，在完整 layer0 Attention 子层之后形成：

```text
完整 Attention 子层 residual hidden [896] signed Q6.10
→ 真实 layer0 post_attention_layernorm
→ 共享 INT8 激活
├─ gate_proj Q28 → Q6.10 + PWL64 SiLU
└─ up_proj Q28

SiLU(gate) Q6.10 × up_proj Q28
→ 完整 signed 80-bit Q38
→ 对称 RNE >> 10 + int64 饱和
→ MLP 中间结果 [4864] signed int64 Q28
→ 真实 down_proj [896,4864] groupwise INT4
→ 76-group signed int64 Q28 精确累加
→ down_proj 输出 [896] signed int64 Q28
```

当前仍不是完整 Qwen 推理；尚未完成第二处残差和完整 MLP，亦未完成完整 Transformer Block、
全模型分层调度、LM Head 和文本生成。

### 当前唯一下一任务：G1 第二处残差

1. down 分支输入必须直接来自已经真实上板逐位通过的 `down_proj` `[896]` signed int64 Q28。
2. residual 分支必须使用进入 `post_attention_layernorm` 之前、即完整 Attention 第一处残差后的
   `[896]` signed Q6.10 hidden state，禁止错误使用归一化输出。
3. 冻结 down Q28→Q6.10 的 signed RNE 右移 18 位、int16 显式饱和、Q6.10 残差相加和
   第二次显式饱和规则。
4. 建立四组连贯真实输入、随机/边界软件参考、固定清单和上传载荷往返校验。
5. 建立独立 RTL 与流式调度，完成 PDS、多角时序、JTAG SRAM 和真实板卡逐位压力。
6. 第二处残差单独全部通过前不得勾选“完整 MLP 与软件参考比较”，也不得进入完整 Transformer Block。


## 三十四、G1 第二处残差与完整 MLP 真实闭环已完成（2026-07-25）

本轮严格执行路线图中的唯一下一任务，只完成 layer0 MLP 第二处残差；在该任务通过后，
才将“完整 MLP 与软件参考比较”标记为完成，没有进入完整 Transformer Block。

正确数据支路与数值规则：

- residual 分支使用进入 `post_attention_layernorm` 之前、即完整 Attention 第一处残差后的
  `[896]` signed int16 Q6.10 hidden state；禁止使用 post-attention RMSNorm 输出；
- down 分支使用已经真实上板逐位通过的 `down_proj` `[896]` signed int64 Q28；
- 四组输入按相同 query/count=`0/1、1/2、5/6、15/16` 严格配对；
- down Q28 对绝对值执行 signed RNE 右移 18 位并恢复符号，`INT64_MIN` 使用无符号二补码幅值；
- 重标定结果第一次显式饱和到 signed int16 Q6.10；
- 两路符号扩展相加，最终第二次显式饱和到 signed int16 Q6.10；
- 输出为完整 layer0 MLP `[896]` signed int16 Q6.10。

软件、清单与回归：

- 新增 `model_tools/mlp_residual_reference.py`、
  `model_tools/mlp_residual_g1_reference.json` 和 5 项单元测试；
- 四组最终输出 SHA256：
  - `630952eaf6fe179639773f2b60d1e9e990f380b0e698fa3d493dd7c279c96104`
  - `1cd96d92e43203abda26cace446c2664a0cbaa1fad29a8658a0d94941fbfbea7`
  - `b2365afdc2857c543628c9d15fd829005ede98aa99d35a8c4f973f61b35ed9dc`
  - `c164aab5251afc3954b8689a826a1241d1c7d6757adef5c5a232e127b59a4032`
- 每组 residual SHA256 与 F6 第一处残差输出完全一致，down SHA256 与 G1 `down_proj` 输出完全一致；
- 上传载荷为 8960 B：1792 B residual Q6.10 与 7168 B down Q28；结果为 1792 B；
- 新增测试 5/5 PASS，完整 `model_tools` 回归 142/142 PASS；
- 上位机 selftest、固定清单和载荷往返通过；
- 软件随机/边界 1000/1000 PASS，seed=`20260817`；
- 覆盖全零、正负 RNE half-way tie、`INT64_MIN/MAX`、Q10 饱和边缘、一般范围、
  第一次饱和和最终残差饱和。

RTL、DDR3 与协议：

- 新增独立 `mlp_residual_g1` 工程，未修改或覆盖任何已有验证工程和位流；
- `mlp_residual_core.v` 使用 1 个 hidden 缓存和 4 个 down bank；每个 256-bit 结果 beat
  读取 16 个 hidden 与 16 个 down 元素，并逐 lane 完成幅值、RNE、两级饱和和打包；
- `mlp_residual_ctrl.v` 实现 UART、DDR3 上传、最大 16 拍 burst 分段读取、核心调度、
  结果写回与回读；
- DDR3 Controller 32-bit 地址基址：hidden=`0x0000000`、down=`0x0001000`、
  result=`0x0003000`；
- 固件命令为 `I/S/L/G`，标识为 `PANGU50K MLP RESIDUAL V1`。

PDS 与位流：

- PDS Compile、Synthesize、Device Map、Place & Route、Timing 和 Bitstream 全部成功；
- 详细布线 89 轮后未布线网络为 0；hold 修复 3 轮；
- 资源：7705 LUT、6868 FF、70 distributed RAM、20 DRM、0 APM、79 IO；
- `Design Summary : All Constraints Met`；
- 慢角 core setup WNS=`+0.727 ns`、TNS=0，hold WHS=`+0.169 ns`、THS=0；
- 快角 core setup WNS=`+3.298 ns`、TNS=0，hold WHS=`+0.100 ns`、THS=0；
- Slow/Fast recovery、removal 和 minimum pulse width 无违例；
- 位流：`mlp_residual_g1\pnr\generate_bitstream\mlp_residual_top.sbit`；
- 位流大小 2101696 B；SHA256=`ddc424fae630fda5ab55acc8d2cb12d80b3f8cca1d5341f4a455ec0aa0a0e42b`。

真实上板：

- 仅通过 JTAG 下载到 FPGA 易失性 SRAM，识别 `PANGO USB CABLE II` 与 `PGL50H`；
- program 100%，`done bit=1`，未执行任何 Flash 擦除或写入；
- COM20 固件标识正确，DDR3 初始化成功；
- 四组连贯真实固定输入全部 896/896 上板逐位一致，四个输出 SHA256 与固定清单完全相同；
- 固定四组耗时约 3.94 秒；
- 同一 seed=`20260817` 连续随机/边界 index=`0..299` 分三批完成 300/300 PASS；
- 三批耗时约 98.79、98.78、98.74 秒；每组 896 项均与 Python 金标准逐位一致。

G1 第二处残差和完整 layer0 MLP 至此真实闭环完成。

## 三十五、最新项目状态与唯一下一任务

当前已完成二十三级真实闭环，layer0 的 Attention 子层和 MLP 子层均已分别建立连贯软件参考、
独立 PDS 工程和真实 FPGA 逐位闭环：

```text
block hidden [896] signed Q6.10
→ input RMSNorm
→ Q/K/V → RoPE → KV Cache/Attention → O_proj
→ 第一处残差 [896] signed Q6.10
→ post_attention_layernorm
→ gate_proj / up_proj
→ SiLU(gate) × up
→ down_proj [896] signed int64 Q28
→ signed RNE >> 18 + 第一次 Q6.10 饱和
→ 与第一处残差后的 hidden 相加 + 第二次饱和
→ 完整 layer0 MLP / Block 候选输出 [896] signed Q6.10
```

当前仍不是完整 Qwen 推理。虽然所有单算子和完整 Attention、完整 MLP 已分别通过，但尚未建立
一个独立工程从同一 block hidden state 连贯调度全部算子，也未完成 28 层分层调度、LM Head
或文本生成。

### 当前唯一下一任务：G2 完整 layer0 Transformer Block 集成

1. 从同一组 block hidden state 出发，建立完整 layer0 Block 软件参考并冻结全部中间张量。
2. 连贯执行 input RMSNorm、Q/K/V、RoPE、KV Cache/Attention、O_proj、第一处残差、
   post_attention_layernorm、gate/up、SiLU、逐元素乘法、down_proj 和第二处残差。
3. 复用已验证数值定义，但新建独立集成调度、DDR3 地址表、状态机和握手边界，不覆盖历史工程。
4. 建立多组真实 hidden state、随机/边界输入、固定清单和关键中间结果 SHA256。
5. 完成 PDS Compile/Synthesize/Map/PnR/Timing/Bitstream，全角 TNS/THS 必须为 0。
6. 仅通过 JTAG SRAM 下载，完成完整 Block 固定与随机/边界真实板卡逐位压力。
7. 完整 Block 单独全部通过前不得进入 28 层全模型调度、LM Head 或文本生成。


## 三十六、G2.1 完整 layer0 Transformer Block 软件全链与集成契约（2026-07-25）

本轮开始执行 G2，但只完成可独立验收的软件基线与硬件集成契约，没有宣称完整 Block RTL、PDS、位流或真实板卡已经通过。

软件全链现在从同一组 block hidden state 出发，严格连贯执行：

```text
hidden Q6.10
→ input RMSNorm
→ Q/K/V
→ RoPE
→ KV history / Attention Score / Softmax / probability×V
→ O_proj
→ 第一处残差
→ post_attention_layernorm
→ gate_proj / up_proj
→ SiLU(gate)
→ SiLU(gate) × up
→ down_proj
→ 第二处残差
→ block output Q6.10
```

新增内容：

- `model_tools/transformer_block_reference.py`：G2 完整软件金标准、DDR3 地址表、阶段 ID、握手契约和动态载荷；
- `model_tools/transformer_block_g2_reference.json`：四组真实固定用例和关键中间张量 SHA256；
- `model_tools/test_transformer_block_reference.py`：5 项集成测试；
- `transformer_block_g2/README.md`：独立工程边界、下一步 RTL 文件与完整验收标准；
- `transformer_block_g2/rtl/transformer_block_contract.vh`：控制器地址和阶段 ID 的 RTL 镜像。

固定真实用例：

- query/count=`0/1、1/2、5/6、15/16`；
- 最终 Block 输出 SHA256 分别为：
  - `630952eaf6fe179639773f2b60d1e9e990f380b0e698fa3d493dd7c279c96104`
  - `1cd96d92e43203abda26cace446c2664a0cbaa1fad29a8658a0d94941fbfbea7`
  - `b2365afdc2857c543628c9d15fd829005ede98aa99d35a8c4f973f61b35ed9dc`
  - `c164aab5251afc3954b8689a826a1241d1c7d6757adef5c5a232e127b59a4032`
- 四组结果与已经真实上板验证的 G1 第二处残差最终结果完全一致；
- 动态载荷长度分别为 2112、4160、12352、32832 B，header/hidden/trig/history K/V 往返校验通过；
- 完整软件链不同 seed/query/window 确定性压力 1/1 PASS，seed=`20260818`；
- 新增测试 5/5 PASS，完整 `model_tools` 回归 147/147 PASS。

地址与调度契约：

- 冻结 26 个 scratch/查表区域、17 个 Linear 参数区域和 21 个状态 ID；
- DDR3 Controller 地址单位仍为 32 bit；低端参数区最高结束字节地址为 `0x01823000`，未越过低端 128 MiB；
- F3 KV Cache 布局保持不变：字节基址 `0x08000000`、layer stride 32 MiB、token stride 2048 B、V offset 1024 B；
- layer27/position16383 的 V 末地址恰好为 1 GiB 末端 `0x40000000`；
- start 只在 idle 接受，busy 覆盖完整执行，done 只在最终 DDR3 写回完成后单拍产生，error/stage/timeout 必须可定位。

## 三十七、G2.1 阶段结束时状态（后续已由第三十九节更新）

在 G2.1 软件基线刚建立时，完整 Block scheduler/controller/top、共享 Linear engine、PDS 工程、时序报告、位流和真实板卡结果尚不存在，因此当时不得把“一个完整 Block 与软件参考比较”标记为完成。后续 G2.2 已完成部分硬件骨架，当前唯一任务以第三十九节为准。

### G2.1 当时计划：继续完整 layer0 Transformer Block 硬件集成

1. 在独立 `transformer_block_g2` 目录实现可覆盖 Q/K/V/O_proj/gate/up/down 的运行时参数化共享 Linear engine。
2. 按已冻结 18 个计算阶段建立顶层顺序 scheduler、明确 start/busy/done/error/watchdog 握手和 DDR3 地址切换。
3. 逐步接入已验证 RMSNorm、RoPE、KV Cache、Attention、Softmax、元素级运算与残差能力，不修改历史工程和位流。
4. 使用四组固定清单逐阶段比较中间 SHA256，再扩展真实 hidden、随机和饱和边界。
5. 完成独立 PDS Compile/Synthesize/Map/PnR/Timing/Bitstream，所有角 TNS/THS 为 0、未布线网络为 0。
6. 仅通过 JTAG SRAM 下载并完成真实 FPGA 完整 Block 逐位压力；在此之前不得进入 28 层调度、LM Head 或文本生成。


## 三十八、G2.2 共享 Linear、运行时调度与量化规格（2026-07-25）

在 G2.1 软件全链基础上，本轮继续向真实完整 Block 硬件集成推进，并发现、修正了一个不能绕过的架构问题：此前 Q/K/V、O_proj、gate/up、down_proj 的独立工程均由主机提前生成 INT8 激活和 `activation_scale × weight_scale` UQ4.28；完整 Block 的中间激活由 FPGA 自身产生，主机不可能提前知道后续 O_proj、gate/up、down_proj 的 activation scale。因此，真正的 G2 必须在 FPGA 上加入运行时激活量化和 combined scale 构建，不能只把既有算子状态机机械串联。

### 1. 精确运行时量化软件规格

新增：

```text
model_tools/runtime_linear_quant_reference.py
model_tools/test_runtime_linear_quant_reference.py
```

数值定义保持与已经真实上板验证的 G1/F6 完全一致：

- Q6.10 输入：精确执行 `round_rne(x * 127 / max_abs)`，输出 symmetric INT8 `[-127,127]`；
- Q28 输入：先严格复现原 `int64 Q28 -> float64 / 2^28 -> IEEE binary32` 舍入，再执行 symmetric INT8；禁止直接用原始 int64 max 做近似替换；
- 全零向量：activation scale 固定为 1.0，激活全部为 0；
- P50 FP16 weight scale 被解释为精确二进制有理数，与 activation scale 相乘后执行 UQ4.28 RNE 和显式饱和；
- 整数除法使用商、余数与 ties-to-even，不依赖近似浮点除法。

新增 6 项测试全部通过，覆盖：

- 正负 ties-to-even；
- Q10 全零和 int16 极值；
- Q28→binary32 的确定性；
- 真实 Q/K/V 三矩阵；
- 真实 gate/up 共享激活；
- 真实 O_proj/down_proj Q28 输入和全部 combined scale。

七个真实矩阵的全部 INT8 激活与 scale 数组逐位复现原 NumPy/G1/F6 定义。

### 2. 集成契约扩展

完整 Block 顶层不再只有 18 个显式算子阶段，而是加入四个不可省略的运行时量化阶段：

```text
INPUT_RMS
QKV_QUANT
Q_LINEAR / K_LINEAR / V_LINEAR
ROPE / KV_WRITE / ATTENTION_SCORE / SOFTMAX / ATTENTION_OUTPUT
OPROJ_QUANT / OPROJ_LINEAR / RESIDUAL1
POST_RMS
GATE_UP_QUANT / GATE_LINEAR / UP_LINEAR
SILU / SILU_UP_MUL
DOWN_QUANT / DOWN_LINEAR / RESIDUAL2
```

当前冻结：

- 22 个计算阶段；加 IDLE/DONE/ERROR 共 25 个状态 ID；
- 28 个 scratch/查表区域；
- 24 个 Linear 权重、combined scale、bias 和原始 FP16 weight-scale 区；
- 七个矩阵调用描述，明确 M/K/groups、activation/weight/raw-scale/combined-scale/bias/result 地址；
- 共享 `linear_activation_int8` scratch 和量化元数据区；
- 低端参数区最高结束字节地址 `0x018a5400`，仍远低于 128 MiB；
- F3 KV Cache 地址公式保持不变，1 GiB 边界检查继续通过。

Python 与 `transformer_block_contract.vh` 的全部地址和 25 个状态宏均由测试逐项比较，防止静默错位。

### 3. 可综合共享 Linear 与调度骨架

新增 RTL：

```text
transformer_block_g2/rtl/int4_unpack16.v
transformer_block_g2/rtl/int8_dot16_pipe.v
transformer_block_g2/rtl/shared_linear_engine.v
transformer_block_g2/rtl/runtime_linear_ctrl.v
transformer_block_g2/rtl/transformer_block_scheduler.v
```

`shared_linear_engine.v` 统一支持：

- K=896、56 个 16-element block、14 groups；
- K=4864、304 个 16-element block、76 groups；
- signed INT8 × signed INT4；
- 每组 signed int32 点积；
- unsigned UQ4.28 combined scale；
- signed int64 Q28 跨组累加和可选 bias。

`runtime_linear_ctrl.v` 统一支持 layer0 七个矩阵：

- Q：M=896、K=896、bias；
- K/V：M=128、K=896、bias；
- O_proj：M=896、K=896、无 bias；
- gate/up：M=4864、K=896、无 bias；
- down_proj：M=896、K=4864、无 bias。

控制器在一次矩阵执行中只加载一次 activation，随后逐行读取 weight/scale/bias，启动共享单行 engine，并每四行把四个 int64 Q28 合并为一个 256-bit DDR3 写拍。

22 阶段 scheduler 对每个子阶段只发一个周期 start，等待明确 done/error，带逐阶段 watchdog 和粘滞错误状态，不按固定周期猜测完成。

### 4. PDS Compile/Synthesize 结果

三个独立子模块均已使用 PDS 2022.2-SP6.4、PGL50H/FBG484 正式 Compile/Synthesize 成功，日志无硬错误：

| 子模块 | LUT | FF | DRM18K | APM |
|---|---:|---:|---:|---:|
| `shared_linear_engine` | 1557 | 3152 | 8 | 12 |
| `runtime_linear_ctrl`（含 engine） | 2377 | 4038 | 8 | 12 |
| 22 阶段 `transformer_block_scheduler` | 159 | 80 | 0 | 0 |

当前综合脚本把子模块单独作为 top，因此未约束端口和 I/O 数量警告只用于语法/结构综合检查；这些结果不代表完整 Block PnR、多角时序或资源最终值。

提交前边界审查还发现：down_proj 只有 76 个有效 groups，但 DDR3 每行按 10 个 256-bit scale beat 上传，即物理上会写入 80 个 word。共享 engine 初版把 scale RAM 深度定义为 76，虽然综合通过，却会让最后一拍的 4 个 padding word 越界。现已按历史已验证 down core 修正为 `MAX_SCALE_WORDS=((MAX_GROUPS+7)/8)*8=80`，新增源码断言测试并重新 PDS Compile/Synthesize；上表为修正后的最终资源。

### 5. 软件回归

- G2 固定清单：4/4 PASS；
- G2 完整软件链确定性压力：1/1 PASS，seed=`20260818`；
- G2 完整软件/地址/矩阵/量化目标测试：14/14 PASS；
- 完整 `model_tools` 回归：156/156 PASS。

## 三十九、G2.2 阶段开始时任务（后续已由第四十节更新）

本节记录运行时量化 RTL 开始实现前的状态。当时已经有完整软件金标准、地址/矩阵/状态契约、共享 Linear engine、DDR3 行控制器和顶层 scheduler 骨架，但尚无 FPGA 运行时量化数据路。后续量化算术核心和 DDR3 controller 已建立，当前状态与唯一任务以第四十节为准。

下一步必须按以下顺序执行：

1. 实现 Q6.10 激活 max-abs、精确 RNE 除法和 symmetric INT8 输出；
2. 实现 Q28 到 IEEE binary32 的逐位等价舍入、max-abs 和 symmetric INT8 输出；
3. 实现 FP16 weight scale 解码以及 `activation_scale × weight_scale -> UQ4.28` 的精确 RNE/饱和；
4. 建立量化 DDR3 controller，把 INT8 写入共享 scratch，把七个矩阵的 combined scale 写入冻结地址；
5. 对 QKV、O_proj、gate/up、down 四类真实输入逐位比较全部 activation/scale；
6. 量化 RTL 独立通过 PDS Compile/Synthesize/PnR/Timing 和真实板卡后，才允许连接完整 `transformer_block_ctrl/top`；
7. 完整顶层仍需独立 PDS、多角时序、JTAG SRAM 和真实 hidden/random/boundary 逐位压力，在此之前不得进入阶段 H。


## 四十、G2.3 运行时量化 RTL 与 DDR3 controller 骨架（后续已由第四十一节更新）

本轮继续完成 G2 收尾，把第三十九节中尚未存在的运行时量化数据路落实为可综合 RTL，并重新以最新源码执行软件回归和 PDS Compile/Synthesize。当前仍未宣称完整 Transformer Block、量化数值闭环、PnR、位流或真实板卡已经通过。

### 1. 运行时量化算术核心

新增 RTL：

```text
transformer_block_g2/rtl/unsigned_divider_rne.v
transformer_block_g2/rtl/runtime_q10_activation_quantizer.v
transformer_block_g2/rtl/q28_to_binary32.v
transformer_block_g2/rtl/runtime_q28_activation_quantizer.v
transformer_block_g2/rtl/runtime_fp16_scale_builder.v
```

关键数值规则：

- Q6.10 输入执行 `round_rne(abs(x) * 127 / max_abs)`，恢复符号后输出 symmetric INT8 `[-127,127]`；
- Q28 输入严格复现原软件的 `int64 -> binary64 -> /2^28 -> binary32` 双重 RNE，再按 binary32 mantissa/exponent 比值生成 INT8；
- 全零向量激活全部为 0，activation scale 固定为 1.0；
- FP16 weight scale 被精确解码为二进制有理数，与 activation scale 组合后生成 UQ4.28，使用 ties-to-even 和显式 uint32 饱和；
- 通用恢复除法器在最终商上执行 RNE，不使用近似除法。

软件侧验证：

- `unsigned_divider_rne` 状态机等价镜像随机 `60000/60000 PASS`；
- Q28 双重舍入对 10000 组随机 signed int64 和 11 个关键边界逐位匹配 NumPy；
- `test_runtime_linear_quant_reference.py` 当前 `7/7 PASS`，覆盖 Q10/Q28、全零、ties-to-even、双重舍入和七个真实矩阵全部 combined scale。

### 2. DDR3 可调用量化 controller

新增 RTL：

```text
transformer_block_g2/rtl/runtime_activation_quantizer_ctrl.v
transformer_block_g2/rtl/runtime_scale_builder_ctrl.v
transformer_block_g2/rtl/runtime_quantizer_ctrl.v
transformer_block_g2/rtl/runtime_quantizer_q28_top.v
```

已实现的结构能力：

- 从 256-bit DDR3 beat 解包 Q6.10 或 Q28 源向量；
- 驱动相应 activation quantizer，并把 32 个 INT8 重新打包为一个 256-bit beat 写回共享 `linear_activation_int8`；
- 连续读取原始 FP16 weight scale，驱动 UQ4.28 scale builder；
- 对每行 14 个有效 group 写出 16 个 word，对 76 个有效 group 写出 80 个 word，显式补齐 padding；
- 顺序调度 activation 和 scale 两段，输出 busy/done/error、失败阶段、max metadata 和饱和计数；
- Q28 参数分支由 `runtime_quantizer_q28_top.v` 固化，避免只综合默认 Q10 generate 分支。

### 3. 最新 PDS Compile/Synthesize

使用 PDS 2022.2-SP6.4、PGL50H-6IFBG484，以最新源码重新执行两类完整量化 DDR3 controller 的 Compile/Synthesize，均完成且日志无硬错误：

| 子模块 | LUT | FF | DRM18K | APM |
|---|---:|---:|---:|---:|
| Q6.10 `runtime_quantizer_ctrl` | 3494 | 2773 | 8 | 2 |
| Q28 `runtime_quantizer_q28_top` | 6500 | 3301 | 32 | 2 |

当前脚本仍把内部 AXI/controller 接口直接暴露为独立 top，因此存在未约束端口、I/O 数量超过封装、宽除法器 fanout 和 constant-probe 警告。这些结果只证明 RTL 可解析和可综合，不代表完整顶层可放置、100 MHz 时序通过或数值正确。

### 4. 最新软件与契约回归

在全部最新代码和文档更新前重新执行：

- Python `compileall`：PASS；
- 完整 `model_tools`：`157/157 PASS`；
- G2 固定清单：`4/4 PASS`；
- G2 完整软件链确定性压力：`1/1 PASS`，seed=`20260818`；
- `git diff --check`：PASS。

### 5. 当时的真实边界与下一任务（后续已完成）

本小节记录 G2.3 骨架完成时的历史边界。当时运行时量化 RTL 和 DDR3 controller 尚无真实 FPGA 逐位证据，也未完成独立 PnR/多角时序、位流和 JTAG SRAM；这些事项现已由第四十一节完成。完整 Block 仍未完成。

当时的下一任务：

1. 为 Q6.10/Q28 量化 DDR3 controller 建立自动数值闭环；
2. 使用真实 QKV、O_proj、gate/up、down 输入逐项比较全部 INT8、max metadata 和 UQ4.28；
3. 检查 14→16、76→80 padding、burst 分段、读写长度和目标地址；
4. 完成量化子系统独立 PDS Map/PnR/多角 Timing/Bitstream 和 JTAG SRAM 板卡压力；
5. 通过后再连接完整 `transformer_block_ctrl.v`、`transformer_block_top.v`、DDR3 仲裁和 host 工具；
6. 完整 Block 单独通过前不得进入 28 层调度、LM Head 或文本生成。


## 四十一、G2.4 运行时量化自动逐位板级闭环完成（2026-07-25）

本轮完成 `PROJECT_ROADMAP.md` 中此前唯一的下一任务：为 Q6.10/Q28 运行时量化 DDR3 controller 建立自动逐位闭环，并完成独立 PDS、多角时序、JTAG SRAM 和真实板卡验收。该结论只适用于运行时量化子系统；完整 layer0 Transformer Block 的 22 阶段连贯硬件闭环尚未完成。

### 1. 自动软件金标准与七矩阵清单

新增：

```text
model_tools/runtime_quantizer_validation.py
model_tools/runtime_quantizer_g2_reference.json
model_tools/test_runtime_quantizer_validation.py
tools/pangu_runtime_quantizer_host.py
```

固定清单覆盖 layer0 七个真实 Linear 调用：

```text
q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj
```

每个事务均冻结并自动核对：

- 实际 G2 source、raw FP16 scale、INT8 activation 和 combined-scale DDR3 地址；
- Q6.10 或 Q28 源向量、原始 FP16 weight scale；
- 全部 symmetric INT8 activation；
- max-abs metadata：Q10 max、binary32 mantissa/exponent/bits、all-zero；
- 每行 14→16 或 76→80 的 padded UQ4.28 combined scale；
- source/raw-scale 读取和 activation/combined-scale 写入的命令数、beat 数、burst 长度与首尾地址；
- 配置、上传和结果载荷的长度与 SHA256。

最终软件验收：

- 完整 `model_tools` 回归：`165/165 PASS`；
- 运行时量化七矩阵固定清单：`7/7 PASS`；
- 地址、burst、padding 和载荷事务压力：`1000/1000 PASS`，seed=`20260819`；
- Python `py_compile` 与 `git diff --check`：PASS。

随机压力金标准明确以精确整数/二进制有理数规格为权威。测试发现旧 NumPy 浮点复刻在数学半整数边界会出现 `63.5 -> 63.49999999999999`，从而错误舍入为 63；精确 Q10 RNE 对 `source=2047、max=4094` 应为 ties-to-even 64。固定真实矩阵仍保留与既有 NumPy/G1/F6 定义逐位一致的断言，随机/边界压力则直接对照 RTL 的精确规格。

### 2. 验证 RTL、协议与时序收敛

新增：

```text
transformer_block_g2/rtl/q28_to_binary32_sequential.v
transformer_block_g2/rtl/runtime_quantizer_trace_checker.v
transformer_block_g2/rtl/runtime_quantizer_validation_ctrl.v
transformer_block_g2/rtl/runtime_quantizer_validation_top.v
transformer_block_g2/pnr/build_runtime_quantizer_validation_ctrl.tcl
transformer_block_g2/pnr/build_runtime_quantizer_validation.tcl
transformer_block_g2/pnr/program_runtime_quantizer_validation_sram.tcl
```

验证顶层通过 UART `I/S/C/L/G` 协议完成配置、DDR3 上传、量化执行和结果回读；同一位流同时覆盖 Q6.10 与 Q28。结果包含 96 B metadata header、全部 INT8 和全部 padded UQ4.28。AXI trace checker 在硬件运行时逐命令检查地址、burst、命令数和 beat 数，错误为粘滞状态并通过状态字和结果头返回。

为满足 100 MHz，先后完成以下不改变逐位结果的多周期重构：

- Q28 int64→binary64→binary32 双重 RNE 改为顺序 MSB 搜索、逐拍规格化和两级 RNE；
- Q28 加载改为输入捕获拍与 max-abs 更新拍；
- 96 位 restoring divider 每一位拆为比较拍和更新拍，最终 ties-to-even 再拆为比较与加一两拍；
- FP16 scale builder 将 FP16 解码/35 位乘积、指数准备、96 位移位和除法拆为多周期；
- trace checker 完成信号由外层状态机明确等待，避免跨拍误判。

### 3. PDS、资源、时序和位流

最终验收使用 PDS 2022.2-SP6.4、PGL50H-6IFBG484，PnR seeds=`5/11`：

- Compile：PASS；
- Synthesize：PASS；
- Device Map：PASS；
- Place & Route：PASS；
- 详细布线：79 轮后未布线网络为 0；
- hold 修复：4 轮；
- Timing：`Design Summary : All Constraints Met`；
- Bitstream：PASS。

资源：

| 资源 | 使用量 |
|---|---:|
| LUT | 16370 / 42800（38.25%） |
| FF | 13887 / 64200（21.63%） |
| DRM18K | 40 / 134（29.85%） |
| APM | 8 / 84（9.52%） |
| I/O | 79 / 296（26.69%） |

多角时序：

| Corner | Clock | Setup WNS/TNS | Hold WHS/THS |
|---|---|---|---|
| Slow | `ref_clk` | `+14.559 ns / 0` | `+0.256 ns / 0` |
| Slow | `ddrphy_clkin` | `+0.187 ns / 0` | `+0.171 ns / 0` |
| Fast | `ref_clk` | `+16.067 ns / 0` | `+0.199 ns / 0` |
| Fast | `ddrphy_clkin` | `+2.908 ns / 0` | `+0.101 ns / 0` |

验收位流：

```text
transformer_block_g2/pnr/generate_bitstream/runtime_quantizer_validation_top.sbit
size   = 2101696 B
SHA256 = 220b771afbf8ea8d99806f3de27512748e2bd54913b1cc5e1f4a894647314236
```

### 4. JTAG SRAM 与真实板卡结果

使用 `cdt_cfg_shell.exe` 仅下载 FPGA 易失性 SRAM：

- USB Cable II 扫描到 PGL50H；
- 下载进度 100%；
- DONE bit=1；
- 未执行任何 Flash 擦除或编程命令。

固件与状态：

```text
INFO   = PANGU50K G2 QUANT V1
DDR3   = initialized
trace_error    = 0
protocol_error = 0
```

七个真实矩阵固定结果全部逐位通过：

| Matrix | Source | Result SHA256 |
|---|---|---|
| q_proj | Q6.10 | `54429490cc7504705bf30c37d5cda345e9c934d630663ecca0eb30d71a3f3e30` |
| k_proj | Q6.10 | `7070a0522562dc56740418c66a74ee2fd1937f784d49be9c7b486287847e5c16` |
| v_proj | Q6.10 | `59b18e3244da7f59f2fead479d2f2f86de363e42e097b50763f84c81691fcbe4` |
| o_proj | Q28 | `c74acb088a82291876a4d68e187d9a64c05175d77677e778d3b3469483a70aaf` |
| gate_proj | Q6.10 | `e7df51ddaae8cf74c173cde982803e61b0684343b23b3147f84be4e3b5ffb3e5` |
| up_proj | Q6.10 | `a430dffb6d2e44594d4c21701bf3ef3fcd95d6053762911d4ae6cb5eb551663d` |
| down_proj | Q28 | `27bf301cdfbd465fa08fac623b157ce95b528acf6fee8e3d665ed2c1ebbc6f88` |

固定真实矩阵：`7/7 PASS`，总耗时约 137.448 秒。每项均比较 metadata、全部 INT8、全部 UQ4.28、padding 和 AXI trace，而非只比较最终摘要。

真实 FPGA 随机/边界压力：

- Q6.10 `k_proj`：`100/100 PASS`，seed=`20260819`，约 134.651 秒；
- Q28 `o_proj`：`24/24 PASS`，seed=`20260819`，约 191.596 秒；
- 覆盖全零、signed 极值、完整随机、稀疏、幂次边界、中等动态范围和精确 half-way tie；
- 最终状态 `core_busy=0、trace_error=0、protocol_error=0`。

### 5. 当前唯一下一任务

运行时量化子系统已经独立完成软件、PDS、时序、位流和真实板卡验收。下一步只允许继续完整 layer0 Transformer Block 集成：

1. 建立统一 DDR3 仲裁；
2. 连接 `transformer_block_ctrl.v`、`transformer_block_top.v` 与 22 阶段 scheduler；
3. 从同一组 block hidden state 连贯执行到第二处残差；
4. 对全部关键中间张量和最终 `[896]` 输出逐位比较；
5. 完成完整 Block 的独立 PnR、多角时序、JTAG SRAM、固定真实 hidden 和随机/边界板级压力；
6. 完整 Block 单独通过前不得进入 28 层调度、LM Head 或文本生成。


## 四十二、G2.5 完整 layer0 Transformer Block 板级闭环完成（2026-08-03）

本轮完成 `PROJECT_ROADMAP.md` 中 G2 的最终任务：把此前分别验证的 Attention、MLP 和运行时量化模块整合为同一个 layer0 Transformer Block，并完成最新 RTL 的独立 PDS、多角时序、JTAG SRAM、四组真实 hidden 的 18 张量逐位比较，以及随机、地址末端和正负饱和边界真实板卡验收。

### 1. 最终时序收敛修复

完整 Block 首版物理实现曾出现 Attention 乘法收尾、RoPE 表写使能和 host/AXI 写回确认路径的 setup 违例。最终采用以下不改变逐位数值结果的寄存边界：

- RoPE DDR 返回只使用已寄存的 `ar_seen`，切断 `current_stage/AXI ready -> trig_mem.CE` 高扇出组合路径；
- Attention Score 的 64×64 顺序乘法器为 G2 增加分段累加和低/高 64 位两拍符号恢复，切断 128 位补码进位链；
- Attention Output 新增 `ST_ACK_RESULT`，把 DDR 写完成转换为本地寄存的一拍 `core_result_ready`，切断 scheduler/AXI owner 到 100 位累加器使能的组合路径；
- 上述参数化流水默认关闭，仅完整 G2 实例开启，因此 E1/F4/F5/F6/G1 已验证工程的默认接口和周期保持不变。

修改后重新执行 G2 聚焦回归 `30/30 PASS`，完整 `model_tools` 回归 `187/187 PASS`。

### 2. 最终 PDS、资源、路由与位流

最终验收工程：`transformer_block_g2_output_ack_fix_full`。目标器件与工具为 PDS 2022.2-SP6.4、PGL50H-6IFBG484，placement/route seeds=`5/11`。

- Compile、Synthesize、Device Map、Place & Route、Timing、Bitstream：全部 PASS；
- 详细路由：162 轮，最终未布线网络 0；
- hold 修复：6 轮；
- PnR 实际耗时 34 分 52 秒，峰值内存 2004 MB；
- Timing：`Design Summary : All Constraints Met`。

资源：

| 阶段 | LUT | FF | Distributed RAM | DRM18K | APM | I/O |
|---|---:|---:|---:|---:|---:|---:|
| Device Map | 29011 | 35053 | 332 | 52 | 36 | 79 |
| 最终 PnR | 29086 | 35053 | 332 | 52 | 36 | 79 |

正式多角时序：

| Corner | Clock | Setup WNS/TNS | Hold WHS/THS |
|---|---|---|---|
| Slow | `ref_clk` | `+11.874 ns / 0` | `+0.343 ns / 0` |
| Slow | `ddrphy_clkin` | `+0.198 ns / 0` | `+0.141 ns / 0` |
| Slow | `ioclk0/ioclk1` | `+1.692 ns / 0` | `+0.450 ns / 0` |
| Fast | `ref_clk` | `+14.220 ns / 0` | `+0.266 ns / 0` |
| Fast | `ddrphy_clkin` | `+2.640 ns / 0` | `+0.067 ns / 0` |
| Fast | `ioclk0/ioclk1` | `+1.834 ns / 0` | `+0.383 ns / 0` |

所有 slow/fast recovery、removal 和 minimum-pulse-width 检查均无违例。

验收位流：

```text
transformer_block_g2/pnr/generate_bitstream/transformer_block_top.sbit
size   = 2101696 B
SHA256 = e4c3494152498583ae4a25540363fe3e828483fa7c0012a117e26e17fc557403
```

### 3. JTAG SRAM、固件与驻留参数

`program_transformer_block_sram.tcl` 只执行 `cfg_connect/cfg_scan_chain/cfg_assign_file/cfg_program`。USB Cable II 扫描到 PGL50H，下载进度 100%、DONE bit=1，未执行 Flash 擦除或编程。固件标识为 `PANGU50K G2 BLOCK V1`，DDR3 初始化成功。

22 个 layer0 驻留参数共 7964352 B 全部写入 DDR3，包括七矩阵 INT4 权重、FP16 scale、Q/K/V bias、两组 RMS gamma、RMS/Softmax/SiLU 查表。三个最大权重的上传 SHA256：

- gate：`992b722108b986cd80fac6247c4e591ad5178f1b2437a910254822c69dccedef`；
- up：`b555608247809ec4c9c18e2603ab60953c303e039071521861b18c7394304112`；
- down：`db626c292a3b0fbf618c4d64b3a0a858e73e930f5b7beb8804319284f49de577`。

### 4. 四组固定真实 hidden：72/72 张量逐位通过

每组均从当前 hidden 开始运行完整 22 阶段 Block，并逐位回读 `input_norm、Q/K、RoPE Q/K、V、score、probability、attention concat、O_proj、第一处残差、post RMSNorm、gate/up、SiLU、SiLU×up、down_proj、block output` 共 18 个张量。

| query/count | 执行周期 | 18 张量 | 最终输出 SHA256 |
|---|---:|---:|---|
| 0/1 | 64666986 | 18/18 PASS | `630952eaf6fe179639773f2b60d1e9e990f380b0e698fa3d493dd7c279c96104` |
| 1/2 | 64672076 | 18/18 PASS | `1cd96d92e43203abda26cace446c2664a0cbaa1fad29a8658a0d94941fbfbea7` |
| 5/6 | 65030857 | 18/18 PASS | `b2365afdc2857c543628c9d15fd829005ede98aa99d35a8c4f973f61b35ed9dc` |
| 15/16 | 66006181 | 18/18 PASS | `c164aab5251afc3954b8689a826a1241d1c7d6757adef5c5a232e127b59a4032` |

累计 `4/4` 用例、`72/72` 张量逐位一致，最终 SHA256 与 Python 固定清单及 G1 第二处残差结果完全相同。

### 5. 随机、地址末端与正负饱和边界

完整 Block 地址/窗口压力 `8/8 PASS`，seed=`20260820`，约 29.114 秒。覆盖 query/count=`0/1、1/2、15/16`，query=`16383`、window=`16368..16383` 的 KV Cache 最后合法窗口，以及四组随机合法 query/window/count；每组均执行完整 Block 并逐位比较最终 `[896]` 输出。

另外构造三组显式 Q6.10 数值边界，并对每组全部 18 张量逐位比较：

| hidden 模式 | 第一处残差饱和 | 第二处残差饱和 | 板卡结果 | 输出 SHA256 |
|---|---:|---:|---|---|
| 交替 `INT16_MAX/MIN` | 434 | 443 | 18/18 PASS | `e37229b536f587bbc6ec9976d9052fc340361aec9e7768eb5e3d57aef7ff705b` |
| 全 `INT16_MAX` | 391 | 456 | 18/18 PASS | `1780b085e23605ead3905e3e141fc0890d3bb48d88acb032ca00fe3ad7389263` |
| 全 `INT16_MIN` | 403 | 431 | 18/18 PASS | `60a0503e934837bfe2671e846c2544d78356aad9041a91fb416710550f727eb9` |

数值边界累计 `3/3` 用例、`54/54` 张量 PASS，明确覆盖正向、负向及交替极值饱和。

### 6. 最终状态与下一阶段

最终 UART 状态：

```text
ddr_init_done     = 1
configured        = 1
payload_committed = 1
result_valid      = 1
block_busy        = 0
block_error       = 0
protocol_error    = 0
stage             = IDLE
error_code        = 0
```

G2 单个完整 layer0 Transformer Block 至此完成真实板级闭环。下一步允许进入阶段 H：完整模型分层调度、权重流式加载、DDR3 分区和 hidden 双缓冲。当前尚未实现或验证 28 层连续执行、最终 RMSNorm、LM Head、logits 或文本生成。
