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

## 十二、当前项目状态

当前已经完成六级真实闭环：

```text
长度16单点积
→ 固定 M=4、K=64 packed INT4 GEMV
→ 运行时参数化 M/K、尾块屏蔽的通用 packed INT4 GEMV
→ GEMV 周期计数、带宽、GMAC/s、利用率和瓶颈分析
→ 真实 q_proj M4K896 分组 UQ4.28 signed INT64 Q28 小闭环
→ 真实 layer0 q_proj M896K896 完整 Linear 层闭环
```

这证明 DDR3 Controller + PHY、长 burst、片上缓存、INT4 解包、流水 MAC16、逐组 scale、64 位定点乘加、完整输出行调度、结果流式写回和 Python 自动验证可以协同工作。

当前仍不是完整 Qwen 推理；第一个完整真实 Linear 层已经完成，尚未实现 RMSNorm、Attention、MLP、Transformer Block 和文本生成。

## 十三、下一阶段路线

### 当前唯一下一步：E1 RMSNorm

1. 从真实 `.p50` 提取 layer0 `input_layernorm.weight`，确认 K=896 的 gamma 布局。
2. 建立 Python 定点金标准，明确输入/输出格式、平方和与均值位宽、epsilon、gamma 格式、舍入和饱和规则。
3. 比较查表和 Newton-Raphson 两类 `rsqrt` 近似的误差、资源和流水延迟，选定第一版实现。
4. 新建独立 RMSNorm 工程，不覆盖任何已有验证工程和位流。
5. 完成 DDR3 输入/gamma 读取、平方和、均值、`rsqrt`、gamma 乘法和逐元素输出。
6. 完成固定向量逐元素比较、随机压力、PDS 全流程、多角时序和真实上板验证。

### 后续算子

按以下顺序逐步实现并逐层验证：

1. RMSNorm。
2. Q/K/V 线性层。
3. RoPE。
4. Attention。
5. MLP。
6. Transformer Block。
7. 完整模型权重加载与分层调度。
8. tokenizer、采样与文本推理验证。

每一步都应保留“FPGA 结果与 Python 参考逐元素自动比较”的闭环，避免直接跳到完整模型后难以定位错误。
