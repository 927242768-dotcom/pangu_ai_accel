# 盘古 PGL50H AI 大模型 FPGA 项目总路线

> 本文件是项目的**唯一权威任务清单**。后续对话和开发会话必须先读取本文件，再决定继续做什么。
>
> 最后更新：2026-07-24

## 1. 最终目标

在盘古 Logos `PGL50H-6IFBG484` 开发板上运行 Qwen2.5-0.5B + LoRA 的量化推理，完成从模型权重加载、Transformer 分层调度、KV Cache 到文本生成的完整闭环。

最终应达到：

```text
输入提示词
→ Tokenizer/Token ID
→ FPGA执行模型前向推理
→ 生成下一个Token
→ 连续自回归生成文本
```

第一阶段可以由电脑负责 Tokenizer、模型文件传输和采样；核心矩阵计算、模型层调度和 KV Cache 必须逐步迁移到 FPGA/DDR3。

## 2. 硬件、软件和模型基线

- FPGA：Pango Logos `PGL50H-6IFBG484`
- DDR3：32 位 Controller + PHY，用户侧 256 bit AXI，容量 1 GiB
- 核心时钟：100 MHz
- 串口：115200 8N1，当前开发环境常用 `COM20`
- 已有计算核：16 路有符号 INT8 MAC，记作 MAC16
- 模型：Qwen2.5-0.5B + LoRA
- INT4 模型文件：`model_output/yanbo_qwen25_0.5b_int4.p50`
- 模型元数据：`model_output/yanbo_qwen25_0.5b_int4.json`
- 大模型文件约 251.63 MiB，不提交到 Git

## 3. 完成状态图例

- `[x]`：已经完成，并具有真实验证证据
- `[ ]`：尚未完成
- `[~]`：正在开发或仅部分完成

任务只有同时满足以下条件才能从 `[ ]` 改成 `[x]`：

1. Python 参考模型结果一致；
2. PDS 编译、综合、Device Map、布局布线成功；
3. 快慢角建立/保持/恢复/移除时序全部通过，TNS=0；
4. 位流真实下载到开发板 SRAM；
5. 固定向量和随机压力测试通过；
6. 文档、协议和地址布局已同步更新。

---

# 4. 已完成的硬件基础

## 阶段 A：INT8 MAC16 基础核

- [x] UART 固件信息读取
- [x] MAC16 自检
- [x] 16 维 INT8 × INT8 点积
- [x] Python 自动比较
- [x] 多轮随机压力测试
- [x] 真实上板验证

## 阶段 B：完整 DDR3 基础

- [x] 使用正确的 PGL50H、FBG484 和 32 位 DDR3 Controller + PHY
- [x] DDR3 初始化和训练
- [x] 完整 1 GiB 地址空间顺序写入与读回
- [x] 地址相关数据校验
- [x] PDS 全流程和多角时序通过
- [x] JTAG SRAM 下载
- [x] 上板状态 `test_main_state=5`、`err_cnt=0`

已验证位流：

```text
ipcore/pangu_ddr3_x32/pangu_ddr3_x32/pnr/generate_bitstream/test_ddr.sbit
```

## 阶段 C：DDR3 + MAC16 + INT4 集成闭环

- [x] 上位机经 UART 写入激活和权重
- [x] FPGA 将数据写入 DDR3
- [x] 一次 2 拍 × 256 bit AXI burst 读取激活和权重
- [x] 片上寄存缓冲与数据拆分
- [x] INT8 权重直接进入 MAC16
- [x] 每字节两个有符号 INT4 权重解包
- [x] INT4 二补码符号扩展为 INT8
- [x] INT4 权重 × INT8 激活点积
- [x] 32 位结果写回 DDR3
- [x] UART 返回结果并与 Python 比较
- [x] MAC 输入一级流水，修复 INT4 路径时序违例
- [x] 最终时序 `All Constraints Met`
- [x] INT8 固定向量和 1000 轮随机测试
- [x] INT4 固定向量和 1000 轮随机测试

最终集成位流：

```text
ddr_mac16_integration/pnr/generate_bitstream/ddr_mac16_top.sbit
SHA256: e625e6dbe0e7f49915b41be805a970ea3977a72a6cb189f98c50497371b0af9f
```

---

# 5. 阶段 D1：通用 packed INT4 GEMV（已完成）

## 阶段 D1：实现通用 packed INT4 GEMV `y = W × x`

**在完成本阶段前，不进入 RMSNorm、Attention 或完整模型。**

目标：不再只计算长度 16 的一个点积，而是支持：

```text
W: M × K 的 packed INT4 矩阵
x: K 维 INT8 激活向量
y: M 维 INT32 累加结果
```

### D1.1 先实现固定小尺寸 GEMV

建议第一版固定：

```text
M = 4
K = 64
```

任务：

- [x] 设计 GEMV DDR3 地址布局
- [x] 激活 `x` 写入 DDR3，并只读取/缓存一次
- [x] 4 行 packed INT4 权重连续写入 DDR3
- [x] 每行 K=64，拆成 4 个 MAC16 分块
- [x] 每行跨 4 个分块进行 INT32 累加
- [x] 生成 4 个 INT32 输出
- [x] 输出向量批量写回 DDR3
- [x] UART 返回整个输出向量
- [x] Python 对 4 个输出逐元素比较
- [x] 固定向量通过
- [x] 至少 1000 轮随机压力测试通过
- [x] PDS 全流程、时序和真实上板通过

D1.1 验证证据（2026-07-23）：

- 独立工程：`gemv_int4_m4k64`
- Python 金标准自检：1000/1000 PASS，seed=`20260725`
- 固定向量：FPGA `[1376, -1344, 416, 256]`，Python 完全一致
- 真实上板随机压力测试：1000/1000 PASS，耗时约 19.70 秒
- PDS：编译、综合、Device Map、布局布线、时序分析、位流生成全部成功
- 布局布线：0 条未布线网络
- 多角时序：`All Constraints Met`，慢速角 100 MHz WNS=`+0.983 ns`、TNS=`0`
- 位流：`gemv_int4_m4k64/pnr/generate_bitstream/gemv_m4k64_top.sbit`
- SHA256：`349a26b45362778849868e68475c5b8f6620bc8edb8375ebb237efbab4d352ed`
- JTAG SRAM 下载：100%，`done bit=1`，未操作 Flash

### D1.2 扩展为参数化 GEMV

- [x] 支持运行时参数 `M` 和 `K`
- [x] `K` 不是 16 整数倍时支持尾块屏蔽
- [x] 支持更长的 AXI 256 bit burst
- [x] 权重行地址自动递增
- [x] 输出地址自动递增
- [x] 32 位累加溢出边界测试
- [x] UART 协议增加 GEMV 配置和启动命令
- [x] Python 工具可自动产生不同 M/K 的随机矩阵
- [x] 至少覆盖 `M={1,4,16,64}`、`K={16,64,256,896}` 的测试

D1.2 验证证据（2026-07-23）：

- 独立工程：`gemv_int4_param`，未覆盖固定 M4K64 已验证工程和位流
- 支持范围：`1 <= M <= 64`、`1 <= K <= 896`
- 激活读取：最多 16 拍 AXI burst，超过 16 拍自动分段；K=896 共 28 拍
- 权重读取：按行 burst，行地址自动递增；输出每 8 个 INT32 一拍写回，地址自动递增
- 尾块：最后一个 MAC16 分块按真实 K 显式屏蔽无效激活字节和 INT4 半字节
- Python 金标准自检：1025 例 PASS，含标准尺寸、尾块尺寸和固定 M4K64 回归，seed=`20260728`
- 多尺寸真实上板：24 种形状、72 例全部 PASS；标准组合完整覆盖 `M={1,4,16,64}`、`K={16,64,256,896}`
- 尾块上板覆盖：`K={1,15,17,63,65,255,257,895}`
- 固定 M4K64 回归：1000/1000 PASS，seed=`20260730`，约 19.89 秒
- 尾块 M16K65：1000/1000 PASS，seed=`20260731`，约 105.27 秒
- 近最大尾块 M4K895：100/100 PASS，seed=`20260801`，约 23.90 秒
- INT32 边界：FPGA `[917504, -802816, 57344, 57344]` 与 Python 一致；当前范围理论绝对上界 `917504`
- PDS：编译、综合、Device Map、布局布线、时序分析、位流生成全部成功，0 条未布线网络
- 资源：LUT=`10715`、Register=`8136`、DRM18K=`4`、APM=`9`
- 多角时序：`All Constraints Met`；慢速角 100 MHz WNS=`+0.682 ns`、TNS=`0`，WHS=`+0.086 ns`、THS=`0`
- 快速角：WNS=`+3.137 ns`、TNS=`0`，WHS=`+0.001 ns`、THS=`0`
- 位流：`gemv_int4_param/pnr/generate_bitstream/gemv_param_top.sbit`
- SHA256：`90c67a74841826b358f4a4de5e0783c587de01a296d7991c3b2a8d3fc1bcd2a3`
- JTAG SRAM 下载：100%，`done bit=1`，未操作 Flash

### D1.3 GEMV 性能基础设施

- [x] 统计 DDR3 读取周期
- [x] 统计 MAC 计算周期
- [x] 统计单次 GEMV 总周期
- [x] 增加性能计数器并可由上位机读取
- [x] 记录实测带宽、GMAC/s 和利用率
- [x] 明确瓶颈是 DDR3、MAC 数量还是控制开销

D1.3 验证证据（2026-07-23）：

- 独立构建目录：`gemv_int4_perf`，未覆盖 D1.2 已验证位流
- 固件协议：升级为 `PANGU50K GEMV PARAM V2`，新增 `P` 命令返回 4 个 `uint32_le` 周期计数
- 计数口径：激活读取、全部权重读取、核心 `busy` 计算周期，以及从激活读取开始到最后结果写回完成的总周期
- Python 金标准与性能计算公式自检：1025 例 PASS，seed=`20260728`
- M4K64 实测：激活读取 32 周期、权重读取 116 周期、MAC 64 周期、总计 244 周期；合并读取带宽 `129.73 MB/s`，核心 `0.4000 GMAC/s`，端到端 `0.1049 GMAC/s`，主瓶颈为 DDR3 读取
- M16K65 尾块实测：33/480/320/919 周期；合并读取带宽 `218.32 MB/s`，核心 `0.3250 GMAC/s`，端到端 `0.1132 GMAC/s`，主瓶颈为 DDR3 读取
- M64K896 最大尺寸实测：86/3152/14336/17912 周期；合并读取带宽 `913.16 MB/s`，核心 `0.4000 GMAC/s`，端到端 `0.3201 GMAC/s`，主瓶颈转为 MAC 数量/计算
- MAC16 理论峰值按 16 路、100 MHz 计为 `1.6 GMAC/s`；最大尺寸核心利用率 `25.00%`，端到端利用率 `20.01%`
- 多尺寸真实上板：24 种形状、72 例全部 PASS
- 固定 M4K64：1000/1000 PASS，seed=`20260730`，约 19.79 秒
- 尾块 M16K65：1000/1000 PASS，seed=`20260731`，约 105.26 秒
- 近最大尾块 M4K895：100/100 PASS，seed=`20260801`，约 23.90 秒
- INT32 边界：FPGA `[917504, -802816, 57344, 57344]` 与 Python 一致
- PDS：编译、综合、Device Map、布局布线、时序分析、位流生成全部成功，0 条未布线网络
- 资源：LUT=`10906`、Register=`8269`、DRM18K=`4`、APM=`9`
- 多角时序：`All Constraints Met`；慢速角 100 MHz WNS=`+0.589 ns`、TNS=`0`，WHS=`+0.142 ns`、THS=`0`
- 快速角：WNS=`+3.074 ns`、TNS=`0`，WHS=`+0.065 ns`、THS=`0`
- 位流：`gemv_int4_perf/pnr/generate_bitstream/gemv_param_top.sbit`
- SHA256：`a727f7427143b874da278ae83d7e8a2cdeff8b82bd7c0bb4361e7a2efed73c35`
- JTAG SRAM 下载：100%，`done bit=1`，未操作 Flash

### D1 验收标准

必须形成以下闭环：

```text
Python生成M×K INT4矩阵和K维INT8向量
→ 写入DDR3
→ FPGA连续burst读取
→ 多次MAC16分块累加
→ 得到M维输出
→ 写回DDR3并返回
→ Python逐元素完全一致
```

---

# 6. GEMV 之后的完整开发路线

## 阶段 D2：真实量化格式与模型张量

目标：从“自定义随机 INT4”转向模型文件中的真实权重格式。

- [x] 完整解析 `.p50` 文件头、张量目录和数据偏移
- [x] 验证 JSON 元数据与二进制张量完全一致
- [x] 明确每个线性层的权重形状和存储顺序
- [x] 明确 INT4 编码方式、分组大小、scale 和 zero point
- [x] Python 可提取任意一行/一块真实模型权重
- [x] FPGA GEMV 支持真实模型的分组反量化或定点缩放
- [x] 选择统一的激活量化格式
- [x] 定义 scale 的定点格式，例如 Q 格式
- [x] 验证一个真实线性层的小切片与 PyTorch/NumPy 一致
- [x] 验证一个完整真实线性层输出误差在规定范围内

D2 模型格式解析验证证据（2026-07-23）：

- 新增轻量解析库：`model_tools/p50_format.py`，只依赖 NumPy
- 新增命令行工具：`model_tools/p50_inspect.py`，支持 `verify/summary/list/describe/row/block`
- 真实镜像：`263,857,920` 字节，SHA256=`f0c0a22886499715fe16832b88ac59bff48fea8f3069c247437726aca6f19e9d`
- 固定头：magic=`P50Q4V1\0`、version=`1`、header size=`4096`、metadata size=`63716`、data offset=`528384`
- 张量目录：共 `290` 个，其中 `169` 个分组 INT4、`121` 个 FP16；名称唯一
- 外部 JSON 与镜像内嵌 JSON：逐字段完全一致
- 全量派生校验：shape、padded columns、groups、data/scale 长度、4 KiB/64 B 对齐、范围和互不重叠全部 PASS
- 真实量化格式：每输出行 row-major、group size=`64`、低半字节在前、4 位二补码、范围 `[-7,7]`、FP16 scale、对称量化 zero point=`0`
- 真实张量提取：完整 INT4 行、跨 group 二维块和 FP16 行均通过
- 独立微型镜像单元测试：5/5 PASS
- 原 BF16 + LoRA 软件参考抽样：4 组反量化误差全部位于理论半 scale 舍入上限内
- 本阶段未修改 FPGA RTL、PDS 工程或任何已验证位流

D2 真实 Linear 量化软件参考验证证据（2026-07-23）：

- 新增 `model_tools/linear_quant_reference.py`，定义真实 P50 INT4 Linear 的三条独立参考路径：P50 反量化浮点基线、INT8 激活量化浮点参考、UQ4.28 硬件等价定点参考
- 激活统一格式：逐向量对称 INT8，范围 `[-127,127]`，zero point=`0`，scale=`max(abs(x))/127`，全零向量 scale=`1.0`
- 所有浮点转整数统一采用 round-to-nearest-even（RNE），随后饱和
- 主机预计算 `combined_scale = activation_scale * weight_scale[row,group]`，保存为 32 位无符号 `UQ4.28`
- FPGA 精确定义：每个 64 元素 group 先产生 INT32 点积；`acc_int32 * combined_scale_uq4_28` 后在带 28 位小数的有符号 INT64 中跨组累加，并加入 `bias_q28`
- 理论定点误差上界：`(sum(abs(group_acc)) + 1) * 0.5 / 2^28`
- 真实张量切片：layer0 `q_proj` 输出行 `0..3`、完整输入列 `0..895`，即 M=4、K=896、14 groups
- 固定激活：跨平台 LCG，seed=`20260723`，激活 scale=`0.0314826064222441`，INT8 饱和数=`0`
- 组合 scale 实际范围：`0.0001496403793..0.0004270635545`，UQ4.28 饱和数=`0`
- P50 浮点基线：`[0.7752590203, -0.6386315781, 1.0810645018, -0.8347725510]`
- 量化激活浮点参考：`[0.7720806824, -0.6458171611, 1.0714217223, -0.8315785984]`
- 定点 Q28 输出：`[207253689, -173360554, 287606739, -223225713]`
- 定点反量化输出：`[0.7720801570, -0.6458183900, 1.0714185946, -0.8315805830]`
- 激活量化最大绝对误差=`0.0096427795`
- UQ4.28 最大绝对误差=`3.1277186e-6`，小于理论上界 `3.8200990e-5`
- 固定向量清单：`model_tools/q_proj_m4k896_reference.json`，包含关键数组 SHA256；完整 NPZ 可由真实镜像确定性生成
- 原有解析测试与新增量化测试共 `13/13 PASS`
- 随机软件压力测试：`1000/1000 PASS`，seed=`20260723`
- 本阶段未修改 FPGA RTL、PDS 工程或任何已验证位流

D2 真实分组 UQ4.28 FPGA 小闭环验证证据（2026-07-24）：

- 新建独立工程：`gemv_int4_group_q28`，未覆盖 `gemv_int4_param`、`gemv_int4_perf` 或其已验证位流
- 固定真实验收对象：layer0 `q_proj` 输出行 `0..3`、输入列 `0..895`，M=4、K=896、group size=64
- UART 固定载荷共 2976 B：896 B 激活、1792 B packed INT4 权重、256 B UQ4.28 scale、32 B bias_q28
- FPGA 每 64 元素 group 执行 4 次流水 MAC16，产生 signed INT32 group 点积
- UQ4.28 scale 以 unsigned uint32 保存，硬件零扩展后与 signed INT32 相乘，并在 signed INT64 Q28 中跨 14 组累加
- 4 个 signed int64 输出写回 DDR3，并通过 UART 返回 Python 逐位比较
- Python 载荷、打包/解包和精确定点金标准：`1000/1000 PASS`，seed=`20260724`
- 固定真实向量 FPGA 输出：`[207253689, -173360554, 287606739, -223225713]`，与软件参考逐位完全一致
- scale bit31 和 `0xFFFFFFFF` 边界向量：真实上板 PASS
- 随机分组 scale 真实上板压力测试：`1000/1000 PASS`，seed=`20260724`，约 266.06 秒
- 首版组合 MAC16 慢角 WNS=`-0.109 ns`、TNS=`-0.163 ns`；改为显式平衡流水后全部修复
- PDS 全流程成功，最终未布线网络 0；多角时序 `All Constraints Met`
- 100 MHz 慢速角建立 WNS=`+0.909 ns`、TNS=0；保持 WHS=`+0.111 ns`、THS=0
- 快速角建立 WNS=`+3.041 ns`、TNS=0；保持 WHS=`+0.051 ns`、THS=0
- 资源：8379 LUT、7492 FF、4 个 DRM、12 个 APM
- 位流：`gemv_int4_group_q28/pnr/generate_bitstream/gemv_group_q28_top.sbit`
- SHA256：`d8c7d194d4d8ce1e5d189df39fae5fc904030fe4be6e981a5876a4df73ea17bd`
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash

D2 完整真实 Linear 层验证证据（2026-07-24）：

- 新建独立工程：`gemv_int4_qproj_full`，未覆盖任何已有验证工程或位流
- 完整验收对象：layer0 `q_proj` 全部输出行和完整输入列，即 M=896、K=896、group size=64、每行 14 groups
- Python 固定载荷共 `488320 B`：896 B 激活、401408 B packed INT4 权重、57344 B padded UQ4.28 scale、28672 B padded bias_q28
- Python 从真实 `.p50` 一次性提取完整权重、FP16 scale 和 bias，并复用模型数据生成不同激活的逐行 signed int64 Q28 金标准
- 完整层载荷打包/解包、补齐区域和独立 Q28 重算全部通过
- 固定完整层输出 SHA256=`ea1f04bf4ff313dad07025ff35e66a088f13afd28d817422b89bb135f63525a0`
- 固定输出前 4 行与已验证 M4K896 小闭环逐位一致
- 软件随机激活压力测试：`1000/1000 PASS`，seed 起点=`20260725`，约 25.88 秒
- FPGA 逐行读取 14 拍权重、2 拍 scale 和 1 拍 padded bias；每 4 行结果组成一个 256 bit 数据拍立即写回 DDR3，不缓存完整输出向量
- 固定完整层真实上板：896 个 signed int64 与 Python Q28 金标准逐位完全一致；上传、计算和回读约 43.03 秒
- 随机激活完整层真实上板回归：`3/3 PASS`，seed=`20260725..20260727`，约 130.13 秒
- PDS 编译、综合、Device Map、布局布线、时序分析和位流生成全部成功，最终未布线网络 0
- 资源：8510 LUT、7619 FF、4 DRM、12 APM
- 多角时序：`All Constraints Met`；慢角 100 MHz WNS=`+0.670 ns`、TNS=0，WHS=`+0.171 ns`、THS=0；快角 WNS=`+3.034 ns`、TNS=0，WHS=`+0.100 ns`、THS=0
- 恢复、移除和最小脉宽均无违例
- 位流：`gemv_int4_qproj_full/pnr/generate_bitstream/gemv_qproj_full_top.sbit`
- SHA256：`432454b80678c11f493856cb725d791e271d86eada1b5cabccefc0d7486f8894`
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash

验收：已完成模型中的第一个完整真实 Linear 层，输出与软件量化参考逐位一致。D2 阶段完成。

## 阶段 E：基础非矩阵算子

### E1 RMSNorm

- [x] 平方和累加
- [x] 均值计算
- [x] `rsqrt` 近似方案确定：查表、Newton-Raphson 或软件辅助
- [x] gamma 权重乘法
- [x] 定点格式和饱和/舍入规则
- [x] Python 逐元素比较
- [x] 随机压力测试、时序和上板验证

E1 验证证据（2026-07-24）：

- 独立工程：`rmsnorm_k896`，未覆盖任何已有验证工程和位流；
- 真实 gamma：`model.layers.0.input_layernorm.weight`，连续 FP16、长度 K=896；
- 算子：`gamma * x * rsqrt(mean(x^2) + epsilon)`，`epsilon=1e-6`；
- 定点格式：输入/gamma/输出为 signed Q6.10 int16，平方和 40 位，均值/epsilon 为 Q12.20，rsqrt 为 UQ12.20 uint32；
- 所有浮点转整数、除法和右移使用 RNE，输出显式饱和；
- rsqrt 比较：256 项中点 LUT 与 32 项种子 LUT + 一次 Newton-Raphson；第一版选择 LUT256；
- 固定标量：`sum_squares=5176164753`、`variance_q20=5776971`、`lut_rsqrt_q20=446797`；
- 固定输出 SHA256：`1f52890780e0f4cc0f734d47a4e3bdb28c3c964b8734b442d7781d4ca155a4f0`；
- 软件相关单元测试：23/23 PASS；RMSNorm 软件随机压力：1000/1000 PASS，seed=`20260726`；
- DDR3 闭环：上传 4608 B，读取输入/gamma/LUT，计算 896 个输出，写回 DDR3 后通过 UART 返回；
- 固定真实上板：896 个 signed Q6.10 输出与 Python LUT256 金标准逐位一致，端到端约 0.61 秒；
- 真实随机上板：300/300 PASS，seed=`20260726..20261025`，约 183.11 秒；
- PDS：编译、综合、Device Map、布局布线、时序分析和位流生成全部成功，最终未布线网络 0；
- 资源：LUT=`8801`、FF=`7051`、DRM=`12`、APM=`9`；
- 多角时序：`All Constraints Met`；慢角 100 MHz WNS=`+0.374 ns`、TNS=0，WHS=`+0.171 ns`、THS=0；快角 WNS=`+2.832 ns`、TNS=0，WHS=`+0.100 ns`、THS=0；
- 恢复、移除和最小脉宽均无违例；
- 位流：`rmsnorm_k896/pnr/generate_bitstream/rmsnorm_k896_top.sbit`；
- SHA256：`94c82d1ef6adf563043c6f90f5744ec258156d85c6db134389132ae4f2938b11`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash。

### E2 元素级运算

- [x] 残差加法
- [x] 定点乘法和缩放
- [x] 饱和与舍入
- [x] SiLU 或 `x·sigmoid(x)` 近似
- [x] element-wise multiply
- [x] Python 参考与误差阈值

E2 验证证据（2026-07-24）：

- 独立工程：`elementwise_k896`，未覆盖任何已验证 GEMV、Linear 或 RMSNorm 工程和位流；
- 统一格式：输入 A/B、标量 scale 和输出均为 signed Q6.10 int16；
- 残差加法使用扩展加法和显式 signed int16 饱和；缩放与元素乘法使用 signed Q12.20 乘积、RNE 右移 10 位和显式饱和；
- SiLU 在完整 65536 个 int16 输入上比较 LUT2048 与 64 段端点 PWL：LUT 最大误差 5 Q10 LSB、表容量 32768 bit；PWL 最大误差 4 Q10 LSB、端点表 1040 bit，第一版选择 PWL64；
- SiLU 覆盖 `[-8,8)`，尾部规则为 `x<-8 -> 0`、`x>=8 -> x`；
- E2 单元测试 11/11 PASS，完整 `model_tools` 回归 34/34 PASS；
- 软件与上传载荷随机压力：1000/1000 PASS，seed=`20260727`；
- 固定边界向量：残差、缩放、元素乘法和 SiLU 四种操作，每种 896 个输出均与 Python 逐位一致，端到端约 1.01 秒；
- 固定输出 SHA256：residual=`dd6cf26e917004e52973ee8506bfdc2e403dac2d31e64abba9c6cd4619196dca`，scale=`8137acd3e9c983380ef1d024858e88ed54b675791cf416539ca3b03fa9c3455c`，multiply=`f07847b17449eb401324b413b4df7765d14377e9b20c340f48e6dc87112f25aa`，SiLU=`1933e7c436030c00285bffb2def77c70c979b32c041af3833f61fa25825fdbf8`；
- 真实随机上板：分三批累计 300/300 PASS，seed=`20260727..20261026`，合计约 312.49 秒；
- PDS 编译、综合、Device Map、布局布线、时序分析和位流生成全部成功，最终未布线网络 0；
- 资源：LUT=`7872`、FF=`7778`、distributed RAM=`70`、DRM=`8`、APM=`2`；
- 多角时序：`All Constraints Met`；慢角 100 MHz WNS=`+0.580 ns`、TNS=0，WHS=`+0.112 ns`、THS=0；快角 WNS=`+2.951 ns`、TNS=0，WHS=`+0.051 ns`、THS=0；
- 恢复、移除和最小脉宽均无违例；
- 位流：`elementwise_k896/pnr_seed17/generate_bitstream/elementwise_k896_top.sbit`；
- SHA256：`809b436f1c369d66a20c5f2faaa8e684a15a3963d659b95d080e342c3a7d9d50`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；固件为 `PANGU50K ELEMENTWISE K896 V1`，DDR3 初始化成功；
- 开发中修复了 SiLU 长组合路径建立违例，以及最高段 `63+1` 的 6 位索引回绕问题。

### E3 Embedding/查表

- [x] Token ID 到 embedding 行地址映射
- [x] DDR3 中读取一个 token 的 embedding
- [x] 转换为统一激活格式
- [x] 与软件参考比较

E3 验证证据（2026-07-24）：

- 独立工程：`embedding_k896`，未覆盖任何已验证 GEMV、Linear、RMSNorm 或元素级工程和位流；
- 真实 tied embedding：`model.embed_tokens.weight`，shape=`[151936,896]`、group size=64、每行 14 groups，Token ID 有效范围 `0..151935`；
- DDR3 行槽：每个 Token 固定 512 B/16 拍，控制器地址 `token_id << 7`；前 448 B 为 packed signed INT4，后 56 B 为 14 个 UQ4.28 scale，末尾 8 B padding；
- 真实全部 FP16 embedding scales 均可被 UQ4.28 精确表示；硬件执行 signed INT4 × unsigned UQ4.28，RNE 右移 18 位后显式饱和为 signed Q6.10 int16；
- E3 单元测试 11/11 PASS，完整 `model_tools` 回归 45/45 PASS；
- 真实 P50 软件/载荷随机压力：1000/1000 PASS，seed=`20260728`，最大 Q6.10 量化误差 `0.00048828125`；
- 固定 Token ID `[0,1,2026,151935]` 的 896 个输出真实上板逐位一致，覆盖最低、相邻、普通和最大 Token ID；
- 真实随机 Token ID 上板压力：300/300 PASS，seed=`20260728`，约 75.53 秒；
- PDS 编译、综合、Device Map、布局布线、时序分析和位流生成全部成功，最终未布线网络 0；
- 资源：LUT=`7637`、FF=`7380`、distributed RAM=`326`、APM=`2`、DRM=`0`；
- 多角时序：`All Constraints Met`；慢角 100 MHz WNS=`+0.679 ns`、TNS=0，WHS=`+0.172 ns`、THS=0；快角 WNS=`+2.964 ns`、TNS=0，WHS=`+0.101 ns`、THS=0；
- 恢复、移除和最小脉宽均无违例；
- 位流：`embedding_k896/pnr/generate_bitstream/embedding_k896_top.sbit`；
- SHA256：`cd0e138e494875035cf5c66d76eaf250729625c172bf51c935b831d31c45c0fa`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；固件为 `PANGU50K EMBEDDING K896 V1`，DDR3 初始化成功。

## 阶段 F：Attention 数据通路

### F1 Q/K/V 线性层

- [ ] 用通用 GEMV 实现 Q 投影
- [ ] 用通用 GEMV 实现 K 投影
- [ ] 用通用 GEMV 实现 V 投影
- [ ] 支持多头/分组查询注意力的张量布局
- [ ] 与软件参考逐元素比较

### F2 RoPE

- [ ] 生成或加载 sin/cos 表
- [ ] 偶数/奇数维旋转
- [ ] 支持位置索引递增
- [ ] 定点误差验证

### F3 KV Cache

- [ ] 定义 DDR3 中每层 K/V Cache 地址布局
- [ ] 当前 token 的 K/V 写入
- [ ] 历史 token 的 K/V 顺序读取
- [ ] 支持上下文长度边界检查
- [ ] 防止层间和 token 间地址覆盖

### F4 Attention Score

- [ ] Q·K 点积
- [ ] 缩放 `1/sqrt(head_dim)`
- [ ] causal mask
- [ ] 支持多头循环调度

### F5 Softmax

- [ ] max reduction
- [ ] 减最大值
- [ ] exp 近似或查表
- [ ] sum reduction
- [ ] reciprocal/归一化
- [ ] 长序列数值稳定性测试

### F6 Attention 输出

- [ ] softmax 权重与 V 的加权和
- [ ] 多头拼接
- [ ] 输出投影 `O_proj`
- [ ] 残差连接
- [ ] 完整 Attention 子层与软件参考比较

## 阶段 G：MLP 和 Transformer Block

### G1 MLP

- [ ] gate projection
- [ ] up projection
- [ ] SiLU(gate)
- [ ] SiLU(gate) × up
- [ ] down projection
- [ ] 残差连接
- [ ] 完整 MLP 与软件参考比较

### G2 单个 Transformer Block

- [ ] 输入 RMSNorm
- [ ] Q/K/V
- [ ] RoPE
- [ ] Attention
- [ ] O projection
- [ ] 第一处残差
- [ ] 第二个 RMSNorm
- [ ] MLP
- [ ] 第二处残差
- [ ] 一个完整 Block 与软件参考比较
- [ ] 多组随机输入和真实 hidden state 验证

## 阶段 H：完整模型分层调度

- [ ] 建立模型层描述表
- [ ] 为每个张量记录 DDR3/主机文件偏移、形状和量化参数
- [ ] 设计权重流式加载方案
- [ ] 决定模型权重是否常驻 DDR3 或按层重载
- [ ] 设计 1 GiB DDR3 内存分区
- [ ] hidden state 双缓冲
- [ ] 激活 scratch buffer
- [ ] KV Cache 区域
- [ ] GEMV 输出区
- [ ] 层间状态机/微码调度器
- [ ] 从第 0 层运行到最后一层
- [ ] 最终 RMSNorm
- [ ] LM Head
- [ ] 完整单 token 前向输出 logits 与软件参考比较

## 阶段 I：文本生成闭环

### I1 第一版主机辅助文本生成

- [ ] 电脑执行 Tokenizer
- [ ] 电脑发送 prompt token IDs
- [ ] FPGA 执行 embedding、全部层和 LM Head
- [ ] FPGA 或电脑执行 argmax/top-k 采样
- [ ] 返回下一个 token
- [ ] 更新 KV Cache
- [ ] 连续生成至少 16 个 token
- [ ] 输出可读文本

### I2 可用的推理接口

- [ ] 支持 BOS/EOS
- [ ] 支持温度、top-k、top-p
- [ ] 支持最大生成长度
- [ ] 支持复位会话和清空 KV Cache
- [ ] 串口/USB/以太网中选择更高效接口
- [ ] 上位机提供命令行聊天工具

## 阶段 J：性能优化

功能正确后再做，禁止在完整闭环前过早优化。

- [ ] 复制 2/4/8 套 MAC16
- [ ] 评估 APM、LUT、FF、BRAM 和时序资源
- [ ] 权重和激活双缓冲
- [ ] DDR3 读取与 MAC 计算重叠
- [ ] 更长 burst 和连续行预取
- [ ] 多输出并行
- [ ] 减少 UART，改为更高速数据接口
- [ ] 优化 Softmax、RMSNorm 和非线性近似
- [ ] 测量首 token 延迟
- [ ] 测量 tokens/s
- [ ] 测量 DDR3 实际带宽
- [ ] 测量功耗和温度
- [ ] 在时序满足的前提下确定最佳核心频率

## 阶段 K：可靠性和发布

- [ ] 每个算子保留独立测试模式
- [ ] 建立自动回归测试套件
- [ ] 固定随机种子和金标准数据
- [ ] 记录每个验证位流 SHA256
- [ ] 错误状态码和超时恢复
- [ ] DDR3 越界保护
- [ ] 模型文件 CRC/哈希校验
- [ ] 断电重启流程
- [ ] 可选：写入 Flash 的安全发布流程
- [ ] 完整使用说明、架构图和性能报告

---

# 7. 当前工程中的关键文件

| 路径 | 作用 |
|---|---|
| `AGENTS.md` | 后续对话和开发者的强制入口说明 |
| `PROJECT_ROADMAP.md` | 本文件，唯一权威任务清单 |
| `PROJECT_PROGRESS_2026-07-23.md` | 当前已验证历史记录 |
| `source/int8_dot16.v` | 已验证 MAC16 |
| `ddr_mac16_integration/rtl/ddr_mac16_ctrl.v` | 当前 UART、AXI 和计算调度状态机 |
| `ddr_mac16_integration/rtl/ddr_mac16_top.v` | DDR3、UART 和计算顶层 |
| `ddr_mac16_integration/rtl/int4_unpack16.v` | packed INT4 解包 |
| `ddr_mac16_integration/pnr/build_ddr_mac16.tcl` | PDS 构建脚本 |
| `ddr_mac16_integration/pnr/program_sram.tcl` | 仅下载 SRAM |
| `gemv_int4_m4k64/rtl/gemv_m4k64_core.v` | 已验证固定 M=4、K=64 GEMV 核心 |
| `gemv_int4_m4k64/rtl/gemv_m4k64_ctrl.v` | 已验证 GEMV UART、DDR3 与计算调度状态机 |
| `gemv_int4_m4k64/pnr/build_gemv_m4k64.tcl` | 固定 GEMV PDS 构建脚本 |
| `gemv_int4_param/rtl/gemv_param_core.v` | 已验证运行时 K、片上缓存、MAC16 分块和尾块屏蔽核心 |
| `gemv_int4_param/rtl/gemv_param_ctrl.v` | 已验证运行时 M/K、UART、DDR3 行与输出地址调度 |
| `gemv_int4_param/pnr/build_gemv_param.tcl` | 参数化 GEMV D1.2 PDS 构建脚本 |
| `gemv_int4_perf/pnr/build_gemv_perf.tcl` | D1.3 性能计数独立 PDS 构建脚本，不覆盖 D1.2 位流 |
| `gemv_int4_perf/README.md` | 性能计数口径、协议、实测结果和瓶颈结论 |
| `tools/pangu_ddr_mac16_host.py` | INT8/INT4 上位机验证工具 |
| `tools/pangu_gemv_m4k64_host.py` | M=4、K=64 GEMV 金标准与上板测试工具 |
| `tools/pangu_gemv_param_host.py` | 参数化 GEMV 金标准、多尺寸、尾块、边界、压力测试与性能分析工具 |
| `tools/pangu_gemv_group_q28_host.py` | 真实 q_proj M4K896 分组 UQ4.28 固定向量、载荷自检和上板压力工具 |
| `gemv_int4_group_q28/README.md` | 分组 Q28 工程协议、地址布局、时序、位流和上板证据 |
| `tools/pangu_gemv_qproj_full_host.py` | 完整 q_proj 真实载荷、逐行 Q28 金标准、固定与随机上板验证工具 |
| `gemv_int4_qproj_full/README.md` | 完整 q_proj 工程协议、DDR3 布局、时序、位流和上板证据 |
| `model_tools/export_qwen25_fpga.py` | 模型转换工具 |
| `model_tools/p50_format.py` | `.p50` 固定头、目录、布局校验和按名提取解析库 |
| `model_tools/p50_inspect.py` | `.p50` 摘要、目录查看、全量校验、行/块提取命令行工具 |
| `model_tools/verify_p50_image.py` | 模型文件与源 BF16/LoRA 抽样量化验证工具 |
| `model_tools/linear_quant_reference.py` | 真实 Linear 的激活 INT8、UQ4.28 分组 scale 与 Q28 定点金标准 |
| `model_tools/q_proj_m4k896_reference.json` | layer0 q_proj 固定切片输出、误差上界和关键数组 SHA256 |
| `model_tools/q_proj_full_reference.json` | layer0 q_proj 完整层固定输出、上传布局和关键数组 SHA256 |
| `model_tools/test_linear_quant_reference.py` | 格式单测、1000 轮软件压力和真实 q_proj 集成回归 |
| `model_tools/rmsnorm_fixed_reference.py` | layer0 RMSNorm Q6.10/Q12.20、LUT/NR rsqrt 和硬件等价金标准 |
| `model_tools/rmsnorm_layer0_reference.json` | RMSNorm 固定向量、关键标量和数组 SHA256 清单 |
| `model_tools/test_rmsnorm_fixed_reference.py` | RMSNorm RNE、边界、真实 gamma 和 1000 轮软件压力测试 |
| `rmsnorm_k896/rtl/rmsnorm_k896_core.v` | 已验证 K=896 平方和、均值、LUT rsqrt、gamma 乘法和饱和核心 |
| `rmsnorm_k896/rtl/rmsnorm_k896_ctrl.v` | 已验证 RMSNorm UART、DDR3 载荷、结果回写和回读调度 |
| `rmsnorm_k896/pnr/build_rmsnorm_k896.tcl` | E1 RMSNorm 独立 PDS 构建脚本 |
| `rmsnorm_k896/README.md` | E1 定点格式、协议、地址、时序、位流和真实上板证据 |
| `tools/pangu_rmsnorm_k896_host.py` | RMSNorm 固定载荷、软件自检、固定与随机上板比较工具 |
| `model_tools/elementwise_fixed_reference.py` | E2 signed Q6.10 残差、缩放、元素乘法和 SiLU LUT/PWL 金标准 |
| `model_tools/elementwise_k896_reference.json` | E2 固定边界向量、SiLU 完整输入域误差和关键数组 SHA256 |
| `model_tools/test_elementwise_fixed_reference.py` | E2 RNE、饱和、最高 PWL 段覆盖和 1000 轮软件压力测试 |
| `elementwise_k896/rtl/elementwise_k896_core.v` | 已验证 K=896 四模式元素级计算、PWL64 SiLU 和结果打包核心 |
| `elementwise_k896/rtl/elementwise_k896_ctrl.v` | 已验证元素级 UART、DDR3 双向量/PWL 载荷、结果回写与回读调度 |
| `elementwise_k896/pnr/build_elementwise_k896.tcl` | 固定 seed17/29 和保持修复参数的 E2 PDS 构建脚本 |
| `elementwise_k896/README.md` | E2 定点规则、SiLU 选择、协议、地址、时序、位流和上板证据 |
| `tools/pangu_elementwise_k896_host.py` | E2 固定载荷、软件自检、四操作固定与随机上板比较工具 |
| `model_tools/embedding_fixed_reference.py` | E3 Token 行地址、真实 INT4/FP16 scale 到 UQ4.28/Q6.10 的硬件等价参考 |
| `model_tools/embedding_k896_reference.json` | E3 四个固定 Token 的载荷、输出和地址 SHA256 清单 |
| `model_tools/test_embedding_fixed_reference.py` | E3 地址边界、RNE、饱和、真实 scale 和 1000 个随机 Token 测试 |
| `embedding_k896/rtl/embedding_k896_core.v` | 已验证 14 组 INT4×UQ4.28、RNE、饱和与 896 元素结果打包核心 |
| `embedding_k896/rtl/embedding_k896_ctrl.v` | 已验证 Token 地址映射、UART、DDR3 行读取、结果回写与回读调度 |
| `embedding_k896/pnr/build_embedding_k896.tcl` | E3 独立 PDS 全流程构建脚本 |
| `embedding_k896/README.md` | E3 格式、协议、地址、时序、位流和真实上板证据 |
| `tools/pangu_embedding_k896_host.py` | E3 软件自检、固定边界 Token 和随机 Token 上板比较工具 |
| `model_tools/README.md` | `.p50` 格式、真实张量布局、量化定点定义、工具用法和验证证据 |

# 8. 后续每次工作的收尾要求

完成一次开发后，必须在本文件中更新：

1. 本轮完成了哪些复选框；
2. 固定测试结果；
3. 随机测试轮数；
4. PDS 时序 WNS/TNS；
5. 位流路径和 SHA256；
6. 真实上板结果；
7. “当前唯一下一任务”。

## 当前唯一下一任务（简明版）

```text
进入 F1 Q/K/V 线性层：围绕 layer0 真实张量建立统一软件参考和独立硬件闭环。
已确认 q_proj=[896,896]、k_proj=[128,896]、v_proj=[128,896]，均为 group size 64 的
INT4 分组对称量化；模型为 14 个 Q heads、2 个 KV heads、head_dim=64 的 GQA 布局。
先复用已验证 q_proj 完整层数据通路，补齐 K/V 的真实权重、scale、bias、Q28 金标准和
输出 head 布局；再建立可按投影类型运行的独立 QKV 工程，验证 Q/K/V 全输出、GQA 张量
排列、固定与随机真实 hidden state、PDS、多角时序和真实上板。不得覆盖任何已有验证工程和位流。
```
