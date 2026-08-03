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

- [x] 用通用 GEMV 实现 Q 投影
- [x] 用通用 GEMV 实现 K 投影
- [x] 用通用 GEMV 实现 V 投影
- [x] 支持多头/分组查询注意力的张量布局
- [x] 与软件参考逐元素比较

F1 验证证据（2026-07-24）：

- 新增独立 `qkv_linear_layer0` 工程，真实运行 layer0 `q_proj=[896,896]`、`k_proj=[128,896]`、`v_proj=[128,896]`，均为 INT4 group size 64；
- Q/K/V 共用同一逐向量对称 INT8 hidden state、UQ4.28 combined scale、signed int64 Q28 输出定义；
- GQA 输出按 head-major 连续排列，Q=`[14,64]`，K/V=`[2,64]`，`head_dim=64`；
- 固定 Q/K/V 全输出真实上板逐位一致，输出 SHA256 分别为 `ea1f04bf4ff313dad07025ff35e66a088f13afd28d817422b89bb135f63525a0`、`20728d329c32c722b0194032897bc3cf9a3a31323317e389d8fd7b6f78745474`、`162622e05e0013ca342f28032cb280c264f428f93a197eb67dbfafd76e20a168`；
- 完整 `model_tools` 回归 48/48 PASS，QKV 软件随机 hidden state 1000/1000 PASS，seed=`20260729`；
- 真实 FPGA 随机完整 Q+K+V 回归 3/3 PASS，seed=`20260729..20260731`；
- seed5/11 PDS 编译、综合、Device Map、布局布线、时序分析和位流生成成功，最终未布线网络为 0；
- 资源：8503 LUT、7641 FF、326 个 distributed RAM、4 DRM、12 APM；
- 多角时序 `All Constraints Met`：慢角 setup WNS=`+0.363 ns`、TNS=0，hold WHS=`+0.169 ns`、THS=0；快角 setup WNS=`+2.985 ns`、TNS=0，hold WHS=`+0.100 ns`、THS=0；恢复、移除和最小脉宽无违例；
- 位流：`qkv_linear_layer0/pnr_seed5/generate_bitstream/qkv_linear_top.sbit`，SHA256=`e3a4b6849a5716f38d6bdd3fbd039d46f2d350a32a0417ee347462d1a8f96e26`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；固件 `PANGU50K QKV LINEAR V1`，DDR3 初始化成功。

### F2 RoPE

- [x] 生成或加载 sin/cos 表
- [x] 偶数/奇数维旋转
- [x] 支持位置索引递增
- [x] 定点误差验证

F2 验证证据（2026-07-24）：

- 已确认 Qwen2.5-0.5B 配置：`head_dim=rotary_dim=64`、`rope_theta=1000000`、`max_position_embeddings=32768`；
- 已确认 Qwen2 实际 `rotate_half` 为 split-half：`dim i` 与 `dim i+32` 配对，不是相邻 `(0,1)、(2,3)` 配对；
- 新增独立 `rope_qk_layer0` 工程，直接消费 F1 已验证的 Q=`[14,64]`、K=`[2,64]` head-major signed int64 Q28 输出；
- sin/cos 使用 signed Q1.30；四个 64×32 乘积精确计算，两项乘积先在 signed 97 bit 中加/减，再执行一次 RNE 右移 30 位并显式饱和到 signed int64 Q28；
- 固定位置 `[0,1,2026,32767]` 的 Q/K 全输出真实上板逐位一致；位置 2026 的 Q/K SHA256 分别为 `6c266ff09ef200af907da2796b8fb1db4e5c050f0cad15ccb62e318a5953b0d6`、`0f8625c3063eb62726c7b3bfc933af4d70652014cd4b63a0ba772916a4c02622`；
- 连续位置 `2026..2033` 自动递增 8/8 PASS，位置表结束状态正确，`Z` 复位后首位置重放逐位一致；
- F2 新增单元测试 7/7 PASS，完整 `model_tools` 回归 55/55 PASS；软件随机 Q/K 与位置压力 1000/1000 PASS，seed=`20260730`；
- 真实 FPGA 随机位置回归 300/300 PASS，seed=`20260731`，约 235.59 秒；
- PDS 编译、综合、Device Map、布局布线、时序分析和位流生成成功，最终未布线网络为 0；
- 资源：8859 LUT、9886 FF、70 个 distributed RAM、1 APM、0 DRM；
- 多角时序 `All Constraints Met`：慢角 setup WNS=`+0.988 ns`、TNS=0，hold WHS=`+0.171 ns`、THS=0；快角 setup WNS=`+3.483 ns`、TNS=0，hold WHS=`+0.100 ns`、THS=0；恢复、移除和最小脉宽无违例；
- 位流：`rope_qk_layer0/pnr/generate_bitstream/rope_qk_top.sbit`，SHA256=`25396ffc894abc15b81ab99f62619f3694e7e662f620f3c6a89e28ae116d153a`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；固件 `PANGU50K ROPE QK V1`，DDR3 初始化成功。

### F3 KV Cache

- [x] 定义 DDR3 中每层 K/V Cache 地址布局
- [x] 当前 token 的 K/V 写入
- [x] 历史 token 的 K/V 顺序读取
- [x] 支持上下文长度边界检查
- [x] 防止层间和 token 间地址覆盖

F3 验证证据（2026-07-24）：

- 新增独立 `kv_cache_f3` 工程，未覆盖 F2 或任何更早阶段的工程和位流；
- 容量结论：K/V 各为 `[2,64]` signed int64 Q28，每 token 共 2048 B；完整 32768 positions 需要 1792 MiB，超过 1 GiB，因此硬件上下文上限确定为 16384；
- DDR3 低端 128 MiB 保留，高端 896 MiB 为 KV Cache；每层固定 32 MiB，28 层恰好结束于 1 GiB；低端 128 MiB 无法常驻约 251.63 MiB 全模型，后续完整集成需流式/分层加载权重或重新平衡上下文与内存分区；
- Controller 地址公式：`K=0x02000000 + layer*0x00800000 + position*0x200`，`V=K+0x100`；
- 支持当前 token 写入后 position 自动递增，以及一次连续 `1..16` token 的历史 K/V 分段 burst 顺序读取；
- F3 新增单元测试 9/9 PASS，完整 `model_tools` 回归 64/64 PASS；软件地址、边界和载荷随机压力 1000/1000 PASS，seed=`20260801`；
- 真实固定 K/V：layer0 position `0..1`、layer13 position `2026`、layer27 position `16383` 全部逐位一致；固定测试约 1.66 秒；
- layer27 最后槽严格结束于 1 GiB，下一 token 写入被错误码 `0x05` 正确拒绝；
- layer3/layer17 在相同 position `4096` 写入不同完整 K/V 后跨配置回读均逐位一致，证明层间不覆盖；
- 真实 FPGA 随机层、随机 position、每批 `1..16` 连续 token 回归 300/300 token PASS，seed=`20260801`，约 124.41 秒；
- PDS 编译、综合、Device Map、布局布线、时序和位流生成成功，最终未布线网络为 0；
- 资源：7572 LUT、9884 FF、70 个 distributed RAM、0 DRM、0 APM；
- 多角时序 `All Constraints Met`：慢角 core setup WNS=`+1.781 ns`、TNS=0，hold WHS=`+0.171 ns`、THS=0；快角 setup WNS=`+4.142 ns`、TNS=0，hold WHS=`+0.100 ns`、THS=0；恢复、移除和最小脉宽无违例；
- 位流：`kv_cache_f3/pnr/generate_bitstream/kv_cache_top.sbit`，大小 2101696 B，SHA256=`11a0240a2ee42f0c92b6a5919f4a4b71ceb7bb806b55f1810b4ef3ff88d23216`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；固件 `PANGU50K KV CACHE V1`，DDR3 初始化成功。

### F4 Attention Score

- [x] Q·K 点积
- [x] 缩放 `1/sqrt(head_dim)`
- [x] causal mask
- [x] 支持多头循环调度

F4 验证证据（2026-07-24）：

- 新增独立 `attention_score_f4` 工程，未覆盖 F3 或更早阶段的工程与位流；
- Q=`[14,64]`、K=`[2,64]` 均为 head-major signed int64 Q28；每 7 个连续 Q head 共用一个 KV head，映射为 Q head `0..6 -> KV0`、`7..13 -> KV1`；
- 64 维点积使用精确 signed Q56 累加；`head_dim=64` 对应 `1/sqrt(64)=1/8`，统一执行 signed RNE 右移 31 位并显式饱和到 signed int64 Q28；
- 固定 score 输出布局为 `[14,16]` head-major，共 1792 B；未来位置和未使用槽统一写入 `INT64_MIN=0x8000000000000000`；
- K 地址完全复用 F3：`0x02000000 + layer*0x00800000 + position*0x200`，F4 只读取 K，不改写 V；
- Q/K 片上缓存按 256 bit beat 同步读，成功推断 8 个 DRM18K；64×64 有符号乘法由 16 个 16×16 部分积顺序精确重构；
- 固定真实窗口覆盖 layer0/query0、layer0/query1、layer13/query2026 的部分未来 mask，以及 layer27/query16383 的最后 16 token 边界，四组完整 score 均真实上板逐位一致；
- 固定 score SHA256 分别为 `0697d1457bbd91a13a86e06b7de87a9928258c51c2b2b23d31a054bbc99325c5`、`30deb88a395f65ebaa92810278a8954f4cb0c8999462eb7071449dbf957a515d`、`c91ad94ac9af6da06aa2143cd81c87c8d3aeb68cd93ee5e50ecc54271bd51096`、`466cea477112b15a43ee5d03529bc34c10880f93887e44ae078bfac3ef527948`；
- F4 新增单元测试 9/9 PASS，完整 `model_tools` 回归 73/73 PASS；软件随机窗口、GQA、RNE、mask 和载荷压力 1000/1000 PASS，seed=`20260802`；
- 真实 FPGA 随机层、随机 query/start、`1..16` token 窗口、随机 Q/K 回归 100/100 PASS，seed=`20260802`，约 170.16 秒；
- PDS Compile、Synthesize、Device Map、Place & Route、Timing 和 Bitstream 全部成功，最终未布线网络为 0；
- 资源：9594 LUT、11621 FF、70 个 distributed RAM、8 DRM、1 APM；
- 多角时序 `All Constraints Met`：慢角 core setup WNS=`+0.482 ns`、TNS=0，hold WHS=`+0.170 ns`、THS=0；快角 setup WNS=`+3.003 ns`、TNS=0，hold WHS=`+0.100 ns`、THS=0；恢复、移除和最小脉宽无违例；
- 位流：`attention_score_f4/pnr/generate_bitstream/attention_score_top.sbit`，大小 2101696 B，SHA256=`669cb5b23cb6c5d33d0003f32452e57cda251751179c318c1b5d8f2ed8c0e0f8`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；固件 `PANGU50K ATTN SCORE V1`，DDR3 初始化成功；
- 本阶段未实现 Softmax，严格停在 F4 score 输出。

### F5 Softmax

- [x] max reduction
- [x] 减最大值
- [x] exp 近似或查表
- [x] sum reduction
- [x] reciprocal/归一化
- [x] 长序列数值稳定性测试

F5 验证证据（2026-07-24）：

- 新增独立 `softmax_f5` 工程，直接消费 F4 的 `[14,16]` head-major signed int64 Q28 score；`INT64_MIN` mask 槽严格输出概率 0，全 mask head 输出全 0；
- 概率格式确定为 unsigned UQ1.31 uint32，`1.0=0x80000000`；单有效 token 精确输出 1.0，相同 16 个有效 score 精确输出 16 个 `0x08000000`；
- 每 head 先执行 mask 感知 max reduction 和减最大值；exp 使用 `[-16,0]`、步长 `1/32` 的 513 点 UQ1.31 端点 LUT，区间内线性插值，差值小于 `-16` 时尾部置 0；
- 最多 16 项 exp 使用 36 位和；倒数为 `RNE(2^62/sum_exp_q31)`，概率为 `RNE(exp_q31*reciprocal_q31/2^31)` 并限制到 `[0,1.0]`；
- 插值乘法按 26 位相邻 LUT 差值拆成四个 13×12/11 部分积和两级加法流水；概率乘法前增加 exp 选择寄存，修复 100 MHz 慢角时序；
- F5 新增单元测试 10/10 PASS，完整 `model_tools` 回归 83/83 PASS；软件随机 mask、窗口、极端差值和载荷压力 1000/1000 PASS，seed=`20260803`，最坏 float64 概率误差 `3.04973546883e-05`；
- 四组真实 F4 固定 score 全部上板逐位一致，概率 SHA256 分别为 `768bd8912f9168473b8978963e805b52b5eeb40b26c517229dc6a4c8d96ce608`、`021ac6fad9854aeede02829734c6afdc3dc9cb41ce79ce078b830a53b695ce81`、`267a18e4d4fef9d1afb118d8f1a025cd9922f14963ae75fe672c30c816e5495f`、`b1ae419016695bb6c2a62ffb1b92c7bcbb70c87b22c1ba6d7cb96e327b201f39`；
- 真实 FPGA 全 mask、单有效、部分窗口、16 token、全等 score、`-16` 截断边界和随机稀疏 mask 回归 100/100 PASS，seed=`20260803`，约 29.05 秒，最坏 float64 概率误差 `2.96390625578e-05`；
- PDS Compile、Synthesize、Device Map、Place & Route、Timing 和 Bitstream 全部成功，最终未布线网络为 0；
- 资源：10515 LUT、12703 FF、70 个 distributed RAM、12 DRM18K、8 APM；
- 多角时序 `All Constraints Met`：慢角 core setup WNS=`+0.227 ns`、TNS=0，hold WHS=`+0.143 ns`、THS=0；快角 setup WNS=`+2.958 ns`、TNS=0，hold WHS=`+0.067 ns`、THS=0；恢复、移除和最小脉宽无违例；
- 位流：`softmax_f5/pnr/generate_bitstream/softmax_top.sbit`，大小 2101696 B，SHA256=`d6e505ea5495c6054a447608406db0f93855ef55dbfc357c8d113b00adba34fe`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；固件 `PANGU50K SOFTMAX F5 V1`，DDR3 初始化成功；
- 本阶段未实现 V 加权和，严格停在 F5 概率输出。

### F6 Attention 输出

- [x] softmax 权重与 V 的加权和
- [x] 多头拼接
- [x] 输出投影 `O_proj`
- [x] 残差连接
- [x] 完整 Attention 子层与软件参考比较

F6 第一段验证证据（2026-07-24）：

- 新增独立 `attention_output_f6` 工程，直接消费 F5 `[14,16]` head-major unsigned UQ1.31 概率，并按 F3 地址读取 `[count,2,64]` signed int64 Q28 V Cache；未覆盖 F5 或更早工程和位流；
- GQA 映射固定为 Q head `0..6 -> KV0`、`7..13 -> KV1`；每项概率×V 为 signed Q59，最多 16 项在 signed 100 bit 中精确累加，全部累加结束后仅执行一次 signed RNE 右移 31 位并显式饱和到 signed int64 Q28；
- 输出为 `[14,64]` head-major，可无损拼接为连续 `[896]`，共 7168 B；全 mask 输出严格全 0，单 token `1.0=0x80000000` 精确复制对应 V，固定未使用概率槽必须为 0；
- F6 新增单元测试 10/10 PASS，完整 `model_tools` 回归 93/93 PASS；软件随机 GQA、Q59、RNE、饱和和载荷压力 1000/1000 PASS，seed=`20260804`；
- 四组真实 F5 概率与逐 position 真实 layer0 `v_proj` 固定 V 全部上板逐位一致，输出 SHA256 分别为 `c5107911c0e6b9f1d9c471d7dde2c26d1192282abc964eb5005051aa5a4c9f71`、`14a6ee14736ed6132ea1357d3af27d55074492a727b196c33cd6905d4e1c9b02`、`86bc89cc77d49ac9451cc2a707510df310566997faf4c76fdfa95599043c248b`、`73e6b464b8c27e6a4d1066df5b5dade1c11c6a085a712d58dc54f6b598a0e407`；
- 真实 FPGA 全 mask、单 token 1.0、14Q/2KV 映射、`INT64_MIN/MAX` 极端 V，以及 16-token Q59 宽累加后的 INT64 正/负双向饱和边界全部通过；随机层、随机 start、count `1..16`、随机概率/V 回归 100/100 PASS，seed=`20260804`，约 182.09 秒；
- PDS Compile、Synthesize、Device Map、Place & Route、Timing 和 Bitstream 全部成功，详细路由 155 轮后未布线网络为 0；
- 资源：9184 LUT、11357 FF、70 个 distributed RAM、12 DRM、1 APM；概率缓存映射为 8 个 DRM9K，16-token V 缓存映射为 8 个 DRM18K；
- 多角时序 `All Constraints Met`：慢角 setup WNS=`+0.825 ns`、TNS=0，hold WHS=`+0.112 ns`、THS=0；快角 setup WNS=`+3.349 ns`、TNS=0，hold WHS=`+0.032 ns`、THS=0；恢复和移除无违例；
- 位流：`attention_output_f6/pnr/generate_bitstream/attention_output_top.sbit`，大小 2101696 B，SHA256=`d7e64c58b73f8ca93f7a7dd981feabe5cc48f9b43e6b2ff0d8f60155886f36a3`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；固件 `PANGU50K ATTN OUTPUT V1`，DDR3 初始化成功；
- 第一段严格停在 Attention 加权和与多头拼接，没有提前实现 `O_proj`、Attention 残差或 MLP。

F6 第二段 O_proj 验证证据（2026-07-24）：

- 新增独立 `attention_oproj_f6` 工程，输入来源为第一段已经真实上板逐位通过的 `[896]` head-major signed int64 Q28 Attention 拼接结果；真实参数为 `model.layers.0.self_attn.o_proj.weight=[896,896]`、group size 64，`.p50` 中不存在 O_proj bias，因此 `bias_q28` 固定全 0；
- Q28 输入先按实数解释并执行逐向量对称 INT8 量化，硬件继续采用每 64 元素 INT32 点积、unsigned UQ4.28 combined scale、signed int64 Q28 跨 14 group 累加；复用已验证完整 Linear controller/core，但建立独立顶层、PDS 工程、位流、软件清单和上位机，不覆盖 q_proj、F6 第一段或更早成果；
- 四组固定输入直接复用 F6 第一段的 1、2、6、16-token 真实窗口输出；O_proj 输出 SHA256 分别为 `19008a25a59cde0f8def0c938ada397b6866dc143774b74c6ff77a2a95a7fcd5`、`0e70753bea148c81d0bce79360d250710a1cc6ee817a40e4b6cbccf7d4f30279`、`c0ffeb8b5a1168b661d52a34f34a5f4f12f3d075805b05b4ace346683cb8b018`、`af63d1efc3913f597fdcd5dbe520ac782a943074301a60b249f4f25a3cf34a65`；四组 896 个输出均真实上板逐位一致；
- O_proj 新增单元测试 7/7 PASS，完整 `model_tools` 回归 100/100 PASS；真实参数、独立 Q28 重算、固定 488320 B 载荷往返和随机/边界软件压力 1000/1000 PASS，seed=`20260805`，约 35.26 秒；
- 真实 FPGA 全零、随机常量、稀疏极值和完整 896 维随机 Attention Q28 输入回归 4/4 PASS，seed=`20260805`，约 172.34 秒；四组固定上传、计算和回读均约 43.03~43.04 秒；
- PDS Compile、Synthesize、Device Map、Place & Route、Timing 和 Bitstream 全部成功，最终未布线网络为 0；
- 资源：8510 LUT、7619 FF、326 个 distributed RAM、4 DRM、12 APM；
- 多角时序 `All Constraints Met`：慢角 core setup WNS=`+0.614 ns`、TNS=0，hold WHS=`+0.171 ns`、THS=0；快角 setup WNS=`+3.023 ns`、TNS=0，hold WHS=`+0.101 ns`、THS=0；恢复、移除和最小脉宽无违例；
- 位流：`attention_oproj_f6/pnr/generate_bitstream/attention_oproj_top.sbit`，大小 2101696 B，SHA256=`017517f877f29e62d945ecd3ae4ba22c2d690b6e6b92778eb0502ba7ac115533`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；复用通用完整 Linear 协议标识 `PANGU50K QPROJ FULL V1`，DDR3 初始化成功；
- 第二段严格停在 O_proj，当时尚未实现 Attention 残差、完整 Attention 子层或 MLP。

F6 第三段残差与完整 Attention 子层验证证据（2026-07-24）：

- 新增独立 `attention_residual_f6` 工程，硬件入口严格接收 `[896]` signed int16 Q6.10 residual hidden state 和 `[896]` signed int64 Q28 O_proj 输出，不覆盖 O_proj、F6 加权和或更早工程和位流；
- 定点边界统一为：O_proj Q28 使用对称 signed RNE 右移 18 位，先显式饱和到 signed int16 Q6.10，再与原 hidden state 符号扩展相加，最终再次显式饱和到 signed int16；重标定饱和与残差饱和相互独立；
- 新增连贯 layer0 Attention 软件参考：每个 token 从对应 hidden state 出发，依次执行真实 input RMSNorm、Q/K/V、RoPE、历史 K/V、Score、Softmax、概率×V、多头拼接、真实 O_proj 和第一处残差，不再拼接此前彼此独立的固定测试向量；
- 1/2/6/16-token 四组连贯固定输出 SHA256 分别为 `36859690e421b96cb8db65a5760a364d165a73b63fd1121040a7d1b42c042eb7`、`2b4a2d9240e6e30c2afe2943fa30ac60decd47f8fc8d377ab7e530e516009378`、`c0c0776d71e3dc97aa1a4d4e0709f38441cc82717c0c3081a79c47c30a21af10`、`7e61dc1fd0eb43b231e25fe1d08b1c08342723f537b6e25924858608235fa61e`；四组 896 个输出均真实上板逐位一致；
- 新增单元测试 5/5 PASS，完整 `model_tools` 回归 105/105 PASS；完整软件链、8960 B 上传载荷、signed RNE tie、`INT64_MIN/MAX`、两级正负饱和和随机/边界软件压力 1000/1000 PASS，seed=`20260806`；
- 真实 FPGA 随机/边界压力累计 300/300 PASS，分三批 seed_start=`20260806`、`20260906`、`20261006`，覆盖全零、RNE 半值、INT64 极值、一般 Q28、重标定饱和和最终残差饱和；
- PDS Compile、Synthesize、Device Map、Place & Route、Timing 和 Bitstream 全部成功，最终未布线网络为 0；
- 资源：7695 LUT、6868 FF、70 个 distributed RAM、20 DRM、0 APM、79 IO；
- 多角时序 `All Constraints Met`：慢角 core setup WNS=`+1.493 ns`、TNS=0，hold WHS=`+0.112 ns`、THS=0；快角 setup WNS=`+3.841 ns`、TNS=0，hold WHS=`+0.051 ns`、THS=0；恢复和移除无违例；
- 位流：`attention_residual_f6/pnr/generate_bitstream/attention_residual_top.sbit`，大小 2101696 B，SHA256=`609e1f569aa1e4579cffb995b0d7d0bc89fa34529790b35e8b26d6778226bcbd`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；固件 `PANGU50K ATTN RESIDUAL V1`，DDR3 初始化成功；
- F6 Attention 已完整通过，当前才允许进入 MLP。

## 阶段 G：MLP 和 Transformer Block

### G1 MLP

- [x] MLP 输入 `post_attention_layernorm`
- [x] gate projection
- [x] up projection
- [x] SiLU(gate)
- [x] SiLU(gate) × up
- [x] down projection
- [x] 残差连接
- [x] 完整 MLP 与软件参考比较

G1 MLP 输入 `post_attention_layernorm` 验证证据（2026-07-24）：

- 新增独立 `post_attention_layernorm_g1` 工程，输入直接复用 F6 已真实上板逐位通过的完整 Attention 子层 `[896]` signed Q6.10 输出；真实 gamma 为 `model.layers.0.post_attention_layernorm.weight`，shape=`[896]`、连续 FP16，epsilon=`1e-6`；
- 定点规则严格复用但隔离 E1：input/gamma/output 为 signed Q6.10，平方和 40 位，均值与 epsilon 为 Q12.20，LUT256 rsqrt 为 UQ12.20，除法与右移采用 RNE，输出显式饱和；未修改或覆盖 E1、F6 已验证 RTL、PDS 工程和位流；
- 四组连贯固定 query/count=`0/1、1/2、5/6、15/16` 的输入 SHA256 与 F6 最终输出完全一致；G1 输出 SHA256 分别为 `93d2d3ee866a7923e3ce9d450ae5d6e43a05c50daeaa952cae052c4584891f80`、`0ef1296dde8e999f6ac707725da227bd8f87b5da848a7a81113f422a03d0cbdf`、`40965e0cb4d96cf8de644d4b7081df5acef34d6c24ec8cd6d448fac4943b83aa`、`fa574c09c76580173c62d59bd5a682cd35bb97b70d25459dcf0ac6e3808e48b1`；四组 896 个输出均真实上板逐位一致；
- 新增单元测试 5/5 PASS，完整 `model_tools` 回归 110/110 PASS；软件固定清单、4608 B 载荷往返和随机/边界压力 1000/1000 PASS，seed=`20260807`，覆盖全零、`INT16_MIN/MAX`、常量、稀疏、小幅值和完整 int16 随机输入；
- PDS Compile、Synthesize、Device Map、Place & Route、Timing 和 Bitstream 全部成功，最终未布线网络为 0；资源为 8801 LUT、7051 FF、70 个 distributed RAM、12 DRM、9 APM、79 IO；
- 多角时序 `All Constraints Met`：慢角 core setup WNS=`+0.411 ns`、TNS=0，hold WHS=`+0.169 ns`、THS=0；快角 setup WNS=`+2.857 ns`、TNS=0，hold WHS=`+0.100 ns`、THS=0；恢复和移除无违例；
- 位流：`post_attention_layernorm_g1/pnr/generate_bitstream/rmsnorm_k896_top.sbit`，大小 2101696 B，SHA256=`b8c87ee10edf435617ab110cfdf0cf2a8d3c3ad3d3b91748c80ef04363305ec2`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash，DDR3 初始化成功；真实 FPGA 随机/边界 300/300 PASS，seed=`20260807`，耗时 182.74 秒。

G1 `gate_proj` 与 `up_proj` 真实双投影验证证据（2026-07-24）：

- 新增独立 `mlp_gate_up_g1` 工程，两路直接消费上述四组 `[896]` signed Q6.10 post-attention RMSNorm 输出；真实 gate/up 权重 shape 均为 `[4864,896]`、group size 64、每行 14 groups 的对称 signed INT4，`.p50` 均不存在 bias；
- 两路严格共享同一份逐向量对称 INT8 激活；主机预计算 UQ4.28 combined scale，硬件按 64 元素 signed INT32 点积并在 signed int64 Q28 中跨 14 groups 累加；通用 bias 槽全零；
- 四组 query/count=`0/1、1/2、5/6、15/16` 的 gate/up 共 8 个完整 `[4864]` 输出全部真实上板 `4864/4864` 逐位一致；对应输出 SHA256 已固定在 `model_tools/mlp_gate_up_g1_reference.json`；
- 新增单元测试 6/6 PASS，完整 `model_tools` 回归 116/116 PASS；双投影软件随机/边界 1000/1000 PASS，seed=`20260808`；
- 真实 FPGA gate/up 各覆盖全零、交替 `INT16_MIN/MAX` 和一般随机输入 3/3 PASS，双路合计 6/6 PASS；每路 2646912 B 完整上传、计算和 38912 B 回读约 232.99~233.26 秒；
- PDS Compile、Synthesize、Device Map、Place & Route、Timing 和 Bitstream 全部成功，最终未布线网络为 0；资源为 8548 LUT、7628 FF、326 个 distributed RAM、4 DRM、12 APM、79 IO；
- 多角时序 `All Constraints Met`：慢角 core setup WNS=`+0.916 ns`、TNS=0，hold WHS=`+0.157 ns`、THS=0；快角 setup WNS=`+3.046 ns`、TNS=0，hold WHS=`+0.089 ns`、THS=0；恢复、移除和最小脉宽无违例；
- 位流：`mlp_gate_up_g1/pnr/generate_bitstream/mlp_gate_up_top.sbit`，大小 2101696 B，SHA256=`e72959d2968a543bf3a2bcfd31f2b2c7a0d31a9888daba9ceac2d7c50cd5db6b`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；固件 `PANGU50K MLP GATEUP V1`，DDR3 初始化成功；gate/up 已全部通过，现允许进入 `SiLU(gate)`，但不得提前执行乘法。

G1 `SiLU(gate)` 真实非线性验证证据（2026-07-25）：

- 新增独立 `mlp_silu_g1` 工程，输入直接复用上述四组已经真实上板逐位通过的 gate projection `[4864]` signed int64 Q28 输出；本阶段没有重算 gate/up，也没有执行 `SiLU(gate) × up`；
- 固定数值规则为 Q28 对称 signed RNE 右移 18 位并显式饱和到 signed int16 Q6.10，再复用 E2 已验证的 PWL64 SiLU；主区间 `[-8,8)`，尾部 `x<-8 -> 0`、`x>=8 -> x`，相对精确 SiLU 的完整 int16 输入域最大误差 4 Q10 LSB；
- 四组真实 gate 范围约为 `[-3.5440,2.8183]`，均未触发 PWL 尾部或 Q6.10 饱和；四组 query/count=`0/1、1/2、5/6、15/16` 的完整 `[4864]` 输出均真实上板 `4864/4864` 逐位一致；输出 SHA256 分别为 `d3a50e88eba59160b61eccaf9a25c0d3f5dd8c5f799dbd28ede20acbd383cd18`、`4dc5e4f4d3240ce628ee7db071ed31faa570212a1a5dc56e5b01c69d9702d310`、`b807ad37514a9bd1702625666f2c13670bfa460c423a2fc53fe483a44900e9c9`、`4f16572a82b583edb041444edf7bdea5841ffcc3a5a7de71a28cae138f2e980e`；
- 新增单元测试 7/7 PASS，完整 `model_tools` 回归 123/123 PASS；软件固定载荷与随机/边界压力 1000/1000 PASS，seed=`20260809`，覆盖全零、`INT64_MIN/MAX`、RNE half-way tie、`±8` 尾部、int16 饱和边界、稀疏、一般范围和完整随机 int64 bit pattern；
- 独立上传载荷 39072 B：38912 B gate Q28 与补齐到 80 项的 160 B PWL 端点；结果为 9728 B signed Q6.10；RTL 使用 4-bank DRM 缓存将 1216 个输入 beat 重排为 304 个输出 beat；
- PDS Compile、Synthesize、Device Map、Place & Route、Timing 和 Bitstream 全部成功，最终未布线网络为 0；资源为 8024 LUT、7901 FF、70 个 distributed RAM、32 DRM、1 APM、79 IO；
- 多角时序 `All Constraints Met`：慢角 core setup WNS=`+1.468 ns`、TNS=0，hold WHS=`+0.169 ns`、THS=0；快角 setup WNS=`+3.793 ns`、TNS=0，hold WHS=`+0.100 ns`、THS=0；恢复、移除和最小脉宽无违例；显式 seed17/29 的首版仍有 1 条 Fast hold `-0.015 ns`，未被接受，最终默认 PnR 版本全角为 0；
- 位流：`mlp_silu_g1/pnr/generate_bitstream/mlp_silu_top.sbit`，大小 2101696 B，SHA256=`87e643c65b70949297d54042921ac62e70454c018b6ff31f1386bbf2c8770550`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；固件 `PANGU50K MLP SILU V1`，DDR3 初始化成功；四组固定输入合计约 17.16 秒；
- 真实 FPGA 随机/边界分六批累计 300/300 PASS，seeds=`20260809..20260814`，所有 4864 项均与 Python 金标准逐位一致；`SiLU(gate)` 已完整通过，现允许进入 `SiLU(gate) × up`，但不得提前进入 down projection。

G1 `SiLU(gate) × up` 真实逐元素乘法验证证据（2026-07-25）：

- 新增独立 `mlp_silu_up_mul_g1` 工程，两路输入直接复用已经分别真实上板逐位通过的 `SiLU(gate)` `[4864]` signed int16 Q6.10 和 `up_proj` `[4864]` signed int64 Q28；本阶段不重算 gate/up，也没有执行 `down_proj`；
- 数值规则冻结为完整 signed 16×64 乘法，保留 signed 80 bit Q38 乘积；对绝对值执行对称 RNE 右移 10 位，恢复符号后显式饱和到 signed int64，输出保持 Q28，可直接作为后续 down projection 输入，禁止隐含截断；
- 四组 query/count=`0/1、1/2、5/6、15/16` 的完整 `[4864]` 输出均真实上板 `4864/4864` 逐位一致；输出 SHA256 分别为 `278ceccc804b8f74266b6000745c1ae21d09cf47ba19041ff13cb5cbdaeac0ca`、`96e1191832febbb2bf246918e489725094567811869ab85bf8452ee8e6520fa9`、`9f01a9589fc9ee4f8b33acd9a64b8a767b37bee8f788697f0043ac395c7a28dc`、`297b982da2fb3ee7bd9202cd8d655dec200a9e19fee9a8c614e2e5412ae97802`；
- 新增单元测试 7/7 PASS，完整 `model_tools` 回归 130/130 PASS；固定清单、48640 B 上传载荷往返和软件随机/边界压力 1000/1000 PASS，seed=`20260815`，覆盖正负 RNE half-way tie、完整 80 位乘积、零乘数、完整 int16/int64 bit pattern、`INT64_MIN/MAX` 与双向饱和；
- RTL 使用 304×256 SiLU 缓存与四个 304×256 up bank，单个 16×16 乘法器分四个 limb 重构 80 位乘积，结果以 1216 个 256-bit beat 流式写回；PDS 推断 40 DRM 和 1 APM；
- 默认 PnR 虽完整生成位流，但 Fast Corner core hold 有一条 `-0.005 ns`，未被接受；独立 seed17/29 PDS Compile、Synthesize、Device Map、Place & Route、Timing 和 Bitstream 全部成功，最终未布线网络为 0；资源为 7895 LUT、6910 FF、70 distributed RAM、40 DRM、1 APM、79 IO；
- seed17/29 多角时序 `All Constraints Met`：慢角 core setup WNS=`+0.511 ns`、TNS=0，hold WHS=`+0.141 ns`、THS=0；快角 setup WNS=`+3.050 ns`、TNS=0，hold WHS=`+0.065 ns`、THS=0；恢复、移除和最小脉宽无违例；
- 验收位流：`mlp_silu_up_mul_g1/pnr_seed17/generate_bitstream/mlp_silu_up_mul_top.sbit`，大小 2101696 B，SHA256=`a83797a8b2ec75d030fc01144e6bf51e7de0ec930fc135c1a0aba89ebf1c4336`；
- JTAG SRAM 下载成功，`done bit=1`，未操作 Flash；固件 `PANGU50K MLP SILUUP V1`，DDR3 初始化成功；四组固定输入耗时 30.55 秒；同一固定 seed 连续真实 FPGA 随机/边界 100/100 PASS，所有 4864 项均与 Python 任意精度金标准逐位一致；
- `SiLU(gate) × up` 已完整通过，当前唯一允许进入的下一任务为独立 `down_proj`，不得提前合并 MLP 残差或完整 Transformer Block。

G1 `down_proj` 真实完整投影验证证据（2026-07-25）：

- 新增独立 `mlp_down_proj_g1` 工程，输入直接复用已经真实上板逐位通过的 `SiLU(gate) × up` `[4864]` signed int64 Q28；真实 `model.layers.0.mlp.down_proj.weight` shape=`[896,4864]`、group size 64、每行 76 groups 的对称 signed INT4，`.p50` 不存在 bias；本阶段没有执行第二处残差；
- 数值规则冻结为 Q28→float32、逐向量对称 INT8 `[-127,127]` RNE、zero point=0，combined scale 使用 unsigned UQ4.28 RNE 与显式饱和；每 64 项执行 signed INT32 点积，76 组乘 scale 后在 signed int64 Q28 中精确累加；理论最坏绝对累加 `18571850900440320 < 2^63-1`，禁止隐含截断或回绕；
- 四组 query/count=`0/1、1/2、5/6、15/16` 的完整 `[896]` 输出均真实上板 `896/896` 逐位一致；输出 SHA256 分别为 `20ada87fb91b6f3a286d554eed7ede0d369e417162683bb4828f4ba2d0a45da3`、`05daecd0467d77bd1cf4f48be22caaece068cd844d263b46f88e016775deacec`、`2e8933ddb0423cf7f7c43d7165f82ce62c128607b883f80ca942d919740a0ccf`、`2dcea63a160554e624edd6f1c42e28a15f17a59e4999badfabdc8a7db80a82ee`；
- 新增单元测试 7/7 PASS，完整 `model_tools` 回归 137/137 PASS；固定清单、2499328 B 上传载荷往返和软件随机/边界压力 1000/1000 PASS，seed=`20260816`，覆盖全零、上游乘法极值/饱和、RNE tie、稀疏、一般范围和完整 int64 bit pattern；
- RTL 缓存 152 拍激活和当前行 76 拍权重，激活与权重均按 AXI 最大 16 拍自动分段；每行 76 个 scale 补齐为 10 拍，每 4 行结果组成 1 拍立即写回；
- PDS Compile、Synthesize、Device Map、Place & Route、Timing 和 Bitstream 全部成功，详细布线 153 轮后未布线网络为 0，hold 修复 6 轮完成；资源为 8915 LUT、9426 FF、70 distributed RAM、8 DRM、12 APM、79 IO；
- 多角时序 `All Constraints Met`：慢角 core setup WNS=`+0.872 ns`、TNS=0，hold WHS=`+0.110 ns`、THS=0；快角 setup WNS=`+3.026 ns`、TNS=0，hold WHS=`+0.015 ns`、THS=0；恢复、移除和最小脉宽无违例；
- 验收位流：`mlp_down_proj_g1/pnr/generate_bitstream/mlp_down_proj_top.sbit`，大小 2101696 B，SHA256=`f4d1013a287fc27003db88905f3c61e25620d213475039ddbb14900580c46757`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；固件 `PANGU50K MLP DOWN V1`，DDR3 初始化成功；四组固定每组上传约 216.78~216.82 秒，计算与回读约 0.65 秒；真实 FPGA 随机/边界 3/3 PASS，seed=`20260816`，覆盖全零、极值/饱和和正负 RNE half-way tie，所有 896 项逐位一致；
- `down_proj` 已独立完整通过，当前唯一允许进入的下一任务为第二处残差；残差单独通过前不得宣称完整 MLP 或完整 Transformer Block 完成。

G1 第二处残差与完整 MLP 真实闭环验证证据（2026-07-25）：

- 新增独立 `mlp_residual_g1` 工程；down 分支直接消费已经真实上板逐位通过的 `down_proj` `[896]` signed int64 Q28，residual 分支严格使用进入 `post_attention_layernorm` 之前、即完整 Attention 第一处残差后的 `[896]` signed int16 Q6.10 hidden state，禁止使用归一化输出；
- 固定数值规则为 down Q28 对称 signed RNE 右移 18 位、显式饱和到 signed int16 Q6.10、与 residual hidden 符号扩展相加、最终再次显式饱和到 signed int16；`INT64_MIN` 采用无符号二补码幅值路径，无隐含截断或回绕；
- 四组 query/count=`0/1、1/2、5/6、15/16` 的 residual SHA256 与 F6 第一处残差输出完全一致，down SHA256 与 G1 `down_proj` 输出完全一致；最终 `[896]` 输出 SHA256 分别为 `630952eaf6fe179639773f2b60d1e9e990f380b0e698fa3d493dd7c279c96104`、`1cd96d92e43203abda26cace446c2664a0cbaa1fad29a8658a0d94941fbfbea7`、`b2365afdc2857c543628c9d15fd829005ede98aa99d35a8c4f973f61b35ed9dc`、`c164aab5251afc3954b8689a826a1241d1c7d6757adef5c5a232e127b59a4032`；四组均真实上板 `896/896` 逐位一致；
- 新增单元测试 5/5 PASS，完整 `model_tools` 回归 142/142 PASS；固定清单、8960 B 上传载荷往返和软件随机/边界压力 1000/1000 PASS，seed=`20260817`，覆盖全零、正负 RNE half-way tie、`INT64_MIN/MAX`、Q10 饱和边缘、一般范围、第一次饱和和最终残差饱和；
- RTL 使用 1 个 hidden 缓存和 4 个 down bank，每个 256-bit 输出 beat 处理 16 个元素；DDR3 32-bit 地址基址为 hidden=`0x0000000`、down=`0x0001000`、result=`0x0003000`；固件命令 `I/S/L/G`，标识 `PANGU50K MLP RESIDUAL V1`；
- PDS Compile、Synthesize、Device Map、Place & Route、Timing 和 Bitstream 全部成功，详细布线 89 轮后未布线网络为 0，hold 修复 3 轮；资源为 7705 LUT、6868 FF、70 distributed RAM、20 DRM、0 APM、79 IO；
- 多角时序 `All Constraints Met`：慢角 core setup WNS=`+0.727 ns`、TNS=0，hold WHS=`+0.169 ns`、THS=0；快角 setup WNS=`+3.298 ns`、TNS=0，hold WHS=`+0.100 ns`、THS=0；恢复、移除和最小脉宽无违例；
- 验收位流：`mlp_residual_g1/pnr/generate_bitstream/mlp_residual_top.sbit`，大小 2101696 B，SHA256=`ddc424fae630fda5ab55acc8d2cb12d80b3f8cca1d5341f4a455ec0aa0a0e42b`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；DDR3 初始化成功；同一 seed=`20260817` 的连续真实 FPGA 随机/边界 index=`0..299` 分三批累计 300/300 PASS，每组 896 项均与 Python 金标准逐位一致；
- G1 第二处残差和完整 layer0 MLP 至此真实闭环完成，现允许进入 G2 单个完整 Transformer Block 集成，但不得跳过该阶段直接进入 28 层调度或文本生成。

### G2 单个 Transformer Block

- [x] 输入 RMSNorm
- [x] Q/K/V
- [x] RoPE
- [x] Attention
- [x] O projection
- [x] 第一处残差
- [x] 第二个 RMSNorm
- [x] MLP
- [x] 第二处残差
- [x] 一个完整 Block 与软件参考比较
- [x] 多组随机输入和真实 hidden state 验证

G2.1 软件全链参考、运行时量化闭环与集成契约（2026-07-25 至 2026-08-03，已完成）：

- [x] 从同一组 block hidden state 建立完整 layer0 软件金标准，连贯执行 input RMSNorm、Q/K/V、RoPE、KV history、Attention Score、Softmax、概率×V、O_proj、第一处残差、post-attention RMSNorm、gate/up、SiLU、逐元素乘法、down_proj 和第二处残差；
- [x] 冻结四组真实 query/count=`0/1、1/2、5/6、15/16` 的关键中间结果与最终输出 SHA256；最终 Block 输出分别为 `630952eaf6fe179639773f2b60d1e9e990f380b0e698fa3d493dd7c279c96104`、`1cd96d92e43203abda26cace446c2664a0cbaa1fad29a8658a0d94941fbfbea7`、`b2365afdc2857c543628c9d15fd829005ede98aa99d35a8c4f973f61b35ed9dc`、`c164aab5251afc3954b8689a826a1241d1c7d6757adef5c5a232e127b59a4032`，与已验证 G1 最终结果完全一致；
- [x] 建立 64 字节执行头、当前 hidden、RoPE trig 和历史 K/V 的动态载荷，四组载荷长度分别为 2112、4160、12352、32832 B，往返校验通过；
- [x] 冻结独立 G2 的 28 个 scratch/查表区域、24 个 Linear 参数/scale 区、七个矩阵调用描述与 25 个状态 ID；低端参数区最高结束字节地址为 `0x018a5400`，未越过 128 MiB；F3 KV Cache 继续从字节地址 `0x08000000` 开始，layer27/position16383 的 V 末地址恰好为 `0x40000000`；
- [x] 明确完整 Block 不能继续依赖主机预生成中间 INT8 激活和 combined scale；新增 `runtime_linear_quant_reference.py`，把 Q6.10/Q28→binary32→symmetric INT8、FP16 weight scale→UQ4.28 改写为精确整数/二进制有理数定义；7 项测试覆盖 ties-to-even、零向量、Q10/Q28、Q28 双重舍入随机/边界和七个真实矩阵全部 scale，逐位复现既有 NumPy/G1/F6 定义；
- [x] 顶层调度显式加入 `QKV_QUANT/OPROJ_QUANT/GATE_UP_QUANT/DOWN_QUANT` 四个阶段，22 个计算阶段均使用明确 start/done/error/watchdog，不允许隐含主机介入；
- [x] 新增可覆盖 Q/K/V/O_proj/gate/up/down 的 `shared_linear_engine.v`、`runtime_linear_ctrl.v` 和 `transformer_block_scheduler.v`；PDS Compile/Synthesize 均成功，无硬错误；边界审查发现 down_proj 76 groups 实际按 10 拍补齐为 80 scale words，已把共享 scale RAM 从 76 修正为 padded 80 并重新综合；最终资源为共享 engine `1557 LUT/3152 FF/8 DRM/12 APM`、runtime controller（含 engine）`2377 LUT/4038 FF/8 DRM/12 APM`、22 阶段 scheduler `159 LUT/80 FF`；这些仅是独立子模块综合，不代表完整 PnR/时序通过；
- [x] 新增完整软件/契约测试；完整软件链不同 seed/query/window 确定性压力累计 `12/12 PASS`，seed=`20260822`；Block 固定清单 `4/4 PASS`；运行时量化七矩阵固定清单 `7/7 PASS`、地址/burst/padding 软件压力 `1000/1000 PASS`，seed=`20260819`；最新完整 `model_tools` 回归 `187/187 PASS`；
- [x] 已实现运行时量化 RTL 与 DDR3 controller：`unsigned_divider_rne.v`、`runtime_q10_activation_quantizer.v`、`q28_to_binary32.v`、`runtime_q28_activation_quantizer.v`、`runtime_fp16_scale_builder.v`、`runtime_activation_quantizer_ctrl.v`、`runtime_scale_builder_ctrl.v`、`runtime_quantizer_ctrl.v` 和 Q28 固化顶层；RNE 除法器软件镜像随机 `60000/60000 PASS`，Q28 int64→binary64→binary32 双重舍入对 10000 组随机值和 11 个关键边界逐位匹配 NumPy；
- [x] Q6.10/Q28 量化 DDR3 controller 已用最新源码重新通过 PDS Compile/Synthesize；为满足 100 MHz，Q28 双重舍入、96 位 RNE 除法、Q28 max-abs 和 FP16 scale 构建均改为保持逐位语义的多周期结构；
- [x] 已建立 `runtime_quantizer_validation_top` 自动逐位闭环和 AXI trace checker：同一位流覆盖 Q6.10/Q28，自动核对 source/raw-scale 读取、activation/combined-scale 写入的地址、命令数、beat 数和 burst，并回读 96 B metadata、全部 INT8 和 padded UQ4.28；
- [x] 七个真实矩阵 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` 已真实上板 `7/7 PASS`，逐位覆盖 max metadata、全部 activation、全部 scale、14→16/76→80 padding 和 AXI trace；Q6.10 `k_proj` 随机/边界 `100/100 PASS`，Q28 `o_proj` 随机/边界 `24/24 PASS`，seed=`20260819`，覆盖全零、signed 极值、稀疏、完整随机、幂次边界和精确 half-way tie；
- [x] 量化验证工程已完成 Compile、Synthesize、Device Map、Place & Route、Timing 和 Bitstream，详细布线 79 轮后未布线网络为 0、hold 修复 4 轮；资源 `16370 LUT/13887 FF/40 DRM/8 APM/79 IO`；多角时序 `All Constraints Met`，慢角用户时钟 setup WNS=`+0.187 ns`、hold WHS=`+0.171 ns`，快角 setup WNS=`+2.908 ns`、hold WHS=`+0.101 ns`，TNS/THS 全 0；
- [x] 验收位流 `transformer_block_g2/pnr/generate_bitstream/runtime_quantizer_validation_top.sbit`，大小 2101696 B，SHA256=`220b771afbf8ea8d99806f3de27512748e2bd54913b1cc5e1f4a894647314236`；JTAG SRAM 下载 100%、DONE bit=1，未操作 Flash；固件 `PANGU50K G2 QUANT V1`，DDR3 初始化成功，最终 `trace_error/protocol_error` 均为 0；
- [x] 完整 Block 首版集成源码已经落地：统一 11 路 DDR3 仲裁、22 阶段 scheduler/controller、UART/DDR3 host controller、板级 top、完整动态/参数载荷、PDS 全流程脚本和板测 host 工具均已建立；固定板测契约将回读并比较 RoPE 前后 Q/K 等 18 个关键中间/最终张量；该勾选只表示“集成源码存在”，不表示上方完整 Block 功能已经通过；
- [x] 最新源码已通过完整顶层 Compile/Synthesize/Device Map；为切断已知物理关键路径，Attention 64×64 乘法收尾、Softmax score 差值、host 写入入口以及 RMSNorm 的规格化/RNE 路径均采用默认保持历史工程不变、仅 G2 开启的参数化流水。综合资源 `29154 LUT / 34677 FF / 52 DRM / 36 APM`，慢角 100 MHz setup WNS=`+0.814 ns`、TNS=0；Device Map 资源 `29163 LUT / 34677 FF / 52 DRM / 36 APM / 79 IO`，无硬错误；
- [x] 完整 Block 时序收敛已完成：依次切断 RoPE 表写使能高扇出路径、Attention Score 128 位符号恢复进位链，以及 Attention Output 的 AXI 写完成到 100 位累加器使能组合路径；上述流水均默认关闭、仅 G2 开启，不改变 E1/F4/F5/F6/G1 已验证工程的默认行为；
- [x] 最终完整 PDS 工程 `transformer_block_g2_output_ack_fix_full` 已通过 Compile、Synthesize、Device Map、Place & Route、Timing 和 Bitstream；Device Map 资源 `29011 LUT / 35053 FF / 52 DRM / 36 APM / 79 IO`，最终物理资源 `29086 LUT / 35053 FF / 52 DRM / 36 APM / 79 IO`；详细路由 162 轮后未布线网络为 0，hold 修复 6 轮；
- [x] 多角时序 `All Constraints Met`：慢角 `ddrphy_clkin` setup WNS=`+0.198 ns`、TNS=0，hold WHS=`+0.141 ns`、THS=0；快角 setup WNS=`+2.640 ns`、TNS=0，hold WHS=`+0.067 ns`、THS=0；`ref_clk` 慢/快 setup WNS 分别为 `+11.874/+14.220 ns`，恢复、移除和最小脉宽均无违例；
- [x] 验收位流 `transformer_block_g2/pnr/generate_bitstream/transformer_block_top.sbit`，大小 2101696 B，SHA256=`e4c3494152498583ae4a25540363fe3e828483fa7c0012a117e26e17fc557403`；通过专用脚本仅下载 JTAG SRAM，进度 100%、DONE bit=1，未执行 Flash 擦除或编程；固件 `PANGU50K G2 BLOCK V1`，DDR3 初始化成功；
- [x] 四组固定真实 hidden query/count=`0/1、1/2、5/6、15/16` 全部完成 18 个中间/最终张量逐位比较，共 `4/4` 用例、`72/72` 张量 PASS；最终输出 SHA256 与软件/G1 固定清单完全一致；
- [x] 完整 Block 板级随机/地址边界 `8/8 PASS`，seed=`20260820`，覆盖 count=`1/2/16`、四组随机窗口以及 query=`16383`、window=`16368..16383` 的 1 GiB KV Cache 末端；另完成交替 `INT16_MAX/MIN`、全 `INT16_MAX`、全 `INT16_MIN` 三组 hidden 数值边界，每组 18/18 张量 PASS，并分别触发第一/第二残差饱和 `434/443`、`391/456`、`403/431` 项；
- [x] 最终板卡状态 `block_busy=0、block_error=0、protocol_error=0、stage=IDLE、error_code=0`。G2 单个完整 layer0 Transformer Block 至此真实闭环完成，现允许进入阶段 H 的完整模型分层调度，但尚未开始真实 24 层连续执行、LM Head 或文本生成。

## 阶段 H：完整模型分层调度

- [x] 建立模型层描述表
- [x] 为每个张量记录 DDR3/主机文件偏移、形状和量化参数

H1 模型层描述与主机文件偏移契约（2026-08-03，已完成）：

- [x] 新增 `model_tools/model_layer_descriptor.py`、冻结清单 `model_layer_descriptor_reference.json` 和 `test_model_layer_descriptor.py`；真实 `.p50` 与外部 JSON 已完整交叉校验；
- [x] 纠正“模型实际层数”和“硬件容量”混淆：当前 Qwen2.5-0.5B 镜像 `num_hidden_layers=24`，层号连续 `0..23`；现有 KV/控制地址契约容量为 28 层，因此仅有 24 个活动层和 4 个未用容量槽，后续调度不得执行不存在的 layer24..27；
- [x] 纠正上下文上限差异：模型元数据 `max_position_embeddings=32768`，当前硬件 KV Cache 上限为 16384，阶段 H/I 第一版必须显式按硬件上限约束；
- [x] 冻结 290 个真实张量：2 个全局张量、24×12=288 个层内张量；Embedding 与 LM Head 权重 tied，共用 `model.embed_tokens.weight=[151936,896]`，另有最终 `model.norm.weight=[896]`；
- [x] 24 层结构完全同构：首层文件偏移 `72851456`，层步长 `7958528 B`，每层有效跨度 `7955456 B`，层间 4 KiB 对齐间隙 `3072 B`；12 类张量的 shape、INT4/FP16、data/scale 相对偏移、长度、group 数均冻结为层模板；
- [x] 描述器可展开任意层或全部层，已把 288 个层内张量的绝对 data/scale 主机文件偏移逐项与 P50 原目录比较；H1 新增 `9/9 PASS`，完整 `model_tools` 回归 `196/196 PASS`。

- [x] 设计权重流式加载方案
- [x] 决定模型权重是否常驻 DDR3 或按层重载
- [x] 设计 1 GiB DDR3 内存分区
- [ ] hidden state 双缓冲
- [x] 激活 scratch buffer
- [x] KV Cache 区域
- [x] GEMV 输出区

H2 参数换层与 1 GiB DDR3 内存契约（2026-08-03，已完成）：

- [x] 新增 `model_tools/full_model_memory_plan.py`、冻结摘要 `full_model_memory_plan_reference.json` 和 `test_full_model_memory_plan.py`；H2 专项 `10/10 PASS`，完整 `model_tools` 回归 `206/206 PASS`；
- [x] 明确 16K 上下文下无法让全部参数常驻：24 层 KV 占 768 MiB，剩余 256 MiB 无法同时容纳 251.63 MiB P50、G2 scratch、运行时量化缓冲和对齐；因此采用 `hybrid_global_resident_layer_reload`；
- [x] 顶部 `0x38000000` 起常驻 tied Embedding/LM Head packed INT4、FP16 raw scale 和最终 RMSNorm gamma，并预留 8,508,416 B LM Head combined-scale 与 1,215,488 B Q28 logits；全局区结束于 `0x3CE41000`，顶部仍余 52,162,560 B；
- [x] 每个 Transformer 层按 19 笔事务加载：P50 源数据 7,926,528 B，DDR 目标 7,961,088 B；INT4/FP16 scale 直拷，两个 gamma 转 Q6.10，Q/K/V bias 转 Q28 并按每行 32 B 展开；七组 combined scale 继续由 FPGA 运行时生成；
- [x] 低端保持 G2 地址兼容：`0x00000000..0x00FFFFFF` 为 runtime/scratch，slot A=`0x01000000..0x01FFFFFF` 为当前活动层，slot B=`0x02000000..0x02FFFFFF` 仅为后续高速预取保留，`0x03000000..0x07FFFFFF` 保留给 DMA/微码/staging；
- [x] 真实 24 层 KV 固定为 `0x08000000..0x37FFFFFF`，层步长 32 MiB、每 token K/V 共 2048 B、硬件上下文 16384；不访问容量中不存在模型参数的 layer24..27；
- [x] 当前 hidden 物理 ping/pong 使用已验证的 `block_hidden_q10@0x00000000` 与 `block_output_q10@0x00034000`，每层交接仅 1792 B；但 G2 顶层仍固定地址，H3 必须实现层末复制或地址选择后才可勾选 hidden 双缓冲；
- [x] 115200 UART 每层加载约 691.067 秒，24 层每 token 约 4.607 小时，只允许用于正确性验证；可用推理必须引入更高速传输，但不得改变已冻结的 DDR3 目标布局。

当前唯一实施点：实现 H3 层间控制与主机换层事务。第一版先使用 slot A 和 1792 B 层末复制，从 layer0 顺序执行到 layer23；必须逐层检查参数加载完成、cfg_layer、KV 层号、hidden 交接和错误状态。slot B 预取与直接地址 ping-pong 留到该基线通过后启用。

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
| `model_tools/qkv_linear_reference.py` | F1 真实 Q/K/V 统一加载、Q28 金标准、载荷和 GQA head-major 布局参考 |
| `model_tools/qkv_layer0_reference.json` | F1 Q/K/V 固定输出、载荷布局和关键数组 SHA256 清单 |
| `model_tools/test_qkv_linear_reference.py` | F1 投影形状、共享 hidden state、head 布局、载荷和真实 P50 集成测试 |
| `qkv_linear_layer0/rtl/qkv_linear_core.v` | F1 Q/K/V 共用的 K=896、group size 64 单行 Q28 GEMV 核心 |
| `qkv_linear_layer0/rtl/qkv_linear_ctrl.v` | F1 投影选择、动态 M=896/128、UART、DDR3 行调度和结果回读控制器 |
| `qkv_linear_layer0/pnr_seed5/run_seed5.tcl` | F1 时序全通过的 seed5/11 独立 PDS 全流程构建脚本 |
| `qkv_linear_layer0/README.md` | F1 量化定义、协议、GQA 布局、时序、位流和真实上板证据 |
| `tools/pangu_qkv_linear_host.py` | F1 Q/K/V 软件自检、固定和随机 hidden state 上板逐位比较工具 |
| `model_tools/rope_fixed_reference.py` | F2 Qwen2 split-half RoPE、Q28/Q1.30、RNE、误差界和位置表金标准 |
| `model_tools/rope_layer0_reference.json` | F2 固定位置、真实 Q/K 输出和关键数组 SHA256 清单 |
| `model_tools/test_rope_fixed_reference.py` | F2 配置、配对规则、RNE、载荷、真实模型和随机压力测试 |
| `rope_qk_layer0/rtl/rope_pair_q28_core.v` | F2 16 位 limb 顺序乘法、97 位旋转、RNE 和饱和核心 |
| `rope_qk_layer0/rtl/rope_qk_ctrl.v` | F2 UART、DDR3、连续位置表、Q/K head-major 调度和结果回读控制器 |
| `rope_qk_layer0/pnr/build_rope_qk.tcl` | F2 独立 PDS 全流程构建脚本 |
| `rope_qk_layer0/README.md` | F2 数学定义、协议、地址、时序、位流和真实上板证据 |
| `tools/pangu_rope_qk_host.py` | F2 软件自检、固定位置、自动递增和随机位置上板比较工具 |
| `model_tools/kv_cache_reference.py` | F3 28 层/16384 token 容量、地址、真实 K/V 和载荷金标准 |
| `model_tools/kv_cache_reference.json` | F3 四个真实固定槽、边界地址和数组/载荷 SHA256 清单 |
| `model_tools/test_kv_cache_reference.py` | F3 容量、地址连续性、越界、载荷和真实清单测试 |
| `kv_cache_f3/rtl/kv_cache_ctrl.v` | F3 UART、当前 K/V 写入、位置推进和历史分段 burst 读取控制器 |
| `kv_cache_f3/pnr/build_kv_cache.tcl` | F3 独立 PDS 全流程构建脚本 |
| `kv_cache_f3/README.md` | F3 容量决策、协议、地址、时序、位流和真实上板证据 |
| `tools/pangu_kv_cache_host.py` | F3 软件自检、真实固定、层间隔离和随机层/位置上板工具 |
| `model_tools/attention_score_reference.py` | F4 Q·K、1/8 缩放、14Q/2KV GQA、causal mask 和固定 score 金标准 |
| `model_tools/attention_score_f4_reference.json` | F4 四组真实固定窗口、K 地址、mask 和完整 score SHA256 清单 |
| `model_tools/test_attention_score_reference.py` | F4 GQA、RNE、缩放、mask、载荷和真实固定窗口测试 |
| `attention_score_f4/rtl/attention_score_core.v` | F4 DRM Q/K 缓存、精确 64×64 部分积重构、Q56 累加、RNE 和多头核心 |
| `attention_score_f4/rtl/attention_score_ctrl.v` | F4 UART、DDR3 Q/K 读取、14Q/2KV 调度、mask 和 score 部分写回控制器 |
| `attention_score_f4/pnr/build_attention_score.tcl` | F4 独立 PDS 全流程构建脚本 |
| `attention_score_f4/README.md` | F4 定点规则、协议、地址、时序、位流和真实上板证据 |
| `tools/pangu_attention_score_host.py` | F4 软件自检、真实固定窗口和随机层/窗口/Q/K 上板逐位比较工具 |
| `model_tools/softmax_fixed_reference.py` | F5 mask 感知 max、PWL exp、36 位求和、Q31 倒数和概率归一化金标准 |
| `model_tools/softmax_f5_reference.json` | F5 exp LUT、四组真实 F4 score 概率输出和 SHA256 固定清单 |
| `model_tools/test_softmax_fixed_reference.py` | F5 RNE、LUT 边界、全 mask、单有效、满窗口、载荷和真实固定测试 |
| `softmax_f5/rtl/softmax_core.v` | F5 score/LUT DRM 缓存、流水 PWL、恢复除法器和概率输出核心 |
| `softmax_f5/rtl/softmax_ctrl.v` | F5 UART、DDR3 score/LUT 载荷、14 heads 调度和概率部分写回控制器 |
| `softmax_f5/pnr/build_softmax.tcl` | F5 独立 PDS 全流程构建脚本 |
| `softmax_f5/README.md` | F5 定点规则、协议、地址、时序、位流和真实上板证据 |
| `tools/pangu_softmax_host.py` | F5 软件自检、真实固定 score 和随机 mask/窗口上板逐位比较工具 |
| `model_tools/attention_output_reference.py` | F6 概率×V、14Q/2KV GQA、Q59 累加、signed RNE、饱和和 `[14,64]/[896]` 金标准 |
| `model_tools/attention_output_f6_reference.json` | F6 四组真实固定概率/V、完整输出和载荷 SHA256 清单 |
| `model_tools/test_attention_output_reference.py` | F6 RNE ties、全 mask、单 token、满窗口、极端 V、拼接和真实清单测试 |
| `attention_output_f6/rtl/attention_output_core.v` | F6 DRM 概率/V 缓存、顺序 16-bit 部分积、100-bit Q59 累加、RNE 和流式输出核心 |
| `attention_output_f6/rtl/attention_output_ctrl.v` | F6 UART、F3 V 地址读取、16-token 装载、输出 byte-enable 写回和结果回读控制器 |
| `attention_output_f6/pnr/build_attention_output.tcl` | F6 独立 PDS 全流程构建脚本 |
| `attention_output_f6/README.md` | F6 加权和与多头拼接的定点规则、协议、地址、资源、时序、位流和真实上板证据 |
| `tools/pangu_attention_output_host.py` | F6 加权和软件自检、真实固定/边界和随机层/窗口/概率/V 上板逐位比较工具 |
| `model_tools/attention_oproj_reference.py` | F6 Q28 拼接输入量化、真实 layer0 O_proj 参数和完整 Q28 金标准 |
| `model_tools/attention_oproj_f6_reference.json` | F6 四组真实 Attention 输入、O_proj 输出和关键数组 SHA256 清单 |
| `model_tools/test_attention_oproj_reference.py` | F6 O_proj 输入转换、独立重算、真实参数、零输入和随机测试 |
| `attention_oproj_f6/rtl/attention_oproj_top.v` | F6 O_proj 独立顶层，直接复用已验证完整 Linear controller/core 并保持 DDR3 约束层级 |
| `attention_oproj_f6/pnr/build_attention_oproj.tcl` | F6 O_proj 独立 PDS 全流程构建脚本 |
| `attention_oproj_f6/README.md` | F6 O_proj 量化规则、复用边界、时序、位流和真实上板证据 |
| `tools/pangu_attention_oproj_host.py` | F6 O_proj 软件自检、四组真实固定和随机/边界上板逐位比较工具 |
| `model_tools/attention_residual_reference.py` | F6 连贯 layer0 Attention 软件链、Q28→Q10 signed RNE 和残差金标准 |
| `model_tools/attention_residual_f6_reference.json` | F6 1/2/6/16-token 完整 Attention 输出和载荷 SHA256 清单 |
| `model_tools/test_attention_residual_reference.py` | F6 RNE tie、INT64 极值、双重饱和、真实完整链和随机测试 |
| `attention_residual_f6/rtl/attention_residual_core.v` | F6 O_proj Q28 重标定、两级饱和、残差相加和结果打包核心 |
| `attention_residual_f6/rtl/attention_residual_ctrl.v` | F6 residual/O_proj UART、DDR3 burst、结果写回和回读控制器 |
| `attention_residual_f6/pnr/build_attention_residual.tcl` | F6 完整 Attention 残差独立 PDS 全流程构建脚本 |
| `attention_residual_f6/README.md` | F6 连贯软件链、定点边界、协议、时序、位流和真实上板证据 |
| `tools/pangu_attention_residual_host.py` | F6 完整 Attention 软件自检、固定和随机/边界上板逐位比较工具 |
| `model_tools/post_attention_layernorm_reference.py` | G1 连贯 Attention 输出、真实 post-attention gamma、LUT256 RMSNorm 和载荷金标准 |
| `model_tools/post_attention_layernorm_g1_reference.json` | G1 四组真实 MLP 输入、归一化输出和载荷 SHA256 清单 |
| `model_tools/test_post_attention_layernorm_reference.py` | G1 张量、Q10 往返、载荷、连贯固定输入和 1000 轮随机/边界测试 |
| `post_attention_layernorm_g1/pnr/build_post_attention_layernorm.tcl` | G1 隔离复用 E1 RTL 的独立 PDS 全流程构建脚本 |
| `post_attention_layernorm_g1/pnr/program_sram.tcl` | G1 独立位流仅下载 FPGA 易失性 SRAM 的脚本 |
| `post_attention_layernorm_g1/README.md` | G1 复用边界、定点规则、固定输入、时序、位流和真实上板证据 |
| `tools/pangu_post_attention_layernorm_host.py` | G1 软件自检、四组真实固定和随机/边界上板逐位比较工具 |
| `model_tools/mlp_gate_up_reference.py` | G1 gate/up 共享输入量化、真实双权重、Q28 独立重算和载荷金标准 |
| `model_tools/mlp_gate_up_g1_reference.json` | G1 四组真实 gate/up 完整输出和关键载荷 SHA256 清单 |
| `model_tools/test_mlp_gate_up_reference.py` | G1 双投影 shape、共享激活、无 bias、载荷、独立重算和零输入测试 |
| `mlp_gate_up_g1/rtl/mlp_gate_up_ctrl.v` | G1 4864 行 DDR3 流式 Linear 控制器 |
| `mlp_gate_up_g1/pnr/build_mlp_gate_up.tcl` | G1 gate/up 独立 PDS 全流程构建脚本 |
| `mlp_gate_up_g1/pnr/program_sram.tcl` | G1 gate/up 位流仅下载 FPGA 易失性 SRAM 的脚本 |
| `mlp_gate_up_g1/README.md` | G1 双投影定点、载荷、时序、位流和真实上板证据 |
| `tools/pangu_mlp_gate_up_host.py` | G1 gate/up 软件自检、四组固定和随机/边界上板逐位比较工具 |
| `model_tools/mlp_silu_reference.py` | G1 gate Q28→Q6.10 signed RNE、显式饱和、PWL64 SiLU 与载荷金标准 |
| `model_tools/mlp_silu_g1_reference.json` | G1 四组真实 `SiLU(gate)` 完整输出、PWL 表和载荷 SHA256 清单 |
| `model_tools/test_mlp_silu_reference.py` | G1 RNE tie、INT64 极值、尾部、饱和、载荷与真实固定输入测试 |
| `mlp_silu_g1/rtl/mlp_silu_core.v` | G1 4-bank gate 缓存、Q28 重标定、PWL64 插值和 Q6.10 打包核心 |
| `mlp_silu_g1/rtl/mlp_silu_ctrl.v` | G1 UART、DDR3 上传/burst、核心调度、结果写回和回读控制器 |
| `mlp_silu_g1/pnr/build_mlp_silu.tcl` | G1 `SiLU(gate)` 独立 PDS 全流程构建脚本 |
| `mlp_silu_g1/pnr/program_sram.tcl` | G1 `SiLU(gate)` 位流仅下载 FPGA 易失性 SRAM 的脚本 |
| `mlp_silu_g1/README.md` | G1 数值定义、协议、地址、时序、位流和真实上板证据 |
| `tools/pangu_mlp_silu_host.py` | G1 软件自检、四组真实固定和随机/边界上板逐位比较工具 |
| `model_tools/mlp_silu_up_mul_reference.py` | G1 完整 80-bit Q38 乘积、对称 RNE、int64 饱和与真实乘法金标准 |
| `model_tools/mlp_silu_up_mul_g1_reference.json` | G1 四组真实 `SiLU(gate) × up` 完整输出和载荷 SHA256 清单 |
| `model_tools/test_mlp_silu_up_mul_reference.py` | G1 正负 RNE tie、全位宽、双向饱和、载荷和真实固定输入测试 |
| `mlp_silu_up_mul_g1/rtl/mlp_silu_up_mul_core.v` | G1 5-bank DRM 缓存、16×16 limb 乘法、80-bit 重构、RNE 和 Q28 输出核心 |
| `mlp_silu_up_mul_g1/rtl/mlp_silu_up_mul_ctrl.v` | G1 UART、DDR3 双输入上传/burst、核心调度、结果写回和回读控制器 |
| `mlp_silu_up_mul_g1/pnr_seed17/build_seed17.tcl` | G1 通过全部多角时序的 seed17/29 独立 PDS 全流程脚本 |
| `mlp_silu_up_mul_g1/pnr_seed17/program_sram.tcl` | G1 验收位流仅下载 FPGA 易失性 SRAM 的脚本 |
| `mlp_silu_up_mul_g1/README.md` | G1 乘法数值规则、协议、地址、时序、位流和真实上板证据 |
| `tools/pangu_mlp_silu_up_mul_host.py` | G1 软件自检、四组真实固定和连续随机序列上板逐位比较工具 |
| `model_tools/mlp_down_proj_reference.py` | G1 verified Q28 输入量化、真实 down_proj 权重、76 组 Q28 累加与载荷金标准 |
| `model_tools/mlp_down_proj_g1_reference.json` | G1 四组真实 down_proj 完整输出、载荷和边界 SHA256 清单 |
| `model_tools/test_mlp_down_proj_reference.py` | G1 shape、Q28 转换、scale 饱和、INT64 安全、载荷与真实来源测试 |
| `mlp_down_proj_g1/rtl/mlp_down_proj_core.v` | G1 K=4864、76 groups、MAC16 与 signed int64 Q28 累加核心 |
| `mlp_down_proj_g1/rtl/mlp_down_proj_ctrl.v` | G1 UART、2.499 MB DDR3 上传、长 burst 分段、896 行调度和结果回读控制器 |
| `mlp_down_proj_g1/pnr/build_mlp_down_proj.tcl` | G1 down_proj 独立 PDS 全流程构建脚本 |
| `mlp_down_proj_g1/pnr/program_sram.tcl` | G1 down_proj 位流仅下载 FPGA 易失性 SRAM 的脚本 |
| `mlp_down_proj_g1/README.md` | G1 数值定义、载荷、地址、时序、位流和真实上板证据 |
| `tools/pangu_mlp_down_proj_host.py` | G1 软件自检、四组真实固定和随机/边界上板逐位比较工具 |
| `model_tools/mlp_residual_reference.py` | G1 第二处残差正确支路配对、Q28→Q6.10 RNE、两级饱和与完整 MLP 金标准 |
| `model_tools/mlp_residual_g1_reference.json` | G1 四组连贯真实完整 MLP 输出和上传载荷 SHA256 清单 |
| `model_tools/test_mlp_residual_reference.py` | G1 RNE tie、INT64 极值、两级饱和、真实支路来源和固定清单测试 |
| `mlp_residual_g1/rtl/mlp_residual_core.v` | G1 hidden/down 多 bank 缓存、Q28 重标定、残差相加和 Q6.10 打包核心 |
| `mlp_residual_g1/rtl/mlp_residual_ctrl.v` | G1 UART、DDR3 上传/burst、核心调度、结果写回和回读控制器 |
| `mlp_residual_g1/pnr/build_mlp_residual.tcl` | G1 第二处残差独立 PDS 全流程构建脚本 |
| `mlp_residual_g1/pnr/program_sram.tcl` | G1 验收位流仅下载 FPGA 易失性 SRAM 的脚本 |
| `mlp_residual_g1/README.md` | G1 第二处残差数值、协议、地址、时序、位流和真实上板证据 |
| `tools/pangu_mlp_residual_host.py` | G1 软件自检、四组真实固定和连续随机序列上板逐位比较工具 |
| `model_tools/runtime_quantizer_validation.py` | G2 七矩阵运行时量化固定清单、精确随机/边界金标准、地址/burst/padding 事务模型 |
| `model_tools/runtime_quantizer_g2_reference.json` | G2 七个真实矩阵的配置、载荷、逐位结果与 AXI trace SHA256 清单 |
| `model_tools/test_runtime_quantizer_validation.py` | G2 量化协议、padding、trace、精确 half-tie 和随机事务回归 |
| `transformer_block_g2/rtl/runtime_quantizer_validation_top.v` | G2 Q6.10/Q28 自动逐位验证、DDR3、UART 与 trace checker 顶层 |
| `transformer_block_g2/pnr/build_runtime_quantizer_validation.tcl` | G2 量化子系统通过全部多角时序的 seed5/11 PDS 全流程脚本 |
| `transformer_block_g2/pnr/program_runtime_quantizer_validation_sram.tcl` | G2 量化验收位流仅下载 FPGA 易失性 SRAM 的脚本 |
| `tools/pangu_runtime_quantizer_host.py` | G2 七矩阵固定与 Q6.10/Q28 随机/边界板级逐位比较工具 |
| `transformer_block_g2/README.md` | G2 软件契约、量化板级闭环证据与完整 Block 下一实施点 |
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
G2 的唯一下一任务：完整 layer0 Transformer Block 的 11 路 DDR3 仲裁、22 阶段
controller/top、host/PDS 和板测工具已经落地，最新源码也已通过 Compile/Synthesize/Device Map，
但仍是未验收硬件。现在必须完成 Place & Route、全部多角时序和新位流，记录 SHA256 后仅通过 JTAG 下载 SRAM；再用
四组固定真实 hidden 逐位比较 18 个关键中间/最终张量，并完成随机/边界板卡压力。旧的违例位流
和旧 placement 数据库均不得复用。完整 Block 单独通过前不得进入 28 层调度、LM Head 或文本生成。
```
