# 盘古 PGL50H AI 大模型 FPGA 项目总路线

> 本文件是项目的**唯一权威任务清单**。后续对话和开发会话必须先读取本文件，再决定继续做什么。
>
> 最后更新：2026-07-23

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

# 5. 当前唯一下一任务

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

- [ ] 设计 GEMV DDR3 地址布局
- [ ] 激活 `x` 写入 DDR3，并只读取/缓存一次
- [ ] 4 行 packed INT4 权重连续写入 DDR3
- [ ] 每行 K=64，拆成 4 个 MAC16 分块
- [ ] 每行跨 4 个分块进行 INT32 累加
- [ ] 生成 4 个 INT32 输出
- [ ] 输出向量批量写回 DDR3
- [ ] UART 返回整个输出向量
- [ ] Python 对 4 个输出逐元素比较
- [ ] 固定向量通过
- [ ] 至少 1000 轮随机压力测试通过
- [ ] PDS 全流程、时序和真实上板通过

### D1.2 扩展为参数化 GEMV

- [ ] 支持运行时参数 `M` 和 `K`
- [ ] `K` 不是 16 整数倍时支持尾块屏蔽
- [ ] 支持更长的 AXI 256 bit burst
- [ ] 权重行地址自动递增
- [ ] 输出地址自动递增
- [ ] 32 位累加溢出边界测试
- [ ] UART 协议增加 GEMV 配置和启动命令
- [ ] Python 工具可自动产生不同 M/K 的随机矩阵
- [ ] 至少覆盖 `M={1,4,16,64}`、`K={16,64,256,896}` 的测试

### D1.3 GEMV 性能基础设施

- [ ] 统计 DDR3 读取周期
- [ ] 统计 MAC 计算周期
- [ ] 统计单次 GEMV 总周期
- [ ] 增加性能计数器并可由上位机读取
- [ ] 记录实测带宽、GMAC/s 和利用率
- [ ] 明确瓶颈是 DDR3、MAC 数量还是控制开销

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

- [ ] 完整解析 `.p50` 文件头、张量目录和数据偏移
- [ ] 验证 JSON 元数据与二进制张量完全一致
- [ ] 明确每个线性层的权重形状和存储顺序
- [ ] 明确 INT4 编码方式、分组大小、scale 和 zero point
- [ ] Python 可提取任意一行/一块真实模型权重
- [ ] FPGA GEMV 支持真实模型的分组反量化或定点缩放
- [ ] 选择统一的激活量化格式
- [ ] 定义 scale 的定点格式，例如 Q 格式
- [ ] 验证一个真实线性层的小切片与 PyTorch/NumPy 一致
- [ ] 验证一个完整真实线性层输出误差在规定范围内

验收：至少完成模型中的一个完整 Linear 层，输出与软件量化参考一致。

## 阶段 E：基础非矩阵算子

### E1 RMSNorm

- [ ] 平方和累加
- [ ] 均值计算
- [ ] `rsqrt` 近似方案确定：查表、Newton-Raphson 或软件辅助
- [ ] gamma 权重乘法
- [ ] 定点格式和饱和/舍入规则
- [ ] Python 逐元素比较
- [ ] 随机压力测试、时序和上板验证

### E2 元素级运算

- [ ] 残差加法
- [ ] 定点乘法和缩放
- [ ] 饱和与舍入
- [ ] SiLU 或 `x·sigmoid(x)` 近似
- [ ] element-wise multiply
- [ ] Python 参考与误差阈值

### E3 Embedding/查表

- [ ] Token ID 到 embedding 行地址映射
- [ ] DDR3 中读取一个 token 的 embedding
- [ ] 转换为统一激活格式
- [ ] 与软件参考比较

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
| `tools/pangu_ddr_mac16_host.py` | INT8/INT4 上位机验证工具 |
| `model_tools/export_qwen25_fpga.py` | 模型转换工具 |
| `model_tools/verify_p50_image.py` | 模型文件验证工具 |

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
建立独立的 GEMV 验证版本：M=4、K=64。
激活读取一次，4行packed INT4权重连续读取；
每行执行4次MAC16并跨分块INT32累加；
返回4个输出，与Python逐元素比较；
完成PDS时序和真实上板1000轮压力测试。
```
