# PGL50H AI/LLM FPGA 加速项目

目标是在盘古 Logos `PGL50H-6IFBG484` 上逐步完成 Qwen2.5-0.5B + LoRA 的 INT4 推理。

## 进入项目后先读

1. [`PROJECT_ROADMAP.md`](PROJECT_ROADMAP.md)：完整开发路线、当前任务和验收标准。
2. [`PROJECT_PROGRESS_2026-07-23.md`](PROJECT_PROGRESS_2026-07-23.md)：已完成能力和真实上板证据。
3. [`ddr_mac16_integration/README.md`](ddr_mac16_integration/README.md)：已验证的 DDR3 + MAC16 + INT4 单点积工程。
4. [`gemv_int4_m4k64/README.md`](gemv_int4_m4k64/README.md)：已验证的固定 M=4、K=64 packed INT4 GEMV 工程。
5. [`gemv_int4_param/README.md`](gemv_int4_param/README.md)：已验证的运行时 M/K、尾块屏蔽参数化 GEMV 工程。
6. [`gemv_int4_perf/README.md`](gemv_int4_perf/README.md)：已验证的 D1.3 周期计数、带宽、GMAC/s 和利用率分析工程。
7. [`model_tools/README.md`](model_tools/README.md)：已确认的 `.p50` 文件头、真实张量目录、INT4 格式和按名提取工具。
8. [`gemv_int4_group_q28/README.md`](gemv_int4_group_q28/README.md)：已验证的真实 q_proj M=4、K=896 分组 UQ4.28 定点小闭环。
9. [`gemv_int4_qproj_full/README.md`](gemv_int4_qproj_full/README.md)：已验证的 layer0 q_proj 完整 M=896、K=896 真实 Linear 层闭环。
10. [`rmsnorm_k896/README.md`](rmsnorm_k896/README.md)：已验证的 layer0 input_layernorm K=896 定点 RMSNorm 闭环。
11. [`elementwise_k896/README.md`](elementwise_k896/README.md)：已验证的 K=896 残差、缩放、元素乘法和 PWL64 SiLU 闭环。
12. [`embedding_k896/README.md`](embedding_k896/README.md)：已验证的真实 tied Embedding Token 行地址、INT4/UQ4.28 到 Q6.10 闭环。
13. [`qkv_linear_layer0/README.md`](qkv_linear_layer0/README.md)：已验证的真实 layer0 Q/K/V、GQA head-major 布局和统一 Q28 闭环。
14. [`rope_qk_layer0/README.md`](rope_qk_layer0/README.md)：已验证的 Qwen2 split-half Q/K RoPE、位置递增和 Q28/Q1.30 闭环。
15. [`kv_cache_f3/README.md`](kv_cache_f3/README.md)：已验证的 28 层、16384 token、真实 K/V 写入、历史顺序读取和防覆盖闭环。
16. [`attention_score_f4/README.md`](attention_score_f4/README.md)：已验证的 14Q/2KV GQA Attention Score、1/8 缩放和 causal mask 闭环。
17. [`softmax_f5/README.md`](softmax_f5/README.md)：已验证的 mask 感知 Softmax、PWL exp、Q31 倒数和概率归一化闭环。
18. [`attention_output_f6/README.md`](attention_output_f6/README.md)：已验证的 F5 概率×F3 V、14Q/2KV GQA、Q59 累加、RNE 和 `[14,64]/[896]` Attention 输出闭环。
19. [`attention_oproj_f6/README.md`](attention_oproj_f6/README.md)：已验证的 F6 `[896]` Attention 输入量化、真实 layer0 O_proj M896K896 分组 INT4 和 signed int64 Q28 闭环。
20. [`attention_residual_f6/README.md`](attention_residual_f6/README.md)：已验证的完整 layer0 Attention 连贯软件链、Q28→Q6.10 重标定和第一处残差闭环。
21. [`post_attention_layernorm_g1/README.md`](post_attention_layernorm_g1/README.md)：已验证的真实 layer0 post_attention_layernorm、连贯 MLP 输入和独立 RMSNorm 闭环。
22. [`mlp_gate_up_g1/README.md`](mlp_gate_up_g1/README.md)：已验证的真实 layer0 gate/up 双投影、共享激活和完整 `[4864]` Q28 闭环。
23. [`mlp_silu_g1/README.md`](mlp_silu_g1/README.md)：已验证的 gate Q28→Q6.10 signed RNE、PWL64 `SiLU(gate)` 和完整 `[4864]` 非线性闭环。
24. [`mlp_silu_up_mul_g1/README.md`](mlp_silu_up_mul_g1/README.md)：已验证的完整 signed 80-bit Q38 乘积、RNE、int64 饱和和 `[4864]` `SiLU(gate) × up` 闭环。
25. [`mlp_down_proj_g1/README.md`](mlp_down_proj_g1/README.md)：已验证的真实 layer0 `down_proj=[896,4864]`、76-group INT4/UQ4.28 和完整 `[896]` Q28 闭环。
26. [`mlp_residual_g1/README.md`](mlp_residual_g1/README.md)：已验证的正确 residual 支路、down Q28→Q6.10 RNE、两级饱和和完整 layer0 MLP 闭环。
27. [`transformer_block_g2/README.md`](transformer_block_g2/README.md)：已验证的 G2 完整 layer0 Transformer Block；统一 11 路 DDR3 仲裁、22 阶段 scheduler/controller、七矩阵运行时量化、完整 PDS 多角时序、JTAG SRAM、四组真实 hidden 的 18 张量逐位比较，以及随机、地址末端和正负饱和边界均已闭环。
28. [`full_model_h3/README.md`](full_model_h3/README.md)：正在进行的 H3 真实 24 层换层基线；456 笔参数事务、配置读回、DDR hidden copy、主机顺序器和独立 PDS 前端已建立，但尚未完成时序、位流或板级 24 层连续执行。

## 当前状态

已经真实上板完成从单点积、完整真实 Linear 层到 RMSNorm、元素级非线性、Embedding、完整 Q/K/V、RoPE、KV Cache、Attention Score、Softmax、Attention 输出加权和、真实 O_proj、两处残差与完整 MLP；G2 已把这些算子连成同一 hidden state 出发的完整 layer0 Transformer Block，并完成独立 PDS、逐位和边界板级闭环：

```text
长度16单点积：
DDR3写入 → 2拍×256位AXI burst读取 → INT8/INT4处理
→ MAC16点积 → 结果写回 → UART返回 → Python比较

固定M=4、K=64 GEMV：
激活2拍读取并缓存一次 → 4行packed INT4权重4拍连续读取
→ 每行4次MAC16分块累加 → 4个INT32输出 → Python逐元素比较

运行时参数化 GEMV：
支持 1<=M<=64、1<=K<=896 → 长burst分段读取 → 权重行地址自动递增
→ ceil(K/16)次MAC16 → 尾块硬件屏蔽 → 输出地址自动递增 → Python逐元素比较
```

参数化工程已覆盖 24 种标准/尾块形状；固定 M4K64 和尾块 M16K65 均分别通过 1000 轮真实上板随机测试。D1.3 已增加激活/权重 DDR3 读取、MAC 计算和总周期计数，上位机可计算实测带宽、GMAC/s、MAC16 利用率并自动判断主要瓶颈。PDS 编译、综合、布局布线、多角时序和真实上板验证全部满足。

D2 的模型格式解析和真实 Linear 软件参考均已完成：真实 `.p50` 镜像的固定头、290 个张量目录、形状、偏移、长度和对齐已全量校验；外部 JSON 与镜像内嵌 JSON 逐字段完全一致。Python 工具可按张量名提取任意 INT4 行、跨 group 二维块或 FP16 数据，并返回量化值、FP16 scale 与反量化结果。

真实 Linear 已统一采用逐向量对称 INT8 激活和 UQ4.28 组合 scale。layer0 `q_proj` 的 M=4、K=896 固定切片已建立 P50 浮点、量化激活浮点和硬件等价 Q28 三条参考路径；定点最大绝对误差 `3.1277186e-6`，低于理论上界 `3.8200990e-5`。原有解析和新增量化测试共 13/13 PASS，另有 1000/1000 随机软件压力测试通过。

D2 的首个真实模型 FPGA 小闭环也已完成。独立工程 `gemv_int4_group_q28` 对 layer0 `q_proj` 前 4 行、完整 K=896 输入执行每 64 元素分组 INT32 点积、UQ4.28 乘法、signed INT64 Q28 跨组累加和 bias 加法。固定向量 FPGA 输出 `[207253689, -173360554, 287606739, -223225713]` 与软件参考逐位一致，scale bit31/`0xFFFFFFFF` 边界通过，随机上板 `1000/1000 PASS`。PDS 多角时序 `All Constraints Met`，慢角 100 MHz WNS=`+0.909 ns`、TNS=0；位流 SHA256=`d8c7d194d4d8ce1e5d189df39fae5fc904030fe4be6e981a5876a4df73ea17bd`。

D2 完整真实 Linear 层现已完成。独立工程 `gemv_int4_qproj_full` 将 layer0 `q_proj` 扩展到完整 M=896、K=896：逐行读取真实 packed INT4 权重、UQ4.28 scale 和 signed Q28 bias，每 4 行结果立即流式写回 DDR3，最终返回 896 个 signed int64。固定完整层真实上板逐位一致，输出 SHA256=`ea1f04bf4ff313dad07025ff35e66a088f13afd28d817422b89bb135f63525a0`；随机激活上板 `3/3 PASS`，软件压力测试 `1000/1000 PASS`。PDS 全流程和多角时序通过，慢角 100 MHz WNS=`+0.670 ns`、TNS=0；位流 SHA256=`432454b80678c11f493856cb725d791e271d86eada1b5cabccefc0d7486f8894`。

E1 RMSNorm 也已完成。独立工程 `rmsnorm_k896` 对真实 `model.layers.0.input_layernorm.weight` 执行 K=896 定点 RMSNorm：输入、gamma 和输出使用 signed Q6.10，40 位平方和、Q12.20 均值/epsilon、UQ12.20 LUT256 rsqrt，全部采用 RNE 和显式饱和。固定向量 896 个输出与 Python LUT 金标准逐位一致，输出 SHA256=`1f52890780e0f4cc0f734d47a4e3bdb28c3c964b8734b442d7781d4ca155a4f0`；软件随机 `1000/1000 PASS`，真实上板随机 `300/300 PASS`。PDS 全流程和多角时序通过，慢角 100 MHz WNS=`+0.374 ns`、TNS=0；位流 SHA256=`94c82d1ef6adf563043c6f90f5744ec258156d85c6db134389132ae4f2938b11`。

E2 元素级运算现已完成。独立工程 `elementwise_k896` 支持 signed Q6.10 残差加法、定点缩放、元素乘法和 64 段端点 PWL SiLU，统一使用 RNE 与显式饱和。PWL64 在完整 int16 输入域最大误差为 4 Q10 LSB，端点表仅 1040 bit。四种操作的固定 K=896 向量均与 Python 逐位一致；软件随机 `1000/1000 PASS`，真实上板随机累计 `300/300 PASS`。PDS 全流程、多角建立/保持/恢复/移除均通过，慢角 100 MHz WNS=`+0.580 ns`、TNS=0；位流 SHA256=`809b436f1c369d66a20c5f2faaa8e684a15a3963d659b95d080e342c3a7d9d50`。

E3 Embedding/查表现已完成。独立工程 `embedding_k896` 对真实 tied `model.embed_tokens.weight`（shape `[151936,896]`、group size 64）实现 Token ID 到 512 B DDR3 行槽映射，读取 448 B packed signed INT4 和 14 个 UQ4.28 scale，逐元素 RNE 转为 signed Q6.10。四个固定 Token `[0,1,2026,151935]` 的 896 个输出均与 Python 逐位一致；软件/载荷随机 `1000/1000 PASS`，真实上板随机 `300/300 PASS`。PDS 全流程和所有角时序通过，慢角 100 MHz WNS=`+0.679 ns`、TNS=0；位流 SHA256=`cd0e138e494875035cf5c66d76eaf250729625c172bf51c935b831d31c45c0fa`。

F1 Q/K/V 线性层现已完成。独立工程 `qkv_linear_layer0` 统一运行真实 layer0 `q_proj=[896,896]`、`k_proj=[128,896]`、`v_proj=[128,896]`，共用逐向量对称 INT8 hidden state、UQ4.28 combined scale 和 signed int64 Q28 数据通路；输出按 Q=`[14,64]`、K/V=`[2,64]` 的 head-major GQA 布局排列。固定 Q/K/V 全输出均与 Python 逐位一致；完整软件回归 `48/48 PASS`，QKV 软件随机 `1000/1000 PASS`，真实上板随机完整 Q+K+V `3/3 PASS`。seed5/11 PDS 全流程和所有角时序通过，慢角 setup WNS=`+0.363 ns`、TNS=0、hold WHS=`+0.169 ns`、THS=0；位流 SHA256=`e3a4b6849a5716f38d6bdd3fbd039d46f2d350a32a0417ee347462d1a8f96e26`。

F2 RoPE 现已完成。独立工程 `rope_qk_layer0` 对 F1 真实 Q=`[14,64]`、K=`[2,64]` 执行 Qwen2 split-half 旋转：`dim i` 与 `dim i+32` 配对，`rope_theta=1000000`，Q/K 为 signed Q28，sin/cos 为 signed Q1.30；四个乘积精确重构后在 97 位中加减并单次 RNE。固定位置 `[0,1,2026,32767]` 的 Q/K 全输出逐位一致，连续位置 `2026..2033` 自动递增和复位重放通过；完整软件回归 `55/55 PASS`，软件随机 `1000/1000 PASS`，真实上板随机位置 `300/300 PASS`。PDS 所有角时序通过，慢角 setup WNS=`+0.988 ns`、TNS=0、hold WHS=`+0.171 ns`、THS=0；位流 SHA256=`25396ffc894abc15b81ab99f62619f3694e7e662f620f3c6a89e28ae116d153a`。

F3 KV Cache 现已完成。独立工程 `kv_cache_f3` 将 1 GiB DDR3 的低端 128 MiB 保留给后续权重/激活，将高端 896 MiB 划分为 28 个 32 MiB 层区；每 token 保存 K/V 各 `[2,64]` signed int64 Q28，共 2048 B，因此硬件上下文上限为 16384。真实 F2-K/F1-V 固定位置 `layer0/0..1、layer13/2026、layer27/16383` 全部逐位一致，位置自动推进、最后槽 1 GiB 边界、下一写入拒绝及 layer3/17 同位置防覆盖通过；完整软件回归 `64/64 PASS`，软件随机 `1000/1000 PASS`，真实随机层/位置 `300/300 token PASS`。PDS 所有角时序通过，慢角 core setup WNS=`+1.781 ns`、TNS=0、hold WHS=`+0.171 ns`、THS=0；位流 SHA256=`11a0240a2ee42f0c92b6a5919f4a4b71ceb7bb806b55f1810b4ef3ff88d23216`。

F4 Attention Score 现已完成。独立工程 `attention_score_f4` 直接消费 F1/F2 的 Q/K Q28 输出并复用 F3 K Cache 地址；实现 64 维精确 Q56 点积、`1/sqrt(64)=1/8` 的 signed RNE 右移 31 位、`14Q -> 2KV` GQA 映射以及 `INT64_MIN` causal mask，固定输出为 `[14,16]` head-major signed int64 Q28。四组真实固定窗口和 mask/末地址边界全部逐位一致；完整软件回归 `73/73 PASS`，软件随机 `1000/1000 PASS`，真实随机层/窗口/Q/K `100/100 PASS`。PDS 所有角时序通过，慢角 core setup WNS=`+0.482 ns`、TNS=0、hold WHS=`+0.170 ns`、THS=0；位流 SHA256=`669cb5b23cb6c5d33d0003f32452e57cda251751179c318c1b5d8f2ed8c0e0f8`。

F5 Softmax 现已完成。独立工程 `softmax_f5` 直接消费 F4 `[14,16]` signed Q28 score，实现 mask 感知 max、减最大值、`[-16,0]`/步长 `1/32` 的 513 点 UQ1.31 PWL exp、36 位求和、恢复除法 Q31 倒数和概率归一化；输出为 `[14,16]` unsigned UQ1.31。全 mask、单有效、部分/满窗口、全等 score 和极端差值行为均已固定。四组真实 F4 窗口上板逐位一致；完整软件回归 `83/83 PASS`，软件随机 `1000/1000 PASS`，真实随机 mask/窗口 `100/100 PASS`，最坏 float64 概率误差 `2.96390625578e-05`。PDS 所有角时序通过，慢角 setup WNS=`+0.227 ns`、TNS=0、hold WHS=`+0.143 ns`、THS=0；位流 SHA256=`d6e505ea5495c6054a447608406db0f93855ef55dbfc357c8d113b00adba34fe`。

F6 Attention 输出的第一段现已完成。独立工程 `attention_output_f6` 直接消费 F5 `[14,16]` unsigned UQ1.31 概率并按 F3 地址读取最多 16 token 的 `[2,64]` signed int64 Q28 V；按 `14Q -> 2KV` GQA 形成 signed Q59 乘积，在 100 bit 中精确累加后单次 signed RNE 右移 31 位，输出 `[14,64]` 并 head-major 拼接为 `[896]`。四组真实固定窗口、全 mask、单 token `INT64_MIN/MAX` 极端 V，以及 16-token INT64 正/负双向饱和全部上板逐位一致；完整软件回归 `93/93 PASS`，软件随机 `1000/1000 PASS`，真实随机层/窗口/概率/V `100/100 PASS`。PDS 所有角时序通过，慢角 setup WNS=`+0.825 ns`、TNS=0、hold WHS=`+0.112 ns`、THS=0；位流 SHA256=`d7e64c58b73f8ca93f7a7dd981feabe5cc48f9b43e6b2ff0d8f60155886f36a3`。

F6 Attention O_proj 现已完成。独立工程 `attention_oproj_f6` 将第一段真实 `[896]` signed int64 Q28 拼接结果按逐向量对称规则量化为 INT8，读取真实 `model.layers.0.self_attn.o_proj.weight=[896,896]` 的 groupwise signed INT4 权重和 FP16 scale；真实 `.p50` 不含 O_proj bias，因此硬件 bias 全 0。四组 1/2/6/16-token 固定输入的 896 个输出全部上板逐位一致；完整软件回归 `100/100 PASS`，软件随机/边界 `1000/1000 PASS`，真实板卡随机/边界 `4/4 PASS`。PDS 所有角时序通过，慢角 setup WNS=`+0.614 ns`、TNS=0、hold WHS=`+0.171 ns`、THS=0；位流 SHA256=`017517f877f29e62d945ecd3ae4ba22c2d690b6e6b92778eb0502ba7ac115533`。

F6 Attention 残差与完整子层现已完成。独立工程 `attention_residual_f6` 将真实 O_proj `[896]` signed int64 Q28 使用 signed RNE 右移 18 位并饱和到 signed Q6.10，再与对应原 hidden state 扩展相加并再次饱和。软件端首次建立同一批 hidden state 连贯经过 input RMSNorm、Q/K/V、RoPE、Score、Softmax、V 加权、多头拼接、O_proj 和残差的完整 layer0 Attention 参考链。1/2/6/16-token 四组固定输出全部上板逐位一致；完整软件回归 `105/105 PASS`，软件随机/边界 `1000/1000 PASS`，真实板卡随机/边界累计 `300/300 PASS`。PDS 所有角时序通过，慢角 setup WNS=`+1.493 ns`、TNS=0、hold WHS=`+0.112 ns`、THS=0；位流 SHA256=`609e1f569aa1e4579cffb995b0d7d0bc89fa34529790b35e8b26d6778226bcbd`。

G1 MLP 输入 `post_attention_layernorm` 现已完成。独立工程 `post_attention_layernorm_g1` 直接消费上述四组 1/2/6/16-token 完整 Attention 子层 `[896]` signed Q6.10 输出，并读取真实 `model.layers.0.post_attention_layernorm.weight`；数值规则严格复用但隔离 E1 RMSNorm 的 Q6.10、Q12.20、LUT256 UQ12.20 rsqrt、RNE 和显式饱和。四组固定输出全部 896/896 上板逐位一致；新增测试 `5/5 PASS`，完整软件回归 `110/110 PASS`，软件随机/边界 `1000/1000 PASS`，真实板卡 `300/300 PASS`。PDS 所有角时序通过，慢角 setup WNS=`+0.411 ns`、TNS=0、hold WHS=`+0.169 ns`、THS=0；位流 SHA256=`b8c87ee10edf435617ab110cfdf0cf2a8d3c3ad3d3b91748c80ef04363305ec2`。

G1 MLP `gate_proj` 与 `up_proj` 真实双投影现已完成。独立工程 `mlp_gate_up_g1` 直接消费上述四组 `[896]` signed Q6.10 post-attention RMSNorm 输出，两路真实权重均为 `[4864,896]`、group size 64 的对称 signed INT4，且均无 bias；两路共享同一份逐向量对称 INT8 激活，combined scale 为 UQ4.28，输出为 signed int64 Q28。四组连贯输入共 8 个完整投影全部 `4864/4864` 上板逐位一致；新增测试 `6/6 PASS`，完整软件回归 `116/116 PASS`，软件随机/边界 `1000/1000 PASS`，真实板卡全零/极值/一般随机双路合计 `6/6 PASS`。PDS 所有角时序通过，慢角 setup WNS=`+0.916 ns`、TNS=0、hold WHS=`+0.157 ns`、THS=0；位流 SHA256=`e72959d2968a543bf3a2bcfd31f2b2c7a0d31a9888daba9ceac2d7c50cd5db6b`。

G1 MLP `SiLU(gate)` 现已完成。独立工程 `mlp_silu_g1` 直接消费四组已验证 gate projection `[4864]` signed int64 Q28 输出，执行对称 signed RNE 右移 18 位、signed int16 Q6.10 显式饱和和 E2 已验证 PWL64 SiLU；没有重算 gate/up，也没有提前执行乘法。四组 query/count=`0/1、1/2、5/6、15/16` 的完整结果全部 `4864/4864` 上板逐位一致；新增测试 `7/7 PASS`，完整软件回归 `123/123 PASS`，软件随机/边界 `1000/1000 PASS`，真实 FPGA 六批累计 `300/300 PASS`。PDS 所有角时序通过，慢角 setup WNS=`+1.468 ns`、TNS=0、hold WHS=`+0.169 ns`、THS=0；快角 hold WHS=`+0.100 ns`、THS=0；位流 SHA256=`87e643c65b70949297d54042921ac62e70454c018b6ff31f1386bbf2c8770550`。

G1 MLP `SiLU(gate) × up` 现已完成。独立工程 `mlp_silu_up_mul_g1` 直接消费已验证的 `[4864]` signed int16 Q6.10 SiLU 输出和 `[4864]` signed int64 Q28 up 输出；完整 signed 16×64 乘法保留 80-bit Q38，随后对绝对值执行 RNE 右移 10 位、恢复符号并显式饱和到 signed int64 Q28。四组连贯真实输入全部 `4864/4864` 上板逐位一致；新增测试 `7/7 PASS`，完整软件回归 `130/130 PASS`，软件随机/边界 `1000/1000 PASS`，同一固定 seed 的真实 FPGA 连续随机/边界 `100/100 PASS`。seed17/29 PDS `All Constraints Met`，慢角 setup WNS=`+0.511 ns`、hold WHS=`+0.141 ns`，快角 setup WNS=`+3.050 ns`、hold WHS=`+0.065 ns`，TNS/THS 全 0；验收位流 SHA256=`a83797a8b2ec75d030fc01144e6bf51e7de0ec930fc135c1a0aba89ebf1c4336`。

G1 MLP `down_proj` 现已完成。独立工程 `mlp_down_proj_g1` 直接消费上述 verified `[4864]` signed int64 Q28 乘法输出，读取真实 `model.layers.0.mlp.down_proj.weight=[896,4864]`，按逐向量对称 INT8、UQ4.28 combined scale、64 元素分组点积和 76 组 signed int64 Q28 精确累加输出 `[896]`；真实模型无 bias。四组 query/count=`0/1、1/2、5/6、15/16` 全部 `896/896` 上板逐位一致；新增测试 `7/7 PASS`，完整软件回归 `137/137 PASS`，软件随机/边界 `1000/1000 PASS`，真实 FPGA 全零、极值/饱和和 RNE tie `3/3 PASS`。PDS `All Constraints Met`，慢角 setup WNS=`+0.872 ns`、hold WHS=`+0.110 ns`，快角 setup WNS=`+3.026 ns`、hold WHS=`+0.015 ns`，TNS/THS 全 0；验收位流 SHA256=`f4d1013a287fc27003db88905f3c61e25620d213475039ddbb14900580c46757`。

G1 MLP 第二处残差与完整 MLP 现已完成。独立工程 `mlp_residual_g1` 严格使用完整 Attention 第一处残差后的 `[896]` signed Q6.10 hidden，而不是 `post_attention_layernorm` 输出；down 分支使用已验证 `[896]` signed int64 Q28。硬件执行 signed RNE `>>18`、第一次 int16 饱和、残差相加和第二次 int16 饱和。四组连贯真实固定输入全部 `896/896` 上板逐位一致；新增测试 `5/5 PASS`，完整软件回归 `142/142 PASS`，软件随机/边界 `1000/1000 PASS`，同一 seed 连续真实 FPGA index=`0..299` 累计 `300/300 PASS`。PDS `All Constraints Met`，慢角 setup WNS=`+0.727 ns`、hold WHS=`+0.169 ns`，快角 setup WNS=`+3.298 ns`、hold WHS=`+0.100 ns`，TNS/THS 全 0；验收位流 SHA256=`ddc424fae630fda5ab55acc8d2cb12d80b3f8cca1d5341f4a455ec0aa0a0e42b`。

G2 完整 layer0 Transformer Block 已完成独立验收。`model_tools/transformer_block_reference.py` 从同一组 block hidden state 连贯执行 input RMSNorm、Q/K/V、RoPE、KV Cache、Attention、O_proj、第一处残差、post RMSNorm、gate/up、SiLU、down_proj 和第二处残差；硬件使用统一 11 路 DDR3 仲裁、22 阶段 scheduler/controller 和七矩阵运行时量化。最新完整 `model_tools` 回归 `187/187 PASS`。最终 PDS 详细路由 162 轮后未布线网络为 0、hold 修复 6 轮，物理资源 `29086 LUT / 35053 FF / 52 DRM / 36 APM / 79 IO`；多角时序 `All Constraints Met`，慢角 100 MHz setup WNS=`+0.198 ns`、hold WHS=`+0.141 ns`，快角 setup WNS=`+2.640 ns`、hold WHS=`+0.067 ns`，TNS/THS 全 0，恢复、移除和最小脉宽均通过。验收位流 SHA256=`e4c3494152498583ae4a25540363fe3e828483fa7c0012a117e26e17fc557403`，仅通过 JTAG 下载 SRAM，DONE bit=1，未操作 Flash。四组固定真实 hidden 共 `72/72` 个中间/最终张量逐位 PASS；地址/窗口压力 `8/8 PASS`，含 query=16383 的 KV Cache 末端；交替极值、全正最大、全负最小三组 hidden 共 `54/54` 张量 PASS，并实际覆盖双向残差饱和。最终固件 `PANGU50K G2 BLOCK V1`，`block_error/protocol_error=0`。

## 当前唯一下一任务

```text
G2 完整 layer0 Block、H1 真实 24 层目录和 H2 1 GiB DDR3 方案均已冻结。H3 已建立真实 24 层
slot A 顺序事务：456 笔参数上传、23 次 1792 B hidden copy；UART 新增配置读回 `L` 与安全 DDR copy `M`，
G2 默认单层行为保持不变。真实 position0/count1 已连贯运行 layer0..23：layer0 的 18 张量与 G2
逐位一致，每层输入等于上一层输出，最终 hidden SHA256=e9708deff485…619a；完整回归 244/244 PASS。
H3 host 可用 `--check-reference` 逐层比对。独立顶层已通过 Compile/Synthesize/Device Map，映射资源
29741 LUT / 35225 FF，但综合 setup WNS=-0.312 ns，尚未时序收敛。下一步完成 H3 PnR、位流、
JTAG SRAM 和真实 layer0→23 逐层板级闭环；最终 RMSNorm、LM Head、logits 与文本生成仍未开始。
```

详细任务以 `PROJECT_ROADMAP.md` 为准。
