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

## 当前状态

已经真实上板完成从单点积、完整真实 Linear 层到 RMSNorm、元素级非线性、Embedding 和完整 Q/K/V 的多级计算闭环：

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

## 当前唯一下一任务

```text
进入 F2 RoPE：在 F1 已验证的 Q=[14,64]、K=[2,64] head-major Q28 输出基础上，
确认 rotary_dim、rope_theta、位置索引和偶奇维配对规则，建立真实 Q/K 软件参考与固定清单；
再新建独立 RoPE 定点工程，完成 sin/cos 表、位置递增、Q/K 旋转、格式转换、误差验证、
软件压力、PDS、多角时序以及真实上板固定和随机位置测试。不得覆盖已有验证工程和位流。
```

详细任务以 `PROJECT_ROADMAP.md` 为准。
