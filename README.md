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

## 当前状态

已经真实上板完成三级计算闭环：

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

真实 Linear 已统一采用逐向量对称 INT8 激活和 UQ4.28 组合 scale。layer0 `q_proj` 的 M=4、K=896 固定切片已建立 P50 浮点、量化激活浮点和硬件等价 Q28 三条参考路径；定点最大绝对误差 `3.1277186e-6`，低于理论上界 `3.8200990e-5`。原有解析和新增量化测试共 13/13 PASS，另有 1000/1000 随机软件压力测试通过。本阶段仍未修改 FPGA RTL 或已验证位流。

## 当前唯一下一任务

```text
继续 D2：新建独立 FPGA 工程，实现真实模型的分组 UQ4.28 定点缩放。
首先复现 layer0 q_proj 的 M=4、K=896 固定向量：
每个 64 元素 group 进行 INT4×INT8 点积，乘主机预计算的 UQ4.28 combined scale，
在有符号 64 位 Q28 中跨组累加并加入 bias_q28，最终 4 个整数必须与软件参考逐位一致。
随后完成随机压力、PDS 全流程、多角时序和真实上板验证。
```

详细任务以 `PROJECT_ROADMAP.md` 为准。
