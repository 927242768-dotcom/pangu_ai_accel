# PGL50H AI/LLM FPGA 加速项目

目标是在盘古 Logos `PGL50H-6IFBG484` 上逐步完成 Qwen2.5-0.5B + LoRA 的 INT4 推理。

## 进入项目后先读

1. [`PROJECT_ROADMAP.md`](PROJECT_ROADMAP.md)：完整开发路线、当前任务和验收标准。
2. [`PROJECT_PROGRESS_2026-07-23.md`](PROJECT_PROGRESS_2026-07-23.md)：已完成能力和真实上板证据。
3. [`ddr_mac16_integration/README.md`](ddr_mac16_integration/README.md)：当前 DDR3 + MAC16 + INT4 集成工程。

## 当前状态

已经真实上板完成：

```text
DDR3写入
→ 2拍×256位AXI burst读取
→ INT8或packed INT4权重处理
→ MAC16点积
→ 结果写回DDR3
→ UART返回
→ Python自动比较
```

INT8 和 INT4 路径分别通过 1000 轮随机测试，最终 PDS 多角时序全部满足。

## 当前唯一下一任务

```text
实现 M=4、K=64 的 packed INT4 GEMV：y=W×x。
激活只读取一次，每行分4个MAC16块累加，输出4个INT32结果，
与Python逐元素比较，并完成时序及真实上板1000轮压力测试。
```

详细任务以 `PROJECT_ROADMAP.md` 为准。
