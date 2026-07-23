# 项目协作说明

本文件用于让后续 ChatGPT/Codex/开发者进入工程后立即知道项目状态和继续方向。

## 开始工作前必须阅读

1. `PROJECT_ROADMAP.md`：唯一权威的完整路线、当前阶段、任务状态和验收标准。
2. `PROJECT_PROGRESS_2026-07-23.md`：截至 2026-07-23 的真实硬件验证记录。
3. 正在修改子工程中的 `README.md`，例如 `ddr_mac16_integration/README.md`。

## 每次开发必须遵守

- 当前唯一优先任务以 `PROJECT_ROADMAP.md` 中“当前唯一下一任务”为准，不要跳过中间验证直接做完整模型。
- 每完成一个可验证里程碑，必须同步更新 `PROJECT_ROADMAP.md`：勾选任务、填写验证证据、修改“当前唯一下一任务”。
- 只有同时满足以下条件，任务才能标记为完成：
  1. Python/软件参考结果一致；
  2. PDS 编译、综合、布局布线成功；
  3. 多角时序全部通过，TNS=0；
  4. 真实开发板上板验证通过；
  5. 随机压力测试通过并记录轮数。
- 已验证工程和位流不得被无意覆盖。新阶段优先建立独立目录或保留可回退版本。
- JTAG 默认只下载 FPGA 易失性 SRAM，不写 Flash，除非用户明确要求。
- 不将模型大文件、位流、PDS 中间数据库和日志提交到 Git；它们由 `.gitignore` 管理。
- 保持 RTL、上位机协议、DDR3 地址布局和 Python 参考模型同步更新。
- 代码注释、提交说明和项目文档默认使用简体中文。

## 项目基本信息

- 工程根目录：`E:\50K\AI_LLM_FPGA\pangu_ai_accel`
- FPGA：Pango Logos `PGL50H-6IFBG484`
- 开发板：盘古 Logos 50K / MES50HP
- DDR3：32 位 Controller + PHY，完整 1 GiB 已验证
- 模型目标：Qwen2.5-0.5B + LoRA，权重已转换为约 251.63 MiB 的 INT4 文件
- 当前阶段：从长度 16 点积扩展为通用 packed INT4 GEMV `y=W×x`
