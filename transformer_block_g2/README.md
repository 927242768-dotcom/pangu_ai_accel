# G2 layer0 完整 Transformer Block 集成工程

## 1. 目标

本目录用于把此前已经分别真实上板验证的 layer0 Attention 与 MLP 算子，整合为一个从同一组 block hidden state 出发的独立 Transformer Block 工程：

```text
block hidden [896] signed Q6.10
→ input RMSNorm
→ Q/K/V
→ Qwen2 split-half RoPE
→ KV Cache 写入与历史读取
→ Attention Score + causal mask
→ Softmax
→ probability × V + 多头拼接
→ O_proj
→ 第一处残差
→ post_attention_layernorm
→ gate_proj / up_proj
→ PWL64 SiLU(gate)
→ SiLU(gate) × up
→ down_proj
→ 第二处残差
→ block output [896] signed Q6.10
```

G2 必须新建独立调度、地址表、状态机和握手边界，不覆盖任何历史工程或已验证位流。

## 2. 当前状态

当前已完成 **G2.1 软件全链参考与集成契约**，并推进到 **G2.2 可综合硬件骨架**；仍未宣称完整 Block 硬件完成：

- `model_tools/transformer_block_reference.py` 已从同一 hidden state 连贯执行完整 layer0 Block。
- `model_tools/runtime_linear_quant_reference.py` 已把原主机侧 Q6.10/Q28→binary32→symmetric INT8、FP16 weight scale→UQ4.28 combined scale 改写为精确整数/二进制有理数规格；Q28 的 int64→binary64→binary32 双重 RNE 已对 10000 组随机值和 11 个关键边界逐位匹配 NumPy。
- 顶层现显式包含 22 个计算阶段，其中四个为 `QKV/OPROJ/GATE_UP/DOWN` 运行时量化；加上 IDLE/DONE/ERROR 共 25 个状态 ID。
- 已冻结 28 个 scratch/查表区域、24 个 Linear 参数/scale 区、七个矩阵调用描述和 F3 KV Cache 地址复用规则。
- 已建立 64 字节动态执行头、hidden、RoPE trig、历史 K/V 的上传载荷格式。
- 已建立四组真实固定 query/count=`0/1、1/2、5/6、15/16` 的关键中间张量 SHA256；最终 Block 输出与已验证 G1 第二处残差结果完全一致；最新固定清单 `4/4 PASS`、完整软件链压力 `1/1 PASS`、完整 `model_tools` 回归 `157/157 PASS`。
- 已实现并通过 PDS Compile/Synthesize 的共享 Linear、DDR3 Linear controller、22 阶段 scheduler、通用 RNE 除法器、Q6.10 激活量化器、Q28→binary32→INT8 激活量化器、FP16→UQ4.28 scale builder，以及 Q6.10/Q28 两类量化 DDR3 controller/beat adapter。
- 量化 controller 已能从 DDR3 解包 Q6.10/Q28、写回 packed INT8、读取 FP16 scale，并按 14→16 或 76→80 padding 写回 UQ4.28；当前仍缺 RTL/DDR3 数值逐位比较、独立 PnR/时序/板卡闭环、完整 DDR3 仲裁/顶层连接、完整位流和真实 Block 板卡结果。

独立子模块综合资源仅用于结构可行性检查，不等同完整 Block 资源或时序：

| RTL 单元 | LUT | FF | DRM18K | APM |
|---|---:|---:|---:|---:|
| shared Linear engine | 1557 | 3152 | 8 | 12 |
| runtime Linear DDR3 controller（含 engine） | 2377 | 4038 | 8 | 12 |
| 22 阶段 scheduler | 159 | 80 | 0 | 0 |
| Q6.10→INT8 activation quantizer | 513 | 290 | 8 | 0 |
| FP16→UQ4.28 scale builder | 1280 | 778 | 0 | 2 |
| Q28→binary32→INT8 activation quantizer | 3508 | 802 | 32 | 0 |
| Q6.10 runtime quantizer DDR3 controller | 3494 | 2773 | 8 | 2 |
| Q28 runtime quantizer DDR3 controller | 6500 | 3301 | 32 | 2 |

这些单元均无 PDS 硬错误，但宽除法器/可变移位在独立综合时存在 constant-probe 与 fanout 警告；当前只证明可综合，不代表数值已由 RTL 仿真验证，也不代表 100 MHz PnR 时序通过。

固定输出 SHA256：

| query/count | block output SHA256 |
|---|---|
| 0/1 | `630952eaf6fe179639773f2b60d1e9e990f380b0e698fa3d493dd7c279c96104` |
| 1/2 | `1cd96d92e43203abda26cace446c2664a0cbaa1fad29a8658a0d94941fbfbea7` |
| 5/6 | `b2365afdc2857c543628c9d15fd829005ede98aa99d35a8c4f973f61b35ed9dc` |
| 15/16 | `c164aab5251afc3954b8689a826a1241d1c7d6757adef5c5a232e127b59a4032` |

## 3. 软件参考与固定清单

主要文件：

```text
model_tools/transformer_block_reference.py
model_tools/runtime_linear_quant_reference.py
model_tools/transformer_block_g2_reference.json
model_tools/test_transformer_block_reference.py
model_tools/test_runtime_linear_quant_reference.py
```

验证命令：

```bat
python -m model_tools.transformer_block_reference summary
python -m model_tools.transformer_block_reference verify
python -m model_tools.transformer_block_reference --rounds 1 --seed 20260818 stress
python -m unittest model_tools.test_transformer_block_reference model_tools.test_runtime_linear_quant_reference
```

完整软件压力函数当前用于检查不同 hidden seed、query/window 和动态载荷的确定性。它不替代最终要求的 1000 轮软件压力与真实 FPGA 随机/边界压力。

## 4. DDR3 地址原则

- DDR3 Controller 地址单位仍为 32 bit。
- 低端 `0x00000000..0x07ffffff` 字节地址用于 layer0 参数、查表和 Block scratch。
- F3 KV Cache 继续从字节地址 `0x08000000` 开始，禁止改变既有公式。
- layer0 参数区当前最高结束字节地址为 `0x018a5400`，未越过低端 128 MiB。
- 共享 `linear_activation_int8` 区保存 FPGA 运行时产生的 INT8 激活；各矩阵 UQ4.28 combined scale 区由量化器动态重建。
- 七份原始 FP16 weight scale 常驻 DDR3，不能再假设主机提前知道中间激活 scale。
- 动态 scratch 与参数区均通过 Python 自动检查不重叠、4 字节对齐和边界；Python/Verilog 全地址和状态宏有单元测试逐项比对。

RTL 常量镜像：

```text
rtl/transformer_block_contract.vh
```

Python 文件是地址与阶段定义的权威来源；后续修改 RTL 常量时必须同步更新 Python 契约哈希和固定清单。

## 5. 顶层阶段与握手

阶段 ID 从 `INPUT_RMS=0x01` 顺序推进到 `RESIDUAL2=0x16`，`DONE=0x17`，`ERROR=0x1f`。四个运行时量化阶段分别为 `QKV_QUANT=0x02`、`OPROJ_QUANT=0x0b`、`GATE_UP_QUANT=0x0f`、`DOWN_QUANT=0x14`。

第一版握手约束：

- `start` 只允许在 idle 状态接受一个周期。
- `busy` 从接受 start 起持续到 done 或 error。
- `done` 只能在最终结果已经提交 DDR3 后产生一个周期。
- 子模块完成必须使用明确 done/valid 握手，禁止按固定周期猜测完成。
- 每个阶段必须有 watchdog；超时记录当前 stage ID 和 error code。
- error 为粘滞状态，复位前不允许重新启动。

## 6. 已有硬件骨架与唯一下一实施点

已存在：

```text
rtl/int4_unpack16.v
rtl/int8_dot16_pipe.v
rtl/shared_linear_engine.v
rtl/runtime_linear_ctrl.v
rtl/transformer_block_scheduler.v
rtl/unsigned_divider_rne.v
rtl/runtime_q10_activation_quantizer.v
rtl/q28_to_binary32.v
rtl/runtime_q28_activation_quantizer.v
rtl/runtime_fp16_scale_builder.v
rtl/runtime_activation_quantizer_ctrl.v
rtl/runtime_scale_builder_ctrl.v
rtl/runtime_quantizer_ctrl.v
rtl/runtime_quantizer_q28_top.v
rtl/transformer_block_contract.vh
pnr/build_shared_linear.tcl
pnr/build_runtime_linear.tcl
pnr/build_scheduler.tcl
pnr/build_q10_quantizer.tcl
pnr/build_q28_quantizer.tcl
pnr/build_scale_builder.tcl
pnr/build_runtime_quantizer.tcl
pnr/build_runtime_quantizer_q28.tcl
```

共享 Linear 数据路已覆盖：

- `q/k/v/o_proj/gate/up`：K=896、M=128/896/4864、14 groups；
- `down_proj`：K=4864、M=896、76 groups；
- packed signed INT4、UQ4.28 combined scale、signed int64 Q28、可选 bias；
- 运行时行数、group 数、activation/weight/scale/bias/result 地址切换；
- activation 一次加载、逐行参数读取、四个 int64 结果合并为一个 256-bit DDR3 写拍。

运行时量化算术核心和 DDR3 可调用 controller 已经建立：

- `runtime_activation_quantizer_ctrl.v` 从 DDR3 解包 Q6.10 或 Q28 源向量，驱动相应 activation quantizer，并把 32 个 INT8 打包为一个 256-bit beat 写入 `linear_activation_int8`；
- `runtime_scale_builder_ctrl.v` 连续读取原始 FP16 weight scale，按每行 14→16 或 76→80 插入 padding，把 UQ4.28 写入对应 combined-scale 区；
- `runtime_quantizer_ctrl.v` 顺序调度 activation 和 scale 两段，记录饱和计数及失败阶段；`runtime_quantizer_q28_top.v` 固化 Q28 参数分支用于独立综合；
- Q6.10/Q28 controller 均已重新通过 PDS Compile/Synthesize，无硬错误，但未做独立 PnR、多角时序或板卡验证。

当前唯一实施点是建立量化 RTL/DDR3 自动数值闭环：使用固定真实 QKV、O_proj、gate/up、down 输入逐项比较全部 INT8、max metadata、combined scale、padding word、burst 长度和目标地址。该闭环及独立 PnR/Timing/JTAG SRAM 通过后，才连接 `transformer_block_ctrl.v`、`transformer_block_top.v` 和完整 host/PDS 工程。

## 7. 完整 G2 验收标准

只有同时满足以下条件，才允许把 G2 标记为完成：

1. 同一 hidden state 的全部中间张量和最终输出与 Python 固定清单逐位一致。
2. 多组真实 hidden、随机和正负饱和边界通过自动比较。
3. 完整 RTL 在独立 PDS 工程中通过 Compile、Synthesize、Device Map、Place & Route、Timing 和 Bitstream。
4. 所有要求时钟角落 setup TNS=0，hold THS=0，恢复、移除和最小脉宽无违例。
5. 最终未布线网络为 0，并记录资源、WNS/WHS、位流路径和 SHA256。
6. 仅通过 JTAG 下载到 FPGA 易失性 SRAM，不擦写 Flash。
7. 完整 Block 固定用例与随机/边界真实板卡逐位通过。
8. G2 全部通过前，不进入 28 层调度、LM Head 或文本生成。
