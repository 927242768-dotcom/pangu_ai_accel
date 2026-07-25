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

当前已完成 **G2 软件全链参考、运行时量化硬件与独立板级验收**；仍未宣称完整 Block 硬件完成：

- `model_tools/transformer_block_reference.py` 已从同一 hidden state 连贯执行完整 layer0 Block，四组固定最终输出与已验证 G1 第二处残差完全一致。
- 顶层契约显式包含 22 个计算阶段，其中四个为 `QKV/OPROJ/GATE_UP/DOWN` 运行时量化；加上 IDLE/DONE/ERROR 共 25 个状态 ID。
- 已冻结 28 个 scratch/查表区域、24 个 Linear 参数/scale 区、七个矩阵调用描述和 F3 KV Cache 地址复用规则。
- Q6.10/Q28 运行时量化子系统已经建立自动逐位闭环：同一验证位流核对全部 INT8、max metadata、全部 UQ4.28、14→16/76→80 padding，以及 source/raw-scale/activation/combined-scale 的 AXI 地址、命令数、beat 数和 burst。
- 七个真实矩阵 `q/k/v/o/gate/up/down` 已真实上板 `7/7 PASS`；Q6.10 随机/边界 `100/100 PASS`，Q28 随机/边界 `24/24 PASS`，seed=`20260819`。
- 最新完整 `model_tools` 回归 `165/165 PASS`；量化七矩阵软件固定清单 `7/7 PASS`，地址/burst/padding 事务压力 `1000/1000 PASS`。
- 量化验证工程已完整通过 PDS Compile、Synthesize、Device Map、PnR、Timing 和 Bitstream；慢/快角 setup TNS 与 hold THS 全部为 0，并已通过 JTAG SRAM 下载和真实板卡压力。
- 当前尚缺的是完整 Block 的统一 DDR3 仲裁、22 阶段 controller/top、完整中间张量与最终 `[896]` 输出的连贯板级闭环。

量化验证工程最终资源和时序：

| 项目 | 结果 |
|---|---|
| LUT / FF | 16370 / 13887 |
| DRM18K / APM / I/O | 40 / 8 / 79 |
| Slow `ddrphy_clkin` setup | WNS=`+0.187 ns`，TNS=0 |
| Slow `ddrphy_clkin` hold | WHS=`+0.171 ns`，THS=0 |
| Fast `ddrphy_clkin` setup | WNS=`+2.908 ns`，TNS=0 |
| Fast `ddrphy_clkin` hold | WHS=`+0.101 ns`，THS=0 |
| Timing 总结 | `All Constraints Met` |
| 位流 SHA256 | `220b771afbf8ea8d99806f3de27512748e2bd54913b1cc5e1f4a894647314236` |

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
model_tools/runtime_quantizer_validation.py
model_tools/runtime_quantizer_g2_reference.json
model_tools/test_transformer_block_reference.py
model_tools/test_runtime_linear_quant_reference.py
model_tools/test_runtime_quantizer_validation.py
tools/pangu_runtime_quantizer_host.py
```

验证命令：

```bat
python -m model_tools.transformer_block_reference summary
python -m model_tools.transformer_block_reference verify
python -m model_tools.transformer_block_reference --rounds 1 --seed 20260818 stress
python -m model_tools.runtime_quantizer_validation verify
python -m model_tools.runtime_quantizer_validation stress --rounds 1000 --seed 20260819
python -m unittest discover -s model_tools -p "test_*.py"
```

完整 Block 软件压力函数用于检查不同 hidden seed、query/window 和动态载荷的确定性。量化子系统已经额外完成七矩阵固定清单、1000 轮地址/burst/padding 软件压力，以及 Q6.10/Q28 真实 FPGA 随机/边界压力；这些结果不替代后续完整 Block 的板级验收。

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
rtl/q28_to_binary32_sequential.v
rtl/runtime_q28_activation_quantizer.v
rtl/runtime_fp16_scale_builder.v
rtl/runtime_activation_quantizer_ctrl.v
rtl/runtime_scale_builder_ctrl.v
rtl/runtime_quantizer_ctrl.v
rtl/runtime_quantizer_q28_top.v
rtl/runtime_quantizer_trace_checker.v
rtl/runtime_quantizer_validation_ctrl.v
rtl/runtime_quantizer_validation_top.v
rtl/transformer_block_contract.vh
pnr/build_shared_linear.tcl
pnr/build_runtime_linear.tcl
pnr/build_scheduler.tcl
pnr/build_q10_quantizer.tcl
pnr/build_q28_quantizer.tcl
pnr/build_scale_builder.tcl
pnr/build_runtime_quantizer.tcl
pnr/build_runtime_quantizer_q28.tcl
pnr/build_runtime_quantizer_validation_ctrl.tcl
pnr/build_runtime_quantizer_validation.tcl
pnr/program_runtime_quantizer_validation_sram.tcl
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
- `runtime_quantizer_trace_checker.v` 与验证顶层已经对七个真实矩阵逐命令检查 AXI 地址、burst、命令数和 beat 数，并回读全部逐位结果；
- 量化子系统已经完成独立 PnR、多角时序、位流、JTAG SRAM、七矩阵固定真实输入和 Q6.10/Q28 随机/边界板卡验证。

当前唯一实施点已经切换为完整 Block：连接统一 DDR3 仲裁、`transformer_block_ctrl.v`、`transformer_block_top.v` 和完整 host/PDS 工程，把 22 个阶段从同一组 hidden state 连贯执行到第二处残差，并逐位比较全部关键中间张量和最终 `[896]` 输出。

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


## 8. 运行时量化验收复现

构建：

```bat
cd /d E:\50K\AI_LLM_FPGA\pangu_ai_accel\transformer_block_g2\pnr
D:\Pango\PDS_2022.2-SP6.4\bin\pds_shell.exe ^
  -file build_runtime_quantizer_validation.tcl ^
  -project_name runtime_quantizer_validation_seed5_11
```

时序验收必须看到：

```text
Design Summary : All Constraints Met.
Slow ddrphy_clkin setup WNS = +0.187 ns, TNS = 0
Slow ddrphy_clkin hold  WHS = +0.171 ns, THS = 0
Fast ddrphy_clkin setup WNS = +2.908 ns, TNS = 0
Fast ddrphy_clkin hold  WHS = +0.101 ns, THS = 0
```

仅下载 SRAM：

```bat
D:\Pango\PDS_2022.2-SP6.4\bin\cdt_cfg_shell.exe ^
  -file program_runtime_quantizer_validation_sram.tcl ^
  -work_dir .
```

该脚本只包含 `cfg_connect/cfg_scan_chain/cfg_assign_file/cfg_program`，不包含 Flash 擦除或编程命令。验收记录为下载 100%、DONE bit=1。

软件与板卡命令：

```bat
python -m unittest discover -s model_tools -p "test_*.py"
python -m model_tools.runtime_quantizer_validation verify
python -m model_tools.runtime_quantizer_validation stress --rounds 1000 --seed 20260819
python tools\pangu_runtime_quantizer_host.py --port COM20 info
python tools\pangu_runtime_quantizer_host.py --port COM20 status
python tools\pangu_runtime_quantizer_host.py --port COM20 --timeout 300 all
python tools\pangu_runtime_quantizer_host.py --port COM20 --timeout 300 stress k_proj --rounds 100 --seed 20260819
python tools\pangu_runtime_quantizer_host.py --port COM20 --timeout 300 stress o_proj --rounds 24 --seed 20260819
```

最终结果：

- 软件回归 `165/165 PASS`；
- 七矩阵软件固定清单 `7/7 PASS`；
- 软件事务压力 `1000/1000 PASS`；
- 七矩阵真实板卡 `7/7 PASS`；
- Q6.10 板级压力 `100/100 PASS`；
- Q28 板级压力 `24/24 PASS`；
- 固件 `PANGU50K G2 QUANT V1`；
- 最终状态 `ddr_init_done=1、core_busy=0、trace_error=0、protocol_error=0`；
- 位流 `generate_bitstream/runtime_quantizer_validation_top.sbit`，SHA256=`220b771afbf8ea8d99806f3de27512748e2bd54913b1cc5e1f4a894647314236`。
