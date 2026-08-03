# H3 真实 24 层换层与 hidden 交接基线

## 1. 目标

本目录把已经真实上板通过的 G2 单个完整 Transformer Block 扩展为阶段 H 的第一版分层执行基线：

```text
真实 layer0..23 参数按层换入 slot A
→ 配置并读回 cfg_layer/query/window/count
→ 执行一个完整 G2 Block
→ block_output_q10 通过 DDR3 内 copy 回到 block_hidden_q10
→ 换入下一层参数
```

第一版只验证正确的换层顺序、层号、KV 层地址和 1,792 B hidden 交接。当前不包含最终 RMSNorm、LM Head、logits 或采样，也不把 28 层硬件地址容量误当作真实模型层数；真实 P50 只有 24 层。

## 2. 当前状态

截至 2026-08-03，已完成 **H3 参数/事务契约、真实 24 层软件金标准、UART 协议、独立顶层和 PDS 前端映射**，但尚未完成 H3 PnR、时序、位流或真实板卡顺序执行：

- `build_layer_parameter_uploads(layer_index)` 可为真实 layer0..23 生成 19 笔 slot A 参数事务，每层 7,961,088 B；layer0 全部内容与 G2 已板测 resident 参数逐字节一致。
- `full_model_layer_sequence.py` 冻结 24 层顺序清单：456 笔参数上传、191,066,112 B，总计 23 次 `block_output_q10 -> block_hidden_q10` copy，每次 1,792 B/56 beats。
- G2 host/controller 默认仍为 `ACTIVE_LAYER_COUNT=1、ENABLE_DDR_COPY=0、FULL_MODEL_MODE=0`，不会改变已验收 layer0 固件行为。
- H3 顶层显式设置真实层数 24、启用 DDR copy，并返回固件标识 `PANGU50K H3 LAYER V1`。
- 新增 `L` 命令读回当前 `layer/query/window/count`；新增 `M` 命令执行 DDR3 内部 32 B beat copy，包含地址范围、对齐、长度和非重叠检查。
- H3 上位机按 `C -> L校验 -> 19笔W -> P -> G -> 可选M` 顺序运行；可执行单层或指定层范围 dry-run/板测。
- `full_model_24layer_reference.py` 已用真实 layer0..23 参数连贯执行 position=0/count=1；layer0 的 18 个关键张量与已验收 G2 query0 逐位完全一致，每层输入严格等于上一层输出。
- 冻结初始 hidden SHA256=`26139d5cacc3a2c2cf018016f370effd02e043b0d2155f89573463683fba80f0`，layer23 最终 hidden SHA256=`e9708deff4856b400fb953575288fdceab6bfef6a895f15739ac18b488f5619a`；24 层完整重算约 136 秒。
- H3 host 已支持 `--check-reference`，未来板测将逐层回读 1,792 B output 并与软件 SHA256 比较。
- 受影响专项 `38/38 PASS`，完整回归 `244/244 PASS`，24 层冻结清单显式重算 verify PASS。
- `transformer_block_host_ctrl` 独立 PDS Compile/Synthesize 成功。
- 独立顶层 `full_model_h3_top` 已通过 Compile、Synthesize 和 Device Map。

当前没有以下证据，因此不得把“层间状态机/微码调度器”或“从第 0 层运行到最后一层”勾选为完成：

- H3 完整 Place & Route、多角时序和未布线网络 0；
- H3 位流 SHA256 和 JTAG SRAM；
- `L/M` 命令真实板卡闭环；
- layer0→23 连续真实 FPGA 执行和每层输出 SHA256。

## 3. 软件事务契约

主要文件：

```text
model_tools/transformer_block_g2_payload.py
model_tools/test_full_model_layer_uploads.py
model_tools/full_model_layer_sequence.py
model_tools/full_model_layer_sequence_reference.json
model_tools/test_full_model_layer_sequence.py
model_tools/test_full_model_h3_protocol.py
model_tools/full_model_24layer_reference.py
model_tools/full_model_24layer_reference.json
model_tools/test_full_model_24layer_reference.py
tools/pangu_full_model_h3_host.py
```

冻结数据：

| 项目 | 数值 |
|---|---:|
| 真实模型层 | 24（layer0..23） |
| 每层参数事务 | 19 |
| 每层上传字节 | 7,961,088 B |
| 24 层参数事务 | 456 |
| 24 层上传字节 | 191,066,112 B |
| hidden copy 次数 | 23 |
| 单次 hidden copy | 1,792 B / 56 beats |
| copy 源 | byte `0x00034000` / controller `0x0000d000` |
| copy 目标 | byte/controller `0x00000000` |

公共 RMS rsqrt LUT、Softmax exp LUT 和 SiLU PWL 表上电后只上传一次，不计入每层 19 笔参数。

验证：

```bat
python -m model_tools.full_model_layer_sequence verify
python -m model_tools.full_model_24layer_reference verify
python -m unittest model_tools.test_full_model_layer_uploads ^
  model_tools.test_full_model_layer_sequence ^
  model_tools.test_full_model_h3_protocol ^
  model_tools.test_transformer_block_g2_integration
python tools\pangu_full_model_h3_host.py dry-run
```

## 4. UART 协议扩展

在 G2 `I/S/C/W/R/P/G` 之外新增：

```text
L
  返回：'L' + <layer:uint16> + <query:uint16> +
        <window:uint16> + <count:uint16> + CRLF

M + <src_controller_addr:uint32> +
    <dst_controller_addr:uint32> +
    <byte_length:uint32>
  返回：K CRLF 或 E + error + CRLF
```

`M` 只接受：

- 源/目标 controller 地址按 8 word，即 32 B 对齐；
- byte length 大于 0 且按 32 B 对齐；
- 源/目标范围不重叠；
- 首尾均不越过 1 GiB DDR3 Controller 地址空间。

## 5. H3 上位机

只查看计划，不访问板卡：

```bat
python tools\pangu_full_model_h3_host.py dry-run
```

板卡命令将在 H3 合法位流生成并仅通过 JTAG SRAM 下载后使用：

```bat
python tools\pangu_full_model_h3_host.py --port COM20 info
python tools\pangu_full_model_h3_host.py --port COM20 config
python tools\pangu_full_model_h3_host.py --port COM20 copy-hidden
python tools\pangu_full_model_h3_host.py --port COM20 run-layer 0 --read-output
python tools\pangu_full_model_h3_host.py --port COM20 run-sequence ^
  --start-layer 0 --end-layer 23 --query 0 --window 0 --count 1 ^
  --load-tables --prepare-query0 --check-reference
```

115200 UART 每层参数换入约 691 秒，24 层每 token 约 4.61 小时，只适合正确性验证。

## 6. PDS 前端结果

构建：

```bat
cd /d E:\50K\AI_LLM_FPGA\pangu_ai_accel\full_model_h3\pnr
set H3_FRONTEND_ONLY=1
D:\Pango\PDS_2022.2-SP6.4\bin\pds_shell.exe ^
  -file build_full_model_h3.tcl ^
  -project_name full_model_h3_frontend_direct
```

结果：

- Compile：PASS；
- Synthesize：PASS；
- Device Map：PASS；
- 映射资源：`29741 LUT / 35225 FF / 332 distributed RAM / 52 DRM / 36 APM / 79 IO`；
- 综合慢角 `ddrphy_clkin` setup WNS=`-0.312 ns`、TNS=`-79.872 ns`；
- 综合 hold、recovery、removal、minimum-pulse 无违例；
- 尚未执行完整 PnR、正式多角时序或 Bitstream。

初版外层 wrapper 曾导致原 DDR3 FDC 无法命中层级；最终顶层直接保留 G2 的 `I_ipsxb_ddr_top` 实例名和层级，现有约束无需改写并已成功完成 Device Map。

## 7. 当前唯一下一任务

修复 H3 综合 setup WNS=`-0.312 ns`、TNS=`-79.872 ns`，完成完整 PnR、多角时序、未布线网络 0、位流 SHA256 和仅 JTAG SRAM 下载；随后使用 `--check-reference` 验证 `L/M`、cfg_layer、KV 层号、hidden copy 和 layer0→23 每层 output SHA256。

在这些证据全部通过前，不进入最终 RMSNorm、LM Head、logits 或文本生成。
