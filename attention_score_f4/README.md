# F4 Attention Score 独立验证工程

本目录只验证 `PROJECT_ROADMAP.md` 中的 F4：

- `Q · K` 64 维点积；
- `1 / sqrt(64) = 1/8` 缩放；
- causal mask；
- Qwen2.5-0.5B 的 `14Q -> 2KV` GQA 调度；
- 结果逐位回读。

本工程**不包含 Softmax、Attention Value 加权或 Transformer Block 调度**，不会覆盖 F3 KV Cache 工程和位流。

## 1. 定点定义

输入来自已验证的 F1/F2：

- Q：`[14,64]`，signed int64 Q28；
- K：`[2,64]`，signed int64 Q28；
- GQA：Q head `0..6` 使用 KV head 0，Q head `7..13` 使用 KV head 1；
- 点积：64 项 signed `Q28 × Q28`，精确累加为 Q56；
- 缩放与输出：对点积执行 signed RNE 右移 31 位，其中 28 位恢复 Q28，额外 3 位完成除以 8；
- 输出：signed int64 Q28；
- causal mask 和固定未使用槽：`INT64_MIN = 0x8000000000000000`。

第一版窗口上限为 16 token，固定结果布局为：

```text
scores[14][16]，head-major，共 224 个 int64 / 1792 B
```

## 2. DDR3 地址布局

低端临时区：

```text
Q      : ctrl 0x0000000，7168 B，224 个 256-bit beat
scores : ctrl 0x0000800，1792 B，56 个 256-bit beat
```

K 直接复用 F3 已验证布局：

```text
K = 0x02000000 + layer * 0x00800000 + position * 0x00000200
```

每个 K 为 `[2,64]` signed int64 Q28，共 1024 B / 32 beat。F4 不改写 V 区。

## 3. 文件

```text
attention_score_f4/
├── README.md
├── rtl/
│   ├── attention_score_core.v
│   ├── attention_score_ctrl.v
│   └── attention_score_top.v
└── pnr/
    ├── build_attention_score.tcl
    └── program_sram.tcl

model_tools/
├── attention_score_reference.py
├── attention_score_f4_reference.json
└── test_attention_score_reference.py

tools/
└── pangu_attention_score_host.py
```

## 4. 软件金标准

运行单元测试：

```bash
python -m unittest model_tools.test_attention_score_reference -v
```

运行固定真实用例和 1000 轮随机压力：

```bash
python model_tools/attention_score_reference.py verify \
  --rounds 1000 \
  --seed 20260802
```

也可从统一上位机入口运行：

```bash
python tools/pangu_attention_score_host.py selftest \
  --rounds 1000 \
  --seed 20260802
```

固定清单覆盖：

- layer 0，query 0，单 token；
- layer 0，query 1，连续 2 token；
- layer 13，query 2026，窗口 2023..2028，其中 2027/2028 必须被 mask；
- layer 27，query 16383，最后 16 token 边界窗口。

## 5. PDS 构建

在 `attention_score_f4/pnr` 目录运行：

```bash
D:/Pango/PDS_2022.2-SP6.4/bin/pds_shell.exe \
  -file build_attention_score.tcl \
  -project_name attention_score
```

验收必须同时满足：

- Compile、Synthesize、Device Map、Place & Route、Timing、Bitstream 全部完成；
- 0 unrouted nets；
- 所有分析角 TNS = 0、THS = 0；
- 位流独立生成于 `attention_score_f4/pnr/generate_bitstream/attention_score_top.sbit`。

## 6. SRAM 下载

只允许下载到易失性 SRAM：

```bash
D:/Pango/PDS_2022.2-SP6.4/bin/cdt_cfg_shell.exe \
  -file program_sram.tcl
```

`cfg_*` 命令属于 `cdt_cfg_shell.exe`，不是普通 `pds_shell.exe`。`program_sram.tcl` 不包含 Flash 擦除或编程命令。

## 7. UART 协议

115200、8N1：

```text
I
  -> "PANGU50K ATTN SCORE V1\r\n"

S
  -> 'S' + flags + layer + query_u16 + start_u16 + count + k_loaded + CRLF

C + layer_u8 + query_u16 + start_u16 + count_u8
  -> "K\r\n"

Q + 7168 B Q
  -> "K\r\n"

K + position_u16 + 1024 B K
  -> 'K' + position_u16 + CRLF

G
  -> 计算并写回固定 14x16 score
  -> "K\r\n"

R
  -> 'D' + layer + query_u16 + start_u16 + count_u8 + 1792 B scores
```

## 8. 真实板卡测试

列出串口并确认固件：

```bash
python tools/pangu_attention_score_host.py ports
python tools/pangu_attention_score_host.py --port COM20 info
python tools/pangu_attention_score_host.py --port COM20 status
```

固定真实用例：

```bash
python tools/pangu_attention_score_host.py --port COM20 fixed
```

随机窗口逐位回归：

```bash
python tools/pangu_attention_score_host.py --port COM20 stress \
  --windows 100 \
  --seed 20260802
```

只有固定用例、causal mask、随机回归、PDS 全流程和多角时序全部通过，才能在路线图中把 F4 标记为完成。
