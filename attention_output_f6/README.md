# F6 Attention 输出加权和独立验证工程

本目录完成 `PROJECT_ROADMAP.md` 中 F6 的第一段闭环：

- 直接消费 F5 `[14,16]` unsigned UQ1.31 Softmax 概率；
- 直接复用 F3 DDR3 V Cache `[2,64]` signed int64 Q28；
- 按 Qwen2.5-0.5B 的 `14Q -> 2KV` GQA 映射执行概率 × V 加权和；
- 输出 `[14,64]` 多头结果并按 head-major 拼接为 `[896]`；
- Python、PDS、多角时序和真实板卡逐位闭环。

本工程**不包含 `O_proj`、Attention 残差、MLP 或 Transformer Block 调度**，不会覆盖 F5 或更早阶段的工程和位流。

## 1. 定点与布局定义

输入和输出定义如下：

- 概率：`[14,16]` head-major unsigned UQ1.31 uint32，`1.0 = 0x80000000`；
- V history：`[count,2,64]` token-major signed int64 Q28，`1 <= count <= 16`；
- GQA：Q head `0..6` 使用 KV head 0，Q head `7..13` 使用 KV head 1；
- 单项乘积：unsigned UQ1.31 × signed Q28，形成 signed Q59；
- 累加：每个输出在 signed 100 bit 中精确累加最多 16 项；
- 舍入：全部 token 累加结束后，仅执行一次 signed round-to-nearest-even 右移 31 位；
- 饱和：RNE 后显式饱和到 signed int64 Q28；
- 输出：`[14,64]` head-major，共 896 个 int64 / 7168 B；
- 拼接：`[14,64] -> [896]` 为 head-major 连续布局，可无损往返。

边界行为：

- 全 mask / 概率全 0：输出严格全 0；
- 单 token 且概率为 `0x80000000`：输出严格复制对应 GQA V；
- `count` 之外固定概率槽必须为 0；
- 满 16-token 窗口必须在一次 RNE 前完成全部累加；
- `INT64_MIN/MAX` V 输入按上述规则计算并显式饱和。

## 2. DDR3 地址布局

低端临时区：

```text
output        : ctrl 0x0000000，7168 B，224 个 256-bit beat
probabilities : ctrl 0x0000A00， 896 B， 28 个 256-bit beat
```

V 直接复用 F3 已验证布局：

```text
V = 0x02000000
  + layer    * 0x00800000
  + position * 0x00000200
  + 0x00000100
```

每个 V 为 `[2,64]` signed int64 Q28，共 1024 B / 32 beat。

## 3. 硬件结构

- `probability_mem`：32×256 bit，同步简单双口读写，映射为 8 个 DRM9K；
- `v_mem`：512×256 bit，最多缓存 16 token，映射为 8 个 DRM18K；
- 乘法器：概率拆 2 个 16-bit limb、V 绝对值拆 4 个 16-bit limb，8 个部分积顺序精确重构 signed 96-bit Q59；
- PDS 最终只使用 1 个 APM，避免形成 32×64 大组合乘法；
- 输出按 `(head, dimension)` 顺序流式写回 DDR3，每个 int64 使用 byte-enable 部分写入。

## 4. 文件

```text
attention_output_f6/
├── README.md
├── rtl/
│   ├── attention_output_core.v
│   ├── attention_output_ctrl.v
│   └── attention_output_top.v
└── pnr/
    ├── build_attention_output.tcl
    └── program_sram.tcl

model_tools/
├── attention_output_reference.py
├── attention_output_f6_reference.json
└── test_attention_output_reference.py

tools/
└── pangu_attention_output_host.py
```

## 5. 软件金标准

单元测试：

```bash
python -m unittest model_tools.test_attention_output_reference -v
```

固定真实清单与随机压力：

```bash
python model_tools/attention_output_reference.py verify \
  --rounds 1000 \
  --seed 20260804
```

统一上位机软件自检：

```bash
python tools/pangu_attention_output_host.py selftest \
  --rounds 1000 \
  --seed 20260804
```

四组真实固定窗口直接复用 F5 概率，并为每个 position 生成不同的真实 layer0 `v_proj` 输出：

- layer 0，window `0..0`，count=1；
- layer 0，window `0..1`，count=2；
- layer 13，window `2023..2028`，count=6；
- layer 27，window `16368..16383`，count=16。

固定 Attention 输出 SHA256：

```text
c5107911c0e6b9f1d9c471d7dde2c26d1192282abc964eb5005051aa5a4c9f71
14a6ee14736ed6132ea1357d3af27d55074492a727b196c33cd6905d4e1c9b02
86bc89cc77d49ac9451cc2a707510df310566997faf4c76fdfa95599043c248b
73e6b464b8c27e6a4d1066df5b5dade1c11c6a085a712d58dc54f6b598a0e407
```

验证结果：

- F6 新增单元测试 10/10 PASS；
- 完整 `model_tools` 回归 93/93 PASS；
- 软件随机压力 1000/1000 PASS，seed=`20260804`。

## 6. PDS 构建与时序

在 `attention_output_f6/pnr` 运行：

```bash
D:/Pango/PDS_2022.2-SP6.4/bin/pds_shell.exe \
  -file build_attention_output.tcl \
  -project_name attention_output
```

最终结果：

- Compile、Synthesize、Device Map、Place & Route、Timing、Bitstream 全部完成；
- 详细路由 155 轮后 0 unrouted nets；
- 资源：9184 LUT、11357 FF、70 distributed RAM、12 DRM、1 APM；
- `Design Summary : All Constraints Met`；
- 慢角：setup WNS=`+0.825 ns`、TNS=0；hold WHS=`+0.112 ns`、THS=0；
- 快角：setup WNS=`+3.349 ns`、TNS=0；hold WHS=`+0.032 ns`、THS=0；
- 慢/快角 Recovery、Removal 均无违例；
- 位流：`attention_output_f6/pnr/generate_bitstream/attention_output_top.sbit`；
- 位流大小：2101696 B；
- 位流 SHA256：`d7e64c58b73f8ca93f7a7dd981feabe5cc48f9b43e6b2ff0d8f60155886f36a3`。

## 7. SRAM 下载

只允许下载到易失性 SRAM：

```bash
D:/Pango/PDS_2022.2-SP6.4/bin/cdt_cfg_shell.exe \
  -file program_sram.tcl \
  -work_dir .
```

`cfg_*` 命令属于 `cdt_cfg_shell.exe`，不是 `pds_shell.exe`。本脚本不包含任何 Flash 擦除或编程命令。实际下载达到 100%，DONE bit=1。

## 8. UART 协议

115200、8N1：

```text
I
  -> "PANGU50K ATTN OUTPUT V1\r\n"

S
  -> 'S' + flags + layer + start_u16 + count + v_loaded + CRLF

C + layer_u8 + start_u16 + count_u8
  -> "K\r\n"

P + 896 B probabilities
  -> "K\r\n"

V + position_u16 + 1024 B V
  -> 'K' + position_u16 + CRLF

G
  -> 计算并写回 [14,64] / [896] Attention 输出
  -> "K\r\n"

R
  -> 'D' + layer + start_u16 + count_u8 + 7168 B output
```

## 9. 真实板卡验证

```bash
python tools/pangu_attention_output_host.py ports
python tools/pangu_attention_output_host.py --port COM20 info
python tools/pangu_attention_output_host.py --port COM20 status
python tools/pangu_attention_output_host.py --port COM20 fixed
python tools/pangu_attention_output_host.py --port COM20 stress \
  --windows 100 \
  --seed 20260804
```

真实结果：

- 固件：`PANGU50K ATTN OUTPUT V1`；
- DDR3 初始化成功；
- 四组真实固定窗口全部逐位一致，SHA256 与软件清单完全相同；
- 全 mask / 全零概率输出严格全 0；
- 单 token 1.0 概率、14Q/2KV GQA 与 `INT64_MIN/MAX` V 精确复制通过；
- 16-token 全 1.0 概率驱动 Q59 宽累加，INT64 正/负双向显式饱和逐位通过；
- 随机层、随机起点、count 1..16、随机概率/V 回归 100/100 PASS；
- 每 10 轮加入一次全 64-bit V 随机模式；
- seed=`20260804`，耗时 182.09 秒。

至此，F6 的“Softmax 权重与 V 加权和”和“多头拼接”已完成。下一任务是独立实现真实 layer0 `o_proj=[896,896]`，然后再接 Attention 残差；不得提前进入 MLP。
