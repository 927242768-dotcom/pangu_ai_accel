# G1 layer0 MLP down_proj 独立真实上板闭环

## 1. 工程目标

本工程独立验证 Qwen2.5-0.5B layer0 MLP 的 `down_proj`，不覆盖任何已有
Attention、gate/up、SiLU 或 `SiLU(gate) × up` 工程和位流。

输入直接来自已经真实上板逐位通过的：

```text
SiLU(gate) × up: [4864] signed int64 Q28
```

真实权重：

```text
model.layers.0.mlp.down_proj.weight
shape = [896, 4864]
group_size = 64
groups_per_row = 76
storage = groupwise symmetric signed INT4
bias = absent
```

本阶段只输出 `[896]` signed int64 Q28，不执行 MLP 第二处残差，也不宣称完整
MLP/Transformer Block 已完成。

## 2. 冻结的数值定义

1. 将 verified `SiLU(gate) × up` signed int64 Q28 按实数转换为 float32。
2. 对完整 `[4864]` 向量执行逐向量对称 INT8 量化：
   `scale=max(abs(x))/127`，RNE，范围 `[-127,127]`，zero point=0；全零向量
   使用 scale=1 并严格量化为全零。
3. 主机计算 `activation_scale × weight_scale`，按 RNE 编码为 unsigned UQ4.28，
   超过 uint32 范围时显式饱和。
4. 每 64 元素执行 signed `INT8 × INT4` 点积，组内累加为 signed int32。
5. 每组点积乘 unsigned UQ4.28，并在 signed int64 Q28 中跨 76 组精确累加。
6. 真实 `down_proj` 不含 bias；上传的通用 bias 槽固定全零。

最坏边界证明：

```text
max_group_acc = 64 × 127 × 7 = 56896
max_output_magnitude = 76 × 56896 × 0xffffffff
                     = 18571850900440320
                     < 2^63 - 1
```

因此 76 组 Q28 累加不会溢出 signed int64，RTL 不允许隐含回绕或截断。

## 3. DDR3 载荷与地址布局

DDR3 控制器地址单位为 32 bit；一个 256 bit 数据拍占 8 个地址单位。

| 区域 | 控制器基地址 | 大小 | 说明 |
|---|---:|---:|---|
| activation | `0x0000000` | 4864 B / 152 beats | `[4864]` INT8 |
| weight | `0x0001000` | 2179072 B / 68096 beats | 896 行，每行 76 beats packed INT4 |
| scale | `0x0090000` | 286720 B / 8960 beats | 每行 76 个 uint32，补齐为 10 beats |
| bias | `0x00a4000` | 28672 B / 896 beats | 每行 1 beat，低 64 bit 有效且全零 |
| result | `0x00a8000` | 7168 B / 224 beats | `[896]` signed int64 Q28 |

完整上传载荷为 `2499328 B`。AXI `arlen` 最大 15，因此：

- 152 拍激活自动拆分为最多 16 拍的 burst；
- 每行 76 拍权重自动拆分为 `16+16+16+16+12`；
- 每行 10 拍 scale 使用单次 burst；
- 每 4 行结果组成一个 256 bit 数据拍立即写回。

## 4. UART 协议

串口参数：`115200, 8N1`。

| 命令 | 请求 | 回复 |
|---|---|---|
| `I` | 固件信息 | `PANGU50K MLP DOWN V1\r\n` |
| `S` | 状态查询 | `S + flags + \r\n` |
| `L` | 后跟固定 2499328 B 载荷 | `K\r\n` |
| `G` | 启动完整 down_proj | `R + 7168 B signed int64 Q28` |

状态字节：bit0 DDR3 初始化完成，bit1 已加载，bit2 结果有效，bit3 核心忙。

## 5. 主要文件

| 文件 | 作用 |
|---|---|
| `rtl/mlp_down_proj_core.v` | K=4864、76 groups 的单行 Q28 GEMV 核心 |
| `rtl/mlp_down_proj_ctrl.v` | UART、DDR3 上传、长 burst 分段、896 行调度、结果流式写回 |
| `rtl/mlp_down_proj_top.v` | PGL50H DDR3/时钟/UART/LED 顶层 |
| `pnr/build_mlp_down_proj.tcl` | PDS Compile 到 Bitstream 全流程 |
| `pnr/program_sram.tcl` | 仅写 FPGA 易失性 SRAM，不操作 Flash |
| `../model_tools/mlp_down_proj_reference.py` | 真实输入链、量化、Q28 金标准、载荷和压力测试 |
| `../model_tools/mlp_down_proj_g1_reference.json` | 四组真实固定输入/输出 SHA256 清单 |
| `../model_tools/test_mlp_down_proj_reference.py` | shape、边界、载荷、INT64 安全和真实来源测试 |
| `../tools/pangu_mlp_down_proj_host.py` | 软件自检、固定和随机/边界真实上板比较 |

## 6. 软件验证

```bat
python -m unittest model_tools.test_mlp_down_proj_reference -v
python model_tools\mlp_down_proj_reference.py check
python model_tools\mlp_down_proj_reference.py stress --rounds 1000 --seed 20260816
python tools\pangu_mlp_down_proj_host.py selftest --rounds 1000 --seed 20260816
python -m unittest discover -s model_tools -p "test_*.py"
```

最终结果：

- 新增单元测试：`7/7 PASS`；
- 完整 `model_tools` 回归：`137/137 PASS`；
- 软件随机/边界：`1000/1000 PASS`，seed=`20260816`；
- 固定真实 query/count=`0/1、1/2、5/6、15/16` 输出 SHA256：
  - `20ada87fb91b6f3a286d554eed7ede0d369e417162683bb4828f4ba2d0a45da3`
  - `05daecd0467d77bd1cf4f48be22caaece068cd844d263b46f88e016775deacec`
  - `2e8933ddb0423cf7f7c43d7165f82ce62c128607b883f80ca942d919740a0ccf`
  - `2dcea63a160554e624edd6f1c42e28a15f17a59e4999badfabdc8a7db80a82ee`

## 7. PDS 实现与多角时序

构建：

```bat
cd mlp_down_proj_g1\pnr
D:\Pango\PDS_2022.2-SP6.4\bin\pds_shell.exe -file build_mlp_down_proj.tcl -project_name mlp_down_proj
```

结果：

- Compile、Synthesize、Device Map、Place & Route、Timing、Bitstream 全部成功；
- 详细布线 153 轮后未布线网络为 0；hold 修复 6 轮完成；
- `Design Summary : All Constraints Met.`；
- 慢角 100 MHz core setup WNS=`+0.872 ns`、TNS=`0`；
- 慢角 core hold WHS=`+0.110 ns`、THS=`0`；
- 快角 core setup WNS=`+3.026 ns`、TNS=`0`；
- 快角 core hold WHS=`+0.015 ns`、THS=`0`；
- 恢复、移除和最小脉宽全部无违例；
- 资源：8915 LUT、9426 FF、70 distributed RAM、8 DRM、12 APM、79 IO；
- 位流：`pnr/generate_bitstream/mlp_down_proj_top.sbit`；
- 位流大小：`2101696 B`；
- SHA256：`f4d1013a287fc27003db88905f3c61e25620d213475039ddbb14900580c46757`。

## 8. 真实上板结果

仅下载易失性 SRAM：

```bat
cd mlp_down_proj_g1\pnr
D:\Pango\PDS_2022.2-SP6.4\bin\cdt_cfg_shell.exe -file program_sram.tcl -work_dir .
```

结果：

- JTAG 识别 `PANGO USB CABLE II` 与 `PGL50H`；
- SRAM 下载 100%，`done bit=1`，未操作 Flash；
- 串口 `COM20`，固件 `PANGU50K MLP DOWN V1`；
- DDR3 初始化完成；
- 四组真实固定输入全部 `896/896 PASS`，四个输出 SHA256 与固定清单一致；
- 每组 2499328 B 上传约 `216.78~216.82 s`；完整计算与 7168 B 回读约 `0.65 s`；
- 真实随机/边界 `3/3 PASS`，seed=`20260816`，global index=`0..2`：
  1. 全零输入；
  2. `INT16/INT64` 极值经完整 SiLU×up 饱和链形成的输入；
  3. 正负 RNE half-way tie 输入；
- 每轮全部 896 项均与 Python 金标准逐位一致。

## 9. 当前边界与下一任务

`down_proj` 已独立完整通过。当前仍未执行 MLP 第二处残差，也未完成完整 MLP。
下一任务只能是独立完成第二处残差：将 `down_proj` signed int64 Q28 按冻结的 signed
RNE/饱和规则重标定到 Q6.10，再与进入 `post_attention_layernorm` 之前的 Attention
子层 residual `[896]` signed Q6.10 相加并再次饱和；该残差单独通过前不得宣称完整
MLP 或完整 Transformer Block 完成。
