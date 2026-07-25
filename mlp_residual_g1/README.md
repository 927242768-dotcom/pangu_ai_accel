# G1 layer0 MLP 第二处残差真实闭环

本目录是独立的 layer0 MLP 第二处残差验证工程，不覆盖此前已经验证的
`attention_residual_f6`、`post_attention_layernorm_g1`、`mlp_gate_up_g1`、
`mlp_silu_g1`、`mlp_silu_up_mul_g1` 或 `mlp_down_proj_g1` 工程与位流。

## 1. 本阶段边界

本阶段只完成：

```text
完整 Attention 第一处残差后的 hidden [896] signed Q6.10
+
真实 down_proj 输出 [896] signed int64 Q28

→ down Q28 对称 signed RNE 右移 18 位
→ 显式饱和到 signed int16 Q6.10
→ 与 residual hidden 符号扩展相加
→ 再次显式饱和到 signed int16 Q6.10
→ layer0 完整 MLP 输出 [896] signed Q6.10
```

residual 分支必须使用进入 `post_attention_layernorm` **之前**的 hidden，也就是完整
Attention 第一处残差后的输出。禁止错误使用 `post_attention_layernorm` 的归一化输出。

## 2. 数值规则

- residual hidden：`[896]` signed int16 Q6.10；
- down_proj：`[896]` signed int64 Q28；
- 重标定位移：`28 - 10 = 18`；
- 舍入：正负对称 round-to-nearest-even；
- `INT64_MIN`：使用无符号二补码幅值路径，避免有符号取反溢出；
- 第一次饱和：重标定结果显式限制到 `[-32768,32767]`；
- 残差相加：两路先符号扩展到 18 位；
- 第二次饱和：最终输出再次显式限制到 signed int16；
- 输出：`[896]` signed int16 Q6.10。

## 3. 软件参考与固定清单

相关文件：

```text
model_tools/mlp_residual_reference.py
model_tools/mlp_residual_g1_reference.json
model_tools/test_mlp_residual_reference.py
tools/pangu_mlp_residual_host.py
```

四组连贯固定 query/count 为：

```text
0/1
1/2
5/6
15/16
```

每组 residual hidden 的 SHA256 与 `attention_residual_f6` 最终输出完全相同，down 分支的
SHA256 与 `mlp_down_proj_g1` 最终输出完全相同。最终 MLP 输出 SHA256：

```text
query=0  count=1  630952eaf6fe179639773f2b60d1e9e990f380b0e698fa3d493dd7c279c96104
query=1  count=2  1cd96d92e43203abda26cace446c2664a0cbaa1fad29a8658a0d94941fbfbea7
query=5  count=6  b2365afdc2857c543628c9d15fd829005ede98aa99d35a8c4f973f61b35ed9dc
query=15 count=16 c164aab5251afc3954b8689a826a1241d1c7d6757adef5c5a232e127b59a4032
```

软件验证：

- 新增单元测试：5/5 PASS；
- 完整 `model_tools` 回归：142/142 PASS；
- 固定清单与 8960 B 上传载荷往返：PASS；
- 随机/边界软件压力：1000/1000 PASS，seed=`20260817`；
- 覆盖全零、正负 RNE half-way tie、`INT64_MIN/MAX`、Q10 饱和边缘、一般范围、
  第一次饱和和最终残差饱和。

常用命令：

```powershell
python -m unittest model_tools.test_mlp_residual_reference
python -m model_tools.mlp_residual_reference check
python -m model_tools.mlp_residual_reference stress --rounds 1000 --seed 20260817
python tools/pangu_mlp_residual_host.py selftest --rounds 1000 --seed 20260817
python -m unittest discover -s model_tools -p "test_*.py"
```

## 4. RTL 与 DDR3 布局

RTL：

```text
rtl/mlp_residual_core.v
rtl/mlp_residual_ctrl.v
rtl/mlp_residual_top.v
```

核心使用 1 个 hidden DRM 缓存和 4 个 down DRM bank。每个 256-bit 输出 beat 同时读取
16 个 hidden 元素和 16 个 down 元素，并逐 lane 执行幅值、RNE、两级饱和和结果打包。

DDR3 Controller 地址单位为 32 bit：

```text
residual hidden : 0x0000000，56 beats，1792 B
down_proj Q28   : 0x0001000，224 beats，7168 B
result Q6.10    : 0x0003000，56 beats，1792 B
```

## 5. UART 协议

115200 8N1：

```text
I -> PANGU50K MLP RESIDUAL V1\r\n
S -> S + flags + \r\n
L + 8960 B -> K\r\n
G -> R + 1792 B little-endian signed int16 Q6.10
```

状态 flags：

```text
bit0 DDR3 初始化完成
bit1 数据已加载
bit2 结果有效
bit3 核心忙
```

## 6. PDS 验证

构建：

```powershell
cd mlp_residual_g1\pnr
D:\Pango\PDS_2022.2-SP6.4\bin\pds_shell.exe -file build_mlp_residual.tcl -project_name mlp_residual
```

结果：

- Compile、Synthesize、Device Map、Place & Route、Report Timing、Generate Bitstream 全部成功；
- 详细布线 89 轮后未布线网络为 0；hold 修复 3 轮；
- `Design Summary : All Constraints Met`；
- 资源：7705 LUT、6868 FF、70 distributed RAM、20 DRM、0 APM、79 IO；
- 慢角 100 MHz core setup WNS=`+0.727 ns`、TNS=0；hold WHS=`+0.169 ns`、THS=0；
- 快角 core setup WNS=`+3.298 ns`、TNS=0；hold WHS=`+0.100 ns`、THS=0；
- Slow/Fast recovery、removal 和 minimum pulse width 无违例。

位流：

```text
pnr/generate_bitstream/mlp_residual_top.sbit
大小：2101696 B
SHA256：ddc424fae630fda5ab55acc8d2cb12d80b3f8cca1d5341f4a455ec0aa0a0e42b
```

## 7. 真实上板验证

只通过 JTAG 下载到 FPGA 易失性 SRAM：

- 识别 `PANGO USB CABLE II` 与 `PGL50H`；
- program 100%，`done bit=1`；
- 未执行任何 Flash 擦除或写入；
- 固件标识 `PANGU50K MLP RESIDUAL V1`；
- DDR3 初始化成功；
- 四组连贯真实固定输入全部 896/896 逐位一致，合计约 3.94 秒；
- 同一 seed=`20260817` 的连续随机/边界序列分三批完成：index `0..99`、`100..199`、
  `200..299`；累计 300/300 PASS；
- 三批耗时分别约 98.79、98.78、98.74 秒；
- 每组全部 896 项均与 Python 金标准逐位一致。

上板命令：

```powershell
cd mlp_residual_g1\pnr
D:\Pango\PDS_2022.2-SP6.4\bin\cdt_cfg_shell.exe -file program_sram.tcl -work_dir .
cd ..\..
python tools/pangu_mlp_residual_host.py --port COM20 info
python tools/pangu_mlp_residual_host.py --port COM20 fixed
python tools/pangu_mlp_residual_host.py --port COM20 stress --rounds 100 --seed 20260817 --start-index 0
python tools/pangu_mlp_residual_host.py --port COM20 stress --rounds 100 --seed 20260817 --start-index 100
python tools/pangu_mlp_residual_host.py --port COM20 stress --rounds 100 --seed 20260817 --start-index 200
```

## 8. 结论

G1 第二处残差和完整 layer0 MLP 软件链已经真实闭环。下一阶段只能进入 G2：建立独立的
完整 layer0 Transformer Block 集成参考、调度、PDS 工程和真实上板逐位验证；不得直接跳到
28 层完整模型或文本生成。
