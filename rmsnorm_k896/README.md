# layer0 input_layernorm K=896 定点 RMSNorm 闭环

本目录是 E1 RMSNorm 的独立验证工程，不覆盖任何已验证的 MAC16、GEMV 或完整 `q_proj` 工程与位流。

目标是在盘古 Logos `PGL50H-6IFBG484` 上完成：

```text
UART 上传真实 layer0 输入、gamma 和 rsqrt LUT
→ 写入 DDR3
→ FPGA 从 DDR3 读取并缓存
→ 计算 K=896 RMSNorm
→ 结果写回 DDR3
→ UART 回读 896 个输出
→ Python 逐元素比较
```

这仍是单算子闭环，不是完整 Qwen 推理。

## 1. 算子定义

真实模型张量：

```text
model.layers.0.input_layernorm.weight
shape = [896]
storage = float16, contiguous C-order
rms_norm_eps = 1e-6
```

Qwen2 RMSNorm 定义为：

```text
y_i = gamma_i * x_i * rsqrt(mean(x^2) + epsilon)
```

`gamma` 直接相乘，不采用 `1 + weight`。

## 2. 定点格式

| 数据 | 格式 | 说明 |
|---|---|---|
| 输入 `x` | signed Q6.10 int16 | 范围 `[-32, 31.9990234375]` |
| gamma | signed Q6.10 int16 | 真实 FP16 gamma 由主机 RNE 量化 |
| `sum(x^2)` | unsigned 40 bit | 保留 20 位小数，可覆盖 K=896 最坏情况 |
| mean/epsilon | unsigned Q12.20 | `epsilon=1e-6` 量化为 `1` |
| rsqrt | unsigned UQ12.20 uint32 | 归一化尾数 LUT 输出与指数校正 |
| 输出 | signed Q6.10 int16 | RNE 后显式饱和 |

所有浮点转整数、除法和右移都使用 round-to-nearest-even（RNE）。输出大于 `32767` 或小于 `-32768` 时饱和到 signed int16 边界。

## 3. rsqrt 方案选择

软件参考比较了两种方案：

1. `lut256_midpoint`：将方差归一化为 `m×2^p`，其中 `m∈[1,2)`；使用 256 项中点采样 `1/sqrt(m)` LUT，再完成指数缩放。
2. `lut32_newton1`：使用 32 项 LUT 产生初值，再执行一次 Newton-Raphson。

固定向量结果：

| 方案 | rsqrt Q20 | 相对精确量化值误差 | 输出最大误差 | LUT 存储 | 计算代价 |
|---|---:|---:|---:|---:|---|
| 精确量化参考 | 446735 | 0 | 0 | 无 | 软件基线 |
| LUT256 | 446797 | `1.3878e-4` | 1 个 Q10 LSB | 8192 bit | ROM + 指数校正 |
| LUT32 + NR1 | 446720 | 更低 | 1 个 Q10 LSB | 1024 bit | 额外乘法流水链 |

第一版硬件选择 LUT256：固定向量和随机压力中的最终输出误差足够小，并避免 Newton-Raphson 的额外乘法链与复杂流水。NR1 保留为后续资源优化候选。

## 4. RTL 结构

| 文件 | 作用 |
|---|---|
| `rtl/rmsnorm_k896_core.v` | K=896 平方和、RNE 除法、LUT rsqrt、gamma 乘法和输出饱和 |
| `rtl/rmsnorm_k896_ctrl.v` | UART 协议、DDR3 地址和 burst 调度、片上缓存装载、结果回写与回读 |
| `rtl/rmsnorm_k896_top.v` | 复用已验证的 32 位 DDR3 Controller + PHY、UART 和开发板约束 |
| `pnr/build_rmsnorm_k896.tcl` | PDS 编译、综合、Device Map、PNR、时序和位流生成 |
| `pnr/program_sram.tcl` | 仅通过 JTAG 下载易失性 SRAM，不操作 Flash |

核心采用串行、易验证的数据通路：

```text
56 拍输入缓存
→ 896 次平方和累加
→ 40 位除以 896，RNE 得到均值
→ 加 epsilon
→ 方差归一化和 LUT256 rsqrt
→ 896 次 x×rsqrt、RNE、gamma 乘法、RNE、饱和
→ 每 16 个 int16 组成 256 bit 结果拍
```

为满足 100 MHz 慢角时序，最终显式拆分了：

- rsqrt 常数乘法、RNE、动态指数移位和提交；
- 输入平方与 40 位累加；
- gamma 乘法后的 RNE、饱和和 256 位打包。

## 5. DDR3 地址布局

AXI 地址按 32 bit word 编址；每个 256 bit beat 地址增加 8。

| 区域 | AXI 基地址 | 大小 | 内容 |
|---|---:|---:|---|
| 输入 | `0x0000000` | 1792 B / 56 拍 | 896 个 little-endian signed Q6.10 int16 |
| gamma | `0x0001000` | 1792 B / 56 拍 | 896 个 little-endian signed Q6.10 int16 |
| LUT | `0x0002000` | 1024 B / 32 拍 | 256 个 little-endian UQ12.20 uint32 |
| 结果 | `0x0003000` | 1792 B / 56 拍 | 896 个 little-endian signed Q6.10 int16 |

一次固定上传载荷共 `4608 B`。

## 6. UART 协议

串口为 `115200 8N1`，当前常用端口为 `COM20`。

| 命令 | 主机发送 | FPGA 返回 |
|---|---|---|
| 信息 | `I` | `PANGU50K RMSNORM K896 V1\r\n` |
| 状态 | `S` | `S + flags + \r\n` |
| 加载 | `L + 4608 B` | `K\r\n` |
| 运行 | `G` | `R + 1792 B` |

状态位：

- bit0：DDR3 初始化完成；
- bit1：载荷已写入；
- bit2：结果有效；
- bit3：计算核心忙。

错误码：

- `0x01`：未知命令；
- `0x02`：DDR3 尚未初始化；
- `0x04`：尚未加载数据；
- `0xFF`：状态机异常。

## 7. 软件参考与测试

生成固定软件参考并更新 JSON 清单：

```bat
python model_tools\rmsnorm_fixed_reference.py ^
  --manifest model_tools\rmsnorm_layer0_reference.json
```

运行全部相关单元测试：

```bat
python -m unittest ^
  model_tools.test_p50_format ^
  model_tools.test_linear_quant_reference ^
  model_tools.test_rmsnorm_fixed_reference -v
```

RMSNorm 上位机 1000 组软件自检：

```bat
python tools\pangu_rmsnorm_k896_host.py selftest ^
  --rounds 1000 --seed 20260726
```

固定向量关键值：

```text
sum_squares      = 5176164753
mean_square_q20  = 5776970
variance_q20     = 5776971
lut_rsqrt_q20    = 446797
output first 16  = [20, -16, -38, -11, -71, 4, -65, -32,
                    140, -32, -36, 13, 43, -1, -71, 68]
```

关键 SHA256：

```text
activation Q6.10 : 673f017fc17e07910e55a4c0dabb131f0385a6e851cb3e53bc5b9b0ad81690ec
gamma Q6.10      : f850ab9bd97859f20e7850e4d414aebd18629dcc50a95daa51712cce340cb0ee
LUT256 UQ12.20   : 76ff1e6f79b00b125ca32deeaf5e62444eb06b6194c8815aea76b45e69e224c1
output Q6.10     : 1f52890780e0f4cc0f734d47a4e3bdb28c3c964b8734b442d7781d4ca155a4f0
upload payload   : f7db5667c1704db0305f01584e2d4a5fee8cd18a9e0006638d17be4cff384bc8
```

## 8. PDS 构建与 SRAM 下载

在 `rmsnorm_k896\pnr` 下运行：

```bat
D:\Pango\PDS_2022.2-SP6.4\bin\pds_shell.exe ^
  -file build_rmsnorm_k896.tcl -project_name rmsnorm_k896
```

仅下载 FPGA SRAM：

```bat
D:\Pango\PDS_2022.2-SP6.4\bin\cdt_cfg_shell.exe ^
  -file program_sram.tcl -work_dir .
```

`program_sram.tcl` 只调用 `cfg_program` 配置 FPGA SRAM，没有 Flash 擦除或写入命令。

## 9. 最终验证证据

2026-07-24 完成：

- 软件相关单元测试：23/23 PASS；
- RMSNorm 软件随机压力：1000/1000 PASS，seed=`20260726`；
- 固定向量 LUT256 与精确定点路径最大差值：1 个 Q10 LSB；
- 软件随机压力中 LUT256 与精确定点路径最大差值：2 个 Q10 LSB；
- PDS 编译、综合、Device Map、布局布线、时序和位流生成全部成功；
- 最终未布线网络：0；
- 多角时序：`All Constraints Met`；
- 慢角 100 MHz 建立 WNS=`+0.374 ns`、TNS=`0`，保持 WHS=`+0.171 ns`、THS=`0`；
- 快角建立 WNS=`+2.832 ns`、TNS=`0`，保持 WHS=`+0.100 ns`、THS=`0`；
- 恢复、移除和最小脉宽全部无违例；
- 资源：LUT=`8801`、FF=`7051`、DRM=`12`、APM=`9`；
- 位流：`rmsnorm_k896/pnr/generate_bitstream/rmsnorm_k896_top.sbit`；
- 位流 SHA256：`94c82d1ef6adf563043c6f90f5744ec258156d85c6db134389132ae4f2938b11`；
- JTAG SRAM 下载 100%，`done bit=1`，未操作 Flash；
- 固定真实上板：896 个输出与 Python LUT256 金标准逐位一致，端到端约 `0.61 s`；
- 真实随机上板：`300/300 PASS`，seed=`20260726..20261025`，约 `183.11 s`。

验收结论：E1 layer0 `input_layernorm` K=896 RMSNorm 已完成软件金标准、真实 DDR3 数据通路、定点计算、结果回写、逐元素比较、随机压力、PDS、多角时序和真实上板闭环。
