# `.p50` 模型格式解析与提取工具

本目录负责 Qwen2.5-0.5B + LoRA 的 `.p50` 镜像导出、结构校验和真实张量提取。

## 1. 当前真实镜像

```text
model_output/yanbo_qwen25_0.5b_int4.p50
大小：263,857,920 字节（251.63 MiB）
SHA256：f0c0a22886499715fe16832b88ac59bff48fea8f3069c247437726aca6f19e9d
```

镜像内共有 290 个张量：

- 169 个二维 `int4_groupwise_symmetric` 张量；
- 121 个连续 `float16` 张量；
- INT4/FP16 主数据合计约 235.68 MiB；
- FP16 分组 scale 合计约 14.72 MiB。

二维权重形状统计：

| 形状 `[rows, columns]` | 数量 | 典型用途 |
|---|---:|---|
| `[151936, 896]` | 1 | Token Embedding，且模型标记为 tied embedding |
| `[896, 896]` | 48 | Q 投影、O 投影 |
| `[128, 896]` | 48 | K 投影、V 投影 |
| `[4864, 896]` | 48 | Gate 投影、Up 投影 |
| `[896, 4864]` | 24 | Down 投影 |

## 2. 固定头格式

固定头使用小端结构：

```python
struct.Struct("<8sIIQQIIII")
```

| 字节偏移 | 类型 | 字段 | 当前值/含义 |
|---:|---|---|---|
| 0 | `char[8]` | magic | `P50Q4V1\0` |
| 8 | `uint32` | version | `1` |
| 12 | `uint32` | header_size | `4096` |
| 16 | `uint64` | metadata_size | 当前为 `63716` |
| 24 | `uint64` | data_offset | 当前为 `528384` |
| 32 | `uint32` | tensor_count | 当前为 `290` |
| 36 | `uint32` | group_size | 当前为 `64` |
| 40 | `uint32` | flags | bit0=LoRA 已合并，bit1=tied embedding |
| 44 | `uint32` | reserved | 必须为 `0` |

结构体本身为 48 字节，整个固定头区域保留 4096 字节。内嵌 JSON 从文件偏移 4096 开始，真实张量数据区按 4 KiB 对齐。

## 3. INT4 权重格式

真实模型二维权重统一使用：

```text
scheme      = symmetric_per_row_group
weight_bits = 4
group_size  = 64
range       = [-7, 7]
zero_point  = 0（对称量化，不保存独立 zero point）
scale_dtype = float16
```

存储规则：

1. 矩阵逻辑形状为 `[输出行, 输入列]`，按输出行 row-major 存放；
2. 每一行按 64 个输入列分组；
3. 输入列不足 64 的尾部补零到 64 的整数倍；
4. 每个字节保存两个有符号 INT4：低半字节对应较小列号，高半字节对应下一列；
5. INT4 按 4 位二补码解释，导出器实际限制到 `[-7, 7]`；
6. scale 按 `[row, group]` row-major 连续保存，每个 scale 为小端 FP16；
7. 反量化公式为 `weight_fp32 = int4_value * fp16_scale`。

二维 INT4 张量的长度可由目录完全推导：

```text
padded_columns = align_up(columns, group_size)
groups_per_row = padded_columns / group_size
data_nbytes     = rows * padded_columns / 2
scale_nbytes    = rows * groups_per_row * 2
```

## 4. FP16 张量格式

一维 bias、RMSNorm 权重等张量使用连续 C-order FP16：

```text
data_nbytes = product(shape) * 2
```

FP16 张量没有 scale、padded columns 或 groups 字段。

## 5. 工具说明

| 文件 | 作用 |
|---|---|
| `export_qwen25_fpga.py` | 合并 LoRA 并导出 `.p50` 与外部 JSON |
| `p50_format.py` | 轻量解析库；校验头、目录、长度、偏移并提取张量 |
| `p50_inspect.py` | 摘要、目录查看、全量校验、行/块提取 CLI |
| `verify_p50_image.py` | 在结构全量校验基础上，对照源模型抽样验证量化误差 |
| `test_p50_format.py` | 使用独立微型镜像验证解析、解包、反量化和错误检测 |

## 6. 常用命令

全量校验真实镜像与外部 JSON：

```bat
python model_tools\p50_inspect.py verify
```

查看摘要：

```bat
python model_tools\p50_inspect.py summary --check-metadata
```

列出某类张量：

```bat
python model_tools\p50_inspect.py list --contains self_attn.q_proj.weight
```

查看一个张量的形状、偏移和长度：

```bat
python model_tools\p50_inspect.py describe ^
  --tensor model.layers.0.self_attn.q_proj.weight
```

按张量名提取任意一行：

```bat
python model_tools\p50_inspect.py row ^
  --tensor model.layers.0.self_attn.q_proj.weight ^
  --row 0 ^
  --output extracted\layer0_q_row0.npz
```

提取任意二维块，允许跨越多个量化 group：

```bat
python model_tools\p50_inspect.py block ^
  --tensor model.layers.23.mlp.gate_proj.weight ^
  --row-start 1024 --row-count 2 ^
  --column-start 60 --column-count 12 ^
  --output extracted\layer23_gate_block.npz
```

INT4 提取结果的 NPZ 包含：

- `values`：FP32 反量化结果；
- `quantized`：有符号 INT8 容器中的原始 INT4 数值；
- `scales`：该块涉及的 FP16 scales；
- `scale_group_start`：`scales` 第一个 group 的全局编号；
- 张量名、存储类型和行列范围元数据。

FP16 提取结果主要包含原始 `values` 和范围元数据。

## 7. 验证结果

2026-07-23 在真实 251.63 MiB 镜像上完成：

- 固定头、版本、flags、reserved 和数据区边界检查通过；
- 290 个张量名称唯一；
- 所有 shape、padded columns、group 数、数据长度和 scale 长度均可正确推导；
- 所有数据偏移满足 4 KiB 对齐，所有 scale 偏移满足 64 字节对齐；
- 所有数据范围均在镜像内且互不重叠；
- 外部 JSON 与镜像内嵌 JSON 逐字段完全一致；
- INT4 行、跨 group 数据块和 FP16 行提取通过；
- 独立微型镜像单元测试 5/5 PASS；
- 对照原 BF16 模型和已合并 LoRA 的 4 组抽样反量化误差全部位于理论半 scale 舍入上限内。

此阶段只确认真实模型文件格式和软件提取能力，没有修改 FPGA GEMV RTL、PDS 工程或已验证位流。
