#!/usr/bin/env python3
"""Qwen2.5-0.5B F3 KV Cache 地址布局与软件金标准。

本模块定义 1 GiB DDR3 中 KV Cache 的唯一地址公式：

- 低端 128 MiB 保留给模型权重、激活和临时缓冲；
- 高端 896 MiB 用于 28 层 KV Cache；
- 每层支持 16384 个 token；
- 每 token 保存 K=[2,64] 与 V=[2,64]，元素为 signed int64 Q28；
- K 与 V 各 1024 B，token 槽固定 2048 B；
- 每层固定 32 MiB，28 层恰好占用 896 MiB并结束于 1 GiB 边界。

DDR3 控制器地址单位为 32 bit，因此字节地址除以 4 即控制器地址。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from .linear_quant_reference import DEFAULT_ACTIVATION_SEED
    from .p50_format import P50Image
    from .qkv_linear_reference import (
        DEFAULT_IMAGE,
        HEAD_DIM,
        KV_HEADS,
        build_qkv_cases,
        load_qkv_models,
        reshape_heads,
    )
    from .rope_fixed_reference import apply_rope_fixed_q28, generate_trig_row
except ImportError:
    from linear_quant_reference import DEFAULT_ACTIVATION_SEED
    from p50_format import P50Image
    from qkv_linear_reference import (
        DEFAULT_IMAGE,
        HEAD_DIM,
        KV_HEADS,
        build_qkv_cases,
        load_qkv_models,
        reshape_heads,
    )
    from rope_fixed_reference import apply_rope_fixed_q28, generate_trig_row

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).with_name("kv_cache_reference.json")

DDR_BYTES = 1 << 30
CTRL_WORD_BYTES = 4
AXI_BEAT_BYTES = 32
NUM_LAYERS = 28
MAX_CONTEXT = 16_384
MODEL_MAX_POSITION = 32_768
ELEMENT_BYTES = 8
KV_VALUES = KV_HEADS * HEAD_DIM
VECTOR_BYTES = KV_VALUES * ELEMENT_BYTES
TOKEN_SLOT_BYTES = VECTOR_BYTES * 2
TOKEN_SLOT_BEATS = TOKEN_SLOT_BYTES // AXI_BEAT_BYTES
K_BEATS = VECTOR_BYTES // AXI_BEAT_BYTES
V_BEATS = K_BEATS
RESERVED_LOW_BYTES = 128 << 20
KV_BASE_BYTES = RESERVED_LOW_BYTES
LAYER_STRIDE_BYTES = MAX_CONTEXT * TOKEN_SLOT_BYTES
KV_TOTAL_BYTES = NUM_LAYERS * LAYER_STRIDE_BYTES
KV_END_BYTES = KV_BASE_BYTES + KV_TOTAL_BYTES
MAX_READ_TOKENS = 16

KV_BASE_CTRL = KV_BASE_BYTES // CTRL_WORD_BYTES
LAYER_STRIDE_CTRL = LAYER_STRIDE_BYTES // CTRL_WORD_BYTES
TOKEN_STRIDE_CTRL = TOKEN_SLOT_BYTES // CTRL_WORD_BYTES
V_OFFSET_CTRL = VECTOR_BYTES // CTRL_WORD_BYTES
BEAT_STRIDE_CTRL = AXI_BEAT_BYTES // CTRL_WORD_BYTES

DEFAULT_FIXED_SLOTS = ((0, 0), (0, 1), (13, 2026), (27, MAX_CONTEXT - 1))
DEFAULT_STRESS_SEED = 20260801


class KVCacheReferenceError(ValueError):
    """表示 KV Cache 配置、地址或载荷不合法。"""


@dataclass(frozen=True)
class KVSlotAddress:
    """一个 layer/token 槽的字节地址与控制器地址。"""

    layer: int
    position: int
    slot_base_bytes: int
    k_base_bytes: int
    v_base_bytes: int
    slot_end_bytes: int
    slot_base_ctrl: int
    k_base_ctrl: int
    v_base_ctrl: int
    slot_end_ctrl: int


@dataclass(frozen=True)
class KVTokenCase:
    """一个待写入 KV Cache 的 K/V token 固定用例。"""

    layer: int
    position: int
    k_q28: np.ndarray
    v_q28: np.ndarray
    address: KVSlotAddress

    @property
    def payload(self) -> bytes:
        return build_token_payload(self.k_q28, self.v_q28)


def _require_shape(array: np.ndarray, shape: tuple[int, ...], label: str) -> None:
    if array.shape != shape:
        raise KVCacheReferenceError(f"{label} 形状错误：{array.shape}，预期 {shape}")


def validate_layout_constants() -> None:
    """检查布局常数能恰好覆盖 1 GiB，且全部满足 AXI 对齐。"""

    if KV_END_BYTES != DDR_BYTES:
        raise KVCacheReferenceError(
            f"KV 区域没有恰好结束于 1 GiB：0x{KV_END_BYTES:x} != 0x{DDR_BYTES:x}"
        )
    if KV_TOTAL_BYTES != 896 << 20:
        raise KVCacheReferenceError("KV Cache 总容量不是 896 MiB")
    for label, value in (
        ("KV_BASE_BYTES", KV_BASE_BYTES),
        ("LAYER_STRIDE_BYTES", LAYER_STRIDE_BYTES),
        ("TOKEN_SLOT_BYTES", TOKEN_SLOT_BYTES),
        ("VECTOR_BYTES", VECTOR_BYTES),
    ):
        if value % AXI_BEAT_BYTES:
            raise KVCacheReferenceError(f"{label} 未按 256 bit AXI beat 对齐")
    if TOKEN_SLOT_BEATS != 64 or K_BEATS != 32 or V_BEATS != 32:
        raise KVCacheReferenceError("K/V beat 数推导错误")


def validate_layer(layer: int) -> int:
    resolved = int(layer)
    if not 0 <= resolved < NUM_LAYERS:
        raise KVCacheReferenceError(
            f"layer 越界：{resolved}，有效范围 0..{NUM_LAYERS - 1}"
        )
    return resolved


def validate_position(position: int) -> int:
    resolved = int(position)
    if not 0 <= resolved < MAX_CONTEXT:
        raise KVCacheReferenceError(
            f"position 越界：{resolved}，硬件上下文范围 0..{MAX_CONTEXT - 1}"
        )
    return resolved


def validate_read_range(start_position: int, count: int) -> tuple[int, int]:
    start = validate_position(start_position)
    resolved_count = int(count)
    if not 1 <= resolved_count <= MAX_READ_TOKENS:
        raise KVCacheReferenceError(
            f"读取 token 数越界：{resolved_count}，有效范围 1..{MAX_READ_TOKENS}"
        )
    if start + resolved_count > MAX_CONTEXT:
        raise KVCacheReferenceError(
            f"读取范围越界：start={start}, count={resolved_count}, max={MAX_CONTEXT}"
        )
    return start, resolved_count


def kv_slot_address(layer: int, position: int) -> KVSlotAddress:
    """计算一个 layer/token 的 K、V 与槽边界地址。"""

    resolved_layer = validate_layer(layer)
    resolved_position = validate_position(position)
    slot_base = (
        KV_BASE_BYTES
        + resolved_layer * LAYER_STRIDE_BYTES
        + resolved_position * TOKEN_SLOT_BYTES
    )
    k_base = slot_base
    v_base = slot_base + VECTOR_BYTES
    slot_end = slot_base + TOKEN_SLOT_BYTES
    if not KV_BASE_BYTES <= k_base < v_base < slot_end <= DDR_BYTES:
        raise KVCacheReferenceError("KV 槽地址超出 DDR3 区域")
    return KVSlotAddress(
        layer=resolved_layer,
        position=resolved_position,
        slot_base_bytes=slot_base,
        k_base_bytes=k_base,
        v_base_bytes=v_base,
        slot_end_bytes=slot_end,
        slot_base_ctrl=slot_base // CTRL_WORD_BYTES,
        k_base_ctrl=k_base // CTRL_WORD_BYTES,
        v_base_ctrl=v_base // CTRL_WORD_BYTES,
        slot_end_ctrl=slot_end // CTRL_WORD_BYTES,
    )


def history_addresses(layer: int, start_position: int, count: int) -> list[KVSlotAddress]:
    start, resolved_count = validate_read_range(start_position, count)
    addresses = [kv_slot_address(layer, start + index) for index in range(resolved_count)]
    for previous, current in zip(addresses, addresses[1:]):
        if previous.slot_end_bytes != current.slot_base_bytes:
            raise KVCacheReferenceError("历史 token 地址不是严格连续布局")
    return addresses


def _normalize_vector(values: np.ndarray | Sequence[int], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim == 1:
        _require_shape(array, (KV_VALUES,), label)
        array = array.reshape(KV_HEADS, HEAD_DIM)
    _require_shape(array, (KV_HEADS, HEAD_DIM), label)
    return array


def build_token_payload(
    k_q28: np.ndarray | Sequence[int],
    v_q28: np.ndarray | Sequence[int],
) -> bytes:
    """生成固定 2048 B token 槽：K[2,64] 后接 V[2,64]。"""

    k = _normalize_vector(k_q28, "k_q28")
    v = _normalize_vector(v_q28, "v_q28")
    payload = (
        np.asarray(k, dtype="<i8").tobytes(order="C")
        + np.asarray(v, dtype="<i8").tobytes(order="C")
    )
    if len(payload) != TOKEN_SLOT_BYTES:
        raise KVCacheReferenceError(
            f"token 载荷长度错误：{len(payload)} != {TOKEN_SLOT_BYTES}"
        )
    return payload


def decode_token_payload(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    if len(payload) != TOKEN_SLOT_BYTES:
        raise KVCacheReferenceError(
            f"token 载荷长度错误：{len(payload)} != {TOKEN_SLOT_BYTES}"
        )
    values = np.frombuffer(payload, dtype="<i8").copy()
    k = values[:KV_VALUES].reshape(KV_HEADS, HEAD_DIM)
    v = values[KV_VALUES:].reshape(KV_HEADS, HEAD_DIM)
    return k, v


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(array: np.ndarray, dtype: str | np.dtype = "<i8") -> str:
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes(order="C")).hexdigest()


def make_deterministic_token(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """生成覆盖完整 64 位位型的确定性 K/V 测试载荷。"""

    rng = np.random.default_rng(int(seed))
    raw = rng.integers(0, 1 << 64, size=KV_VALUES * 2, dtype=np.uint64)
    signed = raw.view(np.int64)
    return (
        signed[:KV_VALUES].reshape(KV_HEADS, HEAD_DIM).copy(),
        signed[KV_VALUES:].reshape(KV_HEADS, HEAD_DIM).copy(),
    )


def load_real_layer0_kv(
    position: int,
    image_path: Path = DEFAULT_IMAGE,
    *,
    activation_seed: int = DEFAULT_ACTIVATION_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 F2 RoPE 后 K 与 F1 V 的真实 layer0 Q28 数据。"""

    if not 0 <= int(position) < MODEL_MAX_POSITION:
        raise KVCacheReferenceError(
            f"模型 RoPE position 越界：{position}，有效范围 0..{MODEL_MAX_POSITION - 1}"
        )
    image = P50Image(image_path)
    image.validate()
    cases = build_qkv_cases(load_qkv_models(image), activation_seed=activation_seed)
    k_before_rope = reshape_heads(cases["k"].expected_q28, cases["k"].spec).astype(np.int64)
    v_q28 = reshape_heads(cases["v"].expected_q28, cases["v"].spec).astype(np.int64)
    k_after_rope = apply_rope_fixed_q28(
        k_before_rope,
        generate_trig_row(int(position)),
        heads=KV_HEADS,
    )
    return k_after_rope, v_q28


def build_fixed_real_cases(
    slots: Iterable[tuple[int, int]] = DEFAULT_FIXED_SLOTS,
    *,
    image_path: Path = DEFAULT_IMAGE,
    activation_seed: int = DEFAULT_ACTIVATION_SEED,
) -> list[KVTokenCase]:
    """建立真实 K/V 固定用例；模型 Linear 只计算一次。"""

    resolved_slots = [(validate_layer(layer), validate_position(position)) for layer, position in slots]
    image = P50Image(image_path)
    image.validate()
    qkv_cases = build_qkv_cases(load_qkv_models(image), activation_seed=activation_seed)
    k_before_rope = reshape_heads(
        qkv_cases["k"].expected_q28, qkv_cases["k"].spec
    ).astype(np.int64)
    v_q28 = reshape_heads(
        qkv_cases["v"].expected_q28, qkv_cases["v"].spec
    ).astype(np.int64)
    by_position: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for _, position in resolved_slots:
        if position not in by_position:
            by_position[position] = (
                apply_rope_fixed_q28(
                    k_before_rope,
                    generate_trig_row(position),
                    heads=KV_HEADS,
                ),
                v_q28,
            )
    return [
        KVTokenCase(
            layer=layer,
            position=position,
            k_q28=by_position[position][0],
            v_q28=by_position[position][1],
            address=kv_slot_address(layer, position),
        )
        for layer, position in resolved_slots
    ]


def fixed_manifest(cases: Sequence[KVTokenCase]) -> dict[str, object]:
    validate_layout_constants()
    return {
        "format_version": 1,
        "layout": {
            "ddr_bytes": DDR_BYTES,
            "reserved_low_bytes": RESERVED_LOW_BYTES,
            "kv_base_bytes": KV_BASE_BYTES,
            "kv_base_ctrl": KV_BASE_CTRL,
            "num_layers": NUM_LAYERS,
            "max_context": MAX_CONTEXT,
            "model_max_position_embeddings": MODEL_MAX_POSITION,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "element_format": "signed int64 Q28",
            "vector_bytes": VECTOR_BYTES,
            "token_slot_bytes": TOKEN_SLOT_BYTES,
            "token_slot_beats": TOKEN_SLOT_BEATS,
            "layer_stride_bytes": LAYER_STRIDE_BYTES,
            "layer_stride_ctrl": LAYER_STRIDE_CTRL,
            "kv_total_bytes": KV_TOTAL_BYTES,
            "kv_end_bytes": KV_END_BYTES,
            "address_formula_ctrl": (
                "K=0x02000000 + layer*0x00800000 + position*0x00000200; "
                "V=K+0x00000100"
            ),
        },
        "cases": [
            {
                "layer": case.layer,
                "position": case.position,
                "address": {
                    "k_base_bytes": case.address.k_base_bytes,
                    "v_base_bytes": case.address.v_base_bytes,
                    "slot_end_bytes": case.address.slot_end_bytes,
                    "k_base_ctrl": case.address.k_base_ctrl,
                    "v_base_ctrl": case.address.v_base_ctrl,
                    "slot_end_ctrl": case.address.slot_end_ctrl,
                },
                "sha256": {
                    "k_q28": sha256_array(case.k_q28),
                    "v_q28": sha256_array(case.v_q28),
                    "token_payload": sha256_bytes(case.payload),
                },
                "preview": {
                    "k_head0_first8": case.k_q28[0, :8].tolist(),
                    "v_head0_first8": case.v_q28[0, :8].tolist(),
                },
            }
            for case in cases
        ],
    }


def validate_manifest(
    cases: Sequence[KVTokenCase],
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, object]:
    actual = fixed_manifest(cases)
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise KVCacheReferenceError(f"KV Cache 固定清单不一致：{manifest_path}")
    return expected


def software_stress(rounds: int = 1000, seed: int = DEFAULT_STRESS_SEED) -> None:
    """随机验证地址、不覆盖、连续读取与载荷往返。"""

    if rounds <= 0:
        raise KVCacheReferenceError("rounds 必须大于 0")
    validate_layout_constants()
    rng = np.random.default_rng(seed)
    virtual_cache: dict[tuple[int, int], bytes] = {}

    for round_index in range(rounds):
        layer = int(rng.integers(0, NUM_LAYERS))
        count = int(rng.integers(1, MAX_READ_TOKENS + 1))
        start = int(rng.integers(0, MAX_CONTEXT - count + 1))
        addresses = history_addresses(layer, start, count)

        for offset, address in enumerate(addresses):
            k, v = make_deterministic_token(seed ^ (round_index << 8) ^ offset)
            payload = build_token_payload(k, v)
            decoded_k, decoded_v = decode_token_payload(payload)
            if not np.array_equal(decoded_k, k) or not np.array_equal(decoded_v, v):
                raise KVCacheReferenceError("K/V 载荷往返不一致")
            if address.v_base_bytes - address.k_base_bytes != VECTOR_BYTES:
                raise KVCacheReferenceError("K/V 区域间距错误")
            if address.slot_end_bytes - address.v_base_bytes != VECTOR_BYTES:
                raise KVCacheReferenceError("V 区域长度错误")
            virtual_cache[(layer, start + offset)] = payload

        readback = b"".join(virtual_cache[(layer, start + offset)] for offset in range(count))
        expected = b"".join(
            virtual_cache[(layer, address.position)] for address in addresses
        )
        if readback != expected:
            raise KVCacheReferenceError("历史 K/V 顺序读取结果不一致")

    first = kv_slot_address(0, 0)
    last = kv_slot_address(NUM_LAYERS - 1, MAX_CONTEXT - 1)
    if first.k_base_bytes != KV_BASE_BYTES or last.slot_end_bytes != DDR_BYTES:
        raise KVCacheReferenceError("首尾边界地址错误")


def _print_summary(manifest: dict[str, object]) -> None:
    layout = manifest["layout"]
    print("KV Cache 地址布局：PASS")
    print(
        f"28 层 × {layout['max_context']} token × {TOKEN_SLOT_BYTES} B = "
        f"{KV_TOTAL_BYTES // (1 << 20)} MiB"
    )
    print(
        f"区域：0x{KV_BASE_BYTES:08X}..0x{KV_END_BYTES - 1:08X}，"
        f"低端保留 {RESERVED_LOW_BYTES // (1 << 20)} MiB"
    )
    print(layout["address_formula_ctrl"])
    for case in manifest["cases"]:
        print(
            f"layer={case['layer']}, position={case['position']}, "
            f"K_ctrl=0x{case['address']['k_base_ctrl']:07X}, "
            f"V_ctrl=0x{case['address']['v_base_ctrl']:07X}, "
            f"payload={case['sha256']['token_payload']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="F3 KV Cache 地址与软件参考")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest", help="输出真实固定清单 JSON")
    manifest.add_argument("--image", type=Path, default=DEFAULT_IMAGE)

    verify = sub.add_parser("verify", help="校验固定清单并运行随机压力")
    verify.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    verify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    verify.add_argument("--rounds", type=int, default=1000)
    verify.add_argument("--seed", type=int, default=DEFAULT_STRESS_SEED)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        cases = build_fixed_real_cases(image_path=args.image)
        manifest = fixed_manifest(cases)
        if args.command == "manifest":
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
            return 0
        validate_manifest(cases, args.manifest)
        software_stress(rounds=args.rounds, seed=args.seed)
        _print_summary(manifest)
        print(f"KV Cache 软件随机压力 PASS：{args.rounds}/{args.rounds}，seed={args.seed}")
        return 0
    except (FileNotFoundError, OSError, KVCacheReferenceError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
