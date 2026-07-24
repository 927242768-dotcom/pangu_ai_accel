#!/usr/bin/env python3
"""``kv_cache_reference`` 的容量、地址、载荷和真实 K/V 测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

try:
    from .kv_cache_reference import (
        AXI_BEAT_BYTES,
        DDR_BYTES,
        DEFAULT_IMAGE,
        DEFAULT_MANIFEST,
        K_BEATS,
        KV_BASE_BYTES,
        KV_BASE_CTRL,
        KV_END_BYTES,
        KV_TOTAL_BYTES,
        LAYER_STRIDE_BYTES,
        LAYER_STRIDE_CTRL,
        MAX_CONTEXT,
        MAX_READ_TOKENS,
        NUM_LAYERS,
        RESERVED_LOW_BYTES,
        TOKEN_SLOT_BEATS,
        TOKEN_SLOT_BYTES,
        TOKEN_STRIDE_CTRL,
        VECTOR_BYTES,
        V_OFFSET_CTRL,
        KVCacheReferenceError,
        build_fixed_real_cases,
        build_token_payload,
        decode_token_payload,
        history_addresses,
        kv_slot_address,
        make_deterministic_token,
        software_stress,
        validate_layout_constants,
        validate_manifest,
        validate_read_range,
    )
except ImportError:
    from kv_cache_reference import (
        AXI_BEAT_BYTES,
        DDR_BYTES,
        DEFAULT_IMAGE,
        DEFAULT_MANIFEST,
        K_BEATS,
        KV_BASE_BYTES,
        KV_BASE_CTRL,
        KV_END_BYTES,
        KV_TOTAL_BYTES,
        LAYER_STRIDE_BYTES,
        LAYER_STRIDE_CTRL,
        MAX_CONTEXT,
        MAX_READ_TOKENS,
        NUM_LAYERS,
        RESERVED_LOW_BYTES,
        TOKEN_SLOT_BEATS,
        TOKEN_SLOT_BYTES,
        TOKEN_STRIDE_CTRL,
        VECTOR_BYTES,
        V_OFFSET_CTRL,
        KVCacheReferenceError,
        build_fixed_real_cases,
        build_token_payload,
        decode_token_payload,
        history_addresses,
        kv_slot_address,
        make_deterministic_token,
        software_stress,
        validate_layout_constants,
        validate_manifest,
        validate_read_range,
    )


class KVCacheReferenceTest(unittest.TestCase):
    def test_capacity_and_alignment(self) -> None:
        validate_layout_constants()
        self.assertEqual(RESERVED_LOW_BYTES, 128 << 20)
        self.assertEqual(KV_TOTAL_BYTES, 896 << 20)
        self.assertEqual(KV_END_BYTES, DDR_BYTES)
        self.assertEqual(NUM_LAYERS, 28)
        self.assertEqual(MAX_CONTEXT, 16384)
        self.assertEqual(VECTOR_BYTES, 1024)
        self.assertEqual(TOKEN_SLOT_BYTES, 2048)
        self.assertEqual(TOKEN_SLOT_BEATS, 64)
        self.assertEqual(K_BEATS, 32)
        self.assertEqual(LAYER_STRIDE_BYTES, 32 << 20)
        self.assertEqual(KV_BASE_BYTES % AXI_BEAT_BYTES, 0)

    def test_controller_address_formula(self) -> None:
        self.assertEqual(KV_BASE_CTRL, 0x02000000)
        self.assertEqual(LAYER_STRIDE_CTRL, 0x00800000)
        self.assertEqual(TOKEN_STRIDE_CTRL, 0x200)
        self.assertEqual(V_OFFSET_CTRL, 0x100)

        case = kv_slot_address(13, 2026)
        expected_k = 0x02000000 + 13 * 0x00800000 + 2026 * 0x200
        self.assertEqual(case.k_base_ctrl, expected_k)
        self.assertEqual(case.v_base_ctrl, expected_k + 0x100)
        self.assertEqual(case.slot_end_ctrl, expected_k + 0x200)

    def test_first_and_last_slot_boundaries(self) -> None:
        first = kv_slot_address(0, 0)
        last = kv_slot_address(NUM_LAYERS - 1, MAX_CONTEXT - 1)
        self.assertEqual(first.k_base_bytes, KV_BASE_BYTES)
        self.assertEqual(first.v_base_bytes, KV_BASE_BYTES + VECTOR_BYTES)
        self.assertEqual(last.slot_end_bytes, DDR_BYTES)

    def test_adjacent_token_and_layer_do_not_overlap(self) -> None:
        token0 = kv_slot_address(5, 100)
        token1 = kv_slot_address(5, 101)
        next_layer = kv_slot_address(6, 0)
        layer_last = kv_slot_address(5, MAX_CONTEXT - 1)
        self.assertEqual(token0.slot_end_bytes, token1.slot_base_bytes)
        self.assertEqual(layer_last.slot_end_bytes, next_layer.slot_base_bytes)

    def test_history_range_is_strictly_contiguous(self) -> None:
        addresses = history_addresses(27, MAX_CONTEXT - MAX_READ_TOKENS, MAX_READ_TOKENS)
        self.assertEqual(len(addresses), MAX_READ_TOKENS)
        for left, right in zip(addresses, addresses[1:]):
            self.assertEqual(left.slot_end_bytes, right.slot_base_bytes)

    def test_payload_roundtrip_full_64_bit_patterns(self) -> None:
        k, v = make_deterministic_token(20260801)
        payload = build_token_payload(k, v)
        self.assertEqual(len(payload), TOKEN_SLOT_BYTES)
        decoded_k, decoded_v = decode_token_payload(payload)
        np.testing.assert_array_equal(decoded_k, k)
        np.testing.assert_array_equal(decoded_v, v)

    def test_invalid_boundaries_raise(self) -> None:
        for layer in (-1, NUM_LAYERS):
            with self.assertRaises(KVCacheReferenceError):
                kv_slot_address(layer, 0)
        for position in (-1, MAX_CONTEXT):
            with self.assertRaises(KVCacheReferenceError):
                kv_slot_address(0, position)
        with self.assertRaises(KVCacheReferenceError):
            validate_read_range(MAX_CONTEXT - 1, 2)
        with self.assertRaises(KVCacheReferenceError):
            validate_read_range(0, MAX_READ_TOKENS + 1)

    def test_small_random_stress(self) -> None:
        software_stress(rounds=50, seed=20260801)

    @unittest.skipUnless(Path(DEFAULT_IMAGE).is_file(), "本地没有真实 .p50 镜像")
    def test_real_kv_manifest(self) -> None:
        cases = build_fixed_real_cases()
        self.assertEqual(len(cases), 4)
        for case in cases:
            self.assertEqual(case.k_q28.shape, (2, 64))
            self.assertEqual(case.v_q28.shape, (2, 64))
            self.assertEqual(len(case.payload), TOKEN_SLOT_BYTES)
        committed = validate_manifest(cases, DEFAULT_MANIFEST)
        self.assertEqual(committed["layout"]["max_context"], MAX_CONTEXT)
        self.assertEqual(committed["layout"]["kv_end_bytes"], DDR_BYTES)


if __name__ == "__main__":
    unittest.main()
