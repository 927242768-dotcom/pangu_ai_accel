#!/usr/bin/env python3
"""阶段 H3 真实 24 层换层、参数转换与 hidden 交接测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

try:
    from .full_model_layer_sequence import (
        ACTIVE_LAYER_COUNT,
        COMMON_RUNTIME_NAMES,
        FullModelLayerSequenceError,
        build_common_runtime_uploads,
        build_layer_sequence_manifest,
        build_layer_uploads,
        hidden_copy_contract,
        load_reference,
        reference_snapshot,
        validate_execution_window,
    )
    from .full_model_memory_plan import (
        build_full_model_memory_plan,
        expand_layer_transfer_plan,
    )
    from .linear_quant_reference import quantize_signed_q28
    from .p50_format import P50Image
    from .rmsnorm_fixed_reference import quantize_gamma_q6_10
    from .transformer_block_g2_payload import build_resident_uploads
except ImportError:
    from full_model_layer_sequence import (
        ACTIVE_LAYER_COUNT,
        COMMON_RUNTIME_NAMES,
        FullModelLayerSequenceError,
        build_common_runtime_uploads,
        build_layer_sequence_manifest,
        build_layer_uploads,
        hidden_copy_contract,
        load_reference,
        reference_snapshot,
        validate_execution_window,
    )
    from full_model_memory_plan import (
        build_full_model_memory_plan,
        expand_layer_transfer_plan,
    )
    from linear_quant_reference import quantize_signed_q28
    from p50_format import P50Image
    from rmsnorm_fixed_reference import quantize_gamma_q6_10
    from transformer_block_g2_payload import build_resident_uploads


IMAGE = Path("model_output/yanbo_qwen25_0.5b_int4.p50")


class FullModelLayerSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = build_full_model_memory_plan()
        cls.image = P50Image(IMAGE)

    def test_common_runtime_uploads_are_only_three_shared_tables(self) -> None:
        uploads = build_common_runtime_uploads()
        self.assertEqual([item.name for item in uploads], list(COMMON_RUNTIME_NAMES))
        self.assertEqual(len(uploads), 3)
        self.assertTrue(all(item.persistent for item in uploads))
        resident = {item.name: item for item in build_resident_uploads()}
        for upload in uploads:
            self.assertEqual(upload.payload, resident[upload.name].payload)
            self.assertEqual(
                upload.controller_address,
                resident[upload.name].controller_address,
            )

    def test_layer0_uploads_are_bit_exact_with_verified_g2_resident_payloads(self) -> None:
        uploads = build_layer_uploads(0)
        self.assertEqual(len(uploads), 19)
        resident = {
            item.name: item
            for item in build_resident_uploads()
            if item.name not in COMMON_RUNTIME_NAMES
        }
        self.assertEqual(len(resident), 19)
        for upload in uploads:
            name = upload.name.removeprefix("layer0_")
            self.assertIn(name, resident)
            self.assertEqual(upload.controller_address, resident[name].controller_address)
            self.assertEqual(upload.payload, resident[name].payload)

    def test_all_24_layers_have_19_aligned_transactions(self) -> None:
        hashes: list[tuple[str, ...]] = []
        for layer_index in range(ACTIVE_LAYER_COUNT):
            uploads = build_layer_uploads(layer_index, plan=self.plan)
            self.assertEqual(len(uploads), 19)
            self.assertEqual(sum(len(item.payload) for item in uploads), 7_961_088)
            self.assertTrue(all(len(item.payload) % 32 == 0 for item in uploads))
            self.assertTrue(all(item.controller_address % 8 == 0 for item in uploads))
            hashes.append(tuple(item.sha256 for item in uploads))
        self.assertEqual(len(set(hashes)), ACTIVE_LAYER_COUNT)
        with self.assertRaises(FullModelLayerSequenceError):
            build_layer_uploads(24, plan=self.plan)

    def test_direct_copy_transactions_equal_real_p50_ranges(self) -> None:
        layer_index = 23
        transfers = expand_layer_transfer_plan(self.plan, layer_index, slot="A")
        uploads = build_layer_uploads(layer_index, plan=self.plan)
        with IMAGE.open("rb") as handle:
            for transfer, upload in zip(transfers, uploads):
                if transfer["transform"] not in {"copy_int4", "copy_fp16_scale"}:
                    continue
                handle.seek(transfer["source_byte_offset"])
                expected = handle.read(transfer["source_nbytes"])
                self.assertEqual(upload.payload, expected)

    def test_gamma_and_bias_transforms_match_independent_reference(self) -> None:
        layer_index = 23
        transfers = expand_layer_transfer_plan(self.plan, layer_index, slot="A")
        uploads = build_layer_uploads(layer_index, plan=self.plan)
        by_region = {
            transfer["destination_region"]: (transfer, upload)
            for transfer, upload in zip(transfers, uploads)
        }
        with IMAGE.open("rb") as handle:
            for name in ("input_rms_gamma_q10", "post_rms_gamma_q10"):
                transfer, upload = by_region[name]
                handle.seek(transfer["source_byte_offset"])
                raw = handle.read(transfer["source_nbytes"])
                values = np.frombuffer(raw, dtype="<f2").astype(np.float32)
                expected = quantize_gamma_q6_10(values)
                self.assertEqual(expected.clipped_count, 0)
                self.assertEqual(
                    upload.payload,
                    np.asarray(expected.quantized, dtype="<i2").tobytes(order="C"),
                )
            for name, rows in (
                ("q_bias_q28", 896),
                ("k_bias_q28", 128),
                ("v_bias_q28", 128),
            ):
                transfer, upload = by_region[name]
                handle.seek(transfer["source_byte_offset"])
                raw = handle.read(transfer["source_nbytes"])
                values = np.frombuffer(raw, dtype="<f2").astype(np.float32)
                expected, saturated = quantize_signed_q28(values)
                self.assertEqual(saturated, 0)
                actual = np.frombuffer(upload.payload, dtype="<i8").reshape(rows, 4)
                np.testing.assert_array_equal(actual[:, 0], expected)
                np.testing.assert_array_equal(actual[:, 1:], 0)

    def test_sequence_baseline_rejects_slot_b_prefetch(self) -> None:
        uploads = build_layer_uploads(7, slot="A", plan=self.plan)
        self.assertEqual(len(uploads), 19)
        with self.assertRaises(FullModelLayerSequenceError):
            build_layer_uploads(7, slot="B", plan=self.plan)

    def test_hidden_copy_is_56_nonoverlapping_beats(self) -> None:
        copy = hidden_copy_contract(self.plan)
        self.assertEqual(copy["command"], "M")
        self.assertEqual(copy["source_byte_address"], 0x0003_4000)
        self.assertEqual(copy["source_controller_address"], 0x0000_D000)
        self.assertEqual(copy["destination_byte_address"], 0)
        self.assertEqual(copy["destination_controller_address"], 0)
        self.assertEqual(copy["length"], 1792)
        self.assertEqual(copy["beats"], 56)

    def test_full_sequence_orders_real_layers_and_copies_between_them(self) -> None:
        manifest = build_layer_sequence_manifest()
        self.assertEqual(manifest["active_model_layers"], 24)
        self.assertEqual(manifest["layer_count"], 24)
        self.assertEqual(manifest["total_upload_transactions"], 456)
        self.assertEqual(manifest["total_upload_bytes"], 191_066_112)
        self.assertEqual(manifest["hidden_copy_count"], 23)
        self.assertEqual(manifest["hidden_copy_total_bytes"], 41_216)
        self.assertEqual(
            [item["layer_index"] for item in manifest["layers"]],
            list(range(24)),
        )
        self.assertTrue(all(item["commit_required"] for item in manifest["layers"]))
        self.assertTrue(all(item["execute_required"] for item in manifest["layers"]))
        self.assertTrue(all(item["copy_output_to_input"] for item in manifest["layers"][:-1]))
        self.assertFalse(manifest["layers"][-1]["copy_output_to_input"])
        self.assertEqual(
            manifest["not_included"],
            ["final_rmsnorm", "lm_head", "logits", "sampling"],
        )

    def test_window_and_layer_boundaries_are_rejected(self) -> None:
        validate_execution_window(0, 0, 1)
        validate_execution_window(16_383, 16_368, 16)
        for args in ((0, 0, 0), (16, 0, 17), (16_384, 16_384, 1), (5, 0, 5)):
            with self.assertRaises(FullModelLayerSequenceError):
                validate_execution_window(*args)
        with self.assertRaises(FullModelLayerSequenceError):
            build_layer_sequence_manifest(start_layer=-1)
        with self.assertRaises(FullModelLayerSequenceError):
            build_layer_sequence_manifest(end_layer=24)
        with self.assertRaises(FullModelLayerSequenceError):
            build_layer_sequence_manifest(start_layer=8, end_layer=7)

    def test_reference_snapshot_matches(self) -> None:
        self.assertEqual(reference_snapshot(), load_reference())


if __name__ == "__main__":
    unittest.main()
