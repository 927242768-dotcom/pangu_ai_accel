#!/usr/bin/env python3
"""阶段 H3 主机辅助参数换层与非零层配置测试。"""

from __future__ import annotations

import struct
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    from .full_model_memory_plan import (
        build_full_model_memory_plan,
        expand_layer_transfer_plan,
    )
    from .transformer_block_g2_payload import (
        TransformerBlockPayloadError,
        build_dynamic_uploads,
        build_layer_parameter_uploads,
        build_resident_uploads,
    )
    from .transformer_block_reference import kv_slot_byte_addresses
except ImportError:
    from full_model_memory_plan import (
        build_full_model_memory_plan,
        expand_layer_transfer_plan,
    )
    from transformer_block_g2_payload import (
        TransformerBlockPayloadError,
        build_dynamic_uploads,
        build_layer_parameter_uploads,
        build_resident_uploads,
    )
    from transformer_block_reference import kv_slot_byte_addresses

from tools.pangu_transformer_block_host import (
    build_config_payload,
    build_parser as build_host_parser,
)


class FullModelLayerUploadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image_path = Path("model_output/yanbo_qwen25_0.5b_int4.p50")
        cls.plan = build_full_model_memory_plan(cls.image_path)

    @staticmethod
    def _fake_case(*, count: int = 2) -> SimpleNamespace:
        history = count - 1
        return SimpleNamespace(
            query_position=count - 1,
            window_start=0,
            count=count,
            block_input_q10=np.zeros(896, dtype=np.int16),
            history_k_q28=np.zeros((history, 2, 64), dtype=np.int64),
            history_v_q28=np.zeros((history, 2, 64), dtype=np.int64),
        )

    def test_layer0_new_uploads_match_board_verified_resident_subset(self) -> None:
        resident = {item.name: item for item in build_resident_uploads()}
        layer0 = build_layer_parameter_uploads(0)
        self.assertEqual(len(layer0), 19)
        self.assertEqual(sum(len(item.payload) for item in layer0), 7_961_088)
        for upload in layer0:
            expected = resident[upload.name]
            self.assertEqual(upload.controller_address, expected.controller_address)
            self.assertEqual(upload.payload, expected.payload)
            self.assertTrue(upload.persistent)

    def test_all_24_layers_follow_h2_transfer_contract(self) -> None:
        for layer_index in range(24):
            uploads = build_layer_parameter_uploads(layer_index)
            transfers = expand_layer_transfer_plan(self.plan, layer_index, slot="A")
            self.assertEqual(len(uploads), 19)
            self.assertEqual(len(transfers), 19)
            self.assertEqual(sum(len(item.payload) for item in uploads), 7_961_088)
            for upload, transfer in zip(uploads, transfers):
                self.assertEqual(upload.name, transfer["destination_region"])
                self.assertEqual(upload.byte_address, transfer["destination_byte_address"])
                self.assertEqual(len(upload.payload), transfer["destination_nbytes"])

    def test_copy_transforms_match_exact_p50_bytes(self) -> None:
        with self.image_path.open("rb") as handle:
            for layer_index in (0, 1, 12, 23):
                uploads = {
                    item.name: item for item in build_layer_parameter_uploads(layer_index)
                }
                transfers = expand_layer_transfer_plan(
                    self.plan, layer_index, slot="A"
                )
                for transfer in transfers:
                    if transfer["transform"] not in {"copy_int4", "copy_fp16_scale"}:
                        continue
                    handle.seek(transfer["source_byte_offset"])
                    raw = handle.read(transfer["source_nbytes"])
                    self.assertEqual(
                        uploads[transfer["destination_region"]].payload,
                        raw,
                        (layer_index, transfer["source_role"]),
                    )

    def test_layer_payload_excludes_fpga_generated_combined_scales(self) -> None:
        names = {item.name for item in build_layer_parameter_uploads(23)}
        self.assertFalse(
            names
            & {
                "q_scale_uq4_28",
                "k_scale_uq4_28",
                "v_scale_uq4_28",
                "oproj_scale_uq4_28",
                "gate_scale_uq4_28",
                "up_scale_uq4_28",
                "down_scale_uq4_28",
            }
        )
        self.assertTrue(
            {
                "q_weight_scale_fp16",
                "k_weight_scale_fp16",
                "v_weight_scale_fp16",
                "oproj_weight_scale_fp16",
                "gate_weight_scale_fp16",
                "up_weight_scale_fp16",
                "down_weight_scale_fp16",
            }
            <= names
        )

    def test_nonzero_layer_dynamic_kv_addresses(self) -> None:
        case = self._fake_case(count=2)
        uploads = build_dynamic_uploads(case, layer_index=23)
        history = [item for item in uploads if item.name.startswith("kv_history_")]
        self.assertEqual(len(history), 2)
        expected_k, expected_v = kv_slot_byte_addresses(23, 0)
        self.assertEqual(history[0].byte_address, expected_k)
        self.assertEqual(history[1].byte_address, expected_v)
        self.assertEqual(history[0].name, "kv_history_k_layer_23_position_0")
        self.assertEqual(history[1].name, "kv_history_v_layer_23_position_0")

    def test_default_dynamic_upload_names_and_addresses_remain_layer0(self) -> None:
        case = self._fake_case(count=2)
        uploads = build_dynamic_uploads(case)
        history = [item for item in uploads if item.name.startswith("kv_history_")]
        expected_k, expected_v = kv_slot_byte_addresses(0, 0)
        self.assertEqual(history[0].name, "kv_history_k_position_0")
        self.assertEqual(history[1].name, "kv_history_v_position_0")
        self.assertEqual(history[0].byte_address, expected_k)
        self.assertEqual(history[1].byte_address, expected_v)

    def test_host_config_payload_carries_layer_without_breaking_default(self) -> None:
        case = self._fake_case(count=6)
        self.assertEqual(
            struct.unpack("<4H", build_config_payload(case)),
            (0, 5, 0, 6),
        )
        self.assertEqual(
            struct.unpack("<4H", build_config_payload(case, layer_index=23)),
            (23, 5, 0, 6),
        )

    def test_host_layer_params_cli(self) -> None:
        args = build_host_parser().parse_args(["layer-params", "23", "--verify"])
        self.assertEqual(args.command, "layer-params")
        self.assertEqual(args.layer, 23)
        self.assertTrue(args.verify)

    def test_invalid_layer_indices_are_rejected(self) -> None:
        with self.assertRaises(TransformerBlockPayloadError):
            build_layer_parameter_uploads(-1)
        with self.assertRaises(TransformerBlockPayloadError):
            build_layer_parameter_uploads(24)
        case = self._fake_case(count=1)
        with self.assertRaises(TransformerBlockPayloadError):
            build_dynamic_uploads(case, layer_index=28)
        with self.assertRaises(ValueError):
            build_config_payload(case, layer_index=28)


if __name__ == "__main__":
    unittest.main()
