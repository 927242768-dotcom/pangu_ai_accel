#!/usr/bin/env python3
"""阶段 H3 UART 配置读回、DDR copy 与独立顶层静态契约测试。"""

from __future__ import annotations

import struct
import unittest
from pathlib import Path

from tools.pangu_full_model_h3_host import (
    CONFIG_FRAME,
    COPY_HEADER,
    build_parser,
    configure_layer,
    copy_region,
    read_configuration,
    verify_configuration,
)


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "transformer_block_g2" / "rtl"
H3_TOP = ROOT / "full_model_h3" / "rtl" / "full_model_h3_top.v"
H3_BUILD = ROOT / "full_model_h3" / "pnr" / "build_full_model_h3.tcl"


class FakePort:
    def __init__(self, response: bytes) -> None:
        self.response = bytearray(response)
        self.written = bytearray()
        self.flush_count = 0
        self.reset_count = 0

    def reset_input_buffer(self) -> None:
        self.reset_count += 1

    def write(self, payload: bytes | memoryview) -> int:
        resolved = bytes(payload)
        self.written.extend(resolved)
        return len(resolved)

    def flush(self) -> None:
        self.flush_count += 1

    def read(self, size: int) -> bytes:
        chunk = bytes(self.response[:size])
        del self.response[:size]
        return chunk


class FullModelH3ProtocolTests(unittest.TestCase):
    def test_configure_layer_writes_real_layer_and_window(self) -> None:
        port = FakePort(b"K\r\n")
        configure_layer(
            port,
            layer=23,
            query_position=16_383,
            window_start=16_368,
            count=16,
            timeout=0.1,
        )
        self.assertEqual(
            bytes(port.written),
            b"C" + struct.pack("<4H", 23, 16_383, 16_368, 16),
        )
        self.assertEqual(port.flush_count, 1)

    def test_configuration_readback_frame_is_checked_field_by_field(self) -> None:
        frame = CONFIG_FRAME.pack(b"L", 17, 2026, 2011, 16, b"\r\n")
        port = FakePort(frame)
        actual = read_configuration(port, 0.1)
        self.assertEqual(bytes(port.written), b"L")
        self.assertEqual(
            actual,
            {
                "layer": 17,
                "query_position": 2026,
                "window_start": 2011,
                "count": 16,
            },
        )
        verify_configuration(
            actual,
            layer=17,
            query_position=2026,
            window_start=2011,
            count=16,
        )
        with self.assertRaises(AssertionError):
            verify_configuration(
                actual,
                layer=16,
                query_position=2026,
                window_start=2011,
                count=16,
            )

    def test_copy_hidden_command_uses_controller_addresses_and_1792_bytes(self) -> None:
        port = FakePort(b"K\r\n")
        elapsed = copy_region(
            port,
            source_controller_address=0x0000_D000,
            destination_controller_address=0,
            length=1792,
            baud=115200,
            timeout=0.1,
        )
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(
            bytes(port.written),
            b"M" + COPY_HEADER.pack(0x0000_D000, 0, 1792),
        )

    def test_copy_rejects_alignment_overlap_and_zero_length(self) -> None:
        invalid = (
            (1, 0, 32),
            (8, 1, 32),
            (8, 16, 0),
            (8, 16, 31),
            (8, 12, 32),
        )
        for source, destination, length in invalid:
            with self.subTest(source=source, destination=destination, length=length):
                with self.assertRaises(ValueError):
                    copy_region(
                        FakePort(b"K\r\n"),
                        source_controller_address=source,
                        destination_controller_address=destination,
                        length=length,
                        baud=115200,
                        timeout=0.1,
                    )

    def test_cli_exposes_dry_run_single_layer_and_sequence(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args(["dry-run"]).command, "dry-run")
        single = parser.parse_args(["run-layer", "23", "--copy-output"])
        self.assertEqual(single.layer, 23)
        self.assertTrue(single.copy_output)
        sequence = parser.parse_args(
            [
                "run-sequence", "--start-layer", "2", "--end-layer", "5",
                "--check-reference",
            ]
        )
        self.assertEqual((sequence.start_layer, sequence.end_layer), (2, 5))
        self.assertTrue(sequence.check_reference)

    def test_g2_defaults_stay_single_layer_and_copy_disabled(self) -> None:
        ctrl = (RTL / "transformer_block_ctrl.v").read_text(encoding="utf-8")
        host = (RTL / "transformer_block_host_ctrl.v").read_text(encoding="utf-8")
        top = (RTL / "transformer_block_top.v").read_text(encoding="utf-8")
        self.assertIn("parameter integer ACTIVE_LAYER_COUNT = 1", ctrl)
        self.assertIn("parameter integer ACTIVE_LAYER_COUNT = 1", host)
        self.assertIn("parameter integer ENABLE_DDR_COPY    = 0", host)
        self.assertIn("parameter ACTIVE_LAYER_COUNT  = 1", top)
        self.assertIn("parameter ENABLE_DDR_COPY     = 0", top)
        self.assertIn("(cfg_layer < ACTIVE_LAYER_COUNT)", ctrl)
        self.assertIn("(new_layer < ACTIVE_LAYER_COUNT)", host)

    def test_h3_top_preserves_ddr_hierarchy_and_enables_real_24_layers(self) -> None:
        top = H3_TOP.read_text(encoding="utf-8")
        self.assertIn("module full_model_h3_top", top)
        self.assertIn(") I_ipsxb_ddr_top (", top)
        self.assertIn("transformer_block_host_ctrl #(", top)
        self.assertIn(".ACTIVE_LAYER_COUNT (24)", top)
        self.assertIn(".ENABLE_DDR_COPY    (1)", top)
        self.assertIn(".FULL_MODEL_MODE    (1)", top)
        self.assertNotIn("ACTIVE_LAYER_COUNT (28)", top)
        self.assertNotIn("transformer_block_top #(", top)

    def test_pds_script_builds_direct_h3_top_with_existing_ddr_constraint(self) -> None:
        script = H3_BUILD.read_text(encoding="utf-8")
        self.assertIn('add_design "../rtl/full_model_h3_top.v"', script)
        self.assertIn("compile -top_module full_model_h3_top", script)
        self.assertIn('add_constraint "$ip_root/pnr/ddr_test.fdc"', script)
        self.assertIn("H3_FRONTEND_ONLY", script)
        self.assertNotIn('add_design "$g2_rtl/transformer_block_top.v"', script)

    def test_rtl_protocol_contains_config_readback_and_safe_copy_states(self) -> None:
        host = (RTL / "transformer_block_host_ctrl.v").read_text(encoding="utf-8")
        for token in (
            "8'h4c, 8'h6c",
            "8'h4d, 8'h6d",
            "ST_SEND_CONFIG",
            "ST_RECV_COPY_HEADER",
            "ST_SETUP_COPY_READ",
            "ST_SETUP_COPY_WRITE",
            "new_copy_nonoverlap",
            "ERR_COPY_DISABLED",
            "ERR_COPY_OVERLAP",
            'FULL_MODEL_MODE ? "H" : "G"',
            'FULL_MODEL_MODE ? "3" : "2"',
        ):
            self.assertIn(token, host)


if __name__ == "__main__":
    unittest.main()
