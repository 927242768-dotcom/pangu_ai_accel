#!/usr/bin/env python3
"""G2 运行时量化 DDR3 验证事务测试。"""

from __future__ import annotations

import unittest

import numpy as np

try:
    from .runtime_quantizer_validation import (
        CONFIG_STRUCT,
        MATRIX_NAMES,
        RESULT_HEADER_STRUCT,
        build_config_payload,
        build_fixed_validation_cases,
        build_upload_payload,
        expected_result_payload,
        random_source_case,
        transaction_stress,
        validate_manifest,
        verify_upload_roundtrip,
        with_source_values,
    )
except ImportError:
    from runtime_quantizer_validation import (
        CONFIG_STRUCT,
        MATRIX_NAMES,
        RESULT_HEADER_STRUCT,
        build_config_payload,
        build_fixed_validation_cases,
        build_upload_payload,
        expected_result_payload,
        random_source_case,
        transaction_stress,
        validate_manifest,
        verify_upload_roundtrip,
        with_source_values,
    )


class RuntimeQuantizerValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = build_fixed_validation_cases()

    def test_seven_real_matrix_cases_and_manifest(self) -> None:
        self.assertEqual([case.name for case in self.cases], list(MATRIX_NAMES))
        self.assertEqual(len(self.cases), 7)
        validate_manifest(self.cases)

    def test_config_and_upload_roundtrip(self) -> None:
        for case in self.cases:
            self.assertEqual(len(build_config_payload(case)), CONFIG_STRUCT.size)
            self.assertEqual(len(build_upload_payload(case)), case.upload_bytes)
            verify_upload_roundtrip(case)

    def test_result_layout_and_padding(self) -> None:
        for case in self.cases:
            payload = expected_result_payload(case)
            self.assertEqual(len(payload), case.result_bytes)
            offset = RESULT_HEADER_STRUCT.size + case.activation_bytes
            combined = np.frombuffer(payload[offset:], dtype="<u4").reshape(
                case.rows, case.padded_groups
            )
            self.assertTrue(np.all(combined[:, case.groups :] == 0))
            if case.groups == 14:
                self.assertEqual(case.padded_groups, 16)
            else:
                self.assertEqual((case.groups, case.padded_groups), (76, 80))

    def test_qkv_and_gate_up_share_activation(self) -> None:
        np.testing.assert_array_equal(
            self.cases[0].activation_int8, self.cases[1].activation_int8
        )
        np.testing.assert_array_equal(
            self.cases[1].activation_int8, self.cases[2].activation_int8
        )
        np.testing.assert_array_equal(
            self.cases[4].activation_int8, self.cases[5].activation_int8
        )

    def test_axi_trace_exact_counts(self) -> None:
        for case in self.cases:
            trace = case.trace
            source_elements_per_beat = 4 if case.source_q28 else 16
            self.assertEqual(trace.source_read_beats, case.vector_length // source_elements_per_beat)
            self.assertEqual(trace.raw_scale_read_beats, case.rows * case.groups // 16)
            self.assertEqual(trace.activation_write_beats, case.vector_length // 32)
            self.assertEqual(
                trace.combined_write_beats,
                case.rows * (case.padded_groups // 8),
            )
            self.assertEqual(trace.source_read_commands, (trace.source_read_beats + 15) // 16)
            self.assertEqual(trace.raw_scale_read_commands, trace.raw_scale_read_beats)
            self.assertEqual(trace.activation_write_commands, trace.activation_write_beats)
            self.assertEqual(trace.combined_write_commands, trace.combined_write_beats)

    def test_q10_exact_half_tie_is_authoritative_for_stress(self) -> None:
        source = np.zeros(self.cases[1].vector_length, dtype=np.int16)
        source[0] = 4094
        source[67] = 2047
        rebuilt = with_source_values(
            self.cases[1],
            source,
            verify_numpy=False,
        )
        self.assertEqual(int(rebuilt.activation_int8[67]), 64)

    def test_random_source_cases_roundtrip(self) -> None:
        rng = np.random.default_rng(20260819)
        for case in (self.cases[1], self.cases[3]):
            for iteration in range(60):
                rebuilt = random_source_case(case, rng, iteration)
                verify_upload_roundtrip(rebuilt)
                self.assertEqual(rebuilt.trace, case.trace)
                self.assertEqual(
                    rebuilt.raw_scale_fp16_bits.tobytes(),
                    case.raw_scale_fp16_bits.tobytes(),
                )

    def test_transaction_stress(self) -> None:
        transaction_stress(rounds=1000, seed=20260819)


if __name__ == "__main__":
    unittest.main()
