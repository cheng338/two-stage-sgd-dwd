"""Bias finiteness retains the scalar-update guard's numerical behavior."""
import unittest

import numpy as np

from two_stage_sgd_dwd._core import kernel_step, linear_step
from test_kernel_cache import ExactAssertions

try:
    from two_stage_sgd_dwd import _sgd_accel
except (ImportError, OSError):
    _sgd_accel = None

requires_native = unittest.skipIf(_sgd_accel is None, "Optional compiled extension unavailable")


@requires_native
class NativeMultiplierCheckTests(ExactAssertions):
    def routes(self, kind, rows):
        yield "callable", rows.__getitem__
        if kind == 0 and getattr(_sgd_accel, "_supports_tuple_rows", False):
            yield "tuple", rows

    def checked_run(self, kind, initial, bias, rows, labels, indices, rates, access):
        arrays = (initial, labels, indices, rates) + tuple(rows)
        before = [array.tobytes() for array in arrays]
        result = _sgd_accel.run_steps(kind, initial, bias, access, labels, indices, rates, .125)
        self.assertEqual([array.tobytes() for array in arrays], before)
        self.assertFalse(np.shares_memory(initial, result[0]))
        return result

    def test_signed_zero_subnormal_and_extreme_finite_updates_match_scalar_steps(self):
        tiny = np.nextafter(0., 1.)
        margins = (-0., 0., np.nextafter(.5, 0.), .5, np.nextafter(.5, 1.), -1e308, 1e308)
        rates = (0., -0., tiny, 1e-300, .125, 1e100, float(np.finfo(np.float64).max))
        rows = (np.array([0., -0.]), np.zeros(2))
        initial = np.array([0., -0.])
        for kind in (0, 1):
            for label in (-1., 1.):
                labels = np.array([label, -label])
                for margin in margins:
                    for rate in rates:
                        bias = float(margin) / label
                        if kind == 0:
                            expected = linear_step(initial, bias, rows[0], label, rate)
                        else:
                            expected = kernel_step(initial, bias, rows[0], 0, label, rate, .125)
                        for route, access in self.routes(kind, rows):
                            with self.subTest(kind=kind, label=label, margin=margin, rate=rate, route=route):
                                actual = self.checked_run(kind, initial, bias, rows, labels,
                                    np.array([0], dtype=np.int64), np.array([rate]), access)
                                self.assertExact(actual[:2], expected)
                                self.assertEqual(actual[2:], (1, None))

    def test_finite_bias_overflow_retains_exact_committed_prefix_and_error(self):
        for kind in (0, 1):
            for sign in (-1., 1.):
                initial = np.array([1., 0.])
                bias = sign * 1e308
                rows = (np.zeros(2), np.array([-bias, 0.]))
                labels = np.array([sign, sign])
                for prefix in (0, 1):
                    indices = np.array(([0] if prefix else []) + [1, 0], dtype=np.int64)
                    rates = np.array(([0.] if prefix else []) + [1e308, .1])
                    for route, access in self.routes(kind, rows):
                        with self.subTest(kind=kind, sign=sign, prefix=prefix, route=route):
                            expected = self.checked_run(kind, initial, bias, rows, labels,
                                                        indices[:prefix], rates[:prefix], access)
                            actual = self.checked_run(kind, initial, bias, rows, labels,
                                                      indices, rates, access)
                            self.assertExact(actual[:2], expected[:2])
                            self.assertEqual(actual[2:], (prefix, "Nonfinite scalar update"))
                            # All inputs to the overflowing subtraction are
                            # finite; the independent step rejects it as well.
                            with self.assertRaises(FloatingPointError):
                                if kind == 0:
                                    linear_step(expected[0], expected[1], rows[1], sign, 1e308)
                                else:
                                    kernel_step(expected[0], expected[1], rows[1], 1, sign, 1e308, .125)

    def test_nonfinite_margin_still_fails_before_scalar_update(self):
        largest = float(np.finfo(np.float64).max)
        initial, labels = np.array([1., 0.]), np.array([-1., 1.])
        rows = (np.array([largest, 0.]), np.zeros(2))
        for kind in (0, 1):
            for route, access in self.routes(kind, rows):
                with self.subTest(kind=kind, route=route):
                    actual = self.checked_run(kind, initial, largest, rows, labels,
                        np.array([0], dtype=np.int64), np.array([.1]), access)
                    self.assertExact(actual[:2], (initial, largest))
                    self.assertEqual(actual[2:], (0, "Nonfinite scalar margin"))

    def test_invalid_steps_are_rejected_before_fetching_a_row(self):
        initial, labels = np.zeros(2), np.array([-1., 1.])
        indices = np.array([0], dtype=np.int64)
        for kind in (0, 1):
            for rate in (float("nan"), float("inf"), -float("inf"), -.1):
                with self.subTest(kind=kind, rate=rate):
                    calls = []

                    def get_row(index):
                        calls.append(index)
                        raise AssertionError("Invalid step reached row callback")

                    rates = np.array([rate])
                    before = [array.tobytes() for array in (initial, labels, indices, rates)]
                    with self.assertRaisesRegex(ValueError, "Invalid sampled observation or step size"):
                        _sgd_accel.run_steps(kind, initial, 0., get_row, labels, indices, rates, .125)
                    self.assertEqual(calls, [])
                    self.assertEqual([array.tobytes() for array in (initial, labels, indices, rates)], before)
                    if kind == 0 and getattr(_sgd_accel, "_supports_tuple_rows", False):
                        # Empty tuple would raise IndexError if accessed first.
                        with self.assertRaisesRegex(ValueError, "Invalid sampled observation or step size"):
                            _sgd_accel.run_steps(kind, initial, 0., (), labels, indices, rates, .125)

    def test_nonfinite_initial_bias_still_fails_before_fetching_a_row(self):
        initial, labels = np.zeros(2), np.array([-1., 1.])
        indices, rates = np.array([0], dtype=np.int64), np.array([.125])
        for kind in (0, 1):
            for bias in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(kind=kind, bias=bias):
                    calls = []

                    def get_row(index):
                        calls.append(index)
                        raise AssertionError('Invalid initial bias reached row callback')

                    with self.assertRaisesRegex(ValueError, 'Invalid scalar solver controls'):
                        _sgd_accel.run_steps(kind, initial, bias, get_row, labels, indices, rates, .125)
                    self.assertEqual(calls, [])

    def test_bias_guard_also_rejects_rate_mutation_inside_a_row_callback(self):
        # A callback runs after input-step validation. Even if it changes that
        # step, a nonfinite multiplier cannot escape the finite-bias check.
        initial, labels = np.zeros(2), np.array([-1., 1.])
        indices = np.array([0, 1], dtype=np.int64)
        for kind in (0, 1):
            for changed_rate in (float('nan'), float('inf'), -float('inf')):
                for prefix in (0, 1):
                    with self.subTest(kind=kind, rate=changed_rate, prefix=prefix):
                        rates = np.array([0., 0.])
                        calls = []

                        def get_row(index):
                            calls.append(index)
                            if len(calls) == prefix + 1:
                                rates[prefix] = changed_rate
                            return np.zeros(2)

                        actual = _sgd_accel.run_steps(
                            kind, initial, .125, get_row, labels, indices, rates, .125)
                        self.assertExact(actual[:2], (initial, .125))
                        self.assertEqual(actual[2:], (prefix, 'Nonfinite scalar update'))
                        self.assertEqual(len(calls), prefix + 1)


if __name__ == "__main__":
    unittest.main()
