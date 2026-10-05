"""Reusing selected margins preserves derivative bytes and input contracts."""
import pickle
import unittest

import numpy as np

from two_stage_sgd_dwd._core import _finite_array, _finite_result, dwd_loss_derivative


def original_derivative(value):
    """Pre-change arithmetic, including both independent Boolean selections."""
    margins = _finite_array(value, "u")
    result = np.full_like(margins, -1.0)
    upper = margins > .5
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        result[upper] = (-.25 / margins[upper]) / margins[upper]
    _finite_result(result, "DWD derivative")
    return float(result) if margins.ndim == 0 else result


class DerivativeSelectionTests(unittest.TestCase):
    def compare(self, value):
        saved = pickle.dumps(value, protocol=5)
        layout = None
        if isinstance(value, np.ndarray):
            layout = (value.strides, value.flags.writeable, value.dtype, value.shape)
        with np.errstate(all="raise"):
            expected = original_derivative(value)
            actual = dwd_loss_derivative(value)
        self.assertIs(type(actual), type(expected))
        a, b = np.asarray(actual), np.asarray(expected)
        self.assertEqual((a.dtype, a.shape), (b.dtype, b.shape))
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertEqual(pickle.dumps(value, protocol=5), saved)
        if layout is not None:
            self.assertEqual((value.strides, value.flags.writeable, value.dtype, value.shape), layout)
            if isinstance(actual, np.ndarray):
                self.assertFalse(np.shares_memory(actual, value))
                self.assertEqual((actual.flags.c_contiguous, actual.flags.f_contiguous),
                                 (expected.flags.c_contiguous, expected.flags.f_contiguous))
        return actual

    def test_python_and_zero_dimensional_scalars_include_threshold_and_underflow(self):
        tiny = np.nextafter(0., 1.)
        values = (-float(np.finfo(np.float64).max), -0., 0., tiny,
                  np.nextafter(.5, 0.), .5, np.nextafter(.5, 1.), 1., 1e308,
                  float(np.finfo(np.float64).max))
        for value in values:
            for supplied in (float(value), np.float64(value), np.array(value)):
                with self.subTest(value=value, type=type(supplied).__name__):
                    self.assertIsInstance(self.compare(supplied), float)
        self.assertEqual(np.float64(dwd_loss_derivative(1e308)).tobytes(),
                         np.float64(-0.).tobytes())

    def test_empty_and_varied_array_layouts_preserve_exact_values_and_ownership(self):
        base = np.array([[-2., -.0, .5, 1.], [.25, np.nextafter(.5, 1.), 1e308, 4.]])
        readonly = base.copy()
        readonly.setflags(write=False)
        cases = (np.empty(0), np.empty((2, 0)), base, base.T, np.asfortranarray(base),
                 base[:, ::2], base[::-1, ::-1], readonly,
                 np.array([-4., -0., 0., .5]), np.array([.6, 1., 3., 1e308]))
        for value in cases:
            with self.subTest(shape=value.shape, strides=value.strides, writable=value.flags.writeable):
                self.compare(value)

    def test_convertible_dtypes_and_python_sequences_match_original(self):
        cases = (True, False, 3, "0.5", [], [.25, .5, 1.], [[-1, 1], [2, 3]],
                 np.array([-2, 0, 1, 100], dtype=np.int32),
                 np.array([False, True]), np.array([.25, .5, .75], dtype=np.float32),
                 np.array(["-1", "0.5", "2.0"]), np.array([-.5, 2], dtype=object),
                 np.array([.25, .5, 1.], dtype=">f8"))
        for value in cases:
            with self.subTest(type=type(value).__name__, dtype=getattr(value, "dtype", None)):
                self.compare(value)

    def test_nonfinite_complex_and_invalid_inputs_keep_errors_and_input_bytes(self):
        invalid = (float("nan"), float("inf"), -float("inf"),
                   np.array([.5, np.nan]), np.array([[np.inf, 1.]]),
                   1 + 0j, np.array([1., 2.], dtype=np.complex128),
                   np.array([1 + 2j], dtype=object), "not a margin", [1., "bad"],
                   [[1.], [2., 3.]], None)
        for value in invalid:
            with self.subTest(type=type(value).__name__, dtype=getattr(value, "dtype", None)):
                saved = pickle.dumps(value, protocol=5)
                errors = []
                for function in (original_derivative, dwd_loss_derivative):
                    try:
                        function(value)
                    except Exception as error:
                        errors.append((type(error), str(error)))
                    else:
                        self.fail("Invalid input unexpectedly succeeded")
                self.assertEqual(errors[0], errors[1])
                self.assertEqual(pickle.dumps(value, protocol=5), saved)


if __name__ == "__main__":
    unittest.main()
