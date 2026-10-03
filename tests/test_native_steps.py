"""Tiny native-prefix and C API safety checks against the NumPy step oracle."""
import unittest

import numpy as np
from two_stage_sgd_dwd._core import kernel_step, linear_step

try:
    from two_stage_sgd_dwd import _sgd_accel
except (ImportError, OSError):
    _sgd_accel = None


def unaligned(values):
    values = np.asarray(values)
    result = np.ndarray(values.shape, dtype=values.dtype,
                        buffer=bytearray(values.nbytes + 1), offset=1)
    result[...] = values
    assert not result.flags.aligned
    return result


class ReadonlyCopy(np.ndarray):
    def __array_function__(self, function, types, args, kwargs):
        if function is np.copy:
            result = np.array(np.asarray(self), copy=True)
            result.setflags(write=False)
            return result
        return super().__array_function__(function, types, args, kwargs)


@unittest.skipIf(_sgd_accel is None, "Optional compiled extension unavailable")
class NativeStepTests(unittest.TestCase):
    def exact(self, a, b):
        a, b = np.asarray(a), np.asarray(b)
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.dtype, b.dtype)
        self.assertEqual(a.tobytes(), b.tobytes())

    def reference(self, kind, coefficient, bias, rows, labels, indices, rates, lambd):
        coefficient = coefficient.copy()
        for i, eta in zip(indices, rates):
            if kind == 0:
                coefficient, bias = linear_step(coefficient, bias, rows[int(i)], labels[i], eta)
            else:
                coefficient, bias = kernel_step(coefficient, bias, rows[int(i)], int(i), labels[i], eta, lambd)
        return coefficient, bias

    def every_prefix(self, kind, coefficient, bias, rows, labels, indices, rates, lambd):
        original = coefficient.copy()
        for stop in range(len(indices) + 1):
            with self.subTest(kind=kind, width=len(coefficient), prefix=stop):
                calls = []

                def get_row(index):
                    calls.append(index)
                    return rows[index]

                actual, intercept, completed, error = _sgd_accel.run_steps(
                    kind, coefficient, bias, get_row, labels, indices[:stop], rates[:stop], lambd)
                expected, expected_bias = self.reference(
                    kind, coefficient, bias, rows, labels, indices[:stop], rates[:stop], lambd)
                self.exact(actual, expected)
                self.exact(np.float64(intercept), np.float64(expected_bias))
                self.assertEqual(completed, stop)
                self.assertIsNone(error)
                self.assertEqual(calls, indices[:stop].tolist())
                self.exact(coefficient, original)
                self.assertFalse(np.shares_memory(coefficient, actual))

    def test_linear_random_every_prefix_across_dimensions_and_seeds(self):
        for seed in (19, 117, 2026):
            for width in (1, 2, 7, 17, 64):
                rng = np.random.default_rng(seed)
                rows = np.ascontiguousarray(rng.normal(size=(7, width)))
                labels = np.array([-1., 1., 1., -1., 1., -1., -1.])
                indices = rng.integers(7, size=13, dtype=np.int64)
                rates = np.ascontiguousarray(rng.uniform(0., .8, size=13))
                self.every_prefix(0, rng.normal(size=width), -.37, rows, labels, indices, rates, .2)

    def test_kernel_nonidentity_gram_every_prefix(self):
        for seed in (41, 1729):
            for n in (1, 3, 11, 31):
                rng = np.random.default_rng(seed)
                X = rng.normal(size=(n, 5))
                distances = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
                gram = np.exp(-.23 * distances)
                rows = np.ascontiguousarray(gram.T)
                labels = np.where(np.arange(n) % 3, 1., -1.)
                indices = rng.integers(n, size=11, dtype=np.int64)
                rates = np.ascontiguousarray(rng.uniform(.01, .7, size=11))
                self.every_prefix(1, rng.normal(size=n) * .2, .17,
                                  rows, labels, indices, rates, .31)

    def test_large_projection_and_unit_ball_boundary_branches(self):
        cases = [np.array([3., 4.]), np.array([1e300, -2e300, 3e300]),
                 np.array([.6, .8]), np.array([1e-300, -1e-300]), np.zeros(5)]
        for initial in cases:
            with self.subTest(initial=initial.tolist()):
                rows = np.zeros((2, len(initial)))
                labels = np.array([-1., 1.])
                self.every_prefix(0, initial, -.2, rows, labels,
                                  np.array([0, 1, 0], dtype=np.int64), np.array([0., .1, .3]), .1)

    def test_loss_boundary_reciprocal_tail_and_kernel_shrink_order(self):
        for margin in (np.nextafter(.5, 0.), .5, np.nextafter(.5, 1.), 8., 1e150):
            for label in (-1., 1.):
                for kind in (0, 1):
                    with self.subTest(margin=margin, label=label, kind=kind):
                        self.every_prefix(kind, np.array([0.]), margin / label,
                                          np.array([[1.]]), np.array([label]),
                                          np.array([0], dtype=np.int64), np.array([.13]), .07)
        # The API also faithfully applies negative shrink factors; callers own
        # learning-rate policy. This catches reassociation of 2*eta*lambd.
        self.every_prefix(1, np.array([.7, -.3]), .2, np.array([[1., .2], [.2, 1.]]),
                          np.array([-1., 1.]), np.array([1, 0], dtype=np.int64),
                          np.array([.7, .4]), 2.)

    def test_readonly_inputs_are_allowed_and_never_modified(self):
        coefficient = np.array([.2, -.1])
        labels = np.array([-1., 1.])
        indices = np.array([1, 0, 1], dtype=np.int64)
        rates = np.array([.1, .2, .3])
        rows = np.array([[1., .4], [.4, 1.]])
        arrays = (coefficient, labels, indices, rates, rows)
        originals = [x.copy() for x in arrays]
        for array in arrays:
            array.setflags(write=False)
        self.every_prefix(1, coefficient, .1, rows, labels, indices, rates, .2)
        for before, after in zip(originals, arrays):
            self.exact(before, after)
            self.assertFalse(after.flags.writeable)

    def fixture(self):
        return [1, np.array([.2, -.1]), .1, lambda i: np.array([1., .4]),
                np.array([-1., 1.]), np.array([0], dtype=np.int64), np.array([.2]), .3]

    def test_invalid_vector_types_dimensions_dtypes_and_layouts_reject(self):
        for position in (1, 4, 5, 6):
            original = self.fixture()[position]
            bad = [original.tolist(), original.reshape(1, -1),
                   original.astype(np.int32 if position == 5 else np.float32),
                   unaligned(original)]
            if len(original) > 1:
                bad.append(np.tile(original, 2)[::2])
            for value in bad:
                with self.subTest(argument=position, dtype=getattr(value, "dtype", None),
                                  shape=getattr(value, "shape", None)):
                    args = self.fixture()
                    args[position] = value
                    with self.assertRaises((ValueError, TypeError, BufferError)):
                        _sgd_accel.run_steps(*args)

    def test_invalid_callback_vectors_reject(self):
        for value in ([1., 2.], np.ones((2, 1)), np.ones(2, dtype=np.float32),
                      np.ones(4)[::2], np.ones(3), unaligned(np.ones(2))):
            with self.subTest(type=type(value).__name__, shape=getattr(value, "shape", None)):
                args = self.fixture()
                args[3] = lambda i: value
                with self.assertRaises((ValueError, TypeError, BufferError)):
                    _sgd_accel.run_steps(*args)

    def test_internal_numpy_copy_must_be_writable(self):
        args = self.fixture()
        args[1] = args[1].view(ReadonlyCopy)
        with self.assertRaises((ValueError, BufferError)):
            _sgd_accel.run_steps(*args)

    def test_scalar_index_length_and_nonfinite_validation(self):
        cases = [(0, 2), (2, float('nan')), (2, float('inf')), (3, None),
                 (7, 0.), (7, -1.), (7, float('nan')),
                 (1, np.array([float('inf'), 0.])), (1, np.array([])),
                 (4, np.array([-1.])), (4, np.array([0., 1.])),
                 (5, np.array([-1], dtype=np.int64)), (5, np.array([2], dtype=np.int64)),
                 (6, np.array([])), (6, np.array([-.1])), (6, np.array([float('inf')]))]
        for position, value in cases:
            with self.subTest(argument=position, value=value):
                args = self.fixture()
                args[position] = value
                with self.assertRaises((ValueError, TypeError)):
                    _sgd_accel.run_steps(*args)

    def test_callback_numerical_exception_returns_exact_last_committed_prefix(self):
        rows = np.array([[1., .2], [.2, 1.]])
        indices, rates = np.array([0, 1, 0, 1], dtype=np.int64), np.array([.1, .2, .3, .4])
        labels, initial = np.array([-1., 1.]), np.array([.2, -.3])
        for kind in (0, 1):
            for error_type in (FloatingPointError, OverflowError):
                for fail_at in (0, 2, 3):
                    with self.subTest(kind=kind, error=error_type.__name__, prefix=fail_at):
                        calls = []

                        def get_row(index):
                            calls.append(index)
                            if len(calls) == fail_at + 1:
                                raise error_type("controlled evaluation failure")
                            return rows[index]

                        actual, bias, completed, error = _sgd_accel.run_steps(
                            kind, initial, .1, get_row, labels, indices, rates, .2)
                        expected, expected_bias = self.reference(
                            kind, initial, .1, rows, labels, indices[:fail_at], rates[:fail_at], .2)
                        self.exact(actual, expected)
                        self.exact(np.float64(bias), np.float64(expected_bias))
                        self.assertEqual(completed, fail_at)
                        self.assertEqual(error, "controlled evaluation failure")
                        self.assertEqual(calls, indices[:fail_at + 1].tolist())

    def test_nonnumerical_callback_exception_is_not_converted_or_swallowed(self):
        for error in (RuntimeError("callback bug"), ValueError("bad callback"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                args = self.fixture()

                def fail(index):
                    raise error

                args[3] = fail
                with self.assertRaises(type(error)) as captured:
                    _sgd_accel.run_steps(*args)
                self.assertIs(captured.exception, error)

    def test_overflow_never_commits_partial_coefficients_or_bias(self):
        for kind in (0, 1):
            rows = np.array([[.2, .4], [0., -1e308]]) if kind == 0 else np.eye(2)
            labels = np.array([-1., -1.])
            indices = np.array([0, 1], dtype=np.int64)
            rates = np.array([.1, 1e308])
            initial = np.array([0., 0.])
            lambd = 10. if kind == 1 else .2
            expected, expected_bias = self.reference(kind, initial, 0., rows, labels, indices[:1], rates[:1], lambd)
            with np.errstate(all='ignore'):
                actual, bias, completed, error = _sgd_accel.run_steps(
                    kind, initial, 0., lambda i: rows[i], labels, indices, rates, lambd)
            self.exact(actual, expected)
            self.exact(np.float64(bias), np.float64(expected_bias))
            self.assertEqual(completed, 1)
            self.assertIsInstance(error, str)
            self.assertIn("Nonfinite", error)


if __name__ == '__main__':
    unittest.main()
