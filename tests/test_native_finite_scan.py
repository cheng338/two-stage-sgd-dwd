"""Linear validation-scan optimization preserves exact updates and failures."""
import pickle
import unittest

import numpy as np

from two_stage_sgd_dwd import KernelSGDDWD, LinearSGDDWD, TwoStageDWDClassifier
from two_stage_sgd_dwd._core import kernel_step, linear_step

try:
    from two_stage_sgd_dwd import _sgd_accel
except (ImportError, OSError):
    _sgd_accel = None

requires_native = unittest.skipIf(_sgd_accel is None, 'Optional compiled extension unavailable')


def scalar_prefix(kind, initial, bias, rows, labels, indices, rates, lambd):
    coefficients = np.array(initial, dtype=np.float64, copy=True)
    for index, rate in zip(indices, rates):
        if kind == 0:
            coefficients, bias = linear_step(coefficients, bias, rows[int(index)], labels[index], rate)
        else:
            coefficients, bias = kernel_step(coefficients, bias, rows[int(index)], int(index),
                                             labels[index], rate, lambd)
    return coefficients, bias


class NormFault(np.ndarray):
    """Private C API fixture: retain custom arrays and inject a bad norm."""
    mutate = False

    def __array_function__(self, function, types, args, kwargs):
        if function is np.copy:
            return np.array(np.asarray(args[0]), copy=True).view(type(self))
        if function is np.empty_like:
            return np.empty_like(np.asarray(args[0])).view(type(self))
        if function is np.dot:
            if args[0] is args[1]:
                if type(self).mutate:
                    args[0][0] = float('inf')
                    return np.float64(0.)
                return np.float64(-1.)
            return np.dot(np.asarray(args[0]), np.asarray(args[1]))
        return super().__array_function__(function, types, args, kwargs)


class MutatingNorm(NormFault):
    mutate = True


@requires_native
class NativeFiniteScanTests(unittest.TestCase):
    def exact(self, a, b):
        first, second = np.asarray(a), np.asarray(b)
        self.assertEqual(first.dtype, second.dtype)
        self.assertEqual(first.shape, second.shape)
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_all_projection_branches_match_scalar_reference_at_every_prefix(self):
        cases = {
            'inside': np.array([.1, -.2]),
            'zero': np.array([0., -0.]),
            'just_below_unit': np.array([np.nextafter(1., 0.)]),
            'unit': np.array([1.]),
            'unit_two_coordinates': np.array([.6, .8]),
            'norm_only_projection': np.array([.8, .8]),
            'just_above_unit': np.array([np.nextafter(1., float('inf'))]),
            'max_scale_projection': np.array([3., 4.]),
            'huge_scale_projection': np.array([1e300, -2e300, 3e300]),
            'subnormal_inside': np.array([np.nextafter(0., 1.), -np.nextafter(0., 1.)]),
        }
        for name, initial in cases.items():
            rows = np.zeros((2, len(initial)))
            rows[1] = np.linspace(-.2, .3, len(initial))
            labels = np.array([-1., 1.])
            indices = np.array([0, 1, 0, 1], dtype=np.int64)
            rates = np.array([0., .1, .03, .5])
            before = initial.tobytes()
            for stop in range(5):
                with self.subTest(branch=name, prefix=stop):
                    actual, bias, completed, error = _sgd_accel.run_steps(
                        0, initial, -.2, rows.__getitem__, labels, indices[:stop], rates[:stop], 0.)
                    expected, expected_bias = scalar_prefix(
                        0, initial, -.2, rows, labels, indices[:stop], rates[:stop], 0.)
                    self.exact(actual, expected)
                    self.exact(np.float64(bias), np.float64(expected_bias))
                    self.assertEqual(completed, stop)
                    self.assertIsNone(error)
                    self.assertEqual(initial.tobytes(), before)

    def test_initial_nonfinite_coefficients_still_reject_without_input_mutation(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                initial = np.array([value, 0.])
                before = initial.tobytes()
                with self.assertRaisesRegex(ValueError, 'Nonfinite initial'):
                    _sgd_accel.run_steps(0, initial, 0., lambda index: np.zeros(2),
                        np.array([1.]), np.array([0], dtype=np.int64), np.array([.1]), 0.)
                self.assertEqual(initial.tobytes(), before)

    def test_proposal_overflow_returns_last_finite_committed_prefix(self):
        rows = np.array([[0., 0.], [1e308, -1e308]])
        labels = np.array([1., 1.])
        indices = np.array([0, 0, 1], dtype=np.int64)
        rates = np.array([.1, .2, 2.])
        initial = np.zeros(2)
        before = initial.tobytes()
        actual, bias, completed, error = _sgd_accel.run_steps(
            0, initial, 0., rows.__getitem__, labels, indices, rates, 0.)
        expected, expected_bias = scalar_prefix(0, initial, 0., rows, labels, indices[:2], rates[:2], 0.)
        self.exact(actual, expected)
        self.exact(np.float64(bias), np.float64(expected_bias))
        self.assertEqual(completed, 2)
        self.assertEqual(error, 'Nonfinite linear proposal')
        self.assertEqual(initial.tobytes(), before)

    def test_nonfinite_projection_division_remains_detected(self):
        # A negative injected squared norm makes sqrt/division nonfinite.
        # The projected branch must still validate before committing the update.
        initial = np.array([3., 4.]).view(NormFault)
        before = initial.tobytes()
        actual, bias, completed, error = _sgd_accel.run_steps(
            0, initial, 0., lambda index: np.zeros(2), np.array([1.]),
            np.array([0], dtype=np.int64), np.array([0.]), 0.)
        self.exact(np.asarray(actual), np.asarray(initial))
        self.assertEqual(bias, 0.)
        self.assertEqual(completed, 0)
        self.assertEqual(error, 'Nonfinite coefficients')
        self.assertEqual(initial.tobytes(), before)

    def test_custom_array_self_dot_cannot_bypass_nonfinite_validation(self):
        # Exact ordinary ndarray self-dot does not mutate its input. Preserve
        # the generic private API's final check when custom array hooks can.
        initial = np.array([.1, -.2]).view(MutatingNorm)
        before = initial.tobytes()
        actual, bias, completed, error = _sgd_accel.run_steps(
            0, initial, 0., lambda index: np.zeros(2), np.array([1.]),
            np.array([0], dtype=np.int64), np.array([0.]), 0.)
        self.exact(np.asarray(actual), np.asarray(initial))
        self.assertEqual(bias, 0.)
        self.assertEqual(completed, 0)
        self.assertEqual(error, 'Nonfinite coefficients')
        self.assertEqual(initial.tobytes(), before)

    def test_callback_numerical_failures_preserve_original_prefix(self):
        rows = np.array([[.2, .1], [-.1, .3]])
        labels = np.array([-1., 1.])
        indices = np.array([0, 1, 0, 1], dtype=np.int64)
        rates = np.array([.1, .2, .3, .4])
        initial = np.array([.1, -.2])
        for fail_at in (0, 2, 3):
            with self.subTest(prefix=fail_at):
                calls = []

                def get_row(index):
                    calls.append(index)
                    if len(calls) == fail_at + 1:
                        raise FloatingPointError('Controlled row failure')
                    return rows[index]

                actual, bias, completed, error = _sgd_accel.run_steps(
                    0, initial, .1, get_row, labels, indices, rates, 0.)
                expected, expected_bias = scalar_prefix(
                    0, initial, .1, rows, labels, indices[:fail_at], rates[:fail_at], 0.)
                self.exact(actual, expected)
                self.exact(np.float64(bias), np.float64(expected_bias))
                self.assertEqual(completed, fail_at)
                self.assertEqual(error, 'Controlled row failure')
                self.assertEqual(calls, indices[:fail_at + 1].tolist())

    def test_kernel_updates_and_nonfinite_proposals_are_unchanged(self):
        rows = np.array([[1., .4, .2], [.4, 1., .3], [.2, .3, 1.]])
        labels = np.array([-1., 1., 1.])
        initial = np.array([.2, -.3, .1])
        indices = np.array([1, 0, 2, 1], dtype=np.int64)
        rates = np.array([.1, .2, .7, .4])
        for stop in range(5):
            with self.subTest(prefix=stop):
                actual, bias, completed, error = _sgd_accel.run_steps(
                    1, initial, .1, rows.__getitem__, labels, indices[:stop], rates[:stop], 2.)
                expected, expected_bias = scalar_prefix(
                    1, initial, .1, rows, labels, indices[:stop], rates[:stop], 2.)
                self.exact(actual, expected)
                self.exact(np.float64(bias), np.float64(expected_bias))
                self.assertEqual(completed, stop)
                self.assertIsNone(error)
        # Finite shrinkage can still overflow a coefficient, so the kernel
        # final scan is essential even though its controls are all finite.
        initial = np.array([1e308, -1e308])
        actual, bias, completed, error = _sgd_accel.run_steps(
            1, initial, 0., lambda index: np.zeros(2), np.array([1., -1.]),
            np.array([0], dtype=np.int64), np.array([2.]), 1.)
        self.exact(actual, initial)
        self.assertEqual(completed, 0)
        self.assertEqual(bias, 0.)
        self.assertEqual(error, 'Nonfinite coefficients')

    def test_complete_public_fits_and_rng_match_python(self):
        x = np.linspace(-1.5, 1.5, 14)
        X = np.column_stack((x, .13 * np.cos(np.arange(14)), x * x / 7))
        y = np.where(x < 0, -1, 1)
        for sampling in ('with_replacement', 'shuffle'):
            for constructor in (LinearSGDDWD, KernelSGDDWD, TwoStageDWDClassifier):
                with self.subTest(sampling=sampling, model=constructor.__name__):
                    rng = np.random.Generator(np.random.PCG64DXSM(2017))
                    external = pickle.dumps(rng, protocol=5)
                    options = dict(max_epochs=5, random_state=rng, sampling=sampling)
                    if constructor is not LinearSGDDWD:
                        options['kernel_cache_bytes'] = len(y) ** 2 * 8
                    if constructor is TwoStageDWDClassifier:
                        options['retention_fraction'] = .8
                    reference = constructor(optimizer_backend='python', **options).fit(X, y)
                    candidate = constructor(optimizer_backend='native', **options).fit(X, y)
                    pairs = [(reference, candidate)]
                    if constructor is TwoStageDWDClassifier:
                        pairs = [(reference.linear_estimator_, candidate.linear_estimator_),
                                 (reference.kernel_estimator_, candidate.kernel_estimator_)]
                        for name in ('stage_seeds_', 'selected_indices_', 'selected_ids_',
                                     'selected_class_counts_', 'selection_scores_'):
                            self.assertEqual(pickle.dumps(getattr(reference, name), protocol=5),
                                             pickle.dumps(getattr(candidate, name), protocol=5), name)
                    for first, second in pairs:
                        self.assertEqual(first.optimizer_backend_, 'python')
                        self.assertEqual(second.optimizer_backend_, 'native')
                        for name in ('initial_state_', 'final_state_', 'selected_state_', 'rng_state_',
                                     'n_updates_', 'n_samples_seen_', 'learning_rate_B_', 'model_status_',
                                     'stop_reason_', 'fit_usable_', 'cv_eligible_'):
                            self.assertEqual(pickle.dumps(getattr(first, name), protocol=5),
                                             pickle.dumps(getattr(second, name), protocol=5), name)
                    self.exact(reference.decision_function(X), candidate.decision_function(X))
                    self.exact(reference.predict(X), candidate.predict(X))
                    self.assertEqual(pickle.dumps(rng, protocol=5), external)


if __name__ == '__main__':
    unittest.main()
