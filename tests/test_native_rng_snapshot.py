"""Independent epoch RNG states preserve exact rollback and fallback copying."""
import copy
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from two_stage_sgd_dwd import KernelSGDDWD, NumericalFailure, _estimators, _native_optimizer
from test_kernel_cache import ExactAssertions

try:
    from two_stage_sgd_dwd import _sgd_accel
except (ImportError, OSError):
    _sgd_accel = None

BITS = (np.random.PCG64, np.random.PCG64DXSM, np.random.Philox,
        np.random.SFC64, np.random.MT19937)
requires_native = unittest.skipIf(_sgd_accel is None, "Optional compiled extension unavailable")


def state_bytes(value):
    return pickle.dumps(value, protocol=5)


class GeneratorSubclass(np.random.Generator):
    pass


class ObservedPCG64(np.random.PCG64):
    @property
    def state(self):
        value = np.random.PCG64.state.__get__(self)
        self.observed.append(value)
        return value

    @state.setter
    def state(self, value):
        np.random.PCG64.state.__set__(self, value)


class EpochStateTests(ExactAssertions):
    def assert_independent_storage(self, first, second):
        if isinstance(first, dict):
            self.assertIsNot(first, second)
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_independent_storage(first[key], second[key])
        elif isinstance(first, np.ndarray):
            self.assertIsNot(first, second)
            self.assertFalse(np.shares_memory(first, second))

    def mutate_snapshot(self, value):
        for key, item in value.items():
            if isinstance(item, dict):
                self.mutate_snapshot(item)
            elif isinstance(item, np.ndarray):
                item[...] = 0
            elif isinstance(item, (int, np.integer)):
                value[key] = 0

    @unittest.skipUnless(np.__version__ == "2.5.2", "Getter proof is specific to NumPy 2.5.2")
    def test_builtin_getters_own_nested_state_and_arrays(self):
        for bit_type in BITS:
            with self.subTest(bit=bit_type.__name__):
                rng = np.random.Generator(bit_type(195))
                rng.integers(17, size=3)
                rng.random(5)
                rng.bit_generator.random_raw(2)
                first, second = rng.bit_generator.state, rng.bit_generator.state
                self.assertExact(first, second)
                self.assert_independent_storage(first, second)
                first_before, second_before = state_bytes(first), state_bytes(second)
                rng.bit_generator.random_raw(31)
                rng.integers(57, size=9)
                self.assertEqual(state_bytes(first), first_before)
                current = state_bytes(rng.bit_generator.state)
                self.mutate_snapshot(first)
                self.assertEqual(state_bytes(second), second_before)
                self.assertEqual(state_bytes(rng.bit_generator.state), current)
                rng.bit_generator.state = second
                self.assertExact(rng.bit_generator.state, second)
                rng.bit_generator.random_raw(13)
                self.assertEqual(state_bytes(second), second_before)
                current = state_bytes(rng.bit_generator.state)
                self.mutate_snapshot(second)
                self.assertEqual(state_bytes(rng.bit_generator.state), current)

    def run_copy_probe(self, rng, version, expected_epoch_copies):
        original_copy = copy.deepcopy
        X, y = np.array([[.2, .3], [-.4, .5], [.6, -.7]]), np.array([-1., 1., -1.])
        requests = []

        def run_steps(kind, coefficient, bias, rows, labels, indices, rates, lambd):
            requests.append(indices.copy())
            return coefficient, bias, len(indices), None

        # Isolate epoch copies from the independently required initial/final
        # model snapshots. The final returned rng_state_ still adds one copy.
        with patch.object(_estimators, "_snapshot", return_value={"snapshot": True}), \
                patch.object(np, "__version__", version), \
                patch.object(_native_optimizer.copy, "deepcopy", wraps=original_copy) as copier:
            _native_optimizer.optimize("linear", X, y, rng,
                SimpleNamespace(max_epochs=3, sampling="with_replacement"),
                SimpleNamespace(run_steps=run_steps))
        state_calls = [call.args[0] for call in copier.call_args_list
                       if call.args and isinstance(call.args[0], dict)
                       and "bit_generator" in call.args[0]]
        self.assertEqual(len(state_calls), expected_epoch_copies + 1)
        self.assertEqual(len(requests), 3)

    def test_other_versions_and_generator_subclasses_keep_each_epoch_copy(self):
        for version in ("2.5.1", "2.5.3", "2.5.2+custom"):
            with self.subTest(version=version):
                self.run_copy_probe(np.random.Generator(np.random.PCG64(44)), version, 3)
        self.run_copy_probe(GeneratorSubclass(np.random.PCG64(44)), "2.5.2", 3)

    def test_bit_generator_subclass_keeps_copies_and_reads_state_each_epoch(self):
        bit = ObservedPCG64(44)
        bit.observed = []
        self.run_copy_probe(np.random.Generator(bit), "2.5.2", 3)
        self.assertEqual(len(bit.observed), 4)  # Three epochs plus returned state.
        self.assertEqual(len({id(state) for state in bit.observed}), 4)
        self.assertEqual(len({state_bytes(state) for state in bit.observed}), 4)

    def test_mutable_custom_getter_cannot_overwrite_the_rollback_snapshot(self):
        class SharedStatePCG64(np.random.PCG64):
            @property
            def state(self):
                fresh = np.random.PCG64.state.__get__(self)
                if not hasattr(self, "shared"):
                    self.shared = {}
                self.shared.clear()
                self.shared.update(fresh)
                return self.shared

            @state.setter
            def state(self, value):
                np.random.PCG64.state.__set__(self, value)

        rng = np.random.Generator(SharedStatePCG64(42))
        expected = np.random.Generator(SharedStatePCG64(42))
        expected.integers(3)  # The failed first observation still consumes a draw.

        def fail(kind, coefficient, bias, rows, labels, indices, rates, lambd):
            rng.bit_generator.state  # Refresh the shared dict after all epoch draws.
            return coefficient, bias, 0, "controlled native failure"

        result = _native_optimizer.optimize("linear", np.ones((3, 2)), np.array([-1., 1., -1.]),
            rng, SimpleNamespace(max_epochs=2, sampling="with_replacement"),
            SimpleNamespace(run_steps=fail))
        self.assertEqual(result["n_updates_"], 0)
        self.assertExact(result["rng_state_"], expected.bit_generator.state)
        self.assertExact(rng.integers(3, size=11), expected.integers(3, size=11))

    @unittest.skipUnless(np.__version__ == "2.5.2", "Optimized getter path is version-specific")
    def test_all_supported_builtin_states_skip_only_epoch_deepcopies(self):
        for bit_type in BITS:
            with self.subTest(bit=bit_type.__name__):
                self.run_copy_probe(np.random.Generator(bit_type(44)), "2.5.2", 0)


@requires_native
class EpochRollbackTests(ExactAssertions):
    def test_later_epoch_failure_matches_python_state_and_continuation(self):
        X = np.array([[-1., .1], [-.7, .2], [-.3, -.1], [.3, .1], [.7, -.2], [1., .3]])
        y = np.array([-1, -1, -1, 1, 1, 1])
        m = len(y)
        original_column = _estimators._kernel_values
        for bit_type in BITS:
            for sampling in ("with_replacement", "shuffle"):
                # First/later prefixes are covered broadly in native_sampling;
                # this focused case additionally checks continuation draws.
                for failed_call in (m + 3,):
                    with self.subTest(bit=bit_type.__name__, sampling=sampling, failure=failed_call):
                        external_rng = np.random.Generator(bit_type(173))
                        external_rng.integers(17)
                        external_before = state_bytes(external_rng)
                        models = []
                        for backend in ("python", "native"):
                            calls = [0]

                            def fail_column(*args):
                                calls[0] += 1
                                if calls[0] == failed_call:
                                    raise FloatingPointError("controlled epoch rollback failure")
                                return original_column(*args)

                            model = KernelSGDDWD(max_epochs=3, random_state=external_rng,
                                sampling=sampling, optimizer_backend=backend, kernel_cache_bytes=0)
                            with patch.object(_estimators, "_kernel_values", side_effect=fail_column):
                                with self.assertRaises(NumericalFailure):
                                    model.fit(X, y)
                            self.assertEqual(calls[0], failed_call)
                            self.assertEqual(model.n_updates_, failed_call - 1)
                            self.assertEqual(model.error_, "controlled epoch rollback failure")
                            self.assertFalse(model.fit_usable_)
                            self.assertEqual(state_bytes(external_rng), external_before)
                            models.append(model)
                        self.assertSameFit(*models)
                        expected = np.random.Generator(bit_type(0))
                        expected.bit_generator.state = copy.deepcopy(models[0].initial_state_["rng_state"])
                        if sampling == "shuffle":
                            for _ in range((failed_call - 1) // m + 1):
                                expected.permutation(m)
                        else:
                            for _ in range(failed_call):
                                expected.integers(m)
                        expected_state = expected.bit_generator.state
                        continuation = expected.integers(m, size=11)
                        raw_continuation = expected.bit_generator.random_raw(5)
                        for model in models:
                            self.assertExact(model.rng_state_, expected_state)
                            resumed = np.random.Generator(bit_type(0))
                            resumed.bit_generator.state = copy.deepcopy(model.rng_state_)
                            self.assertExact(resumed.integers(m, size=11), continuation)
                            self.assertExact(resumed.bit_generator.random_raw(5), raw_continuation)


if __name__ == "__main__":
    unittest.main()
