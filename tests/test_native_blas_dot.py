"""Optional native BLAS dispatch retains array hooks and NumPy error policy."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import warnings

import numpy as np
import pytest


@pytest.fixture
def native():
    try:
        module = importlib.import_module('two_stage_sgd_dwd._sgd_accel')
    except (ImportError, OSError):
        pytest.skip('Optional native extension unavailable')
    if not hasattr(module, '_uses_scipy_ddot'):
        pytest.skip('Legacy native extension has no direct BLAS adapter')
    return module


def fresh_process(native, scenario):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[name] = '1'
    run = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                          scenario, native.__file__], capture_output=True, text=True,
                         encoding='utf-8', env=env, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads(run.stdout)
    if 'skip' in result:
        pytest.skip(result['skip'])
    return result


@pytest.mark.parametrize('scenario', ['numpy_hook', 'wrapped_aliases',
                                     'missing_capsule', 'unknown_capsule'])
def test_fresh_import_preserves_captured_hooks_and_optional_fallback(native, scenario):
    result = fresh_process(native, scenario)
    assert result['uses_scipy_ddot'] is False
    assert result['hook_calls'] == (2 if scenario in ('numpy_hook', 'wrapped_aliases') else 0)
    assert result['completed'] == 1


@pytest.mark.parametrize('scenario', ['memory_error', 'interrupt'])
def test_initialization_does_not_swallow_lookup_failures(native, scenario):
    if not native._uses_scipy_ddot:
        pytest.skip('This environment does not expose the recognized BLAS interface')
    result = fresh_process(native, scenario)
    assert result['exception'] == ('MemoryError' if scenario == 'memory_error' else 'KeyboardInterrupt')


def test_array_subclass_keeps_dot_dispatch_for_margin_and_projection(native):
    calls = []

    class Hooked(np.ndarray):
        def __array_function__(self, function, array_types, args, kwargs):
            if function is np.copy:
                return np.array(np.asarray(args[0]), copy=True).view(type(self))
            if function is np.empty_like:
                return np.empty_like(np.asarray(args[0])).view(type(self))
            if function is np.dot:
                calls.append(args[0] is args[1])
                return np.dot(np.asarray(args[0]), np.asarray(args[1]), **kwargs)
            return super().__array_function__(function, array_types, args, kwargs)

    initial = np.zeros(3).view(Hooked)
    row = np.array([1., -2., .5]).view(Hooked)
    coefficients, bias, completed, error = native.run_steps(
        0, initial, 0., (row,), np.ones(1), np.array([0], dtype=np.int64), np.array([.01]), .1)
    assert calls == [False, True]
    assert completed == 1 and error is None and bias == .01
    np.testing.assert_array_equal(coefficients, np.array([.01, -.02, .005]))


def test_extreme_dots_preserve_numpy_policies_and_committed_prefix(native):
    if not native._uses_scipy_ddot:
        pytest.skip('This environment does not expose the recognized BLAS interface')
    direct = fresh_process(native, 'policies_direct')
    fallback = fresh_process(native, 'policies_fallback')
    assert direct['uses_scipy_ddot'] is True
    assert fallback['uses_scipy_ddot'] is False
    assert direct['observations'] == fallback['observations']
    underflow = fallback['observations']['kernel/underflow/raise']
    # Some numerical libraries suppress underflow flags. When NumPy reports
    # underflow, the second sampled update must not be committed by either path.
    if underflow.get('error') and 'underflow' in underflow['error']:
        assert underflow['completed'] == 1


def load_extension(path):
    spec = importlib.util.spec_from_file_location('_sgd_accel', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observe_policy(native, geometry, kind, mode):
    initial, row = {
        'overflow': ([1e308, 1e308], [1e308, 1e308]),
        'cancellation': ([1e308, -1e308], [1e308, 1e308]),
        'underflow': ([1e-200, -1e-200], [1e-200, 1e-200]),
    }[geometry]
    initial, row = np.array(initial), np.array(row)
    if kind == 0:
        args = (0, initial, 0., (row,), np.ones(1),
                np.array([0], dtype=np.int64), np.array([.01]), .1)
    else:
        rows = (np.zeros(2), row)
        args = (1, initial, 0., lambda index: rows[index], np.ones(2),
                np.array([0, 1], dtype=np.int64), np.array([0., .01]), .1)
    before = (initial.tobytes(), row.tobytes())
    calls, logs, result = [], [], {}

    class Logger:
        def write(self, message):
            logs.append(message)

    previous = np.geterrcall()
    np.seterrcall(Logger() if mode == 'log' else lambda error, flag: calls.append([error, int(flag)]))
    try:
        with warnings.catch_warnings(record=True) as seen, np.errstate(all=mode):
            warnings.simplefilter('always')
            try:
                coefficients, bias, completed, error = native.run_steps(*args)
                result.update(coefficients=coefficients.tobytes().hex(), bias=float(bias).hex(),
                              completed=completed, error=error)
            except Exception as exc:
                result['exception'] = [type(exc).__name__, str(exc)]
            result['warnings'] = [[item.category.__name__, str(item.message)] for item in seen]
    finally:
        np.seterrcall(previous)
    result.update(callbacks=calls, logs=logs,
                  inputs_unchanged=before == (initial.tobytes(), row.tobytes()))
    assert result['inputs_unchanged']
    return result


def child(scenario, path):
    original_dot, calls = np.dot, []
    if scenario in ('numpy_hook', 'wrapped_aliases'):
        def hook(*args, **kwargs):
            calls.append(True)
            return original_dot(*args, **kwargs)

        if scenario == 'wrapped_aliases':
            try:
                from numpy._core import multiarray, _multiarray_umath
            except ImportError:
                return {'skip': 'NumPy dispatcher inspection unavailable'}
            hook.__wrapped__ = _multiarray_umath.dot
            multiarray.dot = hook
        np.dot = hook
    elif scenario in ('missing_capsule', 'unknown_capsule', 'policies_fallback'):
        try:
            from scipy.linalg import cython_blas
            api = cython_blas.__pyx_capi__
        except (ImportError, AttributeError, OSError):
            return {'skip': 'Optional SciPy capsule interface unavailable'}
        if scenario == 'unknown_capsule':
            if 'dasum' not in api:
                return {'skip': 'Alternative capsule signature unavailable'}
            api['ddot'] = api['dasum']
        else:
            api.pop('ddot', None)
    elif scenario in ('memory_error', 'interrupt'):
        try:
            from numpy._core import _multiarray_umath
            dispatcher = _multiarray_umath._ArrayFunctionDispatcher
        except (ImportError, AttributeError):
            return {'skip': 'NumPy dispatcher inspection unavailable'}
        exception = MemoryError if scenario == 'memory_error' else KeyboardInterrupt
        probe = types.ModuleType('numpy._core._multiarray_umath')
        probe._ArrayFunctionDispatcher = dispatcher

        def lookup(name):
            if name == 'dot':
                raise exception('preserve the first lookup failure')
            raise AttributeError(name)

        probe.__getattr__ = lookup
        sys.modules['numpy._core._multiarray_umath'] = probe
        try:
            load_extension(path)
        except BaseException as exc:
            assert type(exc) is exception and str(exc) == 'preserve the first lookup failure'
            return {'exception': type(exc).__name__}
        raise AssertionError('Initialization swallowed the lookup failure')
    module = load_extension(path)
    enabled = bool(module._uses_scipy_ddot)
    if scenario.startswith('policies_'):
        if scenario == 'policies_direct' and not enabled:
            return {'skip': 'Recognized direct BLAS interface unavailable in the child process'}
        result = {'uses_scipy_ddot': enabled, 'observations': {}}
        for kind in (0, 1):
            for geometry in ('overflow', 'cancellation', 'underflow'):
                for mode in ('ignore', 'warn', 'raise', 'call', 'log'):
                    key = f"{'linear' if kind == 0 else 'kernel'}/{geometry}/{mode}"
                    result['observations'][key] = observe_policy(module, geometry, kind, mode)
        return result
    assert not enabled
    coefficients, bias, completed, error = module.run_steps(
        0, np.zeros(3), 0., (np.array([1., -2., .5]),), np.ones(1),
        np.array([0], dtype=np.int64), np.array([.01]), .1)
    assert error is None and bias == .01
    assert coefficients.tobytes() == np.array([.01, -.02, .005]).tobytes()
    return {'uses_scipy_ddot': enabled, 'hook_calls': len(calls), 'completed': completed}


if __name__ == '__main__':
    print(json.dumps(child(sys.argv[1], sys.argv[2]), sort_keys=True))
