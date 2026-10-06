"""Explicit training-only MNIST tuning with saved folds and adaptive grids.

Pixels are divided by 255, as in the Chapter 3 procedure. This example never
opens the official test partition. Search results describe the requested SGD
budget; use larger budgets and independent seeds to assess their stability.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import joblib
from sklearn.model_selection import ParameterGrid, StratifiedKFold
import threadpoolctl
from threadpoolctl import threadpool_limits

import two_stage_sgd_dwd
from two_stage_sgd_dwd import mnist_param_grid
from two_stage_sgd_dwd.datasets import load_mnist_pair
from _adaptive_grid import (AdaptiveSearchConfig, candidate_key, next_search_stage,
                            ordered_eligible, POLICY_STATE_SCHEMA_VERSION)
from _mnist_cv import FoldEvaluator, execution_identity


_CONFIG_SCHEMA_VERSION = 8
_PLAN_SCHEMA_VERSION = 3
_RECEIPT_SCHEMA_VERSION = 1
_FOLD_SCHEMA_VERSION = 1
_fold_parameters = FoldEvaluator._parameters
_CONTROL_STAGE = 'inherited-default-control'
_CONTROL_PARAMS = {'kernel': 'rbf', 'lambd': 1e-6, 'gamma': 'scale', 'retention_fraction': .3}


def _io_path(path):
    """Use extended Windows paths for I/O, retaining explicit namespace paths."""
    path = Path(path)
    if os.name != 'nt':
        return path
    value = os.fspath(path)
    if value.startswith(('\\\\?\\', '\\\\.\\')):
        return path
    absolute = os.path.abspath(value)
    return Path('\\\\?\\UNC\\' + absolute[2:] if absolute.startswith('\\\\')
                else '\\\\?\\' + absolute)


class SearchBusyError(RuntimeError):
    """The search directory already has a coordinator, or cannot be locked."""


class _RunLease:
    """A non-inherited OS lease on a stable lockfile that is never removed."""
    def __init__(self, path):
        self.path, self.stream = Path(path), None

    def __enter__(self):
        descriptor = None
        try:
            descriptor = os.open(_io_path(self.path), os.O_RDWR | os.O_CREAT, 0o600)
            os.set_inheritable(descriptor, False)
            self.stream = os.fdopen(descriptor, 'r+b', buffering=0)
            descriptor = None
            self.stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if os.fstat(self.stream.fileno()).st_size == 0:
                self.stream.write(b'\0')
        except BaseException as error:
            if self.stream is not None:
                self.stream.close()
                self.stream = None
            elif descriptor is not None:
                os.close(descriptor)
            if isinstance(error, OSError):
                raise SearchBusyError('Search directory is busy or inaccessible: '
                                      + str(self.path.parent)) from error
            raise
        return self

    def __exit__(self, *args):
        if self.stream is not None:
            # Closing releases the OS lock, including during exception unwinding.
            self.stream.close()
            self.stream = None


def _validate_records(records, *, expected_folds, execution_identity,
                      allow_blas_change=False, threads=None):
    """Validate completed candidate evidence, including legitimate failures."""
    attempted = set()
    for number, record in enumerate(records):
        prefix = f'Invalid candidate record {number}: '
        required = {'params', 'eligible', 'mean_score', 'folds', 'execution_identity'}
        if not isinstance(record, dict) or not required <= record.keys():
            raise ValueError(prefix + 'missing candidate fields')
        params = record['params']
        if not isinstance(params, dict) or set(params) - {
                'kernel', 'lambd', 'gamma', 'retention_fraction'}:
            raise ValueError(prefix + 'invalid candidate parameters')
        key = candidate_key(params)
        if key in attempted:
            raise ValueError('Saved search contains duplicate candidates')
        attempted.add(key)
        if record['execution_identity'] != execution_identity and not allow_blas_change:
            raise ValueError(prefix + 'execution identity differs from the saved run; '
                             'mixed runs require --allow-blas-change')
        if not isinstance(record['eligible'], bool):
            raise ValueError(prefix + 'eligible must be boolean')
        folds = record['folds']
        if not isinstance(folds, list) or len(folds) != expected_folds:
            raise ValueError(prefix + 'incomplete fold evidence')
        indices, scores, eligibility = [], [], []
        for fold in folds:
            if (not isinstance(fold, dict)
                    or not {'fold', 'eligible', 'accuracy', 'status'} <= fold.keys()):
                raise ValueError(prefix + 'missing fold fields')
            index, eligible, score = fold['fold'], fold['eligible'], fold['accuracy']
            if isinstance(index, bool) or not isinstance(index, numbers.Integral):
                raise ValueError(prefix + 'invalid fold index')
            indices.append(index)
            if not isinstance(eligible, bool):
                raise ValueError(prefix + 'fold eligibility must be boolean')
            eligibility.append(eligible)
            if eligible:
                if (fold['status'] != 'trained' or isinstance(score, bool)
                        or not isinstance(score, numbers.Real) or not math.isfinite(score)
                        or not 0 <= score <= 1):
                    raise ValueError(prefix + 'invalid eligible fold accuracy or status')
                scores.append(float(score))
            elif (score is not None or not isinstance(fold['status'], str)
                  or not fold['status'] or fold['status'] == 'trained'):
                raise ValueError(prefix + 'invalid failed fold accuracy or status')
        if sorted(indices) != list(range(expected_folds)):
            raise ValueError(prefix + 'fold indices must be unique and consecutive')
        if record['eligible'] != all(eligibility):
            raise ValueError(prefix + 'candidate eligibility disagrees with folds')
        mean = record['mean_score']
        if record['eligible']:
            if (isinstance(mean, bool) or not isinstance(mean, numbers.Real)
                    or not math.isfinite(mean) or not 0 <= mean <= 1
                    or not math.isclose(float(mean), math.fsum(scores) / expected_folds,
                                        rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError(prefix + 'mean score disagrees with eligible folds')
        elif mean is not None:
            raise ValueError(prefix + 'failed candidates must have a null mean score')
    _runtime_history(records, execution_identity, threads=threads)
    return attempted


def _verify_config_schema(config):
    if config.get('schema_version') != _CONFIG_SCHEMA_VERSION:
        raise ValueError('Unsupported search configuration schema; start a fresh output directory '
                         'for the current parameter, runtime and candidate receipt identities.')


def _content_sha256(value):
    """Hash JSON content independently of indentation or dictionary key order."""
    content = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(content).hexdigest()


def _unique_json_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError('Duplicate JSON field: ' + name)
        result[name] = value
    return result


def _read_json(path):
    return json.loads(_io_path(path).read_text(encoding='utf-8'),
                      object_pairs_hook=_unique_json_object)


def _run_sha256(config, splits):
    return _content_sha256({'config': config, 'splits': splits})


def _check_blas_change(previous, current, *, threads):
    """Allow a documented numerical-build switch, not an arbitrary runtime edit."""
    message = 'Incompatible BLAS change: Python, dependencies, optimizer or numerical controls differ'
    if (not isinstance(previous, dict) or not isinstance(current, dict)
            or {key: value for key, value in previous.items() if key != 'numerical_runtime'} !=
            {key: value for key, value in current.items() if key != 'numerical_runtime'}):
        raise ValueError(message)
    old, new = previous.get('numerical_runtime'), current.get('numerical_runtime')
    fields = {'schema_version', 'python', 'implementation', 'system', 'machine',
              'packages', 'binaries', 'threadpools'}
    if (not isinstance(old, dict) or not isinstance(new, dict)
            or set(old) != fields or set(new) != fields
            or type(old['schema_version']) is not int or old['schema_version'] != 1
            or type(new['schema_version']) is not int or new['schema_version'] != 1
            or any(old[name] != new[name] for name in fields - {'packages', 'binaries', 'threadpools'})):
        raise ValueError(message)

    def digest(value):
        return isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef')

    def packages(runtime):
        entries = runtime['packages']
        if not isinstance(entries, dict) or set(entries) != {'numpy', 'scipy', 'sklearn'}:
            raise ValueError(message)
        result = {}
        for name, package in entries.items():
            if (not isinstance(package, dict) or set(package) != {'version', 'filename', 'sha256'}
                    or not digest(package['sha256'])):
                raise ValueError(message)
            result[name] = {key: value for key, value in package.items() if key != 'sha256'}
        return result

    def binaries(runtime):
        result = []
        modules = set()
        if not isinstance(runtime['binaries'], list) or not runtime['binaries']:
            raise ValueError(message)
        for binary in runtime['binaries']:
            if (not isinstance(binary, dict) or set(binary) != {'module', 'filename', 'sha256'}
                    or not isinstance(binary['module'], str) or binary['module'] in modules
                    or not digest(binary['sha256'])
                    or binary['module'].split('.', 1)[0] not in ('numpy', 'scipy', 'sklearn')):
                raise ValueError(message)
            modules.add(binary['module'])
            # Same-version vendor builds can differ throughout the numerical
            # stack. Keep its complete module/ABI inventory, not binary equality.
            result.append({key: value for key, value in binary.items() if key != 'sha256'})
        return result

    if packages(old) != packages(new) or binaries(old) != binaries(new):
        raise ValueError(message)
    for runtime in (old, new):
        pools = runtime['threadpools']
        if (not isinstance(pools, list) or not pools or any(not isinstance(pool, dict) for pool in pools)
                or not any(pool.get('user_api') == 'blas' for pool in pools)
                or any(pool.get('user_api') not in ('blas', 'openmp')
                       or (pool['user_api'] == 'blas' and pool.get('internal_api') not in ('blis', 'mkl', 'openblas'))
                       or (pool['user_api'] == 'openmp' and pool.get('internal_api') != 'openmp')
                       or not isinstance(pool.get('filename'), str) or not pool['filename']
                       or not digest(pool.get('sha256'))
                       or type(pool.get('num_threads')) is not int
                       or pool['num_threads'] != threads for pool in pools)):
            raise ValueError(message)
    library_keys = lambda runtime: sorted((pool['user_api'], pool['internal_api'], pool['filename'],
                                           pool['sha256']) for pool in runtime['threadpools'])
    if library_keys(old) == library_keys(new):
        raise ValueError('Incompatible BLAS change: loaded BLAS/OpenMP libraries are unchanged')


def _authorize_runtime(previous, current, *, allow_blas_change, threads):
    if previous == current:
        return
    if not allow_blas_change:
        raise ValueError('Execution identity differs; a compatible switch requires --allow-blas-change')
    _check_blas_change(previous, current, threads=threads)


def _runtime_transition(previous, current):
    return {'schema_version': 1, 'reason': 'explicit_blas_change',
            'from_identity_sha256': _content_sha256(previous),
            'to_identity_sha256': _content_sha256(current)}


def _runtime_history(records, initial_identity, *, threads):
    history = [{'candidate_sequence': None, 'execution_identity': initial_identity}]
    previous = initial_identity
    for sequence, record in enumerate(records):
        current = record['execution_identity']
        transition = record.get('runtime_transition')
        if current == previous:
            if 'runtime_transition' in record:
                raise ValueError('Unexpected runtime transition without an execution identity change')
        else:
            if (not isinstance(transition, dict) or type(transition.get('schema_version')) is not int
                    or transition != _runtime_transition(previous, current)):
                raise ValueError('Missing or invalid runtime transition')
            _check_blas_change(previous, current, threads=threads)
            history.append({'candidate_sequence': sequence, 'execution_identity': current})
        previous = current
    return history


def _runtime_summary(records, initial_identity, *, threads):
    history = _runtime_history(records, initial_identity, threads=threads)
    ranked = ordered_eligible(records)
    return {'runtime_history': history, 'mixed_runtime': len(history) > 1,
            'selected_execution_identity': ranked[0]['execution_identity'] if ranked else None,
            'timings_require_same_runtime': True}


def _consumer_context(evidence, identity):
    return {**evidence, 'consumer_execution_identity': identity,
            'consumer_runtime_differs_from_selected': identity != evidence['selected_execution_identity']}


def _runtime_versions():
    import scipy
    import sklearn
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'scikit_learn': sklearn.__version__,
            'joblib': joblib.__version__, 'threadpoolctl': threadpoolctl.__version__}


def _seal_record(record, *, run_sha256, sequence, previous_sha256):
    """Attach a receipt in the same append as the candidate it protects."""
    if '_receipt' in record:
        raise ValueError('Evaluator records must not supply a candidate receipt')
    receipt = {'schema_version': _RECEIPT_SCHEMA_VERSION,
               'run_sha256': run_sha256, 'sequence': sequence,
               'candidate_key': list(candidate_key(record['params'])),
               'previous_sha256': previous_sha256}
    sealed = {**record, '_receipt': receipt}
    digest = _content_sha256(sealed)
    receipt['sha256'] = digest
    return sealed


def _verify_receipts(records, *, run_sha256):
    """Verify the complete committed prefix before any recovery writes."""
    previous = run_sha256
    required = {'schema_version', 'run_sha256', 'sequence', 'candidate_key',
                'previous_sha256', 'sha256'}
    for sequence, record in enumerate(records):
        receipt = record.get('_receipt')
        if (not isinstance(receipt, dict) or set(receipt) != required
                or type(receipt['schema_version']) is not int
                or receipt['schema_version'] != _RECEIPT_SCHEMA_VERSION
                or type(receipt['sequence']) is not int
                or receipt['sequence'] != sequence
                or receipt['run_sha256'] != run_sha256
                or receipt['candidate_key'] != list(candidate_key(record['params']))
                or receipt['previous_sha256'] != previous):
            raise ValueError(f'Invalid candidate receipt at record {sequence}')
        metadata = {key: value for key, value in receipt.items() if key != 'sha256'}
        if _content_sha256({**record, '_receipt': metadata}) != receipt['sha256']:
            raise ValueError(f'Candidate receipt content differs at record {sequence}')
        previous = receipt['sha256']


def write_json(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    path = _io_path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    stream = temporary.open('w', encoding='utf-8')
    try:
        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        # Opening succeeded, so this attempt owns the temporary file. A
        # cleanup failure must not replace the original write/interruption.
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _validate_fold_prefix(folds, expected_folds):
    if not isinstance(folds, list) or len(folds) > expected_folds:
        raise ValueError('Invalid saved fold prefix length')
    for index, fold in enumerate(folds):
        if (not isinstance(fold, dict) or type(fold.get('fold')) is not int
                or fold['fold'] != index or type(fold.get('eligible')) is not bool):
            raise ValueError('Invalid saved fold order or eligibility')
        score = fold.get('accuracy')
        if fold['eligible']:
            if (fold.get('status') != 'trained' or isinstance(score, bool)
                    or not isinstance(score, numbers.Real) or not math.isfinite(score)
                    or not 0 <= score <= 1):
                raise ValueError('Invalid saved eligible fold')
        elif fold.get('status') != 'single_class_selection' or score is not None:
            raise ValueError('Invalid saved failed fold')
    _content_sha256(folds)  # Reject nonfinite or non-JSON auxiliary evidence too.


class _FoldCheckpoint:
    """One active candidate's durable completed folds, under the run lease."""
    def __init__(self, directory, *, run_sha256, expected_folds, execution_identity,
                 allow_blas_change, threads, records, stages, control=None):
        self.path = _io_path(directory) / 'fold-progress.json'
        self.run_sha256, self.expected_folds = run_sha256, expected_folds
        self.identity = deepcopy(execution_identity)
        self.allow_blas_change, self.threads = allow_blas_change, threads
        self.sequence = len(records)
        self.previous_sha256 = records[-1]['_receipt']['sha256'] if records else run_sha256
        self.active = None
        if not self.path.exists():
            return
        saved = _read_json(self.path)
        required = {'schema_version', 'run_sha256', 'stage', 'params', 'candidate_key',
                    'execution_identity', 'expected_folds', 'candidate_sequence',
                    'previous_sha256', 'folds', 'sha256'}
        if (not isinstance(saved, dict) or set(saved) != required
                or type(saved['schema_version']) is not int
                or saved['schema_version'] != _FOLD_SCHEMA_VERSION
                or saved['run_sha256'] != run_sha256
                or type(saved['expected_folds']) is not int
                or saved['expected_folds'] != expected_folds
                or type(saved['candidate_sequence']) is not int
                or not 0 <= saved['candidate_sequence'] <= len(records)
                or not isinstance(saved['execution_identity'], dict)
                or saved['params'] != _fold_parameters(saved['params'])
                or saved['candidate_key'] != _content_sha256(saved['params'])
                or saved['sha256'] != _content_sha256({key: value for key, value in saved.items()
                                                       if key != 'sha256'})):
            raise ValueError('Invalid fold-progress receipt or run identity')
        _validate_fold_prefix(saved['folds'], expected_folds)
        sequence = saved['candidate_sequence']
        previous = records[sequence - 1]['_receipt']['sha256'] if sequence else run_sha256
        if saved['previous_sha256'] != previous:
            raise ValueError('Fold progress differs from the committed candidate history')
        if saved['stage'] == _CONTROL_STAGE:
            if saved['params'] != _CONTROL_PARAMS or sequence != 0:
                raise ValueError('Invalid inherited-control fold progress')
            committed = control
        else:
            stage = next((item for item in stages if item['name'] == saved['stage']), None)
            if stage is None or saved['params'] not in [_fold_parameters(p) for p in stage['candidates']]:
                raise ValueError('Fold progress does not belong to a saved candidate stage')
            committed = records[sequence] if sequence < len(records) else None
            if committed is not None and (len(records) != sequence + 1
                    or committed.get('stage') != saved['stage']):
                raise ValueError('Stale fold progress disagrees with completed candidate history')
            if committed is None:
                attempted = {candidate_key(record['params']) for record in records}
                pending = [_fold_parameters(p) for p in stage['candidates']
                           if candidate_key(p) not in attempted]
                if stage.get('completed') is not False or not pending or saved['params'] != pending[0]:
                    raise ValueError('Fold progress is not the next pending stage candidate')
        self.active = saved
        if committed is not None:
            self.complete(committed)
        elif sequence != len(records):
            raise ValueError('Fold progress is not the next uncommitted candidate')
        elif saved['folds']:
            _authorize_runtime(saved['execution_identity'], self.identity,
                               allow_blas_change=self.allow_blas_change, threads=self.threads)

    def _publish(self, payload):
        saved = deepcopy(payload)
        saved.pop('sha256', None)
        saved['sha256'] = _content_sha256(saved)
        write_json(self.path, saved)
        self.active = saved

    def begin(self, stage, params):
        params = _fold_parameters(params)
        if self.active is not None:
            if self.active['stage'] != stage or self.active['params'] != params:
                raise ValueError('Pending fold progress belongs to a different candidate')
            if self.active['execution_identity'] != self.identity:
                if self.active['folds']:
                    _authorize_runtime(self.active['execution_identity'], self.identity,
                                       allow_blas_change=self.allow_blas_change, threads=self.threads)
                # An empty marker contains no fitted evidence. The run-level
                # configuration and committed records already authorize the
                # current runtime; a failed attempt need not add a transition.
                reason = ('numerical_runtime_changed' if self.active['folds'] else
                          'no_completed_folds_after_runtime_change')
                evidence = {'reason': reason, 'checkpoint': self.active,
                            'replacement_execution_identity': self.identity}
                archive = self.path.with_name('fold-progress-abandoned-' + _content_sha256(evidence) + '.json')
                if archive.exists():
                    if _read_json(archive) != evidence:
                        raise ValueError('Abandoned fold-progress archive differs')
                else:
                    write_json(archive, evidence)
                self.path.unlink()
                self.active = None
        if self.active is None:
            self._publish({'schema_version': _FOLD_SCHEMA_VERSION, 'run_sha256': self.run_sha256,
                'stage': stage, 'params': params, 'candidate_key': _content_sha256(params),
                'execution_identity': self.identity, 'expected_folds': self.expected_folds,
                'candidate_sequence': self.sequence, 'previous_sha256': self.previous_sha256,
                'folds': []})
        return deepcopy(self.active['folds'])

    def save(self, fold):
        if self.active is None:
            raise ValueError('No active candidate for completed fold')
        saved = deepcopy(self.active)
        saved['folds'].append(deepcopy(fold))
        _validate_fold_prefix(saved['folds'], self.expected_folds)
        self._publish(saved)

    def verify(self, record):
        if (self.active is None or len(self.active['folds']) != self.expected_folds
                or self.active['params'] != _fold_parameters(record['params'])
                or self.active['execution_identity'] != record.get('execution_identity')
                or self.active['folds'] != record.get('folds')):
            raise ValueError('Completed candidate differs from its fold-progress receipt')

    def complete(self, record):
        self.verify(record)
        self.path.unlink()
        self.active = None
        if '_receipt' in record:
            self.sequence = record['_receipt']['sequence'] + 1
            self.previous_sha256 = record['_receipt']['sha256']


def _write_failure_json(path, value):
    """Report a failed evaluation without replacing its original exception."""
    try:
        write_json(path, value)
    except Exception as logging_error:
        try:
            warnings.warn(f'Could not write {path.name}: {logging_error}',
                          RuntimeWarning, stacklevel=2)
        except Exception:
            # A warnings-as-errors policy must preserve the evaluation error.
            pass


def balanced_indices(y, count, seed):
    if count < 4 or count % 2:
        raise ValueError('train-size must be an even integer of at least four')
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(y):
        available = np.flatnonzero(y == label)
        if count // 2 > len(available):
            raise ValueError('train-size exceeds the available class counts')
        selected.extend(rng.permutation(available)[:count // 2].tolist())
    return np.sort(selected)


def read_records(path, *, expected_folds, execution_identity, run_sha256, recover=True,
                 allow_blas_change=False, current_execution_identity=None, threads=None,
                 repair=True):
    path = _io_path(path)
    original = path.read_bytes()
    raw = original.decode('utf-8')
    lines = raw.splitlines(keepends=True)
    records = []
    interrupted = False
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line, object_pairs_hook=_unique_json_object))
        except json.JSONDecodeError:
            if not recover or index != len(lines)-1 or line.endswith('\n'):
                raise
            interrupted = True
    # Never repair a tail until all preceding records are safe to reuse.
    _validate_records(records, expected_folds=expected_folds,
                      execution_identity=execution_identity,
                      allow_blas_change=allow_blas_change, threads=threads)
    _verify_receipts(records, run_sha256=run_sha256)
    if current_execution_identity is not None:
        previous = records[-1]['execution_identity'] if records else execution_identity
        _authorize_runtime(previous, current_execution_identity,
                           allow_blas_change=allow_blas_change, threads=threads)
    if interrupted:
        backup = path.with_name('records-interrupted.jsonl')
        if backup.exists():
            raise ValueError('An interrupted-record backup already exists; review it before resuming')
        if repair:
            backup.write_bytes(original)
            temporary = path.with_suffix('.jsonl.tmp')
            temporary.write_bytes(''.join(lines[:-1]).encode('utf-8'))
            temporary.replace(path)
    if repair and recover and lines and len(records) == len(lines) and not lines[-1].endswith('\n'):
        # The JSON object can be complete even if its trailing newline was
        # interrupted. Separate it from the next append without dropping it.
        temporary = path.with_suffix('.jsonl.tmp')
        temporary.write_bytes(original + b'\n')
        temporary.replace(path)
    return records


def verify_implementation_snapshot(directory, hashes):
    directory = _io_path(directory)
    for name, expected in hashes.items():
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Implementation snapshot is missing or differs: {name}')


def _file_hash(path):
    return hashlib.sha256(_io_path(path).read_bytes()).hexdigest()


def _execution_controls(args):
    backend = getattr(args, 'optimizer_backend', 'python')
    cache_bytes = getattr(args, 'kernel_cache_bytes', 0)
    if backend not in ('python', 'auto', 'native'):
        raise ValueError('optimizer-backend must be python, auto, or native')
    if (isinstance(cache_bytes, (bool, np.bool_)) or
            not isinstance(cache_bytes, numbers.Integral) or cache_bytes < 0):
        raise ValueError('kernel-cache-bytes must be a nonnegative integer')
    return {'optimizer_backend': backend, 'kernel_cache_bytes': int(cache_bytes)}


def _implementation_bytes(identity):
    package = _io_path(Path(two_stage_sgd_dwd.__file__).parent)
    files = list(package.glob('*.py'))
    files += [_io_path(Path(__file__).with_name(name)) for name in
              ('tune_mnist.py', '_adaptive_grid.py', '_mnist_cv.py', '_runtime_identity.py')]
    if identity['extension_available']:
        name = identity['extension_filename']
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError('Native extension identity must use a portable filename')
        binary = package / name
        if not binary.is_file() or _file_hash(binary) != identity['extension_sha256']:
            raise ValueError('Native extension differs from the execution identity')
        files.append(binary)
    contents = {path.name: path.read_bytes() for path in files}
    if len(contents) != len(files):
        raise ValueError('Implementation snapshot source filenames must be unique')
    return contents


def _verify_execution_identity(config, expected=None):
    expected = config['execution_identity'] if expected is None else expected
    if execution_identity(config['optimizer_backend']) != expected:
        raise ValueError('Optimizer or numerical runtime identity differs from the saved run')
    if expected['extension_available']:
        if config['implementation_sha256'].get(expected['extension_filename']) != expected['extension_sha256']:
            raise ValueError('Native extension is missing from the implementation snapshot identity')


def load_qualified_selection(directory, *, allow_blas_change=False):
    """Load locked settings only after the adaptive search earns acceptance.

    This guard neither fits a model nor opens held-out data. Acceptance applies
    to the recorded finite-budget CV search, not solver optimality or a global
    hyperparameter optimum. A subsequent evaluation must use a separately
    reserved partition and the recorded settings without tuning on that data.
    """
    directory = Path(directory)
    with _RunLease(directory / '.run.lock'):
        return _load_qualified_selection_locked(
            _io_path(directory), allow_blas_change=allow_blas_change)


def _load_qualified_selection_locked(directory, *, allow_blas_change=False):
    if type(allow_blas_change) is not bool:
        raise ValueError('allow_blas_change must be boolean')
    config = _read_json(directory / 'config.json')
    _verify_config_schema(config)
    if config['runtime'] != _runtime_versions():
        raise ValueError('Current Python or dependency versions differ from the saved run')
    status = _read_json(directory / 'status.json')
    if status.get('state') != 'accepted' or not status.get('final_evaluation_eligible'):
        raise ValueError('Search needs review; final evaluation is not qualified')
    locked = _read_json(directory / 'qualified-selection.json')
    required = {'config.json', 'splits.json', 'records.jsonl', 'selection.json', 'review.json'}
    if set(locked.get('evidence_sha256', {})) != required:
        raise ValueError('Qualified selection is missing required evidence hashes')
    for name, expected in locked['evidence_sha256'].items():
        if _file_hash(directory / name) != expected:
            raise ValueError(f'Qualified selection evidence differs: {name}')
    with threadpool_limits(config['threads']):
        current_identity = execution_identity(config['optimizer_backend'])
    records = read_records(directory / 'records.jsonl', expected_folds=config['folds'],
                           execution_identity=config['execution_identity'],
                           run_sha256=_run_sha256(config, _read_json(directory / 'splits.json')),
                           recover=False, allow_blas_change=allow_blas_change,
                           current_execution_identity=current_identity, threads=config['threads'])
    if not ordered_eligible(records, expected_folds=config['folds']):
        raise ValueError('Qualified selection requires eligible candidate evidence')
    selection = _read_json(directory / 'selection.json')
    if (selection.get('state') != 'accepted' or selection['review_reasons']
            or not selection.get('eligible_candidates')
            or not isinstance(selection.get('mean_cv_accuracy'), (int, float))
            or not np.isfinite(selection['mean_cv_accuracy'])
            or selection['params'] != locked['params']):
        raise ValueError('Qualified selection disagrees with the saved review')
    for name in ('epochs', 'sampling', 'batch_size', 'stopping', 'random_state',
                 'preprocessing', 'package_version', 'optimizer_backend',
                 'kernel_cache_bytes', 'execution_identity'):
        if locked.get(name) != config[name]:
            raise ValueError(f'Qualified selection controls differ: {name}')
    summary = _runtime_summary(records, config['execution_identity'], threads=config['threads'])
    for name, value in summary.items():
        if locked.get(name) != value or selection.get(name) != value:
            raise ValueError('Qualified runtime provenance differs: ' + name)
    with threadpool_limits(config['threads']):
        _verify_execution_identity(config, current_identity)
    current_sources = {name: hashlib.sha256(content).hexdigest()
                       for name, content in _implementation_bytes(current_identity).items()}
    if current_sources != config['implementation_sha256']:
        raise ValueError('Current implementation differs from the saved run')
    verify_implementation_snapshot(directory / 'implementation-snapshot', config['implementation_sha256'])
    return _consumer_context(locked, current_identity) if allow_blas_change else locked


def _policy_config(args):
    return AdaptiveSearchConfig(
        max_expansions=args.max_expansions, max_candidates=args.max_candidates,
        top_k=args.top_k, expected_folds=args.folds,
        bracket_factor=getattr(args, 'bracket_factor', 2.0),
        stable_refinement_rounds=getattr(args, 'stable_refinement_rounds', 2),
        max_refinements=getattr(args, 'refinement_rounds', None))


def run_search(args, *, param_grid=None):
    """Run training-only adaptive CV; preserve unresolved searches for review.

    ``param_grid`` optionally supplies an explicit RBF grid instead of the MNIST
    preset. Retention fractions are compared as fixed choices. Lambda and gamma
    expand/refine conditionally. Complete stages must fit within the candidate
    budget; no partial stage earns stopping credit. Neither accepted nor held
    searches automatically refit or access the official test partition.
    """
    # Capture provenance and create the evaluator under the same numerical
    # controls used for fitting. Restore the caller's settings on every exit.
    with threadpool_limits(args.threads):
        return _run_search(args, param_grid=param_grid)


def _run_search(args, *, param_grid):
    allow_blas_change = getattr(args, 'allow_blas_change', False)
    if type(allow_blas_change) is not bool:
        raise ValueError('allow_blas_change must be boolean')
    policy = _policy_config(args)
    grid = mnist_param_grid(broad=args.broad_grid) if param_grid is None else param_grid
    coarse = list(ParameterGrid(grid))
    if args.max_candidates < len(coarse):
        raise ValueError('max-candidates must accommodate the complete initial grid')
    # Validate grid and policy before loading data or creating output files.
    next_search_stage([], grid, policy)
    execution_controls = _execution_controls(args)
    native_identity = execution_identity(execution_controls['optimizer_backend'])
    implementation_bytes = _implementation_bytes(native_identity)
    data = load_mnist_pair(args.data_dir, args.digits, partition='train')
    indices = balanced_indices(data.y, args.train_size, args.random_state)
    X, y, ids = data.X[indices], data.y[indices], data.ids[indices]
    if args.train_size // 2 < args.folds:
        raise ValueError('Each selected class must have at least as many rows as folds')
    folds = list(StratifiedKFold(args.folds, shuffle=True,
                                random_state=args.random_state).split(X, y))
    split_payload = {'observation_ids': ids.tolist(),
        'folds': [{'fit': fit.tolist(), 'score': score.tolist()} for fit, score in folds]}
    config = {'schema_version': _CONFIG_SCHEMA_VERSION, 'digits': list(data.digits),
        'train_size': args.train_size, 'folds': args.folds,
        'epochs': args.epochs, 'random_state': args.random_state, 'batch_size': 1,
        'sampling': 'with_replacement', 'stopping': 'fixed',
        **execution_controls, 'execution_identity': native_identity,
        'preprocessing': 'pixels_divided_by_255',
        'broad_grid': args.broad_grid, 'policy': asdict(policy),
        'threads': args.threads, 'data_hashes': data.hashes,
        'official_test_partition_loaded': False,
        'selection_metric': 'mean_fold_accuracy',
        'tie_break': ['smaller_retention_fraction', 'larger_lambd', 'smaller_gamma'],
        'retention_scope': 'fixed choices in the supplied grid; no continuous retention optimum claimed',
        'coarse_candidates': len(coarse), 'grid': grid,
        'package_version': two_stage_sgd_dwd.__version__,
        'runtime': _runtime_versions(),
        'implementation_sha256': {name: hashlib.sha256(content).hexdigest()
                                  for name, content in implementation_bytes.items()}}
    run_sha256 = _run_sha256(config, split_payload)
    out = Path(args.output_dir)
    if not args.resume:
        _io_path(out).mkdir(parents=True, exist_ok=False)
    with _RunLease(out / '.run.lock'):
        out = _io_path(out)
        if args.resume:
            saved = _read_json(out / 'config.json')
            _verify_config_schema(saved)
            if ({key: value for key, value in saved.items() if key != 'execution_identity'} !=
                    {key: value for key, value in config.items() if key != 'execution_identity'}):
                raise ValueError('Resume configuration, implementation, runtime or input data differs from the saved run')
            _authorize_runtime(saved['execution_identity'], native_identity,
                               allow_blas_change=allow_blas_change, threads=args.threads)
            config = saved
            run_sha256 = _run_sha256(config, split_payload)
            if _read_json(out / 'splits.json') != split_payload:
                raise ValueError('Resume observation IDs or folds differ from the saved split')
            verify_implementation_snapshot(out / 'implementation-snapshot', config['implementation_sha256'])
            plan = _read_json(out / 'plan_state.json')
            if plan.get('schema_version') != _PLAN_SCHEMA_VERSION:
                raise ValueError('Unsupported saved search plan schema; start a fresh output directory '
                                 'for exact parameter identities.')
            if (not isinstance(plan.get('policy_state'), dict)
                    or not isinstance(plan.get('completed'), bool)):
                raise ValueError('Invalid saved search phase in plan_state.json')
            if (plan['policy_state'] and
                    plan['policy_state'].get('schema_version') != POLICY_STATE_SCHEMA_VERSION):
                raise ValueError('Unsupported saved search policy schema; start a fresh output directory '
                                 'for exact parameter identities.')
            records = read_records(out / 'records.jsonl', expected_folds=args.folds,
                                   execution_identity=config['execution_identity'], run_sha256=run_sha256,
                                   allow_blas_change=allow_blas_change,
                                   current_execution_identity=native_identity, threads=args.threads,
                                   repair=False)
            stages = _read_json(out / 'stages.json')
        else:
            records, stages = [], []
            snapshot = out / 'implementation-snapshot'
            snapshot.mkdir()
            for name, content in implementation_bytes.items():
                (snapshot / name).write_bytes(content)
            verify_implementation_snapshot(snapshot, config['implementation_sha256'])
            write_json(out / 'config.json', config)
            write_json(out / 'splits.json', split_payload)
            (out / 'records.jsonl').write_text('', encoding='utf-8')
            write_json(out / 'stages.json', stages)
            plan = {'schema_version': _PLAN_SCHEMA_VERSION, 'policy_state': {},
                    'phase': 'coarse', 'completed': False}
            write_json(out / 'plan_state.json', plan)
        control_path = out / 'inherited-default-control.json'
        progress = _FoldCheckpoint(out, run_sha256=run_sha256, expected_folds=args.folds,
            execution_identity=native_identity, allow_blas_change=allow_blas_change,
            threads=args.threads, records=records, stages=stages,
            control=_read_json(control_path) if control_path.exists() else None)
        if args.resume:
            # Validate both durable stores before repairing a permitted torn
            # candidate append. A corrupt partial fold must leave it untouched.
            read_records(out / 'records.jsonl', expected_folds=args.folds,
                         execution_identity=config['execution_identity'], run_sha256=run_sha256,
                         allow_blas_change=allow_blas_change,
                         current_execution_identity=native_identity, threads=args.threads)
        if plan['completed']:
            if progress.active is not None:
                raise ValueError('Completed search has uncommitted fold progress')
            result = _read_json(out / 'selection.json')
            if result['records_sha256'] != _file_hash(out / 'records.jsonl'):
                raise ValueError('Completed search records differ from the saved selection')
            if result['state'] == 'accepted':
                _load_qualified_selection_locked(out, allow_blas_change=allow_blas_change)
            elif result['state'] != 'needs_review':
                raise ValueError('Invalid completed search state')
            return _consumer_context(result, native_identity) if allow_blas_change else result
        attempted = {candidate_key(record['params']) for record in records}
        # A crash can occur after the last candidate append but before the stage
        # mirror is updated. The policy's saved pending stage still grants credit
        # once; repair only the mirror from the committed candidate evidence.
        for stage in stages:
            if not stage['completed'] and all(candidate_key(p) in attempted for p in stage['candidates']):
                stage['completed'] = True
                ranked = ordered_eligible(records, expected_folds=args.folds)
                stage['score_after'] = ranked[0]['mean_score'] if ranked else None
        write_json(out / 'stages.json', stages)
        evaluator = FoldEvaluator(X, y, ids, folds, max_epochs=args.epochs,
                                  random_state=args.random_state, batch_size=1,
                                  **execution_controls)

        def rank():
            return ordered_eligible(records, expected_folds=args.folds)

        def evaluate_candidate(stage, params):
            try:
                completed_folds = progress.begin(stage, params)

                def save_fold(fold):
                    _verify_execution_identity(config, native_identity)
                    progress.save(fold)

                record = evaluator.evaluate(params, completed_folds=completed_folds,
                                            on_fold_completed=save_fold)
                if not isinstance(record, dict):
                    raise ValueError('Invalid candidate record: evaluator must return a dictionary')
                if 'runtime_transition' in record:
                    raise ValueError('Evaluator records must not supply runtime_transition')
                if record.get('execution_identity') != native_identity:
                    raise ValueError('Evaluator execution identity differs from the saved run')
                if stage != 'inherited-default-control':
                    _validate_records([record], expected_folds=args.folds,
                                      execution_identity=native_identity)
                progress.verify(record)
                record['replayed_fold_indices'] = list(range(len(completed_folds)))
                return record
            except (Exception, KeyboardInterrupt, SystemExit) as exc:
                failure = {'stage': stage, 'params': dict(params),
                           'error_type': type(exc).__name__, 'message': str(exc),
                           'completed_candidates': len(records)}
                _write_failure_json(out / 'failure.json', failure)
                state = 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed'
                _write_failure_json(out / 'status.json', {'state': state, **failure,
                           'final_evaluation_eligible': False, 'final_refit_performed': False,
                           'held_out_scored': False, 'official_test_partition_loaded': False})
                raise

        def evaluate_stage(name, proposals):
            existing = next((stage for stage in stages if stage['name'] == name), None)
            if existing is None:
                prior = rank()
                existing = {'name': name, 'candidates': proposals, 'completed': False,
                            'score_before': prior[0]['mean_score'] if prior else None}
                stages.append(existing)
                write_json(out / 'stages.json', stages)
            elif not set(candidate_key(p) for p in proposals).issubset(
                    candidate_key(p) for p in existing['candidates']):
                raise ValueError('Pending policy candidates disagree with the saved stage')
            for params in proposals:
                key = candidate_key(params)
                if key in attempted:
                    continue
                start = time.perf_counter()
                record = evaluate_candidate(name, params)
                if candidate_key(record['params']) != key:
                    raise ValueError('Evaluator returned a different parameter candidate')
                record.update(stage=name, elapsed_seconds_not_a_benchmark=time.perf_counter()-start)
                previous_runtime = (records[-1]['execution_identity'] if records else
                                    config['execution_identity'])
                if previous_runtime != native_identity:
                    record['runtime_transition'] = _runtime_transition(previous_runtime, native_identity)
                record = _seal_record(record, run_sha256=run_sha256,
                                      sequence=len(records),
                                      previous_sha256=(records[-1]['_receipt']['sha256']
                                                       if records else run_sha256))
                with (out / 'records.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                records.append(record)
                attempted.add(key)
                progress.complete(record)
                ranked = rank()
                status = {'state': 'running', 'stage': name, 'completed_candidates': len(records),
                          'best_mean_accuracy': ranked[0]['mean_score'] if ranked else None,
                          'final_evaluation_eligible': False, 'final_refit_performed': False,
                          'held_out_scored': False, 'official_test_partition_loaded': False}
                write_json(out / 'status.json', status)
                if len(records) % 10 == 0:
                    print(json.dumps(status), flush=True)
            existing['completed'] = True
            ranked = rank()
            existing['score_after'] = ranked[0]['mean_score'] if ranked else None
            write_json(out / 'stages.json', stages)

        if not control_path.exists():
            control = evaluate_candidate(_CONTROL_STAGE, _CONTROL_PARAMS)
            write_json(control_path, control)
            progress.complete(control)
        while True:
            proposal = next_search_stage(records, grid, policy, plan['policy_state'])
            plan.update(policy_state=proposal['state'], phase=proposal['phase'])
            write_json(out / 'plan_state.json', plan)
            if proposal['review'] is not None:
                review = proposal['review']
                break
            evaluate_stage(proposal['phase'], proposal['candidates'])

        state = plan['policy_state']
        _validate_records(records, expected_folds=args.folds,
                          execution_identity=config['execution_identity'],
                          allow_blas_change=allow_blas_change, threads=args.threads)
        _verify_receipts(records, run_sha256=run_sha256)
        ranked = rank()
        accepted = review['state'] == 'accepted'
        runtime_summary = _runtime_summary(records, config['execution_identity'], threads=args.threads)
        selection = {
            **runtime_summary,
            'state': review['state'], 'params': ranked[0]['params'] if ranked else None,
            'mean_cv_accuracy': ranked[0]['mean_score'] if ranked else None,
            'eligible_candidates': len(ranked), 'attempted_candidates': len(records),
            'review_reasons': review['reasons'], 'stopping_reason': review['stopping_reason'],
            'winner_bracket': state.get('winner_bracket'),
            'boundary_issues': state.get('boundary_issues', []),
            'boundary_plateau_evidence': state.get('boundary_plateau_evidence', []),
            'expansion_rounds': state.get('expansion_rounds', 0),
            'refinement_rounds': state.get('refinement_rounds', 0),
            'stable_refinement_rounds': state.get('stable_rounds', 0),
            'scope': 'training-only adaptive selection at the recorded SGD budget and fixed retention choices',
            'optimization_convergence_established': False, 'global_optimum_established': False,
            'final_evaluation_eligible': accepted, 'final_refit_performed': False,
            'held_out_scored': False, 'official_test_partition_loaded': False,
            'larger_budget_and_seed_confirmation_recommended': True,
            'records_sha256': _file_hash(out / 'records.jsonl')}
        write_json(out / 'selection.json', selection)
        write_json(out / 'review.json', {**review, 'final_evaluation_eligible': accepted,
                   'winner_bracket': selection['winner_bracket'],
                   'stable_refinement_rounds': selection['stable_refinement_rounds']})
        if accepted:
            _verify_execution_identity(config, native_identity)
            locked = {'params': selection['params'], 'epochs': args.epochs,
                      'sampling': 'with_replacement', 'batch_size': 1, 'stopping': 'fixed',
                      'random_state': args.random_state, 'preprocessing': config['preprocessing'],
                      'package_version': config['package_version'],
                      **execution_controls, 'execution_identity': config['execution_identity'],
                      **runtime_summary,
                      'evidence_sha256': {name: _file_hash(out / name) for name in
                          ('config.json', 'splits.json', 'records.jsonl', 'selection.json', 'review.json')}}
            write_json(out / 'qualified-selection.json', locked)
        write_json(out / 'status.json', selection)
        plan.update(phase='completed', completed=True)
        write_json(out / 'plan_state.json', plan)
        print(json.dumps(selection), flush=True)
        return selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--digits', type=int, nargs=2, default=(7, 8))
    parser.add_argument('--train-size', type=int, default=2000)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--optimizer-backend', choices=('python', 'auto', 'native'), default='python',
                        help='Optimizer implementation; native requires the optional compiled extension')
    parser.add_argument('--kernel-cache-bytes', type=int, default=0,
                        help='Maximum lazy kernel-column payload per fit in bytes; 0 disables caching')
    parser.add_argument('--broad-grid', action='store_true')
    parser.add_argument('--max-candidates', type=int, default=500)
    parser.add_argument('--max-expansions', type=int, default=3)
    parser.add_argument('--refinement-rounds', type=int, default=None,
                        help='Optional safety cap; reaching it without qualification yields needs_review')
    parser.add_argument('--bracket-factor', type=float, default=2.0)
    parser.add_argument('--stable-refinement-rounds', type=int, default=2)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--allow-blas-change', action='store_true',
                        help='Explicitly allow compatible numerical-build changes and mixed-runtime evidence')
    args = parser.parse_args(argv)
    for name in ('train_size', 'folds', 'epochs', 'threads', 'max_candidates', 'top_k'):
        if getattr(args, name) <= 0:
            parser.error(name.replace('_', '-') + ' must be positive')
    if args.folds < 2 or args.random_state < 0:
        parser.error('Require folds>=2 and nonnegative random-state')
    try:
        execution_controls = _execution_controls(args)
        policy = _policy_config(args)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if not args.run:
        print(json.dumps({'prepared_only': True,
            'initial_candidates': len(ParameterGrid(mnist_param_grid(broad=args.broad_grid))),
            'folds': args.folds, 'epochs': args.epochs, 'policy': asdict(policy),
            **execution_controls,
            'official_test_partition_loaded': False}))
        return
    if args.data_dir is None or args.output_dir is None:
        parser.error('--run requires --data-dir and --output-dir')
    return run_search(args)


if __name__ == '__main__':
    main()
