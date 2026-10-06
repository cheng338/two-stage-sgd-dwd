"""Completed-fold recovery preserves real fits and rejects changed evidence."""
from copy import deepcopy
import hashlib
import json
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from test_mnist_cv import mnist_cv
from test_tune_mnist import _RUNNER, Scores, arguments, read, records, synthetic


PARAMS = {'lambd': .02, 'gamma': .3, 'retention_fraction': 1.}


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(1):
        yield


def data():
    magnitude = np.linspace(.1, .9, 18)
    positive = np.column_stack((magnitude, magnitude ** 2))
    X = np.vstack((-positive, positive))
    y = np.array([2] * 18 + [3] * 18)
    ids = np.array([f'train:{i}' for i in range(len(y))])
    folds = list(StratifiedKFold(3, shuffle=True, random_state=17).split(X, y))
    return X, y, ids, folds


def evaluator(**changes):
    controls = {'max_epochs': 2, 'random_state': 13, 'optimizer_backend': 'python'}
    return mnist_cv.FoldEvaluator(*data(), **(controls | changes))


def nontiming(value):
    if isinstance(value, dict):
        return {key: nontiming(item) for key, item in value.items()
                if key not in {'linear_seconds_reused', 'linear_seconds_charged',
                               'kernel_seconds', 'score_seconds', 'selection_seconds',
                               'linear_cache_hit'}}
    if isinstance(value, list):
        return [nontiming(item) for item in value]
    return value


@pytest.mark.parametrize('backend', ['python', 'auto'])
@pytest.mark.parametrize('cut', [1, 2, 3])
def test_real_fold_prefix_avoids_repeated_fits_and_preserves_statistics(backend, cut):
    reference = evaluator(optimizer_backend=backend).evaluate(PARAMS)
    first = evaluator(optimizer_backend=backend)
    prefix = []

    def interrupt_after_completion(fold):
        prefix.append(deepcopy(fold))
        if len(prefix) == cut:
            raise KeyboardInterrupt('controlled completed-fold interruption')

    with pytest.raises(KeyboardInterrupt, match='controlled'):
        first.evaluate(PARAMS, on_fold_completed=interrupt_after_completion)
    assert first.linear_fit_count_ == cut and not first._linear_cache
    preserved = deepcopy(prefix)
    resumed = evaluator(optimizer_backend=backend)
    new_folds = []
    original_fit = mnist_cv.KernelSGDDWD.fit
    kernel_calls = []

    def fit(model, X, y, **kwargs):
        kernel_calls.append(len(y))
        return original_fit(model, X, y, **kwargs)

    with patch.object(mnist_cv.KernelSGDDWD, 'fit', fit):
        actual = resumed.evaluate(PARAMS, completed_folds=prefix,
                                  on_fold_completed=new_folds.append)
    assert resumed.linear_fit_count_ == len(kernel_calls) == 3 - cut
    assert [fold['fold'] for fold in new_folds] == list(range(cut, 3))
    assert actual['folds'][:cut] == prefix == preserved
    assert nontiming(actual) == nontiming(reference)
    assert actual['mean_score'] == np.mean([fold['accuracy'] for fold in actual['folds']])
    json.dumps(actual, allow_nan=False)


def test_callback_and_replayed_prefix_are_independent_copies():
    first = evaluator()
    published = []

    def mutate_copy(fold):
        published.append(deepcopy(fold))
        fold['stage_seeds']['linear_sgd'] = -1
        fold['linear_fit']['n_updates'] = -1
        fold['selected_class_counts'][:] = [-1, -1]

    original = first.evaluate(PARAMS, on_fold_completed=mutate_copy)
    assert original['folds'] == published
    prefix = deepcopy(published[:1])
    actual = evaluator().evaluate(PARAMS, completed_folds=prefix)
    actual['folds'][0]['stage_seeds']['linear_sgd'] = -1
    actual['folds'][0]['selected_class_counts'][0] = -1
    assert prefix == published[:1]


@pytest.mark.parametrize('failure', [KeyboardInterrupt('interrupted'), OSError('disk full')])
def test_callback_failure_propagates_and_clears_cached_screening(failure):
    first = evaluator()

    def fail(_fold):
        raise failure

    with pytest.raises(type(failure)) as caught:
        first.evaluate(PARAMS, on_fold_completed=fail)
    assert caught.value is failure
    assert first.linear_fit_count_ == 1 and first._linear_cache == {}


def test_selection_failure_is_a_completed_fold_with_finalized_timing():
    params = PARAMS | {'retention_fraction': .01}
    expected = evaluator().evaluate(params)
    first, saved = evaluator(), []

    def pause(fold):
        saved.append(fold)
        raise KeyboardInterrupt('after ineligible completed fold')

    with pytest.raises(KeyboardInterrupt):
        first.evaluate(params, on_fold_completed=pause)
    assert saved[0]['status'] == 'single_class_selection'
    assert saved[0]['selection_seconds'] >= 0
    assert saved[0]['kernel_fit'] is None and saved[0]['kernel_updates'] == 0
    resumed = evaluator()
    with patch.object(mnist_cv.KernelSGDDWD, 'fit', side_effect=AssertionError('kernel fit')):
        actual = resumed.evaluate(params, completed_folds=saved)
    assert resumed.linear_fit_count_ == 2
    assert not actual['eligible'] and actual['mean_score'] is None
    assert nontiming(actual) == nontiming(expected)


@pytest.mark.parametrize('kind', ['not_list', 'too_long', 'duplicate', 'reordered',
                                 'fold_bool', 'seed', 'stage_seed', 'train_count',
                                 'validation_count', 'gamma', 'eligible_type',
                                 'accuracy_nan', 'status'])
def test_invalid_prefix_is_rejected_before_fitting(kind):
    rows = evaluator().evaluate(PARAMS)['folds']
    if kind == 'not_list':
        rows = tuple(rows)
    elif kind == 'too_long':
        rows.append(deepcopy(rows[-1]))
    elif kind == 'duplicate':
        rows[1] = deepcopy(rows[0])
    elif kind == 'reordered':
        rows[:2] = rows[1::-1]
    elif kind == 'fold_bool':
        rows[0]['fold'] = False
    elif kind == 'seed':
        rows[0]['seed'] += 1
    elif kind == 'stage_seed':
        rows[0]['stage_seeds']['linear_sgd'] += 1
    elif kind == 'train_count':
        rows[0]['n_train'] += 1
    elif kind == 'validation_count':
        rows[0]['n_validation'] += 1
    elif kind == 'gamma':
        rows[0]['gamma_resolved'] = .4
    elif kind == 'eligible_type':
        rows[0]['eligible'] = 1
    elif kind == 'accuracy_nan':
        rows[0]['accuracy'] = float('nan')
    else:
        rows[0]['status'] = 'untrained'
    resumed = evaluator()
    with patch.object(mnist_cv.LinearSGDDWD, 'fit', side_effect=AssertionError('fit')):
        with pytest.raises(ValueError, match='[Cc]ompleted'):
            resumed.evaluate(PARAMS, completed_folds=rows)
    assert resumed.linear_fit_count_ == 0


def test_invalid_callback_rejected_before_fitting():
    resumed = evaluator()
    with patch.object(mnist_cv.LinearSGDDWD, 'fit', side_effect=AssertionError('fit')):
        with pytest.raises(ValueError, match='callable'):
            resumed.evaluate(PARAMS, on_fold_completed=123)


def test_runtime_change_during_fold_prevents_publication():
    first = evaluator()
    original_fit = mnist_cv.KernelSGDDWD.fit
    identity = deepcopy(first.execution_identity_)
    changed = deepcopy(identity)
    changed['package_version'] = 'changed-after-fit'
    state = {'changed': False}
    saved = []

    def fit(model, X, y, **kwargs):
        result = original_fit(model, X, y, **kwargs)
        state['changed'] = True
        return result

    with patch.object(mnist_cv.KernelSGDDWD, 'fit', fit), \
            patch.object(mnist_cv, 'execution_identity',
                         side_effect=lambda _backend: deepcopy(changed if state['changed'] else identity)):
        with pytest.raises(RuntimeError, match='identity changed'):
            first.evaluate(PARAMS, on_fold_completed=saved.append)
    assert saved == [] and first._linear_cache == {}


def checkpoint(directory, result, **changes):
    options = dict(run_sha256='a' * 64, expected_folds=3,
                   execution_identity=result['execution_identity'],
                   allow_blas_change=False, threads=1, records=[],
                   stages=[{'name': 'coarse', 'candidates': [result['params']], 'completed': False}])
    return _RUNNER._FoldCheckpoint(directory, **(options | changes))


def reseal(saved):
    payload = {key: value for key, value in saved.items() if key != 'sha256'}
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
    saved['sha256'] = hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize('kind', ['checksum', 'schema_bool', 'fold_count_float',
                                 'sequence_bool', 'wrong_run', 'wrong_key',
                                 'wrong_previous', 'wrong_stage', 'adjacent_param',
                                 'duplicate_fold', 'reordered_fold', 'status', 'fold_bool'])
def test_checkpoint_corruption_rejected_without_rewrite(tmp_path, kind):
    result = evaluator().evaluate(PARAMS)
    progress = checkpoint(tmp_path, result)
    progress.begin('coarse', PARAMS)
    for fold in result['folds'][:2]:
        progress.save(fold)
    saved = read(progress.path)
    if kind == 'checksum':
        saved['folds'][0]['selected_ids_sha256'] = 'b' * 64
    elif kind == 'schema_bool':
        saved['schema_version'] = True
    elif kind == 'fold_count_float':
        saved['expected_folds'] = 3.
    elif kind == 'sequence_bool':
        saved['candidate_sequence'] = False
    elif kind == 'wrong_run':
        saved['run_sha256'] = 'b' * 64
    elif kind == 'wrong_key':
        saved['candidate_key'] = 'b' * 64
    elif kind == 'wrong_previous':
        saved['previous_sha256'] = 'b' * 64
    elif kind == 'wrong_stage':
        saved['stage'] = 'unplanned'
    elif kind == 'adjacent_param':
        saved['params']['lambd'] = np.nextafter(saved['params']['lambd'], np.inf)
        saved['candidate_key'] = _RUNNER._content_sha256(saved['params'])
    elif kind == 'duplicate_fold':
        saved['folds'][1] = deepcopy(saved['folds'][0])
    elif kind == 'reordered_fold':
        saved['folds'].reverse()
    elif kind == 'status':
        saved['folds'][0]['status'] = 'untrained'
    else:
        saved['folds'][0]['fold'] = False
    if kind != 'checksum':
        reseal(saved)
    progress.path.write_text(json.dumps(saved), encoding='utf-8')
    before = progress.path.read_bytes()
    with pytest.raises(ValueError):
        checkpoint(tmp_path, result)
    assert progress.path.read_bytes() == before
    assert [path.name for path in tmp_path.iterdir()] == ['fold-progress.json']


def test_checkpoint_duplicate_json_field_rejected(tmp_path):
    result = evaluator().evaluate(PARAMS)
    progress = checkpoint(tmp_path, result)
    progress.begin('coarse', PARAMS)
    raw = progress.path.read_text(encoding='utf-8')
    progress.path.write_text(raw.replace('{', '{"schema_version": 1,', 1), encoding='utf-8')
    before = progress.path.read_bytes()
    with pytest.raises(ValueError, match='[Dd]uplicate'):
        checkpoint(tmp_path, result)
    assert progress.path.read_bytes() == before


def test_checkpoint_must_be_first_pending_candidate_in_saved_stage(tmp_path):
    result = evaluator().evaluate(PARAMS)
    progress = checkpoint(tmp_path, result)
    progress.begin('coarse', PARAMS)
    progress.save(result['folds'][0])
    before = progress.path.read_bytes()
    earlier = result['params'] | {'lambd': .01}
    stages = [{'name': 'coarse', 'candidates': [earlier, result['params']], 'completed': False}]
    with pytest.raises(ValueError, match='pending|next|stage'):
        checkpoint(tmp_path, result, stages=stages)
    assert progress.path.read_bytes() == before


@pytest.mark.parametrize('control', [False, True])
def test_durable_record_with_stale_progress_is_reconciled_without_fitting(tmp_path, control):
    params = deepcopy(_RUNNER._CONTROL_PARAMS) if control else PARAMS
    result = evaluator().evaluate(params)
    progress = checkpoint(tmp_path, result)
    stage = _RUNNER._CONTROL_STAGE if control else 'coarse'
    progress.begin(stage, params)
    for fold in result['folds']:
        progress.save(fold)
    record = deepcopy(result)
    if control:
        options = {'control': record}
    else:
        record['stage'] = stage
        record = _RUNNER._seal_record(record, run_sha256='a' * 64,
                                     sequence=0, previous_sha256='a' * 64)
        options = {'records': [record]}
    with patch.object(mnist_cv.LinearSGDDWD, 'fit', side_effect=AssertionError('fit')):
        resumed = checkpoint(tmp_path, result, **options)
    assert resumed.active is None and not progress.path.exists()
    assert resumed.sequence == (0 if control else 1)


def test_stale_progress_must_match_committed_fold_content(tmp_path):
    result = evaluator().evaluate(PARAMS)
    progress = checkpoint(tmp_path, result)
    progress.begin('coarse', PARAMS)
    for fold in result['folds']:
        progress.save(fold)
    record = deepcopy(result)
    record['folds'][0]['kernel_seconds'] += 1.
    record['stage'] = 'coarse'
    record = _RUNNER._seal_record(record, run_sha256='a' * 64,
                                 sequence=0, previous_sha256='a' * 64)
    before = progress.path.read_bytes()
    with pytest.raises(ValueError, match='differs'):
        checkpoint(tmp_path, result, records=[record])
    assert progress.path.read_bytes() == before


def test_failed_atomic_fold_write_keeps_previous_prefix_and_original_exception(tmp_path):
    result = evaluator().evaluate(PARAMS)
    progress = checkpoint(tmp_path, result)
    progress.begin('coarse', PARAMS)
    progress.save(result['folds'][0])
    before = progress.path.read_bytes()
    active = deepcopy(progress.active)
    failure = OSError('controlled atomic replace failure')
    with patch.object(type(progress.path), 'replace', side_effect=failure):
        with pytest.raises(OSError) as caught:
            progress.save(result['folds'][1])
    assert caught.value is failure
    assert progress.path.read_bytes() == before and progress.active == active
    assert not progress.path.with_suffix('.json.tmp').exists()
    resumed = checkpoint(tmp_path, result)
    assert resumed.begin('coarse', PARAMS) == result['folds'][:1]


def test_changed_runtime_requires_opt_in_and_restarts_unfinished_candidate(tmp_path):
    result = evaluator().evaluate(PARAMS)
    progress = checkpoint(tmp_path, result)
    progress.begin('coarse', PARAMS)
    progress.save(result['folds'][0])
    original = read(progress.path)
    replacement = deepcopy(result['execution_identity'])
    pool = next(pool for pool in replacement['numerical_runtime']['threadpools']
                if pool['user_api'] == 'blas')
    pool.update(internal_api='openblas', prefix='test_different_openblas',
                filename='test_different_openblas.dll', version='test-version', sha256='b' * 64)
    before = progress.path.read_bytes()
    with pytest.raises(ValueError):
        strict = checkpoint(tmp_path, result, execution_identity=replacement)
        strict.begin('coarse', PARAMS)
    assert progress.path.read_bytes() == before
    resumed = checkpoint(tmp_path, result, execution_identity=replacement, allow_blas_change=True)
    assert resumed.begin('coarse', PARAMS) == []
    archives = list(tmp_path.glob('fold-progress-abandoned-*.json'))
    assert len(archives) == 1
    evidence = read(archives[0])
    assert evidence['checkpoint'] == original
    assert evidence['replacement_execution_identity'] == replacement
    assert evidence['reason'] == 'numerical_runtime_changed'
    assert read(progress.path)['execution_identity'] == replacement
    assert read(progress.path)['folds'] == []


def test_interrupted_runtime_archive_allows_another_authorized_target(tmp_path):
    result = evaluator().evaluate(PARAMS)
    progress = checkpoint(tmp_path, result)
    progress.begin('coarse', PARAMS)
    progress.save(result['folds'][0])
    original = read(progress.path)
    replacements = []
    for letter in ('b', 'c'):
        replacement = deepcopy(result['execution_identity'])
        pool = next(pool for pool in replacement['numerical_runtime']['threadpools']
                    if pool['user_api'] == 'blas')
        pool.update(prefix='test_' + letter, filename='test_' + letter + '.dll',
                    version='test-version', sha256=letter * 64)
        replacements.append(replacement)
    first = checkpoint(tmp_path, result, execution_identity=replacements[0], allow_blas_change=True)
    with patch.object(type(progress.path), 'unlink', side_effect=OSError('before removal')):
        with pytest.raises(OSError, match='before removal'):
            first.begin('coarse', PARAMS)
    assert read(progress.path) == original
    second = checkpoint(tmp_path, result, execution_identity=replacements[1], allow_blas_change=True)
    assert second.begin('coarse', PARAMS) == []
    archived = [read(path) for path in tmp_path.glob('fold-progress-abandoned-*.json')]
    assert len(archived) == 2
    assert all(item['checkpoint'] == original for item in archived)
    assert all(any(item['replacement_execution_identity'] == replacement for item in archived)
               for replacement in replacements)


@pytest.mark.parametrize('kind', ['corrupt_checkpoint', 'strict_runtime_change'])
def test_invalid_progress_precedes_torn_ledger_tail_repair(tmp_path, kind):
    args, fake = arguments(tmp_path), Scores()
    original_save = _RUNNER._FoldCheckpoint.save

    def stop_after_saved_fold(progress, fold):
        original_save(progress, fold)
        if progress.active['stage'] != _RUNNER._CONTROL_STAGE:
            raise KeyboardInterrupt('saved first candidate fold')

    with synthetic(fake):
        with patch.object(_RUNNER._FoldCheckpoint, 'save', stop_after_saved_fold):
            with pytest.raises(KeyboardInterrupt):
                _RUNNER.run_search(args)
        progress_path = args.output_dir / 'fold-progress.json'
        saved = read(progress_path)
        if kind == 'corrupt_checkpoint':
            saved['folds'][0]['accuracy'] = .123
        else:
            pool = next(pool for pool in saved['execution_identity']['numerical_runtime']['threadpools']
                        if pool['user_api'] == 'blas')
            pool.update(prefix='test_changed_blas', filename='test_changed_blas.dll',
                        sha256='b' * 64)
            reseal(saved)
        progress_path.write_text(json.dumps(saved), encoding='utf-8')
        ledger = args.output_dir / 'records.jsonl'
        ledger.write_bytes(ledger.read_bytes() + b'{"params":')

        def files():
            return {path.relative_to(args.output_dir).as_posix(): path.read_bytes()
                    for path in args.output_dir.rglob('*')
                    if path.is_file() and path.name != '.run.lock'}

        before = files()
        calls, constructors = deepcopy(fake.calls), len(fake.constructed)
        args.resume = True
        with pytest.raises(ValueError):
            _RUNNER.run_search(args)
        assert fake.calls == calls and len(fake.constructed) == constructors
        assert files() == before
        assert not (args.output_dir / 'records-interrupted.jsonl').exists()


def test_returned_candidate_must_match_published_folds_before_durable_commit(tmp_path):
    args, fake = arguments(tmp_path), Scores()
    original = fake.evaluate

    def changed_after_callback(params, **kwargs):
        record = original(params, **kwargs)
        if params['gamma'] != 'scale':
            record['mean_score'] = .5
            for fold in record['folds']:
                fold['accuracy'] = .5
        return record

    with synthetic(fake), patch.object(fake, 'evaluate', side_effect=changed_after_callback):
        with pytest.raises(ValueError, match='fold-progress'):
            _RUNNER.run_search(args)
    assert records(args.output_dir) == []
    assert read(args.output_dir / 'fold-progress.json')['folds'][0]['accuracy'] != .5
