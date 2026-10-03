"""Explicit training-only MNIST tuning with saved folds and adaptive grids.

Pixels are divided by 255, as in the Chapter 3 procedure. This example never
opens the official test partition. Search results describe the requested SGD
budget; use larger budgets and independent seeds to assess their stability.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from threadpoolctl import threadpool_limits

import two_stage_sgd_dwd
from two_stage_sgd_dwd import mnist_param_grid
from two_stage_sgd_dwd.datasets import load_mnist_pair
from _adaptive_grid import (AdaptiveSearchConfig, candidate_key, next_search_stage,
                            ordered_eligible)
from _mnist_cv import FoldEvaluator


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


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


def read_records(path):
    original = path.read_bytes()
    raw = original.decode('utf-8')
    lines = raw.splitlines(keepends=True)
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines)-1 or line.endswith('\n'):
                raise
            backup = path.with_name('records-interrupted.jsonl')
            if backup.exists():
                raise ValueError('An interrupted-record backup already exists; review it before resuming')
            backup.write_bytes(original)
            temporary = path.with_suffix('.jsonl.tmp')
            temporary.write_bytes(''.join(lines[:index]).encode('utf-8'))
            temporary.replace(path)
    if lines and len(records) == len(lines) and not lines[-1].endswith('\n'):
        # The JSON object can be complete even if its trailing newline was
        # interrupted. Separate it from the next append without dropping it.
        temporary = path.with_suffix('.jsonl.tmp')
        temporary.write_bytes(original + b'\n')
        temporary.replace(path)
    return records


def verify_implementation_snapshot(directory, hashes):
    for name, expected in hashes.items():
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Implementation snapshot is missing or differs: {name}')


def _file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_qualified_selection(directory):
    """Load locked settings only after the adaptive search earns acceptance.

    This guard neither fits a model nor opens held-out data. Acceptance applies
    to the recorded finite-budget CV search, not solver optimality or a global
    hyperparameter optimum. A subsequent evaluation must use a separately
    reserved partition and the recorded settings without tuning on that data.
    """
    directory = Path(directory)
    status = json.loads((directory / 'status.json').read_text(encoding='utf-8'))
    if status.get('state') != 'accepted' or not status.get('final_evaluation_eligible'):
        raise ValueError('Search needs review; final evaluation is not qualified')
    locked = json.loads((directory / 'qualified-selection.json').read_text(encoding='utf-8'))
    required = {'config.json', 'splits.json', 'records.jsonl', 'selection.json', 'review.json'}
    if set(locked.get('evidence_sha256', {})) != required:
        raise ValueError('Qualified selection is missing required evidence hashes')
    for name, expected in locked['evidence_sha256'].items():
        if _file_hash(directory / name) != expected:
            raise ValueError(f'Qualified selection evidence differs: {name}')
    selection = json.loads((directory / 'selection.json').read_text(encoding='utf-8'))
    if (selection.get('state') != 'accepted' or selection['review_reasons']
            or not selection.get('eligible_candidates')
            or not isinstance(selection.get('mean_cv_accuracy'), (int, float))
            or not np.isfinite(selection['mean_cv_accuracy'])
            or selection['params'] != locked['params']):
        raise ValueError('Qualified selection disagrees with the saved review')
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    for name in ('epochs', 'sampling', 'batch_size', 'stopping', 'random_state',
                 'preprocessing', 'package_version'):
        if locked.get(name) != config[name]:
            raise ValueError(f'Qualified selection controls differ: {name}')
    verify_implementation_snapshot(directory / 'implementation-snapshot', config['implementation_sha256'])
    return locked


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
    policy = _policy_config(args)
    grid = mnist_param_grid(broad=args.broad_grid) if param_grid is None else param_grid
    coarse = list(ParameterGrid(grid))
    if args.max_candidates < len(coarse):
        raise ValueError('max-candidates must accommodate the complete initial grid')
    # Validate grid and policy before loading data or creating output files.
    next_search_stage([], grid, policy)
    data = load_mnist_pair(args.data_dir, args.digits, partition='train')
    indices = balanced_indices(data.y, args.train_size, args.random_state)
    X, y, ids = data.X[indices], data.y[indices], data.ids[indices]
    if args.train_size // 2 < args.folds:
        raise ValueError('Each selected class must have at least as many rows as folds')
    folds = list(StratifiedKFold(args.folds, shuffle=True,
                                random_state=args.random_state).split(X, y))
    split_payload = {'observation_ids': ids.tolist(),
        'folds': [{'fit': fit.tolist(), 'score': score.tolist()} for fit, score in folds]}
    implementation_files = list(Path(two_stage_sgd_dwd.__file__).parent.glob('*.py'))
    implementation_files += [Path(__file__).with_name(name) for name in
                             ('tune_mnist.py', '_adaptive_grid.py', '_mnist_cv.py')]
    implementation_bytes = {path.name: path.read_bytes() for path in implementation_files}
    if len(implementation_bytes) != len(implementation_files):
        raise ValueError('Implementation snapshot source filenames must be unique')
    import scipy
    import sklearn
    config = {'schema_version': 2, 'digits': list(data.digits),
        'train_size': args.train_size, 'folds': args.folds,
        'epochs': args.epochs, 'random_state': args.random_state, 'batch_size': 1,
        'sampling': 'with_replacement', 'stopping': 'fixed',
        'preprocessing': 'pixels_divided_by_255',
        'broad_grid': args.broad_grid, 'policy': asdict(policy),
        'threads': args.threads, 'data_hashes': data.hashes,
        'official_test_partition_loaded': False,
        'selection_metric': 'mean_fold_accuracy',
        'tie_break': ['smaller_retention_fraction', 'larger_lambd', 'smaller_gamma'],
        'retention_scope': 'fixed choices in the supplied grid; no continuous retention optimum claimed',
        'coarse_candidates': len(coarse), 'grid': grid,
        'package_version': two_stage_sgd_dwd.__version__,
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                    'scipy': scipy.__version__, 'scikit_learn': sklearn.__version__},
        'implementation_sha256': {name: hashlib.sha256(content).hexdigest()
                                  for name, content in implementation_bytes.items()}}
    out = Path(args.output_dir)
    if args.resume:
        saved = json.loads((out / 'config.json').read_text(encoding='utf-8'))
        if saved != config:
            raise ValueError('Resume configuration, implementation, runtime or input data differs from the saved run')
        if json.loads((out / 'splits.json').read_text(encoding='utf-8')) != split_payload:
            raise ValueError('Resume observation IDs or folds differ from the saved split')
        verify_implementation_snapshot(out / 'implementation-snapshot', config['implementation_sha256'])
        plan = json.loads((out / 'plan_state.json').read_text(encoding='utf-8'))
        if (plan.get('schema_version') != 2 or not isinstance(plan.get('policy_state'), dict)
                or not isinstance(plan.get('completed'), bool)):
            raise ValueError('Invalid saved search phase in plan_state.json')
        records = read_records(out / 'records.jsonl')
        stages = json.loads((out / 'stages.json').read_text(encoding='utf-8'))
        if plan['completed']:
            result = json.loads((out / 'selection.json').read_text(encoding='utf-8'))
            if result['records_sha256'] != _file_hash(out / 'records.jsonl'):
                raise ValueError('Completed search records differ from the saved selection')
            if result['state'] == 'accepted':
                load_qualified_selection(out)
            elif result['state'] != 'needs_review':
                raise ValueError('Invalid completed search state')
            return result
    else:
        out.mkdir(parents=True, exist_ok=False)
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
        plan = {'schema_version': 2, 'policy_state': {}, 'phase': 'coarse', 'completed': False}
        write_json(out / 'plan_state.json', plan)
    attempted = {candidate_key(record['params']) for record in records}
    if len(attempted) != len(records):
        raise ValueError('Saved search contains duplicate candidates')
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
                              random_state=args.random_state, batch_size=1)

    def rank():
        return ordered_eligible(records, expected_folds=args.folds)

    def evaluate_candidate(stage, params):
        try:
            return evaluator.evaluate(params)
        except (Exception, KeyboardInterrupt, SystemExit) as exc:
            failure = {'stage': stage, 'params': dict(params),
                       'error_type': type(exc).__name__, 'message': str(exc),
                       'completed_candidates': len(records)}
            write_json(out / 'failure.json', failure)
            state = 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed'
            write_json(out / 'status.json', {'state': state, **failure,
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
            with (out / 'records.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            records.append(record)
            attempted.add(key)
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

    with threadpool_limits(args.threads):
        control_path = out / 'inherited-default-control.json'
        if not control_path.exists():
            control = evaluate_candidate('inherited-default-control',
                {'kernel': 'rbf', 'lambd': 1e-6, 'gamma': 'scale', 'retention_fraction': .3})
            write_json(control_path, control)
        while True:
            proposal = next_search_stage(records, grid, policy, plan['policy_state'])
            plan.update(policy_state=proposal['state'], phase=proposal['phase'])
            write_json(out / 'plan_state.json', plan)
            if proposal['review'] is not None:
                review = proposal['review']
                break
            evaluate_stage(proposal['phase'], proposal['candidates'])

        state = plan['policy_state']
        ranked = rank()
        accepted = review['state'] == 'accepted'
        selection = {
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
            locked = {'params': selection['params'], 'epochs': args.epochs,
                      'sampling': 'with_replacement', 'batch_size': 1, 'stopping': 'fixed',
                      'random_state': args.random_state, 'preprocessing': config['preprocessing'],
                      'package_version': config['package_version'],
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
    args = parser.parse_args(argv)
    for name in ('train_size', 'folds', 'epochs', 'threads', 'max_candidates', 'top_k'):
        if getattr(args, name) <= 0:
            parser.error(name.replace('_', '-') + ' must be positive')
    if args.folds < 2 or args.random_state < 0:
        parser.error('Require folds>=2 and nonnegative random-state')
    try:
        policy = _policy_config(args)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if not args.run:
        print(json.dumps({'prepared_only': True,
            'initial_candidates': len(ParameterGrid(mnist_param_grid(broad=args.broad_grid))),
            'folds': args.folds, 'epochs': args.epochs, 'policy': asdict(policy),
            'official_test_partition_loaded': False}))
        return
    if args.data_dir is None or args.output_dir is None:
        parser.error('--run requires --data-dir and --output-dir')
    return run_search(args)


if __name__ == '__main__':
    main()
