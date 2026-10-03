"""Explicit training-only MNIST tuning with saved folds and adaptive grids.

Pixels are divided by 255, as in the Chapter 3 procedure. This example never
opens the official test partition. Search results describe the requested SGD
budget; use larger budgets and independent seeds to assess their stability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from threadpoolctl import threadpool_limits

import two_stage_sgd_dwd
from two_stage_sgd_dwd import mnist_param_grid
from two_stage_sgd_dwd.datasets import load_mnist_pair
from _adaptive_grid import candidate_key, boundary_axes, propose_expansion, propose_refinement
from _mnist_cv import FoldEvaluator


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
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


def ordered_eligible(records):
    return sorted((r for r in records if r['eligible']), key=lambda r: (
        -r['mean_score'], r['params']['retention_fraction'],
        -r['params']['lambd'], r['params']['gamma']))


def verify_implementation_snapshot(directory, hashes):
    for name, expected in hashes.items():
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Implementation snapshot is missing or differs: {name}')


def read_plan(path):
    plan = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(plan, dict)
            or plan.get('phase') not in ('coarse', 'expansion', 'refinement', 'completed')
            or type(plan.get('index')) is not int or plan['index'] < 0
            or type(plan.get('nonimproving')) is not int or plan['nonimproving'] < 0
            or type(plan.get('plateau')) is not bool
            or not isinstance(plan.get('reasons'), list)
            or any(not isinstance(reason, str) for reason in plan['reasons'])):
        raise ValueError('Invalid saved search phase in plan_state.json')
    return plan


def run_search(args):
    grid = mnist_param_grid(broad=args.broad_grid)
    coarse = list(ParameterGrid(grid))
    if args.max_candidates < len(coarse):
        raise ValueError('max-candidates must accommodate the complete initial grid')
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
    out = args.output_dir
    config = {'digits': list(data.digits), 'train_size': args.train_size, 'folds': args.folds,
        'epochs': args.epochs, 'random_state': args.random_state, 'batch_size': 1,
        'sampling': 'with_replacement', 'preprocessing': 'pixels_divided_by_255',
        'broad_grid': args.broad_grid, 'max_candidates': args.max_candidates,
        'max_expansions': args.max_expansions, 'refinement_rounds': args.refinement_rounds,
        'top_k': args.top_k, 'threads': args.threads, 'data_hashes': data.hashes,
        'official_test_partition_loaded': False,
        'selection_metric': 'mean_fold_accuracy',
        'tie_break': ['smaller_retention_fraction', 'larger_lambd', 'smaller_gamma'],
        'plateau_rule': 'two_complete_nonimproving_expansion_rounds',
        'coarse_candidates': len(coarse), 'grid': grid,
        'package_version': two_stage_sgd_dwd.__version__,
        'implementation_sha256': {name: hashlib.sha256(content).hexdigest()
                                  for name, content in implementation_bytes.items()}}
    if args.resume:
        saved = json.loads((out / 'config.json').read_text(encoding='utf-8'))
        if saved != config:
            raise ValueError('Resume configuration or input data differs from the saved run')
        if json.loads((out / 'splits.json').read_text(encoding='utf-8')) != split_payload:
            raise ValueError('Resume observation IDs or folds differ from the saved split')
        verify_implementation_snapshot(out / 'implementation-snapshot', config['implementation_sha256'])
        plan = read_plan(out / 'plan_state.json')
        status_path = out / 'status.json'
        if status_path.exists() and json.loads(status_path.read_text(encoding='utf-8')).get('state') == 'exploration_completed':
            if plan['phase'] != 'completed':
                raise ValueError('Completed status disagrees with the saved search phase')
            return json.loads((out / 'selection.json').read_text(encoding='utf-8'))
        records = read_records(out / 'records.jsonl')
        stages = json.loads((out / 'stages.json').read_text(encoding='utf-8'))
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
        plan = {'phase': 'coarse', 'index': 0, 'nonimproving': 0,
                'plateau': False, 'reasons': []}
        write_json(out / 'plan_state.json', plan)
    evaluator = FoldEvaluator(X, y, ids, folds, max_epochs=args.epochs,
                              random_state=args.random_state, batch_size=1)
    attempted = {candidate_key(r['params']) for r in records}

    def advance_plan(**updates):
        plan.update(updates)
        write_json(out / 'plan_state.json', plan)

    def finish_phase(next_phase, reason=None):
        reasons = list(plan['reasons'])
        if reason is not None and reason not in reasons:
            reasons.append(reason)
        advance_plan(phase=next_phase, index=0, reasons=reasons)

    def evaluate_candidate(stage, params):
        try:
            return evaluator.evaluate(params)
        except Exception as exc:
            # Unexpected optimizer/evaluation errors are failures of the run,
            # not scored ineligible candidates. Keep the pending stage and
            # committed records intact so an explicit resume retries this fit.
            failure = {'stage': stage, 'params': dict(params),
                       'error_type': type(exc).__name__, 'message': str(exc),
                       'completed_candidates': len(records)}
            write_json(out / 'failure.json', failure)
            write_json(out / 'status.json', {'state': 'failed', **failure,
                       'official_test_partition_loaded': False})
            raise

    def evaluate_stage(name, proposals):
        existing = next((s for s in stages if s['name'] == name), None)
        if existing is None:
            new = [p for p in proposals if candidate_key(p) not in attempted]
            if len(records) + len(new) > args.max_candidates:
                return False
            prior_ranked = ordered_eligible(records)
            existing = {'name': name, 'candidates': new, 'completed': False,
                        'score_before': prior_ranked[0]['mean_score'] if prior_ranked else None}
            stages.append(existing)
            write_json(out / 'stages.json', stages)
        if existing['completed']:
            return True
        for params in existing['candidates']:
            key = candidate_key(params)
            if key in attempted:
                continue
            start = time.perf_counter()
            record = evaluate_candidate(name, params)
            record.update(stage=name, elapsed_seconds_not_a_benchmark=time.perf_counter()-start)
            with (out / 'records.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            records.append(record)
            attempted.add(key)
            ranked = ordered_eligible(records)
            write_json(out / 'status.json', {'state': 'running', 'stage': name,
                'completed_candidates': len(records),
                'best_mean_accuracy': ranked[0]['mean_score'] if ranked else None})
            if len(records) % 10 == 0:
                print(json.dumps({'stage': name, 'candidates': len(records),
                    'best_mean_accuracy': ranked[0]['mean_score'] if ranked else None}), flush=True)
        existing['completed'] = True
        ranked = ordered_eligible(records)
        existing['score_after'] = ranked[0]['mean_score'] if ranked else None
        write_json(out / 'stages.json', stages)
        return True

    with threadpool_limits(args.threads):
        control_path = out / 'inherited-default-control.json'
        if not control_path.exists():
            control = evaluate_candidate('inherited-default-control',
                {'kernel': 'rbf', 'lambd': 1e-6, 'gamma': 'scale', 'retention_fraction': .3})
            write_json(control_path, control)
        while plan['phase'] != 'completed':
            if plan['phase'] == 'coarse':
                evaluate_stage('coarse', coarse)
                if not ordered_eligible(records):
                    raise RuntimeError('No candidate completed every fold with an eligible model')
                finish_phase('expansion')
                continue

            phase, index = plan['phase'], plan['index']
            limit = args.max_expansions if phase == 'expansion' else args.refinement_rounds
            next_phase = 'refinement' if phase == 'expansion' else 'completed'
            if index >= limit or (phase == 'expansion' and plan['nonimproving'] >= 2):
                finish_phase(next_phase)
                continue
            stage_name = f'{phase}-{index+1}'
            previous_stage = next((s for s in stages if s['name'] == stage_name), None)
            if previous_stage is not None:
                proposals = previous_stage['candidates']
            elif phase == 'expansion':
                proposals = propose_expansion(records, grid['retention_fraction'])
            else:
                proposals = propose_refinement(records, grid['retention_fraction'], top_k=args.top_k)
            if not proposals:
                # Persist stop decisions before the next phase can add records
                # that would change a previously empty proposal set.
                finish_phase(next_phase)
                continue
            if not evaluate_stage(stage_name, proposals):
                finish_phase(next_phase, f'candidate_budget_prevents_complete_{phase}')
                continue
            completed = next(s for s in stages if s['name'] == stage_name)
            improved = completed['score_after'] > completed['score_before'] + 1e-12
            if phase == 'expansion':
                nonimproving = 0 if improved else plan['nonimproving'] + 1
                advance_plan(index=index+1, nonimproving=nonimproving, plateau=nonimproving >= 2)
            else:
                advance_plan(index=index+1, plateau=False if improved else plan['plateau'])
            # The stage is persisted before advancing the cursor. If that last
            # write is interrupted, replay uses its saved before/after scores
            # and advances once, without reevaluating any committed candidate.
        boundaries = boundary_axes(records)
        reasons = list(plan['reasons'])
        if any(boundaries.values()):
            reason = 'boundary_plateau_at_this_epoch_budget' if plan['plateau'] else 'unresolved_boundary'
            if reason not in reasons:
                reasons.append(reason)
        advance_plan(reasons=reasons)
        ranked = ordered_eligible(records)
        selection = {'params': ranked[0]['params'], 'mean_cv_accuracy': ranked[0]['mean_score'],
            'eligible_candidates': len(ranked), 'attempted_candidates': len(records),
            'boundaries': boundaries, 'review_reasons': reasons,
            'scope': 'training-only exploratory selection at the recorded SGD budget',
            'optimization_convergence_established': False, 'official_test_partition_loaded': False,
            'larger_budget_and_seed_confirmation_required': True}
        write_json(out / 'selection.json', selection)
        write_json(out / 'status.json', {'state': 'exploration_completed', **selection})
        print(json.dumps(selection), flush=True)
    return selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--digits', type=int, nargs=2, default=(7,8))
    parser.add_argument('--train-size', type=int, default=2000)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--broad-grid', action='store_true')
    parser.add_argument('--max-candidates', type=int, default=2000)
    parser.add_argument('--max-expansions', type=int, default=3)
    parser.add_argument('--refinement-rounds', type=int, default=2)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    for name in ('train_size', 'folds', 'epochs', 'threads', 'max_candidates', 'top_k'):
        if getattr(args, name) <= 0:
            parser.error(name.replace('_','-') + ' must be positive')
    if args.folds < 2 or args.max_expansions < 0 or args.refinement_rounds < 0 or args.random_state < 0:
        parser.error('Require folds>=2, nonnegative round limits and nonnegative random-state')
    if not args.run:
        print(json.dumps({'prepared_only': True,
            'initial_candidates': len(ParameterGrid(mnist_param_grid(broad=args.broad_grid))),
            'folds': args.folds, 'epochs': args.epochs, 'max_candidates': args.max_candidates,
            'official_test_partition_loaded': False}))
        return
    if args.data_dir is None or args.output_dir is None:
        parser.error('--run requires --data-dir and --output-dir')
    return run_search(args)


if __name__ == '__main__':
    main()
