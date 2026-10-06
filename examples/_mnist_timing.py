"""Read-only component accounting for a completed MNIST candidate record."""

import math
from numbers import Real


_COMPONENTS = ('linear_seconds_charged', 'selection_seconds',
               'kernel_seconds', 'score_seconds')


def _seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite nonnegative number')
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f'{name} is not representable as finite seconds') from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be a finite nonnegative number')
    return value


def summarize_candidate_timing(record):
    """Group recorded costs by work performed in the completion evaluation.

    Supply a previously verified, completed runner record. This helper checks
    timing structure, not receipts, provenance or the expected number of folds.
    Explicit replay metadata is required; its absence in older records does not
    mean fresh work. Component seconds are not elapsed wall time. Replayed costs
    remain historical, and cached screening reference costs are not charged.

    The scope is the evaluation that completed this candidate, not a later
    invocation reading its record, a complete session, or abandoned attempts.
    Missing elapsed time (as for the inherited control) is returned as None.
    Inputs are never modified and all returned containers are independent.
    """
    if not isinstance(record, dict):
        raise ValueError('Expected a completed runner record')
    folds = record.get('folds')
    if not isinstance(folds, list) or not folds:
        raise ValueError('A completed record must contain nonempty ordered folds')
    for index, fold in enumerate(folds):
        if (not isinstance(fold, dict) or type(fold.get('fold')) is not int
                or fold['fold'] != index):
            raise ValueError('Fold IDs must be ordered consecutive integers')
    replayed = record.get('replayed_fold_indices')
    if (not isinstance(replayed, list) or len(replayed) > len(folds)
            or any(type(index) is not int for index in replayed)
            or replayed != list(range(len(replayed)))):
        raise ValueError('Explicit replayed_fold_indices must be a consecutive prefix')

    def group(indices):
        components = {}
        for name in _COMPONENTS:
            try:
                total = math.fsum(_seconds(folds[index].get(name), name) for index in indices)
            except OverflowError as error:
                raise ValueError(f'{name} total is not representable as finite seconds') from error
            components[name] = _seconds(total, name + ' total')
        return {'fold_indices': list(indices), 'component_seconds': components}

    elapsed = (_seconds(record['elapsed_seconds_not_a_benchmark'],
                        'elapsed_seconds_not_a_benchmark')
               if 'elapsed_seconds_not_a_benchmark' in record else None)
    return {'scope': 'candidate_completion_evaluation',
            'completion_elapsed_seconds_not_a_benchmark': elapsed,
            'newly_evaluated_folds': group(range(len(replayed), len(folds))),
            'replayed_folds': group(range(len(replayed)))}
