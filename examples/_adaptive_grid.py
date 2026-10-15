"""Pure proposals for an RBF cross-validation grid; no fitting or I/O.

All valid attempted coordinates define the explored grid, including failed
candidates. Only eligible finite scores participate in selection. Every
proposal evaluates the supplied fixed retention fractions; retention itself
is neither extrapolated nor refined.
"""

from collections.abc import Mapping
from itertools import product
from fractions import Fraction
import math
from numbers import Integral, Real
import struct


_KEYS = ("lambd", "gamma", "retention_fraction")
_AXES = ("lambd", "gamma")
POLICY_STATE_SCHEMA_VERSION = 3


def fold_accuracy(fold, *, require_counts=False):
    """Validate an accuracy receipt, using exact counts when supplied.

    Scalar-only synthetic/custom scores remain supported. Counts are never
    inferred from a rounded score, and a partial count receipt is invalid.
    """
    score = fold.get('accuracy')
    if (isinstance(score, bool) or not isinstance(score, Real)
            or not math.isfinite(score) or not 0 <= score <= 1):
        raise ValueError('Invalid fold accuracy')
    has_counts = 'n_correct' in fold or 'n_validation' in fold
    if not has_counts and not require_counts:
        return float(score)
    correct, size = fold.get('n_correct'), fold.get('n_validation')
    if (any(isinstance(value, bool) or not isinstance(value, Integral)
            for value in (correct, size)) or size <= 0 or not 0 <= correct <= size):
        raise ValueError('Invalid fold accuracy counts')
    value = Fraction(int(correct), int(size))
    if float(score) != float(value):
        raise ValueError('Fold accuracy disagrees with its counts')
    return value


def record_score(record):
    """Exact equal-weight mean of fold accuracies, or the custom scalar score."""
    folds = record.get('folds')
    if not folds:
        return float(record['mean_score'])
    scores = [fold_accuracy(fold) for fold in folds]
    exact = [isinstance(score, Fraction) for score in scores]
    if any(exact) and not all(exact):
        raise ValueError('Accuracy counts must be supplied for every eligible fold')
    if all(exact):
        return sum(scores, Fraction()) / len(scores)
    return float(record['mean_score'])


def score_token(value):
    """JSON-safe score identity for resumed policy comparisons."""
    if value is None:
        return None
    if isinstance(value, Fraction):
        return {'numerator': value.numerator, 'denominator': value.denominator}
    return {'score': float(value)}


def _score_from_token(token):
    if token is None:
        return None
    if set(token) == {'score'}:
        return float(token['score'])
    return Fraction(token['numerator'], token['denominator'])


def _improved(after, before, tolerance):
    if isinstance(after, Fraction) and isinstance(before, Fraction):
        return after - before > Fraction(float(tolerance))
    return after > before + tolerance


def _tied(first, second, tolerance):
    if isinstance(first, Fraction) and isinstance(second, Fraction):
        return abs(first - second) <= Fraction(float(tolerance))
    return math.isclose(first, second, rel_tol=0.0, abs_tol=tolerance)


def _positive(value, name, *, retention=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be representable as a positive finite float") from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    if retention and value > 1.0:
        raise ValueError("retention_fraction must be at most one")
    return value


def _token(value):
    # Preserve every normalized float, including adjacent values and extremes.
    return value.hex()


def _parameters(params):
    if not isinstance(params, Mapping):
        raise ValueError("params must be a mapping with lambd, gamma, and retention_fraction")
    missing = [name for name in _KEYS if name not in params]
    if missing:
        raise ValueError("Missing candidate parameter(s): " + ", ".join(missing))
    if "kernel" in params and params["kernel"] != "rbf":
        raise ValueError("Adaptive proposals support kernel='rbf' only")
    return tuple(_positive(params[name], name, retention=(name == "retention_fraction"))
                 for name in _KEYS)


def candidate_key(params):
    """Return ordered, exact hexadecimal float keys for the three axes.

    lambd and gamma must be positive finite numbers; retention_fraction must
    additionally be at most one. An optional kernel entry must equal 'rbf'.
    Other entries do not contribute to this three-parameter key.
    """
    return tuple(_token(value) for value in _parameters(params))


def _records(records):
    if isinstance(records, (str, bytes, Mapping)):
        raise ValueError("records must be a nonempty iterable of record mappings")
    try:
        entries = list(records)
    except TypeError as error:
        raise ValueError("records must be a nonempty iterable of record mappings") from error
    if not entries:
        raise ValueError("records must not be empty")
    parsed = []
    for number, record in enumerate(entries):
        if not isinstance(record, Mapping) or not {"params", "mean_score", "eligible"} <= record.keys():
            raise ValueError(f"Record {number} must contain params, mean_score, and eligible")
        values = _parameters(record["params"])
        eligible = record["eligible"]
        if not isinstance(eligible, bool):
            raise ValueError(f"Record {number}: eligible must be a boolean")
        score = record["mean_score"]
        if score is None and not eligible:
            score = float('nan')
        if isinstance(score, bool) or not isinstance(score, Real):
            raise ValueError(f"Record {number}: mean_score must be numeric")
        try:
            score = float(score)
        except (OverflowError, ValueError) as error:
            raise ValueError(f"Record {number}: mean_score must be a float") from error
        if eligible and not math.isfinite(score):
            raise ValueError(f"Record {number}: an eligible mean_score must be finite")
        if eligible and record.get('folds'):
            if not valid_record(record):
                raise ValueError(f'Record {number}: invalid fold evidence')
            score = record_score(record)
        parsed.append((values, tuple(_token(value) for value in values), score, eligible))
    if not any(entry[3] for entry in parsed):
        raise ValueError("No eligible finite candidate is available")
    return parsed


def _unique(values):
    return sorted(set(values))


def _coordinates(parsed):
    return {axis: _unique(row[0][index] for row in parsed)
            for index, axis in enumerate(_AXES)}


def _tolerance(tie_tol):
    if isinstance(tie_tol, bool) or not isinstance(tie_tol, Real):
        raise ValueError("tie_tol must be a finite nonnegative number")
    try:
        tie_tol = float(tie_tol)
    except (OverflowError, ValueError) as error:
        raise ValueError("tie_tol must be a finite nonnegative number") from error
    if not math.isfinite(tie_tol) or tie_tol < 0.0:
        raise ValueError("tie_tol must be a finite nonnegative number")
    return tie_tol


def _boundaries(parsed, coordinates, tie_tol):
    best = max(row[2] for row in parsed if row[3])
    tied = [row for row in parsed if row[3]
            and _tied(row[2], best, tie_tol)]
    result = {}
    for index, axis in enumerate(_AXES):
        reached = {row[1][index] for row in tied}
        sides = []
        if _token(coordinates[axis][0]) in reached:
            sides.append("lower")
        if _token(coordinates[axis][-1]) in reached:
            sides.append("upper")
        result[axis] = sides
    return result


def boundary_axes(records, tie_tol=1e-12):
    """Identify explored endpoints reached by any tied-best eligible record.

    Return both axis keys, each with zero or more of 'lower' and 'upper'. A
    singleton explored axis has both endpoints. Ties use absolute score
    tolerance. Failed candidates define explored bounds but cannot be best.
    """
    tolerance = _tolerance(tie_tol)
    parsed = _records(records)
    return _boundaries(parsed, _coordinates(parsed), tolerance)


def _retentions(retention_values):
    if isinstance(retention_values, (str, bytes, Mapping)):
        raise ValueError("retention_values must be a nonempty iterable of fractions")
    try:
        values = [_positive(value, "retention_fraction", retention=True)
                  for value in retention_values]
    except TypeError as error:
        raise ValueError("retention_values must be a nonempty iterable of fractions") from error
    if not values:
        raise ValueError("retention_values must not be empty")
    return _unique(values)


def _outward(values, side, axis):
    endpoint = values[0] if side == "lower" else values[-1]
    if len(values) == 1:
        factor = 10.0
    elif side == "lower":
        factor = min(values[1] / endpoint, 10.0)
    else:
        factor = min(endpoint / values[-2], 10.0)
    # A ratio capped at ten equals exp(min(adjacent log gap, log(10))).
    proposed = endpoint / factor if side == "lower" else endpoint * factor
    outward = proposed < endpoint if side == "lower" else proposed > endpoint
    if (not math.isfinite(proposed) or proposed <= 0.0 or not outward
            or _token(proposed) in {_token(value) for value in values}):
        try:
            return _rounded_geometric_extension(values, 'low' if side == 'lower' else 'high')
        except ValueError as error:
            raise ValueError(f"Cannot represent a distinct positive finite {axis} {side} expansion") from error
    return proposed


def _proposal(values):
    return {"kernel": "rbf", **dict(zip(_KEYS, values))}


def _new_proposals(combinations, evaluated):
    result = {}
    for values in combinations:
        params = _proposal(values)
        key = candidate_key(params)
        if key not in evaluated:
            if key not in result or values < result[key]:
                result[key] = values
    return [_proposal(values) for values in sorted(result.values())]


def propose_expansion(records, retention_values, tie_tol=1e-12):
    """Propose an unevaluated Cartesian shell beyond reached axis endpoints.

    Each reached side moves by its adjacent logarithmic gap, capped at one
    decade. A singleton axis uses a decade in each reached direction. Cross
    corners are included if both axes expand. Every fixed retention fraction
    is used; previously attempted candidates, including failures, are omitted.
    Unrepresentable outward steps raise ValueError instead of returning zero,
    infinity, or a repeated coordinate.
    """
    tolerance = _tolerance(tie_tol)
    parsed = _records(records)
    retentions = _retentions(retention_values)
    coordinates = _coordinates(parsed)
    sides = _boundaries(parsed, coordinates, tolerance)
    enlarged = {}
    for axis in _AXES:
        enlarged[axis] = _unique(coordinates[axis] + [
            _outward(coordinates[axis], side, axis) for side in sides[axis]
        ])
    old = {axis: {_token(value) for value in coordinates[axis]} for axis in _AXES}
    shell = (values for values in product(enlarged["lambd"], enlarged["gamma"], retentions)
             if _token(values[0]) not in old["lambd"] or _token(values[1]) not in old["gamma"])
    return _new_proposals(shell, {row[1] for row in parsed})


def _local_midpoints(values, key):
    index = next(index for index, value in enumerate(values) if _token(value) == key)
    center = values[index]
    local = [center]
    for adjacent in (index - 1, index + 1):
        if not 0 <= adjacent < len(values):
            continue
        lower, upper = sorted((center, values[adjacent]))
        midpoint = _midpoint(lower, upper)
        if midpoint is not None:
            local.append(midpoint)
    return _unique(local)


def propose_refinement(records, retention_values, top_k=3):
    """Refine the best distinct lambd/gamma pairs within explored bounds.

    Rank each pair by its highest eligible finite mean_score over retention
    fractions, breaking ties by larger lambd then smaller gamma. Around each of
    the best top_k pairs, use the pair and geometric midpoints to each explored
    neighbor, form their local Cartesian product, and use every fixed retention
    fraction. An interval is refined only when its computed midpoint is
    representably inside its endpoints. All already attempted keys are excluded.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    parsed = _records(records)
    retentions = _retentions(retention_values)
    coordinates = _coordinates(parsed)
    representatives = {axis: {_token(value): value for value in coordinates[axis]}
                       for axis in _AXES}
    pair_scores = {}
    for _, key, score, eligible in parsed:
        if eligible:
            pair = key[:2]
            pair_scores[pair] = max(score, pair_scores.get(pair, -math.inf))
    ranked = sorted(pair_scores, key=lambda pair: (
        -pair_scores[pair], -representatives["lambd"][pair[0]], representatives["gamma"][pair[1]]
    ))[:int(top_k)]
    combinations = []
    for pair in ranked:
        local_lambd = _local_midpoints(coordinates["lambd"], pair[0])
        local_gamma = _local_midpoints(coordinates["gamma"], pair[1])
        combinations.extend(product(local_lambd, local_gamma, retentions))
    return _new_proposals(combinations, {row[1] for row in parsed})


# The stateful policy below follows the conditional-search rules used by the
# ensemble package. The earlier stateless helpers remain available for callers
# that only need proposals; they do not certify completed adaptive searches.
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class AdaptiveSearchConfig:
    """Search controls; budgets hold unresolved searches for explicit review.

    Only lambd and gamma are continuously refined. Retention fractions remain
    the fixed choices in the initial grid and are not certified between points.
    Set expected_folds when consuming actual CV receipts so partial candidates
    cannot be mistaken for complete evidence. Mean-only synthetic records are
    supported when expected_folds is None.
    """
    max_expansions: int = 3
    max_candidates: int = 500
    top_k: int = 3
    plateau_rounds: int = 2
    improvement_tol: float = 0.0
    tie_tol: float = 1e-12
    refine: bool = True
    bracket_factor: float = 2.0
    stable_refinement_rounds: int = 2
    allow_boundary_plateau: bool = True
    max_refinements: int | None = None
    expected_folds: int | None = None

    def __post_init__(self):
        for name in ('max_expansions', 'max_candidates', 'top_k', 'plateau_rounds',
                     'stable_refinement_rounds', 'max_refinements', 'expected_folds'):
            value = getattr(self, name)
            if value is None and name in ('max_refinements', 'expected_folds'):
                continue
            minimum = 0 if name in ('max_expansions', 'max_refinements') else 1
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}.')
        for name in ('improvement_tol', 'tie_tol'):
            _tolerance(getattr(self, name))
        for name in ('refine', 'allow_boundary_plateau'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be boolean.')
        if (isinstance(self.bracket_factor, bool) or not isinstance(self.bracket_factor, Real)
                or not math.isfinite(self.bracket_factor) or self.bracket_factor <= 1):
            raise ValueError('bracket_factor must be finite and greater than one.')

    @property
    def axes(self):
        return _AXES


def parameter_hash(value):
    """Stable digest for normalized policy parameters and scoring evidence."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _canonical_params(params):
    return _proposal(_parameters(params))


def _finite_score(value):
    return (not isinstance(value, bool) and isinstance(value, Real)
            and math.isfinite(value))


def valid_record(record, *, expected_folds=None):
    """Whether one completed candidate supplies eligible finite fold evidence.

    Failed and nonfinite candidates remain attempted coordinates but are never
    valid neighbors or winners. Passing expected_folds also rejects absent or
    partial fold receipts. All supplied fold indices must be consecutive.
    """
    if not isinstance(record, Mapping) or not {'params', 'eligible', 'mean_score'} <= record.keys():
        raise ValueError('Each record must contain params, eligible, and mean_score')
    _parameters(record['params'])
    if not isinstance(record['eligible'], bool):
        raise ValueError('eligible must be boolean')
    if expected_folds is not None and (isinstance(expected_folds, bool)
            or not isinstance(expected_folds, Integral) or expected_folds < 1):
        raise ValueError('expected_folds must be a positive integer or None')
    if not record['eligible'] or not _finite_score(record['mean_score']):
        return False
    if 'folds' not in record:
        return expected_folds is None
    folds = record['folds']
    if not isinstance(folds, (list, tuple)) or not folds:
        return False
    if expected_folds is not None and len(folds) != expected_folds:
        return False
    if expected_folds is not None and not all(isinstance(fold, Mapping)
            and fold.get('eligible') is True and 'fold' in fold for fold in folds):
        return False
    if not all(isinstance(fold, Mapping) and fold.get('eligible', True) is True
               and _finite_score(fold.get('accuracy')) for fold in folds):
        return False
    if any('fold' in fold for fold in folds):
        indices = [fold.get('fold') for fold in folds]
        if (any(isinstance(index, bool) or not isinstance(index, Integral) for index in indices)
                or sorted(indices) != list(range(len(folds)))):
            return False
    try:
        score = record_score(record)
    except (TypeError, ValueError):
        return False
    mean = (float(score) if isinstance(score, Fraction)
            else math.fsum(float(fold['accuracy']) for fold in folds) / len(folds))
    return math.isclose(float(record['mean_score']), mean, rel_tol=1e-12, abs_tol=1e-12)


def ordered_eligible(records, *, expected_folds=None):
    """Rank valid records by accuracy, retention, stronger lambd, then gamma."""
    return sorted((record for record in records if valid_record(record, expected_folds=expected_folds)),
                  key=lambda record: (-record_score(record),
                      float(record['params']['retention_fraction']),
                      -float(record['params']['lambd']), float(record['params']['gamma'])))


def _initial_grid(grid, config):
    if not isinstance(grid, Mapping) or not set(_KEYS) <= set(grid):
        raise ValueError('The initial grid must include lambd, gamma, and retention_fraction')
    if set(grid) - {*_KEYS, 'kernel'}:
        raise ValueError('The adaptive policy only accepts RBF lambd, gamma, and retention_fraction')
    result = {}
    for name in _KEYS:
        values = grid[name]
        if isinstance(values, (str, bytes, Mapping)):
            raise ValueError('Grid values must be nonempty iterables')
        try:
            values = list(values)
        except TypeError as error:
            raise ValueError('Grid values must be nonempty iterables') from error
        if not values:
            raise ValueError('Grid values must be nonempty iterables')
        result[name] = _unique(_positive(value, name,
                               retention=name == 'retention_fraction') for value in values)
    if list(grid.get('kernel', ['rbf'])) != ['rbf']:
        raise ValueError('The adaptive policy requires kernel=["rbf"]')
    result['kernel'] = ['rbf']
    return result


def _parameter_grid(grids):
    grids = [grids] if isinstance(grids, Mapping) else grids
    for grid in grids:
        names = sorted(grid)
        for values in product(*(grid[name] for name in names)):
            yield dict(zip(names, values))


def _policy_results(entries, config):
    if isinstance(entries, (str, bytes, Mapping)):
        raise ValueError('records must be an iterable of candidate records')
    rows, seen = [], set()
    for record in entries:
        valid = valid_record(record, expected_folds=config.expected_folds)
        params = _canonical_params(record['params'])
        key = candidate_key(params)
        if key in seen:
            raise ValueError('Repeated candidate records are not permitted in the stateful policy')
        seen.add(key)
        score = float(record['mean_score']) if _finite_score(record['mean_score']) else None
        folds = record.get('folds')
        evidence = None if folds is None else [
            {name: (int(fold[name]) if isinstance(fold.get(name), Integral)
                    and not isinstance(fold.get(name), bool)
                    else float(fold[name]) if _finite_score(fold.get(name)) else fold.get(name))
             for name in ('fold', 'eligible', 'status', 'accuracy', 'n_validation', 'n_correct')}
            if isinstance(fold, Mapping) else None for fold in folds]
        # Nonfinite scores have one JSON-safe representation; invalid evidence
        # still contributes to the fingerprint and cannot receive stability credit.
        if evidence is not None:
            for fold in evidence:
                if fold is not None and not _finite_score(fold.get('accuracy')):
                    fold['accuracy'] = None
        exact = record_score(record) if valid else None
        rows.append((params, float(exact) if valid else score, valid, evidence, exact))
    return {'params': [row[0] for row in rows],
            'mean_test_score': [row[1] for row in rows],
            '_valid': [row[2] for row in rows], '_fold_evidence': [row[3] for row in rows],
            '_score_values': [row[4] for row in rows]}


def fully_valid_candidate_mask(results):
    return results['_valid']


def ranked_valid_indices(results):
    return sorted((index for index, valid in enumerate(results['_valid']) if valid),
                  key=lambda index: (-results['_score_values'][index],
                      results['params'][index]['retention_fraction'],
                      -results['params'][index]['lambd'], results['params'][index]['gamma']))


def _policy_records(results):
    return {parameter_hash(parameters): (parameters, results['mean_test_score'][index],
                    results['_valid'][index], results['_fold_evidence'][index],
                    score_token(results['_score_values'][index]))
            for index, parameters in enumerate(results['params'])}


def _rounded_geometric_extension(ordered, side):
    """Round the capped geometric extension once if ordinary arithmetic fails."""
    edge = Fraction(ordered[0] if side == 'low' else ordered[-1])
    if len(ordered) == 1:
        factor = Fraction(10)
    else:
        neighbor = Fraction(ordered[1] if side == 'low' else ordered[-2])
        factor = min(neighbor / edge if side == 'low' else edge / neighbor, Fraction(10))
    exact = edge / factor if side == 'low' else edge * factor
    try:
        value = float(exact)
    except OverflowError:
        value = math.inf
    if (not math.isfinite(value) or value <= 0
            or not (value < edge if side == 'low' else value > edge)):
        raise ValueError('Further boundary values are outside representable positive arithmetic.')
    return value


def _outside(values, side):
    ordered = sorted(values)
    edge = ordered[0] if side == 'low' else ordered[-1]
    if len(ordered) == 1:
        step = math.log(10.)
    elif side == 'low':
        step = math.log(ordered[1]) - math.log(edge)
    else:
        step = math.log(edge) - math.log(ordered[-2])
    # Sparse initial grids can have multi-decade gaps. Extend by at most one
    # decade so a coarse grid does not force an unnecessary extreme jump.
    step = min(step, math.log(10.))
    try:
        value = math.exp(math.log(edge) + (-step if side == 'low' else step))
    except OverflowError:
        value = math.inf
    if (not math.isfinite(value) or value <= 0 or _token(value) in {_token(point) for point in ordered}
            or not (value < edge if side == 'low' else value > edge)):
        # Nearby floats can have identical logarithms. Preserve successful
        # coordinates; repair refusals using the same capped geometric rule.
        return _rounded_geometric_extension(ordered, side)
    return value


def _neighbors(value, values):
    ordered = sorted(values)
    position = ordered.index(value)
    result = [value]
    for adjacent in (position - 1, position + 1):
        if 0 <= adjacent < len(ordered):
            midpoint = _midpoint(value, ordered[adjacent])
            if midpoint is not None:
                result.append(midpoint)
    return sorted(set(result))


def _refinement_grid(results, axes, fixed, config):
    parents, grids = set(), []
    first, second = config.axes
    for index in ranked_valid_indices(results):
        parameters = results['params'][int(index)]
        parent = (parameters[first], parameters[second])
        if parent in parents:
            continue
        parents.add(parent)
        # Coordinates from another conditional slice do not establish support.
        grids.append({**fixed,
                      first: _neighbors(parent[0], _conditional_values(results, parameters, first)),
                      second: _neighbors(parent[1], _conditional_values(results, parameters, second))})
        if len(parents) == config.top_k:
            break
    return grids


def _new_candidates(grids, seen, limit):
    """Return a complete new stage or None when its whole stage exceeds budget."""
    pending, keys = [], set()
    for parameters in _parameter_grid(grids):
        parameters = _canonical_params(parameters)
        key = parameter_hash(parameters)
        if key in seen or key in keys:
            continue
        keys.add(key)
        pending.append(parameters)
        if len(pending) > limit:
            return None
    return pending


def _context(parameters, excluded):
    return {name: value for name, value in parameters.items() if name not in excluded}


def _conditional_values(results, anchor, axis):
    signature = parameter_hash(_context(anchor, (axis,)))
    index = results.get('_conditional_indexes', {}).get(axis)
    if index is not None:
        return index.get(signature, {})
    return {float(parameters[axis]): index
            for index, parameters in enumerate(results['params'])
            if parameter_hash(_context(parameters, (axis,))) == signature}


def _adjacent(value, values):
    lower = [point for point in values if point < value]
    upper = [point for point in values if point > value]
    return max(lower) if lower else None, min(upper) if upper else None


def _rounded_geometric_midpoint(lower, upper):
    """Correctly round sqrt(lower * upper) using exact binary64 comparisons.

    At most 63 bisections suffice; the exact product cannot overflow or
    underflow. Only failed ordinary midpoint proposals use this fallback.
    """
    target = Fraction(lower) * Fraction(upper)
    low = struct.unpack('>Q', struct.pack('>d', lower))[0]
    high = struct.unpack('>Q', struct.pack('>d', upper))[0]

    def value(bits):
        return struct.unpack('>d', struct.pack('>Q', bits))[0]

    while high - low > 1:
        middle = (low + high) // 2
        if Fraction(value(middle)) ** 2 <= target:
            low = middle
        else:
            high = middle
    below, above = value(low), value(high)
    threshold = ((Fraction(below) + Fraction(above)) / 2) ** 2
    return below if target < threshold or (target == threshold and low % 2 == 0) else above


def _midpoint(first, second):
    lower, upper = sorted((first, second))
    value = math.sqrt(lower) * math.sqrt(upper)
    if lower < value < upper:
        return value
    if not lower < math.nextafter(lower, upper) < upper:
        return None
    value = _rounded_geometric_midpoint(lower, upper)
    return value if lower < value < upper else None


def _evidence(records, keys=None):
    keys = sorted(records) if keys is None else sorted(keys)
    return parameter_hash([(key, records.get(key)) for key in keys])


def _bracket(results, anchor, config):
    valid = fully_valid_candidate_mask(results)
    details = {}
    for axis in config.axes:
        conditional = _conditional_values(results, anchor, axis)
        lower, upper = _adjacent(anchor[axis], conditional)
        lower_valid = lower is not None and bool(valid[conditional[lower]])
        upper_valid = upper is not None and bool(valid[conditional[upper]])
        narrow = (lower_valid and upper_valid
                  and anchor[axis] / lower <= config.bracket_factor * (1 + 1e-14)
                  and upper / anchor[axis] <= config.bracket_factor * (1 + 1e-14))
        details[axis] = {'lower': lower, 'upper': upper, 'lower_valid': lower_valid,
                         'upper_valid': upper_valid, 'within_factor': bool(narrow)}
    return all(item['within_factor'] for item in details.values()), details


def _in_region(parameters, region, config):
    return (region is not None
            and parameter_hash(_context(parameters, config.axes)) == region['context']
            and all(lower <= parameters[axis] <= upper
                    for axis, (lower, upper) in region['bounds'].items()))


def _slice_key(anchor, axis, side):
    return parameter_hash((axis, side, _context(anchor, (axis,))))


def _full_strip_certificates(results, anchors, records, stages, leader, best, config):
    """Check actual current slices against two completed outward grid levels.

    A new corner slice may have been evaluated together in the later full strip.
    Its scores are checked here, never inferred from another gamma or fraction.
    """
    valid = fully_valid_candidate_mask(results)
    certificates = []
    for anchor in anchors:
        for axis in config.axes:
            conditional = _conditional_values(results, anchor, axis)
            for side in ('low', 'high'):
                levels = [(stage['label'], item) for stage in stages
                          if _score_from_token(stage['score']) == best
                          and stage['leader_key'] == parameter_hash(leader)
                          for item in stage['levels'] if item['axis'] == axis and item['side'] == side]
                levels = levels[-config.plateau_rounds:]
                if (len(levels) < config.plateau_rounds or levels[-1][1]['to'] != anchor[axis]
                        or any(left[1]['to'] != right[1]['from'] for left, right in zip(levels, levels[1:]))):
                    continue
                start, end = levels[0][1]['from'], levels[-1][1]['to']
                required = {start, *(item['to'] for _, item in levels)}
                if not required.issubset(conditional):
                    continue
                points = sorted((value for value in conditional if min(start, end) <= value <= max(start, end)),
                                reverse=side == 'low')
                indices = [conditional[value] for value in points]
                if not all(valid[index] for index in indices):
                    continue
                scores = [results['_score_values'][index] for index in indices]
                if (any(_improved(after, before, config.improvement_tol)
                        for before, after in zip(scores, scores[1:]))
                        or any(_improved(score, best, config.improvement_tol) for score in scores)):
                    continue
                keys = [parameter_hash(results['params'][index]) for index in indices]
                certificates.append({'slice_key': _slice_key(anchor, axis, side), 'axis': axis, 'side': side,
                    'edge_key': parameter_hash(anchor), 'leader_key': parameter_hash(leader),
                    'conditional_parameters': _context(anchor, (axis,)), 'mean_score': float(best),
                    'score': score_token(best),
                    'completed_outward_rounds': len(levels), 'stage_labels': [label for label, _ in levels],
                    'coordinates': points, 'conditional_mean_scores': [float(score) for score in scores],
                    'conditional_scores': [score_token(score) for score in scores],
                    'evidence_keys': keys, 'evidence': _evidence(records, keys),
                    'basis': 'full_strip_plateau: completed outward stages and valid nonimproving same-slice evidence'})
    return certificates


def _conditional_repairs(results, anchors, axes, fixed, config, certificates=()):
    """Resolve each winning slice; invalid attempts never become valid support."""
    valid = fully_valid_candidate_mask(results)
    grids, issues, waivers = [], [], []
    for anchor in anchors:
        for axis in config.axes:
            conditional = _conditional_values(results, anchor, axis)
            neighbors = _adjacent(anchor[axis], conditional)
            global_neighbors = _adjacent(anchor[axis], axes[axis])
            for side, neighbor, global_neighbor in zip(('low', 'high'), neighbors, global_neighbors):
                if neighbor is not None and valid[conditional[neighbor]]:
                    continue
                if neighbor is not None:
                    point = _midpoint(anchor[axis], neighbor)
                    issue = 'invalid_conditional_neighbor'
                elif global_neighbor is not None:
                    point = global_neighbor
                    issue = 'missing_conditional_neighbor'
                else:
                    certificate = next((item for item in certificates
                                        if item['slice_key'] == _slice_key(anchor, axis, side)
                                        and item['edge_key'] == parameter_hash(anchor)), None)
                    if certificate is not None:
                        waivers.append(certificate)
                        continue
                    try:
                        point = _outside(conditional, side)
                    except ValueError:
                        point = None
                    issue = 'winning_boundary'
                candidate = None if point is None else {**anchor, axis: point}
                issues.append({'parameters': dict(anchor), 'axis': axis, 'side': side,
                               'issue': issue, 'representable': point is not None,
                               'candidate': candidate})
                if point is not None:
                    pair = {name: [point if name == axis else anchor[name]] for name in config.axes}
                    grids.append({**fixed, **pair})
    return grids, issues, waivers


def next_search_stage(records, initial_grid, config, state=None):
    """Pure, bounded training-only policy over completed candidate evidence.

    A pending stage earns credit only after all its candidates have records.
    Repairs at existing coordinates are finite and do not earn refinement credit.
    A documented empirical plateau may waive a remote outer edge only after
    repeated valid outward trials in exactly that conditional slice. It never
    waives the canonical winner's bilateral bracket or invalid/missing evidence.
    """
    if not isinstance(config, AdaptiveSearchConfig):
        raise TypeError('config must be AdaptiveSearchConfig')
    results = _policy_results(records, config)
    grid = _initial_grid(initial_grid, config)
    # Build each conditional index once; a flat surface can have many ties.
    indexes = {axis: {} for axis in config.axes}
    for row, parameters in enumerate(results['params']):
        for axis in config.axes:
            signature = parameter_hash(_context(parameters, (axis,)))
            indexes[axis].setdefault(signature, {})[float(parameters[axis])] = row
    results = {**results, '_conditional_indexes': indexes}
    state = deepcopy({} if state is None else state)
    if state and state.get('schema_version') != POLICY_STATE_SCHEMA_VERSION:
        raise ValueError('Unsupported search policy state schema; start a fresh output directory '
                         'for current parameter and score identities.')
    state['schema_version'] = POLICY_STATE_SCHEMA_VERSION
    identity = parameter_hash((asdict(config), grid))
    if state.get('identity', identity) != identity:
        raise ValueError('Search policy state belongs to different settings or initial grid.')
    state.setdefault('identity', identity)
    state['retention_scope'] = 'fixed_grid_choices_not_continuous_certification'
    state['retention_values'] = list(grid['retention_fraction'])
    state.setdefault('expansion_rounds', 0)
    state.setdefault('refinement_rounds', 0)
    state.setdefault('repair_rounds', 0)
    state.setdefault('stable_rounds', 0)
    state.setdefault('stalled_expansions', 0)
    state.setdefault('full_strip_stages', [])
    records = _policy_records(results)
    seen = set(records)
    ranked = ranked_valid_indices(results) if records else []
    best = None if not len(ranked) else results['_score_values'][ranked[0]]
    leader = None if not len(ranked) else results['params'][int(ranked[0])]
    valid_keys = {parameter_hash(results['params'][int(index)]) for index in ranked}

    def finish(reason, accepted=False):
        review = {'state': 'accepted' if accepted else 'needs_review',
                  'reasons': [] if accepted else [reason],
                  'stopping_reason': reason,
                  'plateau_is_optimality_certificate': False,
                  'plateau_observed': state['stalled_expansions'] >= config.plateau_rounds,
                  'final_refit_performed': False, 'held_out_scored': False,
                  'retention_scope': 'fixed_grid_choices_not_continuous_certification'}
        state['last_reason'] = reason
        return {'state': state, 'phase': None, 'candidates': [], 'review': review}

    pending = state.get('pending')
    if pending:
        missing = [p for p in pending['candidates'] if parameter_hash(p) not in seen]
        if missing:
            return {'state': state, 'phase': pending['label'], 'candidates': missing, 'review': None}
        unchanged = _evidence(records, pending['baseline_keys']) == pending['baseline_evidence']
        baseline_best = _score_from_token(pending['baseline_score'])
        same_score = best == baseline_best
        local_evidence = [parameter_hash(p) for p in pending['candidates']
                          if _in_region(p, pending.get('region'), config)]
        steady = (unchanged and same_score and leader is not None
                  and _in_region(leader, pending.get('region'), config))
        if pending['kind'] == 'refinement':
            if steady and local_evidence and all(key in valid_keys for key in local_evidence):
                state['stable_rounds'] += 1
            else:
                state['stable_rounds'] = 0
                state.pop('stability_region', None)
        elif not (pending['kind'] == 'conditional_repair' and steady):
            state['stable_rounds'] = 0
            state.pop('stability_region', None)
        if pending['kind'] == 'expansion':
            all_valid = all(parameter_hash(p) in valid_keys for p in pending['candidates'])
            if (unchanged and all_valid and best is not None and baseline_best is not None
                    and not _improved(best, baseline_best, config.improvement_tol)):
                state['stalled_expansions'] += 1
            else:
                state['stalled_expansions'] = 0
        elif not unchanged or not same_score:
            state['stalled_expansions'] = 0
        if not unchanged or not same_score:
            state['full_strip_stages'] = []
        elif pending['kind'] == 'expansion':
            state['full_strip_stages'].append({'label': pending['label'],
                'mean_score': float(best), 'score': score_token(best),
                'leader_key': parameter_hash(leader),
                'levels': pending.get('outward_levels', [])})
        state['pending'] = None
    elif state.get('observed_evidence') not in (None, _evidence(records)):
        state['stable_rounds'] = 0
        state['stalled_expansions'] = 0
        state['full_strip_stages'] = []
        state.pop('stability_region', None)
    state['observed_evidence'] = _evidence(records)

    def schedule(grids, kind):
        if (kind == 'refinement' and config.max_refinements is not None
                and state['refinement_rounds'] >= config.max_refinements):
            return finish('refinement_budget: completed local refinement remains unresolved')
        candidates = _new_candidates(grids, seen, config.max_candidates - len(records))
        if candidates is None:
            return finish('candidate_budget: the complete ' + kind + ' stage exceeds max_candidates')
        if not candidates:
            return finish('parameter_limit: no new representable point resolves ' + kind)
        if kind == 'expansion':
            if state['expansion_rounds'] >= config.max_expansions:
                return finish('expansion_budget: a best or tied-best conditional boundary remains unresolved')
            state['expansion_rounds'] += 1
            label = 'expansion-' + str(state['expansion_rounds'])
        elif kind == 'refinement':
            state['refinement_rounds'] += 1
            label = 'local' if state['refinement_rounds'] == 1 else 'local-' + str(state['refinement_rounds'])
        elif kind == 'coarse':
            label = 'coarse'
        else:
            state['repair_rounds'] += 1
            label = 'conditional-repair-' + str(state['repair_rounds'])
        state['pending'] = {'kind': kind, 'label': label, 'candidates': candidates,
                            'baseline_keys': sorted(records), 'baseline_evidence': _evidence(records),
                            'baseline_best': None if best is None else float(best),
                            'baseline_score': score_token(best),
                            'region': deepcopy(state.get('stability_region')),
                            'outward_probes': [deepcopy(item) for item in state.get('boundary_issues', [])
                                               if kind == 'expansion' and item['issue'] == 'winning_boundary'
                                               and item['candidate'] is not None],
                            'outward_levels': []}
        if kind == 'expansion':
            for axis in config.axes:
                for side, select in (('low', min), ('high', max)):
                    probes = [item for item in state['pending']['outward_probes']
                              if item['axis'] == axis and item['side'] == side]
                    if probes:
                        state['pending']['outward_levels'].append({'axis': axis, 'side': side,
                            'from': select(p[axis] for p in results['params']),
                            'to': select(item['candidate'][axis] for item in probes)})
        state['last_reason'] = kind
        return {'state': state, 'phase': label, 'candidates': candidates, 'review': None}

    if not records:
        return schedule(grid, 'coarse')
    if not len(ranked):
        return finish('ineligible: no candidate has finite scores on every validation fold')
    axes = {axis: sorted({float(p[axis]) for p in results['params']}) for axis in config.axes}
    fixed = {name: values for name, values in grid.items() if name not in config.axes}
    tied = [results['params'][int(index)] for index in ranked
            if _tied(best, results['_score_values'][index], config.tie_tol)]
    certificates = []
    if config.allow_boundary_plateau:
        certificates = _full_strip_certificates(results, tied, records, state['full_strip_stages'],
                                                leader, best, config)
    certificates = [item for item in certificates if item['edge_key'] != parameter_hash(leader)]
    repairs, issues, waivers = _conditional_repairs(results, tied, axes, fixed, config, certificates)
    bracketed, bracket = _bracket(results, leader, config)
    state['winner_bracket'] = bracket
    state['winner_bracket_resolved'] = bracketed
    state['boundary_issues'] = issues
    state['boundary_plateau_evidence'] = waivers
    state['best_parameters'] = dict(leader)
    state['best_mean_score'] = float(best)
    state['best_score'] = score_token(best)
    repair_only = bool(issues) and all(item['issue'] == 'missing_conditional_neighbor' for item in issues)
    if state.get('stability_region') is not None and not _in_region(leader, state['stability_region'], config):
        state['stable_rounds'] = 0
        state.pop('stability_region', None)
    if bracketed and (not issues or repair_only) and state.get('stability_region') is None:
        state['stability_region'] = {'context': parameter_hash(_context(leader, config.axes)),
                                     'bounds': {axis: [item['lower'], item['upper']]
                                                for axis, item in bracket.items()}}
    if issues:
        if not repair_only:
            state['stable_rounds'] = 0
            state.pop('stability_region', None)
        # Fill existing-coordinate gaps before extending either axis. Otherwise
        # new coordinates can continually create more missing cross-combinations.
        gaps = [item for item in issues if item['issue'] != 'winning_boundary']
        if gaps:
            repair_grids = [{**fixed, **{axis: [item['candidate'][axis]] for axis in config.axes}}
                            for item in gaps if item['candidate'] is not None]
            return schedule(repair_grids, 'conditional_repair')
        added = {axis: set() for axis in config.axes}
        for item in issues:
            if item['candidate'] is not None:
                added[item['axis']].add(item['candidate'][item['axis']])
        first, second = config.axes
        expansion_grids = []
        if added[first]:
            expansion_grids.append({**fixed, first: sorted(added[first]),
                                    second: sorted(set(axes[second]) | added[second])})
        if added[second]:
            expansion_grids.append({**fixed, first: axes[first], second: sorted(added[second])})
        return schedule(expansion_grids, 'expansion')
    if not config.refine:
        reason = 'valid canonical conditional neighbors; refinement disabled'
        if waivers:
            reason += '; empirical boundary plateau (not an optimality certificate)'
        return finish(reason, accepted=True)
    if bracketed and state['stable_rounds'] >= config.stable_refinement_rounds:
        reason = 'valid conditional bracket and stable completed local rounds'
        if waivers:
            reason += '; empirical boundary plateau (not an optimality certificate)'
        return finish(reason, accepted=True)
    return schedule(_refinement_grid(results, axes, fixed, config), 'refinement')
