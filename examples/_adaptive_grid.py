"""Pure proposals for an RBF cross-validation grid; no fitting or I/O.

All valid attempted coordinates define the explored grid, including failed
candidates. Only eligible finite scores participate in selection. Every
proposal evaluates the supplied fixed retention fractions; retention itself
is neither extrapolated nor refined.
"""

from collections.abc import Mapping
from itertools import product
import math
from numbers import Integral, Real


_KEYS = ("lambd", "gamma", "retention_fraction")
_AXES = ("lambd", "gamma")


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
    # Strings avoid overflowing a maximum finite float merely by rounding its
    # key to 15 significant digits. Proposal coordinates remain numeric.
    return format(value, ".15g")


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
    """Return ordered 15-significant-digit string keys for the three axes.

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
        parsed.append((values, tuple(_token(value) for value in values), score, eligible))
    if not any(entry[3] for entry in parsed):
        raise ValueError("No eligible finite candidate is available")
    return parsed


def _unique(values):
    representatives = {}
    for value in values:
        key = _token(value)
        # Select the same representative regardless of record ordering.
        representatives[key] = min(value, representatives.get(key, value))
    return sorted(representatives.values())


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
            and math.isclose(row[2], best, rel_tol=0.0, abs_tol=tie_tol)]
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
        raise ValueError(f"Cannot represent a distinct positive finite {axis} {side} expansion")
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
        # Unlike sqrt(lower * upper), this does not overflow their product or
        # underflow it before taking the square root.
        midpoint = math.sqrt(lower) * math.sqrt(upper)
        if not math.isfinite(midpoint) or midpoint <= 0.0:
            raise ValueError("Cannot represent a positive finite geometric midpoint")
        if lower < midpoint < upper and _token(midpoint) not in {_token(lower), _token(upper)}:
            local.append(midpoint)
    return _unique(local)


def propose_refinement(records, retention_values, top_k=3):
    """Refine the best distinct lambd/gamma pairs within explored bounds.

    Rank each pair by its highest eligible finite mean_score over retention
    fractions, breaking ties by larger lambd then smaller gamma. Around each of
    the best top_k pairs, use the pair and geometric midpoints to each explored
    neighbor, form their local Cartesian product, and use every fixed retention
    fraction. Coordinates indistinguishable at 15 significant digits are not
    refined. All already attempted keys are excluded.
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
