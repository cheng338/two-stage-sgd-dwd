"""Pure adaptive-grid examples, with no classifier fitting or data access."""

import copy
import importlib.util
from itertools import product
import math
from pathlib import Path
import sys
import unittest


_PATH = Path(__file__).resolve().parents[1] / "examples" / "_adaptive_grid.py"
_SPEC = importlib.util.spec_from_file_location("adaptive_grid_example_support", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
candidate_key = _MODULE.candidate_key
boundary_axes = _MODULE.boundary_axes
propose_expansion = _MODULE.propose_expansion
propose_refinement = _MODULE.propose_refinement


def record(lambd, gamma, retention=0.05, score=0.5, eligible=True):
    return {"params": {"kernel": "rbf", "lambd": lambd, "gamma": gamma,
                       "retention_fraction": retention},
            "mean_score": score, "eligible": eligible}


def grid(lambdas, gammas, retentions=(0.05, 0.1)):
    return [record(lambd, gamma, retention) for lambd, gamma, retention
            in product(lambdas, gammas, retentions)]


def triples(proposals):
    return {(p["lambd"], p["gamma"], p["retention_fraction"]) for p in proposals}


class CandidateKeyTests(unittest.TestCase):
    def test_keys_are_ordered_15_significant_digit_tokens(self):
        params = {"retention_fraction": 0.05, "gamma": 1.234567890123456,
                  "lambd": 0.01, "kernel": "rbf"}
        self.assertEqual(candidate_key(params), ("0.01", "1.23456789012346", "0.05"))
        rounded = dict(params, gamma=1.2345678901234562)
        self.assertEqual(candidate_key(params), candidate_key(rounded))

    def test_extreme_finite_values_have_finite_input_keys(self):
        result = candidate_key({"lambd": math.ulp(0.0), "gamma": sys.float_info.max,
                                "retention_fraction": 1.0})
        self.assertTrue(all(isinstance(value, str) for value in result))
        self.assertNotIn("inf", result)
        self.assertNotIn("0", result)

    def test_invalid_parameter_coordinates_are_rejected(self):
        for name, value in [
            ("lambd", 0), ("gamma", -1), ("gamma", math.inf),
            ("lambd", math.nan), ("retention_fraction", 1.01),
            ("retention_fraction", False), ("gamma", "1.0"),
            ("lambd", 10**400),
        ]:
            with self.subTest(name=name, value=value):
                params = record(1, 1)["params"]
                params[name] = value
                with self.assertRaises(ValueError):
                    candidate_key(params)
        for params in [{}, None, {"lambd": 1, "gamma": 1}, dict(record(1, 1)["params"], kernel="linear")]:
            with self.subTest(params=params), self.assertRaises(ValueError):
                candidate_key(params)


class BoundaryAndExpansionTests(unittest.TestCase):
    def test_tied_best_records_can_reach_both_sides_of_both_axes(self):
        records = grid([1, 10, 100], [2, 20, 200])
        records += [record(1, 200, score=0.9), record(100, 2, score=0.9 - 5e-13)]
        self.assertEqual(boundary_axes(records),
                         {"lambd": ["lower", "upper"], "gamma": ["lower", "upper"]})
        self.assertEqual(boundary_axes(records, tie_tol=0.0),
                         {"lambd": ["lower"], "gamma": ["upper"]})

    def test_expansion_uses_adjacent_log_gap_cap_and_includes_cross_corner(self):
        records = grid([1.0, 100.0], [2.0, 8.0])
        records.append(record(1.0, 8.0, score=0.9))
        proposals = propose_expansion(records, [0.05, 0.1])
        # Lambda's two-decade gap is capped at one decade; gamma's factor-four
        # gap remains four. Include (0.1, 32), not just edge strips.
        new_pairs = {(0.1, 2.0), (0.1, 8.0), (0.1, 32.0), (1.0, 32.0), (100.0, 32.0)}
        self.assertEqual(triples(proposals), {(l, g, r) for l, g in new_pairs for r in (0.05, 0.1)})
        self.assertEqual(len(proposals), 10)
        self.assertTrue(all(p["kernel"] == "rbf" for p in proposals))
        self.assertTrue(all(set(p) == {"kernel", "lambd", "gamma", "retention_fraction"} for p in proposals))

    def test_singleton_axes_expand_by_decades_but_only_new_shell_is_proposed(self):
        records = [record(2.0, 3.0, retention=0.1)]
        self.assertEqual(boundary_axes(records),
                         {"lambd": ["lower", "upper"], "gamma": ["lower", "upper"]})
        proposals = propose_expansion(records, [0.1, 0.05])
        expected = {(l, g, r) for l, g, r in product((0.2, 2.0, 20.0), (0.3, 3.0, 30.0), (0.05, 0.1))
                    if (l, g) != (2.0, 3.0)}
        self.assertEqual(triples(proposals), expected)

    def test_interior_best_produces_no_expansion(self):
        records = grid([1, 10, 100], [1, 10, 100])
        records.append(record(10, 10, score=0.9))
        self.assertEqual(boundary_axes(records), {"lambd": [], "gamma": []})
        self.assertEqual(propose_expansion(records, [0.05, 0.1]), [])

    def test_failed_boundary_defines_explored_grid_and_is_not_retried(self):
        failed = record(1.0, 2.0, score=math.nan, eligible=False)
        records = [failed, record(10.0, 2.0, score=0.9)]
        self.assertEqual(boundary_axes(records)["lambd"], ["upper"])
        proposals = propose_expansion(records, [0.05, 0.1])
        keys = {candidate_key(params) for params in proposals}
        self.assertNotIn(candidate_key(failed["params"]), keys)
        self.assertNotIn(("1", "2", "0.1"), keys)  # No invented lower expansion.
        self.assertEqual({p["lambd"] for p in proposals}, {1.0, 10.0, 100.0})

    def test_ineligible_finite_score_cannot_win(self):
        records = grid([1, 10, 100], [1, 10, 100])
        records += [record(10, 10, score=0.9), record(100, 100, score=1000.0, eligible=False)]
        self.assertEqual(boundary_axes(records), {"lambd": [], "gamma": []})

    def test_unrepresentable_outward_steps_raise_clear_errors(self):
        for endpoint in (math.ulp(0.0), sys.float_info.max):
            with self.subTest(endpoint=endpoint), self.assertRaisesRegex(ValueError, "represent"):
                propose_expansion([record(endpoint, 1.0)], [0.05])

    def test_inputs_and_order_do_not_change_proposals(self):
        records = grid([1.0, 10.0], [2.0, 8.0])
        records.append(record(1.0, 8.0, score=0.9))
        original = copy.deepcopy(records)
        forward = propose_expansion(records, [0.1, 0.05, 0.1])
        reverse = propose_expansion(reversed(records), [0.05, 0.1])
        self.assertEqual(forward, reverse)
        self.assertEqual(records, original)
        self.assertEqual(len({candidate_key(params) for params in forward}), len(forward))


class RefinementTests(unittest.TestCase):
    def test_midpoints_cross_locally_stay_inside_bounds_and_use_all_retentions(self):
        records = grid([1.0, 100.0, 10000.0], [4.0, 16.0, 64.0])
        records.append(record(100.0, 16.0, score=0.9))
        proposals = propose_refinement(records, [0.05, 0.1], top_k=1)
        expected = {(l, g, r) for l, g, r in product((10.0, 100.0, 1000.0), (8.0, 16.0, 32.0), (0.05, 0.1))
                    if (l, g) != (100.0, 16.0)}
        self.assertEqual(triples(proposals), expected)
        self.assertEqual(len(proposals), 16)
        self.assertTrue(all(1 < p["lambd"] < 10000 and 4 < p["gamma"] < 64 for p in proposals))
        self.assertFalse({candidate_key(p) for p in proposals}.intersection(candidate_key(r["params"]) for r in records))

    def test_top_pairs_are_distinct_even_when_best_pair_has_several_retention_scores(self):
        records = [record(l, 9.0, r, score) for l, score in
                   [(1.0, 0.99), (100.0, 0.9), (10000.0, 0.8), (1000000.0, 0.7)]
                   for r in (0.05, 0.1)]
        proposals = propose_refinement(records, [0.05, 0.1], top_k=2)
        self.assertEqual({p["lambd"] for p in proposals}, {10.0, 1000.0})
        default = propose_refinement(records, [0.05, 0.1])
        self.assertEqual({p["lambd"] for p in default}, {10.0, 1000.0, 100000.0})

    def test_tied_pair_ranking_is_deterministic(self):
        records = [record(10000.0, 9.0, score=0.1), record(100.0, 9.0, score=0.9),
                   record(1.0, 9.0, score=0.9)]
        forward = propose_refinement(records, [0.05], top_k=1)
        # The tied stronger coefficient 100 is selected; its upper midpoint
        # 1000 would be absent if the weaker coefficient 1 were selected.
        self.assertEqual({p["lambd"] for p in forward}, {10.0, 1000.0})
        self.assertEqual(forward, propose_refinement(reversed(records), [0.05], top_k=1))

    def test_top_one_tie_prefers_stronger_lambd_then_smaller_gamma(self):
        records = grid([1.0, 100.0], [1.0, 100.0, 10000.0], retentions=(0.05,))
        records += [record(1.0, 10000.0, score=0.9),
                    record(100.0, 100.0, score=0.9),
                    record(100.0, 1.0, score=0.9)]
        proposals = propose_refinement(records, [0.05], top_k=1)
        # Refining (100, 1) gives these three unevaluated neighbors. Neither
        # the weaker-lambda winner nor the larger-gamma winner gives this set.
        self.assertEqual(triples(proposals), {
            (10.0, 1.0, 0.05), (10.0, 10.0, 0.05), (100.0, 10.0, 0.05),
        })

    def test_failed_evaluated_midpoint_is_excluded_and_still_a_neighbor(self):
        records = [record(1.0, 4.0, score=0.9), record(100.0, 4.0, score=0.5),
                   record(10.0, 4.0, score=math.inf, eligible=False)]
        proposals = propose_refinement(records, [0.05], top_k=1)
        self.assertEqual(len(proposals), 1)
        self.assertAlmostEqual(proposals[0]["lambd"], math.sqrt(10.0), places=14)
        self.assertNotIn(candidate_key(records[2]["params"]), {candidate_key(p) for p in proposals})

    def test_extreme_midpoints_remain_finite_without_product_overflow(self):
        records = [record(1e-300, 1.0, score=0.9), record(1e300, 1.0, score=0.5)]
        proposals = propose_refinement(records, [0.05], top_k=1)
        self.assertEqual(len(proposals), 1)
        self.assertTrue(math.isfinite(proposals[0]["lambd"]))
        self.assertAlmostEqual(proposals[0]["lambd"], 1.0, places=14)

    def test_singleton_axes_have_no_new_refinement_coordinates(self):
        self.assertEqual(propose_refinement([record(1.0, 2.0)], [0.05]), [])


class AdaptiveValidationTests(unittest.TestCase):
    def test_empty_malformed_or_all_failed_records_are_rejected(self):
        invalid = [None, [], {}, [None], [{}],
                   [record(1, 2, score=math.nan, eligible=True)],
                   [record(1, 2, score=math.inf, eligible=True)],
                   [record(1, 2, score="invalid")],
                   [record(1, 2, eligible=1)],
                   [record(1, 2, score=math.nan, eligible=False)]]
        for records in invalid:
            for function in (boundary_axes,
                             lambda rows: propose_expansion(rows, [0.05]),
                             lambda rows: propose_refinement(rows, [0.05])):
                with self.subTest(records=records, function=function), self.assertRaises(ValueError):
                    function(records)

    def test_invalid_controls_are_rejected(self):
        records = [record(1.0, 2.0)]
        for tolerance in (-1, math.nan, math.inf, True, "0"):
            with self.subTest(tie_tol=tolerance), self.assertRaises(ValueError):
                boundary_axes(records, tie_tol=tolerance)
        for retentions in (None, [], [0.0], [1.01], [math.nan], "0.05", [True]):
            for function in (propose_expansion, propose_refinement):
                with self.subTest(retentions=retentions, function=function), self.assertRaises(ValueError):
                    function(records, retentions)
        for count in (0, -1, 1.5, True, None):
            with self.subTest(top_k=count), self.assertRaises(ValueError):
                propose_refinement(records, [0.05], top_k=count)


def policy_grid(values=(1.0, 4.0, 16.0), gammas=None, retentions=(0.3,)):
    return {"kernel": ["rbf"], "lambd": list(values),
            "gamma": list(values if gammas is None else gammas),
            "retention_fraction": list(retentions)}


def policy_record(params, score, *, folds=None):
    result = {"params": params, "mean_score": score,
              "eligible": score is not None and math.isfinite(score)}
    if folds is not None:
        result["folds"] = [{"fold": i, "accuracy": score, "eligible": result["eligible"]}
                           for i in range(folds)]
    return result


def policy_pair(params):
    return tuple(float(format(params[name], '.12g')) for name in ('lambd', 'gamma'))


def drive_policy(surface, initial=None, config=None, max_stages=80):
    initial = policy_grid() if initial is None else initial
    config = _MODULE.AdaptiveSearchConfig(top_k=1) if config is None else config
    rows, state, stages = [], None, []
    for _ in range(max_stages):
        answer = _MODULE.next_search_stage(rows, initial, config, state)
        state = answer['state']
        if answer['review'] is not None:
            return answer, rows, stages
        stages.append((answer['phase'], copy.deepcopy(answer['candidates'])))
        rows.extend(policy_record(params, surface(params), folds=config.expected_folds)
                    for params in answer['candidates'])
    raise AssertionError('Policy did not finish within the synthetic stage count')


class ConditionalPolicyTests(unittest.TestCase):
    def test_second_midpoint_peak_requires_repeated_local_refinement(self):
        def surface(params):
            lambd, gamma = policy_pair(params)
            if gamma != 1.0:
                return 0.1
            if math.isclose(math.log10(lambd), 1.5, abs_tol=1e-11):
                return 1.0
            return 0.75 if lambd == 1000.0 else 0.5 if lambd == 1.0 else 0.25
        answer, rows, _ = drive_policy(surface, policy_grid((1e-6, 1.0, 1e6)))
        self.assertEqual(answer['review']['state'], 'accepted')
        self.assertEqual(answer['state']['best_mean_score'], 1.0)
        self.assertTrue(answer['state']['winner_bracket_resolved'])
        self.assertGreater(answer['state']['refinement_rounds'], 2)
        self.assertGreaterEqual(answer['state']['stable_rounds'], 2)
        self.assertLess(len(rows), 500)

    def test_missing_conditional_cross_is_repaired_before_expansion(self):
        initial = policy_grid()
        config = _MODULE.AdaptiveSearchConfig(top_k=1)
        surface = lambda p: .75 if policy_pair(p) == (8., 8.) else .5 if policy_pair(p) == (4., 4.) else .25
        rows = [policy_record(p, surface(p)) for p in _MODULE._parameter_grid(initial)]
        first = _MODULE.next_search_stage(rows, initial, config)
        self.assertEqual(first['phase'], 'local')
        rows.extend(policy_record(p, surface(p)) for p in first['candidates'])
        repair = _MODULE.next_search_stage(rows, initial, config, first['state'])
        self.assertEqual(repair['state']['pending']['kind'], 'conditional_repair')
        self.assertIn((16., 8.), [policy_pair(p) for p in repair['candidates']])
        self.assertIn((8., 16.), [policy_pair(p) for p in repair['candidates']])
        self.assertFalse(repair['state']['winner_bracket_resolved'])
        self.assertEqual(repair['state']['expansion_rounds'], 0)

    def test_other_retention_does_not_supply_conditional_neighbor(self):
        initial = policy_grid(retentions=(.1, .3))
        missing = (16., 4., .1)
        rows = [policy_record(p, .75 if policy_pair(p) == (4., 4.) and p['retention_fraction'] == .1 else .25)
                for p in _MODULE._parameter_grid(initial)
                if (*policy_pair(p), p['retention_fraction']) != missing]
        answer = _MODULE.next_search_stage(rows, initial, _MODULE.AdaptiveSearchConfig())
        self.assertEqual(answer['state']['pending']['kind'], 'conditional_repair')
        self.assertIn(missing, triples(answer['candidates']))
        self.assertTrue(any(item['issue'] == 'missing_conditional_neighbor'
                            for item in answer['state']['boundary_issues']))

    def test_tied_best_retention_specific_boundary_is_checked(self):
        initial = policy_grid(retentions=(.1, .3))
        rows = [policy_record(p, .75 if ((policy_pair(p) == (4., 4.) and p['retention_fraction'] == .1)
                    or (policy_pair(p) == (16., 4.) and p['retention_fraction'] == .3)) else .25)
                for p in _MODULE._parameter_grid(initial)]
        answer = _MODULE.next_search_stage(rows, initial, _MODULE.AdaptiveSearchConfig())
        self.assertEqual(answer['phase'], 'expansion-1')
        self.assertIn((64., 4., .3), triples(answer['candidates']))

    def test_invalid_outer_attempts_cannot_certify_interior_or_plateau(self):
        answer, rows, _ = drive_policy(lambda p: .5 if all(1 <= value <= 4 for value in policy_pair(p)) else None,
            policy_grid((1., 4.)), _MODULE.AdaptiveSearchConfig(refine=False, max_candidates=40))
        self.assertEqual(answer['review']['state'], 'needs_review')
        self.assertFalse(answer['state']['boundary_plateau_evidence'])
        self.assertEqual(answer['state']['stable_rounds'], 0)
        self.assertLessEqual(len(rows), 40)

    def test_remote_tied_edge_plateau_requires_completed_valid_same_slice_evidence(self):
        initial = policy_grid(retentions=(.1, .3))
        config = _MODULE.AdaptiveSearchConfig(refine=False)
        def surface(p):
            lambd, gamma = policy_pair(p)
            return .75 if gamma == 4 and ((lambd == 4 and p['retention_fraction'] == .1)
                or (lambd <= 4 and p['retention_fraction'] == .3)) else .25
        answer, rows, _ = drive_policy(surface, initial, config)
        self.assertEqual(answer['review']['state'], 'accepted')
        self.assertEqual(answer['state']['expansion_rounds'], 2)
        certificates = answer['state']['boundary_plateau_evidence']
        self.assertTrue(certificates)
        self.assertTrue(all(item['completed_outward_rounds'] == 2 for item in certificates))
        self.assertFalse(answer['review']['plateau_is_optimality_certificate'])
        damaged = copy.deepcopy(rows)
        target = certificates[0]['evidence_keys'][0]
        entry = next(row for row in damaged if _MODULE.parameter_hash(row['params']) == target)
        entry.update(eligible=False, mean_score=None)
        reopened = _MODULE.next_search_stage(damaged, initial, config, answer['state'])
        self.assertNotEqual((reopened['review'] or {}).get('state'), 'accepted')
        self.assertFalse(reopened['state']['boundary_plateau_evidence'])

    def test_stagnant_canonical_winner_at_boundary_is_never_waived(self):
        answer, _, _ = drive_policy(lambda p: .5, policy_grid((1., 4.)),
            _MODULE.AdaptiveSearchConfig(refine=False))
        self.assertEqual(answer['review']['state'], 'needs_review')
        self.assertIn('expansion_budget', answer['review']['stopping_reason'])
        self.assertTrue(answer['state']['boundary_issues'])

    def test_invalid_adjacent_attempt_is_repaired_between_winner_and_failure(self):
        initial = policy_grid((2., 4., 8.))
        rows = [policy_record(p, None if policy_pair(p) == (8., 4.) else .75 if policy_pair(p) == (4., 4.) else .25)
                for p in _MODULE._parameter_grid(initial)]
        answer = _MODULE.next_search_stage(rows, initial, _MODULE.AdaptiveSearchConfig())
        self.assertEqual(answer['state']['pending']['kind'], 'conditional_repair')
        self.assertTrue(any(4 < p['lambd'] < 8 and p['gamma'] == 4 for p in answer['candidates']))
        self.assertNotIn((8., 4., .3), triples(answer['candidates']))
        self.assertFalse(answer['state']['winner_bracket_resolved'])

    def test_two_actual_local_rounds_are_required_even_for_narrow_initial_bracket(self):
        answer, _, stages = drive_policy(lambda p: .75 if policy_pair(p) == (4., 4.) else .25,
                                         policy_grid((2., 4., 8.)))
        self.assertEqual(answer['review']['state'], 'accepted')
        self.assertEqual(answer['state']['stable_rounds'], 2)
        self.assertEqual([label for label, _ in stages], ['coarse', 'local', 'local-2'])
        self.assertEqual(answer['review']['retention_scope'], 'fixed_grid_choices_not_continuous_certification')

    def test_partial_stage_replays_only_missing_work_and_no_stability_credit(self):
        initial = policy_grid((2., 4., 8.))
        config = _MODULE.AdaptiveSearchConfig(top_k=1)
        rows = [policy_record(p, .75 if policy_pair(p) == (4., 4.) else .25)
                for p in _MODULE._parameter_grid(initial)]
        first = _MODULE.next_search_stage(rows, initial, config)
        rows.append(policy_record(first['candidates'][0], .25))
        import json
        restored = json.loads(json.dumps(first['state'], allow_nan=False))
        answer = _MODULE.next_search_stage(rows, initial, config, restored)
        self.assertEqual(answer['phase'], first['phase'])
        self.assertEqual(answer['state']['stable_rounds'], 0)
        self.assertEqual(answer['candidates'], first['candidates'][1:])
        self.assertEqual(first['state']['pending']['candidates'], restored['pending']['candidates'])

    def test_changed_baseline_fold_scores_reset_stability_even_with_same_mean(self):
        initial = policy_grid((2., 4., 8.))
        config = _MODULE.AdaptiveSearchConfig(top_k=1, expected_folds=2)
        rows = [policy_record(p, .75 if policy_pair(p) == (4., 4.) else .25, folds=2)
                for p in _MODULE._parameter_grid(initial)]
        first = _MODULE.next_search_stage(rows, initial, config)
        rows.extend(policy_record(p, .25, folds=2) for p in first['candidates'])
        rows[0]['folds'][0]['accuracy'], rows[0]['folds'][1]['accuracy'] = .2, .3
        answer = _MODULE.next_search_stage(rows, initial, config, first['state'])
        self.assertEqual(answer['state']['stable_rounds'], 0)

    def test_invalid_local_stage_does_not_earn_stability(self):
        initial = policy_grid((2., 4., 8.))
        config = _MODULE.AdaptiveSearchConfig(top_k=1)
        rows = [policy_record(p, .75 if policy_pair(p) == (4., 4.) else .25)
                for p in _MODULE._parameter_grid(initial)]
        first = _MODULE.next_search_stage(rows, initial, config)
        rows.extend(policy_record(p, None) for p in first['candidates'])
        answer = _MODULE.next_search_stage(rows, initial, config, first['state'])
        self.assertEqual(answer['state']['stable_rounds'], 0)
        self.assertFalse(answer['state']['winner_bracket_resolved'])

    def test_each_local_pair_keeps_all_fixed_retention_choices(self):
        initial = policy_grid(retentions=(.1, .3))
        rows = [policy_record(p, .75 if policy_pair(p) == (4., 4.) else .25)
                for p in _MODULE._parameter_grid(initial)]
        answer = _MODULE.next_search_stage(rows, initial, _MODULE.AdaptiveSearchConfig(top_k=1))
        grouped = {}
        for p in answer['candidates']:
            grouped.setdefault(policy_pair(p), set()).add(p['retention_fraction'])
        self.assertTrue(grouped)
        self.assertTrue(all(values == {.1, .3} for values in grouped.values()))

    def test_candidate_budget_holds_entire_stage(self):
        for limit in (8, 9):
            answer, rows, _ = drive_policy(lambda p: .75 if policy_pair(p) == (4., 4.) else .25,
                config=_MODULE.AdaptiveSearchConfig(top_k=1, max_candidates=limit))
            self.assertEqual(answer['review']['state'], 'needs_review')
            self.assertIn('candidate_budget', answer['review']['stopping_reason'])
            self.assertFalse(answer['candidates'])
            self.assertEqual(len(rows), 0 if limit == 8 else 9)

    def test_refinement_safety_cap_holds_instead_of_certifying_success(self):
        answer, _, stages = drive_policy(lambda p: .75 if policy_pair(p) == (4., 4.) else .25,
            policy_grid((2., 4., 8.)), _MODULE.AdaptiveSearchConfig(top_k=1, max_refinements=1))
        self.assertEqual(answer['review']['state'], 'needs_review')
        self.assertIn('refinement_budget', answer['review']['stopping_reason'])
        self.assertEqual([label for label, _ in stages], ['coarse', 'local'])

    def test_no_refine_still_requires_valid_bilateral_conditional_support(self):
        answer, _, stages = drive_policy(lambda p: .75 if policy_pair(p) == (4., 4.) else .25,
            config=_MODULE.AdaptiveSearchConfig(refine=False))
        self.assertEqual(answer['review']['state'], 'accepted')
        self.assertEqual(len(stages), 1)
        self.assertFalse(answer['state']['winner_bracket_resolved'])

    def test_configuration_and_schema_are_bound_across_resume(self):
        initial = policy_grid()
        first = _MODULE.next_search_stage([], initial, _MODULE.AdaptiveSearchConfig())
        with self.assertRaisesRegex(ValueError, 'different settings'):
            _MODULE.next_search_stage([], initial, _MODULE.AdaptiveSearchConfig(top_k=1), first['state'])
        state = copy.deepcopy(first['state'])
        state['schema_version'] = 999
        with self.assertRaisesRegex(ValueError, 'schema'):
            _MODULE.next_search_stage([], initial, _MODULE.AdaptiveSearchConfig(), state)

    def test_invalid_controls_are_rejected(self):
        for kwargs in ({'bracket_factor': 1}, {'stable_refinement_rounds': 0},
                       {'max_refinements': -1}, {'max_refinements': True},
                       {'expected_folds': 0}, {'allow_boundary_plateau': 'yes'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                _MODULE.AdaptiveSearchConfig(**kwargs)

    def test_duplicate_candidate_receipts_are_rejected(self):
        entry = record(4., 4., .3)
        with self.assertRaisesRegex(ValueError, 'Repeated candidate'):
            _MODULE.next_search_stage([entry, copy.deepcopy(entry)], policy_grid(), _MODULE.AdaptiveSearchConfig())

    def test_close_float_exhaustion_holds_without_reproposing_an_attempted_key(self):
        values = (1., 1.00000000000001, 1.00000000000002)
        initial = policy_grid(values)
        rows = [policy_record(p, .75 if p['lambd'] == values[1] and p['gamma'] == values[1] else .25)
                for p in _MODULE._parameter_grid(initial)]
        answer = _MODULE.next_search_stage(rows, initial, _MODULE.AdaptiveSearchConfig(top_k=1))
        self.assertEqual(answer['review']['state'], 'needs_review')
        self.assertIn('parameter_limit', answer['review']['stopping_reason'])
        self.assertEqual(answer['candidates'], [])


class CandidateEligibilityTests(unittest.TestCase):
    def test_actual_fold_evidence_must_be_complete_finite_eligible_and_consistent(self):
        base = policy_record(record(4., 4., .3)['params'], .75, folds=5)
        self.assertTrue(_MODULE.valid_record(base, expected_folds=5))
        for defect in ('missing', 'partial', 'empty', 'nonfinite', 'ineligible', 'duplicate',
                       'wrongmean', 'missingflag', 'missingindex'):
            changed = copy.deepcopy(base)
            if defect == 'missing':
                changed.pop('folds')
            elif defect == 'partial':
                changed['folds'].pop()
            elif defect == 'empty':
                changed['folds'] = []
            elif defect == 'nonfinite':
                changed['folds'][0]['accuracy'] = math.nan
            elif defect == 'ineligible':
                changed['folds'][0]['eligible'] = False
            elif defect == 'duplicate':
                changed['folds'][0]['fold'] = 1
            elif defect == 'missingflag':
                changed['folds'][0].pop('eligible')
            elif defect == 'missingindex':
                changed['folds'][0].pop('fold')
            else:
                changed['mean_score'] = .9
            with self.subTest(defect=defect):
                self.assertFalse(_MODULE.valid_record(changed, expected_folds=5))
                self.assertEqual(_MODULE.ordered_eligible([changed], expected_folds=5), [])

    def test_documented_sgd_tie_break_is_independent_of_input_order(self):
        records = [record(1., 4., .1, .8), record(4., 4., .1, .8),
                   record(4., 1., .1, .8), record(100., 1., .3, .8)]
        ordered = _MODULE.ordered_eligible(records)
        self.assertEqual(ordered, _MODULE.ordered_eligible(reversed(records)))
        self.assertEqual(policy_pair(ordered[0]['params']), (4., 1.))
        answer = _MODULE.next_search_stage(records, policy_grid(retentions=(.1, .3)),
                                          _MODULE.AdaptiveSearchConfig())
        self.assertEqual(answer['state']['best_parameters'], ordered[0]['params'])


if __name__ == "__main__":
    unittest.main()
