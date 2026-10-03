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


if __name__ == "__main__":
    unittest.main()
