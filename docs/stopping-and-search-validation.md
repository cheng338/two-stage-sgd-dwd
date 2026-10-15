# Stopping and search validation

## Exact accuracy and grid arithmetic update — October 15, 2026

The adaptive example now compares the equally weighted mean of the saved
integer fold fractions exactly. For example, permutations of fold accuracies
0.1, 0.2, 0.3, 0.4 and 0.5 can produce different last bits under floating-point
averaging even though both exact means are 0.3. That difference previously
could override the parameter tie order or reset a stable refinement round.
Unequal validation-fold sizes do not justify pooling the observations: the
chosen statistic remains the mean of fold fractions. Exact ties retain the
order smaller retention, larger `lambd`, then smaller `gamma`.

Ranking, boundary and plateau checks, stability, resume state and qualified
exports use this same count evidence. A genuine improvement smaller than one
floating-point unit remains an improvement. `selection.json` records a
`cv_score` numerator/denominator identity alongside its rounded display score.
Invalid, partial or inconsistent counts are rejected; custom scalar-only
evaluators remain supported without inventing integer evidence. Configuration
schema 9 and policy schema 3 require a fresh directory for older checkpoints.
Existing experiment records remain historical evidence under their original
implementation. The public scikit-learn `make_search` wrapper is unchanged.

Grid proposals retain every previously successful coordinate. If ordinary
arithmetic refuses an outward proposal or interior midpoint, an exact-rational
fallback rounds the same geometric rule. Independent high-precision checks
cover coordinates near `1e-300`, `1e-8` and `1e308`, adjacent binary64 values,
true underflow/overflow limits, and distinct parameter identities. A missing
representable interior value still cannot earn refinement credit.

These are corrections to numerical comparisons and grid construction. They
do not relax the required search evidence, change the SGD methodology, or
establish new MNIST performance results. Current validation is summarized in
[VALIDATION.md](../VALIDATION.md).

## Historical stopping checks

Historical development baseline: `0.6.0.dev20261003`, October 3, 2026.
The stopping and search interfaces described here are retained in
`0.7.0.dev20261003`; the test counts and MNIST outcomes below concern the
recorded 0.6.0 development checks.

The source suite and a separately installed wheel each passed **369 tests and
364 subtests**. Controlled MNIST checks confirmed unchanged recorded numerical
results and refusal to qualify unresolved searches. **Successful adaptive
qualification on real MNIST remains unproven in these checks.**

## Implementation scope

The fitting interface now reports the chosen stopping rule, completed updates
and epochs, cap status, model usability, CV eligibility, and whether optimality
was checked. Completing a fixed epoch budget can produce a usable, CV-eligible
model without establishing convergence. The SGD update equations, sampling
defaults, and default 200 epochs per stage are unchanged.

Adaptive tuning follows the ensemble package's bounded search policy: valid
conditional neighbors, factor-two brackets, repeated refinement, and two
completed stable local rounds. Defaults allow three expansion rounds and 500
candidates. Budget exhaustion or unresolved evidence returns `needs_review`.
Only an accepted search exports qualified settings. Loading those settings
checks the saved evidence hashes and fitting controls. Ordinary fixed-grid
`make_search` remains a separate interface and does not apply this adaptive
qualification policy.

Deterministic regression tests cover successful acceptance, tied boundaries,
missing or invalid conditional neighbors, repeated refinement, stability
requirements, boundary-plateau evidence and its revocation, candidate and
refinement budgets, all-ineligible searches, interrupted-record recovery,
changed source/data identities, and altered selection evidence or controls.
These tests exercise acceptance without claiming that an actual MNIST search
has earned it.

## Real-MNIST settings

The candidate was compared with unchanged version `0.5.0`. All final runs used
the same frozen runtime and example source bytes. Per-file SHA-256 hashes,
source snapshots, exact observation IDs, folds, stage seeds, and input-file
hashes were recorded outside the distribution. Only the final runs attest to
the final source bytes; preliminary probes remain separate evidence.

Data were a balanced 256-observation subset of official MNIST training digits
2 and 3, with 128 observations per class, pixel values divided by 255, three
shuffled stratified folds, and random seed 314159. Retention was fixed at 30%.
SGD used batch size one, sampling with replacement, and fixed-epoch stopping.

The runtime was Python 3.12.14, NumPy 2.5.2, SciPy 1.18.0, scikit-learn 1.9.0,
and AOCL-BLAS 5.2.0 in `py12_df_aocl`. These small checks ran sequentially with
one native thread. No machine-level CPU partition or memory quota was applied.
An audit hook recorded opening only the two official-training IDX files. No
official-test partition was loaded or scored, and no final refit was performed.

The controlled grids used information from preliminary training-only probes.
Their CV accuracies are implementation evidence, not independent performance
estimates or recommendations for new defaults.

## Fixed-budget numerical parity

At `lambd=1e-6`, `gamma="scale"`, and retention 0.3, the baseline and candidate
agreed exactly on recorded accuracy, DWD loss, correct-prediction counts,
stage update counts, retained observation IDs and class counts, resolved gamma,
and stage seeds across nine fold evaluations at 2, 10, and 30 epochs. Mean CV
accuracies were respectively 0.500000, 0.750342, and 0.949156.

All 18 linear/kernel stage fits reported the completed fixed-epoch criterion,
a reached epoch cap, finite usable state, and CV eligibility. They did not
report an optimality check, established optimality, or convergence. This checks
the recorded outputs and diagnostics, not equality of unrecorded model state.

## Adaptive search outcomes

Every search used 30 epochs, `top_k=1`, factor-two conditional brackets, and
two required stable completed refinement rounds. The package default
`top_k=3` was deliberately reduced for these small controlled checks.

- Grid A: `lambd=[1e-6, 1e-4, .01]`, `gamma=[.001, .01, .1]`.
- Grid B: `lambd=[.001, .003, .01]`, `gamma=[.01, .02, .03]`.

Both grids used RBF kernels and retention 0.3.

| Case | Grid and limits | Candidates | Expansion / refinement / stable rounds | Best mean CV accuracy | Result |
|---|---|---:|---:|---:|---|
| Default expansion allowance | A; 500 candidates, 3 expansions | 34 | 3 / 1 / 0 | 0.949065 | `needs_review`: winning or tied boundary unresolved |
| Candidate-budget hold | A; 9 candidates, 3 expansions | 9 | 0 / 0 / 0 | 0.937392 | `needs_review`: next complete expansion exceeds budget |
| Refinement-budget hold | A; 500 candidates, 3 expansions, no local refinements | 15 | 2 / 0 / 0 | 0.937392 | `needs_review`: required refinement unavailable |
| Larger controlled allowance | B; 500 candidates, 20 expansions | 480 | 17 / 10 / 0 | 0.960830 | `needs_review`: next complete expansion exceeds budget |

The larger case repeatedly expanded boundaries and refined local intervals.
Its final canonical winner had valid factor-two brackets on both coefficients,
but tied weak-regularization boundary slices remained unresolved and stable
round credit was zero. It stopped at 480 because the complete next stage could
not fit within 500; stages are not partially evaluated to claim completion.

All four searches kept `final_evaluation_eligible=False`, and the qualified
settings loader rejected their unresolved selections. A small data set's tied
accuracy surface can therefore exhaust even a larger allowance. Ten completed
refinements alone do not meet the two-stable-round requirement. These results
demonstrate the conservative review behavior and a practical limitation of
the current policy. They do not establish successful qualification, optimizer
convergence, a global optimum, or official-test performance.

Replaying the saved records reproduced the terminal policy state exactly.
Of the 480 eligible candidates, 266 tied at the best accuracy. Stability had
reached one round three times, then reset when new tied representatives
required fresh boundary evidence. The next complete stage required 23
candidates, with only 20 slots remaining. This exposed no counter or recovery
defect. One retained difference from ensemble matters here: SGD breaks score
ties by smaller retention, stronger lambda, then smaller gamma; ensemble
retains the first candidate on a score tie. Both bind remote boundary-plateau
evidence to the current canonical leader, so SGD's changing tied leader can
require renewed evidence. Changing this interaction would require a separately
validated policy revision.
