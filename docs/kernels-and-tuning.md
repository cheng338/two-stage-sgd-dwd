# Kernel choice and tuning: statistical rationale

## Recommended starting point

Use the standalone projected `LinearSGDDWD` as a linear baseline and RBF as
the main nonlinear kernel. A second-stage `kernel="linear"` is also available,
but that penalized RKHS model is not the same constrained estimator as the
first-stage linear baseline.

Kernel and cross-validation choices are starting points for empirical
assessment. They do not establish the benefit of screening or convergence
of finite SGD.

Jointly search the RBF bandwidth and regularization coefficient over positive
logarithmic ranges. Start with a manageable coarse grid, inspect whether the
winner lies on an endpoint, and expand or refine using training data only.
Exact powers, endpoints, and refinement budgets must be declared for the
study; no particular numeric grid is a universal statistical standard.
The defaults `lambd=1e-6`, `gamma="scale"`, and `retention_fraction=0.3`
match the ensemble package's starting values. These defaults
are not an empirical optimality claim. `"scale"` and `"auto"` are convenient
software rules, not substitutes for bandwidth assessment. Feature units affect
RBF distances, so choose
preprocessing deliberately. The prescribed MNIST protocol uses pixels / 255;
the generic standardized synthetic example does not change that protocol.

Polynomial kernels are optional when finite-order interactions make sense.
Other kernels require an appropriate feature geometry or domain rationale.
Searching every available kernel family needlessly can increase selection
variability and computational cost. API availability is not a recommendation
to include every family in a benchmark.

## Supported kernel contracts

Named kernels: `linear`, `rbf`, `poly`/`polynomial`, `laplacian`, `cosine`,
`chi2`, `sigmoid`, and `additive_chi2`. Polynomial degree must be a nonnegative
integer, its coef0 nonnegative, and built-in gamma finite and nonnegative.
Chi-square kernels require nonnegative features. Built-in gamma belongs only
at the top level; use `kernel_kws` for degree/coef0. Unsupported keywords
raise errors rather than being silently ignored.

Matrix callables follow the earlier DWD package convention:
`kernel(X_left, X_right, **kernel_kws)` returns an n_left by n_right matrix.
Only explicitly supplied keyword arguments reach the callable. Its top-level
gamma is ignored; a custom gamma belongs in its keyword dictionary. These are
matrix callables, not sklearn's scalar-pair callable convention. Callables
must be pickleable themselves to serialize a model using them.

Known PSD families need only analytic training diagonals. Sigmoid and
additive_chi2 are not guaranteed PSD, and custom callables are not assumed
valid. Those three cases require a one-time full training-Gram symmetry/PSD
check, O(m^2) temporary memory and an eigendecomposition, before SGD. A failing
check raises an error. The check allows small floating-point eigenvalue
roundoff but does not clip, symmetrize, shift, or repair the matrix. It cannot
certify a custom function on unobserved inputs; symmetric PSD behavior over
its intended domain remains the caller's contract.

No Gram matrix is retained after preparation. Training and prediction then
evaluate bounded blocks on demand. Optional `kernel_cache_bytes` stores only
training columns requested by single-observation updates, within a fixed
payload limit, and releases them when fitting exits. It is disabled by default
and restricted to the analytic built-in families, so enabling it does not
introduce an up-front training Gram computation. It is independent of the
MNIST runner's within-fold first-stage reuse. The optional native optimizer
likewise requires analytic built-ins for kernel fits. For other kernels,
`optimizer_backend="auto"` falls back to Python and `"native"` raises an error.
See [backend controls](optimizer-backends.md).

The SGD bound uses the maximum diagonal
on the actual kernel fitting subset, so changing kernels also changes the
safe step-size constant. Preparation time/work is reported separately from
optimizer and checkpoint work. A precomputed kernel is unsupported because
the two-stage screening procedure needs feature rows.

## Validation and scoring

Use stratified fivefold CV as a reasonable default for independent binary
observations. Five versus ten folds is a design choice, not a mathematical
requirement. Grouped, longitudinal, or time-dependent data require a splitter
that respects that structure; `make_search` accepts an explicit splitter.

Every candidate must use preprocessing, linear screening, retained-row
selection, and a kernel fit determined solely inside its training fold. The
generic helper refits the whole estimator. The dedicated MNIST runner can
reuse an identical first-stage fit and ranking within that fold, since the
linear stage does not depend on kernel parameters; cached and ordinary fits
are checked for equivalence. No fitted state is shared across folds.
The helper applies the same seeded folds across candidates and defaults to
one worker. It cannot
repair preprocessing or screening the caller already performed on all data.
Use the full original feature rows as input and keep fitted preprocessing in
a Pipeline. Gamma based on training variance is resolved separately for each
fit, before its retained subset is selected.

Choose the selection metric in advance. Misclassification error/accuracy is
a conventional classification target and accuracy is the helper default.
`neg_dwd_loss` instead targets the mean normalized margin loss and is useful
when following the implementation guide's loss-based calibration criterion. They can select
different hyperparameters. Neither scorer includes the training regularizer.
Balanced accuracy or another appropriate scorer can be supplied when the
prediction objective calls for it; class imbalance alone does not determine
which scientific target is appropriate.

Use fixed, adequate SGD epoch budgets during these searches. Keep seeds and
folds fixed for candidate comparisons, then assess sensitivity to SGD seeds
in a separate repeatability study. A tiny iteration budget may mostly compare
optimization transients. `make_search` executes its explicit fixed grid;
the separate `examples/tune_mnist.py` runner adds bounded adaptive expansion
and refinement. Neither tool automatically calibrates the epoch budget or
averages over independent optimization seeds.

The helper rejects externally forwarded `validation_data`, including pipeline
fit arguments. Checkpoint selection within CV requires another monitoring
split drawn only from each training fold. A CV scoring fold must not also
choose the fitted checkpoint. The final selected model is refitted on the
provided training data; an untrained fallback, wall-time-truncated fit, or
invalid candidate raises an error rather than becoming an eligible winner.
A finite trained fit that completes its epoch budget remains eligible without
an SGD optimality certificate.

Evaluate final predictive performance on an untouched test set. If there is
no separate test set, use nested CV to assess the entire tuning procedure.
The best score from a tuning search is optimistic as an assessment of its
selected model; see [Varma and Simon (2006)](https://link.springer.com/article/10.1186/1471-2105-7-91).
The package helper builds the inner parameter search; the study must supply
the outer evaluation. Record seeds, folds, full CV results, failure reasons,
and tuning cost with the experiment.

## Similar interface, explicit differences

The API borrows naming and sklearn conventions from `dwd` and `ensemble-dwd`.
Their numerical constants are not automatically scientifically appropriate for
this method, and their MM/SOCP optimizers are not dependencies.

| Choice | Published ensemble-dwd | Two-stage DWD |
| --- | --- | --- |
| Default retention | `sv_selection_ratio=0.3` (30%) | `retention_fraction=0.3` (30%) |
| Retention in MNIST preset | 10%, 15%, 20%, 25%, 30% | 10%, 15%, 20%, 25%, 30% |
| Ensemble settings | Ensemble size `M` and subsampling `theta` | Neither parameter applies to the two-stage method |
| Default RBF gamma | `"scale"` | `"scale"` |
| Default `lambd` | 1e-6 | 1e-6 |
| Norm coefficient | `lambd * ||h||^2` | Same `lambd * ||h||^2` convention |

`lambd` is the only regularization parameter, defaulting to 1e-6; `lambd_`
records its fitted value. The spelling avoids Python's reserved word `lambda`.
The constrained first-stage linear estimator has no regularization parameter.
The mean loss is over the actual retained set; changing retention changes the
fitted problem. The two packages also use different screening and fitting
procedures, so matching parameter values or grid ranges does not produce
equivalent predictors.

## MNIST parameter grids

`mnist_param_grid` is a two-stage starting grid for pixels divided by 255.
It fixes RBF and jointly considers these candidates:

| Parameter | Coarse values | Rationale |
| --- | --- | --- |
| `lambd` | `2.0**e`, e = -30, -22, -14, -6, 0 | Same coefficient endpoints as the ensemble preset |
| `gamma` | `10.0**e`, e = -8, -6, -4, -2, 0 | Same bandwidth endpoints as the ensemble preset |
| `retention_fraction` | 0.1, 0.15, 0.2, 0.25, 0.3 | Same retention fractions as the ensemble preset |

The coarse grid has **125 combinations**. `broad=True` uses lambd exponents
[-30, -26, -22, -18, -14, -10, -6, -2, 0] and all integer base-ten gamma
exponents from -8 through 0. The five retention choices remain fixed, giving
9 x 9 x 5 = **405 combinations**.
These ranges are declared design choices, not established optimal ranges.
This helper's historical `broad` option means more candidates; it does not
expand endpoints. Inspect endpoint winners and extend the ranges explicitly
using training-only CV. Use the same folds for coarse and refined searches.

The helper returns `lambd` directly in the shared norm-coefficient convention.
The dissertation's coefficient conversion remains documented in the
[alignment note](dissertation-batch-alignment.md); it does not determine these
new exploration ranges.
One-class retained subsets raise `SelectionError` rather than being repaired
by quotas or assigned artificial scores. Tiny datasets may require an
explicitly different retention range. The preset does not alter class counts
or guarantee that every candidate is eligible on every fold.

The direct-fit default is 30%, matching the ensemble package. The original
Chapter 3 implementation guide's 5% remains valid as an explicit API setting.
It is omitted from the recommended presets after its poor performance in the
recorded 7-versus-8 training-only checks; this does not establish that 5% fails
on every dataset.
Searching retention changes that guide's fixed-5%, nine-candidate study and
must be reported as an alternative protocol.
Specify the grid in the dissertation's experiment protocol. Select parameters
using training data, then evaluate their performance on independent data.

## Training-only adaptive MNIST runner

`examples/tune_mnist.py` uses a fixed pixels/255 transformation and a balanced
subset of the official training partition. It never opens the official test
partition. Defaults are 2,000 observations, stratified fivefold CV, 200 epochs
per stage, batch size one, independent sampling with replacement, a common
random-state seed, and one numerical-library thread. All candidates share the
same saved folds and stage seeds. Changing an epoch budget creates a different
optimization comparison; the default does not certify convergence.

Run the search explicitly with `--run`, `--data-dir`, and `--output-dir`;
without `--run`, the command prints settings only.

The linear first stage and its ranking depend on the fitting fold and seed,
not on lambd or gamma. The runner caches them within each fold and reuses the
ranking across retention fractions. Equivalence checks compare this route
with ordinary complete two-stage fits. The cache must never reuse a fit or
ranking across different training folds, seeds, preprocessing, or linear
training settings. Gamma="scale" is still determined from the complete
training fold, before retention.

The inherited defaults `lambd=1e-6, gamma="scale", retention_fraction=0.3`
form a separate control evaluated on the same folds. They are not inserted
into the positive numeric grid used for adaptive proposals. Selection among
grid candidates uses mean fold accuracy, with exact score ties broken by
smaller retention, larger lambd, then smaller gamma. Every candidate must
complete all folds with eligible trained models; failures remain recorded.
Earlier exploratory runs used an explicit 5% provisional control; their saved
records retain that setting. The 30% default aligns the software conventions
with the ensemble package and is not selected from those exploratory scores.

The adaptive policy matches the ensemble workflow's default three expansion
rounds and 500 distinct grid candidates, excluding the control. Its search
criterion requires valid conditional lower and upper neighbors of the chosen
winner within a factor of two on each coefficient and two stable completed
local rounds. Geometric refinement repeats until that criterion is met or a
budget prevents further work. The optional `--refinement-rounds` is a safety
cap; reaching it never substitutes for qualification. Retention is compared
only at the fixed choices in the supplied grid; the search does not certify a
continuous retention optimum.

Conditional support holds every other parameter fixed, including retention.
Coordinates evaluated under another gamma, lambda, or retention choice do not
establish that support. Missing combinations are filled, invalid neighbors are
probed inward, and unresolved outer boundaries are extended. Tied-best tuples
also require support. An outer tied edge can use documented same-slice plateau
evidence from two valid completed outward levels, but the canonical winner
still requires a bilateral bracket. Failed or missing fits never supply a
plateau certificate. None of these checks establishes a global optimum.

Every proposed stage must fit the remaining candidate budget in full.
Unresolved searches finish as `needs_review`, retaining provisional winners
and concrete reasons. They cannot export qualified final-evaluation settings.
Accepted searches finish as `accepted`, with `qualified-selection.json`
binding parameters, fitting controls, source identity, data, folds, and scores.
`load_qualified_selection` checks those records before returning the settings.
The runner never performs final fitting or opens the official test partition;
a later evaluation must reserve its own independent data and use the locked
settings. Search acceptance does not establish SGD optimality.

`--resume` reuses committed candidates and pending stages only with identical
configuration, recorded Python/NumPy/SciPy/scikit-learn versions, source hashes,
input hashes, observation IDs, and folds. An uncommitted candidate restarts
from the beginning; this is not an optimizer-state resume.
Checkpoint formats from version 0.5.0 cannot be mixed with this policy.
Runtime/numerical failures stop the run and may be retried by explicit resume;
one-class retained subsets remain recorded ineligible candidates. Completed
fixed-epoch fits are eligible even though SGD optimality remains unchecked.

A boundary plateau at 200 epochs may reflect the optimization budget;
confirmation with larger budgets and independent seeds is required before
recommending new defaults. Computational cost pilots alone do not supply that
evidence.

Ordinary `make_search` remains fail-fast fixed-grid sklearn CV. It refits its
winner automatically and has no adaptive boundary qualification. It rejects
shared `validation_data`, wall-time-truncated fits, and untrained fallbacks.
Do not confuse that helper with this separately invoked adaptive workflow.

The [MNIST development report](mnist-exploration.md) records the completed
7-versus-8 search, lambda endpoint extensions, bandwidth refinement, retention
checks beyond 30%, and fixed-candidate checks across budgets, seeds, and sample
sizes. Its focused follow-up grid is task-specific. The historical 150-candidate
initial grid and 486-candidate broader preset definition included 5%; their
records are preserved. Current recommended presets contain 125/405 candidates
and omit 5%, while the inherited software defaults remain unchanged.
