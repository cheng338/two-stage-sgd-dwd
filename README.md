# Overview

`two-stage-sgd-dwd` implements linear and kernel Distance Weighted Discrimination
(DWD) for binary classification using stochastic gradient descent (SGD). Its
two-stage procedure first fits a linear classifier, then trains a kernel
classifier on observations selected near the linear decision boundary.

The following methods are implemented:

- **Linear DWD** (`LinearSGDDWD`): fitted using projected SGD with a unit-ball
  constraint on the slope and a separate free intercept.
- **Kernel DWD** (`KernelSGDDWD`): fitted using SGD in an RKHS with penalty
  `lambd * ||h||^2` and a free intercept.
- **Two-stage DWD** (`TwoStageDWDClassifier`): fits linear DWD, selects
  observations globally by absolute linear score, then fits kernel DWD on the
  selected subset. Both DWD fits use SGD.

The package implements its own SGD solvers. It does not call `dwd`, a SOCP
solver, CVXPY, or scikit-learn's `SGDClassifier`.

Chang Cheng developed this package from his original research scripts
accompanying his dissertation. The two-stage method follows Chapter 3,
**"Our Earlier Attempt on Kernel DWD"**, of:

> Cheng, Chang (2026). *Overcoming the Curses of Dimensionality and
> Kernelization: Scalable DWD and Supervised NMF*. Doctoral dissertation
> manuscript, Purdue University, Department of Statistics.

# Installation

Python 3.10 or newer is required. From this source directory:

```console
python -m pip install .
python -m pip install ".[test]"
python -m pytest
python examples/synthetic_example.py
```

Dependencies are NumPy, SciPy, and scikit-learn. Validation environments are
documented in [VALIDATION.md](VALIDATION.md). The package does not require
Conda or a particular BLAS vendor.

The compiled scalar optimizer is optional. A compatible native wheel includes
it; a source build needs a suitable C compiler to build it. Setting the build
environment variable `SGD_BUILD_ACCEL=0` requests a portable Python-only build.
The Python backend remains the default in either distribution. Native binary
availability depends on the operating system and architecture; see
[backend and cache details](docs/optimizer-backends.md).
Native builds on CPython 3.10 use its version-specific ABI. Builds on CPython
3.11 and newer retain the Python 3.11 stable ABI (`cp311-abi3`). A Python 3.11
stable-ABI wheel cannot be installed on Python 3.10; that interpreter needs a
matching native build or the portable distribution.

The repository's Windows CI workflow builds and installs native and portable wheels on
Python 3.10 with NumPy 1 and Python 3.12 with NumPy 2. It checks installed
package and extension locations, version agreement, and actual backend use
before running the full suite from a separate test bundle without `src`.
Native jobs require compiled fits; portable jobs require an absent extension
and working automatic Python fallback. Each job saves its wheel, dependency
versions, installed-package check, and test report.

# Example

```python
from two_stage_sgd_dwd import TwoStageDWDClassifier

model = TwoStageDWDClassifier(
    retention_fraction=0.3,
    lambd=1e-6,
    gamma="scale",
    random_state=20261001,
)
model.fit(X_train, y_train, validation_data=(X_validation, y_validation))
scores = model.decision_function(X_test)
predictions = model.predict(X_test)
print(model.selected_ids_, model.selected_class_counts_)
print(model.stop_reason_, model.model_status_)
```

# Kernel choice and parameter tuning

Linear and Gaussian/RBF kernels are the primary recommended choices. RBF is
the default; polynomial, Laplacian, cosine, chi-square, and matrix callables
are available for problems where their geometry is appropriate. Supporting a
kernel does not imply that searching every family is beneficial.

The interface follows `ensemble-dwd`: `kernel`, top-level `gamma`, and
`kernel_kws` for other arguments, such as polynomial `degree` and `coef0`.
Gamma accepts a nonnegative number, `"auto"` (1 / feature count), or `"scale"`
(1 / (feature count * training variance), with 1 for constant data).
Two-stage fitting resolves scale on the complete training fold before selecting
rows. The default is `gamma="scale"`, as in `ensemble-dwd`; it is a
data-dependent starting value, not a statistically optimal bandwidth guarantee.

`lambd` is the sole regularization parameter in version 0.8.0 and
means `mean_loss + lambd * ||h||^2`, as in `dwd`. Python reserves the word `lambda`,
so the argument is spelled `lambd`. Its default is **1e-6**, matching the
ensemble package's starting value. The fitted value is `lambd_`.
The retention default is **0.3 (30%)**, also matching `ensemble-dwd`.
The original Chapter 3 choice of 5% remains available through an explicit
`retention_fraction=0.05`, but is excluded from the recommended presets.
The constrained first-stage `LinearSGDDWD` has no regularization parameter.
The [alignment note](docs/dissertation-batch-alignment.md) explains conversion
from the dissertation's half-norm coefficient.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from two_stage_sgd_dwd import make_search

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("dwd", TwoStageDWDClassifier(random_state=17)),
])
# Illustrative ranges: choose and check their coverage for the actual data.
grid = [
    {"dwd__kernel": ["linear"], "dwd__lambd": [2.0**-22, 2.0**-14, 2.0**-6]},
    {"dwd__kernel": ["rbf"], "dwd__lambd": [2.0**-22, 2.0**-14, 2.0**-6],
     "dwd__gamma": [1e-6, 1e-4, 1e-2, 1.0]},
]
search = make_search(pipeline, grid, cv=5, scoring="accuracy", random_state=17)
search.fit(X_train, y_train)
print(search.best_params_, search.best_score_)
predictions = search.predict(X_test)  # only after parameter selection
```

`fit()` never tunes implicitly. `make_search` requires an explicit parameter
grid and returns a sklearn `GridSearchCV` object with standard `best_*` and
`cv_results_` attributes. Its defaults are seeded stratified fivefold CV,
accuracy, one worker, and failure on invalid fits. For lowest unpenalized
validation DWD loss, pass the exported `neg_dwd_loss` scorer instead.
The choice of metric should be fixed before searching.

Every fold refits any supplied preprocessing, linear screening, and the kernel
stage using only its training rows. The helper rejects `validation_data`; fixed epoch
budgets avoid using the scoring fold to select SGD checkpoints. Early stopping
within CV requires another monitoring split inside each training fold.
Reserve independent test data for final evaluation, or use nested CV if no
such test set exists. The winning tuning score is not a final error estimate.

`mnist_param_grid(prefix="", broad=False)` supplies a two-stage starting grid
for MNIST pixels divided by 255. It fixes RBF and jointly searches:

| Parameter | Starting candidates |
| --- | --- |
| `retention_fraction` | 0.1, 0.15, 0.2, 0.25, 0.3 |
| `lambd` | `2.0**e` for e = -30, -22, -14, -6, 0 |
| `gamma` | `10.0**e` for e = -8, -6, -4, -2, 0 |

There are **125 candidates**. `broad=True` uses lambd exponents
[-30, -26, -22, -18, -14, -10, -6, -2, 0] and every integer gamma exponent
from -8 through 0, giving **405 candidates**. Both grids use the ensemble
package's coefficient ranges and retention fractions. The screening procedures
differ, so matching ranges does not imply equivalent fitted models.
Two-stage DWD has no ensemble `M` or
`theta` parameter. The ranges are starting choices whose coverage must be
checked using training data.
The presets omit 5% following its poor performance in the recorded 7-versus-8
training-only checks. The earlier 150-candidate grid and 486-candidate broader
preset definition included 5%; their historical records remain unchanged.
Use `prefix="dwd__"` for a Pipeline. The grid tunes `lambd` directly.

```python
from two_stage_sgd_dwd import mnist_param_grid

mnist_search = make_search(
    TwoStageDWDClassifier(random_state=17), mnist_param_grid(), cv=5,
    random_state=17,
)
# Call mnist_search.fit(X_train, y_train) to run the search.
```

The grid is never silently applied by `make_search`. The estimator's direct-fit
retention default is 30%, but this search treats retention as a tuning
parameter. This is an alternative to the implementation guide's fixed-5%, nine-candidate
calibration; using it in the dissertation must be reported as a protocol
change. Select parameters using training data, then evaluate their performance
on independent data as described above.

## Training-only adaptive MNIST tuning

[examples/tune_mnist.py](examples/tune_mnist.py) runs an explicit search on a
balanced subset of the official training partition. It divides pixels by 255
and never opens the official test partition. For example:

```console
python examples/tune_mnist.py --data-dir path/to/idx/files --output-dir results/mnist-7-8 --digits 7 8 --train-size 2000 --run
```

Without `--run`, the command only prints the proposed search settings.
Defaults are fivefold stratified CV, 200 SGD epochs per stage, the same folds
and common stage seeds across candidates, and one numerical-library thread.
Within-fold caching reuses the first-stage fit and selection ranking; its
equivalence to ordinary complete fits is checked, and fitted state never
crosses fold boundaries.

The tuner defaults to `--optimizer-backend python --kernel-cache-bytes 0`.
To use the optional compiled optimizer and a 64 MiB lazy kernel-column cache
per kernel fit, add `--optimizer-backend native --kernel-cache-bytes 67108864`
to the command above. `--optimizer-backend auto` permits Python fallback when
the extension is unavailable. These options preserve the folds, stage seeds,
sampling, epoch budgets, and within-fold screening reuse. Saved fold records
report the actual backends and cache diagnostics. Resume and qualified-setting
checks bind the requested controls and, for `auto` or `native`, extension
availability and binary content; changing them requires a new output directory.

The runner evaluates `lambd=1e-6, gamma="scale", retention_fraction=0.3` as a
separate control, then searches the numeric grid. Its defaults match the
ensemble adaptive workflow: at most **three expansion rounds** and **500
candidates**, excluding the control. Lambda and gamma are refined around the
best three distinct pairs until the selected winner has valid conditional
neighbors within a factor of two on both sides and **two completed local
rounds are stable**. Retention is compared at the fixed supplied grid values.

Neighbors must be evaluated at the same values of every other parameter.
Invalid fits cannot establish a bracket, and missing conditional combinations
are filled before further expansion. Tied boundary winners are checked too;
a remote tied edge can be waived only by recorded, repeated valid same-slice
outward evidence. The selected winner always needs its own bilateral bracket.

A complete proposed stage must fit the remaining budget. Unresolved bounds,
invalid support, or exhausted budgets produce `state="needs_review"`, with
provisional settings and reasons. Successful searches produce `state="accepted"`
and `qualified-selection.json`. The example's `load_qualified_selection`
checks the saved evidence before exporting those locked settings. Neither
state automatically starts a final fit or accesses held-out data.

Earlier exploratory runs used an explicit 5% provisional control; their
records retain that setting. The 30% software default aligns with the ensemble package;
this alignment does not establish 30% as the best-performing retention.

`--refinement-rounds` is an optional safety cap, not a success condition; its
default is unlimited within the candidate budget. Saved folds, settings,
implementation snapshots, candidate records, and policy state support
`--resume` defaults to matching settings, numerical runtime identity, and data.
The runtime record includes Python/platform and scientific-package versions,
package entrypoint and loaded scientific-extension hashes, and the loaded
BLAS/OpenMP libraries' metadata, file hashes, and actual thread counts. This
prevents unrequested mixing of environments such as AOCL and MKL even when their package
versions agree. Runtime checks and the evaluator use the requested `--threads`
setting; loading a qualified selection temporarily applies its recorded thread
setting and then restores the caller's setting. The identity uses portable
filenames, so moving unchanged environments alone does not invalidate it.
The evaluator checks runtime identity at candidate boundaries before reusing
screening fits. File hashes are cached while file stat stamps agree; no hashing
occurs inside SGD updates. Keep the environment unchanged during a search:
loading additional scientific extensions can also change its identity. This
works best in a dedicated CV process with needed optional modules imported
before starting. Failed or interrupted candidates discard screening caches;
after a runtime mismatch, restore the recorded environment before retrying. The
guard covers loaded scientific binaries and identified BLAS/OpenMP libraries,
not every dependency file or code already held in memory; it is not a tamper
attestation or a guarantee of bitwise agreement across different hardware.

For an intentional change between processes, `--resume --allow-blas-change`
permits compatible AOCL/BLIS, MKL or OpenBLAS numerical builds. Python version,
platform, NumPy/SciPy/scikit-learn/joblib/threadpoolctl versions, loaded scientific
module names and ABI filenames, package source, native optimizer bytes, data,
folds, seeds, parameters and thread counts must still match. NumPy, SciPy and
scikit-learn build fingerprints can differ: this is permission to migrate
between pinned-version numerical builds, not evidence that only BLAS machine
code changed. Numerical differences can change selected observations or their
order, especially near ties, and alter class predictions, CV scores and candidate
rankings. Matching versions do not guarantee the same selected model.

Completed records, their receipts and the original configuration remain
unchanged. The first completed candidate under a different runtime records the
transition in its existing receipt chain. A new evaluator fits outstanding
candidates without reusing a prior process's screening cache; its runtime must
stay unchanged throughout that process. Mixed-runtime searches require the flag
on every subsequent resume, including a return to the original environment.
Use `load_qualified_selection(path, allow_blas_change=True)` to consume such
evidence. Exports identify the runtime of the selected candidate and the complete
runtime history. Opt-in loading also returns the consuming runtime separately,
without rewriting the export or inventing a fitting transition. Compare timing
measurements only within one runtime; combining runtimes does not establish
comparable fit or prediction performance. This mode applies only to searches
started with the current checkpoint schema; older records are not rewritten.

One coordinator at a time may use an output directory, including completed
resume and qualified-selection export. A nonblocking operating-system lock
rejects overlapping access; different output directories remain independent.
The `.run.lock` file stays in place and does not itself indicate a live run.
Process exit or interruption releases its lock; do not delete the file to
force access. Saved and newly evaluated candidate records must have unique
exact parameter identities and consistent complete fold evidence. Legitimate
failed folds remain recorded and ineligible for selection.
Each committed candidate also has an inline SHA256 receipt binding its complete
JSON content, exact parameter identity, position, and preceding receipt to the
saved configuration and folds. Resume verifies all complete records before
repairing an interrupted final line or fitting another candidate. Changes to
scores, diagnostics, ordering or run identity are rejected even if the edited
JSON remains valid. Whitespace and dictionary key order do not affect content
checksums. These receipts detect inconsistent edits; they are not signatures
and do not independently detect replacement by an older complete valid prefix.
Keep original run archives when preserving an audit trail.
Duplicate JSON fields are rejected to avoid ambiguous checkpoint contents.
Resume and qualified-selection loading require writable access to the lockfile
(and to its directory when creating it). Use a writable copy of a read-only
archive; inaccessible locks fail closed instead of permitting unlocked access.
The adaptive example preserves normalized floating-point parameter values and
uses exact candidate identities; nearby values are not rounded into one
candidate. This can increase the number of candidates relative to older
searches, within the same configured budgets. Older example checkpoints and
qualified selections lack the current identity scheme and cannot be resumed or
loaded by this version. Keep those records and start a fresh output directory.
Ordinary `make_search` remains fixed-grid CV and does not apply this adaptive
review gate.

Search acceptance is a finite empirical stopping decision. It does not
establish solver convergence, a global optimum, or final test accuracy.
Larger-budget and independent-seed checks remain useful when assessing SGD
settings. The [earlier MNIST exploration](docs/mnist-exploration.md) remains
historical evidence for version 0.5.0; it is not a validation of this branch.

Exploratory CV scores and checks across SGD seeds do not establish a screening
advantage or replace replicated evaluation on independent data. The starting
defaults are inherited choices, not universally optimal settings.

See [statistical rationale and kernel details](docs/kernels-and-tuning.md)
and the [small runnable search example](examples/kernel_search_example.py).
An explicitly requested 5% on tiny folds may retain only one class; that is a
recorded selection failure. The synthetic example uses the 30% retention default.

# Single-observation, mini-batch, and full-batch updates

All three estimators accept the same update controls:

```python
single = TwoStageDWDClassifier(batch_size=1, random_state=17)
mini = TwoStageDWDClassifier(batch_size=32, random_state=17)
full = TwoStageDWDClassifier(batch_size="full", random_state=17)
shuffled = TwoStageDWDClassifier(
    batch_size=32, sampling="shuffle", random_state=17
)
```

The defaults are `batch_size=1, sampling="with_replacement"`, matching the
single-observation sampling used in Chapter 5 of the October 4, 2026
dissertation. At fixed hyperparameters and seeds, the single-point update
retains the earlier solver's formula and sequence.
Independent uniform draws
allow repeats and omissions within an epoch. An integer batch size equal
to the fitting-set size is a random multiset, whereas `"full"` is the complete
fitting set.

Optional `sampling="shuffle"` generates a fresh random permutation each epoch
and partitions it into nonoverlapping batches, visiting each row once. Under
either stochastic policy the last batch may be smaller. A requested batch size
larger than the stage's fitting set is capped to that size. `batch_size="full"`
uses all rows once in one gradient update per epoch without sampling RNG draws.

Every batch averages loss gradients using its actual length, evaluates them
at the same old model, and commits one parameter update. Linear fitting projects
the averaged slope step once. Kernel fitting adds its RKHS regularizer once.
Both intercepts remain unrestricted and unregularized. Batch size does not
rescale the loss, penalty, or learning rate automatically. Larger batches make
fewer parameter updates per epoch, so equal epochs do not imply equal progress.

In a two-stage fit, each stage uses its actual data size and the requested
batch policy. The optimizer's `batch_size` is separate from
`prediction_batch_size`, which only controls kernel scoring memory. Full-batch
kernel updates use bounded scoring blocks with frozen parameters and retain
no full Gram cache. An interrupted batch is not partially committed.
Changing a fitted `TwoStageDWDClassifier` with
`set_params(prediction_batch_size=...)` controls subsequent prediction blocks
without refitting or changing its fitted kernel child's parameter. The size
must remain a positive integer; invalid values are rejected at prediction time.

New `KernelSGDDWD` and `TwoStageDWDClassifier` instances default to
`prediction_backend="blas"`, using guarded matrix-product distances for
positive-width RBF prediction. Set `prediction_backend="direct"` to use
direct squared distances. Other kernels and zero-width RBF keep their existing
scoring paths. Training, screening and validation checkpoints are unchanged;
no compiled SGD extension is required.
The setting can be changed on a fitted model without refitting. Older saved
models that lack this option continue to use direct prediction, and explicitly
saved backend choices are preserved. Floating-point scores can differ between
backends. Cancellation and near-zero-score fallbacks recompute directly; they
are engineering safeguards, not a universal label-equivalence certificate.
See [prediction details](docs/optimizer-backends.md#rbf-prediction) and the
AOCL/MKL comparisons in [VALIDATION.md](VALIDATION.md).

Sections 3.1 and 3.2 of the October 4 dissertation explicitly permit averaged
batch updates, including full-gradient updates. The reported Chapter 5 study
uses single-observation updates with replacement; it does not use optional
shuffling. See the [alignment note](docs/dissertation-batch-alignment.md) for
the batch equations, learning-rate clock, and sampling and proof distinctions.
The package's estimators accept binary targets. Multiclass prediction can be
composed externally using scikit-learn's existing one-vs-one wrapper.

# Multiclass composition

[The multiclass example](examples/multiclass_example.py) wraps
`TwoStageDWDClassifier` in scikit-learn's
[`OneVsOneClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html).
For `K` classes, it fits `K * (K - 1) / 2` unchanged binary estimators on raw
feature arrays and delegates voting and score aggregation to scikit-learn.
The wrapper's decision scores combine votes and confidence for tie breaking;
they are not DWD margins or class probabilities.
The example places training-only standardization before the wrapper in a
pipeline and uses serial pair fits with one native thread. Its tiny synthetic
dataset illustrates composition; its accuracy is not a performance study.
The example's `retention_fraction=1.0` avoids single-class selections in this
small demonstration. It is not a retention recommendation or a default change.

The example's `pair_diagnostics(model)` helper copies each successful pair's
sample and retention counts, status, stage diagnostics and seeds. The reported
class pair uses original labels; selected class counts follow that order.
Scikit-learn passes pair-local labels `0` and `1` to the binary estimators.
Their selected indices and default observation IDs refer to pair-local rows,
not the original multiclass data. The helper accepts the fitted wrapper itself,
as shown in the example.

Smaller retention fractions keep the existing global boundary-selection rule:
single-class selections raise `SelectionError` without redraws or class quotas.
If a pair fit fails, the wrapper propagates the error and does not provide a
public record of the failing clone's diagnostics. Discard that wrapper and
create a new one before retrying; do not inspect it as a completed fit.
Do not pass multiclass `validation_data` directly through the wrapper: it does
not filter and relabel that tuple for each pair. This example uses fixed epochs
and passes no fit metadata.

# Optimizer backends and lazy kernel columns

All three estimators accept `optimizer_backend="python"` by default.
`"auto"` selects the optional compiled scalar optimizer when supported and
otherwise uses Python. `"native"` requires the compiled optimizer: unsupported
fitting controls raise `ValueError`, and an unavailable extension raises
`ImportError`.

The compiled optimizer uses guarded direct BLAS calls for eligible dot
products, with NumPy fallback for unsupported interfaces and numerical
exceptions. This training optimization is independent of `prediction_backend`.
See [backend details](docs/optimizer-backends.md#direct-blas-calls-in-native-fitting)
and [measured gains and limitations](VALIDATION.md).

The native path requires `batch_size=1`, C-contiguous aligned feature arrays,
and no `validation_data`, `checkpoint_callback`, or time limit. This includes
`max_seconds` and, for the two-stage linear fit, `linear_max_seconds`. Supplying
validation data is unsupported even when `stopping="fixed"`. Both independent
sampling with replacement and shuffled epochs are supported. Native kernel
fits require the analytic built-in kernels listed below. Other configurations
continue to work with `"python"` or an automatic Python fallback.

`optimizer_backend_` records the actual backend and
`optimizer_backend_reason_` records why it was selected. The two-stage model
applies the requested setting to both stages, exposes their individual
diagnostics, and reports `"mixed"` when their actual backends differ. A native
numerical failure invalidates the fit; it is not retried using Python.

For repeated single-observation kernel updates, `KernelSGDDWD` and
`TwoStageDWDClassifier` optionally reuse training kernel columns:

```python
model = TwoStageDWDClassifier(
    batch_size=1, kernel="rbf", kernel_cache_bytes=64 * 1024**2,
    optimizer_backend="auto", random_state=17,
)
```

The default `kernel_cache_bytes=0` disables caching. A positive integer is a
byte limit on retained float64 column array payload. Each column is computed
only when its training observation is drawn, using the original column
calculation. Later draws reuse it while retained; the least recently used
column is evicted when storage is full. Nothing precomputes a full Gram matrix,
and all cached arrays are released when fitting exits, including failed fits.
A limit smaller than one column stores no columns. Python metadata, allocator
overhead, kernel working memory, and the rest of the model are outside this
payload limit.

Enabled caching requires `batch_size=1` and one of `linear`, `rbf`, `poly`,
`polynomial`, `laplacian`, `cosine`, or `chi2`. Other batch modes and kernels
remain available with the default zero cache. Validation and prediction use
their existing bounded blocks and are not cached. The two-stage cache applies
only to its retained kernel training set; the linear stage is unchanged.

Cache diagnostics are `kernel_cache_hits_`, `kernel_cache_misses_`,
`kernel_cache_evictions_`, `kernel_cache_peak_bytes_`, and
`kernel_cache_capacity_columns_`; they are also available on the two-stage
estimator and in its kernel-stage diagnostics. Disabled-cache counters are zero.
`kernel_evaluation_count_` counts values actually computed during optimization
and validation, so a cache hit does not increase it. Kernel preparation counts
remain separate. At fixed data, settings, seed, and update budget, cache reuse
preserves the update sequence and values. A wall-time limit can stop faster and
slower implementations at different updates; it is not an equality guarantee.

These controls do not change the SGD objective, sampling policy, or number of
updates. Correctness checks compare complete fitted results and RNG state in
the same numerical environment. They do not promise the same runtime or
bitwise agreement across different BLAS libraries, thread settings, hardware,
or compilers. Assess speed with complete cold fits, including preprocessing,
screening, kernel preparation, lazy cache filling, and prediction as applicable.
Scalar-update microbenchmarks alone do not establish an end-to-end advantage.
See [backend and cache details](docs/optimizer-backends.md) for limits and
verification scope.

# Data and labels

Supply finite dense numeric features and exactly two class labels. Labels may
be strings or numbers; `classes_[0]` maps internally to -1 and `classes_[1]`
to +1. A zero decision score predicts `classes_[1]`. This zero convention follows
the dissertation and may differ from other DWD packages.

The estimators do **not** scale data automatically. The original October 1
guide and standalone `examples/tune_mnist.py` use float64 pixels divided by 255.
The completed Chapter 5 study centers and scales features using only the
fitting observations, separately within each cross-validation training fold.
Preprocessing changes the geometry and must be chosen explicitly.
`Pipeline` and `GridSearchCV` work for ordinary
training-only cross-validation. If an external pipeline transforms X, any
explicit `validation_data` must already be transformed consistently using only
the fitted training transformer; sklearn Pipeline does not transform that
fit argument automatically. Division by 255 itself requires no fitted transformer.

For a very small training set, 5% may retain only one class. The estimator
reports this selection failure without adding observations or changing the model.
A `SelectionError` preserves `linear_estimator_`, `selected_indices_`,
`selected_ids_`, and class-count diagnostics; the two-stage predictor remains
unfitted. The selected linear model remains available as a baseline.

# Fitting budgets and stopping

`max_epochs=200` remains the default for each stage. One SGD epoch processes
that stage's observations; it is not equivalent to one MM solver iteration.
The schedule, gradients, regularization, and sampling defaults are unchanged.

`stopping="auto"` preserves prior behavior: fixed epochs without monitoring
data, or validation-loss patience with explicit `validation_data`.
`stopping="fixed"` disables patience, and `stopping="validation"` requires
monitoring data. When monitoring data is supplied, the best validation
checkpoint is selected even in fixed mode. Keep that monitoring set inside
training data; it must not be the final test set or the CV scoring fold.

`termination_criterion_` names the chosen stopping rule, and
`criterion_reached_` says whether that rule was met. Completing a fixed epoch
budget is a budget-completion criterion. Validation patience is a predictive
monitoring criterion. Neither is an optimality test: `optimality_checked_`
remains false, and `converged_`/`optimality_met_` are false because no certificate
was established.

`cap_reached_`, `fit_usable_`, and `cv_eligible_` distinguish these facts.
Finite trained fits completing an epoch cap can receive CV scores without an
automatic budget increase. Wall-time-truncated fits and untrained fallbacks
are excluded by the search helpers. `n_iter_` counts SGD updates;
`n_epochs_completed_` records processed passes. Two-stage models expose both
components through `stage_diagnostics_`.

# Reproducibility and diagnostics

Each stage uses a private NumPy generator. Integer seeds, `Generator`, and
legacy `RandomState` inputs are supported without mutating caller RNG state.
The two-stage estimator records `stage_seeds_` for linear SGD, kernel SGD, and
selection ties. Optional unique `observation_ids` are preserved with selection.
No kernel row or Gram cache is retained after fitting. Optional lazy training
column storage is bounded by `kernel_cache_bytes`; public prediction uses
bounded blocks.

Standalone solvers record initial, final, and selected states as independent
copies; separate optimizer-update and processed-observation counts; the
numerical learning-rate bound; validation
history; RNG state; timing; and stopping/model status. The two-stage object
exposes both fitted stage objects. There is no elapsed-time cap by default.
Set `max_seconds` or `linear_max_seconds` to impose one; a capped fit is not
declared converged.

`fit(..., checkpoint_callback=callback)` supplies independent state and metric
dictionaries at initialization and completed validation checkpoints. For the
two-stage estimator, the callback receives `(stage_name, state, metrics)`;
standalone callbacks receive `(state, metrics)`. Callbacks can persist states
without accumulating full history arrays in RAM. Numerical failure invalidates
the fit even when older diagnostic checkpoints exist.

Standard Python pickle round-trips preserve scores for trusted model files.
State snapshots alone are diagnostic data, not an exact-resume API. This
release supports reproducible restart from seeds, not resuming an interrupted
optimizer. Exact resume would additionally need full patience, best-checkpoint,
budget, and preprocessing state.

Use `KernelSGDDWD.rkhs_norm_squared()` and `.objective(X, y)` when needed.
The squared-norm diagnostic requires O(m^2 d) work with on-demand kernels and
is intentionally separate from timed fitting. `.objective` uses the mean loss
of the supplied observations plus the fitted function's RKHS penalty. To
reproduce the fitted kernel objective, supply its actual fitting subset.

See [method and API notes](docs/method.md),
[committee requirements](docs/committee-requirements.md), and
[verification status](VALIDATION.md).

# Local MNIST data

```python
from two_stage_sgd_dwd.datasets import load_mnist_pair

data = load_mnist_pair("path/to/idx/files", digits=(7, 8), partition="train")
X, y, ids = data.X, data.y, data.ids
```

The optional loader validates IDX headers, dimensions, counts, payload lengths,
and digit labels; returns float64 pixels divided by 255; and records original
partition/row IDs and source SHA256 hashes. It does not download data, make
train/validation splits, or execute experiments at import time.

# Basis and citation

The implementation follows Chapter 3 of the October 4, 2026 manuscript revision
cited in the [overview](#overview). When using the package, cite the dissertation
and the software version:

> Cheng, Chang (2026). *two-stage-sgd-dwd: Two-Stage DWD via Stochastic
> Gradient Descent*, version 0.8.0 [Computer software].

Machine-readable citations are provided in [CITATION.cff](CITATION.cff)
and [BibTeX format](docs/references.bib). Input filenames and checksums are
recorded in [the source provenance record](docs/provenance.json).

Releases are available on [GitHub](https://github.com/cheng338/two-stage-sgd-dwd/releases).
Licensing has not been decided.
