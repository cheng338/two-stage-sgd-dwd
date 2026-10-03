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

Python 3.11 or newer is required. From this source directory:

```console
python -m pip install .
python -m pip install ".[test]"
python -m pytest
python examples/synthetic_example.py
```

Dependencies are NumPy, SciPy, and scikit-learn. Validation environments are
documented in [VALIDATION.md](VALIDATION.md). The package does not require
Conda or a particular BLAS vendor.

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

`lambd` is the sole regularization parameter in version 0.5.0 and means
`mean_loss + lambd * ||h||^2`, as in `dwd`. Python reserves the word `lambda`,
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

The runner evaluates `lambd=1e-6, gamma="scale", retention_fraction=0.3` as a
separate control, then searches the numeric grid. It allows up to three
boundary expansions and two geometric-refinement rounds around the best
three distinct lambd/gamma pairs, with at most 2,000 grid candidates. Two
complete expansion rounds without improvement stop further expansion.
Every proposed round covers all five fixed retention fractions. Completed
candidate records, including ineligible selections, are not evaluated again.
An unexpected fitting error stops the run; an explicit resume retries the
uncommitted candidate from the beginning. Earlier
exploratory runs used an explicit 5% provisional control; their records retain
that setting. The 30% software default aligns with the ensemble package;
this alignment does not establish 30% as the best-performing retention.

The runner saves folds, settings, candidate records, and selection diagnostics
for inspection and supports resuming with `--resume`. It reports boundary
winners, budget limits, and plateaus. A CV winner or plateau at a fixed
epoch budget does not establish an optimal default or optimizer convergence;
larger-budget and independent-seed checks remain part of assessing a candidate.
The [recorded MNIST exploration](docs/mnist-exploration.md) for digits 7 and 8
includes numerical verification, these training-only checks, a larger training
sample, and a focused follow-up grid.
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

The defaults are `batch_size=1, sampling="with_replacement"`, preserving the
dissertation's single-point sampling policy. At fixed hyperparameters and
seeds, the single-point update retains the earlier solver's arithmetic.
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

The dissertation currently states independent single-point sampling. See the
[alignment note](docs/dissertation-batch-alignment.md) for the required sampling,
update, and theory distinctions when describing optional batches or shuffling.
No sampling-policy revision is needed to describe the unchanged default.
The package remains **binary only**; multiclass DWD is not implemented here.

# Data and labels

Supply finite dense numeric features and exactly two class labels. Labels may
be strings or numbers; `classes_[0]` maps internally to -1 and `classes_[1]`
to +1. A zero decision score predicts `classes_[1]`. This zero convention follows
the dissertation and may differ from other DWD packages.

The estimators do **not** scale data automatically. The prescribed MNIST study
uses float64 pixels divided by 255. Other transformations change the geometry
and must be chosen explicitly. `Pipeline` and `GridSearchCV` work for ordinary
training-only cross-validation. If an external pipeline transforms X, any
explicit `validation_data` must already be transformed consistently using only
the fitted training transformer; sklearn Pipeline does not transform that
fit argument automatically. The prescribed MNIST division requires no fitted
transformer.

For a very small training set, 5% may retain only one class. The estimator
reports this selection failure without adding observations or changing the model.
A `SelectionError` preserves `linear_estimator_`, `selected_indices_`,
`selected_ids_`, and class-count diagnostics; the two-stage predictor remains
unfitted. The selected linear model remains available as a baseline.

# Reproducibility and diagnostics

Each stage uses a private NumPy generator. Integer seeds, `Generator`, and
legacy `RandomState` inputs are supported without mutating caller RNG state.
The two-stage estimator records `stage_seeds_` for linear SGD, kernel SGD, and
selection ties. Optional unique `observation_ids` are preserved with selection.
No kernel row or Gram cache is retained. Public prediction uses bounded blocks.

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

The implementation follows Chapter 3 of the October 1, 2026 manuscript revision
cited in the [overview](#overview). When using the package, cite the dissertation
and the software version:

> Cheng, Chang (2026). *two-stage-sgd-dwd: Two-Stage DWD via Stochastic
> Gradient Descent*, version 0.5.0 [Computer software].

Machine-readable citations are provided in [CITATION.cff](CITATION.cff)
and [BibTeX format](docs/references.bib). Input filenames and checksums are
recorded in [the source provenance record](docs/provenance.json).

Publication and licensing have not been decided.
