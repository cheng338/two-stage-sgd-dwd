# Method and API contract

## Mathematical definition

For margin u = y f(x), the normalized DWD loss is 1-u for u <= 1/2 and
1/(4u) otherwise. Its derivative is -1 and -1/(4u^2), respectively. The loss
is continuously differentiable, not twice differentiable at the threshold.
There is no free historical C parameter in this implementation.

The linear objective is mean L(y_i (x_i^T w + b)) subject to ||w||_2 <= 1.
In the dissertation's single-observation version, starting from zero, draw
one observation uniformly with replacement at each update, compute
q = L'(y_i (x_i^T w + b)) y_i using the old state, and update
w = projection(w - eta q x_i), b = b - eta q. Only the slope is projected.

The selected rows are the floor(retention_fraction * N) globally smallest
absolute linear scores. Independent random keys break score ties, then unique
original IDs break any key ties. There are no label or predicted-side quotas.
Selection is a heuristic; these are not DWD support vectors. Duplicate indices
are impossible. The count is rounded down without imposing a minimum count or
adding rows to supply both classes. An empty selection raises `SelectionError`
with status `empty_selection`; a nonempty selection with only one label raises
the same exception with status `single_class_selection`.
The product is evaluated in binary64 arithmetic before applying the floor.
For example, `0.29 * 100` is slightly below 29, so its retained count is 28.
There is no epsilon correction or exact-decimal interpretation of the ratio.
The default `retention_fraction=0.3` matches the ensemble package's 30% starting
value. The original Chapter 3 guide's 5% remains an explicit API setting,
outside the recommended grid; changing the default does not establish its
empirical superiority.

After selection, `training_class_counts_` and `selected_class_counts_` follow
`classes_` order. `selection_composition_` records the original labels, training
and selected sizes, realized selection ratio, and per-class counts and fractions.
For each class, `selected_fraction` divides its selected count by the total
selected count; `class_retention_fraction` divides by that class's training
count. These are descriptive facts, not class weights or selection quotas.
Empty selections retain zero counts and zero retention, with `selected_fraction`
set to `None` because the selected set has no class mixture. Empty and one-class
selection failures preserve this evidence. MNIST CV records use the same
`selection_composition` structure with counts from the fold's training rows.

For actual kernel fitting size m, the objective is
mean L(y_i (h(x_i) + b)) + lambd * ||h||_H^2.
`lambd` is the sole kernel regularization parameter, with default 1e-6, and
uses the same norm-coefficient convention as `dwd`. Python reserves `lambda`.
The source dissertation writes its coefficient as lambda_dissertation/2;
the identical objective uses lambda_new = lambda_dissertation/2, represented
by `lambd` here. See the [alignment note](dissertation-batch-alignment.md).
The constrained linear stage has no regularization parameter.
For a PSD kernel K and h(x) = sum_j a_j K(x_j,x), the update is below.
RBF, K(x,z) = exp(-gamma ||x-z||^2), remains the default.

```text
i ~ Uniform{0,...,m-1}, independently at each update
u = y[i] * (sum_j a[j] * K(x[j], x[i]) + b)
q = L_prime(u) * y[i]
a = (1 - 2 * eta * lambd) * a
a[i] -= eta * q
b -= eta * q
```

The intercept is unrestricted and unregularized in both stages. The coefficient
loss increment is not multiplied by K. The ordinary Euclidean coefficient
gradient is K times the RKHS gradient's coefficient representation; finite
differences confirm that identity but do not change the RKHS update.

The selected objective is an average over retained rows, equivalently a binary
weighted loss divided by the retained count. Full-data weighting divided by N
is a different objective at unchanged regularization. For equivalence, using
lambd_full with 1/N corresponds to lambd_subset = (N/m) lambd_full with 1/m.

## Optional batches and default sampling

The package retains `sampling="with_replacement", batch_size=1` as defaults,
matching the dissertation's independent-sampling formulation. Optional
`sampling="shuffle"` draws one random permutation each epoch and partitions
it into batches, including a smaller last batch if needed.
`batch_size=1` makes single-observation updates; positive integers make
mini-batch updates; `batch_size="full"` uses the entire stage fitting set.
An integer size is capped to the actual stage size m. Full mode uses no
permutation and no sampling RNG draws.

For actual batch size r and old-state q_l = L'(y[i_l] f(x[i_l])) y[i_l],
the linear batch update is:

```text
w = projection(w - eta * sum_l(q_l * x[i_l]) / r)
b = b - eta * sum_l(q_l) / r
```

For kernel DWD it is:

```text
a[j] = (1 - 2 * eta * lambd) * a[j]
       - eta * sum_{l: i_l == j}(q_l) / r
b = b - eta * sum_l(q_l) / r
```

All q values use the same old parameters, projection occurs once, and the
kernel regularization gradient `2 * lambd * h` occurs once. Repeated indices
accumulate loss increments. No extra division by m or batch-size multiplication
of `lambd` is introduced.
Full mode is gradient descent on these same objectives; it does not make
sequential SGD updates while looping through scoring blocks.

Under the default `sampling="with_replacement"`, each epoch consists of m
independent draws, allowing repeated or omitted rows. Under this policy the
batch-averaged gradient is conditionally unbiased for the empirical gradient.
Random reshuffling is common in practice but does not retain that conditional
unbiasedness claim after earlier updates in an epoch. The objectives remain
the same; theoretical statements requiring independent sampling need their
assumptions preserved or a separate reshuffling analysis. See
[dissertation alignment](dissertation-batch-alignment.md).

The averaged-gradient rule and optional reshuffled batching convention are described
in [Dive into Deep Learning, Section 12.5](https://d2l.ai/chapter_optimization/minibatch-sgd.html).
Sections 3.1 and 3.2 of the October 4, 2026 dissertation already permit averaged
batch updates with one projection or regularization update. Its Chapter 5
six-classifier study uses batch size one and independent replacement sampling,
as does the original implementation guide's primary experiment. The package
default is unchanged. That study does not establish results for optional
minibatch, full-gradient or reshuffling modes.

## Schedule and checkpoint selection

eta = (1/B) (1 + s/m)^(-3/4), where s is the number of observations in
previously committed batches. Each epoch processes m observations, and the
final batch is truncated at the epoch boundary. With sampling by replacement,
duplicates count as separate processed observations. For batch size one,
s equals the update index t and this is the implementation guide's exact schedule.
For other batch sizes this is a documented decay by processed passes, without
automatic learning-rate scaling. It does not equate the optimization progress
of different batch sizes or claim a batch-size-independent convergence rate.
Linear B = 4 max_i (||x_i||^2 + 1). For a general PSD kernel,
kernel B = 2 * lambd + 4 (max_i K(x_i,x_i) + 1), which reduces exactly
to 2 * lambd + 8 for RBF. The factor 4 bounds the normalized loss's
derivative Lipschitz constant; the +1 accounts for the free intercept.
This finite training bound does not establish a global population bound.
Both bounds are computed from the actual stage inputs. Float64 is used.

`stopping="auto"` preserves the previous rule: fixed epochs without validation
data, validation patience with it. Explicit `stopping="validation"` requires
validation data. Explicit `stopping="fixed"` disables patience but retains
best-checkpoint selection if validation data is supplied.

With validation data, checks default to every five epochs. The first finite
post-update check initializes the best checkpoint and patience reference.
The best checkpoint follows the smallest actual validation loss, with the
earliest exact tie. The separate patience reference updates only for improvement
greater than tol * max(1, reference_loss). Three consecutive misses and at
least ten completed epochs stop fitting when validation patience is active;
with defaults the earliest stop is epoch 20. The initial zero state is
diagnostic, not a validation candidate.
Without validation data, the final finite state is selected at the epoch cap.

Eligible near-constant RBF validation scores use the same cancellation recovery
as prediction. This changes only their floating-point evaluation, not the loss
definition, update gradients or checkpoint rules. It can change a nearly tied
checkpoint, the stopping epoch and the resulting final state. More accurate
scores do not guarantee uniformly more accurate rounded mean losses. Extra
physical recovery work is reported in `validation_recovery_work_` separately
from the existing logical kernel counters.

The named `termination_criterion_` is `fixed_epochs` or `validation_patience`.
`criterion_reached_` records its completion, not optimality. `n_iter_` is the
number of committed optimizer updates, `returned_iteration_` is the selected
checkpoint update, and `n_epochs_completed_` is processed observations divided
by fitting-set size (possibly fractional after truncation). `cap_reached_`
separates epoch/time limits from `fit_usable_`. `cv_eligible_` permits trained
fixed-epoch fits, but excludes wall-time truncation and untrained fallbacks.
No objective or optimality test is run during fitting: `optimality_checked_`,
`optimality_met_`, and `converged_` remain false. The last two mean no
certificate was established, not that divergence was proved. Two-stage
`stage_diagnostics_` preserves these facts separately for both fits.

No rule claims that finite SGD equals an exact minimizer or that validation
loss improves monotonically. The default maximum is 200 epochs. Optional time
caps are checked before and after updates and bounded validation blocks, after
checkpoint callbacks, and at model finalization. A NumPy operation or user
callback already in progress cannot be interrupted; the cap is cooperative,
so overshoot is possible. A completed finite update remains usable after a cap.

## Fitted state and failure status

Fitted training hyperparameters are frozen for prediction and objective
diagnostics; changing them takes effect on the next fit. A fitted
`KernelSGDDWD` reads its current `prediction_batch_size` for scoring blocks.
A fitted `TwoStageDWDClassifier` uses its own current `prediction_batch_size`
on each prediction call through a read-only child override. The child's stored
parameter is unchanged; direct child predictions and RKHS-norm diagnostics use
the child's own validated block size. Changing this setting after fitting does
not modify the fitted coefficients. When used during fitting, however, the
scoring block size can change floating-point reduction order and the resulting
updates or checkpoints. Equal mathematical update equations do not guarantee
identical fitted results across block sizes.

Standalone models expose `initial_state_`, `final_state_`, `selected_state_`,
`selected_iteration_`, `n_updates_`, `n_samples_seen_`, `n_epochs_`, `history_`, `learning_rate_B_`,
`rng_state_`, `fit_seconds_`, `stop_reason_`, and `model_status_`.
Initial/final/selected arrays and callback arrays are independent copies.
`n_updates_` counts committed parameter updates; `n_samples_seen_` counts
observations in those updates; `n_epochs_` is the latter divided by the actual
stage size. `batch_size_`, `sampling_`, `n_updates_per_epoch_`, and
`learning_rate_clock_` identify the effective policy. Snapshot dictionaries
and validation history also store `n_samples_seen`; `selected_n_samples_seen_`
locates the selected checkpoint's observation count. A cap during kernel
batch scoring preserves prior parameters and these committed-work counts.
Elapsed update time and completed kernel evaluations still include that
interrupted work.

`stop_reason_` distinguishes `epoch_cap`, `validation_patience`, `time_cap`,
`time_cap_no_validation`, and `numerical_failure`. `model_status_` distinguishes
`trained`, `untrained_fallback`, and numerical failure. A standalone cap before
one finite update returns a zero diagnostic model marked `untrained_fallback`.
It must not be ranked as a trained model. A time-capped trained model without
validation may be evaluated later but is ineligible for the guide's calibration.

Two-stage failures additionally include `empty_selection`, `single_class_selection`,
`linear_stage_failure`, and `kernel_stage_unrun`. An untrained or numerically
failed linear stage cannot generate a misleading random subset. A failed
kernel stage does not erase a valid linear baseline. Prediction is blocked
after failed fitting or invalid refitting. Validation and training inputs
must be finite; clipping nonfinite values would alter the specified method.

The library does not inspect test labels while fitting. Experiment code must
handle statistical replication, held-out evaluation, watchdogs, process
peak-memory measurement, calibration, and dissertation tables.
Library timing fields are diagnostics, not a completed controlled benchmark.
`fit_seconds_` includes input validation and publishing fitted arrays, including
the kernel basis copy. `update_seconds_` and `validation_checkpoint_seconds_`
include interrupted work in those phases; the latter also includes completed
checkpoint callbacks. Initialization and other overhead explain why the phase
times need not sum to total fitting time. `kernel_evaluation_count_` counts
individual kernel values in completed kernel calls during fitting, including
completed validation blocks before a cap. Prediction and separately requested
norm/objective diagnostics are excluded from this fitting counter.
With optional lazy training-column reuse, cache hits perform no kernel
evaluation and do not increase this counter. Cache payload capacity, peak
retained bytes, hits, misses, and evictions have separate diagnostics.

Kernel preparation is recorded separately in `kernel_preparation_seconds_`,
`kernel_preparation_evaluation_count_`, and `kernel_validation_`; the combined
count is `kernel_total_evaluation_count_`. Analytic diagonal evaluations count
as one per row; transient Gram checks count all evaluated matrix entries.
An interrupted non-RBF preparation raises `TrainingError` with status
`kernel_preparation_timeout` because validation has not completed.
RBF's diagonal is known to be one and can safely return the usual zero-update
diagnostic if its cap expires before any optimizer step.

The default execution settings are `optimizer_backend="auto"` and
`kernel_cache_bytes=0` on kernel and two-stage estimators. The linear estimator
accepts the backend setting but has no kernel-cache parameter. Optional native
scalar execution preserves the update definitions and scalar sampling stream;
it is restricted to fixed-epoch, single-observation fits without validation,
callbacks, or time caps, using contiguous aligned feature arrays and analytic
built-in kernels. `"auto"` records a Python fallback for unsupported settings;
`"native"` rejects them. Each two-stage component records its actual backend.

Lazy column storage is fit-local, bounded by retained array payload, and
populated only when an update requests a column. It does not reuse a fitted
model across repetitions. Failed numerical fits remain failed under either
backend; no older iterate is silently used as a successful fallback. See
[execution controls and timing scope](optimizer-backends.md).

## Optional post-fit optimization diagnostics

`optimization_diagnostics(estimator, X, y, block_size=256)` is an explicit
post-fit helper for the three binary estimators. It does not participate in
SGD updates, checkpoint selection, fitting budgets or cross-validation. It
assesses the returned fitted coefficients, including a returned validation
checkpoint, and leaves estimator state and `optimality_checked_`,
`optimality_met_` and `converged_` unchanged.

Supply the original preprocessed fitting rows and labels in their original
order. A standalone kernel estimator requires `X` to equal its stored fitting
basis. For a two-stage estimator supply the full fitting set: the helper
evaluates the linear stage there and the kernel stage only on its recorded
selected indices. Each mean uses that stage's own sample count. Fitting-label
identity and the full original linear design are not stored, so the caller
must preserve them. A same-sized replacement data set cannot be interpreted
as a diagnostic of the original training objective. For multiclass work call
the helper separately for each binary fit with that fit's label encoding;
preprocessing pipelines and multiclass wrappers are not accepted directly.

The report contains the objective, free-intercept derivative, a stationarity
residual, a numerical dual lower-bound estimate and the difference between
primal and dual values. For linear fits the residual uses a unit-step
projected gradient of the slope plus the intercept derivative. For kernel
fits it uses the RKHS gradient norm plus the intercept derivative; a Euclidean
coefficient-gradient norm would depend on the kernel representation. The
linear unit-ball constraint and the kernel `lambd * ||h||^2` penalty apply
only to their respective slopes/functions. Both intercepts remain free.

For the normalized loss, the identity
`V(u) = sup_{0 <= alpha <= 1} (sqrt(alpha) - alpha*u)` yields dual objectives
`mean(sqrt(alpha)) - ||X.T @ q||` for linear fits and
`mean(sqrt(alpha)) - q.T @ K @ q / (4*lambd)` for kernel fits, where
`q = y_signed * alpha / n` and a free intercept requires `sum(q) = 0`.
The helper starts from the negative loss derivative and reduces the larger
class total to construct a balanced candidate. It corrects residual imbalance
using a conservative bound on an optimal intercept, and can fall back to the
zero dual candidate. That correction is a reporting calculation, not a new
constraint or penalty on the fitted intercept. No Gram inverse, extra ridge,
or change of kernel is used; singular PSD kernels are supported.

A small numerical gap supports near-optimality for the stated objective.
A large bound can be loose, and it is not the actual remaining objective
error or a test-accuracy deficit. Values for different subsets, penalties or
preprocessing describe different objectives. The helper does not certify the
global selection procedure or provide a universal stopping tolerance.
`formal_certificate` is always false: float64 reductions and kernel evaluation
do not give interval-arithmetic guarantees. The report preserves tiny negative
gaps within its stated roundoff tolerance and rejects materially negative
ones. Quadratic-form tolerances and the amplification by small `lambd` are
reported. An extreme input can make a diagnostic unavailable even when a
fitted model remains usable. Deterministic callable kernels must satisfy the
PSD assumption; the helper cannot control a callable's external side effects.

The helper raises `TypeError` for unsupported estimator types, `ValueError`
for invalid or incompatible data and
`FloatingPointError` for detected numerical invalidity; an unfitted estimator
raises scikit-learn's `NotFittedError`. A failure does not invalidate the fitted
model or install a replacement result. Reports are independent, JSON-compatible
dictionaries. Two-stage reports contain `linear_stage` and `kernel_stage`.

For eligible near-constant RBF blocks, the helper evaluates the small variation
with `expm1` and separates the constant component of quadratic forms. A finite
mean whose unscaled sum overflows is recomputed after scaling its terms. These
changes improve particular floating-point calculations; rounded dual vectors
need not satisfy exact feasibility or exact coefficient identities, and a
reported estimate remains uncertified. The report warns about amplification of
quadratic roundoff by a small regularization parameter.

Kernel reports also include `kernel_matrix_product_uncertainty`, with separate
indicators for coefficient, gradient and dual matrix-vector products. Scaled
estimates use the observed kernel-entry magnitudes and vector absolute sums
to assess inner-product cancellation and its effect on quadratic components.
They are conservative reporting estimates, not formal error bounds, corrected
objective values or acceptance criteria. The report identifies whether the
matrix is the literal kernel or the existing RBF `expm1` contrast representation.
Its qualification states omitted errors, including kernel construction,
separately accumulated constant components and formation of the gradient or
dual vectors. A false warning flag does not establish accuracy. Reporting
thresholds do not change fitting, stopping rules or convergence flags.

Linear work is proportional to the design size. Kernel work uses two blocked
passes over the actual stage data, evaluating `2*n*n` kernel values. The report
includes elapsed time, kernel-value count and largest kernel-block element
count. Working memory includes that block and additional vectors and library
workspace; the block count is not a process peak-RAM measurement. No full
Gram cache is retained. Keep diagnostic timing separate from fit/prediction
timing, and do not enable it automatically for every CV fit.

## Relationship to the previous packages

`dwd` and `ensemble-dwd` are not runtime dependencies. NumPy/SciPy provide
numerical operations; sklearn provides estimator validation, cloning, labels,
and the classifier interface. Tiny SciPy optimization oracles appear only in
tests. No SOCP solver participates in fitting.

The `kernel`/`gamma`/`kernel_kws` interface, `lambd` objective convention,
explicit whole-estimator cross-validation, and a logarithmic MNIST grid helper
follow the previous packages' interface conventions. The actual starting grid
uses the ensemble package's 10%, 15%, 20%, 25%, and 30% retention fractions.
The original 5% remains available explicitly but is excluded from current
recommended presets. Their solvers are not reused.
The sole regularization argument is `lambd`, with default 1e-6; its fitted
value is recorded in `lambd_`.
See [statistical rationale and kernel details](kernels-and-tuning.md).

The practical algorithm is distinct from regularized-intercept theoretical
variants in the dissertation. No consistency, equivalence, noninferiority, or
universal speed/accuracy guarantee is claimed by this package.
