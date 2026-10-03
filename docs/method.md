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

The selected rows are the ceil(retention_fraction * N) globally smallest
absolute linear scores. Independent random keys break score ties, then unique
original IDs break any key ties. There are no label or predicted-side quotas.
Selection is a heuristic; these are not DWD support vectors. Duplicate indices
are impossible. Selection with only one label raises a recorded failure.
The default `retention_fraction=0.3` matches the ensemble package's 30% starting
value. The original Chapter 3 guide's 5% remains an explicit API setting,
outside the recommended grid; changing the default does not establish its
empirical superiority.

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
The implementation guide allows averaged mini-batches with regularization once,
while its primary experiment specifies batch size one. This default is
preserved. No experimental protocol has been frozen or executed with the
optional batch or reshuffling modes.

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
with defaults the earliest stop is
epoch 20. The initial zero state is diagnostic, not a validation candidate.
Without validation data, the final finite state is selected at the epoch cap.

The named `termination_criterion_` is `fixed_epochs` or `validation_patience`.
`criterion_reached_` records its completion, not optimality. `n_iter_` is the
number of committed optimizer updates, `returned_iteration_` is the selected
checkpoint update, and `n_epochs_completed_` is processed observations divided
by fitting-set size (possibly fractional after truncation). `cap_reached_`
separates epoch/time limits from `fit_usable_`. `cv_eligible_` permits trained
fixed-epoch fits, but excludes wall-time truncation and untrained fallbacks.
No objective or optimality test is run: `optimality_checked_`,
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

Two-stage failures additionally include `single_class_selection`,
`linear_stage_failure`, and `kernel_stage_unrun`. An untrained or numerically
failed linear stage cannot generate a misleading random subset. A failed
kernel stage does not erase a valid linear baseline. Prediction is blocked
after failed fitting or invalid refitting. Validation and training inputs
must be finite; clipping nonfinite values would alter the specified method.

The library does not inspect test labels while fitting. Statistical replication,
held-out experiment orchestration, watchdogs, process peak-memory measurement,
calibration, and dissertation tables belong to the later experiment layer.
Library timing fields are diagnostics, not a completed controlled benchmark.
`fit_seconds_` includes input validation and publishing fitted arrays, including
the kernel basis copy. `update_seconds_` and `validation_checkpoint_seconds_`
include interrupted work in those phases; the latter also includes completed
checkpoint callbacks. Initialization and other overhead explain why the phase
times need not sum to total fitting time. `kernel_evaluation_count_` counts
individual kernel values in completed kernel calls during fitting, including
completed validation blocks before a cap. Prediction and separately requested
norm/objective diagnostics are excluded from this fitting counter.

Kernel preparation is recorded separately in `kernel_preparation_seconds_`,
`kernel_preparation_evaluation_count_`, and `kernel_validation_`; the combined
count is `kernel_total_evaluation_count_`. Analytic diagonal evaluations count
as one per row; transient Gram checks count all evaluated matrix entries.
An interrupted non-RBF preparation raises `TrainingError` with status
`kernel_preparation_timeout` because validation has not completed.
RBF's diagonal is known to be one and can safely return the usual zero-update
diagnostic if its cap expires before any optimizer step.

## Relationship to the previous packages

`dwd` and `ensemble-dwd` use different solvers/frameworks. Neither is a runtime
dependency. NumPy/SciPy provide numerical operations; sklearn provides estimator
validation, cloning, labels, and the classifier interface. Tiny SciPy optimization
oracles appear only in tests. No SOCP solver participates in fitting.

The `kernel`/`gamma`/`kernel_kws` interface, `lambd` objective convention,
explicit whole-estimator cross-validation, and a logarithmic MNIST grid helper
follow the previous packages' interface conventions. The actual starting grid
uses the ensemble package's 10%, 15%, 20%, 25%, and 30% retention fractions.
The original 5% remains available explicitly but is excluded from current
recommended presets. Their solvers are not reused.
The sole regularization argument is `lambd`, with default 1e-6; its fitted
value is recorded in `lambd_`.
Fitted hyperparameters are frozen for prediction and objective diagnostics;
changing constructor parameters takes effect on the next fit. Prediction
block size remains adjustable because it only controls memory use.
See [statistical rationale and kernel details](kernels-and-tuning.md).

The practical algorithm is distinct from regularized-intercept theoretical
variants in the dissertation. No consistency, equivalence, noninferiority, or
universal speed/accuracy guarantee is claimed by this package.
