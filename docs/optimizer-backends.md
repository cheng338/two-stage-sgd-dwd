# Optimizer backends and lazy training-column reuse

These execution options retain the DWD objectives, update definitions, seeds,
and requested epoch budgets. They are independent of statistical parameter
selection. The defaults are `optimizer_backend="auto"` and, where
applicable, `kernel_cache_bytes=0`.

The MNIST tuning example retains its explicit Python backend default for its
recorded execution protocol. Pass `--optimizer-backend auto` to select the
guarded automatic backend there; its receipt binds that choice and the loaded
extension for resume validation.

## Backend selection

`LinearSGDDWD`, `KernelSGDDWD`, and `TwoStageDWDClassifier` accept:

| `optimizer_backend` | Behavior |
| --- | --- |
| `"python"` | Use the original Python orchestration and scalar update functions. |
| `"auto"` | Default: use the optional compiled scalar optimizer when supported; otherwise use Python and record the reason. |
| `"native"` | Require the compiled optimizer. Unsupported controls raise `ValueError`; an unavailable extension raises `ImportError`. |

Native execution supports `batch_size=1`, either sampling policy, and
C-contiguous, aligned feature arrays. It requires no supplied validation data,
checkpoint callback, or deadline. This restriction includes monitoring data
with `stopping="fixed"`, as well as `max_seconds` and the two-stage linear
`linear_max_seconds` limit. It does not copy a strided feature layout merely to
make it eligible: changing that layout could change NumPy's dot-product path.

For kernel fits, native execution supports `linear`, `rbf`, `poly`,
`polynomial`, `laplacian`, `cosine`, and `chi2`. Their ordinary domain and
parameter validation still applies. Callables, `sigmoid`, and `additive_chi2`
remain available through Python; `"auto"` records the fallback.

The native loop retains the scalar update operation order and uses the guarded
dot-product path described below.
It prepares at most one epoch of observation indices and learning rates at a
time. On stable NumPy releases 2.5.2 and later, an exact built-in `Generator`
using `PCG64`, `PCG64DXSM`, `Philox`, `SFC64`, or `MT19937` prepares replacement
indices with one array draw. Scalar and array draws use the same bounded-integer
routine in the tested versions 2.5.2, 2.5.3 and 2.5.4; regression checks compare
the indices, complete RNG state and continuation draws. Older releases, prerelease or
custom version strings, generator subclasses, and custom bit generators retain
scalar draws. Shuffle sampling retains its original epoch permutation.
Preparing an array of indices does not batch the gradients: every update
still uses one observation and its original learning rate.

NumPy does not promise scalar/array stream equivalence across all builds and
versions. Later stable releases are eligible under a forward compatibility
policy, without an upper version limit. They have not been tested in advance;
this policy assumes scalar/array stream and state equivalence continues for
the admitted built-in types. The exact stream, continuation and failed-update
rollback regression checks run on eligible installed versions. See
[NumPy's compatibility policy](https://numpy.org/doc/stable/reference/random/compatibility.html).

For those same exact built-in generators on stable NumPy releases 2.5.2 and
later, native optimization uses the independent state returned by the getter
as its epoch rollback snapshot, avoiding a redundant deep copy. This applies to both native sampling
policies and preserves the getter position and failed-update RNG restoration.
Ownership and rollback checks have passed on actual NumPy 2.5.2, 2.5.3 and
2.5.4 implementations. Later stable releases are eligible under a forward
compatibility policy that assumes the getter continues to return independent
state; they have not been tested in advance. Older releases, prerelease or
custom version strings, generator subclasses and custom bit generators retain
the deep copy. Initial, selected, final and returned model snapshots keep their
existing independent copies. Snapshot ownership and index-array stream
equivalence are separate properties tested under the shared version policy.

Eligible epoch learning rates are prepared by a compiled helper that retains
Python's scalar power operation and the existing schedule. Exact integer-range
checks protect the division arithmetic. Older extensions, unusual scalar types
and counters outside that range retain the original Python expression. The
helper polls pending signals every 1,024 entries; it does not combine gradients
or change the number of updates.

It does not prepare a kernel Gram matrix. A failed numerical update retains
the diagnostic state from the preceding committed update and the RNG position
after the attempted draw. The fit remains invalid. Structural errors propagate;
neither error category triggers a retry with a different backend.

`optimizer_backend_` and `optimizer_backend_reason_` record the actual path.
The two-stage model passes the requested setting to both components and stores
their diagnostics in `stage_diagnostics_`. Its top-level backend is `"mixed"`
when the components use different paths, and its top-level reason is a mapping
with `linear` and `kernel` entries.

## Direct BLAS calls in native fitting

For ordinary contiguous, aligned float64 vectors, the compiled optimizer can
call BLAS directly through SciPy's `cython_blas` interface. This removes the
Python dispatch and scalar conversion around each margin or projection-norm
dot product. It uses the existing SciPy dependency and requires no particular
BLAS vendor. The update formulas, learning rates, sampling, and kernel-column
calculations retain their existing paths.

The adapter requires a recognized LP64 C interface, the canonical NumPy dot
dispatcher, exact NumPy arrays, and a vector length representable by that
interface. Missing or unsupported interfaces, replaced dispatchers, and array
subclasses retain the captured NumPy dot call. A direct result with a numerical
exception flag or a nonfinite value is recomputed through NumPy, preserving
its warning, callback, logging, and exception policies and the committed update
prefix. This numerical fallback does not retry a fit or change its backend.

Fixed-budget MNIST comparisons found approximately 37% less complete fitting
time under both AOCL and MKL, with identical tested parameters and predictions.
That is an observed result, not a bitwise-equivalence guarantee across numerical
libraries. Inputs near `1e-160` repeatedly triggered fallback and were 16–19%
slower in a small linear control. See [validation results](../VALIDATION.md).

Direct C calls retain the GIL and bypass any internal NumPy GIL release. A
native epoch can therefore hold it continuously, delaying other Python threads.
Concurrent-training throughput has not been benchmarked. The native loop
returns to Python between epochs. This optimization is independent of
`prediction_backend` and applies only when the compiled optimizer is selected.

## Interrupts during native fitting

On the main Python thread, the native loop checks pending Python signals after
every 1,024 completed scalar updates, after releasing the current row. This
allows a pending interrupt to raise during a long epoch. The native loop
propagates exceptions from this check instead of returning a failed-update
prefix. Public estimators retain their existing exception handling. A normal
`KeyboardInterrupt` leaves the fit unfitted and does not produce a resumable
optimizer checkpoint.

These checks do not release the GIL or make Python signal handlers execute in
worker threads. They provide no wall-clock latency guarantee: one expensive
update, including a kernel evaluation or BLAS call, can still delay handling.
The update formulas, sampling, learning rates and ordinary numerical-failure
recovery are unchanged.

## Lazy kernel-column cache

`KernelSGDDWD` and `TwoStageDWDClassifier` accept `kernel_cache_bytes`, a
nonnegative integer. Zero disables storage. A positive budget enables an LRU
cache for `batch_size=1` and the same analytic built-in kernel families listed
above. An unsupported positive-cache configuration raises `ValueError`.
The linear estimator has no kernel-cache parameter.

For a kernel stage with `m` fitting observations, each float64 column has
`8*m` bytes. Capacity is `min(m, kernel_cache_bytes // (8*m))` columns. A
budget smaller than one column stores none. The first request for a training
observation computes the original `K(X, X[i:i+1])` column; later requests reuse
its exact stored values while it remains present. Storage fills on demand and
evicts the least recently used column when full. A sufficiently large cache
can eventually retain all requested columns, but no Gram matrix is computed
up front. No kernel symmetry shortcut or alternative distance formula is used.

The budget limits retained column array payload only. Dictionary entries,
allocator overhead, the current kernel computation, features, model arrays,
and other working memory are outside it. Monitor process peak memory when
comparing resource use. Cache arrays are released when fitting exits, including
error paths, and do not appear in fitted-model serialization. Repeated fits
start with an empty cache. Two-stage storage applies only to its retained
kernel subset. Validation and prediction retain their existing uncached block
computations. First-stage model reuse in the MNIST tuning runner is a separate
mechanism.

Diagnostics include `kernel_cache_hits_`, `kernel_cache_misses_`,
`kernel_cache_evictions_`, `kernel_cache_peak_bytes_`, and
`kernel_cache_capacity_columns_`. Disabled-cache counters are zero; an enabled
cache with insufficient capacity records misses and retains zero bytes.
`kernel_evaluation_count_` includes actual completed optimization/validation
kernel-block evaluations, so hits add zero. Additional validation cancellation
probes and corrected scores are recorded separately in
`validation_recovery_work_`, with requested and completed rows and entries.
Preparation and total evaluation counts retain their separate meanings.

## Availability and timing

For linear fitting, a compatible native extension retrieves each selected
row directly from the existing tuple of row views. This avoids creating a
Python index and calling a bound method for each update. The same row object
passes through the existing buffer checks, guarded dot-product call and scalar
update. Rows are still checked when selected, and failure recovery is unchanged.
Older extensions retain callable row access through capability detection;
kernel loaders and custom callable providers keep their existing path. Private
wrappers around `run_steps` that inspect the row provider must honor or disable
the `_supports_tuple_rows` capability rather than assume it is always callable.

The native linear loop checks every proposed coefficient while constructing
the proposal. For ordinary NumPy arrays that need no projection, it avoids
scanning that unchanged proposal a second time. Ordinary kernel proposals also
check every coefficient in their fused update loop. Projected linear updates
and unusual array types retain the separate final scan. This changes
neither the update arithmetic nor the sampling and learning-rate choices.

The compiled extension is optional; portable Python-only distributions retain
all estimator functionality. Native wheels are platform and architecture
specific. Source builds require a compatible C compiler for the extension;
setting the build environment variable `SGD_BUILD_ACCEL=0` requests a build
without it. Backend availability and numerical equivalence should be checked
for the actual installed artifact and environment. No compiler is downloaded
when importing the package or using its estimators.

The cache can be used with either backend. Fixed-data, fixed-seed correctness
tests check exact outputs in the tested environment; compiler, NumPy/BLAS,
thread, and hardware changes require their own verification. Time-limited
Python fits can stop at different updates when caching changes execution time.
Such stopped fits are not an equality comparison at a common update budget.

For ordinary native positive-width RBF fits, fitting features and bandwidth are
validated once before the update loop. Each requested column still uses the
same `cdist(X, X[i:i+1], metric="sqeuclidean")`, multiplication, in-place
exponential, and finite-result checks. Columns remain fresh and lazy; cache
lookups, evictions, and evaluation counters keep their existing behavior.
This path requires the standard kernel estimator, ordinary contiguous aligned
float64 arrays, an exact built-in NumPy generator and bit generator, the plain
resolved RBF specification, and unchanged kernel helper functions. Subclasses,
custom generators, other kernel settings, and replaced kernel helpers retain
per-request validation. Fitting inputs must remain unchanged during the fit;
concurrent external mutation is not supported. Python fitting, validation,
and prediction keep their existing paths.

Measure complete cold fits before making speed claims. Include scaling and
screening where used, kernel preparation, lazy cache filling, native setup,
fitted-array publication, and prediction according to the declared comparison.

For ordinary NumPy coefficient buffers, native kernel updates check finite
values while writing all shrunk coefficients and check the sampled coefficient
again after its loss update. This removes a separate scan without changing
the multiplication or subtraction order, the failure message, or the committed
update prefix. Custom coefficient buffers retain the separate final scan.
`fit_seconds_` covers the estimator's fit call; preprocessing performed by an
external Pipeline requires an outer timer. Scalar-update microbenchmarks do
not establish overall runtime or an advantage over another DWD solver. Report
the actual backend, cache budget and peak payload, process peak memory, fitting
controls, seeds, library versions, native-thread settings, and hardware.

## RBF prediction

New `KernelSGDDWD` and `TwoStageDWDClassifier` instances use
`prediction_backend="blas"` by default. Set `prediction_backend="direct"`
to select direct squared distances explicitly. This is separate from
`optimizer_backend`: it changes public prediction only, including scores used
by `predict`, `score`, or explicit objective evaluation. The optimizer,
screening, training kernel values, and validation-checkpoint calculations keep
their original arithmetic. Changing this option after fitting does not refit
or modify model coefficients. The two-stage model's top-level setting controls
its kernel prediction without modifying the nested estimator.

Direct prediction retains squared Euclidean differences computed by SciPy
`cdist`. The BLAS backend evaluates the same positive-width RBF formula through
matrix products in bounded prediction blocks. It falls back to direct
calculation for numerically hazardous distances or uncertain near-zero scores.
These are conservative engineering guards, not a mathematical certificate of
identical scores or labels for all finite inputs. Floating-point reduction
order differs, so bitwise equality is not promised. Non-RBF kernels and
zero-width RBF use their existing direct routes even when `"blas"` is selected.

Finite inputs can produce intermediate squared distances or scaled exponents
outside the floating-point range. Shared direct-distance consumers recover
exceptional representable RBF exponents and the small contrasts used by the
existing `expm1` paths. Ordinary distance scaling keeps its existing arithmetic.
The recovery requires the original numerical helpers, nearest binary64
rounding and preserved subnormals; it does not change the caller's arithmetic
mode. It preserves rejection of genuinely invalid distances and retains the
historical behavior of replaced helpers or unsupported arithmetic modes.
Recovery targets a rounded binary64 exponent, not arbitrary-precision kernel
values or all contrasts below the representable range. Exceptional recovery
can change scores or downstream fitted results and has a computational cost.

The loss term in `KernelSGDDWD.objective(X, y)` uses the selected prediction
backend, so the reported objective can reflect its score rounding. The RKHS
norm calculation retains its original path; fitting and validation checkpoints
do not use the BLAS prediction helper.

The option requires NumPy/SciPy's ordinary numerical libraries, not the optional
compiled SGD extension. It creates no persistent prediction cache; temporary
arrays are discarded when the call returns. `prediction_batch_size` still
bounds the query block size. Serialized models from before this option existed
continue to use direct prediction. Models saved with an explicit backend retain
that choice, including `"direct"`; loading an existing model does not adopt the
new constructor default.

Prediction comparisons on 30 saved MNIST models found substantial BLAS gains
under both AOCL and Intel MKL, with no changed labels in either environment.
Decision scores can differ in their last digits, including between numerical
libraries when both use direct prediction. See [validation results](../VALIDATION.md)
for the measured workloads and safeguards. Compare complete prediction calls
on the intended data and numerical environment before making performance claims.
