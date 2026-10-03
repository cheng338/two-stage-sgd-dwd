# Optimizer backends and lazy training-column reuse

These execution options retain the DWD objectives, update definitions, seeds,
and requested epoch budgets. They are independent of statistical parameter
selection. The defaults remain `optimizer_backend="python"` and, where
applicable, `kernel_cache_bytes=0`.

## Backend selection

`LinearSGDDWD`, `KernelSGDDWD`, and `TwoStageDWDClassifier` accept:

| `optimizer_backend` | Behavior |
| --- | --- |
| `"python"` | Use the original Python orchestration and scalar update functions. This is the default. |
| `"auto"` | Use the optional compiled scalar optimizer when supported; otherwise use Python and record the reason. |
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

The native loop uses NumPy's dot product and the scalar update operation order.
It prepares at most one epoch of observation indices and learning rates at a
time, using the same scalar replacement draws or epoch permutation as Python.
It does not prepare a kernel Gram matrix. A failed numerical update retains
the diagnostic state from the preceding committed update and the RNG position
after the attempted draw. The fit remains invalid. Structural errors propagate;
neither error category triggers a retry with a different backend.

`optimizer_backend_` and `optimizer_backend_reason_` record the actual path.
The two-stage model passes the requested setting to both components and stores
their diagnostics in `stage_diagnostics_`. Its top-level backend is `"mixed"`
when the components use different paths, and its top-level reason is a mapping
with `linear` and `kernel` entries.

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
kernel evaluations, so hits add zero. Preparation and total evaluation counts
retain their separate meanings.

## Availability and timing

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

Measure complete cold fits before making speed claims. Include scaling and
screening where used, kernel preparation, lazy cache filling, native setup,
fitted-array publication, and prediction according to the declared comparison.
`fit_seconds_` covers the estimator's fit call; preprocessing performed by an
external Pipeline requires an outer timer. Scalar-update microbenchmarks do
not establish overall runtime or an advantage over another DWD solver. Report
the actual backend, cache budget and peak payload, process peak memory, fitting
controls, seeds, library versions, native-thread settings, and hardware.
