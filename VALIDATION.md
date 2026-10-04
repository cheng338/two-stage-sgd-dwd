# Development verification — native replacement-index preparation

On October 4, 2026, guarded native index preparation passed **547 tests and
7,413 subtests** with the released native extension and **516 tests and 4,639
subtests** in a separate portable source copy, with 31 native-dependent skips.
Each suite produced only the existing warning from an invalid-label fixture.
Checks compare complete RNG states, continuation draws, generated indices,
scalar fallbacks, shuffle sampling, coefficients, selected observations,
cache diagnostics, external RNG immutability, and failed-update recovery.

On NumPy 2.5.2, exact built-in Generator instances with PCG64, PCG64DXSM,
Philox, SFC64 or MT19937 prepare one epoch of replacement indices with one
array draw. Each coefficient update still processes one observation with its
original learning rate. Other NumPy versions and custom generators retain
scalar draws. No new dependency, tuning parameter, or compiled binary is
required. The Python optimizer remains the default.

A complete-fit benchmark used digit pairs 2–3, 4–9 and 3–8, with saved training
draws 00, 04 and 09 for each pair. Each draw was timed twice in alternating
implementation order: six paired timings per pair, not six independent
training draws. Timings included StandardScaler and both stages at 800 epochs,
30% retention and a 64 MiB lazy-cache payload limit, using py12_df_aocl,
NumPy 2.5.2 and one native thread. Means and sample SDs combine the three
saved draws and their two timing repetitions.

| Pair | Previous fit, s (mean ± SD) | Guarded array draws, s (mean ± SD) | Time saved |
| --- | ---: | ---: | ---: |
| 2–3 | 7.641 ± 0.292 | 4.928 ± 0.269 | 35.5% |
| 4–9 | 7.479 ± 0.342 | 4.748 ± 0.305 | 36.5% |
| 3–8 | 7.646 ± 0.249 | 4.955 ± 0.413 | 35.2% |

The candidate was faster in all 18 complete-fit pairs. All nine saved model
configurations reproduced their original study score hashes; coefficients,
RNG states, selections, predictions and cache/evaluation counters matched
exactly. Tests with cold, evicting, disabled and insufficient caches showed
no material regression; the largest control mean slowdown was 0.65%.
No test labels were read and no statistical settings were retuned.

These results describe the tested workloads and runtime. NumPy does not
generally promise scalar/array stream equivalence across versions and builds;
the fast path therefore uses a narrow version/type guard. Other versions
receive scalar fallback and should not expect this speedup. Historical study
outputs are unchanged. See [optimizer details](docs/optimizer-backends.md).

# Development verification — optional BLAS RBF prediction

On October 4, 2026, the optional prediction backend passed **534 tests and
5,599 subtests** with the released native extension and **509 tests and 2,959
subtests** in a separate portable source copy, with 25 native-dependent skips.
Each suite produced the existing warning from a deliberately invalid-label
fixture; neither had a failure. Checks cover direct defaults, old saved models,
postfit switching, unchanged training/checkpoint paths, large offsets, nearly
equal rows, extreme widths and coefficients, subnormal values, overflow,
near-zero decisions, array layouts, serialization, and bounded local buffers.

A paired prediction benchmark reproduced 30 fixed-30% MNIST models: ten saved
training draws for each of digit pairs 2–3, 4–9, and 3–8. Each direct model
matched its saved study score hash. Identical fitted state was transferred to
the optional BLAS path without retraining or tuning. All training and test
labels stayed unchanged at prediction block sizes 256 and 1024. The largest
absolute score difference was approximately **4e-14**, within the tolerance
declared before timing: absolute 1e-10 plus relative 1e-11. Input arrays and
serialized fitted state stayed unchanged.

Six paired, alternating-order timings per model included StandardScaler and
prediction, using `py12_df_aocl` with one native thread. At the default block
size of 256, means across ten model-level timing means were:

| Pair | Direct prediction, ms (mean ± SD) | BLAS prediction, ms (mean ± SD) | Speedup |
| --- | ---: | ---: | ---: |
| 2–3 | 244.795 ± 3.568 | 30.948 ± 0.720 | 7.91× |
| 4–9 | 233.663 ± 3.442 | 29.736 ± 0.862 | 7.86× |
| 3–8 | 236.359 ± 5.073 | 29.862 ± 1.361 | 7.92× |

At block size 1024, speedups were 7.58–7.84×. Peak traced temporary allocations
increased by about 0.03 MiB at block size 256 and 0.04–0.08 MiB at block size
1024; these allocation measurements are not total process memory. Untimed
test-query diagnostics recorded no direct-distance fallbacks on these models.
Separate adversarial tests exercise the fallback paths.

Direct prediction remains the default. The opt-in backend changes floating-point
reduction order; its safeguards and observed label agreement are not universal
equivalence certificates across inputs or BLAS builds. Training and its update
rules are unchanged. Explicit postfit objective evaluation can reflect the
chosen prediction backend's score rounding. The recorded study results remain
unchanged; this benchmark establishes prediction performance in this environment.

# Development verification — accelerated MNIST tuning example

On October 4, 2026, the optional tuning controls passed **462 tests and 5,599
subtests** with the released native extension and **439 tests and 2,959
subtests** in a separate portable source copy, with 23 native-dependent skips.
Each suite produced the existing warning from a deliberately invalid-label
fixture. Checks cover default preservation, public-estimator parity, fold-local
screening reuse, backend/cache diagnostics, fallback and error propagation,
native-binary snapshots, and incompatible resume/qualification rejection.
No solver source, search grid, stopping rule, or estimator default changed.

A training-only implementation benchmark used 600 balanced observations for
each of MNIST pairs 2–3, 4–9, and 3–8, pixels divided by 255, three CV folds,
30% retention, 20 fixed epochs, and two parameter settings. Five balanced-order
timing repetitions used Python 3.12 in `py12_df_aocl` with one native thread.
The original Python example took 2.306–2.313 seconds per complete two-candidate
evaluation; native execution with a 64 MiB lazy-cache payload bound took
0.186–0.187 seconds, a 12.3–12.4-fold speedup on this bounded workload.
Keeping screening reuse also reduced those accelerated evaluation times by
approximately 29% compared with refitting screening for both candidates.

Retained observations, coefficients, intercepts, RNG states, and validation
scores agreed exactly with the original path and independent public-estimator
fits in this environment. All timed evaluations preserved their CV metrics.
This exposes acceleration already present in version 0.7.0; it is not an
additional speedup over the completed main study, which already used the
compiled optimizer, or a full-study runtime forecast. Official test data were
not loaded. Native build identity and cache capacity remain explicit controls.

# Development verification — prediction buffer reuse

On October 4, 2026, the prediction-only update passed **417 tests and 5,599
subtests** with the native extension and **399 tests and 2,959 subtests** in
the portable configuration, with 18 native-only skips. Checks cover direct
distance arithmetic, large common offsets, noncontiguous and read-only inputs,
invalid fitted state, overflow, serialization, and bounded buffer reuse.
The training and validation-checkpoint evaluation paths are unchanged.

A paired comparison reproduced 30 fixed-30% MNIST models from the three digit
pairs and ten saved training draws per pair. The original decision scores
matched their saved study hashes. The revised prediction path then matched
training and test scores bitwise and preserved all class labels. Six paired,
alternating-order prediction timings per model used Python 3.12.14 in
`py12_df_aocl` with one native thread, including feature scaling.

Peak traced temporary allocations decreased by approximately 9% with the
default prediction block size of 256 rows and 22–23% with the study setting
of 1024 rows. These are Python/NumPy allocation peaks, not total process memory.
Mean prediction-time differences were below 1%; this check establishes no
substantial prediction speedup. Training settings and the recorded study
results were not changed.

# Release verification — 0.7.0

On October 4, 2026, separately installed release wheels passed the complete
correctness suite. The Windows native wheel passed **405 tests and 5,531
subtests**. The portable wheel passed **387 tests and 2,891 subtests**, with
18 native-only tests skipped because that wheel does not include the extension.
Both checks used Python 3.12.14 in `py12_df_aocl`, one native thread, and
verified installed import paths before and after the tests. Each suite
produced one expected warning from a deliberately invalid-label fixture.
No correctness-test failures occurred.

The native and portable wheels and source distribution are rebuilt from the
release source, including its current documentation. Promotion from the
integrated development branch changes version and publication metadata;
the estimator algorithms and defaults remain unchanged. The earlier checks
below retain their original version and artifact scope.

## Development verification — 0.7.0.dev20261003

The installed-wheel checks used artifacts built from feature commit
`85f90d9`. The separately installed native Windows wheel passed **405 tests and
5,531 subtests**. The separately installed portable wheel passed **387 tests and
2,891 subtests**, with 18 native-only tests skipped because that wheel does
not include the extension. Both were tested with Python 3.12.14 in
`py12_df_aocl`, one native thread, and verified installed import paths.
Each suite produced one expected warning from a deliberately invalid-label
fixture. No correctness-test failures occurred.

The integrated `0.7.0.dev20261003` development-branch source also passed
**405 tests and 5,531 subtests** with the verified native extension present.
This source result does not substitute for a separately installed native or
portable wheel check of the integrated branch.

The optional native optimizer and lazy kernel-column cache have focused
synthetic checks in `tests/test_native_optimizer.py` and
`tests/test_kernel_cache.py`. They compare coefficients, intercepts, selected
observations, update counts, and RNG state against the Python path, including
replacement sampling, shuffled epochs, and cache eviction. Additional checks
cover unsupported-mode fallback, native errors, failed-update RNG restoration,
fit-local cleanup, serialization, and payload limits. These checks concern
correctness in the tested numerical environment; they do not establish a
full-fit speed advantage or cross-platform bitwise agreement.

Native availability is optional. Full source and installed-distribution
validation for a particular build must record whether the extension was
present and which backend actually ran. The earlier source/wheel results
below are retained with their original version scope.

## Development baseline verification — 0.6.0.dev20261003

The October 3, 2026 development branch adds explicit stopping diagnostics and
adaptive-search qualification. The source suite and a separately installed
wheel each passed **369 tests and 364 subtests** in `py12_df_aocl` (Python
3.12.14). Successful acceptance, refusal of unresolved searches, interrupted
search recovery, and detection of changed selection evidence are covered by
deterministic regression tests.

[Stopping and search validation](docs/stopping-and-search-validation.md)
records the additional real-MNIST checks. Nine validation folds matched
version 0.5.0 exactly on scores, selected observations, update counts, and
other recorded numerical outputs. Four controlled adaptive searches exercised
budget limits and repeated refinement. All four correctly remained
`needs_review`; none established successful qualification on real MNIST.
The largest completed 480 candidates and ten local refinement rounds but
still had unresolved tied boundaries. This is a recorded limitation, not a
completed tuning result or an official-test accuracy estimate.

These small checks used one native thread per process. There was no six-core
partition or 32 GiB memory cap. The earlier resource settings below describe
historical verification only.

## Historical verification — 0.5.0

The remaining sections record the version 0.5.0 checks and their resource
settings.

Version 0.5.0 adopts the ensemble package's MNIST lambda and gamma ranges and
its 10% through 30% retention choices, and provides explicit adaptive exploration.
The defaults change to `lambd=1e-6`, `gamma="scale"`, and two-stage retention
of 30%, matching the ensemble package. The original 5% choice remains an
explicit API option and comparison in the exploratory records; current
recommended presets omit it after the recorded training-only checks.
Source and installed-wheel verification cover imports, serialization,
cross-validation, and the synthetic examples. Citation records and their
inclusion in the distributions are checked separately.

Validated on October 1, 2026, using Python 3.12.14 in `py12_df_aocl`,
NumPy 2.5.2, SciPy 1.18.0, scikit-learn 1.9.0, and AOCL BLAS 5.2.0.
The source and separately installed wheel receive correctness-suite checks.
Their exact test counts and environment receipts accompany the release artifacts.

## Evidence covered

- Normalized loss and derivative: branch behavior, continuity, finite
  differences, nonfinite rejection, and stable unit-ball projection.
- Correct stochastic directions: unbiased linear gradients and RKHS updates,
  including a nonidentity Gram matrix that distinguishes an RKHS gradient
  from the ordinary coefficient-space gradient.
- Independent small optimization oracles: constrained SLSQP for linear DWD
  and BFGS for a positive-definite finite kernel problem. Finite SGD is compared
  with explicit tolerances; equality with exact minimizers is not claimed.
- Optional batches: independent linear and RKHS finite differences, averaged
  old-state gradients, one projection/regularization application, accumulation
  of duplicate sampled indices, full-gradient updates, truncated final batches,
  shuffled-epoch coverage when requested, and independent manual trajectories.
  Tests distinguish observation counts from optimizer updates and reject
  partial kernel batches interrupted by a deadline.
- Global absolute-score selection: exact retained count, unique observations,
  immutable scores, label-independent ties, and one-class failure.
- Estimator contracts: cloning, labels, shapes, Pipeline/GridSearchCV,
  serialization, input and RNG isolation, and failed-refit cleanup.
- Checkpoints and caps: actual-best selection distinct from patience,
  independent snapshots, partial validation rejection, numerical failure,
  final-update/finalization cap detection, interrupted-phase timing, and
  accounting for completed kernel blocks before interruption.
- Strict local IDX handling: tiny generated binary fixtures test headers,
  payloads, dimensions, labels, scaling, original IDs, and source hashes.
- Kernel choices: independent linear/polynomial update and objective checks,
  analytic diagonal bounds, matrix callable contracts, gamma resolution,
  full-fold bandwidth resolution before screening, fit-state isolation,
  rejection of invalid/indefinite kernels, and serialization.
- Tuning: exact 125/405-candidate grids, independent returned containers,
  DWD regularization scaling, full-Pipeline cross-validation, unpenalized loss scoring,
  finite-score enforcement, untrained-candidate/final-refit rejection, and
  prevention of forwarded shared monitoring data.
- Adaptive exploration: complete boundary shells, tied endpoints, geometric
  midpoints, retention coverage, attempted-candidate deduplication, and finite
  representability checks. The CLI records splits and complete stages and
  verifies the configuration and inputs before resuming.
- Within-fold screening reuse: comparisons against the public two-stage
  estimator check coefficients, predictions, selected IDs, and random seeds.
  Every candidate still fits its kernel stage; scoring-fold labels are not
  used for either training stage or checkpoint selection.

The synthetic usage example also fits with the default 30% retention and
predicts held-out observations. Its accuracy is illustrative, not a benchmark
or a parameter-selection result. A second, small synthetic example runs
explicit threefold selection between linear and RBF kernel candidates, then
evaluates once on its held-out split. It demonstrates the API only; its
accuracy does not establish MNIST performance or screening superiority.

In version 0.4.0, an additional 18 paired fits compared the source with the separately installed
0.1.0 wheel: all three estimators, three seeds, and runs with/without validation.
The unchanged defaults (`batch_size=1, sampling="with_replacement"`) produce
bitwise-identical scores, coefficients, intercepts, selected subsets, and RNG
states under fixed iteration/validation budgets in the same environment.
This statement excludes wall-clock timing and time-capped stopping points.

Also in version 0.4.0, a further 54 paired fits compared the sole `lambd` parameter against version
0.3.0 at equivalent regularization strengths: two estimators, three kernel
families, three batch sizes, and three coefficients. Predictions, coefficients,
intercepts, learning-rate bounds, objective values, selection, and RNG states
are bitwise identical in these checks. The parameter conversion therefore
preserves the tested numerical behavior while matching the `dwd` convention.

## Resource policy

The final verification harness runs one numerical process at a time, with CPU
affinity selecting one logical processor from each of **six physical cores**.
Windows reports 16 physical cores and 32 logical processors. Numerical thread
pools are capped at six, and a Windows Job Object caps committed memory at
**32 GiB**. Exact peak memory is recorded in each receipt. The source suite
uses approximately 150 MiB of working-set memory; the illustrative examples
use one numerical thread.

These are ceilings, not reserved capacity. Other work on the computer can
still share caches, memory bandwidth, and power limits. Verification timings
under concurrent activity are not scientific runtime comparisons. The local
harness does not change system settings or impose resource limits when users
import this portable library in their own programs.

An earlier short synthetic environment comparison found `py12_df` and
`py12_df_aocl` effectively tied; AOCL was marginally faster and was selected
for development. This does not establish an environment-level speed claim.

## Reproducing the checks

Installation uses the declared runtime dependencies. Verification uses isolated
test and build tools without altering the shared Conda environments. The wheel
is installed into a separate verification directory.

To repeat the portable correctness suite after installation:

```console
python -m pytest -q tests
python examples/synthetic_example.py
python examples/kernel_search_example.py
```

## Boundaries

The package has been verified on this Windows/Python environment. Other
supported Python versions and operating systems have not been exercised here.
[Training-only MNIST exploration](docs/mnist-exploration.md) is recorded
separately from these correctness checks. It does not establish optimizer convergence, consistency,
noninferiority, screening superiority, or official-test performance. Selected
parameters are conditional on the stated data, seeds, and finite SGD budgets.

The committee requirements table distinguishes package support from unfinished
experimental and dissertation work. Input hashes and the source-hash difference
between the supplied chapter and the current chapter are recorded in
`docs/provenance.json`.
