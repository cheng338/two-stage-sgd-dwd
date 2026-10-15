# Current Python support

The current development version requires Python 3.11 or newer. Native builds
use the Python 3.11 stable ABI (`cp311-abi3`) for standard CPython; portable
`py3-none-any` wheels have the same Python minimum. Python 3.10 and free-threaded
CPython builds are not supported. Removing the Python 3.10 packaging path does
not change the SGD implementation or numerical code.

Python 3.12 has direct runtime validation. Standard CPython 3.11 and 3.13
compatibility follows the build configuration and stable-ABI contract, without
direct local runtime checks on those versions to date. The Windows workflow
now targets Python 3.11 with NumPy 1 and Python 3.12 with NumPy 2 for both native
and portable installed wheels; a configured workflow is not a hosted-run result.

The Python 3.10 results below are historical evidence collected before support
was removed. They do not describe current installation requirements.

After changing the minimum on October 15, nine build-configuration checks
passed, including simulated 3.11, 3.12 and 3.13 version selection. Fresh native
and portable wheels built on Windows/Python 3.12 with the expected tags and
`Requires-Python: >=3.11`. Each extracted wheel passed tiny linear, kernel and
two-stage fit/predict checks with its expected backend. Pip rejected both
wheels when targeting Python 3.10. Hashes confirmed that all package source
files and the existing optimizer binary were unchanged by this packaging edit.
These small checks used one numerical thread; no numerical study was rerun.

# Current retention-count convention

The current development version uses `floor(retention_fraction * N)` for
two-stage selection. Global ranking and tie breaking are unchanged; the
implementation does not enlarge an empty or one-class selection. These cases
raise `SelectionError` with status `empty_selection` or
`single_class_selection`, respectively. This approved change can alter new
fits whenever the retained-count product is noninteger. It does not promise
equal predictions, improved accuracy or faster execution. Previous saved
models and experiment records are unchanged.

For this change, the complete AOCL/Python 3.12 source suite passed **1,351
tests and 9,641 subtests**, retaining the existing invalid-label cast warning.
Coverage includes integer and adjacent-float count boundaries, empty and
one-class selection, multiclass failure propagation, cached/direct CV parity,
and recovery of a completed empty-selection fold. An independent comparison
with the immediately preceding source checked 528 selections: each new
selection is the floor-length prefix of the previous ranking, with unchanged
tie-key RNG state and input arrays. Thirty-six small fits with integral
retained counts had bitwise-identical scores, labels and selected rows across
linear, RBF and polynomial kernels, single/batch/full updates, and Python or
automatic optimizer selection. The native optimizer binary is unchanged.
These correctness checks used one numerical thread; no numerical study was
rerun. They do not assert result equivalence when the retained count changes.

The validation results below predate this count-convention change. Their
claims that selection and fitting were unchanged apply to the changes tested
at that time, not to the subsequent downward-rounding change.

# October 15, 2026 prediction and search validation

These preceding updates correct exact-count comparisons and geometric proposals
in the adaptive example, add external prediction diagnostics, and stabilize a
narrow near-constant-RBF cancellation case in the BLAS predictor. The SGD
objective, update equations, sampling, selection, fitting code and native
optimizer binary are unchanged.

The final AOCL/Python 3.12 source suite passed **1,334 tests and 9,641
subtests**, retaining one existing invalid-label cast warning. Before the
Python minimum changed, the rebuilt portable wheel passed **193 targeted tests**
on Python 3.10, with two native-only skips. OpenBLAS/Python 3.12 passed
**158 prediction tests**.
The source suite covers exact ranking and stability, resume/export integrity,
geometric arithmetic limits, profiler isolation, numerical correction and
existing estimator behavior. Wheel-source hashes were verified. An initial
harness omitted the local pytest dependency path from subprocesses; correcting
that path resolved the harness failures before the final complete run.

All 30 accepted saved SGD models from the October 6 experiment were verified
against their recorded byte counts and SHA256 hashes before loading. Their
60,170 test scores were bitwise identical between the preceding predictor,
the profiler-enabled and disabled predictor, and the guarded numerical update.
Every original prediction and accuracy was retained. The audit reused the
saved fits without refitting or tuning; original experiment files were read
only. Profiling stores no diagnostic state on fitted estimators.

The numerical correction concerns scores extremely close to zero. Independent
high-precision RBF calculations exposed sign errors in symmetric small fitted
models where `exp(-gamma * distance_squared)` rounded nearly to one. In 30
Kernel/TwoStage full-gradient cases, the guarded formula reduced reference sign
disagreements from 35 to 1; the remaining ordinary-width exact-zero case lies
outside the guard. In 24 further fitted cases with large coordinate offsets,
disagreements decreased from 38 to 0. These are diagnostic fixtures, not
estimates of generalization accuracy. Additional coefficient/intercept stress
cases and exception-policy checks supplement the fitted cases.

The correction rewrites the same RBF function as the sum of the intercept,
coefficients, and coefficient-weighted `expm1` residuals, using compensated
summation. It applies only inside a score-roundoff envelope and when all scaled
direct squared distances are at most `sqrt(float64 epsilon)`. Explicit direct
prediction, other kernels, training and checkpoint evaluation retain their
existing arithmetic. This improves the tested cancellation cases; it does not
promise exact labels arbitrarily close to every decision boundary.

Sequential, alternating prediction timings used 16 numerical threads for one
saved model per MNIST pair, with 31 measurements per implementation and model.
All timing-control score bytes agreed. Final-source median times were:

| Saved model or control | Preceding, ms | Updated, ms |
| --- | ---: | ---: |
| MNIST 2 vs 3, repetition 00 | 11.7375 | 10.7263 |
| MNIST 4 vs 9, repetition 00 | 9.3247 | 9.5157 |
| MNIST 3 vs 8, repetition 00 | 9.3682 | 9.6073 |
| Synthetic, 512 queries and 128 centers | 0.3755 | 0.3884 |
| Tiny, one query and eight centers | 0.0509 | 0.0548 |

The tiny control used one thread and 101 alternating measurements because
threading overhead would dominate. These checks do not establish a consistent
speed advantage. Genuine correction costs more: a deliberately cancelling
66-center, 129-query fixture increased from 0.218 to 3.427 ms while repairing
64 reference label discrepancies. A nonconstant boundary fixture took
0.151 versus 0.160 ms with unchanged labels after the conservative prefilter
eliminated unnecessary distance probes. Those small affected-path measurements
used one thread and 31 paired trials. Absolute timings varied across sessions;
paired comparisons are the relevant evidence.

This is a numerical correction with an explicit performance cost. Enabled
profiling times include instrumentation overhead and must not be reported as
uninstrumented classifier benchmarks.

The historical evidence and resource settings below describe earlier checks.
Current correctness fixtures use one numerical thread for small problems;
the saved-model scoring audit and MNIST timing checks use 16. No former CPU
partition or RAM cap was reinstated.

# Historical Python 3.10 build compatibility

On October 5, 2026, Windows AMD64 builds were checked with CPython 3.10.12,
NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.0, joblib 1.5.1 and threadpoolctl
3.6.0. The native wheel used the `cp310-cp310-win_amd64` tag and linked against
the Python 3.10 runtime. The portable wheel used `py3-none-any` and contained no
compiled extension. Both declared Python 3.10 or newer at that time.

The same source was also rebuilt on CPython 3.12.14, retaining the
`cp311-abi3-win_amd64` native-wheel tag. Both native builds imported successfully
and enabled the guarded SciPy BLAS adapter. Strict arithmetic compiler flags
and the optimizer implementation were unchanged. Python 3.10 used its full C
API because the buffer interface used by the optimizer entered the stable ABI
in Python 3.11; a Python 3.11 stable-ABI binary is not repackaged for Python 3.10.
These build checks were performed on Windows, without claiming validation of
an operating system from matching dependency versions alone.

# Development evidence — direct BLAS calls in native fitting

On October 5, 2026, the guarded direct-BLAS adapter was compared with the
preceding native optimizer using the same 30 saved MNIST training draws:
ten each for pairs 2–3, 4–9 and 3–8. Fresh complete fits included StandardScaler,
both SGD stages, 800 epochs per stage, 30% retention and a 64 MiB lazy cache.
Original seeds and statistical parameters were fixed; no tuning was repeated.
Each environment used one native-library thread and alternating fit order,
with one timed fit per variant per draw. Predictions used a common direct
backend and were outside the fit timer.

| Environment | Previous mean fit, s | Direct-BLAS mean fit, s | Time saved | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| `py12_df_aocl` | 4.0911 | 2.5612 | 37.39% | 29/30 |
| `py12_df` | 4.3048 | 2.7127 | 36.98% | 30/30 |

Both environments used Python 3.12.14, NumPy 2.5.2, SciPy 1.18.0 and
scikit-learn 1.9.0 on the AMD 9955HX, with AOCL 5.2.0 or Intel MKL 2025.3.
These separate sessions compare implementations within each environment;
they do not establish a ranking of BLAS vendors.

All 60 paired comparisons preserved exact fitted parameter bytes, selected
rows, scores, labels, RNG states, update counts, and cache accounting within
their environment. Every candidate reproduced the original study's labels.
Test accuracy means and sample SDs across the ten training draws remained
97.4633% ± 0.3984 percentage points for 2–3, 96.0472% ± 0.4505 for 4–9,
and 96.5222% ± 0.2267 for 3–8. An independent audit recomputed summaries
from saved arrays and checked source, binary, data, protocol and state hashes.

The sole slower original MNIST observation was pair 4–9, draw 7: 4.2710 s
versus 4.3875 s, a 2.73% increase. It remains in the table. Six alternating
paired follow-up timings of that same draw averaged 3.9324 s versus 2.4387 s,
with all six faster and exact results retained. The original slowdown's cause
was not established; the follow-up did not replace its measurement.

Eleven synthetic controls used six paired timings per case and environment,
20 fixed epochs, and small, HDLSS, many-row, uncached, cached, duplicate and
large-offset inputs. All 132 pairs preserved exact parameters, scores, labels,
test DWD losses and controls. Ten case means improved in both environments,
although individual small timings sometimes lost. The high-dimensional
uncached kernel control gained only about 0.84%. A linear control with
features around `1e-160` repeatedly triggered numerical fallback and took
16.47% longer under AOCL and 18.96% longer under MKL, approximately
2.01 → 2.35 ms and 1.99 → 2.37 ms. All twelve pairs in that control were slower.
These fixed synthetic fixtures test implementation behavior, not statistical
accuracy across independently sampled datasets.

Before adoption, the exact guarded candidate passed the complete existing
suite in each environment: **605 tests and 9,160 subtests**, with only the
existing invalid-label warning. Additional checks passed 30 fresh-process
interface, hook, import-error and numerical-policy cases per environment,
plus 21 callback/logging/printing and pre-existing-flag cases. An earlier
unguarded prototype failed numerical-policy checks, including committing an
underflowing update that NumPy would reject; that prototype was not adopted.

After adoption, the rebuilt binary matched the benchmarked binary byte for
byte. The expanded package suite passed **613 tests and 9,160 subtests** in
each native environment. The portable build passed **535 tests and 4,807
subtests**, with 78 native-only skips. All three runs retained only the
existing invalid-label warning. The eight added regression tests cover
captured dot hooks, unsupported capsules, initialization errors, array
subclasses and floating-point policies, including the underflow commit rule.

The adapter preserves the methodology while changing how a scalar inner
product is invoked. Floating-point equality is empirical, not promised for
every NumPy/SciPy/BLAS build. Other platforms require their own validation.
The direct C path can retain the GIL through a native epoch; Python-thread
responsiveness and concurrent-training throughput were not measured.
See [optimizer details](docs/optimizer-backends.md) for dispatch and fallback.

# Development evidence — guarded BLAS prediction default

New `KernelSGDDWD` and `TwoStageDWDClassifier` instances use
`prediction_backend="blas"` for guarded positive-width RBF prediction.
Explicit `"direct"` remains available; saved backend choices and direct
prediction for older models lacking this option are preserved. Other kernels,
zero-width RBF, training updates and validation checkpoints are unchanged.

On October 5, 2026, fresh prediction comparisons used the same 30 fitted MNIST
models: ten original training draws for each of pairs 2–3, 4–9 and 3–8.
Historical models were reconstructed and checked against saved score and label
hashes in `py12_df_aocl`, then loaded without refitting in `py12_df`.
Each model used identical fitted state for the last-study direct implementation,
current direct implementation and guarded BLAS backend. Each environment timed
12 calls per model and route, **1,080 calls in total**, using all six route
orders twice, one numerical-library thread and 1,024-row prediction blocks.
Timings included the warm complete prediction pipeline, scaling and labels;
fitting, loading, correctness checks, warmup and I/O were excluded.

Guarded BLAS was **7.585× faster** than the last-study direct implementation
under AOCL. Under Intel MKL it was **5.775× faster** than current direct and
**5.846× faster** than last-study direct. All 30 models and every timed call
preserved the historical labels and accuracy in both environments. These are
within-environment prediction comparisons; the two sessions do not establish
a controlled ranking of BLAS vendors or a training-speed result.

Both environments used Python 3.12.14, NumPy 2.5.2, SciPy 1.18.0 and
scikit-learn 1.9.0 on the same AMD 9955HX. The loaded BLAS libraries were
AOCL 5.2.0 and Intel MKL 2025.3. Independent checks recomputed saved accuracy,
errors, confusion matrices, model-level sample standard deviations and timing
summaries, and verified source, runtime, model, data and prediction hashes.

Under MKL, **84 focused prediction tests and 68 subtests** passed. An additional
**72 combined high-dimensional and boundary cases** passed under both runtimes:
257/784 features, odd basis counts, large common offsets, signed near-zero
scores, and coefficients scaled from 1e-320 to 1e16. All tested labels matched;
guarded target blocks matched direct scores byte for byte. Coverage includes
old-model loading, explicit backend selection, layouts, bounded buffers,
overflow, underflow and direct recomputation.

The two direct implementations matched score bytes within each environment,
but direct scores differed between AOCL and MKL by up to approximately
9.24e-14. Within MKL, guarded BLAS differed from direct by up to approximately
6.97e-14. Extreme coefficient tests can have larger absolute score differences
away from guarded blocks. These safeguards and measured label agreement do not
promise identical scores or labels for every input or numerical library.

These measurements validate the existing guarded prediction implementation;
they precede the default change. The subsequent default-change regression
checks passed **605 tests and 9,160 subtests** with the native extension in
each of the AOCL and MKL environments. The portable package passed **535 tests
and 4,807 subtests**, with 70 native-only tests skipped. Each run retained the
existing warning from the invalid-label fixture and had no failures.
These checks cover default dispatch and cloning, explicit direct selection,
saved models with and without a backend attribute, non-RBF routing, and
unchanged Python/native training states and validation checkpoints.
Historical study outputs and earlier validation records are unchanged.

# Development verification — native epoch RNG snapshot reuse

On October 5, 2026, independent epoch RNG snapshot reuse passed **603 tests
and 9,160 subtests** with the existing compiled extension. A portable source
copy passed **533 tests and 4,807 subtests**, with 70 native-dependent skips.
Both suites produced only the existing invalid-label fixture warning. These
checks include the previously added Python derivative margin reuse.

For NumPy 2.5.2, the exact built-in `Generator` with `PCG64`, `PCG64DXSM`,
`Philox`, `SFC64`, or `MT19937` returns independent epoch-state dictionaries
and arrays. Native optimization uses that snapshot directly, avoiding another
deep copy. Other versions and custom or subclassed generators keep the copy.
The state getter stays before the same epoch draws. Sampling, update arithmetic,
learning rates and failure restoration/replay are unchanged. Initial, final,
selected and returned model snapshots retain their existing independent copies.
Source checks and tests cover getter/setter ownership, mutation of snapshots,
shared mutable custom getters, failed updates and subsequent RNG draws.

A confirmation benchmark used the current package, py12_df_aocl, NumPy 2.5.2
and one native thread. Sixteen fixed small-data cases had eight paired timing
blocks each, with case order randomized by a declared seed and each variant
running first four times. Blocks contained 4, 8 or 256 fresh fits as declared
before timing: 4,832 timed fits plus 16 warmup fits per implementation.
Construction and complete fitting were timed; input creation, prediction and
correctness comparisons were outside timing. Every repeated model matched
exact non-timing state, scores and labels. Inputs and caller RNGs were unchanged.

All 96 paired blocks across twelve 200-epoch built-in-generator cases were
faster, with 6.62–24.63% mean time savings and gains in both execution orders.
These deliberately small native workloads use 8 or 32 observations, a full
kernel cache and 100% retention for two-stage fixtures. The two default-seed
two-stage cases saved 15.23% and 6.62%, about 1.18–1.20 ms per fit. These gains
do not describe the default Python backend, usual 30% retention or general
large-data training. Earlier MNIST complete-fit means saved only 0.05–0.56%.

The custom-generator controls were approximately unchanged in the confirmation
(0.077% and 0.055% mean savings, four of eight pairs faster each), following
original means that were 0.198% and 0.019% slower. Tiny one-epoch timings remain
unstable: the linear mean was 4.665% slower despite six of eight pairs being
faster and a positive 1.369% median paired saving. One candidate block took
88.455 ms versus its paired baseline of 61.8125 ms. The kernel mean saved
0.211% with seven of eight pairs faster, but one block was 10.15% slower.
Its original run averaged 4.21% slower. No observation was excluded, and the
causes of the slow blocks were not established. These means do not support a
reliable one-epoch speed claim or a guarantee of no regressions.

This is a targeted optimization for repeated native epochs. The seven-line
Python guard adds no native API or compiled code; extending the optimized
path to another NumPy version requires implementation and state-parity checks.
The default backend, kernel-cache settings and statistical method are unchanged.
These are implementation timings on fixed inputs, not new CV repetitions or
accuracy evaluations. See [optimizer details](docs/optimizer-backends.md).

# Development verification — Python derivative margin reuse

On October 4, 2026, reusing the selected margin array in the Python DWD
derivative passed **597 tests and 8,345 subtests** with the existing compiled
extension. A portable source copy passed **528 tests and 4,730 subtests**,
with 69 native-dependent skips. Both suites produced only the existing
invalid-label fixture warning.

The derivative extracts margins above 0.5 once and reuses that array for the
same two sequential divisions. Input conversion, branch selection, result
validation and scalar/array return behavior are unchanged. Tests check exact
bytes at the threshold and extreme finite values, including negative-zero
underflow, array layouts, converted types, nonmutation and rejection errors.
This affects Python single-sample and batch updates. Compiled updates and
validation-loss calculations bypass this helper and are unaffected.

In py12_df_aocl with NumPy 2.5.2 and one native thread, 14 derivative cases
covered Python and zero-dimensional scalars, empty arrays, and vectors of
length 4, 64 and 4,096 with none, all or mixed margins above the threshold.
Each case had six alternating paired timing blocks with fixed call counts.
All 84 pairs were faster; case means used 1.57–19.54% less time.

Six complete Python-fit cases covered linear, kernel and two-stage estimators
with batch sizes 1 and 4, 16 observations, eight features and three fixed
epochs. Each of six paired blocks contained eight fresh fits. All six mean
times improved by 0.06–1.24%, but 14 of 36 individual pairs were slower and
timing order materially affected the small differences. These results do not
establish a reproducible regression or a substantial full-fit acceleration.
Every timed fitted model matched exact non-timing state, scores and labels;
inputs remained unchanged. These are implementation timings on fixed inputs,
not new CV repetitions or an accuracy study.

The change adds one local variable, with no new API, branch, compatibility
path or native build. It removes a repeated allocation and copy; lower peak
memory is not claimed because the temporary stays live through validation.
The benchmark evidence supports a modest optimization of the Python path.

# Development verification — direct native linear row access

On October 4, 2026, direct native linear row access passed **593 tests and
8,280 subtests** with the rebuilt extension. Compatibility with the previous
extension passed **581 tests and 7,658 subtests**, with 12 capability-dependent
skips. A separate portable source copy passed **524 tests and 4,665 subtests**,
with 69 native-dependent skips. Each suite produced only the existing warning
from an invalid-label fixture.

The native linear loop retrieves the same row object directly from its exact
built-in tuple, avoiding per-update Python index creation and a bound-method
call. Buffer validation remains inside the update loop. NumPy dot products,
gradients, projection, finite checks, sampling, learning rates and committed
failure prefixes are unchanged. A capability flag retains callable access
with older extensions; kernel and custom callable providers keep their path.
Tests cover every update prefix, malformed sampled and unsampled rows, read-only
and invalid layouts, reference ownership, numerical and structural failures,
public estimator dispatch, both sampling policies and older-binary fallback.

Complete-fit timings used MNIST pairs 2–3, 4–9 and 3–8, with saved training
draws 00, 04 and 09 per pair. Each draw was timed twice in alternating order:
six paired timings per pair, not six independent training draws. Fresh fits
included StandardScaler and both native stages at 800 epochs, 30% retention
and a 64 MiB lazy-cache payload limit. The runtime was py12_df_aocl with
NumPy 2.5.2 and one native thread. Means and sample SDs combine the three
draws and their two timing repetitions.

| Pair | Previous fit, s (mean ± SD) | Direct rows, s (mean ± SD) | Time saved |
| --- | ---: | ---: | ---: |
| 2–3 | 4.173 ± 0.029 | 4.041 ± 0.042 | 3.15% |
| 4–9 | 4.021 ± 0.062 | 3.945 ± 0.093 | 1.88% |
| 3–8 | 4.092 ± 0.048 | 3.991 ± 0.039 | 2.47% |

The candidate was faster in 17 of 18 complete-fit comparisons. One 4–9 timing
was 2.37% slower. All 90 paired complete and short kernel fits matched exact
non-timing fitted states, coefficients, RNG states, selections, cache/kernel
counters, scores and predicted labels. All nine complete-model configurations
reproduced the original study score hashes. Inputs and historical study
outputs were unchanged; no test labels were loaded or settings retuned.

Separate short kernel fits used one or five epochs on retained draw-00 data,
with full, evicting, disabled or subcolumn cache budgets. Five control means
were slower by 0.017–0.232%; these are not complete two-stage fit timings.
Separate fixed-update controls used 1, 8, 784 and 4,096 features with and
without projection, one/two-update blocks, and unchanged callable routes.
All 72 paired control blocks matched exactly. The tuple-route control means
used 1.23–4.87% less time; the callable linear control averaged 0.044% slower.
Those controls exclude sampling/rate preparation, preprocessing and kernel
evaluation and do not establish complete-fit performance.

The change is included as a modest optimization with limited implementation
complexity. These measurements do not establish a speedup for every supported
input or platform. The Python backend and older native binaries keep their
existing execution paths. See [optimizer details](docs/optimizer-backends.md).

# Development verification — native kernel coefficient validation

On October 4, 2026, combined native kernel shrinkage and finite-value checks
passed **580 tests and 7,655 subtests** with the rebuilt extension. The same
tests passed with the previous extension. A separate portable source copy
passed **523 tests and 4,662 subtests**, with 57 native-dependent skips.
Each suite produced only the existing warning from an invalid-label fixture.

For ordinary NumPy coefficient buffers, the shrinkage loop checks each stored
product while writing all coefficients. The sampled coefficient is then updated
with the original separate subtraction and checked again. Custom buffers keep
the previous separate scan. The update arithmetic, objective, sampling,
learning rates, cache policy and failure commit point are unchanged. Checks
cover signed zeros, subnormal values, product and subtraction overflow,
custom-buffer behavior, complete fitted states and failed-update RNG recovery.
The improvement requires the rebuilt native extension; older extensions and
the Python backend retain their existing paths.

Complete-fit timings used MNIST pairs 2–3, 4–9 and 3–8, with saved training
draws 00, 04 and 09 per pair. Each draw was timed twice in alternating
implementation order: six paired timings per pair, not six independent
training draws. Each fresh fit included StandardScaler and both native stages
at 800 epochs, 30% retention and a 64 MiB lazy-cache payload limit. The runtime
was py12_df_aocl with NumPy 2.5.2 and one native thread. Means and sample SDs
combine the three draws and their two timing repetitions.

| Pair | Previous fit, s (mean ± SD) | Combined checks, s (mean ± SD) | Time saved |
| --- | ---: | ---: | ---: |
| 2–3 | 4.236 ± 0.073 | 4.166 ± 0.079 | 1.7% |
| 4–9 | 4.123 ± 0.079 | 4.039 ± 0.044 | 2.0% |
| 3–8 | 4.191 ± 0.058 | 4.113 ± 0.050 | 1.9% |

The candidate was faster in 17 of 18 complete-fit comparisons. These results
did not meet the predeclared 1.05× complete-fit speedup target in any pair.
The target and measurements are retained; the change is included as a modest
incremental optimization after review of its localized implementation and
exact tested equivalence.

Separate short kernel fits used each pair's standardized retained draw 00
subset, with six paired timings per cache regime. Cold full-cache fits used
one epoch; evicting, disabled and subcolumn-budget fits used five epochs.
Their mean time reductions ranged from 0.003% to 0.376%, providing little
practical gain in these controls. These timings are not complete 800-epoch
two-stage fit times.

All 90 paired comparisons matched exact non-timing fitted states, coefficients,
RNG states, selections, kernel/cache counters, scores and predicted labels.
All nine saved complete-model configurations reproduced original study score
hashes. Inputs and historical study outputs remained unchanged. No test labels
were loaded and no statistical settings were retuned. Results describe the
tested workloads and runtime. See [optimizer details](docs/optimizer-backends.md).

# Development verification — native RBF input validation

On October 4, 2026, guarded RBF column preparation passed **569 tests and
7,558 subtests** with the native extension and **523 tests and 4,662 subtests**
in a separate portable source copy, with 46 native-dependent skips. Each full
suite produced only the existing warning from an invalid-label fixture.
The focused checks passed 14 tests and 78 subtests without warnings.

For eligible native RBF fits, fixed training inputs and kernel width are
validated once. Each cache miss still computes a fresh column using the same
squared-distance orientation and exponential operations, including finite
distance and result checks. Columns remain lazy and independently stored.
Custom estimators, generators, array layouts and replaced kernel hooks retain
the previous path. The objective, sampling, learning rates, sequential
updates, cache policy and compiled binary are unchanged. No new dependency
or public tuning parameter is introduced.

A complete-fit benchmark used digit pairs 2–3, 4–9 and 3–8, with saved training
draws 00, 04 and 09 for each pair. Each draw was timed twice in alternating
implementation order: six paired timings per pair, not six independent
training draws. Timings included StandardScaler and both stages at 800 epochs,
30% retention and a 64 MiB lazy-cache payload limit, using py12_df_aocl,
NumPy 2.5.2 and one native thread. Means and sample SDs combine the three
saved draws and their two timing repetitions.

| Pair | Previous fit, s (mean ± SD) | Prevalidated RBF fit, s (mean ± SD) | Time saved |
| --- | ---: | ---: | ---: |
| 2–3 | 4.372 ± 0.026 | 4.332 ± 0.073 | 0.9% |
| 4–9 | 4.236 ± 0.046 | 4.160 ± 0.046 | 1.8% |
| 3–8 | 4.338 ± 0.062 | 4.260 ± 0.079 | 1.8% |

Separate short kernel-stage checks used the standardized retained subset of
draw 00, with six paired timings per pair and cache regime. A cold full cache
used one epoch; evicting, disabled and subcolumn-budget caches used five.
Across digit pairs, evicting and disabled caches saved 20.5–20.9% of mean fit
time; the cold full cache saved 20.3–20.6%, and subcolumn budgets saved
20.7–20.9%. All 72 short kernel-stage pairs were faster. These timings measure
cache misses over short runs and do not imply a comparable saving in the
800-epoch complete fits.

All 90 paired checks matched exactly for non-timing model state, coefficients,
RNG states, selections, cache/evaluation counters, scores and predicted labels.
All nine complete-fit configurations reproduced their original study score
hashes. Checks also covered failed-column recovery, hook fallbacks and input
immutability. No test labels were loaded and no statistical settings were
retuned. The gains describe the tested workloads; historical study outputs
are unchanged. See [optimizer details](docs/optimizer-backends.md).

# Development verification — native linear coefficient validation

On October 4, 2026, guarded removal of a redundant native linear coefficient
scan passed **555 tests and 7,480 subtests** with the rebuilt extension. The
same source tests also passed with the original released extension. A separate
portable source copy passed **516 tests and 4,639 subtests**, with 39
native-dependent skips. Each suite produced only the existing warning from
an invalid-label fixture.

The proposal loop already checks each new coefficient for finiteness. When
ordinary NumPy scratch arrays need no projection division, the native solver
avoids checking that unchanged proposal a second time. Projected linear
updates, kernel updates and custom array types retain the final scan. Tests
cover projection boundaries, overflow, subnormal values, custom-array mutation,
committed failure prefixes, complete fitted states and RNG states. Update
arithmetic, projection, sampling, learning rates and public defaults are
unchanged. This improvement requires the rebuilt native extension; older
extensions and the Python backend retain their existing execution paths.

Complete-fit timings used digit pairs 2–3, 4–9 and 3–8 and saved training draws
00, 04 and 09 per pair, each timed twice in alternating implementation order.
These are six paired timings per pair, not six independent training draws.
Each fresh fit included StandardScaler and both native stages at 800 epochs,
30% retention and a 64 MiB lazy-cache payload limit. The environment was
py12_df_aocl with NumPy 2.5.2 and one native thread. Means and sample SDs
combine the three training draws and their two timing repetitions.

| Pair | Previous fit, s (mean ± SD) | Guarded scan removal, s (mean ± SD) | Time saved |
| --- | ---: | ---: | ---: |
| 2–3 | 4.647 ± 0.049 | 4.250 ± 0.021 | 8.5% |
| 4–9 | 4.542 ± 0.060 | 4.151 ± 0.042 | 8.6% |
| 3–8 | 4.592 ± 0.042 | 4.232 ± 0.047 | 7.8% |

All 18 complete-fit comparisons were faster. Every measured pair matched
non-timing fitted states, coefficients, RNG states, selected observations,
cache/evaluation counters, scores and predicted labels exactly. All nine
complete-model configurations reproduced their original saved study score
hashes. No test labels were read and no statistical settings were retuned.

Separate native linear-update controls used 200,000 updates and 784 features,
with six alternating paired timings for each fixture. The unprojected fixture
saved 11.6% of update-block time. Mean changes in the two projection fixtures
were slowdowns of 0.28% and 0.31%. These control timings exclude preprocessing,
sampling preparation, kernel fitting and prediction; they are not complete
fit timings. All control results and inputs matched exactly.

These results apply to the tested workloads and runtime. They do not certify
every compiler or numerical platform. The optimization assumes normal NumPy
operations and preserves the existing checks for array subclasses. Historical
study outputs are unchanged. See [optimizer details](docs/optimizer-backends.md).

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

# Historical development verification — optional BLAS RBF prediction

This section records the October 4 introduction of BLAS as an opt-in backend,
when direct prediction was the default. The default for new instances now
uses guarded BLAS, as described above; the measurements below retain their
original implementation and protocol scope.

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

At this validation stage, direct prediction remained the default. The BLAS
backend changes floating-point reduction order; its safeguards and observed
label agreement are not universal
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
