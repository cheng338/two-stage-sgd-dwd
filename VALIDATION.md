# Development verification — 0.6.0.dev20261003

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

## Boundaries

The package has been verified on this Windows/Python environment. Other
supported Python versions and operating systems have not been exercised here.
[Training-only MNIST exploration](docs/mnist-exploration.md) is recorded
separately from these correctness checks. It does not establish optimizer convergence, consistency,
noninferiority, screening superiority, or official-test performance. Selected
parameters are conditional on the stated data, seeds, and finite SGD budgets.

The committee requirements table distinguishes package support from unfinished
experimental and dissertation work. Input hashes and the handoff/current
chapter source-hash discrepancy are recorded in `docs/provenance.json`.

To repeat the portable correctness suite after installation:

```console
python -m pytest -q tests
python examples/synthetic_example.py
python examples/kernel_search_example.py
```

Installation uses the declared runtime dependencies. Verification uses isolated
test and build tools without altering the shared Conda environments. The wheel
is installed into a separate verification directory.
