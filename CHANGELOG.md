# Changelog

## 0.9.2 - 2026-10-17

- State All Rights Reserved while licensing terms remain undecided, include
  the notice in distributions, and update the dissertation title in citations.
  Use current license metadata with setuptools 77.0.3 or newer for source
  builds; runtime dependencies are unchanged.
- Add explicitly requested post-fit `optimization_diagnostics()` for linear,
  kernel and two-stage fits. Report numerical objective-gap and stationarity
  diagnostics using each stage's actual fitting data and free intercept.
  Preserve fitted state, optimization budgets and convergence flags; the
  numerical estimates are not exact certificates or automatic stopping rules.
- Add independent objective regression tests for imbalanced kernel data with
  conflicting duplicate observations and for wide linear data. Retain fixed
  SGD budgets, free intercepts and explicit finite-budget objective tolerances.
- Report training and selected class composition, including retention within
  each class, in two-stage fits, MNIST CV receipts and multiclass pair summaries.
  Preserve original labels and distinguish an empty selection's undefined class
  proportions from its zero retention. Selection and optimization are unchanged.
- Enable the validated vectorized replacement-sampling path on NumPy 2.5.3 and 2.5.4
  for the native single-observation optimizer. Preserve sampled observations,
  generator state, update order and failure recovery; retain scalar sampling
  for unvalidated versions and custom generator types. SGD still updates one
  observation at a time, and the snapshot-copy optimization is unchanged.

## 0.9.1 - 2026-10-15

- Use patch releases for grouped fixes, performance improvements and optional
  diagnostics or workflow controls. Reserve minor releases for substantial new
  estimator capabilities or incompatible changes to the public estimator
  contract. Keep
  existing release numbers and artifacts unchanged.
- Round the two-stage retained count down with
  `floor(retention_fraction * N)`, matching the ensemble classifiers' count
  convention. Preserve global score ranking and tie breaking, without adding
  rows to meet a minimum count or class quota. Report an empty selection as
  `SelectionError` with status `empty_selection`; retain
  `single_class_selection` for a nonempty one-class subset. New fits may use
  one fewer row and produce different fitted results. Existing study records
  and saved models are unchanged. Earlier adaptive searches require a fresh
  output directory because their implementation identity differs.
- Require Python 3.11 or newer. Remove the version-specific Python 3.10 native
  build path, retain the Python 3.11 stable ABI for supported native builds,
  and move the NumPy 1 installed-wheel CI jobs to Python 3.11. Portable wheels
  have the same minimum Python version. Numerical code and SGD updates are
  unchanged by this packaging change.
- Compare adaptive MNIST CV accuracies using exact equal-weight means of the
  recorded correct/validation counts. Preserve the existing parameter tie
  order and genuine improvements, including improvements smaller than one
  floating-point unit. Validate count evidence on replay and bind exact score
  identities to policy state and qualified selections. Custom scalar-only
  examples remain supported. Older adaptive checkpoints require a new output
  directory; ordinary scikit-learn `make_search` behavior is unchanged.
- Recover representable geometric grid extensions and midpoints when the
  original floating-point calculation incorrectly refuses them. Use the same
  geometric rule, retain every previously successful coordinate, and continue
  to refuse true arithmetic limits and intervals with no interior value.
- Add optional external `prediction_profile()` records for bounded RBF scoring:
  kernel, score and guard times, completed blocks, and direct-fallback reasons
  and counts. Preserve numerical operations and fitted state. Profiling is off
  by default; enabled timings include instrumentation overhead.
- Stabilize the BLAS predictor's near-constant RBF scores when coefficient and
  intercept cancellation reaches the floating-point error envelope. Recompute
  only those rows using direct distances, `expm1` and compensated summation,
  subject to a binary64 precision guard. Preserve the explicit direct backend,
  training updates and ordinary prediction scores. Record diagnostic probes
  separately from applied corrections.

## 0.9.0 - 2026-10-06

- Add an optional read-only example helper to summarize charged component times
  separately for newly evaluated and replayed folds in a completed candidate.
  Preserve historical measurements and explicit completion-evaluation scope;
  reject missing replay metadata and invalid times. Leave runner records,
  recovery schemas, fitting and search behavior unchanged.
- Save an atomic completed-fold prefix for the adaptive MNIST runner's active
  candidate and inherited-default control. Verify run, candidate, stage, runtime
  and preceding-record identities before reuse; retain original fold evidence
  and timings. Archive incomplete evidence and restart that candidate after an
  explicitly permitted numerical-runtime change. Preserve completed-candidate
  receipts, search policy and SGD updates; require a fresh output directory for
  older search schemas.
- Check pending Python signals every 1,024 completed native scalar updates on
  the main thread. Propagate exceptions from the check out of the native loop
  instead of returning a failed-update prefix. Preserve public exception
  handling, update arithmetic, row cleanup and ordinary failure recovery.
  The checks do not release the GIL.
- Avoid an extra kernel-block copy for locally allocated cosine and constant
  outputs after shape and finite checks. Preserve independent copies for
  callable, imported-kernel and subclass-hook outputs. Reduce temporary
  allocation without changing kernel arithmetic, SGD updates or defaults.
- Add a small external scikit-learn one-vs-one example with copied per-pair
  diagnostics. Document pair-local labels and observation IDs, selection
  failures and validation-data limitations; preserve binary estimators,
  defaults and SGD updates.
- Add Windows installed-wheel checks for Python 3.10 with NumPy 1 and Python
  3.12 with NumPy 2, covering native and portable builds. Require installed
  package origins and actual backend behavior before running the full suite
  from a separate test bundle without package sources; retain environment,
  wheel and test reports.
- Apply post-fit `TwoStageDWDClassifier.prediction_batch_size` changes to each
  prediction call through a read-only child-scoring override. Validate parent
  and child block sizes, preserve fitted child state, and retain training and
  prediction defaults.
- Align batch-methodology and study-status documentation with the October 4
  dissertation: averaged batch updates are already permitted, while the
  completed replicated study uses single-observation updates with replacement.
  Keep historical guide settings distinct from the reported study.
- Use extended Windows paths for adaptive MNIST output and checkpoint I/O,
  including temporary files, run locks, implementation snapshots, resume and
  qualified-selection reads. Preserve caller-supplied paths and lock messages,
  strict checkpoint validation, and existing failure-recovery behavior.
- Preserve evaluation exceptions and interruptions when adaptive MNIST failure
  diagnostics cannot be written. Attempt failure and status reports separately;
  keep successful checkpoint writes strict. Validate JSON before creating its
  temporary file and clean up owned temporary files after failed writes without
  replacing the original error if cleanup also fails.
- Add explicit `--allow-blas-change` support for compatible numerical-build
  changes between adaptive MNIST processes. Preserve completed receipts and the
  original configuration, record transitions with the next completed candidate,
  and disclose the selected candidate's runtime and full runtime history.
  Keep strict defaults and evaluator-lifetime checks; mixed-runtime resumes and
  qualified loads require opt-in on every invocation. Keep versions, package
  implementation, data and numerical controls fixed, and require a fresh run
  for checkpoints created before this migration-aware schema.
- Add inline candidate receipts to adaptive MNIST records, binding their full
  content and sequence to the saved configuration and folds. Verify complete
  records before resumed fitting or interrupted-tail repair, reject duplicate
  JSON fields in checkpoints, and require a fresh directory for older
  checkpoints without receipts. Preserve SGD updates,
  search policy, and recovery of genuinely interrupted final writes.
- Support Python 3.10 with portable builds and a version-specific CPython 3.10
  native extension. Retain the Python 3.11 stable ABI for native builds on
  Python 3.11 and newer, with the same strict arithmetic and SGD implementation.
- Bind adaptive MNIST resumes and screening-cache reuse to the loaded numerical
  runtime, including scientific extension hashes, BLAS/OpenMP implementations,
  library hashes and active thread counts. Capture provenance under the same
  thread settings used for fitting and restore caller settings after loading
  qualified selections. Discard screening caches after failed or interrupted
  candidates so unverified fits cannot be reused. Require a fresh directory for
  older checkpoints that did not record this runtime identity; preserve SGD
  updates and search policy.
- Prevent overlapping adaptive MNIST coordinators from modifying the same
  output directory, and hold the operating-system lock through resume and
  qualified-selection export. Validate unique candidate records and complete
  fold evidence before saving or reusing them, including completed searches.
- Preserve normalized floating-point coordinates in the adaptive MNIST
  example and use exact candidate identities instead of 15-digit rounding.
  Keep distinct nearby values as separate candidates, and require a fresh
  output directory for searches saved under the older identity scheme.
- Recover representable mean DWD losses when their intermediate sum overflows,
  including objectives, validation and cross-validation scoring. Preserve
  ordinary loss averages and reject genuinely nonfinite kernel objectives.
- Validate prediction block sizes after fitting, including in the standalone
  RKHS-norm diagnostic. Reject invalid sizes before scoring can return
  unwritten output, and preserve valid predictions and fitted model state.

## 0.8.0 - 2026-10-05

- Call BLAS directly for ordinary float64 dot products in the compiled scalar
  optimizer through a guarded SciPy interface. Retain NumPy dispatch for
  unsupported interfaces and custom arrays or hooks; replay exceptional dot
  products through NumPy to preserve floating-point warnings and failures.
  Preserve SGD updates, sampling, projection, selection, and lazy kernel reuse.
  Document measured training gains and the underflow-heavy input slowdown.

- Reuse independent epoch RNG snapshots for native fits using the verified
  NumPy 2.5.2 built-in generators, avoiding redundant deep copies. Preserve
  sampled streams, failed-update restoration and saved model states; retain
  deep copies for other versions and custom generators.

- Reuse selected margins when evaluating the Python DWD derivative,
  avoiding a duplicate array copy while preserving division order,
  input validation, scalar and batch updates, and exact tested results.

- Retrieve native linear fitting rows directly from their existing tuple,
  avoiding a Python index and method call per update. Preserve row validation,
  update arithmetic and failure recovery; retain callable kernel loaders and
  compatibility with older compiled extensions.

- Combine native kernel shrinkage and finite-value checks for ordinary NumPy
  coefficient buffers, then validate the sampled coefficient after its loss
  update. Preserve arithmetic, failure recovery, and custom-array checks.

- Validate fixed positive-RBF fitting inputs once for ordinary native kernel
  fits. Compute fresh training columns lazily with the same SciPy squared
  distances, arithmetic, and finite-result checks; retain the original path
  for custom state and hooks. Preserve cache accounting and SGD updates.

- Avoid a duplicate native linear coefficient scan when an ordinary NumPy
  proposal has already passed finite-value checks and needs no projection.
  Retain the final scan for projected updates, kernel updates, and custom arrays.

- Prepare replacement-sampling indices with one NumPy call per native epoch
  on the verified NumPy 2.5.2 built-in generators. Keep scalar draws for
  other versions and custom generators. Preserve single-observation updates,
  learning rates, epoch budgets, and failed-update RNG restoration.

- Use `prediction_backend="blas"` by default for new kernel and two-stage
  estimators, with guarded matrix-product RBF prediction. Keep `"direct"`
  available explicitly. Preserve saved backend choices and direct prediction
  for older models lacking the option. Training, checkpoint calculations,
  non-RBF kernels, and zero-width RBF prediction paths are unchanged.
- Recompute suspicious distances and uncertain score signs directly. Document
  floating-point score differences, bounded prediction buffers, and measured
  prediction performance separately from training performance.

- Expose optional optimizer-backend and lazy kernel-cache controls in the
  MNIST tuning example while preserving fold-local first-stage reuse and
  the Python/zero-cache defaults. Record actual backends and cache usage.
- Bind tuning resumes and qualified settings to their execution controls
  and native-extension identity; include native bytes in saved
  implementation snapshots when the extension is available.

- Reuse one bounded RBF prediction buffer and validate the fitted basis and
  bandwidth once per prediction call. Preserve direct squared-distance
  arithmetic, finite-result checks, fitted state, and all training paths.
  Reduce temporary prediction allocations; no substantial speedup is claimed.

## 0.7.0 - native SGD and lazy kernel reuse, 2026-10-04

- Add optional fit-local lazy training kernel-column reuse with a byte-bounded
  LRU cache. Keep caching disabled by default, preserve single-observation
  update arithmetic, and report cache usage and actual kernel evaluations.
- Add an optional compiled scalar optimizer with `optimizer_backend` choices
  `python` (default), `auto`, and `native`. Report automatic fallback reasons;
  reject unsupported forced-native fits and preserve numerical failures.
  Support fixed-epoch, single-observation fits without validation, callbacks,
  or wall-time limits. Keep portable Python-only builds available.

### Stopping and tuning updates included from the 0.6.0 development baseline

- Add explicit fixed-epoch and validation-patience stopping controls, with
  separate cap, model usability, CV eligibility, and optimality diagnostics.
  Preserve SGD updates and the default 200 epochs per stage.
- Align adaptive tuning with the ensemble workflow: three expansion rounds,
  500 candidates, valid conditional factor-two brackets, and two stable
  completed local rounds. Refinement repeats until qualified or budget-limited.
- Hold unresolved searches as `needs_review`, including invalid or missing
  conditional neighbors. Export locked final-evaluation settings only after
  acceptance; retain training-only data access and exact-identity resume.

## 0.5.0 - MNIST tuning ranges and adaptive exploration, 2026-10-01

- Adopt the ensemble DWD package's MNIST lambda and gamma ranges and its
  10%, 15%, 20%, 25%, and 30% retention choices. The recommended coarse and
  broader grids contain 125 and 405 combinations respectively.
- Remove 5% from recommended presets after the recorded training-only checks.
  Preserve the historical 150-candidate grid and 486-candidate broader preset
  definition, which included 5%, in the exploration records.
- Set kernel and two-stage defaults to `lambd=1e-6` and `gamma="scale"`, with
  two-stage retention at 30%, matching the ensemble package's starting values.
  Keep 5% as the original Chapter 3 setting and an explicit API option.
  Single-observation updates and independent sampling with replacement remain
  unchanged. Explicit parameter values retain their meaning.
- Add an explicit training-only MNIST tuning example with saved folds,
  within-fold linear-fit reuse, complete boundary-expansion stages,
  geometric refinement, an inherited-default control, and recoverable records.
- Record finite-budget exploration separately from optimizer convergence
  and final held-out performance.

## 0.4.2 - package naming, 2026-10-01

- Name the distribution `two-stage-sgd-dwd` and the Python import package
  `two_stage_sgd_dwd` to identify its stochastic gradient descent optimization algorithm.
- Update imports, examples, and software citations. Solver behavior is unchanged.

## 0.4.1 - citations and source references, 2026-10-01

- Document development from Chang Cheng's original two-stage SGD-DWD research
  scripts accompanying his dissertation, with the methodology in Chapter 3.
- Add software, dissertation, and foundational DWD paper citations in
  Citation File Format and BibTeX, with verified publication identifiers.
- Include citation files in the source distribution. No solver changes.

## 0.4.0 - one regularization parameter, 2026-10-01

- Use only `lambd`, with objective mean DWD loss + lambd * ||h||^2, matching
  the `dwd` convention. Remove the former `penalty_lambda` argument entirely.
- Set the default to 5e-4, preserving the effective regularization strength.
  Use gradient 2 * lambd * h, shrinkage 1 - 2 * eta * lambd, and kernel
  step-size bound 2 * lambd + 4 * (max diagonal + 1).
- Convert the MNIST regularization grid, including half-decade candidates,
  to the same coefficient convention. Bandwidth and retention candidates,
  and the 125/405 combination counts, are unchanged.
- Document the conversion from the dissertation's half-norm coefficient;
  preserve the constrained linear stage and original source references.

## 0.3.0 - kernels and explicit parameter search, 2026-10-01

- Support `kernel`, `gamma`, and `kernel_kws`, including matrix callables and
  numeric/auto/scale bandwidths. Resolve scale on the entire training fold
  before screening, and freeze fitted parameters.
- Use a kernel-specific SGD bound. Validate custom and untrusted training
  Grams for symmetry/PSD once, then discard them.
- Add the previous packages' norm-coefficient convention while preserving
  the original effective regularization strength and RBF bandwidth.
- Add explicit `make_search`, a two-stage `mnist_param_grid` jointly tuning
  regularization, bandwidth and retention (125 or 405 candidates), and a negative
  unpenalized DWD-loss scorer. Require an explicit search grid, reject
  untrained candidates, and reject shared monitoring data in the helper.
- Explain which choices follow statistical practice and which need a
  study-specific justification. Keep original SGD/sampling defaults.
- No MNIST study or implicit tuning is introduced.

## 0.2.0 - optional batches and random reshuffling, 2026-10-01

- Add `batch_size=1`, positive integer mini-batches, and `batch_size="full"`
  to all three binary estimators. Average old-state gradients over the actual
  batch; project linear slopes once and apply the kernel regularizer once.
- Add optional `sampling="shuffle"`: one new permutation per epoch, each row
  visited once. The defaults remain batch size one and independent sampling
  with replacement, retaining the earlier numerical trajectory and matching
  the dissertation's sampling policy.
- Record processed observations separately from optimizer updates, keep
  validation on epoch boundaries, and decay step size by processed passes.
- Evaluate kernel batches in bounded blocks, accumulate repeated indices
  correctly, and discard any incomplete update when capped during scoring.
- Add batch mathematics and estimator tests plus dissertation alignment notes.

## 0.1.0 - local development release, 2026-10-01

- Implement normalized DWD loss and exact projected linear / RKHS kernel SGD
  updates from the dissertation methodology, replacing legacy penalized linear and
  coefficient-space routines.
- Replace aliased quota-based selection with immutable global absolute-score
  selection and label-independent tie keys; record one-class failures.
- Remove ineffective historical C/sigma arguments. Use one explicit RBF gamma
  and subset-average regularization convention.
- Separate scalar free intercepts, deterministic private RNG streams,
  checkpoints, validation selection, patience, and failure status.
- Add sklearn-compatible binary estimators and validated local IDX loading.
- Retain the original scripts and implementation guide outside the package source,
  unchanged. Their hashes and source provenance are recorded in docs/provenance.json.
- Do not carry over historical accuracy comments as validated results.
