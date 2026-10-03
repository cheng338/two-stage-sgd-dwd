# Changelog

## Unreleased - 0.6.0.dev20261003

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
  `two_stage_sgd_dwd` to identify its stochastic gradient descent method.
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
