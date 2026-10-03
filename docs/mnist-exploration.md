# MNIST development exploration: digits 7 and 8

October 1, 2026; package version 0.5.0.

The package adopts the ensemble DWD starting defaults: RBF,
`lambd=1e-6`, `gamma="scale"`, and **30% retention**. The initial
search additionally tested **5% retention**, which was removed from the
recommended grids after poor results persisted in the larger-sample check.
The recommended grids now contain **125 and 405 combinations**. This exploration supports
refining the bandwidth and checking larger retained subsets on this task;
it does not establish a universal optimal default.

## Protocol

Only the official MNIST **training partition** was opened. Pixels were divided
by 255. The initial search used 600 balanced observations, saved stratified
fivefold splits, seed 42, and 40 epochs per stage. Each candidate refitted the
kernel stage; an identical linear fit and ranking were reused only within the
same training fold and settings. Scoring folds never selected SGD checkpoints.
Updates sampled one observation independently with replacement. Selection used
mean fold accuracy, preferring smaller retention, then larger lambd and smaller
gamma in exact score ties.

The initial grid contained 150 candidates: the inherited lambda/gamma grid
crossed with 5%, 10%, 15%, 20%, 25%, and 30% retention. The 486-candidate
initial broader preset filled gaps within the same endpoints; it was not
exhaustively run. Subsequent exploration used recorded boundary extensions
and refinements. Removing 5% leaves the same lambda/gamma ranges and retention
choices 10%, 15%, 20%, 25%, and 30% in the current recommended presets.

## Search development

| Completed step | Additional candidates | Best development CV accuracy |
| --- | ---: | ---: |
| Initial inherited grid, including 5% | 150 | 97.833% |
| Two lower-lambda endpoint extensions | 60 | 97.833% |
| Two local geometric refinements | 180 | 98.333% |
| Further refinement and broader lambda cross-section | 168 | 98.333% |
| Retention midpoint and extensions above 30% | 6 | 98.833% |

There were **564 distinct grid candidates**, all eligible, with 2,820 kernel
fold fits. The separate original 5% default control added five fold fits.
The original 390-candidate search and later 168-candidate follow-up used
different refinement tie priorities; their source snapshots and complete
plans preserve the actual adaptive history.

The useful bandwidth region moved from gamma 0.01 to about 0.0316228.
Decreasing the lower lambda endpoint twice did not improve accuracy. Several
lambda values tied: `2**-10 = 0.0009765625` was the largest tied value on the
later cross-section at gamma 0.0316228 and 30% retention. Tiny lambda values
are therefore not uniquely supported by this finite-budget search.

At that frozen lambda/gamma pair, retention values 27.5%, 35%, 40%, 50%, 75%,
and 100% scored 98.333%, 98.500%, 98.500%, 98.667%, 98.833%, and 98.833%.
This was a retention cross-section, not a joint search over every larger
retention and kernel parameter combination.

## Budget and seed checks on 600 observations

Six candidate settings were frozen before checking 80 and 200 epochs with
SGD seeds 42, 20261001, and 314159 on the same observations and folds.
All 36 candidate evaluations completed (180 kernel fold fits). At 200 epochs:

| Setting | Retention | lambd | gamma | Mean CV accuracy | SD across seed means |
| --- | ---: | ---: | ---: | ---: | ---: |
| Extended-retention candidate | 75% | 0.000976562 | 0.0316228 | 98.833% | 0.000 percentage points |
| Full-retention control | 100% | 0.000976562 | 0.0316228 | 98.889% | 0.096 percentage points |
| 50% candidate | 50% | 0.000976562 | 0.0316228 | 98.667% | 0.000 percentage points |
| Best initial-grid candidate | 25% | 6.10352e-05 | 0.01 | 98.333% | 0.167 percentage points |
| Tuned 5% candidate | 5% | 6.10352e-05 | 0.0316228 | 66.444% | 1.584 percentage points |
| Original 5% control | 5% | 1e-06 | scale | 64.500% | 1.302 percentage points |

The 75% candidate also averaged 98.833% at 80 epochs; full retention averaged
98.944%. These small differences do not establish a screening advantage.
Zero variation in accuracy across three seeds does not imply zero uncertainty
or convergence of coefficients, loss, or the regularized objective.

## Larger training-sample check

The follow-up used 2,000 balanced observations, five saved folds, 200 epochs
per stage, and the same three SGD seeds. Before fitting, six settings were
frozen: the 75% candidate, the initial-grid winner, tuned 5% and 30% candidates,
the inherited 30% default, and the original 5% control. All 18 evaluations
completed (90 kernel fold fits).

| Setting | Retention | lambd | gamma | Mean CV accuracy | SD across seed means |
| --- | ---: | ---: | ---: | ---: | ---: |
| Extended-retention candidate | 75% | 0.000976562 | 0.0316228 | 99.317% | 0.029 percentage points |
| Best initial-grid candidate | 25% | 6.10352e-05 | 0.01 | 99.450% | 0.000 percentage points |
| Tuned 5% candidate | 5% | 6.10352e-05 | 0.0316228 | 67.050% | 0.541 percentage points |
| Tuned 30% candidate | 30% | 0.000976562 | 0.0316228 | 99.567% | 0.029 percentage points |
| 30% inherited default | 30% | 1e-06 | scale | 99.483% | 0.029 percentage points |
| Original 5% control | 5% | 1e-06 | scale | 67.450% | 0.507 percentage points |

Earlier records call the explicit 5% provisional setting `inherited_default`.
That historical label is preserved. The corrected package default and the new
control use 30%; the larger-sample run records both settings separately.

## Practical use and limits

The tuned 5% candidate scored 66.60%, 66.90%, and 67.65% on the larger sample,
while the 25% initial-grid candidate scored 99.45% for all three seeds.
Consequently, 5% is no longer in the recommended grids. Explicit fractions
remain supported by the estimator; this development result does not establish
that 5% must perform poorly on every dataset or training sample size.

`mnist_param_grid()` and `mnist_param_grid(broad=True)` now provide 125 and 405
combinations. For a focused follow-up on similarly scaled 7-versus-8 data,
the observed region suggests the following explicit grid. It is a proposed
84-candidate follow-up, not an additional completed experiment:

```python
focused_grid = {
    "kernel": ["rbf"],
    "lambd": [2.0**e for e in (-14, -10, -8)],
    "gamma": [10.0**e for e in (-2, -1.75, -1.5, -1.25)],
    "retention_fraction": [0.1, 0.2, 0.25, 0.3, 0.5, 0.75, 1.0],
}
```

Use that dictionary with `make_search` on training data. The adaptive example
expands lambda/gamma and refines their gaps; retention choices in that runner
are the five recommended values. Larger-retention comparisons must be specified
explicitly, as they were in the recorded follow-up here.

These adaptively selected CV scores are development results, not independent
test performance. The larger sample overlaps the smaller one. Repeating SGD
seeds on fixed data measures optimization variability, not independent data
replications. Equal epochs also give larger retained sets more kernel updates:
retention and computational effort are intertwined. An equal-update or
equal-time comparison is a separate experiment. No convergence, unique optimum,
universal default, speed advantage, or committee-level replicated comparison
is established here. The official test partition remains untouched.

The [machine-readable evidence](mnist-exploration.json) includes input hashes,
stage counts, all seed summary values, and hashes of the saved run records.
The local run archive preserves folds, selected-ID hashes, full fold metrics,
configuration, source snapshots, and resource receipts. Runs used at most six
assigned physical cores and an aggregate 32 GiB memory cap. Timings under this
shared-machine allocation are operational records, not runtime benchmarks.
