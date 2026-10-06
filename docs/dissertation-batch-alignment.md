# Batch sampling and dissertation alignment

The October 4, 2026 dissertation explicitly permits single-observation and
averaged batch updates in Sections 3.1 and 3.2 (PDF pages 31 and 33-34),
including full-gradient updates. Its Chapter 5 study uses
`batch_size=1, sampling="with_replacement"`, matching the package defaults:
sample one fitting index uniformly and independently at each update.
The linear and kernel objectives, free intercepts, and two-stage selection
rule remain unchanged.
The package's retention default is 30%, matching the ensemble package; the
original guide's 5% is an explicit alternative outside the recommended grid.
Reports must state the retention used independently of the sampling policy.
Minibatches and full-gradient updates are optional package settings already
covered by the methodology. Shuffled sampling is a separate optional policy
that changes the independent-sampling assumption and was not used in the
reported Chapter 5 study. This note records the update equations and the
reporting and proof distinctions; it does not edit the dissertation or
establish a new theoretical result.

## Regularization convention

From version 0.4.0, the sole public regularization argument is `lambd`, the
same convention as `dwd`: mean DWD loss + lambd * ||h||_H^2. Python reserves
the word `lambda`, so the API uses `lambd`. Version 0.5.0 adopts the MNIST
starting default `lambd=1e-6` from the ensemble DWD package.
Section 3.2 of the dissertation and the original handoff guide write the
squared-norm term as lambda_dissertation/2 * ||h||_H^2. Chapter 5 explicitly
states the conversion lambda_dissertation = 2 * lambda_pkg. For the same
objective, use

```text
lambda_new = lambda_dissertation / 2
lambd = lambda_new
```

This is a coefficient conversion, not a change of model. Version 0.4.0 halved
its default and search-grid values to preserve their previous effective
strengths. Version 0.5.0 makes a separate, intentional change to the MNIST
starting defaults and ranges, already expressed in the `lambd` convention.
Keep this conversion consistent across the objective, gradient, shrinkage,
learning-rate bound, and reported settings, as in Chapter 5. Original
guide and dissertation files remain unchanged. The unit-ball-constrained
linear stage has no regularization parameter.

## API and sampling contract

`batch_size=1` remains the default. A positive integer requests that many
observations per update, subject to the remaining observations in the epoch.
`batch_size="full"` uses every fitting observation exactly once in one update.
For a stage with m fitting observations, an integer larger than m has effective
batch size m. The last batch is shortened when m is not divisible by the
effective batch size. Gradients are divided by that batch's actual length r.

`sampling="with_replacement"` is the default. Each batch consists of
independent uniform draws from all m fitting indices; repeated
indices are allowed. An epoch contains m draws, including repeats, and need
not visit every observation. Integer `batch_size=m` in this mode is a random
multiset, whereas `batch_size="full"` always evaluates every observation once.
The full-batch option determines coverage independently of `sampling`.

`sampling="shuffle"` is optional. At each epoch, generate an independent
uniform random permutation of the m fitting indices and split it into
nonoverlapping consecutive batches. Every observation occurs exactly once per
completed epoch. With `batch_size=1`, these are single-observation updates in
a newly shuffled order. This option differs from Chapter 5's sampling and
from the independent draws used in Chapter 3's unbiased-gradient statements,
even when the batch size is one.

`batch_size=1, sampling="with_replacement"` reproduces the former
implementation numerically after the coefficient conversion above, with
matching explicit hyperparameters and the same input, seed, numerical
environment, and iteration-based stopping settings.
Wall-clock caps need not stop at the
same update across executions. The optional `sampling="shuffle"` does
not reproduce the former trajectory merely by using the same seed.

The original portable handoff's `GUIDE.md`, lines 207-211, allowed a separate
minibatch option with averaged gradients and regularization once per update.
It identified batch size one as the primary method. This is historical guide
context; the October 4 methodology already includes batch updates. State the
actual batch and sampling settings in any subsequent experiment.

## Batch update equations

Let I_k = (i_1, ..., i_r) be the current batch, and compute every margin and
q_l = L'(y_i_l f_k(x_i_l)) y_i_l using the same old parameter state. The
normalized DWD loss and all objective normalizations stay unchanged.

For the linear stage:

```text
g_w = (1/r) sum_l q_l * x[i_l]
g_b = (1/r) sum_l q_l
w_next = projection_onto_unit_ball(w - eta_k * g_w)
b_next = b - eta_k * g_b
```

Project w once after the averaged update. Do not project or penalize b, and
do not average separately projected single-observation updates.

For the kernel stage, with h(x) = sum_j a[j] K(x[j], x):

```text
g_h = 2 * lambd * h + (1/r) sum_l q_l * K(x[i_l], .)
a_next[j] = (1 - 2 * eta_k * lambd) * a[j]
            - (eta_k/r) * sum_{l: i_l=j} q_l
b_next = b - (eta_k/r) * sum_l q_l
```

Repeated indices in replacement sampling contribute repeatedly to the sum.
Apply the kernel regularizer once per parameter update, without multiplying
the loss increment by the Gram matrix. Full batches and kernel computation
blocks use the same frozen old state; a computation block is not a separate
optimizer step. No full Gram matrix is mathematically required.

## Epochs, learning rate, and accounting

Chapter 3 permits a decaying learning rate of the implementer's choice. Let
s_k be the number of observations processed before update k, counting
repeated draws in replacement mode. The package uses

```text
eta_k = (1/B) * (1 + s_k/m)^(-3/4)
s_next = s_k + r
```

Here m is the current stage's actual fitting size. Preserve the existing
conservative bounds B = 4 max_i (||x_i||^2 + 1) for linear updates and
B = 2 * lambd + 4 * (max_i K(x_i,x_i) + 1) for a general PSD kernel, reducing
to 2 * lambd + 8 for the RBF kernel stage. They also bound the smoothness of
averaged batch losses; batch size does not require multiplying the step size.

An epoch is m processed observations: one complete permutation under shuffle,
m independent draws under replacement, or one full-gradient update in full
mode. With batch size one, s_k = k and this matches the schedule reported in
Chapter 5 and the original guide. For larger batches, the decay is indexed by
processed passes rather than the number of optimizer updates. Validation and
patience remain indexed by epochs, with the same checkpoint rules.

Report actual processed observations, parameter updates, epochs, batch size,
sampling mode, and stage fitting size separately. A completed epoch has
ceil(m/effective_batch_size) parameter updates, or one in full mode. Equal
epoch counts across batch sizes do not imply equal trajectories, optimization
progress, runtime, or convergence. Full mode is gradient descent on the same
objective. Historical single-observation operation counts need corresponding
batch-size and kernel-evaluation accounting before being reused.

## Reporting batch variants and theoretical claims

Sections 3.1 and 3.2 of the October 4 dissertation already give the averaged
batch updates, with single-index updates recovered at r = 1. The older
`reference/methodology-sections.txt` excerpt, including its sampling paragraphs
at lines 33-37 and 83-87, is historical and does not describe the full current
methodology.
When comparing coefficient values or writing the update equations in package
notation, apply the regularization conversion above.

If replacement minibatches are used, describe their independent draws and use
the averaged batch equations already present in the methodology and reproduced
above. Report actual batch lengths, processed observations, parameter
updates, the samples-processed learning-rate index, and epoch/checkpoint
definitions. Update any complexity or finite-iteration statement whose counts
depend on single-observation updates. These reporting changes are unnecessary
merely because the library exposes an unused minibatch option.

If shuffled sampling is used, revise the relevant sampling description to
state fresh permutations and nonoverlapping batches. Any theoretical claim
applied to those runs must then address reshuffling. If full mode is used,
identify it as gradient descent on the same objective, with one full-gradient
update per epoch, rather than single-observation SGD.

Under independent replacement sampling, conditional on the training history,
the expected batch gradient equals the full average-loss gradient at the
current iterate; the kernel regularizer is applied once. Thus the default retains
the conditional-unbiasedness property stated in Chapter 3. Under the
optional random-reshuffling scheme,
later batches are drawn from the remaining observations, and the current
iterate depends on observations already processed. Their gradients are not
generally conditionally unbiased for the full-data gradient at that iterate.
A permutation position being marginally uniform does not supply the missing
conditional-unbiasedness property. An exact full gradient needs no stochastic
unbiasedness argument.

For optional variants, any theorem or proof using independent samples,
conditional unbiasedness, martingale differences, or the original
single-update iteration count must retain its stated replacement-sampling
algorithm or be justified separately for the algorithm actually used. The
batch-average gradient's unbiasedness under replacement sampling alone does
not establish every bound for that extension. Likewise, an unchanged objective
does not establish finite-iterate equivalence or transfer a generalization
bound to reshuffling. The default requires no new sampling assumptions; this
note does not certify the dissertation's other proof steps.

Keep the normalized loss, linear unit-ball constraint, unregularized
intercepts, RKHS penalty, retained-set averaging, and selection rule intact.
Identify subsequent experiments by their actual batch and sampling settings;
do not relabel earlier verification or historical accuracy comments as results
for an optional variant. No experiments or proof revisions are completed by this
alignment note.
