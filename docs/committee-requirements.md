# Committee requirements and package scope

Source: WangComments-compressed.pdf supplied by the author, 13 pages. Its chapter
and equation numbering refer to the earlier dissertation. The review was read
visually because this scanned PDF has no extractable text. The October 1 code
guide translates the relevant comments into a proposed new experiment.

| Concern | Package response | Remaining work |
|---|---|---|
| Section 4.8, PDF p.6: three rounded 99.50% scores on one pair support no comparison. | Preserve old scripts as unverified history; make repeatable new fitting possible; make no performance claim. | Pilot, independent tuning, paired repetitions, controls, and honest interpretation. |
| R6, PDF p.12: at least ten replications and standard errors for accuracy tables. | Independent seeds, actual selected IDs, stored stage models and diagnostics support repeated evaluations. | Ten paired repetitions and uncertainty calculations; none have been run here. Ensemble M and theta requests apply to Chapter 5, not this classifier. |
| R4, PDF p.12; PDF p.2: generalization bounds do not establish consistency. | Preserve the practical free-intercept SGD model and document that code tests do not validate a proof. | Theoretical claims and proof revisions remain in the dissertation project. |
| R13, PDF p.12: remove exaggerated claims. | No screening superiority, consistency, universal speedup, or solver-convergence claims. | Apply restrained language to actual experimental findings when available. |
| R18, PDF p.13 and p.9: report tuning costs or identify post-tuning fits. | Expose standalone fit and stage diagnostics; expensive norm diagnostics remain explicit calls. | Experiment layer must measure calibration and full pipeline time separately. |
| R19 / Section 4.8, PDF pp.6,13: explain computational limitations and shorten Chapter 3. | On-demand kernel evaluation without a persistent full Gram matrix; no claim that RKHS SGD mathematically requires one. | Measure actual runtime/peak memory; decide chapter compression and interpretation in dissertation work. |
| R21, PDF pp.10,13: code, seeds, versions, hardware/parallelism. | Versioned package source, reproducible seeds, hashes for IDX sources, tests, and an environment receipt. | Publish the repository when requested; record benchmark hardware/power/thread settings in the eventual study. |

The committee does not prescribe an SOCP dependency, different DWD optimizer,
or altered intercept treatment. Fidelity to the supplied method comes first.
The global 5% rule, three digit pairs, random-subset control, proposed learning
rate, and 24-hour study design are the author's guide choices, not verbatim
committee requirements. R1/R2 ensemble benchmarks, R5 SVM selection, R7-R11
and R14-R16 NMF work, R17 ensemble kernel approximations, editorial tasks, and
figure permissions are outside this package implementation.

The statement on PDF p.6 about practical Gram-matrix cost is a concern to test.
It is not a mathematical requirement to store an N by N matrix. The package
uses on-demand kernels consistently; future results must report that policy.

Current status: implementation, verification, and preliminary exploration using
training data only. A full replicated comparison remains pending. Empirical
review concerns remain open until the agreed experiments and reports are
completed. A working package does not establish committee approval or
dissertation acceptance.
