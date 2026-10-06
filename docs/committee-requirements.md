# Committee requirements and package scope

This note maps the committee's comments to package support and the remaining
experimental and dissertation work.

Status as of October 6, 2026: the six-classifier MNIST comparison and its
fixed-30% follow-up are complete. The follow-up includes 180 fresh final fits:
six classifiers, three digit pairs (2–3, 4–9 and 3–8), and ten paired training
draws per pair. Accuracy and timing summaries retain sample standard deviations
and standard errors separately. Completion of this comparison does not establish
committee approval, dissertation acceptance, or resolution of every concern.

The ten-repetition requirement applies to the numerical study. One call to
`fit()` trains one model, including both stages for `TwoStageDWDClassifier`;
the experiment driver repeats those calls.

The fixed-30% follow-up reused completed CV evidence and the ten saved training
draws; it did not repeat tuning. It was selected after the original evaluation
and reused the official test sets. Its uncertainty summaries are conditional on
the fixed calibration and those test sets, not estimates from ten independent
tuning exercises or a new untouched test set. The historical SGD implementation
was `0.7.0.dev20261003`; later package benchmarks have their own implementation
and timing records, described in [VALIDATION.md](../VALIDATION.md).

The separately archived study evidence is in
`mnist-six-classifiers-fixed-30pct-portable-20261004.zip`, including the
fixed-retention records under `current-run/`: `selection-lock.json`,
`preparation.json`, final-fit records, and `summary/summary.json`,
`summary/validation.json` and `summary/RUN_RECORD.txt`. Earlier CV records and
initial grids remain under `previous-study/cv/`. These study artifacts are
separate from the installed package.

Source: WangComments-compressed.pdf, a scanned 13-page review supplied by the
author, with no extractable text. Its chapter and equation numbering refer to
the earlier dissertation. The author's October 1 code guide proposed an
experiment addressing relevant comments; its original settings are historical
and differ from the subsequently recorded six-classifier protocol.

| Concern | Package response | Remaining work |
|---|---|---|
| Section 4.8, PDF p.6: three rounded 99.50% scores on one pair support no comparison. | The recorded comparison covers three pairs, six classifiers, separate development-reserve tuning and ten paired training draws. Original scripts remain historical material. | Interpret the observed results with the recorded tuning, test-exposure and task-dependence limits; no universal performance claim follows. |
| R6, PDF p.12: at least ten replications and standard errors for accuracy tables. | All 18 method/pair groups have ten eligible final results. Summaries report sample SD and SE = SD/sqrt(10), conditional on fixed calibration and reused tests. | Preserve that uncertainty scope; the repetitions do not measure repeated-tuning variability. Ensemble M and theta requests apply to Chapter 5, not this classifier. |
| R4, PDF p.12; PDF p.2: generalization bounds do not establish consistency. | Preserve the practical free-intercept DWD model fitted with SGD and document that code tests do not validate a proof. | Theoretical claims and proof revisions remain in the dissertation project. |
| R13, PDF p.12: remove exaggerated claims. | No screening superiority, consistency, universal speedup, or solver-convergence claims. | Apply restrained language when interpreting the recorded experimental findings. |
| R18, PDF p.13 and p.9: report tuning costs or identify post-tuning fits. | The study preserves CV work records and reports fresh post-tuning fit and prediction times separately. | Keep tuning cost, post-tuning measurements and later implementation benchmarks distinct; do not add reused CV costs to a fresh final fit. |
| R19 / Section 4.8, PDF pp.6,13: explain computational limitations and shorten Chapter 3. | For two-stage SGD DWD, on-demand kernels and the study's 64 MiB fit-local cache avoid a persistent full Gram matrix. Runtime and process-lifetime peak-memory records are saved. | Process-lifetime peaks are not incremental fit allocation. Chapter compression and interpretation remain dissertation work. |
| R21, PDF pp.10,13: code, seeds, versions, hardware/parallelism. | The study archive preserves source/runtime identities, data hashes, seeds, selected settings and execution controls alongside individual results. | Complete hardware/power details absent from the original receipts; keep those records with the tables and identify later implementation comparisons separately. |

The committee does not prescribe an SOCP dependency, different DWD optimizer,
or altered intercept treatment. Fidelity to the supplied method comes first.
The global 5% rule, three digit pairs, random-subset control, proposed learning
rate, and 24-hour study design are the author's guide choices, not verbatim
committee requirements. R1/R2 ensemble benchmarks, R5 SVM selection, R7-R11
and R14-R16 NMF work, R17 ensemble kernel approximations, editorial tasks, and
figure permissions are outside this package implementation.

The MNIST comparison does not supply a dedicated replicated HDLSS study with
feature count much larger than training size, or the guide's random-subset
control. Small and HDLSS software fixtures in [VALIDATION.md](../VALIDATION.md)
check implementation behavior; they are not substitutes for that statistical
study. Theoretical and other chapter-specific concerns remain separate work.

The statement on PDF p.6 about practical Gram-matrix cost is not a mathematical
requirement to store an N by N matrix. Report the actual on-demand/cache policy
and measured costs; do not relabel historical results with newer implementations.
