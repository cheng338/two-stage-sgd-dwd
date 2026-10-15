# Release policy

Version numbers describe grouped releases, not the number of fixes, commits,
validation rounds or builds. The current published release is **0.9.0**;
the next planned maintenance release is **0.9.1**.

This policy follows the release practice of Chang Cheng's
[ensemble-dwd](https://github.com/cheng338/ensemble-dwd/releases) and
[supervised-nmf](https://github.com/cheng338/supervised-nmf/releases).
Ensemble DWD grouped fixes, performance work and optional diagnostics or
workflow controls into 0.3.1 through 0.3.8. Supervised NMF grouped substantial
projection and classification capabilities into 0.2.0. These are examples of
release scope, not a requirement to keep the packages' numbers synchronized.

## Choosing a version

| Change scope | Version choice |
| --- | --- |
| Fixes, numerical corrections, performance or memory improvements, packaging, dependencies, documentation, tests, or optional diagnostics and workflow controls that preserve the estimator contract | Patch, such as 0.9.0 to 0.9.1 |
| A substantial new estimator or modeling capability, or an incompatible change to the public estimator contract during 0.x development | Minor, such as 0.9.1 to 0.10.0 |
| An explicit decision that the API is ready for a stable compatibility commitment | 1.0.0 |
| An incompatible public API change after 1.0 | Major |

The size of a diff, number of changes and elapsed time do not by themselves
justify a minor release. Patch numbers can continue through 0.9.9, 0.9.10 and
beyond. A growing patch number does not automatically trigger a minor release;
0.9 does not automatically advance to 1.0.

A numerical correction may change rounded scores or labels without changing
the estimator contract. Release notes must describe the affected cases,
validation and any performance cost. Changes to the objective, sampling rule,
selection method or estimator defaults require an explicit scope assessment;
they are not ordinary implementation optimizations.

For planned 0.9.1, the author requested downward rounding of the retained
observation count to match the ensemble classifiers. This is an explicitly
approved count-convention change within the existing two-stage method and
parameter interface, included in the maintenance release. It can change new
fits when `retention_fraction * N` is noninteger, including an empty-selection
failure for a count below one; it is not a result-preserving optimization.
The change does not require a separate minor release under this policy.
Published artifacts, saved models and original experiment records retain
their earlier convention.

Python and dependency support changes may accompany a maintenance release,
but their compatibility impact must be stated. In particular, planned 0.9.1
requires Python 3.11 or newer; Python 3.10 users retain the existing 0.9.0
release. Workflow checkpoint incompatibilities must also be disclosed even
when the estimator API is unchanged. This is a project release policy, not a
promise that every patch preserves every runtime or saved workflow format.

## Development and release preparation

1. Accumulate validated changes in `CHANGELOG.md` under `Unreleased`. Keep the
   package version and release citation at the last published version while
   developing; a commit or successful benchmark does not create a release.
2. Group related, ready changes into one release. Confirm the proposed patch
   or minor scope before preparing its artifacts.
3. During release preparation, update `src/two_stage_sgd_dwd/__init__.py`,
   `CITATION.cff`, the software entry in `docs/references.bib`, README citation,
   and changelog together. `pyproject.toml` reads the package version from
   `__init__.py`. Use the actual release date in release metadata.
4. Build native and portable wheels with the same distribution version.
   Repeated local builds do not require another version bump. Identify local
   validation artifacts by their source revision, uncommitted changes and
   hashes; they are not replacements for published assets of the same version.
5. Publish only when a release is requested. Preserve every published version,
   tag and artifact. Never renumber earlier releases to make the sequence
   smaller or replace their assets with newly changed code.

For the present unreleased work, use **0.9.1** when release preparation is
requested. Do not increment to 0.10.0 solely for the accumulated fixes,
profiling, search corrections or Python support change.
