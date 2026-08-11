# 12.9 Issue 56 complete repository test CI design

**Status:** Proposed for written-spec review
**Date:** 2026-08-11
**Decision:** Add a separate, unconditional pull-request job that runs the repository's exact
`make test` contract while retaining the existing focused NNx, verifier, documentation, and
notebook-execution jobs.

## 12.9.1 Purpose

Issue #56 closes a CI coverage gap: local contributors can run the complete repository-owned
pytest suite with `make test`, but pull requests currently exercise only selected portions through
specialized jobs. A regression outside those selections can therefore merge even though the
canonical local test command would reject it.

The change makes the complete pytest contract an explicit pull-request gate without weakening the
faster diagnostics already provided by the NNx surface job, repository verifier, documentation
checks, and notebook execution tiers. It is a CI and documentation change only. It does not start
Atlas, initialize its submodule, execute notebooks through JupyterHub, or launch Ollama or ComfyUI.

## 12.9.2 Current evidence and constraints

The issue's historical test count is not a durable contract. The suite has already grown since the
issue was written, and platform-specific optional backends produce different justified skips on
macOS and Linux. CI must therefore lock the command and collection boundary rather than a mutable
number of collected tests.

The repository already defines that boundary in two places:

- `make test` invokes `pytest tests/ -v`.
- `pyproject.toml` sets `testpaths` to `tests` and excludes `infra`, `notebooks/archive`, and
  `.venv` from recursive collection.

The complete suite includes documentation-diagram tests. Those tests use CairoSVG and otherwise
skip when its native or Python dependencies are unavailable. A job that installs only runtime
requirements would therefore appear green without exercising the complete contract. The CI image
must install `libcairo2` and the locked documentation requirements in addition to the pinned Torch
stack and normal runtime requirements.

Atlas remains a pinned infrastructure input, not part of pytest collection. Repository-owned
Atlas integration tests inspect fixtures, scripts, and Git metadata; they do not require a live
service or a recursive submodule checkout. The new job must not initialize submodules.

## 12.9.3 Considered approaches

### 12.9.3.1 Replace the focused NNx job with the complete suite

This avoids a second dependency installation, but it removes the fast, clearly named NNx and PyPI
compatibility signal. It also combines two distinct contracts: broad repository correctness and
focused facade/package diagnostics. This approach is rejected.

### 12.9.3.2 Split the uncovered tests across another selective job

A helper-and-documentation job could be combined conceptually with the existing NNx job to cover
the current suite. No individual job would run `make test`, however, and the union could silently
drift whenever tests move or are added. This approach is rejected.

### 12.9.3.3 Add an independent complete-suite job

This is the selected approach. A dedicated `pytest-repository` job runs the canonical command on
every pull request and relevant branch push. Intentional overlap with focused jobs is acceptable:
the complete job is the merge gate, while the focused jobs retain faster failure localization and
their domain-specific setup.

## 12.9.4 Workflow contract

Add `pytest-repository` to `.github/workflows/ci.yml` with the following stable contract:

- run unconditionally whenever the workflow itself runs;
- use `ubuntu-24.04`, Python 3.11, and a 15-minute job timeout;
- check out the repository with persisted credentials disabled and without submodules;
- install the native `libcairo2` dependency;
- use the setup-python pip download cache with dependency paths
  `requirements.txt`, `torch-core-requirements.txt`, `torch-requirements.txt`, and
  `docs-requirements.txt`;
- run `make install-torch-stack`, then install `requirements.txt` and
  `docs-requirements.txt`;
- run the exact command `make test`; and
- use normal failure semantics, with neither `continue-on-error` nor a shell construct that masks
  a nonzero exit.

The 15-minute timeout is deliberately above observed installation and test durations while still
bounding a hung test. The setup-python cache accelerates package downloads; it does not replace
dependency installation, and the documentation must not claim otherwise.

No NLP asset download, notebook execution, service startup, Atlas checkout, Docker action, or model
runtime belongs in this job. The existing notebook tiers remain the authority for executable
notebook reproducibility.

## 12.9.5 Existing job responsibilities

The implementation preserves the current role split:

| CI surface | Responsibility |
| --- | --- |
| `pytest-repository` | Complete repository-owned pytest contract through `make test`. |
| Focused NNx surface job | Fast NNx facade, package-compatibility, quantization, and Ruff diagnostics. |
| Repository verifier | Structural, dependency, documentation, fast notebook, and policy contracts. |
| Documentation workflows | Deterministic three-surface generation and strict publication checks. |
| Notebook execution tiers | Papermill-backed notebook execution and reproducibility evidence. |

The complete-suite job supplements these signals; it does not subsume or rename them.

## 12.9.6 Skip and failure policy

Test failures and collection errors must produce a failing job. A skip is acceptable only when the
test declares a specific platform, dependency, or backend reason. The design does not introduce a
blanket pytest exclusion, an allowed-failure path, or a hard-coded expected test count.

The live Linux run is expected to differ from a local macOS/ARM run. Linux installs the pinned PyG
extension stack, so sampler tests that can be skipped locally should execute there. Conversely,
backend smoke tests whose APIs require a newer Torch release may skip under the repository's pinned
Torch version, provided their skip reasons remain explicit and the surrounding facade and guard
tests execute. The pull-request evidence records the actual collected, passed, and skipped results
instead of turning those values into permanent prose.

## 12.9.7 Regression protection

Tests in `tests/test_verify_repo.py` will encode the workflow contract before production YAML is
changed. They must prove that:

1. the complete-suite job exists and is not conditionally disabled;
2. runner, Python version, timeout, checkout behavior, Cairo installation, cache inputs, dependency
   installation, and the exact `make test` command match this design;
3. the job has no submodule checkout, test selection, failure masking, or allowed-failure setting;
4. `pyproject.toml` retains the canonical `tests` collection root and required exclusions; and
5. the Makefile retains `pytest tests/ -v` as the `test` target.

The new workflow-contract test name must also be included in the verifier job's narrow CI-contract
selection. This self-reference prevents removal of the complete-suite job from simultaneously
removing the only CI check that would detect its removal.

Tests follow red-green-refactor discipline: first demonstrate focused failures against the current
workflow, then make the smallest production change that satisfies them. No test may assert a
fixed total test count.

## 12.9.8 Documentation and diagram changes

The canonical documentation will describe the complete job and the preserved separation of
responsibilities:

- `CONTRIBUTING.md` will state that `make test` runs on every pull request and distinguish it from
  the focused NNx and notebook jobs.
- `docs/conventions.md` will document the timeout, cache manifests, dependency setup, collection
  boundary, explicit-skip policy, and pre-PR command.
- `docs/architecture.md` and the system diagram will show complete pytest as a separate CI signal.
- the OM-068 maintenance finding will move from Deferred to resolved with current evidence.
- `CHANGELOG.md` will record the durable workflow change.

The system diagram's HTML/SVG master is edited first and its committed PNG is regenerated and
visually inspected. Generated site and wiki trees remain derived outputs and are never hand-edited.
All canonical documentation changes must pass deterministic three-surface generation, strict
MkDocs, wiki parity, and repository self-containment checks.

## 12.9.9 GitHub enforcement and rollout

Adding an unconditional workflow job makes the check run, but it does not by itself make GitHub
block a merge when the check is absent. The active `gitflow` repository ruleset currently enforces
pull requests, deletion protection, and non-fast-forward protection without a required-status-check
rule.

The feature pull request establishes the stable `pytest-repository` check context. After that check
has run successfully, update the existing ruleset in place to require that context while preserving
every existing rule and branch target. Do not replace the ruleset or weaken another protection.
Verify the updated ruleset through the GitHub API and prove the required check is green on both the
feature-to-`develop` and `develop`-to-`main` pull requests.

The rollout follows repository Gitflow: feature branch to `develop`, `develop` to `main`, then a
content-neutral `main` to `develop` synchronization if the merge topology requires it. Issue and
project records receive links to the live runs and observed skip evidence before the branches and
PRs are cleaned up.

## 12.9.10 Verification and acceptance criteria

The implementation is complete only when all of the following are true:

1. Focused workflow tests fail for the missing job before implementation and pass afterward.
2. `make test` passes locally, with every skip carrying an explicit reason.
3. The repository verifier, Ruff, documentation checks, wiki generation, strict MkDocs build, and
   diagram regression tests pass.
4. Generated documentation is deterministic and the tracked worktree is clean apart from intended
   source and committed diagram changes.
5. The live feature and release pull requests show a successful, non-allowed-failure
   `pytest-repository` check within its timeout.
6. The existing NNx, verifier, documentation, and notebook jobs retain their established roles and
   remain green.
7. The active GitHub ruleset requires the stable `pytest-repository` context without losing any
   existing protection.
8. No Atlas, JupyterHub, Ollama, ComfyUI, or unrelated container is started or modified.
9. The change merges through feature-to-`develop` and `develop`-to-`main` pull requests, the two
   long-lived branches finish with identical content, and temporary local and remote branch state
   is removed.
