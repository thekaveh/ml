# 12.11 Issue 57 Atlas consumer policy CI design

**Status:** Approved
**Date:** 2026-08-11
**Decision:** Add a separate, unconditional, required `atlas-consumer-policy` CI job for the
parent-owned Atlas wrappers and host-service policy while retaining the path-scoped
`atlas-contract` workflow for direct validation of the pinned infrastructure submodule.

## 12.11.1 Purpose

Issue #57 closes a gap between the Atlas infrastructure contract and this repository's own
consumer behavior. The existing `atlas-contract` workflow validates the pinned `infra/` submodule,
consumer manifest, and compose overlay without starting services, but it does not run the
repository-owned tests for the lifecycle wrappers, runtime probe, dotenv helper, port and
preflight checks, shutdown behavior, or host-service admission policy.

The change creates a fast, attributable pull-request gate for those parent-owned contracts. It
does not replace the complete `pytest-repository` job or the direct Atlas submodule validation.
No CI command may start Atlas, JupyterHub, Ollama, ComfyUI, Docker Compose, or any other service.
Containerized Ollama remains categorically prohibited for this consumer.

## 12.11.2 Current evidence and constraints

The repository already has substantial non-live coverage:

- `tests/test_atlas_consumer_contract.py` verifies the committed manifest, pinned gitlink, and
  native-only LLM and image-service source policy.
- `tests/test_atlas_lifecycle.py` exercises parent wrappers with isolated fixtures and fake
  commands, including port validation, loopback readiness, environment isolation, preflight,
  normal and cold shutdown, and ComfyUI admission.
- `tests/test_atlas_runtime_probe.py` verifies the safe runtime-evidence schema, capability
  classification, notebook import discovery, and secret/path redaction without importing the
  real Atlas runtime stack.
- three Atlas-specific tests in `tests/test_makefile_contract.py` lock the public setup, start,
  stop, connect, validate, and explicit cold-shutdown commands.

The focused four-file baseline passes locally in seconds and does not require an initialized
submodule or a live service. The Python tests require only pytest and PyYAML; the runtime probe
uses the standard library and dynamically inspects optional packages rather than importing NLTK,
Torch, NNx, or PyG at module load time.

The existing `.github/workflows/atlas-contract.yml` pull-request filter omits
`scripts/atlas_runtime_probe.py`, `scripts/lib/atlas-dotenv.sh`, and the parent Atlas test files.
Its single job performs direct non-live Atlas validation after recursive submodule checkout. That
job must remain separate because parent-policy tests and upstream-infrastructure validation have
different dependency, checkout, failure, and rollback boundaries.

## 12.11.3 Considered approaches

### 12.11.3.1 Extend only the path-scoped Atlas workflow

The focused tests and ShellCheck could run after the existing submodule validation. This is a
small YAML change, but the workflow cannot safely become a required GitHub status check: a pull
request outside its path filter would never emit the required context and could remain blocked.
It also unnecessarily couples parent unit tests to recursive submodule checkout. This approach is
rejected.

### 12.11.3.2 Rely only on complete repository pytest

The required `pytest-repository` job already collects these tests. Relying on that broad job would
avoid another dependency installation, but Atlas policy failures would remain buried in the
complete suite, ShellCheck would not be part of the focused signal, and the accepted rollback
boundary would not exist. This approach is rejected.

### 12.11.3.3 Add an unconditional parent-policy job and retain direct Atlas validation

This is the selected approach. The main CI workflow gains a small `atlas-consumer-policy` job that
runs on every pull request and protected-branch push, so its check context can be required safely.
The existing `atlas-contract` workflow retains its path filter and direct, recursive-submodule
validation; its filter expands to cover every input capable of changing the consumer contract.
Intentional overlap with complete pytest is acceptable because the focused job provides early,
domain-specific diagnostics and an independently removable rollback surface.

## 12.11.4 Focused test boundary

Add a public `make test-atlas-consumer` target whose pytest selection consists only of:

- `tests/test_atlas_consumer_contract.py`;
- `tests/test_atlas_lifecycle.py`;
- `tests/test_atlas_runtime_probe.py`; and
- `tests/test_atlas_makefile_contract.py`.

Move the three existing Atlas-specific Makefile tests from the general
`tests/test_makefile_contract.py` module into `tests/test_atlas_makefile_contract.py`. This is a
behavior-neutral test organization change that gives the focused target a complete file-based
boundary without brittle node IDs or unrelated Makefile tests. The normal `make test` collection
continues to include all four modules.

The target is the only pytest command the focused workflow may call. Its exact file inventory and
normal failure semantics are regression-tested; it must not use `-k`, blanket exclusions,
allowed-failure settings, or a fixed expected test count.

## 12.11.5 Minimal dependency contract

Add `atlas-contract-requirements.txt` containing exact direct pins for pytest and PyYAML. The
focused job installs only that file and uses it as the setup-python pip-cache dependency path. It
must not install the Torch stack, NNx, notebook dependencies, documentation toolchain, or NLP
assets.

The pinned pytest and PyYAML versions must match the corresponding exact versions in the locked
documentation requirements. A repository regression test compares the manifests so a dependency
bump cannot silently leave the focused job on a divergent test/parser version. The new file is
also an Atlas workflow path-filter input.

## 12.11.6 Required parent-policy job

Add `atlas-consumer-policy` to `.github/workflows/ci.yml` with this stable contract:

- run unconditionally whenever the CI workflow runs;
- use `ubuntu-24.04`, Python 3.11, and a 15-minute timeout;
- check out with persisted credentials disabled and without submodules;
- configure the setup-python pip cache only against `atlas-contract-requirements.txt`;
- install exactly `atlas-contract-requirements.txt`;
- run ShellCheck against `scripts/atlas-up.sh`, `scripts/atlas-down.sh`,
  `scripts/atlas-connect.sh`, and `scripts/lib/atlas-dotenv.sh`;
- run exact `make test-atlas-consumer`; and
- use ordinary failure semantics with no job or step condition, service, container, environment
  override, `continue-on-error`, or masking shell construct.

The job must not initialize `infra/`. It must not contain `docker`, `docker compose`, `ollama
serve`, `make atlas-up`, `make atlas-down`, `curl`, or any localhost runtime probe. Tests may use
temporary fake executables and fixture-local files to prove the wrapper behavior, but the workflow
itself never contacts or mutates a runtime.

## 12.11.7 Path-scoped direct Atlas workflow

Retain `.github/workflows/atlas-contract.yml` as the direct submodule-consumer validator and
expand its pull-request paths to include every parent input that can change the contract:

- the existing gitlink, manifest, example environment, overlay, notebook specifications, parser,
  verifier, documentation, Makefile, and workflow files;
- `atlas-contract-requirements.txt`;
- `scripts/atlas_runtime_probe.py`;
- `scripts/lib/atlas-dotenv.sh`;
- all `scripts/atlas-*.sh` wrappers;
- `tests/test_atlas_*.py`; and
- `tests/test_makefile_contract.py` during the test-boundary move.

The direct job keeps recursive submodule checkout, the pinned Atlas runner, environment backfill,
compose validation, doctor report, and dirty-infra gate. It does not start services and does not
become a required ruleset context. Its path filter exists for attributable infrastructure
validation, while the unconditional parent-policy job supplies enforceable host-policy coverage.

## 12.11.8 Host-service policy assertions

The focused suite must continue to prove all of the following without live services:

1. `LLM_PROVIDER_SOURCE=ollama-localhost` is the only admitted LLM source.
2. automatic and containerized Ollama sources are rejected, including CPU and GPU container
   variants.
3. the wrapper clears ambient source overrides before resolving the committed policy.
4. the native Ollama port is numeric, within 1–65535, and checked only on loopback.
5. normal start performs the host-native readiness preflight; validation mode remains non-live.
6. ComfyUI is admitted only as disabled, localhost, or explicitly managed localhost/MPS, while
   automatic and containerized variants are rejected.
7. ordinary shutdown preserves volumes, cold shutdown is explicit, and public Make targets pass
   the correct arguments.
8. runtime-probe output cannot leak credentials, environment values, host paths, or notebook
   paths.

Tests must fail if the consumer begins admitting containerized Ollama or ComfyUI. A future service
policy change therefore requires an explicit contract, test, and documentation change rather than
an ambient environment override.

## 12.11.9 Regression protection and TDD

Production workflow, Makefile, dependency, or documentation edits follow red-green-refactor
discipline. Focused regressions in `tests/test_verify_repo.py` and the Makefile-contract tests must
first fail against the current repository, then prove:

1. the new job has the exact runner, timeout, checkout, cache, installation, ShellCheck, and Make
   target contract;
2. job-level or workflow-level controls cannot skip, mask, containerize, or redirect the gate;
3. the Atlas workflow path inventory includes every wrapper, probe, helper, test, and dependency
   input;
4. the focused Make target contains the four exact test modules and remains part of `.PHONY` and
   help output;
5. direct dependency pins match their canonical locked versions; and
6. the verifier job's narrow workflow-contract selector runs the new self-referential tests.

Mutation-style tests must demonstrate that adding a container, service, conditional, live command,
failure mask, omitted path, or reduced test inventory makes the contract test fail.

## 12.11.10 Documentation and diagram changes

Canonical documentation will distinguish the two Atlas CI boundaries:

- `CONTRIBUTING.md` will add the focused local command and identify the required policy job.
- `docs/conventions.md` will describe the unconditional parent-policy gate, dependency boundary,
  ShellCheck surface, non-live restriction, and its relationship to complete pytest.
- `docs/jupyterhub-integration.md` will explain which committed consumer inputs trigger direct
  Atlas validation and which host-policy behavior is enforced on every pull request.
- `docs/architecture.md` and the system-diagram master will show the required Atlas policy signal
  beside complete pytest, NNx, verifier, documentation, and notebook gates.
- `CHANGELOG.md` will record the durable CI and policy change.

The system diagram's HTML/SVG master is edited before its committed PNG is regenerated and visually
inspected. Generated site and wiki trees remain derived, ignored output. The design and
implementation plan are manifest-declared canonical pages and must project cleanly to all three
documentation surfaces.

## 12.11.11 GitHub enforcement and rollout

The feature pull request establishes the stable `atlas-consumer-policy` check context. After its
first successful live run, update the existing `gitflow` ruleset in place so both
`pytest-repository` and `atlas-consumer-policy` are required. Preserve all existing rules, bypass
settings, targets, and enforcement state. Verify the ruleset through the GitHub API and prove both
contexts green on the feature-to-`develop` and `develop`-to-`main` pull requests.

The path-filtered `atlas-contract` check must also run and pass on both PRs because the change
touches its own workflow and Atlas inputs, but it remains non-required to avoid the missing-context
problem on unrelated changes.

Rollout follows repository GitFlow: feature to `develop`, `develop` to `main`, then a content-neutral
`main` to `develop` synchronization PR when merge topology requires it. Issue and project records
receive the focused local evidence, live check URLs, ruleset verification, and explicit no-service
statement before temporary PR and branch state is removed.

## 12.11.12 Verification and acceptance criteria

The implementation is complete only when:

1. workflow, dependency, Makefile, policy, and path-filter regressions demonstrate red before
   implementation and green afterward;
2. `make test-atlas-consumer`, direct ShellCheck, complete `make test`, Ruff, repository
   verification, documentation checks, wiki generation, strict MkDocs, and diagram tests pass;
3. the live feature and release PRs show successful `atlas-consumer-policy`, `atlas-contract`, and
   complete repository checks;
4. GitHub requires `atlas-consumer-policy` alongside `pytest-repository` without weakening the
   existing `gitflow` ruleset;
5. canonical repository, site, and wiki documentation remain deterministic and synchronized;
6. no Atlas, JupyterHub, Ollama, ComfyUI, Docker Compose, or unrelated container is started,
   stopped, or modified;
7. issue #57 and its project item contain final evidence and are closed only after the release
   reaches `main`; and
8. `develop` and `main` finish content-equivalent, with no dangling feature branch, worktree, or
   pull request locally or remotely.
