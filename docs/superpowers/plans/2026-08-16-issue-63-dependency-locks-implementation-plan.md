# 12.24 Issue 63 Dependency Locks and Immutable Build Inputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or
> `superpowers:subagent-driven-development` and execute each task test-first. Do not begin a task
> until the prior task's commit and review boundary are clean.

**Goal:** Replace resolver-time dependency and mutable base-image inputs with reviewed,
hash-checked, platform-qualified locks and immutable image identities without changing the Issue
#62 Torch/PyG/NNx runtime contract.

**Architecture:** Human-authored dependency inputs plus one source-policy file compile into a
bootstrap/compiler/tool lock family and three cumulative platform runtime lock families. One shared
sanitized pip runner installs non-runtime locks; the existing four-stage installer consumes the
platform locks. An offline verifier proves repository coherence, while explicitly networked lock
and image checks prove registry truth. Advisory observations are derived from exact locks rather
than re-resolving source manifests.

**Tech stack:** Python 3.11, pip 26.2.1, setuptools 81.0.0, wheel 0.47.0, packaging 26.2,
uv 0.11.19, TOML, JSON, GNU Make, GitHub Actions, Docker Buildx, pip-audit 2.10.0, pytest, Ruff,
MkDocs, papermill, and the exact Issue #62 Torch/PyG/NNx matrix.

**Spec:** `docs/superpowers/specs/2026-08-16-issue-63-dependency-locks-design.md`

## 12.24.1 Global constraints

- Work only on `codex/issue-63-lock-dependencies`, based on the clean Issue #62 `develop` merge
  `36ad187`. Preserve design commit `1ca131a6dfe7269f61117eba6844a81eb2c2f763`.
- Keep the exact Issue #62 runtime versions and behavior: Torch 2.11.0, TorchVision 0.26.0,
  TorchAudio 2.11.0, Lightning 2.6.1, TorchMetrics 1.9.0, torchao 0.18.0,
  torch-geometric 2.8.0.post1, pyg-lib 0.8.0, torch-scatter 2.1.2,
  torch-sparse 0.6.18, and `thekaveh-nnx[lm]==0.2.0`.
- Preserve the Issue #62 warning-debt, QAT-debt, sampler, JUnit, smoke-output, advisory, and
  dual-SHA PR evidence contracts. Locking may change their installation inputs, never weaken their
  assertions, warning-as-error policy, or required job conclusions.
- Support exactly `darwin-arm64` / `aarch64-apple-darwin`, `linux-x86_64` /
  `x86_64-manylinux_2_28`, and `linux-aarch64` / `aarch64-manylinux_2_28`, with resolution floor
  Python 3.11.0 and consumer range `>=3.11.0,<3.12`.
- Pin GitHub Actions and final Darwin qualification to CPython 3.11.15. The pinned Jupyter Docker
  base must report CPython 3.11.10. Do not silently change the supported interpreter floor.
- `compiler-requirements.txt` is the sole uv version authority. `requirements/lock-policy.toml`
  stores its path and shape, never another uv version. The Atlas input's uv pin must equal it.
- The lock policy fixes the reviewed package-upload horizon at `2026-08-17T02:21:18Z`; every
  compiler command passes it through exactly one `--exclude-newer`. A cutoff advance is an
  intentional policy/lock update, never an ambient consequence of a newly published package.
- Never use `--extra-index-url`, unsafe index strategies, editable/VCS/local requirements, an
  unapproved direct URL, or an unnamed sdist. The sole direct URL is the reviewed
  `en_core_web_sm-3.8.0` wheel. The sole sdist is `python-louvain==0.16`, built with the locked
  toolchain and `--no-build-isolation`.
- Never claim an offline parser proves remote artifact bytes or registry state. Valid-but-wrong
  coordinated hashes/digests must pass syntax/coherence and fail a networked regeneration,
  hash-required clean install, or image-ledger check.
- Do not initialize or start Atlas, JupyterHub, Docker Compose, Ollama, ComfyUI, or a persistent
  repository service. Docker is limited to builds, `buildx imagetools inspect`, and `run --rm`
  probes.
- Keep the `infra` gitlink exact at `61c7c5103660e2226bf107c115dae42bf46f8374`; Issue #65 owns
  Atlas runtime changes. Issue #64 owns NLTK VADER payload integrity. Issue #66 owns the manual-only
  quantization notebook.
- Do not edit notebook code, outputs, execution counts, metadata, or tier inventories. Final
  notebook results go only to disposable output roots.
- Do not rewrite released changelog history, old advisory evidence, or prior maintenance records.
  Add current Issue #63 truth only to current sections.
- Generated site/wiki derivatives are never committed. Edit canonical sources, then prove exact
  source-to-site/wiki parity.
- A tracked change after final-SHA freeze invalidates qualification. Any lock, installer, Docker,
  advisory, CI, or notebook failure stops publication and requires a reviewed correction plus the
  complete fresh qualification at the new SHA.

## 12.24.2 Final file and interface map

### 12.24.2.1 Human-authored inputs

- `bootstrap-requirements.txt`: `pip==26.2.1`, `setuptools==81.0.0`, `wheel==0.47.0`,
  `packaging==26.2`.
- `compiler-requirements.txt`: exactly `uv==0.11.19`.
- `nlp-model-requirements.txt`: exact official spaCy model wheel URL plus SHA-256
  `1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85`.
- Existing root/Torch/docs/audit/Atlas manifests remain direct-input authorities. Remove the PyG
  `--find-links` directive from `torch-requirements.txt`; source policy owns it. Add the exact uv
  pin to `atlas-contract-requirements.txt`.
- `requirements/lock-policy.toml`: schema, compiler-input path, Python floor, platform triples,
  source bindings, input/output inventories, binary exceptions, and direct-URL exception.
- `requirements/image-lock.json`: schema plus exact source tag, index digest, and amd64/arm64 child
  digests for the Jupyter and devcontainer bases.

### 12.24.2.2 Generated outputs

```text
requirements/locks/bootstrap.txt
requirements/locks/compiler.txt
requirements/locks/audit.txt
requirements/locks/atlas-contract.txt
requirements/locks/darwin-arm64/{core,runtime,root}.txt
requirements/locks/linux-x86_64/{core,runtime,root}.txt
requirements/locks/linux-aarch64/{core,runtime,root}.txt
docs-requirements.txt
```

All are exact, SHA-256 hashed, normalized, and free of timestamps, absolute paths, credentials,
cache paths, or hostnames. The four non-platform locks and docs lock are universal across exactly
the three supported targets—not arbitrary operating systems/interpreters. Root locks include
runtime, root, docs, and the spaCy model closure.

### 12.24.2.3 Python interfaces

`scripts/dependency_locks.py` is the shared read-only model:

```python
@dataclass(frozen=True)
class PlatformPolicy:
    key: str
    uv_platform: str
    system: str
    machine: str
    torch_index: str
    pyg_find_links: str

@dataclass(frozen=True)
class TargetEnvironment:
    key: str
    marker_environment: Mapping[str, str]

@dataclass(frozen=True)
class LockPolicy:
    schema: int
    compiler_input: Path
    python_floor: Version
    macos_deployment_target: str
    platforms: tuple[PlatformPolicy, ...]
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]

@dataclass(frozen=True)
class LockedRequirement:
    name: str
    version: Version
    hashes: tuple[str, ...]
    source: str | None
    marker: str | None

def load_policy(repo: Path) -> LockPolicy: ...
def parse_compiler_identity(repo: Path) -> Version: ...
def parse_lock(path: Path) -> tuple[LockedRequirement, ...]: ...
def project_for_target(
    requirements: Sequence[LockedRequirement],
    target: TargetEnvironment,
) -> tuple[LockedRequirement, ...]: ...
def exact_version_projection(
    requirements: Sequence[LockedRequirement],
    target: TargetEnvironment,
) -> str: ...
```

`scripts/lock_dependencies.py` owns generation:

```python
Mode = Literal["write", "check", "update-compiler"]

@dataclass(frozen=True)
class CompileCommand:
    output: Path
    argv: tuple[str, ...]
    env: Mapping[str, str]

Runner = Callable[[CompileCommand], CompletedProcess[str]]

def build_compile_plan(repo: Path, policy: LockPolicy, mode: Mode) -> tuple[CompileCommand, ...]: ...
def compile_locks(repo: Path, mode: Mode, runner: Runner = run_command) -> None: ...
```

`scripts/install_locked_requirements.py` owns dedicated installs:

```python
Role = Literal["bootstrap", "compiler", "docs", "audit", "atlas-contract"]

@dataclass(frozen=True)
class InstallCommand:
    role: Role
    argv: tuple[str, ...]

def command_for(repo: Path, role: Role) -> InstallCommand: ...
def install_role(repo: Path, role: Role, runner: Runner = run_command) -> None: ...
```

`scripts/verify_dependency_locks.py` owns offline validation:

```python
@dataclass(frozen=True)
class LockFinding:
    category: str
    path: str
    detail: str

def verify_dependency_locks(repo: Path) -> tuple[LockFinding, ...]: ...
```

`scripts/check_image_locks.py` owns networked registry validation:

```python
@dataclass(frozen=True)
class ImageIdentity:
    source: str
    index_digest: str
    children: Mapping[str, str]

Inspector = Callable[[str], Mapping[str, object]]

def check_image_locks(repo: Path, inspect: Inspector = inspect_with_buildx) -> None: ...
```

All CLIs report stable categories and repository-relative paths, redact subprocess output, reject
unknown arguments/roles/platforms, and exit nonzero on the first unsafe boundary without hiding
which stable stage failed.

## 12.24.3 Task 0 — publish the reviewed plan

**Files:**

- Modify: `docs/manifest.yaml`
- Add: `docs/superpowers/plans/2026-08-16-issue-63-dependency-locks-implementation-plan.md`

- [ ] Add this plan as manifest section 12.24 immediately after the Issue #63 design.
- [ ] Run the complete docs manifest/build/wiki test set under `-W error`.
- [ ] Run `make docs-check`, `make docs-wiki`, and `git diff --check`.
- [ ] Obtain independent plan review with zero Critical/Important/Minor findings.
- [ ] Commit only these two paths as `docs: plan Issue 63 dependency locks`.
- [ ] Re-read the committed plan and confirm a clean worktree before Task 1.

## 12.24.4 Task 1 — define lock inputs, policy, and parser contracts

**Files:**

- Add: `bootstrap-requirements.txt`
- Add: `compiler-requirements.txt`
- Add: `nlp-model-requirements.txt`
- Modify: `atlas-contract-requirements.txt`
- Modify: `torch-requirements.txt`
- Add: `requirements/lock-policy.toml`
- Add: `requirements/image-lock.json`
- Add: `scripts/dependency_locks.py`
- Add: `tests/test_dependency_locks.py`

### 12.24.4.1 RED

- [ ] Write parser tests before production code. Require exact input/output inventory; sole compiler
  pin; exact three platform key/triple/system/machine mappings; Python 3.11.0; Darwin target 13.0;
  exact source bindings; exact direct-URL/model hash; sole python-louvain sdist exception; and exact
  image-ledger schema.
- [ ] Preserve PEP 508 markers as normalized text and evaluate them only with the exact supported
  target environments: CPython/posix, Python 3.11.0, Darwin arm64 or Linux x86_64/aarch64, with
  exact `implementation_name`, `os_name`, `platform_machine`, `platform_python_implementation`,
  `platform_system`, `python_version`, and `sys_platform` values. Reject `extra`,
  `implementation_version`, `python_full_version`, unknown marker variables, nondeterministic
  environment reads, and marker syntax outside `packaging.markers.Marker`; patch-sensitive markers
  cannot represent the promised complete `>=3.11.0,<3.12` consumer range.
- [ ] Add negative tests for duplicate TOML/JSON keys, duplicate normalized packages in an
  overlapping marker domain, uv version in
  policy, Atlas/compiler uv mismatch, stale PyG source directive in the runtime manifest, unknown
  input/output, missing architecture, malformed digest/hash, tag-only/digest-only image, child/index
  substitution, variable interpolation, unsafe index, VCS/editable/local requirement, and a second
  direct URL or sdist.
- [ ] Allow same-name supported-matrix-lock branches only when their markers are disjoint across all three
  supported target environments. Add positive disjoint-branch tests and killing mutations for
  overlapping branches, unmarked+marked overlap, an unsupported-variable marker, a marker that
  applies to no supported target, and incorrect
  Darwin/Linux/architecture/Python evaluation.
- [ ] Add explicit killing mutations for `python_full_version >= '3.11.10'`,
  `python_full_version >= '3.11.15'`, and `implementation_version` patch thresholds. Each must fail
  as an unsupported patch-sensitive marker rather than being evaluated only at the 3.11.0 floor.
  Positive projections evaluated under clean CPython 3.11.0, 3.11.10, and 3.11.15 marker
  environments must select identical branches because the accepted marker language can observe only
  `python_version == '3.11'`.
- [ ] Run `pytest -p no:cacheprovider -W error tests/test_dependency_locks.py -q` and record the
  expected import/missing-contract failures.

### 12.24.4.2 GREEN

- [ ] Add the three exact manifests and edit the two existing manifests minimally.
- [ ] Implement strict TOML/JSON/requirements parsing without network access. Use `tomllib`, reject
  bool-as-int and duplicate semantic bindings, normalize names with PEP 503 rules, and never execute
  manifest content.
- [ ] Populate the image ledger with the already resolved Jupyter index digest
  `sha256:14379f24c840f27375e3a4b29a9fa55e449633ac85c9cf3806ca62d11e5603ec`, devcontainer index
  digest `sha256:8e95c16fbc98a4a6a8f11f5b5bd152d0ffcd4fd0f4b31bd03e95965c777d2577`, and freshly verified
  amd64/arm64 child digests from `docker buildx imagetools inspect`.
- [ ] Rerun the focused tests. Mutate every authoritative input, platform tuple, source binding,
  image index/child, compiler identity, model URL/hash, and exception; assert the mutation changes
  bytes and the expected stable category fails.
- [ ] Run Ruff, py_compile, and diff checks.
- [ ] Stage exactly the nine listed paths, commit `build: define dependency lock policy`, and obtain
  independent review before Task 2.

## 12.24.5 Task 2 — compile and verify the complete lock family

**Files:**

- Add: `scripts/lock_dependencies.py`
- Add: `scripts/verify_dependency_locks.py`
- Add: `scripts/check_image_locks.py`
- Add: `scripts/install_locked_requirements.py`
- Add: `tests/test_lock_dependencies.py`
- Add: `tests/test_verify_dependency_locks.py`
- Add: `tests/test_check_image_locks.py`
- Add: `tests/test_install_locked_requirements.py`
- Add/modify: every generated lock in Section 12.24.2.2
- Modify: `Makefile`
- Modify: `scripts/verify_repo.py`
- Modify: `tests/test_verify_repo.py`

### 12.24.5.1 RED — compiler

- [ ] Before generator code, test the sanitized `bootstrap` and `compiler` install roles: exact
  repository-relative paths, `--require-hashes`, binary-only, neutralized ambient pip configuration,
  stop-on-nonzero, stable/redacted failure, and rejection of every other role until its generated
  lock exists. Add `install-bootstrap` and `install-compiler-lock` Make contracts in this task.
- [ ] Test exact generation order and argv for all targets. Compiler invocations use uv 0.11.19,
  Python 3.11.0, exact platform triples, explicit source bindings, `--generate-hashes`, binary-only
  policy plus only the python-louvain exception, the exact policy-owned
  `--exclude-newer 2026-08-17T02:21:18Z`, and `MACOSX_DEPLOYMENT_TARGET=13.0` on Darwin.
  Bootstrap, compiler, docs, audit, and Atlas inputs are each compiled once per exact supported
  target into temporary files, then deterministically merged into one supported-matrix lock; do not
  use uv's unbounded `--universal` world. Equal name/version/marker branches union their approved
  hashes; target-specific versions receive normalized disjoint target markers. Every retained
  branch must apply to at least one of the three supported targets and no overlapping branch may
  select two versions for one target. Only core/runtime/root remain separate platform outputs.
- [ ] Test temporary exact-version projections for bootstrap→docs, core→runtime, and
  bootstrap/docs/runtime→root. Projections add no package, contain no hash/source option, and are
  deleted after success or failure. Every projection evaluates markers for its exact target first;
  supported-matrix locks retain marker branches, while a platform projection has at most one selected
  branch per normalized name.
- [ ] Test normalized headers, deterministic environment, complete temporary validation,
  transactional replacement/backup restoration, stable errors, and no checkout writes in `check`.
- [ ] Test the explicit compiler-update five-state transition: old committed identity, first-pass
  candidate, new RECORD-owned environment, second-pass identical candidate, Atlas equality, then
  final family. Every out-of-order, ambiguous, missing, or mismatched state fails.
- [ ] Run the new compiler tests and record intended RED failures.

### 12.24.5.2 GREEN — generated locks

- [ ] Implement the runner seam and modes. Never shell-interpolate paths or credentials. The normal
  modes reject compiler drift; only `--update-compiler` accepts the named old/new intermediate.
- [ ] Perform the one-time seed ceremony explicitly: resolve
  `UV_SEED="$(pyenv prefix 3.11.0)/bin/uv"`, require that exact file to report `uv 0.11.19` (never
  use the ambient Homebrew/PATH uv), and use it to generate candidate bootstrap and
  compiler locks; create a new disposable Python 3.11 environment; hash-install those two candidates
  through the new roles pointed at the validated temporary candidate root (never by copying
  unvalidated candidates into the worktree); verify the RECORD-owned uv identity; regenerate bootstrap/compiler and the
  remaining family through `python -m uv`; and require the seed and locked-regeneration candidates
  to be byte-identical. No later workflow may use the standalone seed path.
- [ ] From that locked disposable environment, run the exact write command and generate all 14
  outputs.
- [ ] Evaluate every merged marker under CPython 3.11.0, 3.11.10, and 3.11.15 for each applicable
  platform, require identical package/version selections across patch releases, and run
  hash-required clean-install controls on the available exact 3.11.0 and 3.11.15 Darwin
  interpreters. Docker Task 5 supplies the exact 3.11.10 Linux install control.
- [ ] Inspect the diff for unexpected upgrades, source changes, sdists, local tags, and unbounded
  requirements. Prove exactly one sdist and one direct URL.
- [ ] Run write twice in independent temporary roots and byte-compare every output.
- [ ] After the release-boundary stale-lock diagnostic, reproduce the drift from packages uploaded
  after the original lock commit, add the fixed upload horizon test first, regenerate without
  selecting those later releases, and prove a fresh networked `lock-check` is byte-stable.

### 12.24.5.3 RED/GREEN — offline and image verifiers

- [ ] Write the offline verifier tests first: inventory/header/input hashes, exact pins and hash
  shape, cumulative package/version subsets, shared packaging/docs projections, platform tags,
  model/source exceptions, and stable/redacted CLI categories. This task validates authority and
  generated outputs only; Tasks 3, 5, and 6 add installer, CI/image, and documentation consumer
  projections as those consumers change, so the verifier never requires a future task's source
  state.
- [ ] Explicitly prove a coordinated syntactically valid wrong artifact hash passes offline syntax
  and then fails `lock-check` or a hash-required clean install. Do not write a vacuous offline oracle.
- [ ] Write image tests with injected inspector output: exact tag/index/child success; moved tag,
  wrong/missing/extra platform, stale child, wrong media type, malformed JSON, nonzero process, and
  redacted error failure. A coordinated source+ledger false digest passes offline coherence and fails
  the injected trusted registry response.
- [ ] Add Make targets `lock-write`, `lock-check`, `image-lock-check`, and
  `verify-dependency-locks`; include only the offline verifier in `make verify`.
- [ ] Add D10 integration to `scripts.verify_repo` and tests that mutate every Task 2
  Make/script/input/output seam. Do not add a transitional allowlist for still-unmodified consumers.
- [ ] Run focused tests, all repository-verifier tests, networked `make lock-check`, networked
  `make image-lock-check`, Ruff, py_compile, and diff checks.
- [ ] Stage exactly the Task 2 files and generated family, commit
  `build: generate verified dependency locks`, and obtain independent review before Task 3.

## 12.24.6 Task 3 — install locks through one sanitized boundary

**Files:**

- Modify: `scripts/install_locked_requirements.py`
- Modify: `tests/test_install_locked_requirements.py`
- Modify: `scripts/install_torch_stack.py`
- Modify: `tests/test_install_torch_stack.py`
- Modify: `scripts/verify_dependency_locks.py`
- Modify: `tests/test_verify_dependency_locks.py`
- Modify: `Makefile`
- Modify: `tests/test_makefile_contract.py`

### 12.24.6.1 RED

- [ ] Expand the bootstrap/compiler helper to the closed five-role map and test exact argv. Every
  dedicated role uses `--require-hashes` and is binary-only. Separately, the four-stage installer's
  root stage owns the sole python-louvain non-binary exception. Reject ambient pip index/find-links/constraint/requirement/
  config state, unknown roles, absolute/out-of-repo locks, symlink escapes, missing locks, nonzero
  subprocess results, shell execution, output leakage, and continuation after failure.
- [ ] Update four-stage tests to expect bootstrap/core/runtime/root locks, exact platform source
  flags including the Linux-only Torch CPU `torchao` project page required by uv's CPU backend
  output, root's python-louvain build exception, and all three supported host mappings.
- [ ] Add functional disposable-venv controls proving bootstrap failure stops before core, core
  failure stops before runtime, and no role installs after final verification.
- [ ] Add Make contract RED tests for the five dedicated install targets, four-stage runtime target,
  Codespaces recipe, and `nlp-assets` containing NLTK VADER only—no `spacy download`, pip invocation,
  or model package mutation.

### 12.24.6.2 GREEN

- [ ] Implement the shared runner and refactor the four-stage installer to reuse its environment
  sanitization and stable failure protocol without changing stage count.
- [ ] Add exact Make seams. Runtime/Codespaces/Docker targets use `install-torch-stack` (whose stage
  0 is bootstrap). `install-docs-lock`, `install-compiler-lock`, `install-audit-lock`, and
  `install-atlas-contract-lock` install only their named role; Task 5 wires each dedicated consumer
  as `install-bootstrap` once followed by its role target. Runtime consumers never install the docs
  lock again.
- [ ] Extend the offline verifier and mutations to require the final installer and Make target
  references introduced in this task. It must reject legacy source-manifest install commands after
  this commit without making claims about workflow/image/docs consumers owned by later tasks.
- [ ] Remove spaCy's package-changing download. Verify `python -c 'import en_core_web_sm'` after the
  root lock and keep only the NLTK VADER data download before final provenance checks.
- [ ] Run focused installer/Make tests and mutation tests for omitted hash/binary/source flags,
  reordered/duplicated stages, late pip commands, role substitution, and fail-open runners.
- [ ] Perform fresh Darwin installs for each dedicated role and the complete four-stage runtime;
  run `pip check`, Torch verifier, NNx verifier, spaCy model import, and record exact interpreter.
- [ ] Run Ruff, py_compile, verify, and diff checks. Commit exact owned paths as
  `build: install reviewed dependency locks`; obtain independent review.

## 12.24.7 Task 4 — derive advisory evidence from locks

**Files:**

- Modify: `scripts/advisory_baseline.py`
- Modify: `tests/test_advisory_baseline.py`
- Modify: `security/accepted-advisories.json`
- Modify: `scripts/verify_repo.py`
- Modify: `tests/test_verify_repo.py`

### 12.24.7.1 RED

- [ ] Replace source-manifest resolver fixtures with exact validated lock-union fixtures. Require
  exactly four logical surfaces: combined-runtime, torch, documentation, and atlas-contract.
- [ ] Test bootstrap inclusion on all four, audit-tool subset of combined-runtime, all three platform
  roots/core/runtime projections, public-version normalization with retained local-tag provenance,
  and exact supplemental equality.
- [ ] Test exact `--no-deps --disable-pip --strict` commands. No source manifest may be passed to
  pip-audit and no package may be installed/resolved during observation.
- [ ] Test explicit non-PyPI evidence for only `en-core-web-sm` and `pyg-lib`, including canonical
  name/version/source/hash, `audited=false`, and `reason=non-pypi`. Scatter/sparse may not use it.
- [ ] Mutate source-manifest re-resolution, bootstrap omission, range input, URL/PyG omission,
  surface membership, local/public versions, unsupported allowlist, audit-tool subset, and accepted
  advisory surface sets. Require exact stable failure.

### 12.24.7.2 GREEN

- [ ] Implement lock parsing/projection through `scripts.dependency_locks`; never duplicate a lock
  parser in the advisory module.
- [ ] Run the exact live audit. Update accepted setuptools advisory surfaces to every observed
  bootstrap-bearing logical surface, or remove it only if the selected locked version is proven
  clean. Accept no unrelated advisory.
- [ ] Preserve six-command/four-surface output identity where applicable, but update evidence schema
  so each observation records contributing lock path/hash and non-PyPI records.
- [ ] Run full advisory tests, D10 verifier tests, and live `make audit-advisories`. Prove a source
  range mutation no longer changes the live command and a lock-version mutation does.
- [ ] Run Ruff, verify, diff checks; commit exact owned paths as
  `security: audit exact dependency locks`; obtain independent review.

## 12.24.8 Task 5 — wire CI, Docker, Codespaces, and image identities

**Files:**

- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/docs.yml`
- Modify: `.github/workflows/pages.yml`
- Modify: `.github/workflows/atlas-contract.yml`
- Modify: `Dockerfile`
- Modify: `.devcontainer/devcontainer.json`
- Modify: `Makefile`
- Modify: `scripts/verify_dependency_locks.py`
- Modify: `tests/test_verify_dependency_locks.py`
- Modify: `tests/test_verify_repo.py`
- Modify: `tests/test_makefile_contract.py`

### 12.24.8.1 RED

- [ ] Add workflow tests requiring exact CPython 3.11.15, bootstrap before every Python-package
  lock install, runtime jobs using only the canonical four-stage installer, dedicated jobs using
  their exact role, final verification after NLTK data, and no later package mutation.
- [ ] Require cache keys to hash every consumed lock and policy/input that can change its meaning.
  Reject source-manifest-only keys, wrong platform lock, missing bootstrap/compiler lock, and
  cross-role substitutions.
- [ ] Add the fail-closed dependency-change classifier contract: schedules/dispatches run both
  network checks; relevant dependency diffs run lock-check; relevant image diffs run image-check;
  missing/ambiguous base data runs both; code-only diffs may skip the steps but not the always-on
  offline verifier/audit job. Mutate every path group and fail-open condition.
- [ ] Preserve Issue #62 job names, PR label/action selection, Tier B/C activation, checkout
  submodules, warning policy, failure masking protections, and service/container prohibition.
- [ ] Add Docker/devcontainer parser tests for exact tag@index references, ledger equality, native
  multi-platform indexes, no `--platform`, canonical installer use, and final in-image pip/Torch/NNx
  checks. Mutate tags, indexes, children, architecture, install path, and late package commands.

### 12.24.8.2 GREEN

- [ ] Change all setup-python inputs to exact 3.11.15. Wire bootstrap/dedicated targets and platform
  runtime locks. Remove direct installs of human source manifests and duplicate docs/root solves.
- [ ] Add exact dependency and image selector steps without masking. The network lock step installs
  compiler lock only when selected; the image step introduces no Python package.
- [ ] Pin Docker and devcontainer tag@index references to the design digests. Preserve native arm64/
  amd64 selection and never use an architecture child as the source digest. Build the Docker lock
  into its exact `/home/jovyan/.venvs/ml-eng-lab` environment so preinstalled base-image packages
  cannot contaminate `pip check` or the runtime, and set `CONDA_AUTO_ACTIVATE_BASE=false` so the
  Jupyter entrypoint cannot reactivate the ambient conda base ahead of the locked venv.
- [ ] Update Codespaces and Docker comments/help to name lock behavior and VADER-only post-lock data.
- [ ] Extend the offline verifier and mutations to require exact workflow, Docker, devcontainer,
  cache, image-ledger, and Codespaces consumer references. After this commit no executable package
  consumer may remain outside the reviewed install roles/platform installer.
- [ ] Run focused workflow/Make tests, all workflow YAML parsing, comment-stripped devcontainer JSON
  parsing, Ruff, verify, docs checks, and diff checks.
- [ ] Perform a no-cache native arm64 Docker build and three `run --rm` probes: Python/pip check,
  Torch stack verifier, NNx verifier. Inspect exact image architecture/base provenance; remove only
  the issue-owned image afterward.
- [ ] Stage exact owned files, commit `ci: consume immutable dependency inputs`, and obtain
  independent review.

## 12.24.9 Task 6 — synchronize current documentation and release records

**Files:**

- Modify: `README.md`
- Modify: `CONTRIBUTING.md`
- Modify: `SECURITY.md`
- Modify: `CHANGELOG.md` (Unreleased only)
- Modify: `docs/env-setup.md`
- Modify: `docs/architecture.md`
- Modify: `docs/dependency-contracts.md`
- Modify: `docs/notebook-infrastructure.md`
- Modify: current manifest-owned notebook pages/specs that describe installation
- Modify: `tests/test_check_docs.py`
- Modify: `tests/test_build_docs.py`
- Modify: `tests/test_wiki.py`
- Modify: `scripts/verify_dependency_locks.py`
- Modify: `tests/test_verify_dependency_locks.py`
- Add: `scripts/verify_issue63_report.py`
- Add: `tests/test_verify_issue63_report.py`

### 12.24.9.1 RED

- [ ] Define report schema 1 and its validator before final freeze. A complete synthetic fixture
  must bind final SHA/tree, design/plan/commit range, lock/input/image identities, compiler and
  interpreter identities, JUnit totals, advisory surfaces/non-PyPI records, Tier counts, Docker,
  dual-SHA PR/run/check/log evidence, Pages/wiki, ruleset/issues, and cleanup. Delete/change/type-
  mutate each required field, add unknown keys, escape repository paths, or forge URL/hash/count
  relationships and require stable failure. Issue evidence uses the exact canonical substantive
  projection and Issue #63 project/item/field/option identifiers from Task 7, never timeline or
  timestamp equality. The validator is read-only and accepts the eventual
  ignored report path; Task 7 supplies live values but introduces no new validation code.
- [ ] Run the focused report-validator tests, Ruff, py_compile, and diff check; commit exactly
  `scripts/verify_issue63_report.py` and `tests/test_verify_issue63_report.py` as
  `test: define Issue 63 qualification evidence`, and obtain independent review before editing
  documentation.
- [ ] Inventory every current claim containing install commands, requirements filenames, Python
  3.11, mutable image tags, spaCy download, Torch/PyG sources, audit surfaces, or reproducibility.
  Classify current versus historical before editing.
- [ ] Add canonical tests for the exact lock inventory, platform mapping, Python floor/accepted patch
  versions, bootstrap/compiler identities, model and sdist exceptions, tag@index images, network vs
  offline proof boundary, advisory lock projections, update/rollback commands, and #64/#65/#66
  boundaries.
- [ ] Add stale-claim mutations for direct source installs, mutable tags, `spacy download`, generic
  cross-platform reproducibility, uv-as-runtime-manager, and offline remote-hash claims.

### 12.24.9.2 GREEN

- [ ] Update current README/setup/contributor/security/architecture/dependency/notebook surfaces from
  the design and live implementation. Say “reproducible for the qualified platform lock,” not
  perfectly cross-platform reproducible.
- [ ] Record Issue #63 in Unreleased only. Do not alter released sections; retain/extend the released
  history hash guard.
- [ ] Complete the offline verifier's documentation-consumer projection and mutations so every
  current install/lock/image/update claim is sourced from the canonical inventory and no historical
  section is treated as a live consumer.
- [ ] Document controlled `lock-write`, `lock-check`, `image-lock-check`, offline verify, exact clean
  install, Docker/devcontainer pins, advisory non-PyPI records, VADER limitation, and rollback.
- [ ] Run complete docs tests, link tests, `make docs-check`, strict MkDocs, `make docs-wiki`, exact
  canonical→site/wiki parity, stale-current-text sweep, Ruff, verify, and diff checks.
- [ ] Stage only planned current docs/tests, commit `docs: document immutable dependency locks`, and
  obtain independent review.

## 12.24.10 Task 7 — freeze, qualify, publish, and clean up

### 12.24.10.1 Pre-freeze closure

- [ ] Rebase/merge no outside work. Require a clean feature worktree, exact infra gitlink, no
  generated docs, no notebook source/output diff, and no issue-owned container/image/process.
- [ ] Run strict full tests under `PYTEST_ADDOPTS='-p no:cacheprovider -W error'`, Ruff, py_compile,
  `make verify`, docs/site/wiki gates, workflow YAML, devcontainer semantics, diff check, and all
  focused mutation suites.
- [ ] Obtain final independent code/design/plan review. Correct findings in tracked commits and repeat
  this boundary until zero Critical/Important/Minor findings.
- [ ] Freeze `FINAL_SHA=$(git rev-parse HEAD)`. From this point any tracked edit invalidates the
  freeze and restarts this entire task.

### 12.24.10.2 Fresh local and container qualification

- [ ] Create a brand-new detached worktree, Python 3.11.15 venv, pip/cache/kernelspec/output roots,
  and locally populated exact infra gitlink. Prove empty artifact roots and exact interpreter path.
- [ ] Create a separate compiler-check venv that installs bootstrap+compiler locks and runs networked
  byte regeneration. The notebook runtime venv installs only the complete Darwin four-stage stack;
  prove uv is absent there. Run image-ledger checks, offline verification, runtime `pip check`, Torch
  verifier, NNx verifier, strict warning-debt probe, exact QAT evidence, full NNx
  surface with JUnit, and full repository tests with JUnit. Require zero failures/errors/skips where
  the Issue #62 contract requires them.
- [ ] Run live lock-derived advisory reconciliation and preserve the exact four-surface inputs,
  outputs, lock hashes, non-PyPI records, and accepted-policy result.
- [ ] Run Ruff, py_compile, repository verifier with zero findings/warnings, docs strict/wiki/parity,
  workflow YAML, devcontainer checks, frozen-SHA/status/diff/source-clean gates.
- [ ] Build the root Docker image no-cache and natively without `--platform`. Require Linux arm64,
  Python 3.11.10, pinned base index, `pip check`, Torch/PyG backend verifier, NNx verifier, and three
  fresh `docker run --rm` probes. No persistent container.
- [ ] Execute Tier A to a fresh root and verify exactly 18 notebooks/210 code cells; Tier B exactly
  6/75; Tier C exactly 4/56. Require integer non-bool execution counts, list/dict output shapes,
  zero errors/unexecuted cells, exact independent roots, and source-clean after every tier.

### 12.24.10.3 GitHub qualification and GitFlow

- [ ] Before any push/PR, query Issues #64/#65/#66 with exactly
  `state,title,body,labels,assignees,projectItems`, require `state=OPEN`, and write a canonical JSON
  projection containing state/title/body; labels sorted as `(name,color,description)`; assignee
  logins sorted; and project items sorted as `(title,status.name,status.optionId)`. Exclude only
  timestamps and timeline/cross-reference events. Hash each canonical projection for final exact
  comparison.
- [ ] Preflight Issue #63's unique project binding before publication: exactly one item
  `PVTI_lAHOABrzpM4Bcwqgzg2ByY8` in project `PVT_kwHOABrzpM4Bcwqg` number 11 titled
  `ml-eng-lab Maintenance & Roadmap`; exactly one Status field
  `PVTSSF_lAHOABrzpM4BcwqgzhXXF18`; current option `47fc9ee4`/`In Progress`; and available Done
  option `98236657`. Any missing/duplicate/different binding stops publication.
- [ ] Push exact FINAL_SHA to the
  feature ref and create/reuse only a narrowly owned ready PR to `develop` with the Issue #63 body,
  exact `tier-b-smoke` label, and no ambiguous/obsolete scoped PR. For a new PR, treat its opened CI
  run as the bounded Tier-B-skip contaminant and select the uniquely triggered labeled CI run; for a
  reused current PR, select the exact synchronize or labeled run required by the schema-2 verifier.
- [ ] Run the explicit ten-job workflow dispatch at FINAL_SHA and require exact job set/all success
  under the established 210-minute bound.
- [ ] Require selected PR-event evidence through `scripts.verify_pr_run_evidence`: source metadata
  SHA, current synthetic merge SHA/parents/tree, exact selected labeled/synchronize CI run, Docs and
  Atlas runs, applicable job checkout logs, required successes, expected Tier C skip only, and
  bounded opened-run contaminant handling.
- [ ] Read the exact-three protected ruleset and preserve it byte-for-byte; Issue #63 does not add a
  required context. Merge the feature PR with merge commit, require develop tree equals FINAL_SHA.
- [ ] Create/reuse the narrowly owned `develop`→`main` release PR, repeat dual-identity/run/check
  proof with the exact `tier-b-smoke` label and selected labeled/synchronize CI run, merge with merge
  commit, require main tree equals develop tree, and validate the exact new
  Pages push run plus live site/wiki content.
- [ ] If ancestry requires it, create/reuse a content-neutral `main`→`develop` sync PR, require tree
  equality, no Tier-B label, selected CI-only opened/synchronize evidence with expected Tier B/C
  skips, merge, capture final develop SHA,
  and wait for every exact-SHA run to complete successfully.

### 12.24.10.4 Immutable report and cleanup

- [ ] Re-query and canonicalize Issues #64/#65/#66 with the exact pre-push projection, require OPEN,
  and byte/hash-compare each projection to its pre-push snapshot. Prove ruleset unchanged,
  main/develop trees equal, zero scoped open/ambiguous PRs, zero owned queued/in-progress runs, and
  live Pages/wiki parity.
- [ ] Remove exact feature local/remote/tracking refs, `refs/issue63/*`, disposable worktrees/venvs/
  notebook outputs, and only exact issue-owned Docker images, while retaining the bounded raw
  evidence directory until the report is validated. Prune stale worktree registrations and prove
  only intended worktrees/containers/images remain.
- [ ] Write ignored `.superpowers/sdd/issue63-qualification-report.md` in the primary checkout after
  cleanup. Its versioned schema binds FINAL_SHA, tree, design/plan/commit range, every
  lock/input/image digest, compiler identity, local/Docker interpreters, test/JUnit totals,
  advisory observations, Tier counts, PR/run/check/log URLs and hashes, Pages/wiki evidence,
  ruleset snapshots, issue snapshots, and the post-cleanup inventory. Mutate every required field
  through the already-frozen `scripts.verify_issue63_report` contract and require validation
  failure; do not create or edit validator code after freeze.
- [ ] After report validation, remove the exact bounded raw evidence directory and prove no other
  path was touched; the ignored immutable report is the retained evidence artifact.
- [ ] Post the immutable report to feature PR, release PR, and Issue #63. Comment on Issue #53 that
  Issue #63 completed while #64/#65/#66 remain open; prove #53 stays OPEN.
- [ ] Immediately before the only project mutation, repeat the exact Issue #63 project-item/field/
  current-option/Done-option preflight above. Update only that item and Status field to option
  `98236657`; re-query the same unique item and require `Done`. Then close Issue #63 as the final
  intended external mutation. If project automation closes it on Done, verify CLOSED/COMPLETED and
  do not reopen merely to add another comment.
- [ ] Confirm primary checkout clean on synchronized `develop`, Issue #53 OPEN, and no remaining
  issue-owned state. Only then mark this plan complete and continue the roadmap goal with the next
  accepted issue.

## 12.24.11 Commit and review boundaries

Expected implementation commits, each independently reviewed before the next task:

1. `build: define dependency lock policy`
2. `build: generate verified dependency locks`
3. `build: install reviewed dependency locks`
4. `security: audit exact dependency locks`
5. `ci: consume immutable dependency inputs`
6. `test: define Issue 63 qualification evidence`
7. `docs: document immutable dependency locks`

Review corrections use separate narrowly scoped commits. No commit may combine unrelated task
ownership merely to make staging convenient. Generated lock changes belong with the compiler/verifier
task that proves them; current documentation belongs only to Task 6. Final evidence remains ignored
and must not create a post-freeze tracked commit.
