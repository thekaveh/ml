# 12.14 Issue 58 canonical NNx wheel contract implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make local and CI NNx compatibility evidence fail closed unless Python resolves the exact
`thekaveh-nnx[lm]==0.2.0` named-release wheel, while preserving a validated explicit editable
development mode.

**Architecture:** A standard-library verifier parses the single canonical pin from
`requirements.txt`, inspects public distribution metadata, and compares the distribution-owned
package path with the unresolved import specification. A session fixture gates every NNx-surface
entry point, while both the required repository CI job and focused NNx job install NNx with
binary-only selection and run an explicit verifier step immediately before tests. Canonical docs
publish the same contract to the repository, site, and wiki.

**Tech Stack:** Python 3.11 standard library (`importlib.metadata`, `importlib.util`, `json`,
`pathlib`, `urllib.parse`), pytest, GNU Make, GitHub Actions YAML, MkDocs documentation tooling.

## 12.14.1 Global constraints

- Keep `thekaveh-nnx[lm]==0.2.0` unchanged; Issue #61 owns any version upgrade.
- Do not add a wheel hash, dependency lock, or second NNx pin; Issue #63 owns hash locking.
- Canonical mode rejects every `direct_url.json`, not only editable direct URLs.
- Only exact `NNX_ALLOW_EDITABLE=1` admits local editable mode; the override validates rather than
  bypasses provenance.
- CI must not define `NNX_ALLOW_EDITABLE` or `PYTHONPATH` at workflow, job, or step scope.
- Both `pytest-repository` and `pytest-nnx-surface` must select the NNx wheel with
  `--only-binary=thekaveh-nnx` and validate after installation and immediately before tests.
- Keep GitHub required contexts exactly `pytest-repository` and `atlas-consumer-policy`.
- Do not start or contact Atlas, JupyterHub, Ollama, ComfyUI, Docker Compose, or any runtime
  service. Never use containerized Ollama.
- Do not stop or modify unrelated containers.
- Generated site/wiki trees remain ignored; edit canonical sources only.
- Every production change follows red-green-refactor and receives a task review before the next
  task starts.

---

## 12.14.2 Task 1: Implement the fail-closed verifier and local test boundary

**Files:**
- Create: `scripts/verify_nnx_install.py`
- Create: `tests/test_verify_nnx_install.py`
- Modify: `tests/nnx_surface/conftest.py`
- Modify: `Makefile`
- Modify: `tests/test_makefile_contract.py`
- Modify: this plan

**Interfaces:**
- Produces: `NnxPin`, `NnxInstallEvidence`, `VerificationError`, `parse_nnx_pin()`,
  `verify_nnx_install()`, and CLI `python -m scripts.verify_nnx_install`.
- Produces: public Make target `verify-nnx-install`.
- Consumes later: Task 2 calls exact `make verify-nnx-install`; Task 3 documents exact
  `NNX_ALLOW_EDITABLE=1 make test-nnx-surface`.

- [ ] **Step 1: Add pin-parser and canonical-wheel unit tests**

Create `tests/test_verify_nnx_install.py` with fixture-owned fake distributions and import specs.
The fake distribution must expose only public-compatible members used by production:

```python
class FakeDistribution:
    def __init__(self, *, name="thekaveh-nnx", version="0.2.0", root, files, direct_url=None):
        self.metadata = {"Name": name}
        self.version = version
        self.files = [Path(value) for value in files]
        self._root = root
        self._direct_url = direct_url

    def read_text(self, filename):
        if filename == "direct_url.json":
            return self._direct_url
        return None

    def locate_file(self, path):
        return self._root / path
```

Add focused tests proving:

```python
assert parse_nnx_pin("thekaveh-nnx[lm]==0.2.0\n").version == "0.2.0"
```

and rejection of missing, duplicate, ranged, URL, editable, wrong-extra, and malformed NNx
requirements. Add a canonical accepted fixture with `WHEEL`, `RECORD`, and `nnx/__init__.py`, plus
rejected cases for wrong version, wrong normalized name, duplicate distributions, any
`direct_url.json`, missing owned files, import-origin mismatch, repository ownership, and symlink
escape. Assert diagnostics contain neither fake host paths nor direct URLs.

- [ ] **Step 2: Run and record parser/canonical RED**

```bash
pytest -p no:cacheprovider tests/test_verify_nnx_install.py -q
```

Expected: collection fails because `scripts.verify_nnx_install` does not exist. Record the exact
failure in `.superpowers/sdd/task-1-report.md` before adding production code.

- [ ] **Step 3: Implement the minimal parser and canonical verifier**

Create `scripts/verify_nnx_install.py` with these public shapes:

```python
@dataclass(frozen=True)
class NnxPin:
    version: str
    extras: frozenset[str]


@dataclass(frozen=True)
class NnxInstallEvidence:
    mode: str
    distribution: str
    version: str


class VerificationError(RuntimeError):
    """The active NNx installation does not satisfy the selected contract."""
```

Expose `parse_nnx_pin(requirements_text: str) -> NnxPin`. Expose
`verify_nnx_install(requirements_path: Path = REPO_ROOT / "requirements.txt", *, environ:
Mapping[str, str] | None = None, distributions: Iterable[metadata.Distribution] | None = None,
find_spec: Callable[[str], ModuleSpec | None] = importlib.util.find_spec, repo_root: Path =
REPO_ROOT) -> NnxInstallEvidence`. Passing `None` for `environ` or `distributions` selects
`os.environ` or `importlib.metadata.distributions()` respectively; injected values make unit tests
independent of the active installation.

Normalize distribution names by lowercasing and collapsing runs of `[-_.]` to `-`. Parse only an
active, uncommented exact requirement line. In canonical mode, require one matching distribution,
matching version, absent direct URL, exact owned files, matching resolved import origin, and an
owned path outside `repo_root`. Inspect `find_spec("nnx").origin` without importing `nnx`.

Errors use stable categories such as `NNx requirement is not an exact pin`, `NNx distribution
metadata is not canonical`, or `NNx import is not owned by the installed distribution`; never
interpolate a path, URL, JSON body, environment value, or exception text. The CLI prints only:

```text
NNx installation verified: thekaveh-nnx 0.2.0 (canonical-wheel)
```

and exits nonzero with a redacted one-line error on failure.

- [ ] **Step 4: Run canonical unit tests GREEN**

```bash
pytest -p no:cacheprovider tests/test_verify_nnx_install.py -q
ruff check --no-cache scripts/verify_nnx_install.py tests/test_verify_nnx_install.py
```

Expected: all new tests pass and Ruff is clean.

- [ ] **Step 5: Add editable-mode tests RED**

Add tests requiring valid PEP 610 JSON shaped as:

```json
{"dir_info":{"editable":true},"url":"file:///workspace/NNx"}
```

Prove exact `NNX_ALLOW_EDITABLE=1` accepts an import beneath the declared directory while still
requiring the exact pin and installed version. Reject an unknown override value, malformed JSON,
missing/non-boolean `dir_info.editable`, non-`file` URL, non-editable directory/archive/VCS
metadata, and an import outside the declared source. Confirm diagnostics redact the URL/path.

Run the focused tests and record that editable cases fail before implementation:

```bash
pytest -p no:cacheprovider tests/test_verify_nnx_install.py -q -k editable
```

- [ ] **Step 6: Implement validated editable mode GREEN**

Use `urllib.parse.urlparse()` and `urllib.request.url2pathname()` to decode only a local `file:` URL.
Treat absent or empty `NNX_ALLOW_EDITABLE` as canonical; reject every value except `1`. Resolve the
declared source and import origin, require the latter to be beneath the former and end in
`nnx/__init__.py`, and return mode `editable-development`.

```bash
pytest -p no:cacheprovider tests/test_verify_nnx_install.py -q
ruff check --no-cache scripts/verify_nnx_install.py tests/test_verify_nnx_install.py
```

- [ ] **Step 7: Add local-boundary and Make contract tests RED**

In `tests/test_makefile_contract.py`, require an exact `.PHONY` member and help entry for
`verify-nnx-install`, and require `make -n verify-nnx-install` to emit exactly:

```text
python -m scripts.verify_nnx_install
```

Add a unit test that loads `tests/nnx_surface/conftest.py` as text and requires one session-scoped,
autouse fixture to call `verify_nnx_install()` before NNx API tests. Mutation cases must reject
function scope, `autouse=False`, swallowed `VerificationError`, or an environment mutation inside
the fixture.

```bash
pytest -p no:cacheprovider tests/test_makefile_contract.py tests/test_verify_nnx_install.py -q
```

Expected: the new Make/fixture contract tests fail because those surfaces do not exist.

- [ ] **Step 8: Wire the local boundary GREEN**

Add `verify-nnx-install` to `.PHONY`, help output, and the Makefile:

```make
verify-nnx-install:
	$(PYTHON) -m scripts.verify_nnx_install
```

In `tests/nnx_surface/conftest.py`, import `verify_nnx_install` and add:

```python
@pytest.fixture(scope="session", autouse=True)
def _verify_nnx_installation_contract():
    verify_nnx_install()
```

Do not catch verification failures or change environment variables. Keep the existing per-test
seed/chdir fixture unchanged.

Run:

```bash
pytest -p no:cacheprovider tests/test_makefile_contract.py tests/test_verify_nnx_install.py -q
make verify-nnx-install
NNX_ALLOW_EDITABLE=1 make verify-nnx-install
NNX_ALLOW_EDITABLE=1 make test-nnx-surface
```

Expected: default `make verify-nnx-install` fails against the currently reproduced editable local
environment; the explicit editable verifier and focused suite pass. Record both negative and
positive evidence rather than treating the intentional default failure as a blocker.

- [ ] **Step 9: Commit Task 1**

Mark Task 1 only complete in this plan. Re-run Step 8, stage only Task 1 files, run
`git diff --cached --check`, and commit:

```bash
git commit -m "test: verify NNx installation provenance"
```

Write the full red/green results and any concerns to `.superpowers/sdd/task-1-report.md`.

## 12.14.3 Task 2: Enforce canonical wheel validation in CI

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `tests/test_verify_repo.py`
- Modify: this plan

**Interfaces:**
- Consumes: Task 1's exact `make verify-nnx-install` command.
- Produces: mutation-resistant contracts for required `pytest-repository` and focused
  `pytest-nnx-surface` jobs.
- Preserves: current required GitHub contexts and every non-NNx CI job.

- [ ] **Step 1: Add repository-job workflow regressions RED**

Extend the existing `test_ci_runs_complete_repository_test_contract` boundary or add a narrowly
named companion that requires this exact ordered repository-job sequence:

```text
Checkout
Install system dependencies for cairosvg
Set up Python 3.11
Install dependencies
Verify canonical NNx installation
Run complete repository tests
```

Require the installation body to retain `make install-torch-stack`, use exact
`python -m pip install --only-binary=thekaveh-nnx -r requirements.txt`, and then install
`docs-requirements.txt`. Require the verifier step to be exact `make verify-nnx-install` and the
test step exact `make test`.

Mutation cases must fail for removed/reordered verifier, validation before the last pip install,
removed or widened binary selection, editable/direct install, job/workflow/step
`NNX_ALLOW_EDITABLE` or `PYTHONPATH`, extra step, condition, service, container, failure mask,
shell override, or test-command weakening.

```bash
pytest -p no:cacheprovider tests/test_verify_repo.py -q -k 'repository_test_contract and nnx'
```

Expected: fail because the current job lacks canonical validation and binary-only selection.

- [ ] **Step 2: Add focused-job workflow regressions RED**

Add `test_ci_nnx_surface_job_enforces_canonical_wheel_contract` requiring the exact job identity,
runner, timeout, checkout, Python/cache manifests, install command, Ruff command, verifier command,
and focused Make command. The exact ordered steps are:

```text
Checkout
Set up Python 3.11
Install dependencies
Lint (ruff check)
Verify canonical NNx installation
Run NNx-surface tests
```

Apply the same mutation controls as the repository job. Require the verifier job's
`Test repository workflow contracts` selector to contain both new/updated NNx workflow test names,
so deleting the production gate cannot be hidden by deleting its CI self-test.

```bash
pytest -p no:cacheprovider tests/test_verify_repo.py -q -k 'nnx_surface_job or repository_workflow_contract_tests'
```

Expected: fail against the print-only smoke and missing selector terms.

- [ ] **Step 3: Update CI minimally GREEN**

In only `pytest-repository` and `pytest-nnx-surface`:

- replace runtime `pip install -r requirements.txt` with
  `python -m pip install --only-binary=thekaveh-nnx -r requirements.txt`;
- use `python -m pip` for the docs requirements command in the repository job;
- replace the print-only NNx step with `Verify canonical NNx installation` running exact
  `make verify-nnx-install`;
- add the same exact verifier step after all installs and immediately before `make test` in the
  repository job; and
- extend only the narrow verifier selector with the exact new workflow test names.

Do not alter triggers, permissions, required contexts, action pins, cache files, unrelated jobs,
or the focused Ruff/API test roles.

- [ ] **Step 4: Run workflow mutation tests GREEN**

```bash
pytest -p no:cacheprovider tests/test_verify_repo.py -q -k 'nnx or repository_workflow_contract_tests'
pytest -p no:cacheprovider tests/test_verify_repo.py -q
python - <<'PY'
from pathlib import Path
import yaml
yaml.safe_load(Path('.github/workflows/ci.yml').read_text())
PY
ruff check --no-cache tests/test_verify_repo.py scripts/verify_nnx_install.py
git diff --check
```

Expected: all verifier tests pass, YAML parses, Ruff and diff checks are clean.

- [ ] **Step 5: Mutation-audit the contract**

Using an isolated temporary copy of `.github/workflows/ci.yml`, demonstrate at least these
independent RED mutations before restoring the exact file each time: remove
`--only-binary=thekaveh-nnx`; set `NNX_ALLOW_EDITABLE=1`; move validation before installation;
replace the verifier with `python -c "import nnx"`; and add `continue-on-error: true`. Run the
focused workflow tests for each mutation and record the expected failure in the task report.

- [ ] **Step 6: Commit Task 2**

Mark Task 2 only complete, rerun Steps 4–5, stage only Task 2 files, run
`git diff --cached --check`, and commit:

```bash
git commit -m "ci: enforce canonical NNx wheel validation"
```

Write `.superpowers/sdd/task-2-report.md` with exact RED/GREEN and mutation evidence.

## 12.14.4 Task 3: Synchronize canonical documentation and maintenance records

**Files:**
- Modify: `README.md`
- Modify: `CONTRIBUTING.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/conventions.md`
- Modify: `docs/dependency-contracts.md`
- Modify: `docs/nnx-library.md`
- Modify: `docs/maintenance/overnight-2026-07-04.md`
- Modify: `docs/manifest.yaml`
- Modify: `tests/test_check_docs.py`
- Modify: `tests/test_verify_repo.py`
- Modify: this plan

**Interfaces:**
- Consumes: Task 1 public commands and Task 2 exact job roles.
- Produces: synchronized repository/site/wiki contract documentation and a resolved OM-006 record.

- [ ] **Step 1: Add documentation regressions RED**

Add focused assertions that canonical sources state all of these exact facts:

- `make verify-nnx-install` is the canonical local evidence command;
- `NNX_ALLOW_EDITABLE=1 make test-nnx-surface` is the explicit local-development command;
- editable results are development-surface, not released-wheel evidence;
- both `pytest-repository` and `pytest-nnx-surface` run canonical validation;
- canonical mode rejects any `direct_url.json` and proves distribution-owned import origin;
- binary-only selection is not a cryptographic hash lock and Issue #63 owns that future work; and
- no private `dist._path` example or incorrect top-level `{"editable": true}` claim remains.

Add a manifest inventory assertion for design `12.13` and implementation `12.14`. Add or update an
OM-006 regression requiring status `Resolved` and wording grounded in the implemented verifier,
but do not claim live CI until Task 4 records it.

```bash
pytest -p no:cacheprovider tests/test_check_docs.py tests/test_verify_repo.py -q -k 'nnx_wheel or om_006 or manifest'
```

Expected: fail against stale canonical prose and the missing plan manifest entry.

- [ ] **Step 2: Update canonical documentation**

- `README.md`: keep the concise NNx overview, link to `docs/dependency-contracts.md`, and name the
  explicit editable test invocation.
- `CONTRIBUTING.md`: add canonical verification to local pre-PR steps and label the editable escape
  hatch as non-release evidence.
- `docs/conventions.md`: document the binary-selected install, explicit verifier order, and the
  blocking-vs-focused CI role split.
- `docs/dependency-contracts.md`: replace the `dist._path` snippet with the public Make command;
  define exact pin, direct-URL, owned-file, import-origin, redaction, and editable-mode semantics.
- `docs/nnx-library.md`: update the upstream development and release-validation workflow.
- `docs/maintenance/overnight-2026-07-04.md`: change OM-006 from deferred to resolved by the
  verifier and CI contract; state live evidence is recorded on Issue #58 after rollout.
- `CHANGELOG.md`: add an Unreleased Changed entry for the verifier, two CI jobs, binary selection,
  and validated editable mode.
- `docs/manifest.yaml`: add this plan as `12.14` immediately after the Issue #58 design.

Do not add a diagram, a hash, a new required context, or a claim that the whole dependency graph is
locked.

- [ ] **Step 3: Verify three documentation surfaces GREEN**

```bash
pytest -p no:cacheprovider tests/test_check_docs.py tests/test_verify_repo.py -q -k 'nnx_wheel or om_006 or manifest'
make docs-check
make docs-wiki
python scripts/verify_repo.py --check docs --fast
ruff check --no-cache tests/test_check_docs.py tests/test_verify_repo.py scripts/docs
git diff --check
```

Inspect generated site and wiki copies of the changed canonical pages. Confirm manifest numbering,
link rewrites, and byte-equivalent content. Do not commit generated output.

- [ ] **Step 4: Commit Task 3**

Mark Task 3 only complete, rerun Step 3, stage canonical sources and tests, run
`git diff --cached --check`, and commit:

```bash
git commit -m "docs: document canonical NNx wheel evidence"
```

Write `.superpowers/sdd/task-3-report.md` with exact surface and test evidence.

## 12.14.5 Task 4: Review, verify, publish, close, and clean up

**Files:**
- Modify only for review fixes: Task 1–3 files
- External evidence: GitHub PRs, Actions, Issue #58, parent #53, project item, branches/worktrees

**Interfaces:**
- Consumes: all task commits and reports.
- Produces: reviewed code, live canonical-wheel evidence, GitFlow merges, closed records, and clean
  synchronized state.

- [ ] **Step 1: Complete per-task and broad review**

For each task, generate a review package from that task's recorded base to head, dispatch a fresh
reviewer for specification compliance and code quality, send all Important/Critical findings to one
fix subagent, require covering test evidence, and re-review until clean. After Tasks 1–3 pass,
generate a whole-branch review package from base
`123caef69a2d29d2e0fe46828cd3f73cdc83d11c` and repeat the review/fix/re-review loop.

- [ ] **Step 2: Prove both active local modes**

The user's active environment is intentionally editable. Run and record:

```bash
make verify-nnx-install
NNX_ALLOW_EDITABLE=1 make verify-nnx-install
NNX_ALLOW_EDITABLE=1 make test-nnx-surface
```

Expected: canonical mode rejects the active editable checkout with a redacted diagnostic; explicit
editable mode and the NNx surface suite pass.

Create a temporary Python 3.11 virtual environment outside the repository, install only the exact
NNx wheel without dependencies, and run the repository verifier module from the checkout:

```bash
python3.11 -m venv /private/tmp/ml-eng-lab-issue58-wheel-venv
/private/tmp/ml-eng-lab-issue58-wheel-venv/bin/python -m pip install --only-binary=thekaveh-nnx --no-deps 'thekaveh-nnx[lm]==0.2.0'
PYTHONPATH="$PWD" /private/tmp/ml-eng-lab-issue58-wheel-venv/bin/python -m scripts.verify_nnx_install
```

Expected: canonical-wheel mode passes. Remove only this explicit temporary environment afterward.
Do not alter the user's editable NNx installation.

- [ ] **Step 3: Run fresh final verification**

Because the active environment is editable, use its explicit validated mode for repository tests:

```bash
NNX_ALLOW_EDITABLE=1 make test-nnx-surface
NNX_ALLOW_EDITABLE=1 make test
make lint
make verify
make docs-check
make docs-wiki
git diff --check
git status --short --branch
```

Record exact pass/skip/warning counts and reasons. Confirm no Atlas, JupyterHub, Ollama, ComfyUI,
Docker Compose, or other service was started or contacted.

- [ ] **Step 4: Open and validate feature → develop PR**

Push the reviewed feature branch and open a ready PR targeting `develop` with `Refs #58` rather
than closing the issue before the default-branch release. Include TDD evidence, both local modes,
full verification, documentation surfaces, and the no-service statement. Wait for every applicable
check, specifically green `pytest-repository`, `atlas-consumer-policy`, `pytest-nnx-surface`, and
`atlas-contract`. Capture job URLs, conclusions, durations, and canonical verifier output.

- [ ] **Step 5: Merge through GitFlow and synchronize**

Merge feature → `develop` through its PR and delete the remote feature branch. Open a separate
`develop` → `main` release PR containing `Closes #58`; wait for both required contexts and every
applicable check, then merge normally. If the release merge breaks ancestry, open a content-neutral
`main` → `develop` sync PR, wait for required checks, and merge it. Verify:

```bash
git fetch --prune origin
git diff --exit-code origin/main origin/develop
git merge-base --is-ancestor origin/main origin/develop
```

- [ ] **Step 6: Close records and clean state**

Check every Issue #58 acceptance box, comment with feature/release/sync PRs, final SHAs, local and
live test evidence, editable/canonical mode evidence, and no-service confirmation. Ensure the issue
is closed after `main`, move its project item to Done, mark #58 complete in parent #53, and identify
#59 as next.

Delete the conclusively merged local feature branch, update local `main` and `develop`, and verify:

```bash
git fetch --prune origin
git worktree list
git branch --format='%(refname:short)'
git branch -r --format='%(refname:short)'
gh pr list --state open --json number,title,headRefName,baseRefName
git status --short --branch
git diff --exit-code origin/main origin/develop
git merge-base --is-ancestor origin/main origin/develop
docker ps --format '{{.Names}}'
```

Expected: one clean worktree; only `main` and `develop` locally/remotely; no open PR; synchronized
trees and correct ancestry; no ml-eng-lab, Atlas, JupyterHub, Ollama, or ComfyUI container. Do not
stop or modify unrelated containers.
