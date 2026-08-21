# 12.28 Issue 65 Atlas Pin Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the evidence-based Atlas retain decision, synchronize every current pin projection, prove the already-consumed migration live through JupyterHub, and publish Issue #65 through the full GitFlow cycle.

**Architecture:** The `infra` tree entry remains the only Atlas revision authority. A fail-closed repository verifier compares three explicit current-document markers and the dependency ledger with that gitlink, while documentation records the empty current-to-main range and the reviewed historical migration. Static gates run before a live, volume-preserving Atlas check; the final clean feature SHA is then published without any later tracked mutation.

**Tech Stack:** Git submodules; Python 3.11; pytest; PyYAML; Bash/ShellCheck; Docker Compose through Atlas; JupyterHub; host-native Ollama; MkDocs/site/wiki pipeline; GitHub CLI and Actions.

## 12.28.1 Global Constraints

- Retain `infra` exactly at `41ba856f7cd35f0b559d6875e08443eac3e98a98`; Atlas fetched `origin/main` must equal the same SHA.
- Review the historical `61c7c5103660e2226bf107c115dae42bf46f8374..41ba856f7cd35f0b559d6875e08443eac3e98a98` range as 30 first-parent commits and 43 total commits.
- Never edit or commit Atlas source from ml-eng-lab; the parent owns only its gitlink, consumer files, tests, and documentation.
- Preserve `BASE_PORT=auto`, `JUPYTERHUB_SOURCE=container`, and `LLM_PROVIDER_SOURCE=ollama-localhost`.
- Never start or add containerized Ollama or ComfyUI. ComfyUI remains disabled.
- Live shutdown is ordinary `make atlas-down`; never use `COLD=1`, and preserve volumes.
- Keep `thekaveh-nnx[lm]==0.2.0`; Atlas and local/CI Torch stacks remain independent contracts.
- Historical specifications, plans, and released changelog entries keep their historical `61c7c510...` statements.
- Current README, environment guide, runbook, dependency ledger, and Unreleased changelog must describe the current `41ba856f...` contract.
- Any tracked change after the final qualification SHA invalidates all qualification evidence.

---

### 12.28.1.1 Task 1: Enforce current Atlas pin projections

**Files:**
- Modify: `scripts/verify_repo.py`
- Modify: `tests/test_verify_repo.py`
- Modify: `tests/test_atlas_consumer_contract.py`
- Modify: `README.md`
- Modify: `docs/env-setup.md`
- Modify: `docs/atlas-pin-bump-runbook.md`

**Interfaces:**
- Consumes: the staged gitlink returned by `git ls-files --stage -- infra`.
- Produces: `_atlas_current_pin_projection_findings(repo: Path, *, gitlink_sha: str) -> list[Finding]` and the exact marker `Current reviewed Atlas pin: \`<40-lowercase-hex>\`.` in three current documents.

- [ ] **Step 1: Add RED verifier tests**

Add a fixture helper and focused tests to `tests/test_verify_repo.py`:

```python
_ATLAS_CURRENT_PIN_DOCS = (
    "README.md",
    "docs/env-setup.md",
    "docs/atlas-pin-bump-runbook.md",
)


def _write_current_atlas_pin_docs(repo: Path, sha: str) -> None:
    marker = f"Current reviewed Atlas pin: `{sha}`.\n"
    for relative in _ATLAS_CURRENT_PIN_DOCS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker, encoding="utf-8")


def test_atlas_current_pin_projection_matches_gitlink(tmp_path):
    sha = "41ba856f7cd35f0b559d6875e08443eac3e98a98"
    _write_current_atlas_pin_docs(tmp_path, sha)
    findings = _load_verify_module()._atlas_current_pin_projection_findings(
        tmp_path, gitlink_sha=sha
    )
    assert findings == []


@pytest.mark.parametrize("relative", _ATLAS_CURRENT_PIN_DOCS)
def test_atlas_current_pin_projection_rejects_one_stale_surface(tmp_path, relative):
    sha = "41ba856f7cd35f0b559d6875e08443eac3e98a98"
    _write_current_atlas_pin_docs(tmp_path, sha)
    path = tmp_path / relative
    path.write_text(
        path.read_text(encoding="utf-8").replace(sha, "61c7c5103660e2226bf107c115dae42bf46f8374"),
        encoding="utf-8",
    )
    findings = _load_verify_module()._atlas_current_pin_projection_findings(
        tmp_path, gitlink_sha=sha
    )
    assert [finding.id for finding in findings] == ["D10.atlas_current_pin_projection"]
    assert findings[0].location == relative
```

Add separate tests for a missing marker, malformed SHA, duplicate marker, and a partial document set. The missing-all fixture is the only accepted no-op so unrelated minimal dependency-ledger fixtures remain isolated. Add a real-repository control that passes the actual staged gitlink.

- [ ] **Step 2: Run RED**

Run:

```bash
PYTEST_ADDOPTS='-p no:cacheprovider -W error' \
  python -m pytest tests/test_verify_repo.py \
  -k 'atlas_current_pin_projection' -q
```

Expected: collection or assertion failures because `_atlas_current_pin_projection_findings` and the exact current markers do not exist.

- [ ] **Step 3: Implement the fail-closed projection check**

Add to `scripts/verify_repo.py`:

```python
_ATLAS_CURRENT_PIN_DOCUMENTS = (
    "README.md",
    "docs/env-setup.md",
    "docs/atlas-pin-bump-runbook.md",
)
_ATLAS_CURRENT_PIN_MARKER_RE = re.compile(
    r"^Current reviewed Atlas pin: `([0-9a-f]{40})`\.[ \t]*$",
    re.MULTILINE,
)


def _atlas_current_pin_projection_findings(
    repo: Path, *, gitlink_sha: str
) -> list[Finding]:
    present = [relative for relative in _ATLAS_CURRENT_PIN_DOCUMENTS if (repo / relative).exists()]
    if not present:
        return []
    findings: list[Finding] = []
    for relative in _ATLAS_CURRENT_PIN_DOCUMENTS:
        path = repo / relative
        matches = _ATLAS_CURRENT_PIN_MARKER_RE.findall(_read_text(path)) if path.exists() else []
        if len(matches) != 1 or matches[0] != gitlink_sha:
            findings.append(Finding(
                id="D10.atlas_current_pin_projection",
                check="docs",
                severity="error",
                location=relative,
                message="current Atlas pin projection must occur once and equal the infra gitlink",
                detail={"matches": matches, "gitlink_sha": gitlink_sha},
            ))
    return findings
```

At the end of the successful gitlink parse in `_dependency_ledger_findings`, extend the findings with this helper regardless of whether the ledger SHA equals the gitlink:

```python
findings.extend(
    _atlas_current_pin_projection_findings(repo, gitlink_sha=gitlink_sha)
)
```

Do not search historical documentation and do not accept prefix, uppercase, or abbreviated SHAs.

- [ ] **Step 4: Add the three exact current markers**

Add this sentence exactly once to the current Atlas section in each owned document:

```markdown
Current reviewed Atlas pin: `41ba856f7cd35f0b559d6875e08443eac3e98a98`.
```

Replace the stale current-pin prose in those same sections. Do not change `CHANGELOG.md` or historical specs/plans in this task.

- [ ] **Step 5: Strengthen the focused consumer contract**

In `tests/test_atlas_consumer_contract.py`, define:

```python
CURRENT_PIN_DOCUMENTS = (
    "README.md",
    "docs/env-setup.md",
    "docs/atlas-pin-bump-runbook.md",
    "docs/dependency-contracts.md",
)
```

Extend the real gitlink test to require `PINNED_ATLAS_REVISION` in every path and add a mutation test replacing it independently in each current document. Keep the byte-exact NLP projection tests unchanged.

- [ ] **Step 6: Run GREEN and mutations**

Run:

```bash
PYTEST_ADDOPTS='-p no:cacheprovider -W error' python -m pytest \
  tests/test_verify_repo.py tests/test_atlas_consumer_contract.py \
  -k 'atlas_current_pin_projection or atlas_submodule or atlas_nlp_projection' -q
ruff check scripts/verify_repo.py tests/test_verify_repo.py tests/test_atlas_consumer_contract.py
git diff --check
```

Expected: all selected tests pass. Independently replacing the marker in each current document, duplicating it, deleting it, or changing one hex digit must make a selected test fail.

- [ ] **Step 7: Commit**

```bash
git add -- \
  scripts/verify_repo.py \
  tests/test_verify_repo.py \
  tests/test_atlas_consumer_contract.py \
  README.md \
  docs/env-setup.md \
  docs/atlas-pin-bump-runbook.md
git commit -m "test: enforce current Atlas pin projections"
```

---

### 12.28.1.2 Task 2: Record the migration review and retain decision

**Files:**
- Modify: `docs/dependency-contracts.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/notebook-infrastructure.md`
- Modify: `tests/test_check_docs.py`

**Interfaces:**
- Consumes: the two exact ranges and classifications in design §12.27.3-§12.27.4.
- Produces: one current Issue #65 ledger record and current three-surface assertions; it does not claim new live runtime evidence yet.

- [ ] **Step 1: Add RED current-document assertions**

Add to `tests/test_check_docs.py`:

```python
_ISSUE65_CURRENT_DOC_PATHS = (
    "README.md",
    "CHANGELOG.md",
    "docs/env-setup.md",
    "docs/atlas-pin-bump-runbook.md",
    "docs/dependency-contracts.md",
    "docs/notebook-infrastructure.md",
)


def _assert_issue65_current_docs(documents: Mapping[str, str]) -> None:
    current = "41ba856f7cd35f0b559d6875e08443eac3e98a98"
    rollback = "61c7c5103660e2226bf107c115dae42bf46f8374"
    marker = f"Current reviewed Atlas pin: `{current}`."
    for path in ("README.md", "docs/env-setup.md", "docs/atlas-pin-bump-runbook.md"):
        assert documents[path].count(marker) == 1, path
    ledger = documents["docs/dependency-contracts.md"]
    assert "Issue #65 Atlas pin review" in ledger
    assert "30 first-parent commits" in ledger
    assert "43 total commits" in ledger
    assert "empty current-to-main range" in ledger
    assert "Torch 2.13.0" in ledger
    assert "fastmcp==3.4.4" in ledger
    assert "managed-host-process" in ledger
    assert "ComfyUI remains disabled" in ledger
    assert current in ledger and rollback in ledger
    unreleased = documents["CHANGELOG.md"].split("## [0.1.0]", 1)[0]
    assert "Issue #65" in unreleased and current in unreleased
    assert "Issue #65" not in documents["docs/notebook-infrastructure.md"]
```

The last assertion replaces the stale ownership statement with a completed
retain contract rather than leaving Issue #65 pending. Add parametrized
mutations for every exact count/identity/classification and for a stale
`Issue #65 owns` sentence.

- [ ] **Step 2: Run RED**

Run:

```bash
PYTEST_ADDOPTS='-p no:cacheprovider -W error' \
  python -m pytest tests/test_check_docs.py -k issue65 -q
```

Expected: failures for the absent migration ledger and Unreleased entry.

- [ ] **Step 3: Update the dependency ledger**

Replace §6.1.7 with a current record that includes:

```markdown
### Issue #65 Atlas pin review

Current-to-main result: empty range at
`41ba856f7cd35f0b559d6875e08443eac3e98a98`.

Historical migration reviewed:
`61c7c5103660e2226bf107c115dae42bf46f8374..41ba856f7cd35f0b559d6875e08443eac3e98a98`
(`30` first-parent commits; `43` total commits).
```

Add a compact classification table for JupyterHub/Torch/PyG/NLP, FastMCP/MCP,
track synthesis, managed-host-process, host Ollama doctors, and ComfyUI. State
the compatibility consequence and explicitly record the rollback SHA. Preserve
the exact existing `Current Atlas \`infra\` gitlink SHA:` line for D10.

- [ ] **Step 4: Synchronize current narrative surfaces**

Add an Unreleased Issue #65 bullet that states the retain decision and exact
SHA. Replace `Atlas runtime ownership remains Issue #65` in
`docs/notebook-infrastructure.md` with the completed retained-pin contract and
leave Issue #66 manual-only language unchanged.

Do not change the historical Atlas migration bullet under a released section
and do not rewrite historical plans/specifications.

- [ ] **Step 5: Run docs GREEN**

Run:

```bash
PYTEST_ADDOPTS='-p no:cacheprovider -W error' python -m pytest \
  tests/test_check_docs.py tests/test_build_docs.py tests/test_wiki.py -q
make docs-check
make docs-wiki
python -m scripts.verify_repo --check docs --fast
git diff --check
```

Expected: all tests pass, strict MkDocs emits no warning, and wiki generation
reports the complete manifest entry count.

- [ ] **Step 6: Commit**

```bash
git add -- \
  docs/dependency-contracts.md \
  CHANGELOG.md \
  docs/notebook-infrastructure.md \
  tests/test_check_docs.py
git commit -m "docs: record Issue 65 Atlas retain decision"
```

---

### 12.28.1.3 Task 3: Prove the non-live consumer boundary

**Files:**
- Create ignored evidence: `.superpowers/sdd/issue65-evidence/non-live.json`
- Modify only if a test exposes a scoped defect: files already owned by Tasks 1-2

**Interfaces:**
- Consumes: committed Task 1-2 current state and the fresh Python 3.11 environment.
- Produces: clean non-live evidence and a reviewed pre-live commit boundary.

- [ ] **Step 1: Re-prove exact Atlas identities**

Run:

```bash
test "$(git rev-parse HEAD:infra)" = 41ba856f7cd35f0b559d6875e08443eac3e98a98
git -C infra fetch origin main --tags --prune
test "$(git -C infra rev-parse origin/main)" = 41ba856f7cd35f0b559d6875e08443eac3e98a98
test "$(git -C infra rev-list --first-parent --count 61c7c5103660e2226bf107c115dae42bf46f8374..41ba856f7cd35f0b559d6875e08443eac3e98a98)" = 30
test "$(git -C infra rev-list --count 61c7c5103660e2226bf107c115dae42bf46f8374..41ba856f7cd35f0b559d6875e08443eac3e98a98)" = 43
test -z "$(git -C infra status --porcelain)"
```

- [ ] **Step 2: Run the complete non-live gate**

Run under the fresh canonical Python 3.11 environment:

```bash
make atlas-setup
make atlas-contract
make test-atlas-consumer
PYTEST_ADDOPTS='-p no:cacheprovider -W error' make test
make verify
make lint
make docs-check
make docs-wiki
shellcheck scripts/atlas-up.sh scripts/atlas-down.sh scripts/atlas-connect.sh scripts/lib/atlas-dotenv.sh
```

Expected: every command exits 0; pytest has zero failures/errors and only
explicitly documented skips; verifier has zero findings; parent and submodule
remain clean.

- [ ] **Step 3: Run mutation controls**

In a temporary copy, independently replace each current marker with the
rollback SHA, delete the ledger current line, add `branch = main` to
`.gitmodules`, set `LLM_PROVIDER_SOURCE=ollama-container-cpu`, and set
`COMFYUI_SOURCE=container-cpu`. Require the appropriate focused test or
verifier to fail for each source-altered mutation. The clean copy must pass.

- [ ] **Step 4: Record ignored non-live evidence**

The JSON object must contain exact HEAD/tree/gitlink/Atlas-main identities,
range counts, command exit codes, pytest totals, verifier totals, ShellCheck
targets, and clean parent/submodule statuses. Reject absolute home/tmp paths,
tokens, URLs containing credentials, and environment dumps.

- [ ] **Step 5: Review the tracked range**

Review `develop..HEAD` for spec compliance, historical/current scoping, test
non-vacuity, and rollback safety. Resolve every finding in a separate tracked
commit and rerun the affected gates before live validation.

---

### 12.28.1.4 Task 4: Run live Atlas JupyterHub validation and record current evidence

**Files:**
- Modify: `docs/dependency-contracts.md`
- Modify: `tests/test_check_docs.py`
- Create ignored evidence: `.superpowers/sdd/issue65-evidence/live-atlas.json`

**Interfaces:**
- Consumes: exact clean gitlink/main identity and non-live green boundary.
- Produces: observed JupyterHub runtime evidence, volume-preserving shutdown proof, and the last tracked documentation commit before freeze.

- [ ] **Step 1: Preflight the host and project namespace**

Require the host-native Ollama loopback health endpoint to answer. Record only
status and port, never model prompts or environment values. Resolve the exact
Atlas project name from the materialized environment and require no existing
task-owned container. Prove that any ambient Ollama process is host-native,
not Docker-owned.

- [ ] **Step 2: Start the reviewed consumer**

Run:

```bash
make atlas-up
```

Require the lifecycle wrapper to use `--track ml-eng`, the exact consumer
manifest/overlay, `BASE_PORT=auto`, and host-native Ollama. Capture the resolved
Compose service inventory. Require JupyterHub running/healthy and reject any
service name/image matching Ollama or ComfyUI.

- [ ] **Step 3: Run mounted runtime probes**

Inside the exact JupyterHub container, run:

```bash
cd /home/jovyan/work/ml-eng-lab
python scripts/atlas_runtime_probe.py
python -m jupyter nbconvert --execute --to notebook \
  --ExecutePreprocessor.kernel_name=python3 \
  --output /tmp/issue65-cell-output.ipynb \
  /tmp/issue65-cell-input.ipynb
```

Parse the runtime probe JSON and require zero failed mandatory capabilities,
the exact mounted repository, NNx 0.2.0, the exact Issue #64 NLP asset
identities, and successful required notebook imports. Before the nbconvert
command, create `/tmp/issue65-cell-input.ipynb` with `nbformat` as one code cell
containing `import nnx, torch, torch_geometric, fastmcp, spacy, nltk;
print("atlas-cell-ok")`. Capture Python, Torch, torchvision, PyG, FastMCP,
spaCy, and NLTK versions. The executed notebook must contain exactly one stream
line, `atlas-cell-ok`; remove both temporary notebooks afterward and do not
print a token-bearing Jupyter URL.

- [ ] **Step 4: Stop without data loss**

Run:

```bash
make atlas-down
```

Require the exact task project to have no running container, require volumes
still present, require any preexisting host Ollama state unchanged, and require
clean parent/submodule status. Never run `COLD=1`.

- [ ] **Step 5: Update current runtime evidence with observed values**

Replace the stale 2026-07-30 §6.1.8 table in
`docs/dependency-contracts.md` with the exact observed live values and date.
State that Atlas Torch 2.13 is independent from local/CI Torch 2.11. Extend
`_assert_issue65_current_docs` with the exact observed values. Do not soften a
failed import into availability evidence.

- [ ] **Step 6: Verify and commit the live evidence projection**

Run:

```bash
PYTEST_ADDOPTS='-p no:cacheprovider -W error' python -m pytest \
  tests/test_check_docs.py tests/test_atlas_runtime_probe.py -q
make docs-check
make docs-wiki
make verify
git diff --check
git add -- docs/dependency-contracts.md tests/test_check_docs.py
git commit -m "docs: record current Atlas runtime evidence"
```

The ignored `live-atlas.json` must bind the pre/post host state, exact project,
container image/architecture, runtime-probe hash/result, cheap-cell result,
volume inventory hash, shutdown result, and clean status without secret or
machine-local path leakage.

---

### 12.28.1.5 Task 5: Freeze, qualify, and publish Issue 65

**Files:**
- Create ignored evidence: `.superpowers/sdd/issue65-evidence/`
- Create ignored report: `.superpowers/sdd/issue65-qualification-report.md`
- No tracked file may change after `FINAL_SHA` is selected.

**Interfaces:**
- Consumes: independently reviewed clean Task 1-4 commits.
- Produces: immutable qualification/report, merged feature/release/sync PRs, synchronized branches, cleanup, and closed Issue #65.

- [ ] **Step 1: Final pre-freeze review and freeze**

Review the complete feature range. Require zero Critical/Important/Minor
findings, clean parent/submodule/index, exact manifest/site/wiki parity, and
exact `41ba856f...` gitlink/Atlas-main identity. Set `FINAL_SHA=$(git rev-parse
HEAD)` and `FINAL_TREE=$(git rev-parse HEAD^{tree})`. Any later tracked change
restarts Task 5.

- [ ] **Step 2: Fresh detached qualification**

Create a new `/private/tmp/ml-eng-lab-issue65-final.*` root with a detached
worktree at `FINAL_SHA`, fresh Python 3.11 environment/cache, and exact locally
populated gitlink. Install the canonical locked stack and run:

```bash
python -m pip check
make verify-torch-stack
make verify-nnx-install
make test-atlas-consumer
PYTEST_ADDOPTS='-p no:cacheprovider -W error' make test
make verify
make lint
make docs-check
make docs-wiki
make audit-advisories
```

Parse JUnit and require zero failures/errors/undocumented skips. Re-run the
live Task 4 flow from the frozen checkout, then stop normally and prove clean
state. Do not reuse Task 4's mutable development evidence as final evidence.

- [ ] **Step 3: Build and validate the immutable report**

Record final SHA/tree, unchanged gitlink, fetched Atlas main, both range
counts, migration classifications, package/runtime evidence, tests/static/docs
totals, live container/service inventory, host-native Ollama boundary,
volume-preserving shutdown, and clean status. Hash every referenced evidence
file. Reject credentials, token-bearing URLs, home/tmp paths, and environment
dumps.

- [ ] **Step 4: Publish and qualify the feature PR**

Push exact `FINAL_SHA` to `codex/issue-65-atlas-pin-review`. Create one ready
PR to `develop` with the Issue #65 title/body and Atlas/dependency labels, then
apply `tier-b-smoke` before selecting CI. Run one explicit `workflow_dispatch`
at the exact SHA and require its ten-job set to complete successfully. For the
selected labeled/synchronize PR event, require CI, Docs, and Atlas runs to bind
the exact source SHA and current synthetic merge tree; Tier B must pass and
Tier C is the only intentional PR skip. Merge only after every applicable
check is green and the synthetic tree equals `FINAL_TREE`.

- [ ] **Step 5: Publish release and content-neutral sync PRs**

Create the exact `develop` -> `main` release PR, require selected CI/Docs/Atlas
evidence and synthetic tree equality, then merge. Require the exact new Pages
run's build/deploy/wiki jobs to pass and live site/wiki text to contain the
current Atlas SHA and retain decision. If ancestry requires it, create one
content-neutral `main` -> `develop` sync PR, require tree equality and its
applicable CI, merge, and wait for the exact final develop push CI. Final
`main` and `develop` trees must equal `FINAL_TREE` and main must be an ancestor
of develop.

- [ ] **Step 6: Publish report, preserve state, and clean up**

Post the immutable report with its SHA-256 to the feature PR, release PR, and
Issue #65. Recompare exact substantive snapshots of Issues #53 and #66 and
require both OPEN. Recompare the protected ruleset semantically and require its
unchanged update identity. Remove only the exact Issue #65 feature branch,
remote-tracking ref, `refs/issue65/*`, detached worktrees, task images, and task
containers; preserve Atlas volumes and unrelated host processes. Fast-forward
the primary checkout to final `origin/develop` and require clean exact gitlink.

- [ ] **Step 7: Complete the issue as the final mutation**

Post one concise Issue #65 completion note to Issue #53 and prove Issue #53
remains OPEN. Requery the unique Issue #65 project item, Status field, current
option, and Done option. Set only Status to Done, requery Done, then close Issue
#65 with reason `completed`. If project automation closes it, verify
CLOSED/COMPLETED and do not reopen or add a later mutation.
