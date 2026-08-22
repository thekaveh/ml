# 12.38 Issue 72 NNx Upstream Triage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give all five ml-eng-lab NNx findings an evidence-backed upstream disposition, document every local workaround and affected notebook, and leave only the one genuinely required local migration follow-up.

**Architecture:** Keep `docs/FINDINGS-NNX.md` as the canonical record, with one summary-table row and one detailed disposition per finding. Create three narrowly scoped upstream issues, link the existing resolved regression issue, record the released ReLU-only contract as a non-bug disposition, and create one local follow-up for the blocked NNx/Atlas upgrade. A focused documentation contract test prevents the triage set from drifting.

**Tech Stack:** GitHub CLI/API, Markdown, pytest, MkDocs Material, existing `scripts.docs` projection/check tooling, repository verifier, Ruff.

## 12.38.1 Global Constraints

- Do not modify the NNx repository, push an NNx branch, or open an NNx pull request.
- Keep ml-eng-lab pinned to `thekaveh-nnx[lm]==0.2.0` in this issue.
- Search both open and closed upstream issues before creating a record; link duplicates instead of recreating them.
- Treat released NNx tags and changelog entries as release evidence, not current `main` alone.
- Create exactly one ml-eng-lab follow-up, for the coordinated NNx/Atlas upgrade and diabetes-loader migration.
- Do not start Atlas, Ollama, ComfyUI, or notebook execution; no runtime code or notebook changes are in scope.
- Keep upstream-triage documentation separately rollbackable from future NNx implementation work.

---

## 12.38.2 Task 1: Create the Durable Upstream and Local Triage Records

**Files:**
- No repository files modified.
- External records: `thekaveh/NNx` issues and one `thekaveh/ml-eng-lab` follow-up issue.

**Interfaces:**
- Consumes: Issue #72, upstream issue search results, NNx `v0.2.0`–`v0.2.3` source, and the design's per-finding dispositions.
- Produces: three literal upstream issue URLs and one literal local follow-up URL used by Tasks 2 and 3.

- [ ] **Step 1: Re-run duplicate searches immediately before mutation**

Run title/body searches for `NNDataset` full-batch defaults, regression `EarlyStopping`, and absolute `Run saved` paths across all open and closed `thekaveh/NNx` issues.

Expected: `thekaveh/NNx#81` is the only matching issue among the five finding families; no issue covers the three actionable unresolved contracts.

- [ ] **Step 2: Open the `NNDataset` default/discoverability issue**

Create an upstream `documentation` + `enhancement` issue that records:

- `batch_sizes=(None, None, None)` resolves to one batch per split;
- stochastic workloads may perform only one optimizer step per epoch;
- the source comment and tests know the semantic, while the public class docs expose only the signature;
- a backward-compatible direction is prominent public documentation and examples, with any default change treated separately;
- acceptance covers docstring/API docs, a quickstart example with explicit mini-batches, and a retained full-batch compatibility test;
- downstream evidence is ml-eng-lab Issue #69 and Issue #72.

Expected: `gh issue create` returns a new numbered issue URL under
`https://github.com/thekaveh/NNx/issues/`.

- [ ] **Step 3: Open the regression-safe `EarlyStopping` issue**

Create an upstream `bug` + `enhancement` issue that records:

- the default remains `monitor="val_edp.error"` in `v0.2.3` and `main`;
- regression evaluation points carry `loss` but may have `error=None`;
- `_lookup_monitored()` returns `None`, so the callback silently never advances patience;
- construction-time loss guessing is not viable because the callback has no model context;
- a backward-compatible auto/fallback monitor contract should be explicit and testable;
- explicit `monitor="val_edp.loss"` must remain supported;
- downstream evidence is the diabetes notebook and ml-eng-lab Issue #72.

Expected: `gh issue create` returns a distinct upstream issue URL.

- [ ] **Step 4: Open the portable training-completion message issue**

Create an upstream `bug` + `documentation` issue that records:

- `NNModel.train()` and `Trainer.train()` build `runs_root_path` from `os.getcwd()` and print it;
- this behavior exists in every released tag through `v0.2.3` and current `main`;
- committed notebook output consequently leaks worktree/container paths;
- the message should be relative to the supplied run root/current working directory while saved metadata may retain absolute paths where semantically needed;
- acceptance covers both training entry points and an execution from a non-repository temporary directory;
- downstream evidence is ml-eng-lab verifier rule `E13.stale_active_notebook_path` and Issue #72.

Expected: `gh issue create` returns a third upstream issue URL.

- [ ] **Step 5: Create the single ml-eng-lab follow-up**

Create an issue labeled `dependencies`, `nnx`, `atlas`, and `notebook` whose scope is:

- coordinate the root and Atlas JupyterHub NNx pin beyond `0.2.0`;
- requalify the canonical wheel and complete Tier A/B/C matrix;
- migrate `tabular_regression-diabetes-mlp-pytorch` from its manual float-target `TensorDataset`/`DataLoader` to `NNTabularDataset(target_dtype=torch.float32)` only after the runtime upgrade;
- preserve Issue #61's retained-version decision until the compatibility gate passes;
- link upstream `thekaveh/NNx#81` and ml-eng-lab Issues #61 and #72.

Expected: one new local issue URL; no follow-up issues for the other four findings.

- [ ] **Step 6: Validate external records**

Run `gh issue view` for all four newly created issue numbers and upstream #81.

Expected: all URLs resolve; titles, labels, states, backlinks, and issue bodies match the intended disposition; #81 remains closed.

---

## 12.38.3 Task 2: Add a Failing Canonical-Triage Documentation Contract

**Files:**
- Modify: `tests/test_check_docs.py`

**Interfaces:**
- Consumes: the four literal URLs returned by Task 1 and the five canonical finding headings.
- Produces: `_assert_nnx_finding_triage_docs(text: str) -> None`, a direct test, and mutation cases guarding each disposition.

- [ ] **Step 1: Add the assertion helper and direct test**

Immediately after the existing NNx retained-release documentation tests, add a helper with literal expected URLs from Task 1 and assertions equivalent to:

```python
def _assert_nnx_finding_triage_docs(text: str) -> None:
    headings = (
        "### 9.1.1.1 `NNDataset` default `batch_size` packs the whole train set into one batch",
        "### 9.1.1.2 `nnx.deepen` is function-preserving only for `Activations.RELU`",
        "### 9.1.1.3 `NNTabularDataset` coerces targets to `torch.long` (classification-only)",
        "### 9.1.1.4 `EarlyStopping(monitor=...)` default is `\"val_edp.error\"`, doesn't exist for regression EDPs",
        "### 9.1.1.5 training completion messages expose absolute run paths",
    )
    for heading in headings:
        assert text.count(heading) == 1

    assert "| F1 |" in text
    assert "| F2 |" in text
    assert "| F3 |" in text
    assert "| F4 |" in text
    assert "| F5 |" in text
    assert "thekaveh/NNx/issues/81" in text
    assert "v0.2.2" in text
    assert "documented design constraint" in text
    assert "not an open upstream bug" in text
    assert "`NNModel.train()` and `Trainer.train()`" in text
    assert "`NNRun.save()` prints an absolute path" not in text


def test_nnx_findings_have_durable_upstream_triage():
    text = (REPO_ROOT / "docs/FINDINGS-NNX.md").read_text(encoding="utf-8")
    _assert_nnx_finding_triage_docs(text)
```

Add literal assertions for the three new upstream URLs and the one new local follow-up URL.

- [ ] **Step 2: Add mutation coverage**

Parameterize at least these removals/replacements and require the helper to raise `AssertionError`:

```python
(
    ("| F1 |", "| missing-F1 |"),
    ("thekaveh/NNx/issues/81", "thekaveh/NNx/issues/999999"),
    ("v0.2.2", "unreleased"),
    ("documented design constraint", "unresolved bug"),
    ("`NNModel.train()` and `Trainer.train()`", "`NNRun.save()`"),
)
```

- [ ] **Step 3: Prove the test is red**

Run:

```bash
/private/tmp/ml-eng-lab-issue72-venv/bin/python -m pytest \
  tests/test_check_docs.py::test_nnx_findings_have_durable_upstream_triage -q
```

Expected: FAIL because the status table and new upstream/local URLs do not yet exist and the fifth heading still has the incorrect attribution.

---

## 12.38.4 Task 3: Implement the Canonical Documentation Dispositions

**Files:**
- Modify: `docs/FINDINGS-NNX.md`
- Modify: `docs/nnx-library.md`
- Modify: `docs/manifest.yaml`
- Test: `tests/test_check_docs.py`

**Interfaces:**
- Consumes: Task 1's issue URLs and Task 2's documentation contract.
- Produces: a five-row status table and five normalized detailed finding sections, projected through the existing manifest.

- [ ] **Step 1: Add the status table**

After the introductory paragraphs in `docs/FINDINGS-NNX.md`, add five rows keyed `F1` through `F5`. Each row must state upstream disposition, released status, local workaround/status, affected notebooks, and remaining work.

Use the literal issue URLs created in Task 1, the existing `thekaveh/NNx#81` URL, and the literal local follow-up URL. Mark F2 as a documented design constraint, not an open upstream bug.

- [ ] **Step 2: Normalize all five detailed findings**

Keep the existing numbered headings and technical explanation, but add the standard labels from the design:

```markdown
**Upstream disposition:** ...

**Release evidence:** ...

**Affected notebooks:** ...

**Local workaround/status:** ...

**Remaining work:** ...
```

For F3, separate upstream resolution in `v0.2.2` from unavailability under the retained `0.2.0` consumer pin. For F5, rename the heading and text so the emitters are `NNModel.train()` and `Trainer.train()`, while `NNRun.save()` is only the persistence call.

- [ ] **Step 3: Synchronize the NNx overview summary**

Update `docs/nnx-library.md` §7.5 to summarize each disposition and link readers to the canonical table. Correct the F5 attribution and describe F3 as upstream-resolved/local-pending instead of classification-only without qualification.

- [ ] **Step 4: Declare the design and plan in the documentation manifest**

Append the Issue #72 design and implementation plan to the manifest's
`design-records` children as entries `12.37` and `12.38`, preserving the
existing order and making both records available to the generated site/wiki.

- [ ] **Step 5: Run the focused green tests**

Run:

```bash
/private/tmp/ml-eng-lab-issue72-venv/bin/python -m pytest \
  tests/test_check_docs.py::test_nnx_findings_have_durable_upstream_triage \
  tests/test_check_docs.py::test_nnx_finding_triage_docs_reject_mutations -q
```

Expected: all focused triage tests pass.

- [ ] **Step 6: Commit the triage documentation and guard**

```bash
git add docs/FINDINGS-NNX.md docs/nnx-library.md docs/manifest.yaml tests/test_check_docs.py \
  docs/superpowers/plans/2026-08-22-issue-72-nnx-upstream-triage-implementation-plan.md
git commit -m "docs: record NNx upstream dispositions"
```

---

## 12.38.5 Task 4: Verify Documentation, Links, and the Complete Repository

**Files:**
- Modify only if verification exposes an issue: the smallest file owned by the failed contract.

**Interfaces:**
- Consumes: committed Task 3 documentation and tests.
- Produces: clean documentation, link, repository, and regression evidence at one reviewed feature SHA.

- [ ] **Step 1: Validate every external issue link semantically**

Run `gh issue view` against the three new upstream issues, upstream #81, the new local follow-up, Issues #61/#69/#72, and confirm title/state/repository rather than relying on HTTP status alone.

Expected: every target resolves and supports the adjacent documentation claim.

- [ ] **Step 2: Run focused and complete documentation checks**

```bash
/private/tmp/ml-eng-lab-issue72-venv/bin/python -m pytest tests/test_check_docs.py -q
make PYTHON=/private/tmp/ml-eng-lab-issue72-venv/bin/python docs-check
make PYTHON=/private/tmp/ml-eng-lab-issue72-venv/bin/python docs-build
make PYTHON=/private/tmp/ml-eng-lab-issue72-venv/bin/python docs-wiki
```

Expected: every command exits zero; no generated documentation drift remains.

- [ ] **Step 3: Run repository verification and lint**

```bash
/private/tmp/ml-eng-lab-issue72-venv/bin/python scripts/verify_repo.py --check all --fast
/private/tmp/ml-eng-lab-issue72-venv/bin/python -m ruff check . --no-cache
git diff --check
```

Expected: zero verifier findings, Ruff clean, and no whitespace errors.

- [ ] **Step 4: Run the complete repository suite**

```bash
/private/tmp/ml-eng-lab-issue72-venv/bin/python -m pytest tests -q
```

Expected: at least the 3,037-test clean baseline passes with no failures.

- [ ] **Step 5: Record final evidence and prepare review**

Confirm a clean worktree, capture the reviewed commit SHA, and add a concise Issue #72 comment listing the upstream dispositions, local follow-up, and verification commands. Do not close Issue #72 before GitFlow integration completes.

---

## 12.38.6 Task 5: Integrate and Clean Up Through GitFlow

**Files:**
- No additional repository changes expected.
- External records: feature PR, release PR, sync PR, Issue #72, parent Issue #53, project item.

**Interfaces:**
- Consumes: reviewed, verified feature branch from Task 4.
- Produces: identical `main`/`develop` trees, closed/Done Issue #72, checked parent item, deleted feature branch/worktree, and no disposable issue #72 resources.

- [ ] **Step 1: Push and merge the feature PR to `develop`**

Push `codex/issue-72-nnx-upstream-triage`, open a PR that closes Issue #72 only through the later release merge, wait for all required checks, and merge with a merge commit. Delete the remote feature branch.

- [ ] **Step 2: Merge `develop` to `main`**

Open the release PR, wait for required checks, and merge with a merge commit.

- [ ] **Step 3: Synchronize `main` back to `develop`**

Open and merge the ancestry-sync PR after required checks. Verify `origin/main^{tree}` equals `origin/develop^{tree}` and no PR remains open.

- [ ] **Step 4: Finish issue/project bookkeeping**

Confirm Issue #72 is closed, set its project item to Done, check `#72` in parent Issue #53, and add the feature/release/sync SHAs plus verification evidence to Issue #72. Keep the persistent goal active for Issue #73.

- [ ] **Step 5: Remove disposable resources**

Fast-forward the main checkout's `develop`, remove the clean issue #72 worktree, delete the local feature branch, move the issue #72 virtual environment and upstream research clone to Trash, and verify no issue #72 temporary path or Atlas container remains. Preserve Atlas volumes.
