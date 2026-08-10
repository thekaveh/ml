# 12.7 Issue 54 documentation reconciliation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` to implement this plan task-by-task. Every task receives an independent requirements and quality review before integration.

**Goal:** Correct every factual and stale-documentation finding in GitHub issue #54 and prove that the repository, generated MkDocs site, and native GitHub wiki remain synchronized.

**Architecture:** Preserve the existing single-source documentation architecture. Edit only committed canonical Markdown and records; let `scripts.docs.build_docs` project those sources into the gitignored site and wiki trees, then use the existing checker, strict MkDocs build, and focused tests as parity gates.

**Tech Stack:** Markdown, YAML, Python 3.11, pytest, MkDocs Material, the existing `scripts.docs` pipeline, GitHub Pages, and the native GitHub wiki.

## 12.7.1 Global constraints

- The repository uses feature → `develop` → `main` GitFlow; no implementation commits go directly to `develop` or `main`.
- Canonical sources are committed; `generated/site/`, `generated/wiki/`, root `mkdocs.yml`, and `site/` remain generated and gitignored.
- Repository, site, and wiki surfaces must be self-contained and content-equivalent after documented transforms.
- The README and `docs/index.md` retain the identical opener tagline and executive-summary opening sentence.
- Documentation claims must be grounded in current source, notebook, configuration, test, or preserved-output evidence.
- Historical records remain historical: correct their present-tense status without rewriting what was true at the recorded time.
- Do not re-execute expensive notebooks solely to support a blanket documentation claim.
- Do not start Atlas for this issue. Never run containerized Ollama or ComfyUI.
- Generated documentation trees and root `mkdocs.yml` must not be committed.

---

## 12.7.2 Task 1: Correct current user-facing factual claims

**Files:**
- Modify: `docs/concepts.md:66-69`
- Modify: `docs/notebooks/node_classification-reddit-gnn-pyg.md:342-345,391-394`
- Modify: `notebooks/image_classification-mnist-ffnn-pytorch/README.md:60-63`
- Modify: `docs/notebooks/image_classification-mnist-ffnn-pytorch.md:399-414`
- Modify: `README.md:235-243`

**Interfaces:**
- Consumes: the sentiment task specification and notebook loss function; the Reddit task README and preserved result table; the shipped NNx EarlyStopping API; the intentional Reddit Phase-2 output exception.
- Produces: current, mutually consistent canonical prose that the docs generator can project without special cases.

- [ ] **Step 1: Verify each claimed contradiction against its authoritative source**

Run:

```bash
rg -n "cross.?entropy|output_dim|num_classes" notebooks/sentiment-analysis-lstm-pytorch docs/notebooks/sentiment-analysis-lstm-pytorch.md
rg -n "depth-3|0\.9164|0\.0904" notebooks/node_classification-reddit-gnn-pyg/README.md docs/notebooks/node_classification-reddit-gnn-pyg.md
rg -n "class EarlyStopping|EarlyStopping" docs notebooks tests requirements.txt
rg -n '"execution_count": null' notebooks/node_classification-reddit-gnn-pyg/phase2_advanced/4_deeper_model.ipynb
```

Expected: sentiment is three-class cross-entropy; the Reddit README already agrees with the preserved depth-3 result; EarlyStopping exists in the consumed NNx contract; the documented Phase-2 notebook exception has unexecuted cells.

- [ ] **Step 2: Correct the sentiment and Reddit descriptions**

Change the BCE paragraph so sentiment is not named as a binary case; retain link prediction as the grounded BCE example. Remove the obsolete Reddit discrepancy note and the completed reconciliation extension without changing preserved metrics or experimental interpretation.

- [ ] **Step 3: Correct the EarlyStopping future-work status**

Keep the early-stopping example as useful future work, but remove language claiming that it awaits an upstream primitive. In the deep dive, describe it as adoption of the available callback and avoid promising a specific wall-time reduction without execution evidence.

- [ ] **Step 4: Qualify committed-output coverage**

Replace the README's blanket executed-cell statement with a precise rule: active notebooks normally retain outputs, while documented long-running or preservation exceptions may intentionally omit them. Retain the stronger Tier-C in-place execution warning.

- [ ] **Step 5: Run focused validation**

Run:

```bash
python -m scripts.docs.build_docs --site --wiki
pytest -p no:cacheprovider tests/test_check_docs.py tests/test_build_docs.py tests/test_wiki.py -q
python scripts/verify_repo.py --check docs --fast
git diff --check
```

Expected: all commands pass, generated projections contain the corrected wording, and no unrelated notebook source/output changes appear.

- [ ] **Step 6: Commit Task 1**

```bash
git add README.md docs/concepts.md docs/notebooks/node_classification-reddit-gnn-pyg.md docs/notebooks/image_classification-mnist-ffnn-pytorch.md notebooks/image_classification-mnist-ffnn-pytorch/README.md
git commit -m "docs: correct notebook and loss claims"
```

---

## 12.7.3 Task 2: Disposition completed and historical records accurately

**Files:**
- Modify: `docs/superpowers/plans/2026-08-01-opener-visual-remediation-implementation-plan.md:404-462`
- Modify: `CHANGELOG.md:22-30`
- Modify: `docs/maintenance/overnight-2026-07-04.md:587-588,629`

**Interfaces:**
- Consumes: merged PRs #51 and #52, current Tier-A temporary-output CI, current Pages/wiki publication, and the Atlas migration that removed `vendor/genai-vanilla`.
- Produces: historical records that preserve chronology while no longer presenting resolved work as pending.

- [ ] **Step 1: Ground every completion status**

Run:

```bash
gh pr view 51 --repo thekaveh/ml-eng-lab --json state,mergedAt,mergeCommit,url
gh pr view 52 --repo thekaveh/ml-eng-lab --json state,mergedAt,mergeCommit,url
rg -n "/tmp/ml-tier-a|tracked source notebooks|check-tier-a-clean" .github/workflows/ci.yml Makefile CHANGELOG.md
rg -n "vendor/genai-vanilla|infra/|github-pages|push_wiki" .gitmodules README.md docs .github/workflows
```

Expected: PRs #51/#52 are merged; Tier-A CI now writes execution artifacts to `/tmp`; Pages/wiki publication and Atlas migration supersede the cited deferrals.

- [ ] **Step 2: Complete the opener implementation record**

Mark Task 6 and its six steps complete. Add a concise completion note with the merged PR numbers and publication outcome. Do not alter commands as though they had been planned differently.

- [ ] **Step 3: Preserve but time-bound the v0.1.0 known issue**

Rewrite the present-tense `currently red` sentence as a historical v0.1.0 limitation and point readers to the Unreleased temporary-output design that superseded the byte-diff gate.

- [ ] **Step 4: Annotate resolved maintenance deferrals**

For OM-011, OM-012, and OM-053, retain the original finding and validation text but change status/disposition so later backlog extraction cannot interpret them as open. Name the resolving release/migration evidence without erasing historical context.

- [ ] **Step 5: Run focused validation**

Run:

```bash
python -m scripts.docs.build_docs --site --wiki
pytest -p no:cacheprovider tests/test_manifest.py tests/test_check_docs.py tests/test_build_docs.py tests/test_wiki.py -q
python scripts/verify_repo.py --check docs --fast
git diff --check
```

Expected: all commands pass and every generated surface carries the updated historical disposition.

- [ ] **Step 6: Commit Task 2**

```bash
git add CHANGELOG.md docs/maintenance/overnight-2026-07-04.md docs/superpowers/plans/2026-08-01-opener-visual-remediation-implementation-plan.md
git commit -m "docs: close stale maintenance records"
```

---

## 12.7.4 Task 3: Audit, review, publish, and record completion

**Files:**
- Modify: any canonical documentation file only if the complete A-L audit exposes an issue in scope.
- Modify: `docs/superpowers/plans/2026-08-10-issue-54-docs-reconciliation-implementation-plan.md`

**Interfaces:**
- Consumes: reviewed Tasks 1–2 and the complete three-surface documentation audit checklist.
- Produces: a clean audit report, verified generated projections, reviewed GitFlow PRs, live Pages/wiki publication, and a completed implementation record.

- [ ] **Step 1: Run the complete three-surface audit**

Apply checklist A–L from `three-surface-docs-audit`, including single-source generation, self-containment, wiki configuration, CI, diagrams, numbering, completeness, strict build, reproducibility separation, content grounding, and project-opening parity.

- [ ] **Step 2: Correct every in-scope audit finding through the same review loop**

For each finding, record file:line evidence, apply the smallest canonical-source correction, regenerate both projections, and rerun the focused checker that exposed it. Do not hand-edit generated trees.

- [ ] **Step 3: Run the complete local gate**

Run:

```bash
make docs-check
make docs-wiki
pytest -p no:cacheprovider tests/ -q
ruff check --no-cache .
python scripts/verify_repo.py --check all --fast
git diff --check
git status --short
```

Expected: zero documentation findings, zero strict-build warnings, all repository tests pass with only documented platform skips, lint passes, and only intended canonical files are changed before the final record commit.

- [ ] **Step 4: Complete independent requirements and quality reviews**

Review the full branch against issue #54 and this plan. Fix all Critical or Important findings, rerun affected tests, then rerun the complete gate.

- [ ] **Step 5: Mark this implementation record complete**

Set all completed checkboxes to `[x]` and add the merged feature, release, and synchronization PRs after they exist. The final record update may be completed during GitFlow bookkeeping if PR numbers are not yet available at the pre-PR commit.

- [ ] **Step 6: Complete GitFlow and publication**

Open and merge the reviewed feature PR into `develop`, then the `develop` → `main` PR. If the merge commit leaves branch SHAs different, merge a no-content `main` → `develop` synchronization PR. Verify Pages and wiki workflows and confirm the live surfaces contain the corrected wording.

- [ ] **Step 7: Clean final state**

Delete only the conclusively merged feature branch and its clean worktree if any, prune remote references, update local `main` and `develop`, and confirm no open PR, dangling merged branch, dirty worktree, or ml-eng-lab Atlas container remains.
