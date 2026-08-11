# 12.8 Issue 55 security policy implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` to implement this plan task-by-task. Apply test-driven development to the documentation-pipeline behavior change, and complete independent requirements and quality reviews before integration.

**Goal:** Add a native root security policy with an actionable private-reporting route, evidence-backed support and disclosure expectations, explicit dependency handling, and synchronized repository, site, and wiki publication.

**Architecture:** Make root `SECURITY.md` the single canonical policy source so GitHub discovers it natively and the existing manifest projects that same file into the MkDocs site and native wiki. Generalize the documentation and repository verifiers to treat manifest-declared root Markdown as canonical input: accept it in completeness checks, scan it for cross-surface coupling and broken links, and keep the focused documentation workflow responsive to policy changes. Continue requiring every canonical `docs/**/*.md` file to be declared and every declared source to exist.

**Tech Stack:** Markdown, YAML, Python 3.11, pytest, MkDocs Material, GitHub private vulnerability reporting, the existing `scripts.docs` pipeline, GitHub Pages, and the native GitHub wiki.

## 12.8.1 Global constraints

- Use feature → `develop` → `main` GitFlow and a final `main` → `develop` synchronization PR when required.
- `SECURITY.md` is the canonical policy; do not create a manually duplicated `docs/security.md`.
- Preserve strict single-source, completeness, numbering, self-containment, and deterministic generation guarantees.
- Keep promises proportional to actual practice: no fixed acknowledgement, remediation, disclosure, support-window, bounty, CVE, or backport guarantee.
- Treat current `main` as the maintained security line; older commits and tags are historical unless explicitly stated otherwise.
- Do not claim dependency audit automation, a fully locked dependency graph, or a current vulnerability count without fresh evidence.
- Report upstream-only NNx or Atlas vulnerabilities upstream, while accepting cross-boundary reports here when this consumer configuration makes them reachable.
- Do not start Atlas. Never run containerized Ollama or ComfyUI.
- Generated documentation trees, `mkdocs.yml`, and `site/` remain generated and uncommitted.

---

## 12.8.2 Task 1: Admit and verify root canonical Markdown

**Files:**
- Modify: `tests/test_check_docs.py`
- Modify: `tests/test_verify_repo.py`
- Modify: `scripts/docs/check_docs.py`
- Modify: `scripts/verify_repo.py`
- Modify: `.github/workflows/docs.yml`

**Interfaces:**
- Consumes: manifest-declared section sources and the canonical `docs/**/*.md` inventory.
- Produces: completeness, self-containment, link, and workflow checks that accept and validate manifest-declared root Markdown without weakening the requirement that every file under `docs/` is manifest-indexed.

- [x] **Step 1: Write a failing regression**

Add fixtures with `SECURITY.md` as a manifest section source plus the existing `docs/index.md` and notebook documentation. Assert completeness accepts an existing declared root source, manifest loading still rejects a missing declared root source, an unmanifested `docs/extra.md` still fails, a forbidden site/wiki link in the root policy is caught by repository self-containment, a broken internal link in the root policy is caught by the repository verifier, and the focused documentation workflow watches `SECURITY.md`.

- [x] **Step 2: Verify the regression fails for the intended reason**

Run:

```bash
pytest -p no:cacheprovider tests/test_check_docs.py -q
```

Expected before production changes: the new tests fail because completeness compares all declared sources only to `docs/**/*.md`, the self-containment and repository scanners omit manifest-declared root policy files, and the workflow does not watch `SECURITY.md`.

- [x] **Step 3: Implement the minimal pipeline correction**

Validate declaration coverage against `docs/**/*.md` while allowing other manifest-declared Markdown paths to rely on `load_manifest`'s existing file-existence check. Include manifest-declared root Markdown in repository self-containment scans, include `SECURITY.md` in the verifier's in-scope text files, and add it to the focused docs workflow path filter. Do not add a broad root-file allowlist or weaken unmanifested canonical-doc detection.

- [x] **Step 4: Verify green and commit**

Run the focused test files, then `tests/test_manifest.py`, `tests/test_build_docs.py`, and `tests/test_wiki.py`. Commit only the tests, checkers, and workflow change with message `docs: support root canonical manifest sources`.

---

## 12.8.3 Task 2: Publish the repository security policy

**Files:**
- Create: `SECURITY.md`
- Modify: `docs/manifest.yaml`
- Modify: `README.md`
- Modify: `CONTRIBUTING.md`
- Modify: `docs/conventions.md`
- Modify: `CHANGELOG.md`
- Modify: `tests/test_manifest.py`
- Modify: `tests/test_transforms.py`
- Modify: `tests/test_build_docs.py`
- Modify: `tests/test_wiki.py`
- Modify: `tests/test_check_docs.py`
- Modify: `docs/superpowers/plans/2026-08-11-issue-55-security-policy-implementation-plan.md`

**Interfaces:**
- Consumes: the live private-vulnerability-reporting setting, current release/GitFlow practice, `docs/dependency-contracts.md`, and upstream ownership boundaries in `CONTRIBUTING.md`.
- Produces: one native and manifest-indexed policy projected to all three documentation surfaces.

- [x] **Step 1: Enable and verify the private reporting route**

Enable GitHub private vulnerability reporting for this repository, then verify the repository setting reports enabled. Use the native `Report a vulnerability` form as the primary route; do not invent an email alias.

- [x] **Step 2: Add the concise, numbered policy**

Cover supported versions, private reporting, requested report contents, best-effort response and coordinated disclosure, in-scope surfaces, NNx/Atlas upstream boundaries, dependency-advisory handling, and current limitations. Include notebook trust/arbitrary-code risk, model and data provenance, unsafe serialized artifacts, secret handling and rotation, Atlas/JupyterHub mounted-workspace boundaries, and the host-native-only Ollama rule. State explicitly that containers and mounts are not trust boundaries and that no fixed SLA, hash-enforcement, or backport promise exists.

- [x] **Step 3: Index the canonical source**

Add `SECURITY.md` as standalone manifest section 13 with strictly numbered headings. Link it from the README repository tree and governance index, route vulnerability reports from `CONTRIBUTING.md` to the private policy, update `docs/conventions.md` so canonical sources include manifest-declared root governance pages, and add an Unreleased changelog entry. Keep policy links limited to exact manifest source keys so site/wiki transforms remain valid.

- [x] **Step 4: Generate and inspect both projections**

Run:

```bash
python -m scripts.docs.build_docs --site --wiki
```

Expected: `generated/site/SECURITY.md`, `generated/wiki/13-Security-policy.md`, and both navigation entries contain the same policy after documented link transforms.

- [x] **Step 5: Run focused validation and commit**

Run:

```bash
make docs-check
make docs-wiki
pytest -p no:cacheprovider tests/test_manifest.py tests/test_transforms.py tests/test_check_docs.py tests/test_build_docs.py tests/test_wiki.py tests/test_verify_repo.py -q
python scripts/verify_repo.py --check docs --fast
git diff --check
```

Commit the canonical policy, manifest, indexes, changelog, and completed Task 1–2 record with message `docs: add repository security policy`.

---

## 12.8.4 Task 3: Audit, review, publish, and clean up

**Files:**
- Modify: in-scope canonical sources only if review or the complete docs audit finds an issue.
- Modify: `docs/superpowers/plans/2026-08-11-issue-55-security-policy-implementation-plan.md`

**Interfaces:**
- Consumes: reviewed Tasks 1–2 and the complete three-surface audit checklist.
- Produces: a green repository gate, reviewed GitFlow integration, live policy publication, synchronized long-lived branches, and a clean issue record.

- [x] **Step 1: Run the complete three-surface documentation audit**

Apply checklist A–L from `three-surface-docs-audit`, including single-source generation, self-containment, wiki configuration, CI, numbering, completeness, strict build, reproducibility separation, content grounding, and opener parity.

- [x] **Step 2: Correct all in-scope findings through the same review loop**

Apply the smallest canonical-source or tested-pipeline correction, regenerate both projections, and rerun the focused check. Never hand-edit generated output.

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

- [ ] **Step 4: Complete independent requirements and quality reviews**

Fix every Critical or Important finding, rerun affected tests, then rerun the complete gate.

- [ ] **Step 5: Complete GitFlow and publication**

Merge the green feature PR into `develop`, promote `develop` to `main` through a second green PR, and merge `main` back into `develop` through a reviewed synchronization PR when needed. Verify final `main` CI, Pages deployment, native wiki sync, and live policy discoverability.

- [ ] **Step 6: Record and clean final state**

Update issue #55, project status, and tracker #53 with PRs, final main SHA, tests, docs, setting verification, and cleanup. Delete the conclusively merged feature branch locally/remotely, prune refs, update both long-lived branches, and confirm content parity, one clean worktree, no open PR, and no ml-eng-lab Atlas or prohibited Ollama/ComfyUI container.
