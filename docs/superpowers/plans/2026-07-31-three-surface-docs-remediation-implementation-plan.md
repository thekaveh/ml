# 12.4 Three-surface documentation remediation implementation plan

> **Implementation record:** This plan was executed task by task; completed steps use checked boxes.

**Goal:** Make every canonical document and diagram publish identically to the repository, MkDocs site, and GitHub wiki while preventing the audit findings from recurring.

**Architecture:** Extend `scripts.docs.check_docs` with independent repository-containment, bidirectional-completeness, and hierarchical-numbering probes. Normalize the manifest and canonical content to satisfy those probes, lock the documentation toolchain, and make all PR/publishing workflows run the same comprehensive gate.

**Tech Stack:** Python 3.11, PyYAML, pytest, Ruff, MkDocs Material, CairoSVG, GitHub Actions, GNU Make, `uv pip compile`, Git/GitHub CLI.

**Status:** Completed

## 12.4.1 Global constraints

- Preserve the Atlas gitlink at `61c7c5103660e2226bf107c115dae42bf46f8374`; do not edit `infra/`.
- Keep `LLM_PROVIDER_SOURCE=ollama-localhost`; never introduce containerized Ollama.
- Keep ComfyUI disabled or host-native only.
- Keep notebook page numbers `8.1` through `8.21`.
- A manifest section is a source leaf or a children group, never both.
- H1 equals the manifest number; each lower heading extends its parent number by one component.
- Generated site/wiki trees and root `mkdocs.yml` remain ignored and untracked.
- Feature work merges to `develop` before a separate `develop` to `main` release PR.

---

## 12.4.2 Task 1: Add regression gates for repository containment, completeness, and numbering

**Files:**

- Modify: `tests/test_check_docs.py`
- Modify: `scripts/docs/check_docs.py`

**Interfaces:**

- Produces: `check_repo_self_containment(repo_root: Path) -> list[Finding]`
- Produces: `manifest_markdown_sources(manifest: Manifest) -> set[str]`
- Produces: `check_numbering(manifest: Manifest, repo_root: Path) -> list[Finding]`
- Extends: `check_completeness(manifest, repo_root)` to report manifest-to-disk and disk-to-manifest drift.

- [x] **Step 1: Write failing repository-containment tests**

Add tests that create `README.md` and `docs/page.md` with `.io` or wiki links, assert both are reported, and prove normal relative repository links remain clean:

```python
def test_repo_self_containment_rejects_site_and_wiki_links(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text(f"{SITE_URL}\n", encoding="utf-8")
    (tmp_path / "docs/page.md").write_text(
        f"[wiki]({WIKI_URL}/Page)\n", encoding="utf-8"
    )
    findings = check_repo_self_containment(tmp_path)
    assert len(findings) == 2
```

- [x] **Step 2: Run the focused tests and confirm RED**

Run: `pytest tests/test_check_docs.py -q`

Expected: collection failure because `check_repo_self_containment` does not exist.

- [x] **Step 3: Write failing completeness and numbering tests**

Add fixtures proving:

```python
def test_completeness_rejects_unmanifested_markdown(tmp_path):
    manifest = _seed_manifest_files(tmp_path)
    (tmp_path / "docs/extra.md").write_text("# 9 Extra\n", encoding="utf-8")
    assert any("not declared" in f.message for f in check_completeness(manifest, tmp_path))

def test_numbering_requires_manifest_h1_and_hierarchical_children(tmp_path):
    manifest = _seed_manifest_files(tmp_path)
    (tmp_path / "docs/index.md").write_text(
        "# Overview\n\n## 1. Wrong depth\n\n### Child without a number\n", encoding="utf-8"
    )
    messages = [f.message for f in check_numbering(manifest, tmp_path)]
    assert any("H1" in message for message in messages)
    assert any("H2" in message for message in messages)
    assert any("H3" in message for message in messages)
```

- [x] **Step 4: Run the focused tests and confirm RED**

Run: `pytest tests/test_check_docs.py -q`

Expected: failures because inverse completeness and numbering validation are absent.

- [x] **Step 5: Implement the minimal independent probes**

Use the manifest's section sources and notebook docs as the declared Markdown set. Scan canonical Markdown except generated trees, extract `#` through `######` headings outside fenced blocks, and require each numbered heading to extend its parent prefix by exactly one numeric component. Add the probes to `check()` after deterministic generation.

- [x] **Step 6: Run focused tests and confirm GREEN**

Run: `pytest tests/test_check_docs.py tests/test_manifest.py -q`

Expected: all tests pass.

- [x] **Step 7: Commit the gate slice**

```bash
git add scripts/docs/check_docs.py tests/test_check_docs.py
git commit -m "test: gate canonical docs completeness and numbering"
```

---

## 12.4.3 Task 2: Normalize the manifest and canonical numbering

**Files:**

- Modify: `docs/manifest.yaml`
- Modify: all manifest-declared Markdown under `docs/`
- Modify: `tests/test_build_docs.py`
- Modify: `scripts/docs/build_docs.py` only if tests expose nested-section traversal gaps.

**Interfaces:**

- Consumes: `check_completeness` and `check_numbering` from Task 1.
- Produces: one manifest declaration for every `docs/**/*.md` file and every diagram master.

- [x] **Step 1: Write a failing real-repository manifest coverage test**

Add a test that loads the real manifest and asserts:

```python
declared = manifest_markdown_sources(manifest)
actual = {str(path.relative_to(REPO_ROOT)) for path in (REPO_ROOT / "docs").rglob("*.md")}
assert declared == actual
```

Also assert every top-level manifest section is exclusively a source leaf or children group.

- [x] **Step 2: Run the focused test and confirm RED**

Run: `pytest tests/test_check_docs.py -q -k 'real_repository or source_leaf'`

Expected: the nine prior omissions plus this design and plan are reported.

- [x] **Step 3: Restructure the manifest**

Use this hierarchy without changing notebook numbers:

```text
1 Overview
2 Architecture -> 2.1 System & context view
3 Concepts
4 Environment & runtimes -> 4.1 Environment setup, 4.2 JupyterHub, 4.3 VS Code, 4.4 Notebook infrastructure
5 Repository conventions
6 Dependency contracts -> 6.1 Dependency ledger, 6.2 Atlas pin-bump runbook
7 NNx
8.1–8.21 Notebooks
9 Findings -> 9.1 NNx, 9.2 Atlas
10 Diagram provenance
11 Maintenance -> 11.1–11.4 existing maintenance records
12 Design records -> 12.1 Atlas migration design, 12.2 Atlas implementation record,
                     12.3 audit-remediation design, 12.4 audit-remediation implementation record
```

Declare the `notebook-sequence` and `docs-publishing` diagram masters in `diagrams:`.

- [x] **Step 4: Normalize headings mechanically and review the diff**

For each page, set H1 to its manifest number, append one numeric component for each H2, append one more component for each H3, and continue the hierarchy for deeper headings. Preserve heading text and update prose/table cross-references to changed section numbers. Do not change code-fence comments beginning with `#`.

- [x] **Step 5: Make manifest traversal recursive if the real hierarchy requires it**

If site/wiki rendering or source maps only visit one child level, first add a failing nested-section test in `tests/test_build_docs.py`, then implement one recursive section iterator shared by rendering, nav, source maps, wiki sidebar, and checks.

- [x] **Step 6: Run gates and confirm GREEN**

Run: `pytest tests/test_check_docs.py tests/test_build_docs.py tests/test_wiki.py tests/test_transforms.py -q`

Expected: all tests pass and `python -m scripts.docs.check_docs` reports no heading/completeness findings.

- [x] **Step 7: Commit the hierarchy slice**

```bash
git add docs scripts/docs tests/test_check_docs.py tests/test_build_docs.py tests/test_wiki.py tests/test_transforms.py
git commit -m "docs: synchronize the complete canonical hierarchy"
```

---

## 12.4.4 Task 3: Correct README, Atlas, migration, and diagram content

**Files:**

- Modify: `README.md`
- Modify: `docs/index.md`
- Modify: `docs/architecture.md`
- Modify: `docs/env-setup.md`
- Modify: `docs/jupyterhub-integration.md`
- Modify: `docs/vscode-remote-access.md`
- Modify: `docs/notebook-infrastructure.md`
- Modify: `docs/atlas-pin-bump-runbook.md`
- Modify: `docs/diagrams/README.md`
- Modify: `docs/diagrams/ml-eng-lab-docs-publishing.html`
- Modify: `docs/diagrams/ml-eng-lab-notebook-sequence.html`
- Modify: `docs/superpowers/specs/2026-07-30-atlas-infrastructure-migration-design.md`
- Modify: `docs/superpowers/plans/2026-07-30-atlas-infrastructure-migration-implementation-plan.md`
- Test: `tests/test_check_docs.py`
- Test: `tests/test_verify_repo.py`

**Interfaces:**

- Consumes: repository self-containment and grounding checks.
- Produces: current, security-accurate user guidance and two current diagram masters.

- [x] **Step 1: Add failing content-contract tests**

Assert the real README contains none of `thekaveh.github.io`, `MkDocs`, wiki synchronization, or deployment mechanics. Assert token guidance contains `token-bearing` but not `short-lived`. Assert notebook infrastructure says track defaults are not notebook authorization and does not claim all other services are inactive.

- [x] **Step 2: Run the content tests and confirm RED**

Run: `pytest tests/test_check_docs.py tests/test_verify_repo.py -q -k 'readme or token or atlas'`

Expected: failures on the audited wording.

- [x] **Step 3: Correct user-facing and Atlas prose**

Remove README build-system prose and generated `mkdocs.yml` tree entry. Replace Codespaces quota text with `120 included compute hours for Free and 180 for Pro; on a two-core machine this is 60/90 machine-hours`. Replace every unqualified `short-lived` claim with `token-bearing` and advise reconnecting after restart. State that Atlas track defaults may run but JupyterHub is the only currently authorized notebook dependency.

- [x] **Step 4: Convert migration documents into completed records**

Set the 2026-07-30 design status to `Implemented`. Change the plan title/status to an implementation record, convert all 86 `[ ]` markers to `[x]`, and correct the Jupyter-only statement to distinguish task dependencies from track defaults.

- [x] **Step 5: Correct and publish diagram content**

Remove styling/rasterization filler from diagram-adjacent prose. Update the docs-publishing status to current main-triggered Pages/wiki publication. Update the notebook-sequence master only where its labels do not match current Makefile/runtime behavior. Embed both diagrams in their manifest-declared pages and make regeneration instructions require `make docs-check` plus `make docs-wiki`.

- [x] **Step 6: Run focused tests and render diagrams**

Run: `pytest tests/test_check_docs.py tests/test_verify_repo.py tests/test_render_diagrams.py -q`

Run: `python -m scripts.docs.render_diagrams`

Expected: tests pass and eleven committed PNGs exist.

- [x] **Step 7: Commit the content slice**

```bash
git add README.md docs tests/test_check_docs.py tests/test_verify_repo.py
git commit -m "docs: correct Atlas and publishing guidance"
```

---

## 12.4.5 Task 4: Lock documentation dependencies and harden workflows

**Files:**

- Create: `docs-requirements.in`
- Regenerate: `docs-requirements.txt`
- Modify: `Makefile`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/docs.yml`
- Modify: `.github/workflows/pages.yml`
- Modify: `tests/test_verify_repo.py`
- Modify: `tests/test_makefile_contract.py`

**Interfaces:**

- Produces: hashed universal documentation lock installed by every docs workflow.
- Produces: warning-free `docs-build`/`docs-check` commands and complete Gitflow trigger coverage.

- [x] **Step 1: Add failing workflow and Makefile contract tests**

Assert CI push and PR branches both equal `{develop, main}`, every diagram-rendering Ubuntu job installs `libcairo2`, docs path filters include README/Atlas/Pages inputs, Pages runs `python -m scripts.docs.check_docs`, and MkDocs commands receive `NO_MKDOCS_2_WARNING=1`.

- [x] **Step 2: Run tests and confirm RED**

Run: `pytest tests/test_verify_repo.py tests/test_makefile_contract.py -q -k 'docs or gitflow or cairo or mkdocs'`

Expected: failures for missing develop push, Cairo, path inputs, Pages gate, and warning suppression.

- [x] **Step 3: Create and compile the documentation lock**

Create `docs-requirements.in` with these exact direct requirements, selected to preserve the tested local toolchain while matching the current live Material release:

```text
mkdocs-material==9.7.7
pyyaml==6.0.3
cairosvg==2.9.0
ruff==0.9.10
pytest==9.0.3
```

Then run:

```bash
uv pip compile docs-requirements.in --universal --generate-hashes \
  --custom-compile-command 'uv pip compile docs-requirements.in --universal --generate-hashes -o docs-requirements.txt' \
  -o docs-requirements.txt
```

The generated file must include exact versions and hashes for all transitive dependencies.

- [x] **Step 4: Harden Make and workflows**

Add `NO_MKDOCS_2_WARNING=1` to MkDocs build/serve commands. Add `develop` to CI pushes; install `libcairo2` in CI docs-build; expand docs path filters; run the comprehensive checker in Pages before strict build; and ensure the wiki uses the same locked requirements.

- [x] **Step 5: Run focused tests and confirm GREEN**

Run: `pytest tests/test_verify_repo.py tests/test_makefile_contract.py -q`

Expected: all workflow and Makefile contracts pass.

- [x] **Step 6: Commit the tooling slice**

```bash
git add docs-requirements.in docs-requirements.txt Makefile .github/workflows tests
git commit -m "ci: harden three-surface documentation gates"
```

---

## 12.4.6 Task 5: Verify, finalize records, and publish through Gitflow

**Files:**

- Modify: `docs/superpowers/plans/2026-07-31-three-surface-docs-remediation-implementation-plan.md`

**Interfaces:**

- Consumes: all prior tasks.
- Produces: verified feature branch, feature-to-develop PR, and develop-to-main PR.

- [x] **Step 1: Run complete local verification**

```bash
python -m scripts.docs.build_docs --site --wiki
make docs-check
NO_MKDOCS_2_WARNING=1 mkdocs build --strict
pytest tests/test_manifest.py tests/test_notebook_infrastructure.py tests/test_links.py \
  tests/test_transforms.py tests/test_render_diagrams.py tests/test_build_docs.py \
  tests/test_wiki.py tests/test_check_docs.py tests/test_push_wiki.py -q
pytest tests/ -q
ruff check .
make verify
make check-tier-a-clean
git diff --check
```

Expected: zero failures, zero project warnings, and no unintended notebook changes.

- [x] **Step 2: Verify inventory and deterministic cleanliness**

Confirm every `docs/**/*.md` appears in the manifest, all eleven diagram masters have committed PNG/site SVG/wiki PNG outputs, generated trees remain ignored, and a second generation produces identical hashes.

- [x] **Step 3: Mark this implementation record complete and commit**

Convert each completed checkbox in this plan to `[x]`, set its status to completed, rerun `make docs-check`, then commit the record and any regenerated committed diagram assets.

- [x] **Step 4: Push and open the feature PR to develop**

```bash
git push
gh pr create --base develop --head codex/fix-three-surface-docs-audit \
  --title "docs: remediate three-surface synchronization audit" \
  --body $'## Summary\n- synchronize every canonical document and diagram across repository, site, and wiki\n- harden completeness, numbering, self-containment, CI, and publishing gates\n- correct Atlas, token, Codespaces, and migration-record claims\n\n## Verification\n- make docs-check\n- pytest tests/ -q\n- ruff check .\n- make verify'
```

Wait for required checks, address failures, and merge without force-pushing.

- [x] **Step 5: Open and merge the develop-to-main release PR**

Update local `develop`, confirm it is content-identical to the merged feature, push if needed, then create a separate `develop` to `main` PR. Wait for required checks and merge it.

- [x] **Step 6: Resynchronize develop and clean up**

If GitHub's merge commit makes branch SHAs differ, merge `main` back to `develop` through the repository's normal synchronization PR. Confirm main/develop trees are identical, no open remediation PR remains, and delete the merged feature branch locally and remotely when it is no longer attached to the workspace.

---

## 12.4.7 Task 6: Lock the project opener and correct remaining documentation drift

**Files:**

- Modify: `README.md`
- Modify: `docs/index.md`
- Modify: `docs/conventions.md`
- Modify: `docs/diagrams/ml-eng-lab-docs-publishing.html`
- Modify: `docs/diagrams/img/docs-publishing.png`
- Modify: `scripts/docs/check_docs.py`
- Modify: `tests/test_check_docs.py`

**Interfaces:**

- Produces: `check_project_opening(repo_root: Path) -> list[Finding]`.
- Enforces: the canonical tagline and executive summary are identical in `README.md` and
  `docs/index.md`, both entry points contain the runtime-flow poster, and the summary contains
  100–150 words.

- [x] **Step 1: Add failing project-opener tests**

Add focused fixtures for a missing poster, divergent tagline, divergent summary, and an
out-of-range summary. Add a real-repository assertion that the canonical opener is clean.

- [x] **Step 2: Run the focused test and confirm RED**

Run: `pytest tests/test_check_docs.py -q -k project_opening`

Expected: collection fails because `check_project_opening` does not exist.

- [x] **Step 3: Implement the minimal opener gate**

Define the canonical tagline and summary opening in `scripts/docs/check_docs.py`. Extract the
summary between `project-summary` markers, compare its complete normalized text across both files,
count words, and require the surface-appropriate runtime-flow poster path.

- [x] **Step 4: Correct all eight audited content findings**

Install the shared opener in both entry points; remove publishing mechanics from the landing
opening; state that all 21 deep-dives exist; make Gitflow mandatory; document CI pushes to
`develop` and `main`; and relabel the wiki as a complete generated documentation mirror.

- [x] **Step 5: Run focused tests and regenerate the diagram**

Run: `pytest tests/test_check_docs.py -q -k project_opening`

Run: `python -m scripts.docs.render_diagrams`

Expected: opener tests pass and `docs/diagrams/img/docs-publishing.png` reflects the corrected
wiki label.

- [x] **Step 6: Run complete verification and independent review**

Run the full documentation and repository verification suite, request an independent review, and
resolve every Critical or Important finding before publishing the feature branch through the
required feature-to-`develop` and `develop`-to-`main` pull requests.
