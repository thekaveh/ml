# 12.6 Project Opener Visual Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the runtime-diagram opener with a branded, centered, badge-rich project opener that remains self-contained and synchronized across the README, MkDocs site, and GitHub wiki.

**Architecture:** Canonical raster/SVG assets live under `docs/assets/` and are copied byte-for-byte into each generated surface by one shared asset helper. README and landing-source markup differ only in their local asset prefixes and numbered H1, while `check_project_opening` validates their normalized semantic structure, complete badge inventory, two-paragraph summary, and local file existence.

**Tech Stack:** Markdown/HTML, PNG, SVG shields, Python 3.11, `pathlib`, `shutil`, pytest, MkDocs Material, the existing three-surface docs generator, built-in image generation.

## 12.6.1 Global Constraints

- The poster targets a 2.4:1 panoramic display ratio and contains only `ML ENG LAB` plus `NOTEBOOKS · SYSTEMS · REPRODUCIBILITY`.
- The runtime-flow diagram remains an architecture artifact and must not appear in either opener.
- Badge categories and membership exactly match design section 12.5.6.
- Poster and badge files are committed locally and physically copied into the site and wiki.
- README, site, and wiki remain self-contained; no new cross-surface or repository-file-view links.
- The executive summary remains 100–150 words, uses exactly two paragraphs, and is byte-equivalent after whitespace normalization in both hand-authored sources.
- The default runtime copy continues to prefer local VS Code connected to Atlas JupyterHub on the ML Engineering track.
- Ollama remains host-native only; no containerized Ollama service is launched during implementation or verification.
- All implementation commits stay on `codex/docs-opener-visual-remediation`, then merge by PR into `develop` and by a separate PR from `develop` into `main`.

---

## 12.6.2 Task 1: Add shared project-asset projection

**Files:**
- Create: `scripts/docs/project_assets.py`
- Modify: `scripts/docs/build_docs.py:26-79`
- Modify: `scripts/docs/wiki.py:18-94`
- Modify: `tests/test_build_docs.py:42-71`
- Modify: `tests/test_wiki.py:39-64`

**Interfaces:**
- Produces: `copy_project_assets(repo_root: Path, out_dir: Path, expected: set[Path]) -> list[Path]`.
- Consumes: canonical files recursively beneath `docs/assets/`.
- Guarantees: byte-identical output beneath `<surface>/assets/`, with every copied file added to the caller's stale-file `expected` set.

- [x] **Step 1: Seed representative poster and badge fixtures**

Add binary-safe fixture writes to both `_seed` helpers:

```python
poster = repo / "docs/assets/ml-eng-lab-poster.png"
poster.parent.mkdir(parents=True, exist_ok=True)
poster.write_bytes(b"poster")
badge = repo / "docs/assets/badges/python.svg"
badge.parent.mkdir(parents=True, exist_ok=True)
badge.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
```

- [x] **Step 2: Write failing site/wiki projection assertions**

```python
assert (out / "assets/ml-eng-lab-poster.png").read_bytes() == b"poster"
assert (out / "assets/badges/python.svg").exists()
```

Add both assertions to the primary site and wiki render tests.

- [x] **Step 3: Run the focused tests and confirm RED**

Run:

```bash
pytest tests/test_build_docs.py::test_render_site_writes_pages_and_assets tests/test_wiki.py::test_render_wiki_writes_home_sidebar_pages_and_images -q
```

Expected: both tests fail because generated project assets do not exist.

- [x] **Step 4: Implement the shared recursive copier**

Create `scripts/docs/project_assets.py`:

```python
from __future__ import annotations

import shutil
from pathlib import Path


def copy_project_assets(repo_root: Path, out_dir: Path, expected: set[Path]) -> list[Path]:
    source_root = repo_root / "docs/assets"
    if not source_root.exists():
        return []
    written: list[Path] = []
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        destination = out_dir / "assets" / source.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        expected.add(destination)
        written.append(destination)
    return written
```

Import and call it in both renderers before their stale-file sweeps:

```python
from scripts.docs.project_assets import copy_project_assets

copy_project_assets(repo_root, out_dir, expected)
```

- [x] **Step 5: Run the focused tests and confirm GREEN**

Run the Task 1 command again. Expected: `2 passed`.

- [x] **Step 6: Commit the projection boundary**

```bash
git add scripts/docs/project_assets.py scripts/docs/build_docs.py scripts/docs/wiki.py tests/test_build_docs.py tests/test_wiki.py
git commit -m "feat(docs): project opener assets to every surface"
```

---

## 12.6.3 Task 2: Create the poster and local shield inventory

**Files:**
- Create: `docs/assets/ml-eng-lab-poster.png`
- Create: `docs/assets/badges/*.svg` (19 files)

**Interfaces:**
- Produces: the canonical project poster and badge files consumed by Task 1 and referenced by Task 3.
- Consumes: the exact badge inventory in design section 12.5.6.

- [x] **Step 1: Generate the project poster with the built-in image tool**

Use this production prompt:

```text
Use case: ads-marketing
Asset type: GitHub repository and documentation hero poster
Primary request: Create a panoramic branded poster for an engineering repository named ML ENG LAB.
Scene/backdrop: Near-black technical laboratory space with subtle notebook-cell grids, neural-network traces, graph nodes, and one restrained cyan remote-runtime beam suggesting Atlas infrastructure.
Subject: The exact large centered wordmark "ML ENG LAB" is dominant; the exact smaller descriptor "NOTEBOOKS · SYSTEMS · REPRODUCIBILITY" sits beneath it.
Style/medium: Premium technical editorial illustration; crisp, minimal, sophisticated, dark-first.
Composition/framing: Wide panoramic composition targeting 2.4:1; generous safe margins; readable when reduced to GitHub README width.
Lighting/mood: Controlled cyan and violet glow with small emerald and amber accents; confident and precise, not cinematic fantasy.
Text (verbatim): "ML ENG LAB" and "NOTEBOOKS · SYSTEMS · REPRODUCIBILITY"
Constraints: exact spelling; no other text; no people; no vendor logos; no architecture boxes; no terminal screenshot; no watermark.
Avoid: busy dashboard composition, tiny labels, photorealistic server rooms, stock-photo aesthetics, Atlas mythology, illegible decorative glyphs.
```

- [x] **Step 2: Inspect and iterate once if necessary**

Validate exact text, wordmark dominance, absence of extra glyphs, and readability at approximately 1000px display width. If one condition fails, issue one targeted image edit correcting only that condition. Save the accepted result as `docs/assets/ml-eng-lab-poster.png`.

- [x] **Step 3: Fetch and commit local shield SVGs**

Create `docs/assets/badges/` and download these exact shield endpoints once; the documentation never references the remote URLs:

```text
python.svg                 Python-3.11-3776AB?logo=python&logoColor=white
jupyter.svg                Jupyter-Notebook-F37626?logo=jupyter&logoColor=white
numpy.svg                  NumPy-013243?logo=numpy&logoColor=white
pandas.svg                 pandas-150458?logo=pandas&logoColor=white
pytorch.svg                PyTorch-2.4-EE4C2C?logo=pytorch&logoColor=white
pytorch-geometric.svg      PyTorch_Geometric-graph_ML-3C2179
scikit-learn.svg           scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white
spacy.svg                  spaCy-NLP-09A3D5?logo=spacy&logoColor=white
nltk.svg                   NLTK-NLP-2C6E49
networkx.svg               NetworkX-graphs-4C72B0
atlas.svg                  Atlas-ML_Engineering-0B1220
docker.svg                 Docker-runtime-2496ED?logo=docker&logoColor=white
vscode.svg                 VS_Code-editor-007ACC?logo=visualstudiocode&logoColor=white
github-codespaces.svg      GitHub_Codespaces-cloud_dev-181717?logo=github&logoColor=white
nnx.svg                    NNx-0.2.0-6D28D9
papermill.svg              Papermill-reexecution-A16207
pytest.svg                 pytest-tests-0A9EDC?logo=pytest&logoColor=white
ruff.svg                   Ruff-lint-D7FF64?logo=ruff&logoColor=111827
github-actions.svg         GitHub_Actions-CI-2088FF?logo=githubactions&logoColor=white
```

Use `https://img.shields.io/badge/<endpoint>` as the one-time source, verify every response starts with `<svg`, and retain the files locally.

- [x] **Step 4: Verify assets**

Run:

```bash
file docs/assets/ml-eng-lab-poster.png docs/assets/badges/*.svg
test "$(find docs/assets/badges -name '*.svg' | wc -l | tr -d ' ')" = 19
```

Expected: one PNG, nineteen SVG documents, exit 0.

- [x] **Step 5: Commit the visual assets**

```bash
git add docs/assets/ml-eng-lab-poster.png docs/assets/badges
git commit -m "feat(docs): add branded opener artwork and stack badges"
```

---

## 12.6.4 Task 3: Install the centered opener on both canonical sources

**Files:**
- Modify: `README.md:1-19`
- Modify: `docs/index.md:1-19`
- Modify: `docs/stylesheets/extra.css:58`

**Interfaces:**
- Produces: equivalent opener blocks whose only intentional differences are the landing H1 number and asset prefix.
- Consumes: `docs/assets/ml-eng-lab-poster.png` and the nineteen Task 2 badges.

- [x] **Step 1: Replace the README opener markup**

Use centered HTML wrappers in this order:

```html
<p align="center">
  <img src="docs/assets/ml-eng-lab-poster.png" alt="ML Eng Lab — notebooks, systems, and reproducibility" width="100%">
</p>

<h1 align="center">ML ENG LAB</h1>

<p align="center"><strong>Local notebooks. Remote Atlas execution. Explicit infrastructure contracts.</strong></p>
```

Follow it with four `<p align="center">` badge rows. Each begins with a `<sub><strong>…</strong></sub><br>` category label and contains the category's local `<img>` shields in the design-specified order.

- [x] **Step 2: Replace the landing-source opener markup**

Use identical markup with:

```html
<img src="assets/ml-eng-lab-poster.png" ...>
<h1 align="center">1 · ML ENG LAB</h1>
```

and `assets/badges/...` sources. Preserve all content beginning at `## 1.1 Repository map`.

- [x] **Step 3: Split the synchronized summary into two paragraphs**

Keep the existing 136-word content and insert one blank line after `the reusable thekaveh-nnx toolkit evolve together.` Remove backticks around `ml-eng-lab` and `thekaveh-nnx` so the opener reads as prose rather than code UI. Keep the summary markers around both paragraphs.

- [x] **Step 4: Add narrowly scoped landing spacing**

Append to `docs/stylesheets/extra.css`:

```css
.md-typeset > p:first-child {
  margin-bottom: 0.75rem;
}

.md-typeset > h1[align="center"] {
  margin: 1rem 0 0.5rem;
  font-size: clamp(2.5rem, 6vw, 4.5rem);
  font-weight: 800;
  letter-spacing: 0.08em;
}
```

- [x] **Step 5: Render both canonical openers for inspection**

Run:

```bash
python -m scripts.docs.build_docs --site --wiki
mkdocs build --strict
```

Expected: strict build succeeds and generated `index.md`/`Home.md` contain only surface-local asset paths.

- [x] **Step 6: Commit the opener markup**

```bash
git add README.md docs/index.md docs/stylesheets/extra.css
git commit -m "docs: install centered badge-rich project opener"
```

---

## 12.6.5 Task 4: Strengthen opener and numbering validation with TDD

**Files:**
- Modify: `scripts/docs/check_docs.py:16-36,106-123,220-274`
- Modify: `tests/test_check_docs.py:144-247,279-287`

**Interfaces:**
- Produces: `check_project_opening(repo_root: Path) -> list[Finding]` with structural poster/header/badge/summary validation.
- Produces: `_markdown_headings(text: str) -> list[tuple[int, str]]` supporting centered HTML H1 as well as Markdown headings.
- Consumes: the canonical poster and badge constants corresponding to Task 3 markup.

- [x] **Step 1: Rewrite the opener fixture around a shared builder**

Define test constants for `BADGE_GROUPS`, construct complete README/landing badge HTML from a supplied prefix, and make `_write_project_opening` emit centered poster, HTML H1, bold centered tagline, four badge rows, and two summary paragraphs.

- [x] **Step 2: Add failing defect tests**

Add tests asserting findings for:

```python
def test_project_opening_rejects_runtime_diagram_as_poster(tmp_path): ...
def test_project_opening_rejects_left_aligned_markdown_title(tmp_path): ...
def test_project_opening_rejects_missing_badge_and_plain_text_stack(tmp_path): ...
def test_project_opening_rejects_missing_local_asset(tmp_path): ...
def test_project_opening_rejects_single_paragraph_summary(tmp_path): ...
def test_numbering_accepts_centered_html_h1(tmp_path): ...
```

Each test mutates exactly one valid fixture and asserts the specific message fragment: `runtime-flow`, `centered HTML title`, `badge`, `asset missing`, `two paragraphs`, or an empty numbering finding list.

- [x] **Step 3: Run the focused tests and confirm RED**

Run:

```bash
pytest tests/test_check_docs.py -q -k 'project_opening or centered_html_h1'
```

Expected: new tests fail against the old string-only checker.

- [x] **Step 4: Implement HTML-H1-aware numbering**

Add:

```python
_HTML_H1_RE = re.compile(r'^<h1\s+align=["\']center["\']>(.+?)</h1>$', re.IGNORECASE)
```

In `_markdown_headings`, when outside a fence, append `(1, html_match.group(1))` for that form. Keep Markdown H1–H6 behavior unchanged. The landing title constant becomes `1 · ML ENG LAB`, whose numeric prefix remains `1`.

- [x] **Step 5: Implement structural opener validation**

Replace `_PROJECT_POSTERS` and `_PROJECT_TITLES` with exact per-surface poster/title paths plus an ordered `PROJECT_BADGE_GROUPS` tuple. Validate:

```python
if "runtime-flow" in opener:
    findings.append(Finding("error", f"runtime-flow diagram cannot be the project poster in {relative_path}"))
if centered_title not in opener:
    findings.append(Finding("error", f"centered HTML title missing from {relative_path}"))
if len(re.split(r"\n\s*\n", matches[0].strip())) != 2:
    findings.append(Finding("error", f"project summary in {relative_path} must contain exactly two paragraphs"))
```

For every expected badge, require the exact `<img alt="…" src="…">` fragment and verify `(repo_root / resolved_source).is_file()`. Normalize `docs/assets/` and `assets/` to one logical prefix before comparing opener structure across the two sources.

- [x] **Step 6: Run focused and full docs tests**

Run:

```bash
pytest tests/test_check_docs.py -q -k 'project_opening or centered_html_h1'
pytest tests/test_manifest.py tests/test_notebook_infrastructure.py tests/test_links.py tests/test_transforms.py tests/test_render_diagrams.py tests/test_build_docs.py tests/test_wiki.py tests/test_check_docs.py tests/test_push_wiki.py -q
```

Expected: focused tests pass 11/11 and the complete docs-script suite passes 89/89.

- [x] **Step 7: Commit the stronger gate**

```bash
git add scripts/docs/check_docs.py tests/test_check_docs.py
git commit -m "test(docs): enforce deliberate opener structure"
```

---

## 12.6.6 Task 5: Update documentation records and implementation evidence

**Files:**
- Modify: `CHANGELOG.md:5-21`
- Modify: `docs/diagrams/README.md:21-30`
- Modify: `docs/superpowers/plans/2026-08-01-opener-visual-remediation-implementation-plan.md`

**Interfaces:**
- Produces: current user-facing release notes, accurate diagram provenance, and a checked implementation record.
- Consumes: final asset paths, checker behavior, and verification results from Tasks 1–4.

- [x] **Step 1: Add the Unreleased changelog entry**

Under `### Changed`, document the dedicated local poster, centered shared header, nineteen categorized badges, two-paragraph summary, generated asset projection, and strengthened CI gate. State explicitly that `runtime-flow` remains an architecture diagram only.

- [x] **Step 2: Clarify runtime-flow provenance**

Update `docs/diagrams/README.md` so the runtime-flow entry says it is embedded only in `docs/architecture.md` and is not a project-branding asset.

- [x] **Step 3: Mark completed plan steps**

Change each executed `- [ ]` to `- [x]`. Replace expected-output descriptions only where actual results differ, without weakening any acceptance criterion.

- [x] **Step 4: Run documentation integrity checks**

Run:

```bash
rg -n '\b(T[O]DO|T[B]D|F[I]XME|X[X]X)\b' README.md docs generated
git diff --check
```

Expected: no placeholder hits outside literal test/spec examples and no whitespace errors.

- [x] **Step 5: Commit the records**

```bash
git add CHANGELOG.md docs/diagrams/README.md docs/superpowers/plans/2026-08-01-opener-visual-remediation-implementation-plan.md
git commit -m "docs: record opener visual remediation"
```

---

## 12.6.7 Task 6: Verify, review, and complete Gitflow integration

**Files:**
- Verify all files changed in Tasks 1–5.
- No new implementation files unless verification exposes an in-scope defect.

**Interfaces:**
- Produces: a reviewed feature PR into `develop`, followed by a content-synchronizing PR from `develop` into `main`.
- Consumes: the complete feature branch.

**Completion:** PR #51 merged the reviewed feature into `develop`, and PR #52 merged the release into
`main`. The main-branch Pages deployment and wiki synchronization both completed successfully, publishing
the remediated opener to the live surfaces.

- [x] **Step 1: Run all local gates**

```bash
make docs-check
make docs-wiki
pytest tests/ -q
ruff check scripts/docs tests/test_build_docs.py tests/test_wiki.py tests/test_check_docs.py
python scripts/verify_repo.py --check all --fast
make check-tier-a-clean
git diff --check
git status --short
```

Expected: all commands pass; the final status is clean after commits.

- [x] **Step 2: Visually inspect the poster and built landing page**

Open the committed poster and the locally built site landing page. Confirm exact wordmark text,
centered hierarchy, four badge categories, readable summary, no missing images, and acceptable dark
and light theme rendering.

- [x] **Step 3: Request code review and address only verified findings**

Use the requesting-code-review skill against `origin/develop...HEAD`. Re-run the affected focused
tests after each accepted correction, then repeat the full Task 6 gate.

- [x] **Step 4: Push and open the feature-to-develop PR**

```bash
git push origin codex/docs-opener-visual-remediation
gh pr create --base develop --head codex/docs-opener-visual-remediation --title "docs: rebuild the project opener" --body "Replaces the architecture-diagram opener with a dedicated branded poster, centered title and tagline, nineteen local stack badges, a two-paragraph executive summary, three-surface asset projection, and stricter opener validation. Verified with the complete local documentation and repository gates."
```

Wait for required checks, merge the PR, and delete the remote feature branch.

- [x] **Step 5: Open and merge the develop-to-main release PR**

```bash
gh pr create --base main --head develop --title "release: publish project opener remediation" --body "Publishes the reviewed project-opener remediation from develop to main so GitHub Pages and the wiki receive the dedicated poster, centered header, local technology badges, and synchronized executive summary."
```

Wait for required checks, merge the PR, and confirm `origin/develop^{tree}` equals
`origin/main^{tree}`.

- [x] **Step 6: Verify published surfaces and clean local state**

Confirm the Pages and wiki workflows succeed at the main merge commit; fetch the live site and wiki
and verify the new wordmark/tagline. Switch local state to updated `main`, remove the merged local
feature branch, prune remote references, and confirm no dangling worktrees, branches, or open PRs.
