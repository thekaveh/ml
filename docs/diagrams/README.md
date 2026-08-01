# 10 Diagram Provenance

This directory holds the diagram **masters**: standalone HTML/SVG architecture
artifacts. Each master is the single source
for one diagram; the docs pipeline derives per-surface artifacts from it so all three
documentation surfaces embed the same geometry.

## 10.1 Pipeline

Each diagram declared in `docs/manifest.yaml` (`diagrams[].master` → a file in this
directory) is rendered once into two artifacts by `scripts/docs/render_diagrams.py` and
`scripts/docs/build_docs.py`:

- **SVG** → `generated/site/assets/img/<id>.svg` — embedded by the generated site.
- **PNG** → `docs/diagrams/img/<id>.png` (committed) — embedded by the in-repo Markdown
  and the generated wiki.

Because the same master feeds all three surfaces, diagrams never drift: updating a master
and re-running the pipeline refreshes every surface.

## 10.2 Current Artifacts

- `ml-eng-lab-system.html` — repository context and primary components (declared as
  `system` in the manifest; embedded by `docs/architecture.md`).
- `ml-eng-lab-runtime-flow.html` — local VS Code → Atlas JupyterHub as the default runtime,
  fallback entry paths, and task-local artifact behavior; embedded only in
  `docs/architecture.md` and never used as a project-branding asset.
- `ml-eng-lab-notebook-sequence.html` — notebook execution from parameters through
  training, ranking, visualization, persistence, and verification.
- `ml-eng-lab-docs-publishing.html` — README, docs, MkDocs, GitHub Pages, wiki, and
  repository metadata surfaces.
- `ml-eng-lab-docs-sync.html` — the three-surface documentation sync pipeline (added with
  the docs-overhaul foundation).

## 10.3 Generation Contract

The checked-in HTML masters are the source of truth. Do not hand-edit generated geometry
without updating this provenance note in the same change. When a diagram needs to change:

1. Update the affected HTML master while preserving its standalone, deterministic SVG contract.
2. Run `python -m scripts.docs.render_diagrams` to refresh the committed PNG
   (`docs/diagrams/img/<id>.png`); the site SVG is refreshed by the next `build_docs` run.
3. Run `make docs-check` and `make docs-wiki` so the repository, site, and wiki projections are
   regenerated and validated from the same commit.
4. Inspect the rendered diagram at normal browser zoom for overlapping labels, boxes,
   legends, or arrows before committing.

## 10.4 Review Rules

- Diagram content must describe current repository behavior, not aspirational design.
- Arrows and labels should remain readable on desktop and narrow screens.
- New architecture, runtime, notebook-flow, or documentation-publishing changes should
  update the matching master and add or refresh its manifest entry in the same pull request.
- A diagram referenced by the manifest must have its master present, or
  `scripts/docs/manifest.py:load_manifest` and `render_diagrams.py` will fail.
