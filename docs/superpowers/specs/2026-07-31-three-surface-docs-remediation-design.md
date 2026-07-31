# Three-surface documentation remediation design

**Status:** Approved for implementation
**Date:** 2026-07-31
**Decision:** Resolve every finding from the 2026-07-31 three-surface documentation audit in one Gitflow feature.

## 1. Goal

Restore the repository, generated MkDocs site, and GitHub wiki to a single explicit documentation contract. Every meaningful document under `docs/` must be declared by the manifest, every declared page must use baked hierarchical numbering, every user-facing claim must match the repository, and CI must reject the classes of drift found by the audit.

## 2. Chosen approach

Three approaches were considered:

1. Patch only the incorrect prose. This is small but leaves the checker unable to detect the same drift again.
2. Keep internal documents repository-only through exclusions. This reduces the published page set but preserves a second, unsynchronized documentation class.
3. Publish and validate the entire canonical tree, correct the prose, and harden the gates. This creates the largest focused change but makes the three-surface promise true by construction.

Use approach 3. Historical maintenance and implementation records remain available, but their status and headings become explicit and they are generated into the site and wiki like every other canonical page. No historical document is deleted.

## 3. Manifest and numbering architecture

`docs/manifest.yaml` remains the source of truth. A section is either a source leaf or a children group; no section combines both. The Environment and Dependency sections become children groups with overview pages as their first children. Findings, diagram provenance, maintenance records, and design/implementation records receive explicit numbered sections after the notebook catalog.

Numbering is hierarchical on every page:

- H1 equals the page's exact manifest number.
- H2 appends one numeric component to the H1.
- H3 appends one numeric component to its parent H2.
- Deeper headings extend the same hierarchy.

The existing notebook page numbers remain `8.1` through `8.21`. Cross-references that name a changed section number are updated in the same change.

The two existing planned diagram masters are retained, corrected to describe current behavior, declared in the manifest, embedded in the appropriate canonical pages, and rendered into repository, site, and wiki assets.

## 4. Gate behavior

`scripts.docs.check_docs` gains explicit probes for:

- repository self-containment, including `README.md` and canonical Markdown;
- bidirectional manifest completeness for every Markdown document under `docs/`;
- exact manifest-number-to-H1 agreement;
- hierarchical H2/H3 numbering and parent-child progression;
- placeholder and deterministic generated-output checks already present.

The checker reports normal `Finding` values and exits non-zero without rewriting canonical files. Tests must demonstrate each new failure before implementation.

## 5. Content corrections

The README describes only the project and supported user workflows. It removes the `.io` URL, publication status, generated-file tree entries, and documentation-pipeline implementation details. Codespaces quota language uses GitHub's current compute-hour terminology.

Atlas pages distinguish two concepts:

- the `ml-eng` track may run default services;
- JupyterHub is currently the only service authorized as a notebook dependency.

Connection URLs are described as token-bearing. The docs do not promise a short lifetime because `scripts/atlas-connect.sh` accepts a configured token without enforcing a TTL.

The completed Atlas design and implementation records receive completed status. Their checkboxes become an implementation record rather than an apparently unfinished plan, and their service-scope language matches the actual track behavior.

Diagram-adjacent prose explains constraints and consequences, not rasterization, styling, or obvious visual contents. Diagram regeneration instructions explicitly cover repository PNGs, site SVGs, and wiki PNGs.

## 6. Reproducible documentation toolchain

Create a human-edited `docs-requirements.in` and compile the complete, hashed, cross-platform lock into `docs-requirements.txt` with `uv pip compile --universal --generate-hashes`. CI continues installing `docs-requirements.txt`, so all direct and transitive documentation dependencies resolve identically.

Set `NO_MKDOCS_2_WARNING=1` only for documentation build commands. This suppresses Material's upstream policy banner while preserving `mkdocs build --strict` for actual project warnings.

## 7. CI and publishing

- General CI runs on pushes to both `develop` and `main` and on PRs to both branches.
- The general docs-build job installs `libcairo2` before Python packages.
- The focused docs workflow watches the README, Atlas consumer/runtime inputs, submodule gitlink, Pages workflow, manifest sources, and existing documentation tooling.
- Pages runs the comprehensive documentation checker before uploading its artifact. The wiki is generated from the same checked commit after deployment.
- Publishing remains main-triggered; feature work still merges to `develop` first and then to `main` through a second PR.

## 8. Verification and success criteria

The change is complete only when:

1. Each new regression test is observed failing before its implementation and passes afterward.
2. `make docs-check` and a standalone `mkdocs build --strict` complete without warnings.
3. The full docs-script test set, full repository test suite, Ruff, and `make verify` pass.
4. Fresh site and wiki generation is deterministic and leaves the tracked worktree clean except for intended committed diagram assets.
5. Every Markdown document under `docs/` is declared in the manifest, and every diagram master has repository/site/wiki outputs.
6. The feature branch merges through a reviewed PR to `develop`, then `develop` merges through a separate reviewed PR to `main`.
