# 12.36 Issue 71 Notebook Source/Output Freshness Hashes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stamp deterministic source hashes beside retained notebook outputs
and fail repository verification when active output/source pairs are missing,
malformed, stale, or orphaned.

**Architecture:** A standard-library JSON stamper owns source canonicalization
and atomic notebook updates. Papermill targets call it only after successful
execution, while E8 imports the same digest helper and enforces the contract
over the authoritative active-notebook inventory.

**Tech Stack:** Python 3.11, JSON, SHA-256, argparse, pytest, Make, Papermill,
nbformat, MkDocs.

**Spec:**
`docs/superpowers/specs/2026-08-22-issue-71-notebook-freshness-hashes-design.md`

## 12.36.1 Global constraints

- Hash exact UTF-8 logical source with SHA-256; do not normalize whitespace,
  comments, magics, line endings, or syntax.
- Concatenate JSON source arrays without a separator; preserve source strings
  exactly.
- Store a bare 64-character lowercase hexadecimal digest in
  `cell.metadata.source_hash` only when a code cell has non-empty outputs.
- Never hash output bytes and never stamp a failed or partial execution.
- Enforce all active notebooks; exclude only archives and outputless code cells.
- Preserve code, prose, outputs, execution counts, cell IDs, notebook metadata,
  and unrelated cell metadata during the one-time migration.
- Follow feature → `develop` → `main` → `develop` GitFlow.

---

## 12.36.2 Task 1: Build the deterministic atomic stamper

**Files:**

- Create: `scripts/stamp_notebook_source_hashes.py`
- Create: `tests/test_stamp_notebook_source_hashes.py`

**Interfaces:**

- Produces: `logical_source(source: object) -> str`
- Produces: `compute_source_hash(source: object) -> str`
- Produces: `stamp_document(document: dict[str, object]) -> int`
- Produces: `stamp_path(notebook: pathlib.Path) -> int`
- Produces: `active_notebook_paths(repo: pathlib.Path) -> tuple[pathlib.Path, ...]`
- Produces: CLI accepting one or more paths or `--all-active`.

- [ ] **Step 1: Write source and stamping tests first**

```python
def test_compute_source_hash_canonicalizes_string_and_list_equally():
    source = "x = 1\nprint(x)\n"
    assert compute_source_hash(source) == compute_source_hash(["x = 1\n", "print(x)\n"])


def test_stamp_document_marks_only_output_bearing_code_cells():
    document = {
        "cells": [
            {"cell_type": "code", "source": ["x = 1\n"], "outputs": [{"output_type": "stream"}], "metadata": {}},
            {"cell_type": "code", "source": "y = 2", "outputs": [], "metadata": {"source_hash": "0" * 64}},
            {"cell_type": "markdown", "source": "# title", "metadata": {}},
        ]
    }
    assert stamp_document(document) == 2
    assert document["cells"][0]["metadata"]["source_hash"] == compute_source_hash("x = 1\n")
    assert "source_hash" not in document["cells"][1]["metadata"]
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_stamp_notebook_source_hashes.py`

Expected: collection fails because `scripts.stamp_notebook_source_hashes` does
not exist.

- [ ] **Step 3: Implement canonicalization, mutation, and atomic writes**

```python
def logical_source(source: object) -> str:
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(part, str) for part in source):
        return "".join(source)
    raise NotebookStampError("cell source must be a string or list of strings")


def compute_source_hash(source: object) -> str:
    return hashlib.sha256(logical_source(source).encode("utf-8")).hexdigest()
```

`stamp_path` must parse and validate before creating a sibling temporary file,
copy the original permission bits, `os.replace` only after a complete write,
and skip writing when no bytes changed.

- [ ] **Step 4: Add idempotence, preservation, invalid-input, and inventory tests**

Require byte-identical second stamping, unchanged file bytes after malformed
input, exact exclusion of `notebooks/archive/`, and discovery only through
`active_task_dirs` in `scripts/verify_repo_config.yaml`.

- [ ] **Step 5: Run GREEN and commit**

Run: `pytest -q tests/test_stamp_notebook_source_hashes.py`

Expected: all tests pass.

Commit: `feat(notebooks): add deterministic source hash stamper`

## 12.36.3 Task 2: Make E8 fail closed over all active notebooks

**Files:**

- Modify: `scripts/verify_repo.py`
- Modify: `tests/test_verify_repo.py`

**Interfaces:**

- Consumes: `compute_source_hash(source: object) -> str`
- Produces findings: `E8.source_hash_missing`, `E8.source_hash_invalid`,
  `E8.stale_output`, and `E8.source_hash_orphan`, all error severity.

- [ ] **Step 1: Add failing verifier fixtures**

Create active notebook fixtures for:

```python
("outputs present, marker absent", "E8.source_hash_missing")
("outputs present, marker malformed", "E8.source_hash_invalid")
("outputs present, marker for old source", "E8.stale_output")
("outputs absent, marker present", "E8.source_hash_orphan")
```

Also prove a current marker passes and the same malformed cell under
`notebooks/archive/` is not inspected.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_verify_repo.py -k 'source_hash or stale_output'`

Expected: missing/malformed/orphan tests fail and stale severity remains a
warning.

- [ ] **Step 3: Implement one E8 helper over `_iter_notebooks(repo)`**

Replace the Tier-A-only optional loop with active-inventory enforcement. Check
non-empty `outputs` first, distinguish absent from invalid markers, compare
valid markers with `compute_source_hash(cell.source)`, and reject markers on
outputless code cells.

- [ ] **Step 4: Run fixture GREEN and commit**

Run: `pytest -q tests/test_verify_repo.py -k 'source_hash or stale_output'`

Expected before migration: all fixture tests pass. Defer the live repository
inventory assertion until Task 4 so no commit intentionally leaves the test
suite RED.

Commit the enforcement and its unit tests separately from notebook metadata:
`test(execution): require active notebook freshness hashes`.

## 12.36.4 Task 3: Stamp every successful Papermill output

**Files:**

- Modify: `Makefile`
- Modify: `tests/test_makefile_contract.py`

**Interfaces:**

- Consumes CLI: `python scripts/stamp_notebook_source_hashes.py NOTEBOOK...`
- Produces: post-success stamping in `run-tier-a`, `smoke-tier-a`,
  `smoke-tier-b`, and `smoke-tier-c`.

- [ ] **Step 1: Add failing recipe and behavioral tests**

Require every Papermill loop to call the configurable
`SOURCE_HASH_STAMPER` after execution. Use a valid minimal notebook in the fake
Papermill tests and assert its output-bearing cell receives the expected hash.
Use a failing fake Papermill process and assert the stamper sentinel is absent.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_makefile_contract.py -k 'source_hash or smoke_output or tier_a_output'`

Expected: no recipe invokes the stamper.

- [ ] **Step 3: Add the Make variable and ordered calls**

```make
SOURCE_HASH_STAMPER ?= $(PYTHON) scripts/stamp_notebook_source_hashes.py
```

Invoke `$(SOURCE_HASH_STAMPER) "$$nb"` after in-place Tier A success and
`$(SOURCE_HASH_STAMPER) "$$out"` after each smoke-output success.

- [ ] **Step 4: Run GREEN and commit**

Run: `pytest -q tests/test_makefile_contract.py`

Expected: all Makefile contract tests pass.

Commit: `build(notebooks): stamp source hashes after execution`

## 12.36.5 Task 4: Migrate all retained active outputs

**Files:**

- Modify: all 29 active `.ipynb` files containing retained outputs.
- Test: `tests/test_stamp_notebook_source_hashes.py`
- Test: `tests/test_verify_repo.py`

**Interfaces:**

- Consumes CLI: `python scripts/stamp_notebook_source_hashes.py --all-active`
- Produces: 189 valid `metadata.source_hash` markers and no orphan markers.

- [ ] **Step 1: Record the pre-migration inventory**

Run the inventory helper and require 29 notebooks, 189 output-bearing cells,
and zero existing markers.

- [ ] **Step 2: Run the all-active migration once**

Run: `python scripts/stamp_notebook_source_hashes.py --all-active`

Expected: 189 markers written across 29 notebooks.

- [ ] **Step 3: Prove surgical preservation**

For every changed notebook, remove only `metadata.source_hash` from the working
document and require exact structural equality with `HEAD`. Independently
require unchanged code/markdown source, outputs, execution counts, cell IDs,
notebook metadata, and all other cell fields. Run the stamper again and require
zero byte changes.

- [ ] **Step 4: Commit the metadata-only migration**

Commit all and only the 29 migrated notebooks so the metadata rollback stays
independent of implementation and test enforcement:
`notebooks: stamp retained output source hashes`.

- [ ] **Step 5: Add the live repository inventory assertion**

The assertion must report 29 active notebooks, 189 output-bearing cells, 189
current valid markers, zero missing/invalid/stale markers, and zero orphan
markers after migration. Commit the guard separately as
`test(execution): lock live source hash inventory`.

- [ ] **Step 6: Run the inventory and verifier GREEN**

Run:

```bash
pytest -q tests/test_stamp_notebook_source_hashes.py
pytest -q tests/test_verify_repo.py -k 'source_hash or stale_output'
python scripts/verify_repo.py --check execution --fast
```

Expected: 29 notebooks, 189 valid markers, zero E8 findings.

- [ ] **Step 7: Confirm both rollback boundaries**

Require the migration commit to contain notebooks only and the live-inventory
guard commit to contain tests only.

## 12.36.6 Task 5: Document the enforced refresh contract

**Files:**

- Modify: `CHANGELOG.md`
- Modify: `CONTRIBUTING.md`
- Modify: `README.md`
- Modify: `docs/conventions.md`
- Modify: `docs/notebook-infrastructure.md` through its canonical generator if required.
- Modify: `docs/maintenance/overnight-2026-07-04.md`
- Modify: `docs/manifest.yaml`
- Modify: `tests/test_check_docs.py`

**Interfaces:**

- Produces: one documented algorithm, execution flow, exception boundary, and
  rollback split across repository/site/wiki surfaces.

- [ ] **Step 1: Add documentation assertions and run RED**

Require the canonical docs to name `metadata.source_hash`, SHA-256, UTF-8
logical source, source-array concatenation, post-success stamping, active-only
enforcement, and archive/outputless exclusions.

- [ ] **Step 2: Update canonical documentation and maintenance state**

Mark OM-010 resolved with the 29/189 migration evidence. Register this plan as
section 12.36. Do not document output-byte hashing or a general exemption tag.

- [ ] **Step 3: Run docs GREEN and commit**

Run:

```bash
pytest -q tests/test_check_docs.py
make docs-check
make docs-wiki
```

Expected: tests, strict MkDocs build, and wiki dry-run pass.

Commit: `docs: publish notebook freshness hash contract`

## 12.36.7 Task 6: Qualify, review, and publish

- [ ] Run focused stamper, verifier, Makefile, docs, and NNx-surface suites.
- [ ] Run `make test`, `make verify`, `make lint`, `make docs-check`, and
  `git diff --check` from a clean feature head.
- [ ] Execute Tier A to temporary output and validate all artifacts are stamped.
- [ ] Execute Tier B and Tier C to temporary output on Atlas JupyterHub, validate
  every output-bearing code cell is stamped, then shut Atlas down while
  preserving volumes and host-native Ollama.
- [ ] Obtain independent review and fix only reproduced findings.
- [ ] Re-run exact-head qualification, push the feature branch, and enable the
  Tier B PR smoke label.
- [ ] Merge feature → `develop`, release → `main`, and zero-content sync →
  `develop`; wait for required CI and Pages evidence.
- [ ] Close #71, check #71 on #53, set the project item Done, delete the feature
  branch, remove only Issue #71 temporary artifacts, and verify clean identical
  `main`/`develop` trees with no open PRs or Atlas containers.
