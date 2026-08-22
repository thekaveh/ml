# Task 1 report: deterministic atomic notebook stamper

Status: DONE
Commit SHA: e54dc8b36034c532a671d4c3cb4bb26e91b8408f

## RED

Command: `pytest -q tests/test_stamp_notebook_source_hashes.py`

Outcome: failed during collection with `ModuleNotFoundError` because
`scripts/stamp_notebook_source_hashes.py` did not yet exist.

## GREEN

Command: `pytest -q tests/test_stamp_notebook_source_hashes.py`

Outcome: 17 passed.

Additional verification:

- `ruff check scripts/stamp_notebook_source_hashes.py tests/test_stamp_notebook_source_hashes.py` passed.
- `git diff --check` passed.
- Active inventory discovery returned 29 notebooks and no archive paths.

## Files changed

- `scripts/stamp_notebook_source_hashes.py`
- `tests/test_stamp_notebook_source_hashes.py`
- `.superpowers/sdd/task-1-report.md`

## Self-review

The implementation validates notebook structure before creating a sibling
temporary file, hashes exact UTF-8 logical source, updates only output-bearing
code-cell markers, removes orphan markers, preserves permissions, and uses a
fully written atomic replacement. Repeated stamping is byte-stable, and
inventory discovery is deterministic and sourced only from
`active_task_dirs`. Tests cover canonicalization, marker behavior, atomic
failure cleanup, preservation, permissions, inventory/archive behavior, and
CLI argument rules.

## Concerns

None.
