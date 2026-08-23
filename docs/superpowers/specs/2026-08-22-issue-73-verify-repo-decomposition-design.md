# 12.39 Issue 73 Repository Verifier Decomposition Design

## 12.39.1 Objective and measured starting point

Decompose `scripts/verify_repo.py` into cohesive, directly testable validators
without changing verifier behavior. The script remains the public command-line
entry point and compatibility facade. Existing command arguments, check order,
JSON schema, finding identifiers, diagnostics, exit codes, output-file behavior,
configuration loading, and subprocess timeout semantics remain exact.

At synchronized `develop` commit
`411a04af23c3bf6be24c6246e93ef3e7ce14ba24`, the script is 3,788 lines. Its
largest orchestration functions are `check_execution` (307 lines and 46 AST
branch nodes), `_dependency_ledger_findings` (257/36), `check_structure`
(232/57), and `check_docs` (198/35). The matching
`tests/test_verify_repo.py` module is 8,045 lines. These measurements supersede
the older 2,455-line figure in the issue description.

## 12.39.2 Considered approaches

### 12.39.2.1 Mechanical moves with facade re-exports

Move functions to modules, import them into `verify_repo.py`, and otherwise
leave their globals unchanged. This minimizes edits, but imported functions
would read their defining module's globals. Existing config reloads and tests
that monkeypatch facade-level `_run`, `_runtime_available`,
`_phase3_code_cells_unchanged`, or task inventories would silently stop
controlling validator behavior.

### 12.39.2.2 Validator class or plugin framework

Introduce a base validator class, registration decorators, and lifecycle
objects. This could support future third-party checks, but no such extension
contract is requested. It would combine a new framework with the extraction,
make rollback harder, and violate the issue's no-wholesale-rewrite constraint.

### 12.39.2.3 Compatibility facade with explicit context and hooks (selected)

Keep configuration and CLI serialization in `verify_repo.py`. Extract data
models, shared primitives, and the five validator responsibilities into a
package. Facade wrappers snapshot the current config globals into an immutable
context and pass the currently bound runtime hooks to validators that execute
commands. This makes dependencies explicit while retaining current import and
monkeypatch behavior.

## 12.39.3 Module responsibilities and interfaces

Create `scripts/repo_verifier/` with these ownership boundaries:

- `models.py` owns `Finding`, `CheckResult`, and immutable `VerifierConfig`.
- `common.py` owns repository text/notebook iteration and subprocess execution
  primitives that are shared by more than one validator.
- `structure.py` owns structure rules, import discovery, Markdown-link
  resolution, and `check_structure(repo, config=...)`.
- `assets.py` owns NLP asset detection and
  `check_assets(repo, config=...)`.
- `docs.py` owns documentation, dependency-ledger, action-pin, and runtime
  contract rules plus `check_docs(repo, config=..., run=...)`.
- `comments.py` owns comment-hygiene scanning,
  `check_comments(repo, config=...)`, and
  `export_phase_b_candidates(repo, out_path, config=...)`.
- `execution.py` owns notebook execution metadata, Atlas static contracts,
  freshness hashes, baseline comparisons, shellcheck/submodule checks, and
  `check_execution(repo, fast, config=..., run=...,
  runtime_available=..., phase3_check=...)`.

`VerifierConfig` contains exactly `active_task_dirs`, `required_sections`, and
`tier_a_notebooks`. It is a value snapshot, not a mutable global registry.
Command execution uses the existing `(return_code, stdout, stderr)` result
shape and the existing 120-second default timeout. No validator catches or
relabels exceptions that currently propagate.

The facade continues to expose the internal helper names exercised by the
current test contract. Pure helpers may be direct aliases. Helpers whose
behavior depends on config or runtime hooks remain thin wrappers so changing a
facade global or monkeypatch still affects the next call. `CHECKS` retains the
current insertion order: structure, assets, docs, comments, execution.

## 12.39.4 CLI and configuration compatibility

`scripts/verify_repo.py` continues to own:

- the six `--check` choices and required-check error;
- `--fast`, `--out`, hidden `--repo-root`, and `--phase-b-out`;
- adjacent and alternate-repository YAML config loading;
- result aggregation and schema version 1 serialization;
- stdout versus output-file behavior;
- stderr summaries and exit status 0/1;
- the Phase-B export diagnostic;
- the executable `sys.exit(main())` boundary.

The existing copied-script `--help` behavior is also compatibility surface.
Argument parsing must be able to exit for help before importing the extracted
package, so copying only `verify_repo.py` to a directory without config or
package files still returns help successfully.

## 12.39.5 Characterization and red/green contract

Before extraction, add characterization tests for the current facade:

- exact check registry order and `all` dispatch order;
- one-check dispatch and execution-only forwarding of `--fast`;
- unchanged JSON summary aggregation and finding serialization;
- zero-error, warning-only, and error exit/diagnostic behavior;
- output-file versus stdout behavior;
- Phase-B bypass behavior;
- the existing timeout return code and partial-output normalization.

Existing focused tests remain authoritative for each finding identifier and
validator rule. Add a separate architecture test that initially fails because
the package and explicit `VerifierConfig` interface do not yet exist. Make it
green with the smallest extraction, then keep it green through each validator
move. Delegation tests must prove facade-level config and runtime-hook
monkeypatches reach the extracted implementations.

No golden copy of the 3,788-line legacy implementation is retained. Behavior
is characterized at public and injected seams rather than by duplicating
production source into fixtures.

## 12.39.6 Extraction and rollback sequence

Use independently revertible commits in dependency order:

1. Add facade characterization tests without production changes.
2. Add models/common primitives and the explicit architecture contract.
3. Extract structure and assets validators.
4. Extract documentation validator.
5. Extract comments and Phase-B export validator.
6. Extract execution validator and runtime hooks.
7. Reduce the facade to CLI/config/delegation, record metrics, and update
   maintainer documentation.

Each extraction commit moves one cohesive responsibility, updates only the
tests needed for that boundary, and leaves the complete focused suite green.
Reverting the latest extraction restores the previous facade-owned
implementation without requiring a feature rollback or config migration.

## 12.39.7 Complexity and size evidence

Record before/after metrics generated from the Python AST and line counts. The
completed change must demonstrate:

- `scripts/verify_repo.py` reduced from 3,788 lines to no more than 450 lines;
- no extracted module larger than 1,200 lines;
- facade validator wrappers no longer contain rule branches;
- the maximum per-file sum of AST branch nodes is lower than the original
  monolith's sum;
- the public check order, finding inventory on the clean repository, and
  verifier exit status remain unchanged.

These are maintainability measurements, not new runtime gates on arbitrary
future rule growth. Document the measured result in
`docs/maintenance/overnight-2026-07-04.md` beside issue #73's disposition.

## 12.39.8 Verification

Qualification runs in this order:

- new facade and architecture characterization tests;
- complete `tests/test_verify_repo.py` plus the new architecture module;
- `python scripts/verify_repo.py --check all --fast`;
- full `pytest tests/`;
- `ruff check .`;
- strict documentation check/build/wiki projection;
- `python -m pip check`;
- a clean diff and source-file inventory check;
- independent code review against this design and the implementation plan;
- GitHub CI, including the repository suite and Tier-A Papermill qualification.

No Atlas service starts. The verifier may inspect the initialized `infra`
submodule exactly as it already does; completion leaves no ml-eng-lab Atlas
containers.

## 12.39.9 Non-goals

This issue does not add finding types, change finding severity, alter notebook
or dependency policy, rename checks, parallelize checks, change timeout values,
introduce plugins, or split unrelated maintenance scripts. The 8,045-line test
module is not broadly reorganized; new architecture tests live separately so
the extraction does not mix production decomposition with a wholesale test
rewrite.
