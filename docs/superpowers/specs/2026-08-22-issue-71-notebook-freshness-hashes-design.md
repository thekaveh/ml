# 12.35 Issue 71 Notebook Source/Output Freshness Hashes Design

## 12.35.1 Purpose and observed state

The 29 active notebooks contain 341 code cells. Of those, 189 retain one or
more outputs, but none records the optional `metadata.source_hash` consumed by
the existing E8 verifier path. E8 therefore cannot distinguish a preserved
output whose source is current from one whose source changed after execution.

Issues 68–70 completed the accepted active-notebook source migrations before
this contract is enabled. Issue 71 is the one-time transition from optional,
Tier-A-only drift hints to a required active-notebook freshness marker.

## 12.35.2 Decision and rejected alternatives

Each active code cell with a non-empty `outputs` list records a bare lowercase
SHA-256 hexadecimal digest in `metadata.source_hash`. The marker stays beside
the output it qualifies, survives notebook moves, and uses the verifier field
that already exists. Code cells without outputs must not carry the marker.

Two alternatives are rejected:

1. A notebook-level cell-index map would couple freshness evidence to cell
   order and require a second synchronization structure.
2. A repository-level manifest would keep notebooks visually cleaner but
   separate each output from its evidence and create another inventory that
   can drift.

No output bytes are hashed. Output rendering can be nondeterministic and is
not part of the freshness claim.

## 12.35.3 Canonical source and marker contract

The logical source string is the notebook JSON `source` string unchanged. If
JSON stores `source` as a list of strings, the strings are concatenated in
order with no separator, matching the notebook-format logical source value.
The UTF-8 bytes of that exact string are hashed with SHA-256 and serialized as
64 lowercase hexadecimal characters. Whitespace, comments, magics, and line
endings are intentionally significant; there is no AST parsing or newline
normalization.

E8 inspects every notebook under the authoritative `active_task_dirs`
inventory. An output-bearing code cell fails verification if its marker is
missing, is not a 64-character lowercase hexadecimal string, or differs from
the current source digest. A code cell without outputs fails if it retains an
orphan marker. Markdown and raw cells never carry source hashes.

The only exceptions are structural and explicit: notebooks under
`notebooks/archive/` are outside the active inventory, and active code cells
without outputs have nothing to qualify. There is no tag, wildcard, or
path-based escape hatch for an output-bearing active cell.

## 12.35.4 Stamping and migration flow

`scripts/stamp_notebook_source_hashes.py` accepts explicit notebook paths for
post-execution use and `--all-active` for the one-time repository migration.
It reads notebook JSON without `nbformat` coercion, updates only
`cell.metadata.source_hash`, and writes atomically with the repository's
existing one-space JSON indentation. Invalid notebook structure fails before
replacement. Repeated stamping is byte-identical.

All Papermill targets invoke the stamper only after Papermill exits
successfully. `run-tier-a` stamps the deliberately refreshed in-place source;
Tier A/B/C smoke targets stamp only their temporary output notebooks. Failed
or partial execution never receives a current marker.

The one-time migration stamps the 189 retained-output cells in all 29 active
notebooks without changing output bytes, execution counts, code, prose, cell
IDs, or unrelated metadata. This is an explicit metadata migration over the
source-frozen state after Issues 68–70; the issue's non-goal forbids expensive
reruns solely to add metadata. Tier A/B/C execution still qualifies the
execution-time hook and current runtime contract.

## 12.35.5 Testing, documentation, and release

Red/green tests cover string/list canonicalization, output-bearing stamping,
orphan removal, idempotence, atomic failure, archive exclusion, and Makefile
post-execution ordering. Verifier tests cover missing, malformed, stale,
current, orphaned, and archived markers. A repository inventory assertion
requires every active output-bearing cell to be valid after migration.

CONTRIBUTING, conventions, notebook execution documentation, the maintenance
ledger, and the changelog describe the algorithm and refresh workflow. Full
qualification includes focused stamper/verifier/Makefile tests, complete
pytest, Ruff, strict docs, repository verification, and non-mutating Tier
A/B/C execution. PyG execution uses Atlas JupyterHub with host-native Ollama
only, followed by volume-preserving shutdown.

After independent review, publish through feature to `develop`, `develop` to
`main`, and content-neutral `main` back to `develop`. Rollback can revert E8
enforcement independently from the notebook metadata commit; the execution
stamper remains safe while enforcement is disabled.
