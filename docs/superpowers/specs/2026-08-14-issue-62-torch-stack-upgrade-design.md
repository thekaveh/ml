# 12.21 Issue 62 Coordinated Torch Stack Upgrade Design

## 12.21.1 Purpose

The repository's local, CI, Docker, and Codespaces runtime pins Torch 2.4.1 together with
TorchVision 0.19.1, TorchAudio 2.4.1, PyTorch Lightning 2.4.0, TorchMetrics 1.4.2, PyG 2.6.1,
four legacy compiled PyG extensions, and an open-ended `torchao>=0.17` requirement. That stack is a
single compatibility contract: changing one member without qualifying the others can produce
binary-extension ABI failures, silently skipped graph tests, unintended Linux CUDA installs, or
an unusable quantization import.

Issue #62 replaces that contract with one reviewed and reproducible matrix. The change must
advance the vulnerable Torch and Lightning pins, make the required torchao API importable in the
canonical environment, preserve the NNx 0.2.0 consumer contract, and prove the graph and notebook
surfaces on Linux CPU and Darwin arm64. It does not promote the complete quantization notebook,
change Atlas, or broaden the supported accelerator matrix.

## 12.21.2 Selected compatibility matrix

The selected matrix is:

| Component | Selected version | Contract rationale |
| --- | ---: | --- |
| Python | 3.11 | Existing repository, CI, devcontainer, and notebook-kernel baseline |
| Torch | 2.11.0 | Latest coherent line with the required TorchAudio pair, mature PyG extension wheels, and proven local QAT evidence |
| TorchVision | 0.26.0 | Official Torch 2.11 companion |
| TorchAudio | 2.11.0 | Official Torch 2.11 companion |
| PyTorch Lightning | 2.6.1 | Upstream-recommended patched version before the active `>=2.6.2` supply-chain advisory range |
| TorchMetrics | 1.9.0 | Current stable metric layer compatible with Python 3.11 and the selected Lightning line |
| torchao | 0.18.0 | Current stable release; exact pin replaces the non-reproducible lower bound |
| torch-geometric | 2.8.0.post1 | Current stable PyG patch with Torch 2.11 support and bounded import-warning debt |
| pyg-lib | 0.8.0 | Preferred graph sampling and operator wheel from the Torch 2.11 PyG index |
| torch-scatter | 2.1.2 | Current extension release with Torch 2.11 wheels |
| torch-sparse | 0.6.18 | Torch 2.11 wheel and NeighborLoader fallback with bounded import-warning debt |
| thekaveh-nnx | 0.2.0 | Unchanged root and Atlas-compatible consumer contract established by Issue #61 |

Primary compatibility evidence is the upstream
[PyTorch release matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md),
[Torch 2.11 release announcement](https://dev-discuss.pytorch.org/t/pytorch-2-11-0-general-availability/3328),
[torchao 0.18 release](https://github.com/pytorch/ao/releases/tag/v0.18.0),
[PyG 2.8 changelog](https://github.com/pyg-team/pytorch_geometric/blob/master/CHANGELOG.md#280---2026-06-05),
[PyG installation guidance](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html),
[PyTorch 2.11 support issue](https://github.com/pyg-team/pytorch_geometric/issues/10508),
[PyG extension-migration tracking issue](https://github.com/pyg-team/pytorch_geometric/issues/10716),
[Torch 2.11 `torch.jit.script` warning source](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/jit/_script.py#L1488-L1494),
[torch-sparse 0.6.18 storage decorator](https://github.com/rusty1s/pytorch_sparse/blob/0.6.18/torch_sparse/storage.py#L21),
[PyG 2.8 typing import catch](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/typing.py#L166-L172),
[PyG 2.8 `SelectOutput` decorator](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/nn/pool/select/base.py#L65),
[Torch 2.11 PyG wheel index](https://data.pyg.org/whl/torch-2.11.0+cpu.html), and the
[Lightning security advisory](https://github.com/Lightning-AI/pytorch-lightning/security/advisories/GHSA-w37p-236h-pfx3).
The implementation refreshes these sources before installation because this dated design record
is not a permanent latest-version oracle.

Three clean Darwin arm64 design-review runs refine the selected boundary without claiming that
implementation is complete. Run r1 exposed missing build tooling in the then-selected legacy
extension path. After that tooling correction, r2 completed installation and exposed an overly
strict `pyg-lib` provenance rule, which was corrected independently. Run r3 then accepted the
native `pyg-lib` artifact but failed the clean source-built spline import/ABI gate. In parallel, a
static consumer review found no active notebook import of either legacy package. Together with
PyG 2.8's documented migration of their functionality to `pyg-lib>=0.6.0`, this evidence removes
both separate packages from the selected repository contract rather than weakening the failed
gate.

A fourth clean Darwin arm64 review run, r4, installed the final three-extension matrix and proved
both the pyg-lib-preferred and forced torch-sparse `NeighborLoader` paths. It then found that the
selected torch-sparse 0.6.18 and torch-geometric 2.8.0.post1 import paths invoke deprecated
`torch.jit.script` under Torch 2.11. No warning-free release exists within the approved matrix.
With `simplefilter("always")`, the PyG import produced 19 identical warnings in that run; this is
diagnostic evidence, not a count that the contract fixes permanently.

## 12.21.3 Why Torch 2.11 is the selected line

Torch 2.11 is not selected merely because Atlas currently uses it. It is the intersection of the
repository's actual constraints:

- Python 3.11 wheels exist for the Torch, TorchVision, and TorchAudio trio on Linux CPU and Darwin
  arm64;
- torchao 0.18 supports the required quantization surface and Issue #61 already proved a tiny QAT
  train, conversion, checkpoint, reload, and inference path on this line;
- the official PyG index provides Linux x86_64 and Darwin arm64 or universal wheels for pyg-lib,
  scatter, and sparse;
- PyG 2.8 explicitly supports Torch 2.11; and
- PyG 2.8 deprecates the two legacy package integrations in favor of `pyg-lib>=0.6.0`, matching
  both the upstream installation command and this repository's active consumer surface.

Torch 2.12.1 is rejected for this issue because it raises the Darwin wheel floor, lacks a matching
TorchAudio 2.12 release, and does not improve the repository's currently recorded residual
advisory result. Torch 2.13 is rejected because the corresponding PyG
compiled-extension index is not yet complete. A partial core-only upgrade is rejected because it
would preserve the ecosystem drift that this issue exists to remove.

## 12.21.4 Manifest and installation architecture

The current `torch-core-requirements.txt` mixes the platform-specific Torch trio with packages
served only by PyPI. That shape cannot safely express Linux's CPU-only index: beginning with Torch
2.11, a bare Linux PyPI install can select CUDA wheels, while pointing the mixed file exclusively
at the PyTorch CPU index makes Lightning and TorchMetrics unavailable.

The selected manifest structure is:

1. `torch-core-requirements.txt` contains only exact Torch, TorchVision, and TorchAudio pins.
2. A new `torch-ecosystem-requirements.txt` contains exact Lightning, TorchMetrics, and torchao
   pins.
3. `torch-requirements.txt` includes the ecosystem manifest, the exact Torch 2.11 CPU PyG wheel
   selector, pyg-lib, torch-scatter, torch-sparse, and torch-geometric.
4. `torch-audit-requirements.txt` is the selector-free resolver projection of the core,
   ecosystem, and torch-geometric inputs.
5. `pyg-extension-audit-requirements.txt` remains the pre-resolved extension supplement.
   It contains only the two exact PyPI supplement pins, torch-scatter and torch-sparse. pyg-lib is
   explicitly classified as an exact external-index wheel whose version and provenance are
   verified locally because it is not distributed through PyPI's ordinary project index.
6. `requirements.txt` no longer carries a second torchao constraint. The stack manifests become
   its one authoritative version source.

`make install-torch-stack` performs exactly four ordered stages:

0. run `python -m pip install --upgrade pip` and install no other bootstrap package;
1. install the exact Torch trio from `https://download.pytorch.org/whl/cpu` on Linux, or from the
   native PyPI path on Darwin;
2. install `torch-requirements.txt` in one resolver pass with the exact
   `--only-binary=pyg-lib,torch-scatter,torch-sparse` policy, taking those three selected wheels
   from `https://data.pyg.org/whl/torch-2.11.0+cpu.html` and resolving the ecosystem and
   pure-Python PyG packages from PyPI; and
3. install the remaining root requirements without reinstalling or replacing the qualified stack.

Docker and Codespaces reuse this order. CI invokes the Make target rather than implementing a
second package algorithm. Every cache that can execute or audit the stack includes all affected
manifests.

## 12.21.5 Platform contract

The supported qualification platforms are Linux x86_64 CPU and Darwin arm64 under Python 3.11.
Because Docker Desktop on an Apple Silicon host builds Linux arm64 by default, the container path
also explicitly qualifies Linux arm64 rather than silently relying on emulation. The implementation
must prove:

- Linux resolves CPU Torch artifacts and no CUDA runtime packages;
- Darwin resolves native arm64 or universal wheels rather than an x86-only fallback;
- native Linux arm64 Docker resolution uses the matching Torch and PyG artifacts;
- pyg-lib, torch-scatter, and torch-sparse each resolve as a compatible Torch 2.11 wheel; and
- `python -m pip check` succeeds after the complete repository install.

No CUDA, ROCm, XPU, Windows, direct-host Linux arm64, or source-built Torch support claim is added.
Failure of any of the three selected PyG wheel contracts on a supported native or container
platform returns the work to design review; the implementation does not mask the failure.

## 12.21.6 Runtime verifier

A focused, fail-closed stack verifier becomes the reusable installation oracle. It derives the
expected versions from the canonical manifests and checks:

- distribution metadata and direct imports for exactly Torch, TorchVision, TorchAudio, Lightning,
  TorchMetrics, torchao, PyG, pyg-lib, scatter, and sparse;
- Torch CPU/CUDA provenance appropriate to the platform;
- extension ABI compatibility through representative scatter and sparse operations rather than
  accepting imports alone;
- a real one-batch PyG `NeighborLoader` path that selects pyg-lib when available and exercises the
  torch-sparse fallback independently; and
- the exact NNx 0.2.0 distribution and import through the existing canonical-wheel verifier.

Version comparison uses the manifest's public version as the package identity while separately
validating PyG wheel provenance. Linux PyG-index local versions require the exact `pt211cpu` tag.
Darwin arm64 accepts the exact `pt211` local tag or an absent local tag. An absent local tag is
accepted only after independent WHEEL/RECORD, Python ABI/platform architecture, and mandatory
runtime-canary checks pass. Darwin artifacts must carry a native arm64 or universal macOS wheel
tag. The verifier does not treat either exact-string inequality or import success alone as
compatibility evidence.

Diagnostics disclose stable component and failure categories, not credentials, private index
tokens, temporary paths, or captured installer output. CI and Docker call the same verifier after
`pip check` and before repository tests or notebook workloads.

The verifier has one local import-boundary exception with two immutable debt keys:

- Torch exactly 2.11.0 with outer component torch-geometric exactly 2.8.0.post1; or
- Torch exactly 2.11.0 with outer component torch-sparse exactly 0.6.18.

These values are explicit verifier constants independent of generic manifest-derived expectations.
Any Torch or outer-component version drift fails the exception even when the warning category,
message, and origin still match. The verifier wraps each selected `hooks.import_module()` call in a
fresh warning capture with `simplefilter("always")`. Capturing zero warnings is normal success and
does not exercise the exception; a transitive import, repeated call, or different selected-import
order may legitimately reuse Python's module cache. Production verification never deletes or
replaces entries in `sys.modules`. Any nonempty captured group for either immutable debt key is
accepted only when every record has all three exact properties:

- `record.category is DeprecationWarning`, so subclasses do not match;
- `str(record.message)` equals `` `torch.jit.script` is deprecated. Please switch to
  `torch.compile` or `torch.export`. `` including punctuation; and
- the resolved `record.filename` equals the exact selected Torch-owned warning file described
  below.

Any nonempty group for every other outer component fails.

The expected warning file is derived from the selected Torch distribution's `Distribution.files`
inventory. Exactly one RECORD-owned `PackagePath` must have POSIX path
`torch/jit/_script.py`, and it must resolve through that selected distribution's location and
ownership to one concrete file. Missing, duplicate, unlocatable, or non-owned entries fail closed.
Warning-origin equality compares the fully resolved paths exactly; basename, suffix, package-root,
and other path heuristics are prohibited. Source line number is not part of the predicate.

The entire captured group must match; a mixed group fails. The wrapper consumes only that exact
group before returning the imported module, so the CLI's outer warning capture remains strict.
The existing `_run_warning_free` boundaries around scatter, sparse, the real sampler, and NNx are
unchanged. Pytest still runs with `-W error`; JUnit still requires a positive test count and zero
failures, errors, and skips; both graph backends remain mandatory; and any warning outside this
single import wrapper fails. No pytest, global, environment, or conftest warning filter is allowed.

Verifier tests accept an exact one-or-more matching group and reject a wrong category or subclass,
message or punctuation, origin path, outer component, a mixed group, and a matching warning emitted
outside the import boundary. Version mutations independently change Torch, torch-geometric, and
torch-sparse while keeping the warning records exact; each must fail the relevant debt key. Origin
mutations include same-basename and matching-suffix files outside the selected Torch inventory.
Mutation tests also reject broad message prefixes, `issubclass`, omitted origin validation, a
wrapper applied to every import, and any broader filter.

Cached, repeated, and order-sensitive fakes prove that zero warnings remain ordinary success and
that any nonempty group must match completely regardless of which debt-key component imports first.
They never evict a production module. Separately, acceptance starts a fresh interpreter in which
neither `torch_geometric` nor `torch_sparse` is preloaded, imports the actual selected
torch-geometric boundary, and requires a nonempty exact group without fixing its count. The same
fresh-r4 acceptance then proves the pyg-lib-preferred and forced torch-sparse sampler paths under
the unchanged warning-as-error and JUnit gates.

This exception is temporary upstream compatibility debt. Every Torch or PyG upgrade requires
explicit design review; no changed version inherits either debt key. Retirement is decided only by
the fresh-interpreter probe, never by a cached zero-warning import. When that probe becomes
warning-free, the exception and its debt-specific machinery are removed rather than carried
forward.

## 12.21.7 PyG compatibility boundary

The Reddit graph notebooks and NNx surface tests require a working neighborhood sampler. Existing
tests can skip if pyg-lib and torch-sparse are both unusable, which is appropriate for optional
developer environments but insufficient for this upgrade's Linux acceptance evidence.

The canonical stack therefore treats compiled graph support as mandatory:

- pyg-lib is the preferred backend;
- torch-sparse remains an independently importable fallback;
- SAGE, GraphConv, and GAT execute against a real sampled batch;
- scatter and sparse execute tiny representative operations; and
- the Linux CI contract rejects a skip caused by missing or ABI-incompatible extensions.

The active notebooks do not import the separate `torch-cluster` package or exercise `SplineConv`;
those unused operators are not Issue #62 consumer-acceptance scope. PyG may supply
their functionality through `pyg-lib` when a future repository consumer requires it, but that
consumer must introduce its own explicit runtime canary and platform validation before the scope
can be claimed. This boundary removes unused legacy packages; it does not claim untested operator
coverage.

The implementation does not edit Reddit notebook code or committed outputs merely to accommodate
the dependency upgrade. A consumer-visible API or behavior failure is diagnosed against the
selected matrix and corrected only when the notebook is relying on a stale public contract.

## 12.21.8 Quantization boundary

The canonical Torch 2.11 stack removes the current `torch.int1` import incompatibility. The
existing quantization surface test changes from backend-skippable to fail-closed and proves the
NNx PTQ and QAT imports plus a tiny prepare, train-step, convert, and inference path.

The complete `quantization-mnist-ffnn-pytorch` notebook remains manual-only and outside Tier A,
Tier B, and Tier C. Issue #66 owns any later notebook execution, output refresh, threshold review,
or tier promotion. Issue #62 updates the current rationale from "the canonical stack cannot
import torchao" to "the dependency surface is qualified, while complete notebook acceptance and
tiering remain separately gated." Historical Torch 2.8 outputs and Issue #61 side-environment
evidence remain explicitly historical.

## 12.21.9 Security and advisory reconciliation

The accepted-advisory policy binds Torch 2.4.1 and Lightning 2.4.0. The coordinated upgrade
therefore triggers intentional accepted-version drift and cannot land separately from a fresh
security review.

The implementation runs the complete four-surface advisory audit after the final clean solve and
updates `security/accepted-advisories.json` and the current human ledger together. It records:

- exact manifest and JSON hashes;
- resolved package counts and versions;
- raw feed records and alias-aware identities;
- disappeared, retained, re-keyed, and new findings;
- explicit risk acceptance for any no-fix or not-yet-fixed identity; and
- pyg-lib's external-index provenance and PyPI-audit limitation. The pre-resolved extension
  supplement contributes only torch-scatter and torch-sparse to the combined-runtime and Torch
  advisory projections.

The upgrade does not claim zero vulnerabilities merely because most recorded fix floors are below
Torch 2.11. At least one existing record lists Torch 2.13 as its fix and several records list no
fix. Feed disappearance is reconciliation evidence, never proof of remediation or
non-reachability. The Lightning 2.6.1 pin also records why newer published 2.6 releases are not
accepted while the upstream supply-chain advisory remains open.

## 12.21.10 CI and Docker contract

No new protected-branch context is required. The existing required jobs remain the integration
gate, and the focused jobs remain diagnostic evidence. Workflow contract tests enforce:

- the exact platform-aware install target and complete cache manifest set;
- `pip check` and stack verification after the final install;
- absence of later package installation before the workload;
- Linux CPU provenance and no CUDA package selection;
- compatible pyg-lib, torch-scatter, and torch-sparse wheels with no failure masking or optional
  compiled-extension skips in canonical jobs;
- only the verifier-local, exact import-warning exception, with no workflow, pytest, environment,
  or conftest warning filter;
- no Atlas initialization or service, container, Ollama, or ComfyUI startup; and
- unchanged binary-only NNx selection and canonical verification.

The Docker image uses the same four-stage manifest order, then runs `pip check` and the stack
verifier, whose scatter, sparse, and real-sampler canaries provide the tiny PyG runtime probe. CI
proves Linux x86_64 and the Apple Silicon local path proves a native Linux arm64 build; neither is
replaced by an unrecorded emulated architecture. A successful image build without those runtime
checks is not sufficient evidence.

## 12.21.11 Repository and notebook validation

The mandatory clean-environment sequence is:

1. install the selected matrix under Python 3.11;
2. run `python -m pip check` and the stack verifier;
3. run `make audit-advisories` and the focused advisory/parser contracts;
4. run `make verify-nnx-install` and `make test-nnx-surface`;
5. run `make test`, `make lint`, and `make verify`;
6. run `make docs-check`, `make docs-wiki`, and strict MkDocs validation;
7. build and probe the Docker image;
8. run complete Tier A, Tier B, and Tier C workloads; and
9. prove source notebooks, tracked files, the Atlas gitlink, and generated-output boundaries remain
   clean.

Implementation inserts a separately reviewed Task 2.1 verifier correction before Task 3 resumes.
Task 2.1 changes only the verifier and its owned tests for the import-warning boundary above. Task
3 retains its existing dirty work-in-progress files, then reruns the full clean-r4 sequence from a
fresh environment, including selected imports, both real sampler paths, `pytest -W error`, and the
zero-failure/error/skip JUnit gate before continuing.

Tier A must complete 18/18, Tier B 6/6, and Tier C 4/4. Tier B is enabled on the feature pull
request through its label-controlled job; Tier C runs through manual workflow dispatch on the
exact feature SHA. The ordinary feature PR alone is not complete evidence because those jobs are
not both unconditional.

## 12.21.12 Documentation surfaces

Current documentation changes cover:

- the selected matrix and platform-specific installation path;
- local, Docker, Codespaces, CI, and default remote-Atlas boundaries;
- the now-importable but still manual-only quantization surface;
- the PyG sampler and three-wheel compiled-extension acceptance contract, including the explicit
  exclusion of unused legacy operators;
- the temporary exact-match TorchScript import-warning debt and its upgrade-time removal trigger;
- the refreshed advisory snapshot and residual risk;
- contributor and rollback procedures; and
- durable Unreleased history in `CHANGELOG.md`.

Historical Issue #59/#60/#61 specifications, plans, snapshots, released changelog entries,
maintenance records, Atlas probe evidence, and committed notebook outputs are not mass-rewritten.
The Issue #62 design and implementation plan become manifest entries 12.21 and 12.22. Generated
site and wiki trees remain ignored derivatives and are never edited directly.

## 12.21.13 Atlas and external-runtime boundary

Issue #62 makes no change to `.gitmodules`, the `infra` gitlink, Atlas requirements, Atlas images,
consumer compose, track selection, or service policy. The observed Atlas Torch 2.11 surface is
useful compatibility evidence, but Atlas remains an independently owned runtime whose pin changes
belong to Issue #65.

No live Atlas run is required for a root-only stack upgrade. If a later acceptance concern truly
requires it, validation uses the existing remote Jupyter kernel contract, host-native Ollama only,
and volume-preserving normal shutdown. Containerized Ollama is prohibited.

## 12.21.14 Alternatives considered

### 12.21.14.1 Upgrade only the Torch trio

This minimizes the manifest diff but retains stale Lightning, TorchMetrics, PyG, and open-ended
torchao behavior. It cannot prove a coherent runtime and does not satisfy Issue #62.

### 12.21.14.2 Select Torch 2.12.1

This is newer but raises the Darwin deployment floor, has no matching TorchAudio 2.12 release, and
moves away from the selected PyG 2.11 wheel surface. It does not clear the remaining
Torch-2.13/no-fix advisory boundary and offers no repository-visible capability required by the
issue.

### 12.21.14.3 Select Torch 2.13

This reaches the highest recorded Torch fix floor, but the required PyG compiled-extension wheel
surface is incomplete and the repository has no equivalent graph/notebook qualification evidence.
Security progress that breaks the graph runtime is not an acceptable upgrade.

### 12.21.14.4 Keep quantization in a side environment

This preserves the current local stack but leaves its accepted Torch findings and canonical
torchao incompatibility untouched. A side environment remains useful evidence, not a substitute
for the coordinated root upgrade.

### 12.21.14.5 Retain every legacy PyG extension as a separate package

This preserves the older manifest shape but conflicts with the PyG 2.8 migration boundary, adds
packages that no active notebook imports, and keeps a clean Darwin failure in the mandatory path.
Weakening or bypassing that failed import would make the contract less trustworthy. The selected
design instead validates the three upstream-listed wheels and requires future consumers of the
migrated operators to add focused acceptance evidence when they appear.

### 12.21.14.6 Suppress TorchScript deprecations globally

A pytest, environment, conftest, or process-wide filter would hide unrelated warnings and weaken
the repository's warning-free acceptance gate. The selected exception instead proves component,
category identity, complete message, and owned origin inside one import call, then leaves every
outer warning boundary unchanged.

## 12.21.15 Failure handling and rollback

The candidate is rejected or returned to design review if:

- official release or wheel evidence changes materially;
- clean Linux or Darwin resolution, `pip check`, or provenance fails;
- any selected compiled extension cannot resolve as the required wheel, import, or execute its
  canary;
- any import warning falls outside the exact temporary group or appears outside its local wrapper;
- PyG sampling, NNx 0.2.0, torchao PTQ/QAT, repository tests, Docker, or a notebook tier fails;
- Linux selects CUDA packages;
- an advisory identity or accepted version is not explicitly reconciled;
- Atlas-owned state changes; or
- the three documentation surfaces cannot remain deterministic and self-contained.

Rollback reverts the complete runtime contract as one unit: core, ecosystem, PyG, and audit
manifests; installer and verifier logic; CI and Docker contracts; security policy and current
ledger; and current operational documentation. It restores the executable baseline at
`df0652b48554b1c3562177ce9e3ee28084be88cf` in a fresh environment or rebuilt image. The design
record remains durable, and a post-merge rollback appends changelog and dependency-ledger history
that records the attempted upgrade, evidence, reason, and restored matrix rather than erasing the
event. An in-place downgrade is prohibited because Torch 2.11 extension wheels and cached artifacts
can survive and produce misleading ABI results. The Atlas gitlink is unchanged in both directions.

## 12.21.16 GitFlow integration and completion evidence

Implementation occurs only on `codex/issue-62-torch-stack-upgrade`. After local review and full
validation, the exact feature SHA is pushed and merged by pull request into `develop`. A separate
`develop`-to-`main` pull request publishes the accepted result. Main and develop are then audited
for content equality and synchronized through the protected-branch flow if necessary.

Issue #62 is complete only after required feature and release checks are green, Tier A/B/C evidence
is attached to the issue, the issue and project item are closed, documentation is published, and
feature branches, obsolete PRs, temporary worktrees, disposable environments, and repository-owned
services are cleaned up.
