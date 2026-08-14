# 12.21 Issue 62 Coordinated Torch Stack Upgrade Design

## 12.21.1 Purpose

The repository's local, CI, Docker, and Codespaces runtime pins Torch 2.4.1 together with
TorchVision 0.19.1, TorchAudio 2.4.1, PyTorch Lightning 2.4.0, TorchMetrics 1.4.2, PyG 2.6.1,
four compiled PyG extensions, and an open-ended `torchao>=0.17` requirement. That stack is a
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
| torch-geometric | 2.8.0.post1 | Current stable PyG patch with Torch 2.11 support |
| pyg-lib | 0.8.0 | Preferred graph sampling and operator wheel from the Torch 2.11 PyG index |
| torch-scatter | 2.1.2 | Current extension release with Torch 2.11 wheels |
| torch-sparse | 0.6.18 | Current extension release with Torch 2.11 wheels and existing NeighborLoader fallback |
| torch-cluster | 1.6.3 | Current extension release with Torch 2.11 wheels |
| torch-spline-conv | 1.2.2 | Existing extension contract; retained subject to source-build qualification |
| thekaveh-nnx | 0.2.0 | Unchanged root and Atlas-compatible consumer contract established by Issue #61 |

Primary compatibility evidence is the upstream
[PyTorch release matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md),
[Torch 2.11 release announcement](https://dev-discuss.pytorch.org/t/pytorch-2-11-0-general-availability/3328),
[torchao 0.18 release](https://github.com/pytorch/ao/releases/tag/v0.18.0),
[PyG release history](https://github.com/pyg-team/pytorch_geometric/blob/master/CHANGELOG.md),
[Torch 2.11 PyG wheel index](https://data.pyg.org/whl/torch-2.11.0+cpu.html), and the
[Lightning security advisory](https://github.com/Lightning-AI/pytorch-lightning/security/advisories/GHSA-w37p-236h-pfx3).
The implementation refreshes these sources before installation because this dated design record
is not a permanent latest-version oracle.

## 12.21.3 Why Torch 2.11 is the selected line

Torch 2.11 is not selected merely because Atlas currently uses it. It is the intersection of the
repository's actual constraints:

- Python 3.11 wheels exist for the Torch, TorchVision, and TorchAudio trio on Linux CPU and Darwin
  arm64;
- torchao 0.18 supports the required quantization surface and Issue #61 already proved a tiny QAT
  train, conversion, checkpoint, reload, and inference path on this line;
- the official PyG index provides Linux x86_64 and Darwin arm64 or universal wheels for pyg-lib,
  scatter, sparse, and cluster;
- PyG 2.8 explicitly supports Torch 2.11; and
- the selected line retains the C++17 extension-build boundary used by the current repository.

Torch 2.12.1 is rejected for this issue because it raises the Darwin wheel floor, lacks a matching
TorchAudio 2.12 release, and adds a torch-cluster source build without improving the repository's
currently recorded residual advisory result. Torch 2.13 is rejected because the corresponding PyG
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
   selector, pyg-lib, the four retained compiled extensions, and torch-geometric.
4. `torch-audit-requirements.txt` is the selector-free resolver projection of the core,
   ecosystem, and torch-geometric inputs.
5. `pyg-extension-audit-requirements.txt` remains the pre-resolved extension supplement.
   pyg-lib is explicitly classified as an exact external-index wheel whose version and provenance
   are verified locally because it is not distributed through PyPI's ordinary project index.
6. `requirements.txt` no longer carries a second torchao constraint. The stack manifests become
   its one authoritative version source.

`make install-torch-stack` performs the ordered install:

1. run `python -m pip install --upgrade pip wheel` to upgrade pip and provision wheel;
   `wheel` supplies the `bdist_wheel` command required by the later `--no-build-isolation`
   source build;
2. install the exact Torch trio from `https://download.pytorch.org/whl/cpu` on Linux, or from the
   native PyPI path on Darwin;
3. install `torch-requirements.txt` in one resolver pass with
   `--only-binary=pyg-lib,torch-scatter,torch-sparse,torch-cluster` and
   `--no-binary=torch-spline-conv`, taking the four wheel-only extensions from
   `https://data.pyg.org/whl/torch-2.11.0+cpu.html`, building only torch-spline-conv with build
   isolation disabled, and resolving the ecosystem and pure-Python PyG packages from PyPI; and
4. install the remaining root requirements without reinstalling or replacing the qualified stack.

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
- every wheel-backed PyG extension reports a build compatible with Torch 2.11;
- torch-spline-conv builds and imports in every clean native and container qualification
  environment; and
- `python -m pip check` succeeds after the complete repository install.

No CUDA, ROCm, XPU, Windows, direct-host Linux arm64, or source-built Torch support claim is added.
Failure of the retained torch-spline-conv contract on a supported native or container platform
returns the work to design review; the implementation does not silently drop the package or mask
the failure.

## 12.21.6 Runtime verifier

A focused, fail-closed stack verifier becomes the reusable installation oracle. It derives the
expected versions from the canonical manifests and checks:

- distribution metadata for every selected component;
- Torch, TorchVision, TorchAudio, Lightning, TorchMetrics, torchao, PyG, pyg-lib, scatter, sparse,
  cluster, and spline-conv imports;
- Torch CPU/CUDA provenance appropriate to the platform;
- extension ABI compatibility by executing representative operators rather than accepting an
  import alone;
- a real one-batch PyG `NeighborLoader` path; and
- the exact NNx 0.2.0 distribution through the existing canonical-wheel verifier.

Version comparison uses the manifest's public version as the package identity while separately
validating PyG wheel provenance. Local wheel versions such as `0.8.0+pt211cpu` satisfy the public
`0.8.0` pin only when their local tag, installed `WHEEL` tags, platform architecture, and runtime
operator evidence match Torch 2.11. Darwin extension wheels that publish no local version tag must
carry a native or universal macOS wheel tag and pass the same runtime ABI canary. The verifier does
not treat either exact-string inequality or import success alone as compatibility evidence.

Diagnostics disclose stable component and failure categories, not credentials, private index
tokens, temporary paths, or captured installer output. CI and Docker call the same verifier after
`pip check` and before repository tests or notebook workloads.

## 12.21.7 PyG compatibility boundary

The Reddit graph notebooks and NNx surface tests require a working neighborhood sampler. Existing
tests can skip if pyg-lib and torch-sparse are both unusable, which is appropriate for optional
developer environments but insufficient for this upgrade's Linux acceptance evidence.

The canonical stack therefore treats compiled graph support as mandatory:

- pyg-lib is the preferred backend;
- torch-sparse remains an independently importable fallback;
- SAGE, GraphConv, and GAT execute against a real sampled batch;
- scatter and cluster execute tiny representative operations;
- spline-conv must import and pass a new tiny `torch_geometric.nn.SplineConv` forward canary; and
- the Linux CI contract rejects a skip caused by missing or ABI-incompatible extensions.

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
- pyg-lib's external-index provenance and PyPI-audit limitation.

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
- no failure masking or optional compiled-extension skips in canonical jobs;
- no Atlas initialization or service, container, Ollama, or ComfyUI startup; and
- unchanged binary-only NNx selection and canonical verification.

The Docker image uses the same manifest order, then runs `pip check`, the stack verifier, and a
tiny PyG sampler probe. CI proves Linux x86_64 and the Apple Silicon local path proves a native
Linux arm64 build; neither is replaced by an unrecorded emulated architecture. A successful image
build without those runtime checks is not sufficient evidence.

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

Tier A must complete 18/18, Tier B 6/6, and Tier C 4/4. Tier B is enabled on the feature pull
request through its label-controlled job; Tier C runs through manual workflow dispatch on the
exact feature SHA. The ordinary feature PR alone is not complete evidence because those jobs are
not both unconditional.

## 12.21.12 Documentation surfaces

Current documentation changes cover:

- the selected matrix and platform-specific installation path;
- local, Docker, Codespaces, CI, and default remote-Atlas boundaries;
- the now-importable but still manual-only quantization surface;
- the PyG sampler and compiled-extension acceptance contract;
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

This is newer but introduces another PyG source build, a higher Darwin deployment floor, and no
matching TorchAudio 2.12 release. It does not clear the remaining Torch-2.13/no-fix advisory
boundary and offers no repository-visible capability required by the issue.

### 12.21.14.3 Select Torch 2.13

This reaches the highest recorded Torch fix floor, but the required PyG compiled-extension wheel
surface is incomplete and the repository has no equivalent graph/notebook qualification evidence.
Security progress that breaks the graph runtime is not an acceptable upgrade.

### 12.21.14.4 Keep quantization in a side environment

This preserves the current local stack but leaves its accepted Torch findings and canonical
torchao incompatibility untouched. A side environment remains useful evidence, not a substitute
for the coordinated root upgrade.

## 12.21.15 Failure handling and rollback

The candidate is rejected or returned to design review if:

- official release or wheel evidence changes materially;
- clean Linux or Darwin resolution, `pip check`, or provenance fails;
- any retained compiled extension cannot build, import, or execute;
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
