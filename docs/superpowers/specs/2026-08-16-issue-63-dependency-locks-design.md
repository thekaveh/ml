# 12.23 Issue 63 Dependency Locks and Immutable Build Inputs Design

## 12.23.1 Purpose

Issue #62 qualified one exact Python 3.11, Torch 2.11, PyG, torchao, and NNx runtime across
Darwin arm64, Linux x86_64, and native Linux arm64 Docker. The selected direct requirements are
exact where ABI compatibility matters, but most of the transitive notebook environment still
resolves from ranges or unpinned names. CI installs those source manifests directly, the bootstrap
stage upgrades pip without an exact artifact identity, and Docker and Codespaces consume mutable
base-image tags. A future resolver run or upstream tag move can therefore change an otherwise
unchanged commit.

Issue #63 closes that reproducibility gap without changing the qualified runtime matrix. It adds
reviewed, hash-checked lock artifacts for every supported installation surface, makes CI and the
four-stage installer consume those artifacts, pins bootstrap tools and container bases, and adds
fail-closed drift checks and controlled update procedures. It does not claim that one wheel set is
portable across all systems; compiled Torch and PyG artifacts remain explicit platform variants.

Primary behavior references are pip's
[hash-checking mode](https://pip.pypa.io/en/stable/topics/secure-installs/), uv's
[requirements locking](https://docs.astral.sh/uv/pip/compile/) and
[platform-specific resolution](https://docs.astral.sh/uv/concepts/resolution/), uv's
[explicit package-index binding](https://docs.astral.sh/uv/concepts/indexes/), and Docker's
[digest-pinning guidance](https://docs.docker.com/build/building/best-practices/#pin-base-image-versions).

## 12.23.2 Selected strategy

The repository retains pip-compatible requirement files and the Issue #62 four-stage installer.
Pinned `uv==0.11.19` is the lock compiler, not the runtime package manager. A repository-owned
compiler front end translates the existing human-authored manifests plus an explicit package
source policy into deterministic, pip-consumable lock files with SHA-256 hashes.

This strategy is selected over two alternatives:

1. **A project-wide `uv.lock` migration** has strong universal-lock semantics and first-class
   index binding, but would replace the existing pip/Make/Docker/Codespaces contract, introduce a
   project model into a notebook repository that is deliberately not a package, and expand this
   issue beyond reproducibility hardening.
2. **Exact constraints without hashes** would stabilize versions but would not bind downloaded
   artifacts, bootstrap tools, or compiled wheel provenance, so it does not meet the acceptance
   criteria.

The selected pip-compatible family is the smallest design that preserves established consumers,
uses the docs lock pattern already present in the repository, and supports different compiled
artifacts for the three qualified hosts.

## 12.23.3 Authoritative inputs and generated outputs

The human-authored input inventory is exact:

- `requirements.txt` — notebook, test, lint, documentation compatibility, and NNx direct inputs;
- `bootstrap-requirements.txt` — exact `pip`, `setuptools`, `wheel`, and `packaging` bootstrap
  inputs;
- `compiler-requirements.txt` — exact `uv==0.11.19` regeneration tooling;
- `nlp-model-requirements.txt` — the official
  `https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl`
  artifact with the upstream-published SHA-256
  `1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85`;
- `torch-core-requirements.txt` — the exact Torch trio;
- `torch-ecosystem-requirements.txt` — exact Lightning, TorchMetrics, and torchao inputs;
- `torch-requirements.txt` — the selected PyG runtime direct inputs; its current `--find-links`
  directive moves to `requirements/lock-policy.toml` so source authority is not duplicated;
- `torch-audit-requirements.txt` and `pyg-extension-audit-requirements.txt` — the two
  human-authored advisory supplements;
- `vulnerability-audit-requirements.txt` — the direct audit tool;
- `atlas-contract-requirements.txt` — focused Atlas contract tooling, including the exact uv
  executable consumed by the standalone Atlas contract workflow; and
- `docs-requirements.in` — documentation direct inputs.

`requirements/lock-policy.toml` is the second human-authored input surface. It contains only the
lock schema version, the path and expected shape of the compiler input (not a compiler version),
the Python `3.11.0` support floor,
three supported platform keys and target triples, explicit package-to-index bindings, expected
source manifests, and expected output inventory. It contains no dependency version that already belongs to a source
manifest. The offline verifier rejects either a duplicated dependency authority or an unrepresented
manifest input. `requirements/image-lock.json` is a separate human-reviewed image ledger containing
the selected tag, multi-platform index digest, and qualified amd64/arm64 child manifests for each
container base; dependency versions do not appear in it.

Generated lock artifacts live under `requirements/locks/` except for the existing
`docs-requirements.txt`, whose established path remains the documentation lock:

```text
requirements/locks/
  bootstrap.txt
  compiler.txt
  audit.txt
  atlas-contract.txt
  darwin-arm64/
    core.txt
    runtime.txt
    root.txt
  linux-x86_64/
    core.txt
    runtime.txt
    root.txt
  linux-aarch64/
    core.txt
    runtime.txt
    root.txt
docs-requirements.txt
```

`compiler.txt` is the complete hashed wheel closure for `uv==0.11.19` and is consumed only by the
networked lock-regeneration job. It is not installed into notebook runtime environments. Each
platform directory is an ordered, cumulative closure:

- `core.txt` compiles the Torch trio and its transitive dependencies;
- `runtime.txt` compiles core plus the ecosystem and PyG runtime;
- `root.txt` compiles core plus runtime plus `requirements.txt`, `docs-requirements.in`, and
  `nlp-model-requirements.txt`, so the runtime, test, documentation, and spaCy model package closure
  is installed once and cannot be changed by a later solve.

Cumulative files deliberately repeat earlier exact requirements. Docs compilation consumes the
bootstrap projection so shared `packaging` cannot diverge. Runtime compilation consumes an
exact-version projection of its freshly generated core lock; root consumes projections of its
fresh runtime, bootstrap, and docs locks. The docs lock remains a smaller standalone surface for
docs-only jobs. Every package shared by `root.txt` and
`docs-requirements.txt` must have the same normalized version; each platform root's approved hashes
must equal that platform's projection of the universal docs hashes. Root compilation consumes the
compiler-generated, exact-version-only constraint projection of the validated docs lock (not the
hashed lock file as a pip constraint) after it is generated. The projection is temporary, carries
no independent versions, and is deleted after the transaction. The ordered
installer verifies
the first download of each artifact in the stage that introduces it; later stages see the already
installed, already-hash-checked requirement at the same exact version. The lock verifier requires
the normalized package/version set of `core` to be a subset of `runtime`, and `runtime` to be a
subset of `root`, while rejecting conflicting duplicates.

Every generated file carries a normalized header with the exact compiler version parsed from
`compiler-requirements.txt`, Python resolution
floor, platform key, source-manifest SHA-256 map, package-source-policy SHA-256, and generation
command identity. Absolute paths, cache paths, credentials, timestamps, and hostnames are forbidden
so clean regeneration is byte-stable.

The policy also fixes the resolver's package-upload horizon at
`2026-08-17T02:21:18Z`, the timestamp of the originally reviewed generated-lock commit. Every uv
compile receives that value through exactly one `--exclude-newer` argument. This cutoff is an
authoritative policy input, not a generated-header timestamp: the header binds it through the
policy SHA-256. A package uploaded later cannot silently make `lock-check` stale; advancing the
cutoff is a deliberate policy and generated-lock change that receives the same review and clean
installation controls as any dependency update.

Generation order is fixed: bootstrap, compiler, docs, audit, Atlas, then core/runtime/root for each
platform in policy order. A downstream compile reads only already-generated temporary outputs from
that same transaction; no step reads a half-updated committed lock. All outputs validate before the
commit phase. Per-file replacements use `os.replace`; a handled replacement failure restores
backups before reporting failure, and validation failures never touch committed outputs.

## 12.23.4 Explicit package-source policy

Index order is not a security or reproducibility policy. The lock compiler uses an exact mapping:

- on Linux x86_64 and Linux aarch64, `torch`, `torchvision`, and `torchaudio` resolve only from
  `https://download.pytorch.org/whl/cpu`;
- on Darwin arm64, those three packages and torchao resolve only from PyPI;
- `pyg-lib`, `torch-scatter`, and `torch-sparse` resolve only from the flat
  `https://data.pyg.org/whl/torch-2.11.0+cpu.html` source on all three targets;
- `en-core-web-sm` resolves only from the exact official URL and hash in
  `nlp-model-requirements.txt`;
- Linux torchao resolves from the exact Torch CPU project page selected by uv's CPU backend; its
  installed wheel metadata is exactly `0.18.0+cpu`, while its public `0.18.0` identity remains the
  manifest and advisory projection;
- all remaining runtime packages resolve only from PyPI; and
- no extra index may become a fallback source for an unbound package.

The compiler must express those bindings through uv's explicit named-index/flat-index semantics,
not through `--extra-index-url`, `--index-strategy unsafe-best-match`, or candidate precedence.
Pip requirements cannot preserve uv's per-package index binding, so installation does not pretend
otherwise. The ordered installer exposes only the sources needed by each stage: Linux core sees
only the Torch CPU index; Darwin core sees only PyPI; runtime and root see PyPI plus the exact PyG
flat page and, on Linux only, the exact Torch CPU `torchao` project page. They rely on the
already-installed core closure. The committed SHA-256 set is the final
artifact authority. A same-version artifact from any reachable source is rejected unless its bytes
match an approved hash. If the ordered core stage did not install Linux's `+cpu` artifacts, later
stages cannot resolve or substitute them.

The offline verifier rejects an unapproved global source option, unbound compiled package, VCS
requirement, editable, local path, or unnamed source distribution. Direct URLs are forbidden except
for the one exact official `en_core_web_sm==3.8.0` wheel declared by
`nlp-model-requirements.txt`. The one named python-louvain sdist exception is governed by Section
12.23.5.

The selected local-version contracts remain those from Issue #62: Linux PyG wheels require
`pt211cpu`; Darwin accepts `pt211` or the already-qualified absent-local-tag case only with the
existing WHEEL/RECORD and runtime-canary evidence. The lock does not broaden that verifier.

## 12.23.5 Hash and binary policy

All generated lock requirements are exact and carry one or more SHA-256 hashes. Runtime installs
pass `--require-hashes` explicitly even though pip also enables hash checking when it sees a hash.
This makes omission mutations visible at the command boundary.

The generated family is wheel-only on supported targets except for one named existing dependency:
`python-louvain==0.16` publishes no wheel. The compiler uses `--only-binary=:all:` together with
`--no-binary=python-louvain`; its exact sdist SHA-256 is locked. The root install uses
`--no-build-isolation`, so that sdist builds only with the already hash-installed bootstrap
setuptools and wheel rather than resolving a hidden build environment. This locks the build inputs;
it does not claim that independently built wheel bytes are identical across platforms.

This intentionally strengthens Issue #62's package-list binary boundary for every other package
after clean qualification proves that the rest of the environment has wheels on every supported
target. It does not weaken or remove the existing compiled-PyG and NNx provenance checks. A second
source-built package, a changed python-louvain backend, or an isolated build requires a design
amendment naming its build constraints and qualification. The compiler and offline verifier reject
every other sdist selection.

pip's hash mode is all-or-nothing: every transitive requirement is pinned and hashed. An unhashed
line, range, bare name, missing transitive dependency, non-SHA-256 digest, or unexpected option
fails before installation.

## 12.23.6 Bootstrap contract

`bootstrap-requirements.txt` is the machine-readable version authority. Stage 0 stops using
`--upgrade pip`; it installs the generated `bootstrap.txt` with
`--require-hashes`, containing exact wheel hashes for:

- `pip==26.2.1` (the fixed release selected after `PYSEC-2026-3721` made the prior
  `26.1.2` bootstrap pin unacceptable);
- `setuptools==81.0.0`; and
- `wheel==0.47.0`; and
- `packaging==26.2`, required by wheel 0.47.0.

The Python 3.11 environment's bundled pip is trusted only far enough to validate and install those
four locally declared hashes. Bootstrap installation is binary-only and may not install uv or any
package outside this exact packaging-tool closure. Because `packaging` is also used by later
closures, the lock verifier requires the same version and platform-projected hashes in bootstrap,
docs, and root. Every repository pip consumer, including docs, Pages, audit, Atlas, Docker,
Codespaces, and each CI job, runs the same `make install-bootstrap` seam before installing its
dedicated hashed lock. The lock compiler then installs uv from `compiler.txt`. Separately, the
focused Atlas contract environment installs the same exact uv version from `atlas-contract.txt`
after bootstrap because `infra/start.sh` consumes it; that consumer role is not part of the notebook
bootstrap stage.

The bootstrap versions are repository policy, not "latest" aliases. Updating any of them requires
the same lock-refresh review, clean installs, and rollback path as a runtime dependency change.

`scripts.install_locked_requirements` owns the common subprocess boundary for non-runtime locks.
Its closed role map is `bootstrap`, `compiler`, `docs`, `audit`, and `atlas-contract`; each role maps
to one repository-relative lock and exact binary/source arguments. Make exposes only
`install-bootstrap`, `install-compiler-lock`, `install-docs-lock`, `install-audit-lock`, and
`install-atlas-contract-lock`. The helper and four-stage installer share the same sanitized pip
environment and stable redacted failure protocol, so a dedicated consumer cannot inherit an ambient
index, constraint, requirement, or config any more than a runtime stage can.

## 12.23.7 Four-stage installation contract

`scripts.install_torch_stack` keeps exactly four ordered stages and selects one supported platform
key from `platform.system()` and `platform.machine()`:

0. install `requirements/locks/bootstrap.txt` with `--only-binary=:all:` and `--require-hashes`;
1. install `<platform>/core.txt` with `--only-binary=:all:`, `--require-hashes`, and the qualified
   Torch-only `--index-url` on Linux or PyPI on Darwin;
2. install `<platform>/runtime.txt` with `--only-binary=:all:`, `--require-hashes`, PyPI as the
   index, the exact PyG page as `--find-links`, and on Linux the exact Torch CPU `torchao` project
   page as a second flat source; and
3. install `<platform>/root.txt` with `--only-binary=:all:`,
   `--no-binary=python-louvain`, `--no-build-isolation`, `--require-hashes`, PyPI as the index, and
   the exact PyG page plus, on Linux, the exact Torch CPU `torchao` project page as `--find-links`.

The stage-specific lock, binary, build-isolation, and source options are exact arguments, not
ambient pip configuration; `PIP_INDEX_URL`, `PIP_EXTRA_INDEX_URL`, `PIP_FIND_LINKS`,
`PIP_CONSTRAINT`, `PIP_REQUIREMENT`, and pip config files are neutralized or rejected so a caller
cannot redirect the reviewed sources.

The selected platform keys and compiler targets are exactly `darwin-arm64` /
`aarch64-apple-darwin`, `linux-x86_64` / `x86_64-manylinux_2_28`, and `linux-aarch64` /
`aarch64-manylinux_2_28`. Resolution retains Issue #62's support floor with
`--python-version 3.11.0`; the supported consumer range is `>=3.11.0,<3.12`. Darwin compilation fixes
`MACOSX_DEPLOYMENT_TARGET=13.0`. Acceptance records exact CPython 3.11.15 for Darwin and GitHub
Actions and exact CPython 3.11.10 for the pinned Jupyter image. Unsupported systems, interpreter
versions, or deployment baselines fail before running a command. Every stage stops on the first
nonzero result and reports only the stable stage name. No consumer may run an additional package
install after the final verifier boundary.

Docker, Codespaces, CI runtime jobs, and local setup continue to call
`make install-torch-stack`; none reimplements lock selection. Documentation, audit, and focused
Atlas jobs run `make install-bootstrap` and their exact dedicated Make install target because they
do not need the runtime stack. The official
`en_core_web_sm==3.8.0` package is already installed and hash-checked by `root.txt`;
`make nlp-assets` therefore removes `spacy download` and downloads only NLTK VADER data. Issue #64
owns the VADER payload's internal content-integrity policy. It is the sole post-lock data mutation,
runs before final `pip check` and runtime verification, and is named as a data-only exception in
workflow and Docker contract tests.

## 12.23.8 Lock compiler and offline verifier

`scripts/lock_dependencies.py` provides three explicit modes and runs the compiler through
`python -m uv` after bootstrap, installation from `compiler.txt`, and verification that the
installed uv distribution is RECORD-owned and equals the sole version authority in
`compiler-requirements.txt`:

- `--write` runs exact `uv==0.11.19` compile commands, normalizes headers, validates the complete
  temporary family, applies the policy's exact `--exclude-newer` cutoff, and transactionally
  replaces only the expected lock inventory; and
- `--check` copies each committed output to a temporary directory, recompiles without `--upgrade`
  and with the same fixed upload cutoff, normalizes the header, and byte-compares every expected
  output without modifying the checkout; and
- `--update-compiler` is the only mode that permits the compiler manifest and installed old
  compiler to differ, and performs the two-pass transition below.

Normal `--write` and `--check` require exact equality among the compiler manifest, committed
compiler-lock direct pin, installed distribution, and every generated header; the policy file stores
none of those versions. A compiler-version update is an explicit state machine:

1. `--update-compiler` verifies the running old uv against the unmodified committed
   `compiler.txt` and old generated headers, while separately parsing exactly one new uv pin from
   the changed `compiler-requirements.txt`;
2. the old compiler generates only a candidate compiler lock in a temporary directory;
3. a new disposable environment bootstraps, hash-installs that candidate, and verifies the new
   RECORD-owned uv identity equals the changed manifest;
4. the new compiler regenerates the candidate compiler lock and the entire remaining family from
   source inputs, with all headers naming only the new identity; and
5. the first- and second-pass candidate compiler locks must be byte-identical, the exact uv pin in
   `atlas-contract-requirements.txt` must equal the new compiler manifest, and the complete final
   verifier must pass before the transactional commit phase.

Any old/new mismatch outside these named intermediate states, or a compiler input change under
`--write`/`--check`, fails closed. This eliminates both self-update deadlock and a second compiler
version authority.

Networked regeneration is never part of `make verify`. `scripts/verify_dependency_locks.py` is the
offline, fail-closed repository gate. It validates:

- the exact source and output inventories;
- schema/header/compiler/platform/source-policy identities;
- source-manifest hashes against live bytes;
- exact pins, SHA-256 hashes, allowed global options, and absence of unsafe requirement forms;
- platform-specific compiled wheel versions and local tags;
- cumulative core/runtime/root package-set relationships;
- direct-input presence in the appropriate closure;
- bootstrap, docs, audit, and Atlas lock direct-input projections; and
- that installer, Make, CI, Docker, Codespaces, and docs consumers reference only the expected
  locked paths.

The verifier reports stable categories and repository-relative paths. It never performs network
access, resolves a registry, or writes a file. It proves internal coherence: hashes are
present, use SHA-256 syntax, and agree wherever the same artifact identity is projected across
committed files, but only a hash-required download binds those values to artifact bytes. The
offline gate does not claim that an arbitrary coordinated lock and source-policy edit reflects
current registry bytes. Offline tests cover missing/malformed hashes and cross-file mismatches;
valid-but-wrong coordinated hash mutations must survive the offline parser and then fail the
network regeneration or clean hash-required install control. Other tests mutate source bytes,
headers, pins, package sets, indexes, local tags, stage paths, and consumer commands and require the
appropriate failure category.

`make lock-check` runs trusted network resolution and byte-regeneration for pull requests that
change an authoritative dependency input, lock, compiler/verifier/installer consumer, or for the
scheduled dependency audit; clean hash-required installs prove the resolved artifacts.
`make image-lock-check` resolves each configured tag from its registry and requires the live index
and qualified child manifests to equal `requirements/image-lock.json`. These networked checks kill
coordinated false-ledger edits that an offline parser cannot detect. `make verify-dependency-locks`
runs the offline verifier and is included in `make verify`.

## 12.23.9 Documentation, audit, and Atlas tool locks

`docs-requirements.txt` remains the universal, hash-checked output of
`docs-requirements.in`. Every docs and Pages install runs `make install-bootstrap` followed by
`make install-docs-lock`; the shared helper supplies explicit `--require-hashes`. Root lock
generation constrains this closure exactly as described in Section 12.23.3, preventing a second
solve from changing an installed docs package. Runtime jobs already receive the docs closure from
their platform `root.txt` and must not invoke `install-docs-lock` afterward.

`requirements/locks/audit.txt` is the complete hashed closure of
`vulnerability-audit-requirements.txt`; `requirements/locks/atlas-contract.txt` is the complete
hashed closure of `atlas-contract-requirements.txt`, including `uv==0.11.19`, `pytest==9.0.3`, and
`pyyaml==6.0.3`, plus exact `nltk==3.10.3` required by the VADER runtime-probe tests. The
corresponding CI jobs install bootstrap followed only by those locks. Advisory input surfaces
remain the human-authored manifests established by Issues #59–#62; the audit policy does not
mistake generated lock duplication for a new logical surface.

Advisory execution, however, never gives those source manifests back to pip. The advisory tool
parses validated locks into temporary exact `name==public-version` projections and invokes
`pip-audit --requirement <projection> --no-deps --disable-pip --strict --format=json`, so no
dependency is resolved or installed during the audit. Linux local versions retain their original
lock provenance while using the PEP 440 public version as the vulnerability identity. The exact
non-PyPI allowlist is derived from validated source policy: the direct-URL `en-core-web-sm` wheel
and `pyg-lib`. Those identities do not enter the PyPI-service projection; instead the observation
records canonical name, locked version, source, artifact hash, `audited=false`, and
`reason=non-pypi`. Any other unsupported package, changed allowlist member, or missing explicit
record is an error. `torch-scatter` and `torch-sparse` remain auditable through their exact public
versions and may not be hidden by that exception.

The four logical surfaces are exact lock unions:

- combined-runtime = bootstrap + each supported platform root + audit-tool lock;
- torch = bootstrap + each supported platform core/runtime, with the two human-authored Torch/PyG
  audit supplements required to equal the corresponding lock projection;
- documentation = bootstrap + docs lock; and
- atlas-contract = bootstrap + Atlas-contract lock.

Shared packages must resolve to one public version across a logical surface; platform-local artifact
tags and source lock paths remain evidence metadata. The standalone bootstrap+audit-tool environment
must be a subset of combined-runtime. The accepted-advisory policy is reconciled against these four
unions, so the selected setuptools advisory must appear on every affected bootstrap-bearing surface
unless a newly locked version removes it. Tests delete or change a bootstrap finding, substitute a
source manifest for a projection, re-resolve a range, omit a direct-URL/PyG identity, and change a
surface union; each mutation must fail before policy acceptance.

## 12.23.10 CI contract

Every Python dependency-manifest install in CI consumes a reviewed lock. Runtime jobs call the
canonical installer; docs, Pages, audit, and Atlas jobs first run the common bootstrap target and
then the corresponding dedicated locked-install target. Pip cache keys include the selected lock
files rather than only source manifests. `make nlp-assets` performs only the NLTK VADER data
download, the explicit Issue #64 exception described in Section 12.23.7, ahead of final
package/provenance checks. No CI step may install a Python package after those checks.

The dependency-audit job always runs the offline verifier, common bootstrap target, committed audit
lock, and advisory reconciliation. A fail-closed path classifier enables `make lock-check` only for
scheduled/workflow-dispatch runs or changes to dependency inputs, outputs, policy, compiler,
installer, or their consumers; it enables `make image-lock-check` for schedules/dispatches or
changes to the image ledger, Dockerfile, devcontainer, or image verifier. Missing base SHA,
ambiguous event, or classifier error selects the networked checks rather than skipping them. Thus a
normal code-only PR avoids unrelated registry resolution, while a relevant diff cannot bypass
compiler drift, non-byte-stable regeneration, or a moved image tag. Normal repository jobs remain
independent of networked resolution and install the committed locks exactly.

When `lock-check` is selected, the job installs `compiler.txt` through
`make install-compiler-lock` before invoking the compiler; otherwise uv is not installed. The image
check uses the Docker Buildx CLI already required by Docker qualification and never introduces a
Python package.

Workflow contract tests reject direct source-manifest installs, un-hashed installs, late package
mutations, wrong platform selection, omitted or fail-open lock/image selectors and checks, cache keys
that exclude the consumed lock,
and masking through conditions, `continue-on-error`, shell pipelines, or warning configuration.
The required job names and GitFlow activation conditions remain unchanged.

## 12.23.11 Docker and devcontainer image identities

The root Dockerfile pins the qualified Jupyter base as a tag plus the reviewed multi-platform digest:

`quay.io/jupyter/datascience-notebook:python-3.11@sha256:14379f24c840f27375e3a4b29a9fa55e449633ac85c9cf3806ca62d11e5603ec`.

The devcontainer pins:

`mcr.microsoft.com/devcontainers/python:3.11-bookworm@sha256:8e95c16fbc98a4a6a8f11f5b5bd152d0ffcd4fd0f4b31bd03e95965c777d2577`.

These are manifest-list/index digests, preserving native Linux amd64 and arm64 selection.
`requirements/image-lock.json` records each exact source reference plus the current qualified amd64
and arm64 child-manifest digests. Source references use the multi-platform digest so Docker Desktop
arm64 and GitHub-hosted amd64 resolve from one reviewed identity.

The Docker consumer creates `/home/jovyan/.venvs/ml-eng-lab` with the base image's qualified Python
and places it first on `PATH` before invoking the canonical installer. It also sets
`CONDA_AUTO_ACTIVATE_BASE=false`, preventing the Jupyter startup hook from replacing that venv with
the base conda environment when the built image launches. Together these controls prevent unrelated
packages preinstalled in the Jupyter base environment from contaminating the exact root lock or
making build-time and runtime verification select different environments; the locked root supplies
the Notebook/JupyterLab runtime inside that environment.

The offline verifier parses image references structurally, requires the exact registry, repository,
tag, and index digest recorded by the ledger, and reconciles source with that committed ledger. It
rejects a tag-only reference, digest-only reference without the human-readable tag, wrong
algorithm/length, platform-child digest substituted for the index digest, variable interpolation,
or an unledgered image. This is an internal-consistency proof; only the networked
`make image-lock-check` establishes that the registry's current tag/index/children equal the ledger.

## 12.23.12 Controlled update procedure

Dependency and image updates are deliberate pull requests:

1. change the smallest human-authored source manifest or approved image tag;
2. run the pinned lock compiler with `--write` in a clean Python 3.11 environment;
3. inspect direct and transitive diffs, source-index changes, wheel tags, hashes, and advisories;
4. resolve image tags with `docker buildx imagetools inspect`, record the current index digest and
   supported child manifests in `requirements/image-lock.json`, and update tag plus digest together;
5. run the offline verifier, networked byte-regeneration check, and networked image-ledger check;
6. perform clean installs for Darwin arm64, Linux x86_64 CI, and native Linux arm64 Docker;
7. run pip check, the Torch/NNx verifiers, audit reconciliation, full tests, documentation gates,
   devcontainer validation, Docker qualification, and the required notebook tiers; and
8. merge only when source inputs, generated outputs, docs, and evidence are one coherent commit
   range.

`--upgrade` is allowed only in the explicit refresh command. Routine `--check` compilation retains
the committed versions. A changed source manifest with unchanged locks and a changed lock with
unchanged source metadata both fail.

## 12.23.13 Qualification

Acceptance freezes one final feature SHA and uses fresh, disposable environments. Required evidence
is:

- byte-stable networked lock regeneration, networked image-ledger validation, and offline
  internal-consistency verification;
- clean locked installs and `pip check` on exact CPython 3.11.15 Darwin arm64 and Linux x86_64;
- a native, no-emulation Linux arm64 Docker build from the pinned base digest using its exact
  CPython 3.11.10 interpreter;
- exact Torch, PyG, NNx, warning-debt, graph-backend, and QAT verifier results preserved from the
  Issue #62 contract;
- complete repository tests, Ruff, repository verification, docs/site/wiki parity, workflow YAML,
  and devcontainer JSON/semantic validation;
- dependency advisory reconciliation from the locked environment;
- Tier A 18/18, Tier B 6/6, and Tier C 4/4 notebook output contracts at the frozen SHA; and
- feature, release, Pages/wiki, and content-neutral main-to-develop synchronization evidence under
  the existing dual-SHA GitFlow contract.

No Atlas service, JupyterHub service, Docker Compose stack, Ollama, ComfyUI, or persistent container
is started. Atlas runtime ownership remains Issue #65. NLP asset content hashes remain Issue #64;
this issue locks the Python packages and setup commands but does not silently absorb that asset
integrity scope.

## 12.23.14 Documentation and release surfaces

README setup instructions, CONTRIBUTING clean-install commands, dependency contracts, architecture,
CHANGELOG Unreleased notes, Make help, Docker/Codespaces comments, and the three generated
documentation surfaces must describe the same lock inventory, supported platform mapping,
regeneration command, digest policy, and limitations. Historical release text remains immutable.

The documentation must say "reproducible for the qualified platform lock" rather than claiming a
single cross-platform binary environment. The PyG flat-index split, external NLP asset boundary,
Atlas ownership boundary, and deliberate update process remain visible.

## 12.23.15 Rollback

Rollback is one coherent revert of source-manifest changes, generated lock artifacts, bootstrap
pins, installer/consumer wiring, image tag-plus-digest pins, verifier rules, and documentation.
Partial rollback is forbidden: restoring a source file without its locks, a mutable image tag without
its digest, or direct CI installs without removing the corresponding verification contract leaves an
internally false state.

After rollback, rerun the previous commit's exact lock verifier, clean installs, Docker build,
tests, and docs gates. Never regenerate old locks with a newer compiler and call that a rollback.

## 12.23.16 Completion boundary

Issue #63 is complete only after the implementation and qualification evidence are independently
reviewed, merged through feature to `develop` and `develop` to `main`, live Pages/wiki are verified,
`main` is synchronized back to `develop` when ancestry requires it, Issue #63 is marked Done and
closed, Issue #53 remains open, and all issue-owned branches, refs, worktrees, environments, images,
containers, and temporary evidence are removed. Any tracked correction after the frozen SHA
invalidates the affected qualification and restarts it.
