# 12.26 Issue 64 NLP Asset Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrity-lock the spaCy and NLTK assets consumed by every supported ml-eng-lab setup path, eliminate runtime download fallbacks, and publish a qualified Atlas-compatible release.

**Architecture:** Keep spaCy as the existing hash-required package from Issue #63. Add one strict TOML identity for the official VADER ZIP and one standard-library installer/verifier, route every parent consumer through it, and project the exact same manifest/script/model requirement into the pinned Atlas JupyterHub build. Land the Atlas projection as a main-based hotfix before pinning it in ml-eng-lab.

**Tech Stack:** Python 3.11, `tomllib`, `urllib.request`, `hashlib`, `zipfile`, NLTK, pytest, Make, Docker, Git submodules, GitHub Actions, MkDocs, GitHub wiki.

**Spec:** `docs/superpowers/specs/2026-08-17-issue-64-nlp-asset-integrity-design.md`

## 12.26.1 Global Constraints

- The authoritative VADER URL is `https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/sentiment/vader_lexicon.zip`.
- The authoritative VADER size is `90486`; the SHA-256 is `8adba4294eef3964d820bf655e37e61bdc3a341994356af59b74fb3b4a36ce5c`.
- The authoritative NLTK resource is `sentiment/vader_lexicon.zip`; its required member is `vader_lexicon/vader_lexicon.txt`; its license is `MIT License`.
- The spaCy model remains `en-core-web-sm==3.8.0` from the exact URL in `nlp-model-requirements.txt`, SHA-256 `1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85`.
- Do not vendor either third-party asset. Do not restore `nltk.download`, `nltk.downloader`, or `spacy download` to a current setup or active notebook.
- The installer must use only Python's standard library plus the already-installed NLTK runtime for resource lookup and the sentiment smoke test.
- Verification is offline. Valid existing data must short-circuit without opening the network. Missing/corrupt data must fail closed.
- A corrupt existing target is never overwritten. A missing target is installed with same-filesystem atomic no-clobber semantics.
- Diagnostics are stable categories and never reveal URLs, response bodies, tokens, local paths, or exception representations.
- Supported parent platforms remain Darwin arm64, Linux x86_64, and Linux arm64 on Python `>=3.11.0,<3.12`.
- Do not start Atlas, Docker Compose, Ollama, ComfyUI, or unrelated services. Docker builds and disposable `docker run --rm` probes are allowed.
- Preserve all Issue #63 lock versions, resolver cutoff, hashes, source bindings, advisory policy, and image digests.
- Keep Issues #53, #65, and #66 open and substantively unchanged.

---

### 12.26.1.1 Task 1: Add the authoritative VADER manifest and fail-closed installer

**Files:**
- Create: `requirements/nlp-assets.toml`
- Create: `scripts/nlp_assets.py`
- Create: `tests/test_nlp_assets.py`

**Interfaces:**
- Produces: `VaderAsset`, `NLPAssetError`, `load_manifest`, `default_nltk_data_dir`, `verify_vader`, `install_vader`, and `main` exactly as §12.25.4 specifies.
- Consumes: Python 3.11 standard library and installed NLTK; no repository lock compiler or downloader internals.

- [ ] **Step 1: Write exact parser and identity RED tests**

Create the canonical fixture and assertions:

```python
EXPECTED = nlp_assets.VaderAsset(
    url="https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/sentiment/vader_lexicon.zip",
    sha256="8adba4294eef3964d820bf655e37e61bdc3a341994356af59b74fb3b4a36ce5c",
    size=90486,
    resource=PurePosixPath("sentiment/vader_lexicon.zip"),
    member=PurePosixPath("vader_lexicon/vader_lexicon.txt"),
    license="MIT License",
)

def test_real_manifest_is_the_exact_reviewed_identity() -> None:
    assert nlp_assets.load_manifest(REPO / "requirements/nlp-assets.toml") == EXPECTED
```

Parameterize missing/extra/duplicate keys, bool/zero/negative size, uppercase/short/non-hex digest, non-HTTPS URL, wrong host/path, URL credentials/query/fragment/port, absolute/traversing/backslash resource/member, and wrong license. Each case must assert `NLPAssetError.category == "manifest"` and that `str(error) == "manifest"` contains none of the mutated input.

- [ ] **Step 2: Run the parser RED gate**

Run:

```bash
PYTEST_ADDOPTS='-p no:cacheprovider -W error' python -m pytest \
  tests/test_nlp_assets.py -k 'manifest or identity' -q
```

Expected: collection fails because `scripts.nlp_assets` does not exist.

- [ ] **Step 3: Add the exact manifest and parser**

Write `requirements/nlp-assets.toml` exactly as §12.25.3. In `scripts/nlp_assets.py`, define:

```python
@dataclass(frozen=True, slots=True)
class VaderAsset:
    url: str
    sha256: str
    size: int
    resource: PurePosixPath
    member: PurePosixPath
    license: str

class NLPAssetError(RuntimeError):
    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)
```

Use `tomllib.loads(path.read_text(encoding="utf-8"))`, compare the top-level and nested key sets exactly, validate types before values, parse the URL with `urllib.parse.urlsplit`, and build `PurePosixPath` only after rejecting unsafe strings. Catch `OSError`, `UnicodeError`, and `tomllib.TOMLDecodeError` and raise `NLPAssetError("manifest") from None`.

- [ ] **Step 4: Write verification and install RED tests**

Create a deterministic in-memory ZIP fixture with member `vader_lexicon/vader_lexicon.txt`, then derive a fixture-specific asset identity. Test:

```python
def test_valid_existing_asset_never_opens_network(tmp_path, fixture_asset, fixture_zip):
    target = _write_target(tmp_path, fixture_asset, fixture_zip)
    def forbidden(_: str):
        raise AssertionError("network opened")
    assert install_vader(fixture_asset, tmp_path, offline=False, opener=forbidden) == target

def test_missing_offline_fails(tmp_path, fixture_asset):
    with pytest.raises(NLPAssetError) as caught:
        install_vader(fixture_asset, tmp_path, offline=True, opener=lambda _: None)
    assert caught.value.category == "missing"
```

Add explicit cases for corrupt existing bytes, target directory, target symlink, symlinked `sentiment` child, truncated and oversized streams, opener exception, wrong digest, malformed ZIP, missing/wrong/directory member, smoke failure, hard-link race winner valid/invalid, and temporary-file cleanup. Assert a corrupt target's bytes and inode never change.

- [ ] **Step 5: Implement verification, bounded streaming, and atomic no-clobber install**

Use these focused helpers:

```python
def _digest(path: Path) -> tuple[int, str]: ...
def _validate_archive(path: Path, asset: VaderAsset) -> None: ...
def _smoke(data_dir: Path) -> None: ...
def _safe_target(data_dir: Path, resource: PurePosixPath) -> Path: ...
```

`_validate_archive` compares exact size/hash, opens `zipfile.ZipFile`, requires
the complete non-directory member set to equal `{asset.member}` (directory
entries are allowed), and reads that member once to make CRC validation execute.
`_smoke` temporarily prepends only the resolved data directory to
`nltk.data.path`, constructs `SentimentIntensityAnalyzer`, requires
`compound("good") > 0` and `compound("bad") < 0`, and restores the original
path in `finally`.

For online install: create the resource directory with mode `0o755`; reject a symlink at either repository-created child; stream at most `asset.size + 1` bytes into `tempfile.NamedTemporaryFile(dir=target.parent, delete=False)`; flush and `os.fsync`; close; verify; call `os.link(temp_path, target)` to atomically create without overwrite; if `FileExistsError`, verify the race winner; finally unlink the temporary name. Translate failures to the exact categories in §12.25.4 with `from None`.

- [ ] **Step 6: Implement and test the CLI**

Use `argparse` subcommands `install` and `verify`, default the manifest from the repository root, obtain the default target from `nltk.downloader.Downloader().default_download_dir()`, and print only:

```text
nlp asset install passed
nlp asset verification passed
nlp asset operation failed: <category>
```

CLI tests call `main([...])` directly and a subprocess. Mutate a manifest URL to include a sentinel secret and ensure the sentinel, URL, and temp path are absent from both streams.

- [ ] **Step 7: Run GREEN and mutations**

Run the full new file under warnings-as-errors. In temporary source copies, delete or neutralize the hash, size, ZIP/member, offline, no-network-short-circuit, symlink, cleanup, `os.link`, race-winner verification, and smoke guards. Before each mutated run assert source bytes changed; every mutation must fail at least one named test.

- [ ] **Step 8: Commit Task 1**

```bash
git add requirements/nlp-assets.toml scripts/nlp_assets.py tests/test_nlp_assets.py
git commit -m "build: integrity-lock the VADER lexicon"
```

---

### 12.26.1.2 Task 2: Route parent consumers, notebook code, and Atlas evidence through the contract

**Files:**
- Modify: `Makefile`
- Modify: `Dockerfile`
- Modify: `.github/workflows/ci.yml`
- Modify: `scripts/verify_dependency_locks.py`
- Modify: `scripts/verify_repo.py`
- Modify: `scripts/atlas_runtime_probe.py`
- Modify: `notebooks/sentiment_classification-vader-mlp-pytorch/notebook.ipynb`
- Modify: `notebooks/sentiment_classification-vader-mlp-pytorch/docs/spec.yaml`
- Modify: `tests/test_makefile_contract.py`
- Modify: `tests/test_verify_dependency_locks.py`
- Modify: `tests/test_verify_repo.py`
- Modify: `tests/test_atlas_runtime_probe.py`
- Create: `tests/test_nlp_asset_consumers.py`

**Interfaces:**
- Consumes: `python -m scripts.nlp_assets install|verify` from Task 1.
- Produces: canonical Make targets, no-download active notebook behavior, repository drift finding `D11.nlp_asset_contract`, and hash-bearing Atlas runtime evidence.

- [ ] **Step 1: Write consumer-order RED tests**

Parse Make recipes and workflow YAML rather than substring-only source. Require:

```python
assert recipe("nlp-assets") == ("$(PYTHON) -m scripts.nlp_assets install",)
assert recipe("verify-nlp-assets") == ("$(PYTHON) -m scripts.nlp_assets verify",)
assert recipe("codespace-setup") == (
    "$(MAKE) nlp-assets",
    "$(MAKE) verify-nlp-assets",
    "$(PYTHON) -m pip check",
    "$(MAKE) verify-torch-stack",
    "$(MAKE) verify-nnx-install",
)
```

Require Docker and Tier A to install then verify assets before pip/stack verification and before workloads. Reject missing/duplicate/late targets, direct module invocations outside Make, shell masking, `continue-on-error`, custom shells, pipes, `nltk.download`, `nltk.downloader`, and `spacy download`. Preserve every existing Tier B/C condition and artifact oracle.

- [ ] **Step 2: Write notebook and repository-guard RED tests**

Parse active notebook code cells and require the VADER setup cell to contain `nltk.data.find("sentiment/vader_lexicon.zip")`, a `LookupError` message naming `make nlp-assets`, and no callable whose qualified name is `nltk.download`. Add mutations for an aliased import/call, attribute assignment, `getattr(nltk, "download")`, and shell/magic download commands.

Add `D11.nlp_asset_contract` to `scripts/verify_repo.py`. At this task boundary its clean control checks the real manifest, Make/Docker/CI/devcontainer/current notebook source, lock verifier, and runtime probe. Task 3 extends the same finding to the newly available Atlas projection and exact gitlink. Malformed or missing files return one stable D11 finding rather than raising.

- [ ] **Step 3: Implement parent consumer changes**

Replace the Make target and add `verify-nlp-assets` to `.PHONY` and help text. Insert verification immediately after every `make nlp-assets`. Do not add it to jobs that do not install the asset. Keep all later pip/stack/NNx checks unchanged.

Change the sentiment notebook only in the setup code cell:

```python
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError as exc:
    raise RuntimeError("VADER lexicon is missing; run `make nlp-assets` before this notebook") from exc
```

Preserve outputs, execution counts, cell IDs, cell metadata, notebook metadata, and every unrelated code/Markdown cell. Update the task spec's execution/dependency statement without changing task behavior.

- [ ] **Step 4: Strengthen the dependency-lock and repository verifiers**

`verify_dependency_locks.py` must load the asset manifest through `scripts.nlp_assets.load_manifest`, verify the spaCy source manifest still equals the Issue #63 URL/hash line, and reject any current consumer legacy downloader. `verify_repo.py` adds D11 to `check_all` and the exact `--check assets` selector; it reports only repository-relative paths and categories.

Mutation tests must alter real temporary repository inputs—not helper constants—and kill wrong manifest URL/hash/size/member, missing consumer verification, late mutation, restored notebook fallback, removed spaCy identity, and malformed workflow shapes.

- [ ] **Step 5: Strengthen Atlas runtime evidence**

Add safe fields to the VADER asset projection:

```json
{
  "asset": "vader_lexicon",
  "status": "ok",
  "expected_sha256": "8adba429...ce5c",
  "observed_sha256": "8adba429...ce5c",
  "expected_size": 90486,
  "observed_size": 90486,
  "member": "vader_lexicon/vader_lexicon.txt",
  "import": {"module": "nltk.sentiment.vader", "status": "ok"}
}
```

Resolve only ZIP resources backed by real files under an existing `nltk.data.path` entry. Reject symlinks, directories, path escape, wrong size/hash/member, malformed ZIP, duplicate matches with different identities, and smoke failure. spaCy evidence requires version exactly `3.8.0`, import success, and `en_core_web_sm.load()` success; sanitize all failures.

- [ ] **Step 6: Run focused GREEN and mutation gates**

```bash
PYTEST_ADDOPTS='-p no:cacheprovider -W error' python -m pytest \
  tests/test_makefile_contract.py \
  tests/test_verify_dependency_locks.py \
  tests/test_verify_repo.py \
  tests/test_atlas_runtime_probe.py \
  tests/test_nlp_asset_consumers.py -q
python -m scripts.verify_dependency_locks
python scripts/verify_repo.py --check assets --fast
```

Run notebook JSON parse and a structural comparator against Task 1's parent commit. Require exactly the intended setup code cell and task spec to change; outputs/counts/IDs/metadata remain byte-equivalent after JSON projection.

- [ ] **Step 7: Commit Task 2**

Stage exactly the listed Task 2 files and commit:

```bash
git commit -m "build: route NLP consumers through verified assets"
```

---

### 12.26.1.3 Task 3: Land the Atlas JupyterHub asset hotfix and pin it

**Files in `thekaveh/atlas`:**
- Create: `services/jupyterhub/build/nlp-assets.toml`
- Create: `services/jupyterhub/build/install_nlp_assets.py`
- Create: `services/jupyterhub/build/nlp-model-requirements.txt`
- Create: `bootstrapper/tests/test_jupyterhub_nlp_assets.py`
- Modify: `bootstrapper/tests/test_security_dependency_floors.py`
- Modify: `services/jupyterhub/build/Dockerfile`
- Modify: `services/jupyterhub/build/requirements.txt`
- Modify: `docs/CHANGELOG.md`

**Files in ml-eng-lab:**
- Modify: `infra` gitlink
- Modify: `tests/test_atlas_consumer_contract.py`
- Modify: `docs/dependency-contracts.md` current Atlas pin row only

**Interfaces:**
- Consumes: byte-exact Task 1 manifest/script and existing `nlp-model-requirements.txt`.
- Produces: a reviewed Atlas `main` merge SHA and synchronized Atlas `develop`, then one exact parent gitlink pin.

- [ ] **Step 1: Preflight Atlas remote state without mutation**

Fetch `main` and `develop`; record their exact SHAs, trees, open PRs for the Issue #64 branch/base pair, required checks, and clean local submodule status. Require no ambiguous owned PR. Create `codex/hotfix-ml-eng-lab-issue-64-nlp-assets` from exact `origin/main` in a separate Atlas worktree.

- [ ] **Step 2: Write Atlas RED tests before copying projections**

`test_jupyterhub_nlp_assets.py` must require all three projected files, exact manifest/script/model contents supplied as immutable expected fixtures, Dockerfile copy/install/verify order, `NLTK_DATA=/home/jovyan/nltk_data`, and absence of legacy downloaders/raw download commands. Add mutations for wrong hash/URL/version, omitted `--require-hashes`, omitted `--no-deps`, omitted offline verification, verification before installation, later asset mutation, and a copied archive.

- [ ] **Step 3: Copy projections and update the Atlas Dockerfile**

Copy, do not rewrite:

```bash
cp "$ML_ROOT/requirements/nlp-assets.toml" services/jupyterhub/build/nlp-assets.toml
cp "$ML_ROOT/scripts/nlp_assets.py" services/jupyterhub/build/install_nlp_assets.py
cp "$ML_ROOT/nlp-model-requirements.txt" services/jupyterhub/build/nlp-model-requirements.txt
```

The Dockerfile block is exactly ordered:

```dockerfile
COPY --chown=${NB_UID}:${NB_GID} nlp-model-requirements.txt /tmp/nlp-model-requirements.txt
COPY --chown=${NB_UID}:${NB_GID} nlp-assets.toml /tmp/nlp-assets.toml
COPY --chown=${NB_UID}:${NB_GID} install_nlp_assets.py /tmp/install_nlp_assets.py
RUN python -m pip install --no-cache-dir --no-deps --require-hashes -r /tmp/nlp-model-requirements.txt \
 && python /tmp/install_nlp_assets.py install --manifest /tmp/nlp-assets.toml --data-dir /home/jovyan/nltk_data \
 && python /tmp/install_nlp_assets.py verify --manifest /tmp/nlp-assets.toml --data-dir /home/jovyan/nltk_data \
 && rm -f /tmp/nlp-model-requirements.txt /tmp/nlp-assets.toml /tmp/install_nlp_assets.py
ENV NLTK_DATA=/home/jovyan/nltk_data
```

The three projected inputs are owned by the active non-root notebook user so
the verified cleanup is executable. The Atlas contract test must kill a
mutation that omits any required `--chown=${NB_UID}:${NB_GID}`.

Remove the old spaCy/NLTK block. Update requirements comments and the Unreleased changelog only.

- [ ] **Step 4: Verify and publish the Atlas hotfix**

Run the focused Atlas test, its complete parent module, Ruff, syntax, docs checks, and native JupyterHub image build. Probe inside a disposable container:

```bash
python -c 'import en_core_web_sm; en_core_web_sm.load()'
python /tmp/install_nlp_assets.py verify --manifest /tmp/nlp-assets.toml --data-dir /home/jovyan/nltk_data
```

Because build inputs are removed in the final image, also run an inline Python evidence probe that hashes `/home/jovyan/nltk_data/sentiment/vader_lexicon.zip` and checks the spaCy distribution version. No persistent container or service is allowed.

Commit, push, open the hotfix PR to Atlas `main`, wait for all applicable checks, and merge. Then open a content-neutral Atlas `main` -> `develop` sync PR, require the sync tree to equal Atlas main, wait for all checks, and merge. Record main merge, final develop, image, and check URLs. Delete only the exact hotfix branch/ref after remote absence is proved.

- [ ] **Step 5: Pin and verify the parent consumer**

Update `infra` to the exact Atlas main merge. Extend `test_atlas_consumer_contract.py` to compare the three projections byte for byte against parent sources and parse Dockerfile order. Replace `PINNED_ATLAS_REVISION` with the exact merge SHA; update the current dependency ledger gitlink. Mutation tests change each projected byte, Docker order/flags, gitlink, and ledger SHA independently.

- [ ] **Step 6: Commit Task 3**

```bash
git add infra tests/test_atlas_consumer_contract.py docs/dependency-contracts.md
git commit -m "infra: pin Atlas verified NLP assets"
```

---

### 12.26.1.4 Task 4: Synchronize current documentation and immutable guards

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/env-setup.md`
- Modify: `docs/dependency-contracts.md`
- Modify: `docs/notebook-infrastructure.md`
- Modify: `docs/notebooks/sentiment_classification-vader-mlp-pytorch.md`
- Modify: `tests/test_check_docs.py`
- Modify: `tests/test_build_docs.py`
- Modify: `tests/test_wiki.py`

**Interfaces:**
- Consumes: final manifest, commands, Atlas SHA, and update process from Tasks 1-3.
- Produces: one consistent repository/site/wiki description and non-vacuous current-doc guards.

- [ ] **Step 1: Add current-section RED assertions**

Scope assertions to current README, Unreleased changelog, current dependency ledger, env setup, notebook infrastructure, and canonical sentiment page. Require exact VADER URL/hash/size, `make nlp-assets`, `make verify-nlp-assets`, offline behavior, update steps, spaCy Issue #63 boundary, Atlas projection/gitlink, and absence of current legacy downloader claims. Do not inspect historical changelog/maintenance sections for removal tokens.

- [ ] **Step 2: Update canonical documentation**

State plainly:

- `make install-torch-stack` installs the hash-required spaCy model wheel;
- `make nlp-assets` installs only the official hash-verified VADER ZIP;
- `make verify-nlp-assets` is offline and fails on missing/corrupt/wrong identity;
- a valid installed ZIP needs no network; clean installation requires the official URL;
- update order is official index review -> manifest -> Atlas projection -> clean install/build -> docs; and
- Atlas is not started for installation.

Replace stale §6.1.5 content completely. Update the canonical notebook page from its notebook/spec source rather than hand-diverging generated code excerpts.

- [ ] **Step 3: Regenerate and verify all three surfaces**

```bash
PYTEST_ADDOPTS='-p no:cacheprovider -W error' python -m pytest \
  tests/test_check_docs.py tests/test_build_docs.py tests/test_wiki.py -q
make docs-check
make docs-wiki
```

Require exact source-to-site and source-to-wiki parity for the design, plan, dependency contract, env setup, notebook infrastructure, and sentiment page. Mutate each required current claim and prove a test fails.

- [ ] **Step 4: Run repository guards and commit Task 4**

```bash
python scripts/verify_repo.py --check docs --fast
python scripts/verify_repo.py --check assets --fast
git diff --check
git add README.md CHANGELOG.md docs/env-setup.md docs/dependency-contracts.md \
  docs/notebook-infrastructure.md \
  docs/notebooks/sentiment_classification-vader-mlp-pytorch.md \
  tests/test_check_docs.py tests/test_build_docs.py tests/test_wiki.py
git commit -m "docs: document verified NLP assets"
```

The commit must not include generated site output, notebook outputs, lock files, policy files, image locks, or unrelated documentation.

---

### 12.26.1.5 Task 5: Freeze and run complete qualification

**Files:**
- Create ignored evidence under `.superpowers/sdd/issue64-evidence/`
- Create ignored report `.superpowers/sdd/issue64-qualification-report.md`

**Interfaces:**
- Consumes: the reviewed final feature SHA and exact Atlas gitlink.
- Produces: immutable local, Docker, Atlas, notebook, CI, publication, and cleanup evidence; no tracked commit after freeze.

- [ ] **Step 1: Independent pre-freeze review**

Review the complete feature range for spec compliance, fail-closed behavior, mutation quality, scope, and rollback. Resolve every Critical/Important/Minor finding in tracked commits, rerun affected gates, then obtain a clean approval. Freeze the resulting exact SHA. Any later tracked change invalidates all evidence and restarts this task.

- [ ] **Step 2: Create a fresh detached qualification root**

Create a new `/private/tmp/ml-eng-lab-issue64-final.*` root, detached worktree at `FINAL_SHA`, fresh Python 3.11 venv/cache/NLTK data directories, isolated kernelspec, and locally populated exact Atlas submodule. Prove clean status/diff/index, no preexisting outputs, exact interpreter prefix, exact gitlink, and no task containers/services.

- [ ] **Step 3: Run core tests and static gates**

Install the canonical locked stack, docs, audit tools, and VADER asset into the explicit empty evidence directory. Run offline verification with an opener/network guard that raises on access. Record asset path-relative identity, size, hash, member, smoke scores, install/verify timing, and no-fetch result.

Then run:

```bash
PYTEST_ADDOPTS='-p no:cacheprovider -W error' make test
make test-nnx-surface
make verify
make lint
make docs-check
make docs-wiki
make audit-advisories
```

Parse JUnit and require zero failures/errors/undocumented skips. Run `pip check`, Torch, NNx, dependency-lock, and NLP asset verifiers after the final package/data mutation.

- [ ] **Step 4: Run Docker and Atlas image qualification**

Build the parent image natively on arm64 without `--platform` and run the existing three disposable probes. Build the pinned Atlas JupyterHub image with no cache and run disposable probes for Python, spaCy version/load, VADER exact hash/size/member/smoke, and NLTK search-path resolution. Inspect both image architectures and record immutable image IDs. Remove both images after evidence capture; assert no container remains.

- [ ] **Step 5: Run all notebook tiers from fresh roots**

Run Tier A `18/18`, Tier B `6/6`, and Tier C `4/4` using the isolated kernelspec and separate absolute output roots. Apply the existing artifact verifier to every tier; require exactly `210`, `75`, and `56` executed code cells, zero error outputs, and clean source notebooks. No output from a failed/sandbox-blocked attempt may be reused.

- [ ] **Step 6: Build the immutable report**

Record final SHA/tree, Atlas SHA, Python/platform/package provenance, manifest/spaCy/installed VADER identities, test JUnit hashes/totals, advisory evidence, Docker/Atlas image evidence, tier counts/hashes/timings, clean-state evidence, and every command/exit code. Validate all referenced files/hashes and scan for local secrets/paths before publication.

---

### 12.26.1.6 Task 6: Publish through GitFlow, synchronize, and close Issue 64

**Files:** none tracked after Task 5 freeze.

**Interfaces:**
- Consumes: immutable Task 5 report and reviewed final SHA.
- Produces: merged feature/release/sync PRs, published Pages/wiki, clean branches/worktrees/images, Issue #64 Done/closed.

- [ ] **Step 1: Snapshot protected state**

Record exact substantive projections (`state,title,body,labels,assignees,projectItems`) for Issues #53/#65/#66 and require OPEN. Record the protected ruleset, main/develop SHAs/trees, open owned PR inventory, and Issue #64's unique project item/Status field/current option/Done option.

- [ ] **Step 2: Publish and qualify the feature PR**

Push exact `FINAL_SHA`, create or reuse only a canonical Issue #64 PR to `develop`, apply required labels before selecting CI, and trigger one explicit workflow dispatch. Require exact ten-job dispatch success and dual source/synthetic-merge evidence for every applicable PR workflow. Tier B must pass for the selected labeled/synchronize PR run; Tier C is the only intentional PR skip. Merge only after checks, source SHA, parents, and tree are exact.

- [ ] **Step 3: Publish and qualify the release PR**

Create or reuse exact `develop` -> `main`, wait for selected CI/Docs/Atlas evidence, distinguish legitimate push-run contamination by exact run/job URLs, and merge only when the synthetic tree equals the qualified feature tree. Require the exact new Pages run, `build` and `deploy` success, and live Pages/wiki content containing the VADER identity and update procedure.

- [ ] **Step 4: Synchronize main back to develop**

If ancestry requires it, create one content-neutral `main` -> `develop` sync PR, require tree equality and selected CI success with Tier B/C skips, merge, and wait for the exact final develop push run. Re-prove main/develop tree identity.

- [ ] **Step 5: Publish report and clean exact owned state**

Post the immutable report to feature PR, release PR, and Issue #64. Remove only validated Issue #64 worktrees, local/remote/tracking feature refs, temporary evidence roots, and exact Docker images. Preserve the preexisting dirty Issue #63 diagnostic worktree. Require zero open/ambiguous Issue #64 PRs and zero owned noncompleted runs.

- [ ] **Step 6: Final preservation and issue transition**

Recompare Issues #53/#65/#66 substantive snapshots byte for byte and require OPEN. Recompare protected ruleset. Post one concise completion note to Issue #53 and verify it remains OPEN. Requery Issue #64's exact project IDs/options, set only Status to Done, requery Done, then close Issue #64 with reason `completed` as the final external mutation. If project automation closes it, verify CLOSED/COMPLETED and do not reopen or add a later mutation.

- [ ] **Step 7: Final local audit**

Require primary checkout on clean `develop` at `origin/develop`, main/develop tree equality, exact Atlas gitlink, no Issue #64 refs/worktrees/images/containers, no open owned PRs, no owned noncompleted runs, and the ignored report present with its published SHA-256.
