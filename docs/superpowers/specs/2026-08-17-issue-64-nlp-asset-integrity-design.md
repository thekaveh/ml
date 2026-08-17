# 12.25 Issue 64 NLP Asset Integrity Design

## 12.25.1 Purpose

Issue #63 made the Python environment deterministic, including the spaCy
`en-core-web-sm==3.8.0` wheel. It did not lock the separate NLTK
`vader_lexicon` data download. `make nlp-assets`, Tier A, Docker, Codespaces,
and the pinned Atlas JupyterHub image still delegate that download to NLTK's
moving package index. The sentiment notebook also retries the same unverified
download at runtime.

Issue #64 closes that remaining asset gap. One repository-owned manifest binds
the official VADER archive URL, byte size, SHA-256 digest, NLTK resource path,
archive member, and license. One standard-library installer verifies that
identity before atomically placing the archive in an NLTK data directory. Every
ml-eng-lab setup path calls that installer, notebooks never download data, and
the pinned Atlas consumer carries byte-identical manifest and installer inputs.

The selected upstream identity is the current official NLTK data index entry:

- URL:
  `https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/sentiment/vader_lexicon.zip`
- archive size: `90486` bytes;
- SHA-256:
  `8adba4294eef3964d820bf655e37e61bdc3a341994356af59b74fb3b4a36ce5c`;
- installed resource: `sentiment/vader_lexicon.zip`;
- required member: `vader_lexicon/vader_lexicon.txt`; and
- license: `MIT License`.

The upstream NLTK index is evidence for a reviewed update, not a runtime source
of truth. Installation reads only the committed manifest.

## 12.25.2 Selected strategy

The selected strategy is a direct, hash-verified download of the official NLTK
archive with a repository-owned installer.

Two alternatives are rejected:

1. **Vendor the archive in ml-eng-lab or Atlas.** This removes the network
   dependency but creates a third-party redistribution and license-maintenance
   contract that Issue #64 explicitly excludes.
2. **Keep `nltk.download()` and verify afterward.** A post-check detects bad
   bytes only after NLTK has selected a moving index entry and written them. It
   leaves source selection outside the reviewed repository contract and makes
   stable offline behavior difficult to prove.

The direct installer is small, uses only the Python standard library, never
extracts the archive, and is portable across Darwin arm64, Linux x86_64, and
Linux arm64. The archive remains in the ZIP form NLTK consumes directly.

## 12.25.3 Authoritative identities

`requirements/nlp-assets.toml` is the sole authority for non-package NLP data.
It contains exactly one `[vader_lexicon]` table and no unrecognized keys:

```toml
schema_version = 1

[vader_lexicon]
url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/sentiment/vader_lexicon.zip"
sha256 = "8adba4294eef3964d820bf655e37e61bdc3a341994356af59b74fb3b4a36ce5c"
size = 90486
resource = "sentiment/vader_lexicon.zip"
member = "vader_lexicon/vader_lexicon.txt"
license = "MIT License"
```

The parser requires `schema_version` to be the integer `1`, rejects booleans as
integers, requires the exact HTTPS origin/path, requires a lowercase 64-digit
SHA-256, requires the exact positive size, and rejects absolute paths, `..`,
backslashes, duplicate tables, and extra keys. The URL must contain no query,
fragment, username, password, or non-default port.

The spaCy asset remains package-managed. `nlp-model-requirements.txt`,
`requirements/lock-policy.toml`, and all three platform root locks already bind
the official `en_core_web_sm-3.8.0` wheel to SHA-256
`1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85`.
Issue #64 verifies that identity and removes every remaining `spacy download`
consumer; it does not introduce a second spaCy version authority.

## 12.25.4 Installer and verifier interface

`scripts/nlp_assets.py` exposes these public interfaces:

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
    category: str

def load_manifest(path: Path) -> VaderAsset: ...
def default_nltk_data_dir() -> Path: ...
def verify_vader(asset: VaderAsset, data_dir: Path) -> Path: ...
def install_vader(
    asset: VaderAsset,
    data_dir: Path,
    *,
    offline: bool,
    opener: Callable[[str], ContextManager[BinaryIO]],
) -> Path: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

The CLI has two commands:

```text
python -m scripts.nlp_assets install [--manifest PATH] [--data-dir PATH] [--offline]
python -m scripts.nlp_assets verify  [--manifest PATH] [--data-dir PATH]
```

The default manifest is `requirements/nlp-assets.toml`. The default data
directory is `nltk.downloader.Downloader().default_download_dir()`, which
preserves NLTK's existing per-user locations. `--data-dir` is an explicit test,
container, or operator override. The resolved target must be an absolute path
under that data directory. Symlinked target files or parent directories are
rejected.

`verify` performs no network operation. It requires the exact archive path,
byte size, SHA-256, one readable ZIP central directory, and the exact required
member as a regular file. It then constructs `SentimentIntensityAnalyzer` with
an NLTK search path containing the selected data directory and proves the
stable smoke scores classify `"good"` as positive and `"bad"` as negative.

`install` first calls `verify`. A valid existing archive returns success without
opening the network, including in `--offline` mode. A missing archive with
`--offline` fails. Online installation streams to a uniquely created temporary
file in the target directory, bounds the response to the expected size, fsyncs,
and verifies the full file and ZIP identity. It then uses a same-filesystem hard
link to create the missing target atomically without overwriting a concurrent
writer and unlinks the temporary name. If a target exists but is corrupt or has
the wrong identity, the installer fails rather than overwriting operator data.
A race-created target is independently verified before success. Temporary files
are removed on every exit path.

CLI diagnostics expose only stable categories:

```text
nlp asset operation failed: manifest
nlp asset operation failed: destination
nlp asset operation failed: missing
nlp asset operation failed: network
nlp asset operation failed: size
nlp asset operation failed: hash
nlp asset operation failed: archive
nlp asset operation failed: smoke
```

No URL, response body, token, local path, or exception representation appears
in stdout/stderr.

## 12.25.5 Repository consumers

The Make contract becomes:

```make
nlp-assets:
	$(PYTHON) -m scripts.nlp_assets install

verify-nlp-assets:
	$(PYTHON) -m scripts.nlp_assets verify
```

`codespace-setup` runs `nlp-assets`, `verify-nlp-assets`, pip-check, and the
existing Torch/NNx verifiers, with no later package or asset mutation. The root
Dockerfile runs the same Make targets. Tier A CI runs installation and
verification before notebook execution. Jobs and tiers that do not consume NLP
assets do not download them.

The sentiment notebook removes its `LookupError -> nltk.download()` fallback.
It requires `nltk.data.find("sentiment/vader_lexicon.zip")` and raises a concise
setup error instructing the operator to run `make nlp-assets` when the verified
asset is absent. Notebook code never performs network I/O. Its canonical
Markdown page and task specification describe the same behavior.

Repository verification rejects:

- any `nltk.download`, `nltk.downloader`, or `spacy download` command in current
  Make, Docker, workflow, devcontainer, script, or active-notebook code;
- an `nlp-assets` recipe not equal to the canonical module command;
- a consumer that installs assets after verification or workload execution;
- a consumer that omits `verify-nlp-assets` where it installs the asset; and
- any direct VADER URL/hash outside the authoritative manifest, the pinned Atlas
  projection described below, and the exact current documentation/test
  assertions that derive from and cross-check those two surfaces.

Historical documentation is not rewritten. Current documentation and active
notebook guidance must not describe the legacy downloader as supported.

## 12.25.6 Atlas consumer projection

Atlas cannot import ml-eng-lab code while building its JupyterHub image. The
cross-repository contract therefore uses reviewed, byte-identical projections:

- `infra/services/jupyterhub/build/nlp-assets.toml` equals
  `requirements/nlp-assets.toml` byte for byte;
- `infra/services/jupyterhub/build/install_nlp_assets.py` equals
  `scripts/nlp_assets.py` byte for byte; and
- `infra/services/jupyterhub/build/nlp-model-requirements.txt` equals
  `nlp-model-requirements.txt` byte for byte.

The Atlas Dockerfile copies those three files with
`--chown=${NB_UID}:${NB_GID}` so its active non-root notebook user can remove
the verified temporary inputs, installs the spaCy model with
`python -m pip install --no-deps --require-hashes -r`, installs VADER with the
shared script into `/home/jovyan/nltk_data`, verifies it offline with the same
script, sets `NLTK_DATA=/home/jovyan/nltk_data`, and removes temporary build
inputs. It contains no `spacy download`, `nltk.download`, raw curl/wget asset
download, unverified pip install, or copied third-party archive.

The Atlas change is a consumer hotfix, not an Atlas feature release. It branches
from the current Atlas `main`, merges through a reviewed PR back to `main`, and
then uses a content-neutral `main`-to-`develop` sync PR. This avoids releasing
unrelated Atlas `develop` work while leaving both long-lived branches with the
same asset contract. ml-eng-lab then updates its detached `infra` gitlink to the
reviewed Atlas `main` merge. The parent Atlas consumer test checks the exact
gitlink plus all three byte-equality projections and Dockerfile order. A dirty
or unavailable submodule fails closed; tests never fetch a moving branch.

`scripts/atlas_runtime_probe.py` strengthens the VADER evidence. It resolves the
resource through `nltk.data.find`, requires a filesystem ZIP owned by an entry
in `nltk.data.path`, hashes it, compares exact size/digest/member identity, and
includes only safe expected/observed digest and size fields. A missing, corrupt,
non-file, symlink, wrong-member, or wrong-hash resource produces
`status="asset_identity_mismatch"`. spaCy evidence requires distribution version
`3.8.0`, an importable `en_core_web_sm`, and a loadable model.

No Atlas service is started merely to install assets. Static consumer tests and
a native Atlas JupyterHub image build are mandatory. If runtime evidence is
needed, use the existing remote Atlas lifecycle and shut it down through the
existing lifecycle command; do not start Docker Compose, Ollama, or unrelated
services locally.

## 12.25.7 Update procedure

A VADER update is one reviewed transaction:

1. Read the official NLTK `index.xml` entry and the upstream archive license.
2. Download the official archive to a disposable directory; record URL, size,
   SHA-256, ZIP member inventory, and a VADER sentiment smoke result.
3. Update only `requirements/nlp-assets.toml` first and observe manifest/current
   consumer tests fail.
4. Update the shared installer projection and Atlas projection in the same
   reviewed change; all byte-equality and mutation tests must pass.
5. Run clean online installation followed by network-disabled verification on
   every supported platform available in CI or qualification.
6. Rebuild the native Atlas JupyterHub image, prove the spaCy model and VADER
   identities, and record the new Atlas main-ancestor gitlink.
7. Update current docs and regenerate site/wiki surfaces.

The spaCy model remains updated through Issue #63's dependency-lock procedure.
Changing `nlp-model-requirements.txt` requires regenerating all three root locks
and updating the byte-identical Atlas projection; Issue #64 adds no shortcut.

## 12.25.8 Tests and mutation gates

Focused tests cover:

- exact manifest parsing and rejection of missing, duplicate, extra, malformed,
  non-HTTPS, credentialed, queried, fragmented, wrong-host, wrong-path, unsafe
  resource/member, boolean-size, wrong-size, and wrong-hash inputs;
- valid existing archive success with an opener that raises if called;
- missing offline, corrupt existing, symlink, directory, truncated stream,
  oversized stream, network error, wrong digest, malformed ZIP, wrong member,
  and failed sentiment smoke behavior;
- temporary-file cleanup, no replacement of corrupt existing data, and atomic
  replacement of a missing target;
- redacted CLI diagnostics and success output;
- exact Make, Docker, CI, Codespaces, active-notebook, and Atlas consumer order;
- byte-identical Atlas manifest/script/model projections and exact pinned
  gitlink;
- Atlas probe success plus missing/corrupt/symlink/wrong-size/wrong-hash/member
  mutations; and
- current documentation and generated site/wiki parity.

Mutation tests must first assert that each mutation changed the intended source.
They must kill removal of the hash check, size bound, ZIP/member check, offline
guard, no-network short circuit, symlink guard, temporary cleanup, atomic
replacement, VADER smoke, spaCy version check, consumer verification step, and
Atlas byte-equality check.

## 12.25.9 Qualification and publication

The final immutable ml-eng-lab feature SHA is qualified from a fresh detached
worktree and fresh Python 3.11 environment. Required evidence is:

- focused asset tests and all mutation controls;
- full repository tests under warnings-as-errors with zero failures, errors, or
  undocumented skips;
- dependency-lock, advisory, lint, repository, strict docs, and wiki gates;
- clean online VADER install into an empty explicit data directory, followed by
  successful offline verification and a network-denied no-fetch control;
- root native Linux arm64 Docker build and the existing three runtime probes;
- Atlas native JupyterHub image build with exact spaCy/VADER identity evidence;
- Tier A `18/18`, Tier B `6/6`, and Tier C `4/4` fresh notebook outputs with the
  existing artifact and source-clean oracles; and
- clean tracked state, exact Atlas gitlink, and no service/container residue.

The repository follows feature -> `develop` -> `main` GitFlow. Feature and
release PR evidence must bind the source SHA and executed synthetic merge SHA,
all applicable checks must succeed, Pages and wiki must publish the exact main
tree, and any content-neutral main-to-develop sync must finish successfully.
Issue #64 moves to Done and closes only after the immutable report is posted,
cleanup is complete, and Issues #53, #65, and #66 remain open and substantively
unchanged.

## 12.25.10 Rollback

Rollback is independently scoped from the dependency locks. Revert the
ml-eng-lab feature commit and the Atlas NLP-asset commit, repin `infra` to its
previous reviewed main ancestor, and rebuild affected images. Do not restore
unverified runtime download fallbacks. If the official archive is unavailable,
existing verified copies continue to work offline; clean installation fails
closed until a reviewed upstream identity is available.
