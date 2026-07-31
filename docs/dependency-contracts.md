# 6.1 Dependency Contracts

This ledger records consumed dependency contracts that are intentionally pinned,
manual-only, or known to carry security/tooling constraints. It complements
`requirements.txt`, `torch-core-requirements.txt`, `torch-requirements.txt`,
`docs-requirements.in`, its compiled `docs-requirements.txt` lock, and the CI workflow; the manifests remain the source
of truth for installation.

## 6.1.1 Audit Snapshot

Last reviewed: 2026-07-04, on branch `codex/overnight-maintenance`.

Command:

```bash
pip-audit -r requirements.txt -r torch-requirements.txt
```

Result: 23 known vulnerabilities across three resolved packages:

Re-run on 2026-07-04 after adding `mkdocs-material`; the finding count and
accepted package set were unchanged.

| Package | Manifest Constraint | Audited Resolved Version | Finding Count | Current Disposition |
| --- | --- | ---: | ---: | --- |
| `torch` | `torch==2.4.1` | `2.4.1` | 21 | Local/CI compatibility baseline; upgrade requires a coordinated PyTorch/PyG/torchao compatibility pass. |
| `pytorch-lightning` | `pytorch-lightning==2.4.0` | `2.4.0` | 1 | Accepted temporarily because it is pinned to the current Torch stack; revisit with the Torch upgrade. |
| `nltk` | `nltk>=3.9.3` | `3.9.4` | 1 | Review on the next dependency bump; VADER usage is local/offline and does not deserialize untrusted corpus files. |

The audit must be re-run after any dependency pin change. New unreviewed audit
findings are maintenance issues until either fixed or added here with rationale.
Because several manifest entries are intentionally ranged or floating today,
the audited resolved versions and advisory IDs below are the accepted state, not
just the package-level counts.

Documentation-only CI jobs intentionally install `docs-requirements.txt` rather
than the full ML runtime stack. This keeps MkDocs and GitHub Pages builds from
resolving Torch, PyG, or NNx dependencies that are unrelated to rendered docs.
The five direct documentation tools are exactly pinned in `docs-requirements.in`;
`uv pip compile --universal --generate-hashes` locks their transitive dependencies
in `docs-requirements.txt`, which CI installs without re-resolving an open range.
The broader local development manifest still includes `mkdocs-material` so an
existing full-dev install can run `make docs-build` without a second setup step.

Accepted advisory IDs from the 2026-07-04 audit. `pip-audit` currently emits
23 feed records; two Torch advisory IDs appear twice from overlapping sources.

| Package | Advisory ID | Feed Records | Fix Versions |
| --- | --- | ---: | --- |
| `torch` | `PYSEC-2025-191` | 2 | `2.7.1rc1` / none listed |
| `torch` | `PYSEC-2025-41` | 2 | `2.6.0` |
| `torch` | `PYSEC-2024-259` | 1 | `2.5.0` |
| `torch` | `PYSEC-2025-205` | 1 | `2.7.1` |
| `torch` | `PYSEC-2025-206` | 1 | `2.9.0` |
| `torch` | `PYSEC-2025-207` | 1 | `2.7.1` |
| `torch` | `PYSEC-2025-204` | 1 | `2.9.0` |
| `torch` | `PYSEC-2026-139` | 1 | none listed |
| `torch` | `PYSEC-2025-209` | 1 | `2.7.1` |
| `torch` | `PYSEC-2025-208` | 1 | `2.7.1` |
| `torch` | `PYSEC-2025-198` | 1 | `2.7.0` |
| `torch` | `PYSEC-2025-203` | 1 | `2.9.0` |
| `torch` | `CVE-2025-3730` | 1 | `2.8.0` |
| `torch` | `CVE-2025-2148` | 1 | none listed |
| `torch` | `CVE-2025-2149` | 1 | none listed |
| `torch` | `CVE-2025-2998` | 1 | none listed |
| `torch` | `CVE-2025-2999` | 1 | `2.9.1` |
| `torch` | `CVE-2025-3000` | 1 | none listed |
| `torch` | `CVE-2025-3001` | 1 | `2.10.0` |
| `pytorch-lightning` | `CVE-2026-31221` | 1 | none listed |
| `nltk` | `PYSEC-2026-597` | 1 | none listed |

## 6.1.2 Torch Stack Pin

`torch-core-requirements.txt` pins the core Torch stack:

- `torch==2.4.1`
- `pytorch-lightning==2.4.0`
- `torchvision==0.19.1`
- `torchaudio==2.4.1`
- `torchmetrics==1.4.2`

`torch-requirements.txt` includes `torch-core-requirements.txt` and then pins:

- PyG wheels resolved from `https://data.pyg.org/whl/torch-2.4.0+cpu.html`
- `torch-scatter==2.1.2`
- `torch-sparse==0.6.18`
- `torch-cluster==1.6.3`
- `torch-spline-conv==1.2.2`
- `torch_geometric==2.6.1`

Reason: this is the deliberately stable local/CI compatibility baseline. It is
not required to match the separately pinned Atlas JupyterHub runtime.

Upgrade criteria:

1. Select a Torch version with matching `torchvision`, `torchaudio`, and PyG CPU
   wheels.
2. Confirm `torchao>=0.17` imports under that Torch version.
3. Re-run `make test`, `make verify`, `make test-nnx-surface`, and at least the
   smoke Tier-B/Tier-C notebooks on Linux.
4. Update README, environment docs, and this ledger in the same change.

## 6.1.3 Manual-Only Quantization Notebook

`notebooks/quantization-mnist-ffnn-pytorch/notebook.ipynb` depends on `torchao>=0.17`.
That torchao API references `torch.int1` at import time, which is unavailable in
the pinned `torch==2.4.1` environment. The notebook remains an active task but is
manual-only until the Torch stack is upgraded.

Expected local side environment for this notebook:

- `torch>=2.5`
- `torchao>=0.17`

Do not add the quantization notebook back to `Makefile` Tier-A/B/C until the
repository-wide local/CI Torch stack supports it. Atlas has a newer observed
package surface, but package availability alone is not a full notebook smoke.

## 6.1.4 Papermill CLI Contract

`requirements.txt` pins `papermill==2.7.0` because notebook re-execution is a
consumed CLI contract, not just a Python import. The Makefile invokes it as
`python -m papermill` by default through `PAPERMILL ?= $(PYTHON) -m papermill`
so stale console-script shebangs cannot break notebook targets.

Verified contract for `papermill==2.7.0`:

- `python -m papermill --version` resolves the installed module.
- `python -m papermill --help` exposes `--kernel` / `-k`, `--parameters` /
  `-p`, `--start-timeout`, and `--execution-timeout`, which the Tier-A/B/C
  Makefile targets use.
- The injected `SMOKE_TEST` parameters cell remains parser-friendly for
  papermill 2.7; `tests/test_inject_smoke_test_cell.py` guards this shape.
- The Makefile centralizes notebook launch limits through
  `PAPERMILL_START_TIMEOUT` and `PAPERMILL_EXECUTION_TIMEOUT`; override those
  variables locally rather than deleting timeout flags from the targets.

Upgrade criteria:

1. Confirm `python -m papermill --version` reports the intended version.
2. Confirm `python -m papermill --help` still accepts the Makefile flags.
3. Run `pytest tests/test_inject_smoke_test_cell.py tests/test_verify_repo.py`.
4. Run at least one cheap notebook target through `make run-tier-a` or a
   targeted papermill command from the notebook directory.

## 6.1.5 External Assets

`make nlp-assets` downloads:

- spaCy `en_core_web_sm`
- NLTK `vader_lexicon`

These assets are consumed by the text-classification and sentiment notebooks.
They are not locked by checksum today. If reproducibility becomes stricter than
the current educational-notebook standard, add a lock/verification mechanism and
update this section.

## 6.1.6 NNx PyPI Pin and Editable Override Boundary

`requirements.txt` pins `thekaveh-nnx[lm]==0.2.0`. That PyPI distribution is
the canonical contract for ml-eng-lab notebook verification and CI. The static
NNx surface tests intentionally inspect the installed `nnx` import surface, so
they are only exact release-contract evidence when the environment resolves
`nnx` from the pinned PyPI wheel.

Editable installs are allowed only for active upstream NNx development, using
the workflow in README §6. When an
editable checkout is active, local tests are development-surface evidence, not
release-contract evidence. Before treating local `tests/nnx_surface` results as
release evidence, confirm:

```bash
python - <<'PY'
import importlib.metadata as md
import json
from pathlib import Path

dist = md.distribution("thekaveh-nnx")
direct_url = Path(dist._path) / "direct_url.json"
print(md.version("thekaveh-nnx"))
print(json.loads(direct_url.read_text()) if direct_url.exists() else "wheel install")
PY
```

Expected release-contract state: version `0.2.0` and no editable
`direct_url.json`. If the output reports `{"editable": true}`, reinstall from
`requirements.txt` before recording exact pinned-contract evidence, or document
that the run intentionally used a local NNx development checkout.

## 6.1.7 Atlas Infra Submodule Contract

`.gitmodules` consumes `https://github.com/thekaveh/atlas.git` as the active
`infra` submodule.
Current Atlas `infra` gitlink SHA: `61c7c5103660e2226bf107c115dae42bf46f8374`.

This reviewed superproject gitlink is the active Atlas dependency contract.
Consumer configuration is deliberately outside `infra/`: `atlas.consumer.yml`,
`atlas.env.user.example`, and `compose/ml-eng-lab-atlas.yml` define the track,
native-source policy, and mount. Pin changes follow
[atlas-pin-bump-runbook.md](atlas-pin-bump-runbook.md).

## 6.1.8 Atlas Jupyter Runtime Evidence

Last verified: 2026-07-30 against the pinned `ml-eng` Atlas track. The live
runtime probe ran inside the JupyterHub container after the consumer mounted
this checkout and reported zero failed checks. The host-native Ollama probe
also succeeded; no Ollama or ComfyUI container was running for the consumer.

| Surface | Observed in Atlas JupyterHub | Contract meaning |
| --- | --- | --- |
| Python | CPython 3.11.10 | Remote notebook interpreter |
| NNx + language extras | `thekaveh-nnx` / `nnx` 0.2.0; `datasets` 5.0.0; `tokenizers` 0.22.2 | Matches notebook imports and the `[lm]` extra |
| Torch | `torch` 2.11.0+cpu; `torchvision` 0.26.0+cpu; `torchaudio` 2.11.0+cpu | Atlas runtime is newer than local/CI; do not infer a local pin bump |
| Torch extensions | `torchao` 0.17.0; `torch-geometric` 2.6.1; `python-louvain` 0.16 | Required import surfaces are present |
| NLP | spaCy 3.8.14, `en_core_web_sm` 3.8.0; NLTK 3.10.0 with VADER | Both task assets resolve |
| Notebook imports | 62 imports across active notebooks; zero failures | Import-level compatibility evidence |

The same live check imported the NumPy MNIST sibling modules from the mounted
checkout. This validates the consumer mount separately from package metadata.
It does not turn a successful import into a completed training or performance
smoke.

## 6.1.9 Atlas Versus Local/CI Dependency Boundaries

Atlas's runtime image is infrastructure-owned and may advance independently of
the checked-in local/CI manifests. The local/CI Torch 2.4.1 contract remains the
source of truth for `make test`, papermill CI, Dockerfile, and Codespaces.
Conversely, notebooks using the remote kernel rely on the observed Atlas package
surface recorded above. Any change that makes a notebook depend on a version
only Atlas provides must add an explicit task contract and update both this
ledger and the relevant runbook.

The quantization notebook is still manual-only. Atlas now imports the needed
package layer, but its complete PTQ/QAT execution has not been run there. A
future reclassification requires a targeted Atlas notebook smoke as well as the
local/CI compatibility decision; do not promote it based only on `torchao`
metadata.

## 6.1.10 GitHub Actions Pins

Workflow actions are pinned to exact commit SHAs, with an inline version comment
showing the reviewed upstream major tag. On 2026-07-04, the reviewed tag refs
were:

| Action | Reviewed Tag | Pinned SHA |
| --- | --- | --- |
| `actions/checkout` | `v7` | `9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0` |
| `actions/setup-python` | `v6` | `ece7cb06caefa5fff74198d8649806c4678c61a1` |
| `actions/upload-artifact` | `v7` | `043fb46d1a93c77aae656e7c1c64a875d1fc6a0a` |
| `actions/configure-pages` | `v6` | `45bfe0192ca1faeb007ade9deae92b16b8254a0d` |
| `actions/upload-pages-artifact` | `v5` | `fc324d3547104276b827a68afc52ff2a11cc49c9` |
| `actions/deploy-pages` | `v5` | `cd2ce8fcbc39b97be8ca5fce6e763baed58fa128` |

Upgrade criteria:

1. Resolve the intended tag with `git ls-remote --tags`.
2. Update the workflow SHA and inline tag comment together.
3. Parse workflow YAML and run the relevant local contract checks.

## 6.1.11 Bootstrap Tooling Gap

The bootstrap paths still upgrade or install the Python packaging toolchain
without exact pip/setuptools pins:

- `Makefile` target `install-torch-stack` runs `pip install --upgrade pip`.
- `Dockerfile` upgrades `pip` and `setuptools` before project requirements.

This is accepted temporarily because pinning bootstrap tools changes every
environment creation path and belongs with the coordinated dependency-lock
work. Until then, maintenance passes should treat unexpected resolver behavior
or build-isolation changes as dependency-contract findings.

## 6.1.12 Deferred Reproducibility Hardening

The current manifests still include floating and ranged Python dependencies, and
the Docker/devcontainer bases are tag-pinned rather than digest-pinned. A full
lockfile, CI install against that lock, `pip-audit` comparison against accepted
advisory IDs, and base-image digest pinning are intentionally deferred to a
coordinated dependency-refresh pass because they can change every notebook
runtime at once.
