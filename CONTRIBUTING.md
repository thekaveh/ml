# Contributing

A short guide for adding new notebook experiment folders and modifying shared code in this lab.

## 1. Conventions

- This is a notebook-driven ML lab. Each active task is a self-contained directory under `notebooks/` using the `[task]-[dataset]-[model]-[framework]` naming convention. Do not introduce a `tasks/` subdirectory or family-prefixed dirs (`vision/`, `nlp/`, ...).
- Shared library code lives in **`thekaveh-nnx`** — the PyTorch toolkit installed from PyPI ([source: `thekaveh/NNx`](https://github.com/thekaveh/NNx)), pinned in `requirements.txt` to `thekaveh-nnx[lm]==0.2.0` for compatibility with the recommended Atlas JupyterHub runtime. Notebooks import via `from nnx.X import Y`. Do not reintroduce a local `common/` directory — `scripts/verify_repo.py` enforces this via `S7.forbidden_toplevel`.
- The `notebooks/archive/` directory holds preserved-as-is experiments. Read-only.
- New notebooks should include a top markdown cell stating purpose and dataset, plus the canonical §1–§6 hierarchy (Overview / Setup / Data / Model / Training / Evaluation & Results). Phase-1 exploration notebooks use a variant: §1, §2, §3 Dataset deep-dive.

Report suspected vulnerabilities through the private route in
[SECURITY.md](SECURITY.md#132-report-a-vulnerability-privately). Do not open a public issue,
discussion, or pull request containing an undisclosed vulnerability or sensitive proof of concept.

## 2. Workflow

1. Open a feature branch off `develop`; merge it there first, then promote `develop` to `main` in a separate PR.
2. Make your change.
3. Run `make verify` (wraps `python scripts/verify_repo.py --check all --fast`) — must exit 0 (no error-severity findings; warnings are OK).
4. Run `make verify-nnx-install` to record canonical local evidence for the exact released NNx wheel.
   Intentional upstream work may instead run
   `NNX_ALLOW_EDITABLE=1 make test-nnx-surface`, but those results are
   development-surface evidence, never released-wheel evidence. Return to canonical mode before
   recording release compatibility.
5. Run `make test` (wraps `pytest tests/`) locally. CI runs the same complete contract on every PR as the required `pytest-repository` job; `pytest-nnx-surface` remains the faster focused NNx/PyPI compatibility and Ruff signal.
6. Before a PR that changes the parent-owned Atlas consumer policy, run `make test-atlas-consumer`. `atlas-consumer-policy` is unconditional on every pull request and is intended to be a required gate. `atlas-contract` remains a separate, path-scoped, non-required direct validator of the recursive `infra/` submodule.
7. Before a PR that changes an audited dependency manifest or the accepted-advisory policy, run `make audit-advisories`. The isolated `dependency-audit` job validates the resolver-safe and pre-resolved PyG audit projections against the runtime manifest, then merges their strict results into the same four logical surfaces. After the controller updates the GitHub ruleset, it is required alongside `pytest-repository` and `atlas-consumer-policy`.
8. If you touched a notebook, re-run it at its tier (Tier-A: `make run-tier-a` only when deliberately refreshing a committed snapshot; `make smoke-tier-a` is the non-mutating CI-equivalent target; Tier-B: `make smoke-tier-b`; Tier-C: `make smoke-tier-c`). Tier-C **code cells** must remain identical to the `pre-cleanup-baseline` tag — verify check E5 enforces this (markdown and embedded outputs are not compared).
9. Open a PR. CI runs complete pytest, Atlas consumer policy, dependency audit, and Tier-A automatically; Tier-B runs on schedule, on `workflow_dispatch`, and on PRs labeled `tier-b-smoke`; Tier-C runs on schedule and on `workflow_dispatch`.

## 3. Adding a new task folder

Convention: active experiment directory named `notebooks/[task]-[dataset]-[model]-[framework]/`.

1. Survey [`thekaveh/NNx`'s `src/nnx/`](https://github.com/thekaveh/NNx/tree/main/src/nnx) for reusable primitives.
2. Identify gaps. If you need new primitives, **land them in [`thekaveh/NNx`](https://github.com/thekaveh/NNx) first** (open a PR upstream), wait for the next NNx PyPI release, then bump `requirements.txt`'s `thekaveh-nnx` version pin here.
3. Scaffold the new task folder with a `README.md` (use [`notebooks/node_classification-reddit-gnn-pyg/README.md`](notebooks/node_classification-reddit-gnn-pyg/README.md) as template) and notebook(s). At the top of §3 "What's in the notebook(s)", include the nbviewer tip — GitHub's notebook renderer chokes on cells with large embedded matplotlib PNGs:

   ```markdown
   > **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-eng-lab/blob/main/notebooks/<folder>/<notebook>.ipynb) for full rendering.
   ```

   For folders with multiple notebooks, link to the folder view at `https://nbviewer.org/github/thekaveh/ml-eng-lab/tree/main/notebooks/<folder>/` instead.
4. Add `docs/spec.yaml` with the required `atlas:` mapping: `executor: jupyterhub`, `default_mode`, `required_services`, `workspace_access`, `artifact_policy`, and constraints. Use `default_mode: vscode-remote` with `workspace_access: remote` for the normal local-editor/remote-kernel path. Use `default_mode: mounted-workspace` only with `workspace_access: mounted-required`, then direct the task to Browser JupyterLab or VS Code attached to the JupyterHub container. Start with `required_services: [jupyterhub]`; only add a service after the future-service admission sequence below.
5. Add every active notebook to `required_sections` in [`scripts/verify_repo_config.yaml`](scripts/verify_repo_config.yaml); ordinary task notebooks should copy the canonical six-section block.
6. Run `make docs-sync-notebook-infrastructure`, then keep the generated task-contract table in [`docs/notebook-infrastructure.md`](docs/notebook-infrastructure.md) in the same commit. If Tier-A, add the notebook path to `tier_a_notebooks` in the same YAML and to `TIER_A` in [`Makefile`](Makefile).
7. For a service beyond JupyterHub, follow future-service admission: declare the need in the task spec, wire only consumer-owned in-network configuration, add contract coverage and a targeted JupyterHub smoke, then update the task docs, dependency ledger, and diagrams as needed. See [docs/atlas-pin-bump-runbook.md](docs/atlas-pin-bump-runbook.md#623-future-service-admission).
8. Update the root README's task table.
9. Tick the box on the root README roadmap.
10. YAGNI: don't add abstractions to `nnx` speculatively. Only land features when a concrete task needs them.

## 4. Modifying shared code

- **`thekaveh-nnx` is a PyPI dep.** Don't bump the `requirements.txt` pin without a corresponding upstream release on [`thekaveh/NNx`](https://github.com/thekaveh/NNx). Workflow:
  1. Open a PR against `thekaveh/NNx` with the new feature + a smoke test.
  2. After merge, wait for the next NNx PyPI release. For editable iteration, clone `thekaveh/NNx` outside the ml-eng-lab tree, `pip install -e <path>[lm]` into your venv, and run `NNX_ALLOW_EDITABLE=1 make test-nnx-surface`; this validates the development surface only.
  3. Bump `thekaveh-nnx[lm]==X.Y.Z` in ml-eng-lab's `requirements.txt` to the new version; open a PR here. Every NNx release review must run the complete Tier A, Tier B, and Tier C matrix plus execute the quantization notebook manually in a fresh canonical environment installed by `make install-torch-stack`; the notebook remains outside Tier A/B/C under Issue #66, even when the candidate appears not to touch those surfaces.
- **`infra/` is the pinned Atlas submodule.** Do not edit it from this repository. Consumer-owned behavior belongs in `atlas.consumer.yml`, `atlas.env.user.example`, and `compose/ml-eng-lab-atlas.yml`; runbook changes belong under `docs/`. Update the gitlink only through [docs/atlas-pin-bump-runbook.md](docs/atlas-pin-bump-runbook.md).
- **`notebooks/archive/` is read-only.** Preserved Aug-2023 work.

Found an issue in the `thekaveh-nnx` library? Append to [docs/FINDINGS-NNX.md](docs/FINDINGS-NNX.md) (and open an upstream issue at [`thekaveh/NNx`](https://github.com/thekaveh/NNx/issues)). For Atlas-consumer behavior, append to [docs/FINDINGS-ATLAS.md](docs/FINDINGS-ATLAS.md); fixes in Atlas itself belong upstream.

## 5. Running notebooks

Primary runtime: the `ml-eng` Atlas track, accessed from local VS Code through the running Atlas JupyterHub server. Set it up with `git submodule update --init --recursive`, `make atlas-setup`, `make atlas-up`, and `make atlas-connect`; see [docs/jupyterhub-integration.md](docs/jupyterhub-integration.md). Most tasks use the local-editor/remote-kernel path. The NumPy MNIST task is `mounted-required`: use Browser JupyterLab or VS Code attached to the JupyterHub container from `/home/jovyan/work/ml-eng-lab`, rather than a local notebook paired with the remote kernel. The local/CI quantization path remains manual-only; Atlas package availability alone is not a full notebook validation.

- **Native Ollama only** — Atlas is fixed to `LLM_PROVIDER_SOURCE=ollama-localhost`. Start or manage `ollama serve` on the host; never add or start an Ollama container for this consumer.
- **ComfyUI deferred** — `ml-eng` disables it by default. A future task may request only a reviewed host-native source (`localhost` or managed MPS); container and automatic sources are prohibited.
- **Fallback runtime** — attach VS Code to the running JupyterHub container or use browser JupyterLab only when remote-kernel editing cannot meet the task's needs. Keep the default local-editor / remote-kernel workflow.
- **Editable-iteration on NNx itself** — clone `thekaveh/NNx` outside the ml-eng-lab tree, then `pip install -e <path>[lm]` into your venv to override the PyPI install. No in-repo override script.
- **Zero-click cloud dev (GitHub Codespaces)** — `Code → Codespaces → Create codespace on main` on github.com/thekaveh/ml-eng-lab. `.devcontainer/devcontainer.json`'s `postCreateCommand` runs `make codespace-setup` (full pip install + NLP assets, ~2-3 min one-time). See [README.md §3.4](README.md#34-github-codespaces-zero-click-cloud-dev) for the motivation + scenario list (and the GPU + persistence caveats).
- Atlas operations, pin bumps, and future-service admission: [docs/atlas-pin-bump-runbook.md](docs/atlas-pin-bump-runbook.md).

### 5.1. One-time NLP-task setup

Two Tier-A tasks need a model + a lexicon that `pip install -r requirements.txt` doesn't pull on its own. Run these once after the venv is set up (CI runs them automatically in `.github/workflows/ci.yml`'s `tier-a-papermill` job):

```bash
# spaCy English model — needed by text_classification-agnews-spacy-mlp-pytorch
# and sentiment_classification-vader-mlp-pytorch
python -m spacy download en_core_web_sm

# NLTK VADER lexicon — needed by sentiment_classification-vader-mlp-pytorch
# (the notebook also has a lazy fallback download, but pre-downloading avoids
# the per-run delay)
python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"
```

## 6. Verification

`scripts/verify_repo.py` is the repo's four-check oracle. Run before commits / PRs:

- `python scripts/verify_repo.py --check all --fast` — structure, docs, comments, env-limited execution. Fast (<30s).
- `python scripts/verify_repo.py --check all` — adds the full Tier-A/B/C papermill smoke. Requires the Atlas JupyterHub runtime or an equivalent fully-provisioned environment.

Exit code 0 iff zero error-severity findings; warnings are informational. Tier-C **code-cell source** equality with the `pre-cleanup-baseline` git tag is enforced by check E5 (markdown / outputs are not compared). Edits to phase3 markdown cells should still use `scripts/edit_notebook_markdown.py` for safety.

After the last package or data install, run `python -m pip check`, `make verify-torch-stack`, and
`make verify-nnx-install`; never mutate the environment between those gates and the workload. Keep
pytest at -W error. Temporary debt assertions are limited to the verifier-local exact TorchScript
import group and the quantization test's exact NNx 0.2.0 8da4w model.train warning; neither permits
a global, CLI, environment, pytest, or conftest filter. A warning-free fresh probe or QAT call stops
qualification for debt retirement. Roll back manifests, installer, verifier, CI/Docker, advisory
policy/ledger, and documentation atomically in a fresh environment or rebuilt image.

### 6.1. Helper scripts

- `scripts/verify_repo.py` — the four-check oracle described above.
- `scripts/edit_notebook_markdown.py` — Tier-C-safe markdown-cell editor (changes a single markdown cell's source in-place).
- `scripts/inject_smoke_test_cell.py` — adds a papermill `parameters`-tagged cell (`SMOKE_TEST = 0`) to a notebook. Use when promoting a notebook to Tier-B / Tier-C so `make smoke-tier-b/c` can truncate via `-p SMOKE_TEST 1`.
- `scripts/rewrite_imports.py` — applies the `common/* → nnx/*` module-path rewrite plus the per-net-Params consolidation (`{FeedFwdNN, GraphAtt, GraphConv, GraphSage}Params → NNParams`). Idempotent; safe to re-run.

## 7. One concern per PR

- Don't bundle unrelated cleanup with a feature change.
- Tier-C notebook re-execution belongs in its own PR if you ever need to (rare; preserved outputs are intentional).
