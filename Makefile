# Notebook re-execution targets, organized by execution-cost tier.
#
# Tier A: cheap (<5 min), re-executed in place only when deliberately refreshing
# committed snapshots; CI writes generated copies to a temporary output tree.
# Tier B: moderate, smoke-runs to /tmp (preserves original outputs).
# Tier C: expensive, smoke-runs via SMOKE_TEST parameter to /tmp.
#
# Tier A's temporary-output smoke target runs on every PR. B/C smoke targets can
# run locally or via workflow_dispatch; CI also runs both on the weekly schedule
# and Tier B on PRs labeled `tier-b-smoke`.
#
# All targets assume the selected Python can run papermill and the notebooks'
# kernel can import nnx. nnx is consumed from PyPI via the `thekaveh-nnx[lm]==0.2.0`
# pin in requirements.txt (as of 2026-06-14). The `[lm]` extra pulls
# tokenizers+datasets for the two notebooks that call train_bpe /
# NNTokenizerParams (text_generation-tinyshakespeare-... and
# preference_alignment-toy-dpo-...) — issue #12. Without it those
# notebooks ImportError at the first tokenizer call.

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
PAPERMILL ?= $(PYTHON) -m papermill
PAPERMILL_START_TIMEOUT ?= 300
PAPERMILL_EXECUTION_TIMEOUT ?= 3600
PAPERMILL_TIMEOUT_FLAGS = --start-timeout $(PAPERMILL_START_TIMEOUT) --execution-timeout $(PAPERMILL_EXECUTION_TIMEOUT)

TIER_A := \
    notebooks/image_classification-mnist-ffnn-numpy/notebook.ipynb \
    notebooks/tabular_classification-iris-mlp-pytorch/notebook.ipynb \
    notebooks/model_surgery-mnist-ffnn-pytorch/notebook.ipynb \
    notebooks/pruning-mnist-ffnn-pytorch/notebook.ipynb \
    notebooks/knowledge_distillation-mnist-ffnn-pytorch/notebook.ipynb \
    notebooks/text_generation-tinyshakespeare-transformer-pytorch/notebook.ipynb \
    notebooks/peft-mnist-to-fmnist-dora-vs-lora-pytorch/notebook.ipynb \
    notebooks/dim_reduction-iris-autoencoder-pytorch/notebook.ipynb \
    notebooks/tabular_regression-diabetes-mlp-pytorch/notebook.ipynb \
    notebooks/diffusion-mnist-ddpm-pytorch/notebook.ipynb \
    notebooks/moe-fmnist-mixture-of-experts-pytorch/notebook.ipynb \
    notebooks/clustering-iris-kmeans-vs-ae-pytorch/notebook.ipynb \
    notebooks/link_prediction-karate-graphsage-pyg/notebook.ipynb \
    notebooks/community_detection-karate-louvain-vs-gnn-pyg/notebook.ipynb \
    notebooks/text_classification-agnews-spacy-mlp-pytorch/notebook.ipynb \
    notebooks/sentiment_classification-vader-mlp-pytorch/notebook.ipynb \
    notebooks/preference_alignment-toy-dpo-pytorch/notebook.ipynb \
    notebooks/self_supervised-fmnist-jepa-pytorch/notebook.ipynb

TIER_B := \
    notebooks/image_classification-mnist-ffnn-pytorch/notebook.ipynb \
    notebooks/node_classification-reddit-gnn-pyg/phase1-dataset-exploration-notebook.ipynb \
    notebooks/node_classification-reddit-gnn-pyg/phase2-model-selection-notebook1.ipynb \
    notebooks/node_classification-reddit-gnn-pyg/phase2-model-selection-notebook2.ipynb \
    notebooks/node_classification-reddit-gnn-pyg/phase2-model-selection-notebook3.ipynb \
    notebooks/node_classification-reddit-gnn-pyg/phase2-model-selection-notebook4.ipynb

# notebooks/quantization-mnist-ffnn-pytorch/notebook.ipynb was previously the 2nd entry
# above. Removed 2026-06-16 after the weekly smoke-tier-b cron failed at the
# quantization import: `torchao>=0.17` (requirements.txt pin, smallest version
# exposing nnx.quantize_int8's `Int8WeightOnlyConfig` API) references
# `torch.int1` at module load; `torch.int1` was added in torch 2.5; the local/CI
# contract pins `torch==2.4.1` (see torch-core-requirements.txt + issue #10).
# No torchao version satisfies both nnx's API requirement AND
# the torch 2.4.1 import surface, so the notebook cannot execute under
# CI's pinned environment. Notebook stays in the repo as a manual-only task
# (run locally under a `torch>=2.5` env). The Tier-B move (PR #11) was made
# under the assumption the weekly cron would still exercise it — that turned
# out to be wrong; removing it here unblocks the remaining Tier-B notebooks
# the cron was supposed to cover.

TIER_C := \
    notebooks/node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook.ipynb \
    notebooks/node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook2.ipynb \
    notebooks/node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook3.ipynb \
    notebooks/node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook4.ipynb

ATLAS_CONSUMER_TESTS := tests/test_atlas_consumer_contract.py \
	tests/test_atlas_lifecycle.py \
	tests/test_atlas_runtime_probe.py \
	tests/test_atlas_makefile_contract.py

SMOKE_OUT := /tmp/ml-smoke
TIER_A_OUT ?= /tmp/ml-tier-a
TIER_A_OUT_ABS := $(abspath $(TIER_A_OUT))

.PHONY: help run-tier-a smoke-tier-a check-tier-a-artifacts check-tier-a-clean smoke-tier-b smoke-tier-c test verify-nnx-install test-nnx-surface test-atlas-consumer audit-advisories lint docs-build docs-serve docs-check docs-wiki docs-sync-notebook-infrastructure nlp-assets verify install-torch-stack codespace-setup atlas-setup atlas-up atlas-down atlas-connect atlas-contract

help:
	@echo "Targets:"
	@echo "  run-tier-a        Re-execute Tier-A notebooks in place to deliberately refresh snapshots."
	@echo "  smoke-tier-a      Execute Tier-A notebooks into $(TIER_A_OUT_ABS)/ without changing source notebooks."
	@echo "  check-tier-a-artifacts Fail if a Tier-A temporary notebook output is missing or empty."
	@echo "  check-tier-a-clean Fail if Tier-A execution changed tracked source notebooks."
	@echo "  smoke-tier-b      Papermill Tier-B notebooks with SMOKE_TEST=1 to $(SMOKE_OUT)/ (preserves source outputs)."
	@echo "  smoke-tier-c      Papermill Tier-C notebooks with SMOKE_TEST=1 to $(SMOKE_OUT)/."
	@echo "  test              Run pytest on tests/ directory."
	@echo "  verify-nnx-install Verify the active NNx installation provenance."
	@echo "  test-nnx-surface  Run only tests/nnx_surface (matches the CI pytest-nnx-surface job)."
	@echo "  test-atlas-consumer Run the focused Atlas consumer contract tests."
	@echo "  audit-advisories Run the accepted-advisory policy audit."
	@echo "  lint              Run ruff check . using the [tool.ruff] config in pyproject.toml."
	@echo "  docs-build        Build the MkDocs site: render diagrams, generate the site input, then mkdocs build --strict."
	@echo "  docs-serve        Render diagrams + generate the site input, then mkdocs serve for live preview."
	@echo "  docs-check        Render diagrams, run the docs gate (check_docs), then mkdocs build --strict."
	@echo "  docs-sync-notebook-infrastructure Render the canonical Atlas task-contract table explicitly."
	@echo "  docs-wiki         Generate the wiki Markdown (build_docs --wiki), then push_wiki --check (dry-run)."
	@echo "  nlp-assets        Download spaCy en_core_web_sm + NLTK vader_lexicon (needed by the 2 NLP Tier-A notebooks)."
	@echo "  verify            Run repo verifier (scripts/verify_repo.py --check all --fast)."
	@echo "  install-torch-stack Install pinned Torch core first, then PyG/runtime deps."
	@echo "  codespace-setup   Full dep install + NLP assets. Invoked by .devcontainer/devcontainer.json's postCreateCommand."
	@echo "  atlas-setup       Initialize Atlas and prepare machine-local environment files."
	@echo "  atlas-up          Prepare, validate, and start the Atlas ml-eng track."
	@echo "  atlas-down        Stop Atlas while preserving volumes (set COLD=1 to destroy them)."
	@echo "  atlas-connect     Print interactive VS Code remote-Jupyter connection steps."
	@echo "  atlas-contract    Run Atlas preparation and non-live contract validation."

atlas-setup:
	git submodule update --init --recursive infra
	./scripts/atlas-up.sh --prepare

atlas-up:
	./scripts/atlas-up.sh

atlas-down:
	./scripts/atlas-down.sh $(if $(COLD),--cold,)

atlas-connect:
	./scripts/atlas-connect.sh

atlas-contract:
	./scripts/atlas-up.sh --validate

run-tier-a:
	@for nb in $(TIER_A); do \
		echo "==> $$nb"; \
		dir=$$(dirname "$$nb"); base=$$(basename "$$nb"); \
		(cd "$$dir" && $(PAPERMILL) $(PAPERMILL_TIMEOUT_FLAGS) --kernel python3 "$$base" "$$base") || exit 1; \
	done

smoke-tier-a:
	@for nb in $(TIER_A); do \
		out="$(TIER_A_OUT_ABS)/$$nb"; \
		echo "==> $$nb -> $$out"; \
		dir=$$(dirname "$$nb"); base=$$(basename "$$nb"); \
		mkdir -p "$$(dirname "$$out")"; \
		(cd "$$dir" && $(PAPERMILL) $(PAPERMILL_TIMEOUT_FLAGS) --kernel python3 "$$base" "$$out") || exit 1; \
	done

check-tier-a-artifacts:
	@for nb in $(TIER_A); do \
		out="$(TIER_A_OUT_ABS)/$$nb"; \
		if [ ! -s "$$out" ]; then \
			printf 'missing expected Tier-A notebook output: %s\n' "$$out" >&2; \
			exit 1; \
		fi; \
	done

check-tier-a-clean:
	git diff --exit-code -- $(TIER_A)

smoke-tier-b:
	@mkdir -p $(SMOKE_OUT)
	@for nb in $(TIER_B); do \
		out=$(SMOKE_OUT)/$$(basename "$$nb"); \
		echo "==> $$nb -> $$out"; \
		dir=$$(dirname "$$nb"); base=$$(basename "$$nb"); \
		(cd "$$dir" && $(PAPERMILL) $(PAPERMILL_TIMEOUT_FLAGS) --kernel python3 -p SMOKE_TEST 1 "$$base" "$$out") || exit 1; \
	done

smoke-tier-c:
	@mkdir -p $(SMOKE_OUT)
	@for nb in $(TIER_C); do \
		out=$(SMOKE_OUT)/$$(basename "$$nb"); \
		echo "==> $$nb -> $$out"; \
		dir=$$(dirname "$$nb"); base=$$(basename "$$nb"); \
		(cd "$$dir" && $(PAPERMILL) $(PAPERMILL_TIMEOUT_FLAGS) --kernel python3 -p SMOKE_TEST 1 "$$base" "$$out") || exit 1; \
	done

test:
	pytest tests/ -v

verify-nnx-install:
	$(PYTHON) -m scripts.verify_nnx_install

test-nnx-surface:
	pytest tests/nnx_surface -v

test-atlas-consumer:
	pytest $(ATLAS_CONSUMER_TESTS) -v

audit-advisories:
	$(PYTHON) -m scripts.advisory_baseline

lint:
	ruff check .

docs-build:
	$(PYTHON) -m scripts.docs.render_diagrams
	$(PYTHON) -m scripts.docs.build_docs --site
	NO_MKDOCS_2_WARNING=1 mkdocs build --strict

docs-serve:
	$(PYTHON) -m scripts.docs.render_diagrams
	$(PYTHON) -m scripts.docs.build_docs --site
	NO_MKDOCS_2_WARNING=1 mkdocs serve

docs-check:
	$(PYTHON) -m scripts.docs.render_diagrams
	$(PYTHON) -m scripts.docs.check_docs
	NO_MKDOCS_2_WARNING=1 mkdocs build --strict

docs-sync-notebook-infrastructure:
	$(PYTHON) -m scripts.docs.notebook_infrastructure --write

docs-wiki:
	$(PYTHON) -m scripts.docs.build_docs --wiki
	$(PYTHON) -m scripts.docs.push_wiki --check

nlp-assets:
	$(PYTHON) -m spacy download en_core_web_sm
	$(PYTHON) -c "import nltk; nltk.download('vader_lexicon', quiet=True)"

verify:
	$(PYTHON) scripts/verify_repo.py --check all --fast

install-torch-stack:
	$(PIP) install --upgrade pip
	$(PIP) install -r torch-core-requirements.txt
	$(PIP) install --no-build-isolation -r torch-requirements.txt

# Full one-shot dep install for the GitHub Codespaces / "Reopen in Container"
# path (README §3.4). Reuses the same Torch-first install order as CI and
# Docker so PyG source-build fallback can import torch during extension builds.
# Recursively invokes nlp-assets so the spaCy + NLTK download steps stay in
# one place across the §3.2 (Docker), §3.3 (venv), and §3.4 (Codespaces) paths.
codespace-setup: install-torch-stack
	$(PIP) install -r requirements.txt
	$(MAKE) nlp-assets
