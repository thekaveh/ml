# 12.27 Issue 65 Atlas Pin Review Design

## 12.27.1 Purpose and observed state

Issue #65 requires an evidence-based retain-or-bump decision for the Atlas
submodule consumed at `infra/`. The original issue text names
`61c7c5103660e2226bf107c115dae42bf46f8374`, but Issue #64 necessarily advanced
the gitlink while landing the shared NLP asset contract. The repository now
consumes:

```text
41ba856f7cd35f0b559d6875e08443eac3e98a98
```

A fresh fetch on 2026-08-21 proves Atlas `origin/main` is the same commit.
Therefore the current-to-candidate range is empty and no gitlink bump is
available or warranted. The current README, environment guide, and pin-bump
runbook still call the historical `61c7c510...` pin current; those statements
are stale even though the dependency ledger and executable consumer test use
the real `41ba856f...` gitlink.

Issue #65 closes both parts of that mismatch:

1. review the already-consumed historical Atlas range from `61c7c510...` to
   `41ba856f...`; and
2. retain the exact current Atlas `main` commit while synchronizing the current
   consumer contract, validation, and documentation.

Historical specifications and released changelog entries continue to describe
the pin that was true for their release boundary. They are not rewritten.

## 12.27.2 Decision and rejected alternatives

The selected decision is **retain `41ba856f...`**.

Two alternatives are rejected:

1. **Roll back to `61c7c510...`, then replay the bump.** This would discard the
   already-qualified Issue #64 NLP asset projections, invalidate the current
   consumer contract, and create churn without changing the reviewed end
   state.
2. **Pin an unmerged Atlas commit from another branch.** Atlas `main` is the
   only reviewed moving source for this consumer. Selecting an unpublished
   descendant would bypass Atlas's own integration boundary and make the
   consumer responsible for upstream work in progress.

The gitlink is the authority. Documentation and test constants are derived
projections and must agree with the tree entry; they do not independently
select an Atlas revision.

## 12.27.3 Reviewed commit ranges

The final evidence records two exact ranges:

| Purpose | Base | Tip | Result |
| --- | --- | --- | --- |
| Current retain decision | `41ba856f7cd35f0b559d6875e08443eac3e98a98` | Atlas `origin/main` at the same SHA | empty range |
| Historical migration review | `61c7c5103660e2226bf107c115dae42bf46f8374` | `41ba856f7cd35f0b559d6875e08443eac3e98a98` | 30 first-parent commits, 43 total commits |

The historical range is reviewed by first-parent integration boundary and by
consumer-relevant path diff. The review must not infer compatibility merely
from the absence of a newer Atlas commit.

## 12.27.4 Consumer-facing migration review

The historical range contains these material consumer changes:

### 12.27.4.1 JupyterHub runtime

- Torch moves from 2.11.0/torchvision 0.26.0/torchaudio 2.11.0 to Atlas's
  independently owned Torch 2.13.0/torchvision 0.28.0 runtime. Atlas omits
  Torchaudio because no executable notebook imports it.
- PyG moves from the Torch 2.11 legacy extension set to the Torch 2.13 CPU
  index with `pyg_lib==0.8.0` and `torch_geometric==2.7.0`; Atlas no longer
  installs the unavailable legacy scatter/sparse/cluster wheels for 2.13.
- JupyterHub adds `fastmcp==3.4.4`, the MCP endpoint contract, locked runtime
  constraints, Coursier integrity verification, and current security patch
  floors.
- Issue #64 adds exact spaCy and VADER projections, `NLTK_DATA`, and offline
  verification to the image.
- The JupyterHub service gains MCP, Docling, and Parakeet endpoint/token
  projections. None authorizes those services for an ml-eng-lab notebook.

Atlas's package graph remains intentionally independent from ml-eng-lab's
local/CI Torch 2.11 lock. Compatibility is proved by the runtime probe and
notebook imports, not by requiring both environments to have identical
versions. Atlas still carries `thekaveh-nnx[lm]==0.2.0`, preserving the default
notebook API boundary established by Issue #61.

The four immutable Reddit Phase-3 notebooks retain a historical
`from torch_sparse import SparseTensor` statement whose binding is never
loaded. Their August-2023 code/output contract forbids rewriting or
re-execution. The runtime probe may exclude `torch_sparse` only when those
four exact notebooks are the complete import context and every binding remains
unused. A new import, a fifth context, or any `SparseTensor` load restores the
mandatory fail-closed import. This is a bounded historical-source exception,
not evidence that `torch_sparse` exists in Atlas.

### 12.27.4.2 Track and source synthesis

Atlas replaces duplicated track logic with a central track registry and
linear startup path. Explicit consumer-declared source values survive an
out-of-track default, while undeclared off-track services remain disabled.
The ml-eng-lab contract therefore continues to require:

- `BASE_PORT=auto`;
- JupyterHub on the `ml-eng` track;
- `LLM_PROVIDER_SOURCE=ollama-localhost`;
- no containerized or automatic Ollama source; and
- ComfyUI disabled unless a later task completes service admission.

The parent wrapper clears inherited source variables before every Atlas
command and validates the materialized values, so ambient host configuration
cannot silently override those rules.

### 12.27.4.3 FastMCP and managed host processes

The standalone MCP runtime and JupyterHub client now share FastMCP 3.4.4. The
MCP service is part of Atlas's dependency graph, but it is not listed in any
ml-eng-lab notebook's `required_services`; availability is not authorization.

Atlas also adds a generic managed-host-process lifecycle with explicit argv,
loopback-by-default binding, consumer-root path confinement, PID ownership,
health checks, and volume-independent stop/remove behavior. ml-eng-lab does
not declare a managed host process in this issue. The framework is reviewed as
a compatible optional capability, not enabled as a new dependency.

### 12.27.4.4 ComfyUI and host-native Ollama

ComfyUI gains consumer custom-node projections and managed-MPS provisioning.
Those changes do not change this consumer's policy: ComfyUI remains disabled,
and container/automatic sources remain prohibited. The live check must prove
that no ComfyUI container is started.

Atlas adds host Ollama parallelism and model-residency doctor checks. They are
advisory/read-only for `ollama-localhost`; Atlas still does not own or start the
host daemon. The live check uses the already-running host-native loopback
daemon and proves no Ollama container exists.

## 12.27.5 Repository changes

The implementation changes no Atlas source and does not change the `infra`
gitlink. It updates only consumer-owned current state:

1. README, environment setup, and the pin-bump runbook name `41ba856f...` as
   the current reviewed pin.
2. The dependency ledger records the two ranges above, the retain decision,
   the migration review, Atlas/local version boundaries, and rollback SHA.
3. Unreleased changelog text records Issue #65 completion without rewriting
   historical release entries.
4. Current-document tests and repository verification require each current
   pin projection to equal the actual `infra` tree entry. Mutations that
   restore `61c7c510...`, use a malformed SHA, or make any one current surface
   disagree must fail.

The exact rollback target remains
`61c7c5103660e2226bf107c115dae42bf46f8374`. A rollback is a new reviewed
gitlink commit; shared branches are never reset and Atlas history is never
modified from this repository.

## 12.27.6 Validation contract

### 12.27.6.1 Non-live validation

Before any live Atlas operation, run:

```bash
make atlas-setup
make atlas-contract
make test-atlas-consumer
make verify
make test
make lint
make docs-check
make docs-wiki
```

Run ShellCheck on the parent lifecycle wrappers. Require clean parent and
submodule status, exact gitlink/main equality, current-document parity, and no
consumer-local Atlas source edit.

### 12.27.6.2 Live JupyterHub validation

Although the final decision does not advance the current gitlink, the
historical range materially changed JupyterHub and was consumed before this
dedicated review. The completion evidence therefore includes one live runtime
validation:

1. Prove the host-native Ollama daemon answers on loopback before Atlas starts.
2. Start with `make atlas-up` and the exact consumer manifest.
3. Prove the resolved project contains JupyterHub but no Ollama or ComfyUI
   container.
4. From the running JupyterHub container, execute
   `scripts/atlas_runtime_probe.py --json /tmp/issue65-runtime.json` against
   the mounted checkout and require every mandatory capability to pass. The
   probe deduplicates repeated NLTK search-path entries only when they resolve
   to the same exact VADER archive; distinct duplicates remain failures.
5. Execute one cheap Python/import cell through the Jupyter kernel registered
   inside the running JupyterHub container, without logging its token-bearing
   URL.
6. Stop with ordinary `make atlas-down`, preserving volumes. Prove the project
   is stopped, the host-native Ollama process was not lifecycle-owned, and the
   parent/submodule remain clean.

No containerized Ollama or ComfyUI source may be started as a workaround. No
`COLD=1` shutdown is permitted.

## 12.27.7 Evidence, publication, and completion

The immutable report records:

- feature SHA/tree and exact unchanged gitlink;
- fetched Atlas `main` identity and both reviewed ranges;
- categorized migration findings and the retain decision;
- focused/full/static/docs/ShellCheck results;
- live project/container inventory, runtime probe, cheap-cell result, and
  volume-preserving shutdown proof;
- host-native Ollama pre/post state without secrets or token-bearing URLs; and
- clean parent/submodule state.

After independent review, freeze one feature SHA and publish it through the
normal feature -> `develop` -> `main` -> `develop` GitFlow cycle. Any tracked
change after freeze invalidates the evidence. Preserve Issues #53 and #66 as
OPEN, preserve the protected ruleset, publish the immutable report, remove only
Issue #65 refs/worktrees/images/containers, set Issue #65's project status to
Done, and close Issue #65 with reason `completed` as the final mutation.
