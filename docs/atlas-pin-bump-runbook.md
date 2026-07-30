# Atlas pin-bump and service-admission runbook

This runbook governs the `infra/` Atlas submodule and the consumer boundary around it. Atlas is
the successor to the old infrastructure seam; this repository consumes it rather than maintaining
a fork. The current reviewed pin is `61c7c5103660e2226bf107c115dae42bf46f8374`.

## 1. Ownership and invariants

- Keep Atlas implementation changes upstream. This repository owns only `atlas.consumer.yml`,
  `atlas.env.user.example`, `compose/ml-eng-lab-atlas.yml`, lifecycle wrappers, task contracts,
  and documentation.
- Preserve `BASE_PORT=auto`; notebooks and documentation must not hard-code a published Atlas
  port.
- Keep `LLM_PROVIDER_SOURCE=ollama-localhost`. Run host-native Ollama; never start or add a
  containerized Ollama service for this consumer.
- Leave ComfyUI disabled unless a concrete task completes the admission path in §3. If admitted,
  it must use a reviewed host-native source, never an automatic or containerized source.
- The primary notebook interaction remains local VS Code connected to the running remote JupyterHub
  kernel for remote-workspace tasks. A `mounted-workspace` / `mounted-required` task (currently
  NumPy MNIST) must use Browser JupyterLab or VS Code attached to the JupyterHub container from
  the mounted checkout.

## 2. Atlas pin bump

### 2.1. Prepare the working tree

1. Start from an up-to-date feature branch based on `develop`; do not bump infrastructure directly
   on `main`.
2. Initialize the current pin and verify both the parent and child are clean:

   ```bash
   git submodule update --init --recursive infra
   git status --short
   git -C infra status --short
   ```

3. Read the candidate Atlas release/commit and its migration notes before selecting a SHA. Confirm
   that `start.sh`, `stop.sh`, consumer manifests, `ml-eng`, and the native localhost Ollama source
   remain available.

### 2.2. Advance the gitlink deliberately

1. Fetch and detach `infra/` at the exact reviewed commit; never make a consumer-local commit in
   the submodule.
2. Update `infra` as a gitlink in the parent repository and update the exact SHA in
   [dependency-contracts.md](dependency-contracts.md).
3. Re-run the non-live gates before Docker work:

   ```bash
   make atlas-setup
   make atlas-contract
   make verify
   make test
   ```

4. Review the materialized consumer configuration. The expected control plane is `ml-eng`, a
   JupyterHub container, `BASE_PORT=auto`, and native Ollama. A pin bump must not silently make a
   previously disabled service containerized.

### 2.3. Validate the live runtime

1. Verify the host-native Ollama daemon answers on loopback. Start it in a separate terminal only
   if needed, then return to the lifecycle terminal:

   ```bash
   ollama serve
   make atlas-up
   ```

2. Confirm the running project has neither an Ollama nor a ComfyUI container.
3. Run the container-side runtime probe and a mounted-checkout import smoke:

   ```bash
   docker exec <project>-jupyterhub sh -lc \
     'cd /home/jovyan/work/ml-eng-lab && python scripts/atlas_runtime_probe.py'
   ```

4. Run `make atlas-connect` from an interactive terminal and use its short-lived URL to execute a
   cheap cell from local VS Code against the remote kernel. Do not expose the URL in a command log.
5. Record observed package changes, supported notebook imports, and any manual-only exceptions in
   [dependency-contracts.md](dependency-contracts.md). Successful imports do not replace a
   notebook smoke.
6. Stop normally with `make atlas-down`; reserve `COLD=1 make atlas-down` for an intentional,
   destructive volume reset. Verify `git -C infra status --short` remains clean.

### 2.4. Integrate with Gitflow

Open a PR from the feature branch to `develop` with the gitlink, contract, test, and documentation
changes together. After it merges and checks pass, open a separate `develop` to `main` PR. Do not
merge a pin bump straight to `main`; keep the two review boundaries visible.

## 3. Future service admission

Do this before enabling any additional Atlas service for a notebook:

1. **Specify the need.** Add the service to the relevant task's `docs/spec.yaml` contract,
   including workspace access, artifact policy, required environment values, and why JupyterHub
   alone is insufficient.
2. **Select the narrowest source.** Prefer disabled until needed, then a scoped in-track or
   explicitly declared source. For AI engines, prefer a host-native source when the workstation
   owns the model runtime. Do not inherit an ambient terminal value.
3. **Wire it in-network.** Place configuration in the consumer manifest or parent overlay, not in
   notebook literals or a patched `infra/` file. Use service names and injected environment values,
   never a hard-coded host port.
4. **Prove the contract.** Add unit/contract coverage and run a targeted Atlas JupyterHub smoke
   that reaches the service with the intended notebook dependency. Include a negative check that
   undesired container sources are absent.
5. **Document and review.** Update `docs/notebook-infrastructure.md`, the task documentation,
   `docs/dependency-contracts.md`, diagrams if topology changed, and this runbook. The service
   change then follows the normal feature → `develop` → `main` PR sequence.

The `ml-eng` track may expose other default services, but availability is not authorization for a
notebook dependency. Each service must earn an explicit task contract and targeted evidence.

## 4. Failure handling

- **A pin breaks consumer validation:** return the `infra` gitlink to the last reviewed SHA in a
  new commit; do not force-reset shared branches or edit the submodule history.
- **A native Ollama probe fails:** repair the host daemon and retry. Do not work around it by
  containerizing Ollama.
- **The package probe drifts:** distinguish a version change from an import failure, and decide
  whether local/CI manifests must stay independent. Update the ledger with observed evidence.
- **A change dirties `infra/`:** stop and inspect it. Do not commit child changes from the parent
  repository; discard only generated/ignored child artifacts according to Atlas's own workflow.
