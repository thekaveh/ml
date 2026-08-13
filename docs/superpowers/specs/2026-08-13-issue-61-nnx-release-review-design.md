# 12.19 Issue 61 NNx Release Review Design

## 12.19.1 Purpose

The repository consumes `thekaveh-nnx[lm]==0.2.0` as its shared PyTorch toolkit. Stable
releases 0.2.1 and 0.2.2 have since shipped changes relevant to this repository's training
lifecycle, persistence, QAT checkpoints, pandas integration, graph loading, Conv and MoE
networks, DPO metrics, generation, and tabular regression.

Issue #61 decides whether the repository can adopt the latest stable release without combining
the broader Torch, PyG, Lightning, and torchao upgrade owned by Issue #62. This is a
compatibility decision, not an automatic version bump. The repository adopts 0.2.2 only if the
published wheel satisfies the existing canonical-installation contract and every affected
consumer surface passes. Otherwise it retains 0.2.0 and records the precise incompatibility.

## 12.19.2 Upstream release evidence

The complete stable release sequence after the current pin is:

| Version | Tag commit | Published wheel SHA-256 | Status |
| --- | --- | --- | --- |
| 0.2.1 | `074ef22944a39eca8d65052ffa8d01520c5f1f1c` | `d0296143a714f9bb310b508406c6e3b3a21fa0a03d867345bef346e9fb556f20` | Stable, universal, not yanked |
| 0.2.2 | `edfd197f3f54d4eb67313d46a80e823e6239c5b6` | `ee56474926fdfd5329721f067cf1b8ae31955627c6949844e09ee4a7bb2bb9d7` | Latest stable, universal, not yanked, immutable GitHub release |

Both releases retain Python `>=3.10`, Torch `>=2.0`, torchvision `>=0.15`, and
torch-geometric `>=2.4`. The `[lm]` extra remains available. Version 0.2.2 adds
`filelock>=3.13` to the runtime requirements and moves IPython out of the core set into a
notebook extra. These floors fit the repository's existing Python 3.11 and Torch/PyG pins, but
metadata compatibility alone is not acceptance evidence.

Primary evidence is the upstream [0.2.1 release](https://github.com/thekaveh/NNx/releases/tag/v0.2.1),
[0.2.2 release](https://github.com/thekaveh/NNx/releases/tag/v0.2.2), and
[PyPI project metadata](https://pypi.org/project/thekaveh-nnx/). The implementation must refresh
these facts before changing the pin rather than treating this dated design record as a permanent
latest-version oracle.

The upstream project changed its license from MIT to Apache-2.0. The repository records that
license transition in the release review; it does not copy, modify, or redistribute NNx source.

## 12.19.3 Release delta and consumer risk

### 12.19.3.1 Version 0.2.1

Version 0.2.1 is additive at the public facade: it introduces `ConvNN`, `NNConvParams`,
`FeedFwdMoENN`, `NNMoEParams`, `Nets.CONV`, `Nets.FEED_FWD_MOE`, `EvalStepContext`,
`EvalStepFn`, and `NNCheckpointTransform` without removing the 0.2.0 exports.

Its compatibility-sensitive changes are behavioral:

- callback cleanup becomes exception-safe and runs in reverse order;
- callback mutations made during `on_train_end` are persisted to the LAST checkpoint;
- converted QAT topology is recorded and reconstructed from checkpoint metadata;
- `NNModel.train` gains an optional whole-validation-loader `eval_step_fn`;
- graph loading gains full-batch operation, an explicit sampler, and a seed;
- pandas 2 type contracts are hardened; and
- invalid Conv and MoE configurations fail earlier.

### 12.19.3.2 Version 0.2.2

Version 0.2.2 retains the 0.2.1 facade and adds optional behavior without removing existing
parameters:

- `NNTabularDataset(target_dtype=<floating dtype>)` supports regression targets shaped
  `(batch, 1)` while the default `None` preserves classification;
- `NNRun` and `NNModel.train` accept an optional identity salt whose default preserves existing
  identities;
- DPO evaluation data includes implicit reward metrics;
- generation accepts an optional per-token callback;
- persistence becomes transactional, with LAST serving as the epoch commit marker;
- warm resume restores and validates optimizer, scheduler, scaler, loader, sampler, and RNG
  state;
- final partial gradient-accumulation windows perform their pending optimizer step; and
- evaluation and classification metrics use corrected normalization and target handling.

NNx is pre-1.0 and explicitly permits behavior-changing bug fixes. Source compatibility is
therefore necessary but insufficient: training metrics, checkpoint contents, run artifacts, and
recorded notebook output may change even when imports and function signatures do not.

## 12.19.4 Selected approach

Use the feature branch itself as the compatibility trial:

1. update the single canonical requirement to `thekaveh-nnx[lm]==0.2.2`;
2. install that version from the package index with
   `--only-binary=thekaveh-nnx` in every job that provides NNx execution evidence;
3. run the existing fail-closed canonical-wheel verifier immediately before each test or
   notebook workload;
4. add only consumer-facing tests needed to exercise newly relevant 0.2.2 behavior;
5. run the complete repository and affected notebook tiers; and
6. retain the bump only if all mandatory gates pass.

This preserves `requirements.txt` as the sole production version source. The verifier continues
to derive the expected version and `[lm]` extra from that file; there is no independent version
constant, candidate manifest, compatibility mode, or editable bypass in CI.

The published SHA-256 is recorded as reviewed release evidence, not added as a second install
contract. Comprehensive hash-locked dependency inputs remain owned by Issue #63. The canonical
wheel verifier continues to prove one matching distribution, exact version, absence of direct
URL metadata, WHEEL and RECORD ownership, exact `nnx/__init__.py` import origin, and an origin
outside this repository.

## 12.19.5 Alternatives considered

### 12.19.5.1 Retain 0.2.0 without a compatibility trial

This minimizes immediate change but leaves known lifecycle, QAT persistence, pandas, and
regression improvements unreviewed. It does not satisfy the issue's requirement to study and
test the latest compatible release.

### 12.19.5.2 Add a candidate manifest or permanent candidate CI job

A second requirements file could test 0.2.2 without changing the canonical pin. It is rejected
because it duplicates the version source established by Issue #58 and creates ambiguity about
which wheel the repository, notebook tiers, cache keys, and verifier actually certify.

### 12.19.5.3 Upgrade NNx together with Torch and torchao

This would allow the converted-QAT path to run in the main environment, but it would combine two
independent dependency decisions and make failures difficult to attribute. Issue #62 owns the
coordinated Torch/PyG/Lightning/torchao upgrade; Issue #66 owns restored quantization execution
and tiering.

### 12.19.5.4 Add downstream compatibility shims

No shim is permitted. A consumer failure means 0.2.2 is not accepted for this issue. The
repository either corrects a stale local assumption that is already invalid or retains 0.2.0
with evidence; it does not emulate old NNx behavior downstream.

## 12.19.6 Consumer changes

The pin bump itself is intentionally small. Release-coupled references in the root requirements,
Makefile commentary, README, contributor guide, NNx overview, dependency contracts, relevant
notebook documentation, and the NNx badge must describe 0.2.2. Historical Issue #58 records,
historical changelog entries, and observed Atlas image evidence remain unchanged.

The active tabular-regression notebook keeps its manual `TensorDataset` and `DataLoader`
pipeline. That pipeline guarantees the same shared train, validation, and test split used by the
sklearn baselines. Replacing it with an NNx-owned split would alter the experiment and recorded
results. Documentation instead marks the former classification-only limitation as resolved in
0.2.2 and describes `target_dtype=torch.float32` for future callers.

The following adjacent improvements remain out of scope:

- migrating three notebooks from manual loader reconstruction to `NNDataset(batch_sizes=...)`
  remains Issue #69;
- adding `NNGraphDataset(seed=...)` to historical Reddit notebooks remains Issue #70;
- replacing deep NNx imports with the public facade remains Issue #68;
- replacing the existing MoE model with `FeedFwdMoENN` would change architecture and results;
  and
- making ConvNN a new notebook architecture is separate roadmap work.

`docs/FINDINGS-NNX.md` records the regression-target finding as resolved by 0.2.2. The whole-split
`NNDataset` default, ReLU-only `deepen`, regression early-stopping default, and absolute run-save
path findings remain open. The unrelated `NNRun.load("best")` documentation contradiction is
corrected to the existing `NNCheckpoint.load(run=<id>, type=Checkpoints.BEST)` contract.

## 12.19.7 Consumer-facing test additions

The existing NNx surface suite remains the primary API contract. Focused additions cover gaps
made material by this release:

- `NNTabularDataset` with a floating target dtype produces floating `(batch, 1)` targets,
  `output_dim == 1`, and supports a tiny MSE training pass;
- default `target_dtype=None` retains the existing contiguous integer classification behavior;
- the released facade exposes and can construct valid Conv and first-class MoE configurations;
- graph dataset signatures admit the released sampler and seed fields without changing locked
  historical Tier-C code; and
- QAT facade signatures remain exact.

The repository does not duplicate upstream lifecycle, persistence, interruption, and callback
unit suites. Those behaviors are exercised here through consumer training, notebook execution,
ordinary checkpoint use, and the optional QAT compatibility probe. Any consumer-visible failure
rejects the bump.

Synthetic 0.2.0 metadata in verifier unit tests remains where it is merely arbitrary fixture
data. Only tests that assert the repository's current production pin change to 0.2.2. This keeps
wrong-version and mismatch cases meaningful.

## 12.19.8 Notebook execution matrix

All 27 tiered NNx notebooks are affected by shared lifecycle, params, metrics, persistence, or
dataset behavior. The mandatory validation matrix is therefore:

| Surface | Required evidence | Rationale |
| --- | --- | --- |
| Tier A | Complete `make smoke-tier-a`, artifact, and clean-source checks | Covers 17 NNx consumers plus the NumPy control across custom and standard train steps |
| Tier B | Complete `make smoke-tier-b` | Covers the image baseline and every active Reddit exploration/model-selection notebook |
| Tier C | Complete `make smoke-tier-c` | Covers all four historical Reddit final pipelines without overwriting source outputs |
| Quantization | Existing static/backend-gated tests plus a best-effort isolated Torch >=2.5 QAT checkpoint round trip | Main Torch 2.4.1 cannot import the required torchao surface |

Tier B must run on the feature pull request through its label-controlled job. Tier C runs through
manual workflow dispatch on the feature ref. Both use temporary outputs; no Tier-C code cell or
historical committed result is edited.

The isolated QAT probe installs no repository-wide dependency change and cannot substitute for
Issue #62 or #66. If a suitable disposable environment cannot run it, the limitation is recorded
as a concern rather than weakening existing CI. Canonical Tier A/B/C and NNx surface failures are
blocking.

## 12.19.9 CI contract

No new required job is introduced. The protected-branch contexts remain:

- `pytest-repository`;
- `atlas-consumer-policy`; and
- `dependency-audit`.

`pytest-repository` remains the required complete test signal and `pytest-nnx-surface` remains the
focused diagnostic signal. Both already use binary-only selection and canonical verification.

The Tier A, Tier B, and Tier C jobs are strengthened to:

1. install all dependencies using the existing Torch-first order;
2. select NNx with `python -m pip install --only-binary=thekaveh-nnx -r requirements.txt`;
3. perform no later package installation;
4. run `make verify-nnx-install`; and
5. immediately execute the tier workload.

Workflow-contract tests enforce binary-only selection, verifier placement, absence of editable or
path overrides, no failure masking, and no install between verification and workload. Existing
cache keys already include `requirements.txt`, so the pin change invalidates each relevant cache.

## 12.19.10 Vulnerability and dependency evidence

Changing `requirements.txt` invalidates the Issue #59 audit input hash and may change the resolved
combined-runtime graph through NNx's new `filelock` dependency. The implementation reruns the
four-surface advisory audit and refreshes the current human ledger evidence.

The machine-readable accepted-advisory policy is not automatically expanded. A new advisory
identity or accepted-version drift blocks the bump and requires explicit security review. An
empty PyPI project-level vulnerability field is not sufficient evidence.

The Atlas submodule's JupyterHub image separately pins NNx 0.2.0. That is Atlas-owned runtime
state, not a second root requirement. This issue does not edit the submodule, rebuild the image,
or rewrite historical Atlas probe results to claim 0.2.2.

## 12.19.11 Documentation surfaces

Canonical documentation updates cover:

- current pin and release rationale in the repository and contributor entry points;
- the 0.2.1 and 0.2.2 delta and validation evidence in the NNx and dependency pages;
- the resolved tabular-regression limitation in the finding ledger, task README, task spec, and
  canonical notebook page;
- exact Tier A/B/C and quantization evidence boundaries;
- the audit snapshot refresh and immutable historical boundaries; and
- durable release history in `CHANGELOG.md`.

This design and its implementation plan become manifest entries 12.19 and 12.20. Generated site
and wiki trees remain ignored derivatives and are never edited directly.

## 12.19.12 Failure handling and rollback

The bump is rejected if any of the following occurs:

- the latest artifact is yanked, a prerelease, missing, or not the reviewed universal wheel;
- binary-only installation or canonical provenance fails;
- an existing consumer import, signature, constructor, training, checkpoint, or notebook contract
  fails;
- the default run identity changes for an existing representative configuration;
- a Tier A, Tier B, or Tier C smoke workload fails its existing acceptance boundary;
- the vulnerability audit reports an unreviewed advisory or accepted-version drift; or
- documentation and generated surfaces cannot remain internally consistent.

Rollback restores the single pin to 0.2.0, reverts only release-coupled current documentation and
tests, reinstalls the binary wheel, and reruns the complete contract. Historical evidence and
existing 0.2.0 fixtures remain available. No provenance check, audit rule, skip guard, or notebook
threshold is weakened to make either version pass.

## 12.19.13 GitFlow acceptance

The feature branch merges into `develop` through a ready pull request after local verification,
independent task reviews, a broad final review, green required contexts, green focused NNx and
Tier A jobs, a labeled green Tier B run, and a manually dispatched green Tier C run.

`develop` then merges into `main` through a separate pull request. Main is synchronized back into
develop when necessary. Final bookkeeping records both pull requests, main and develop merge
commits, test evidence, the chosen adopt-or-retain decision, ruleset readback, issue and project
status, branch/worktree cleanup, and the absence of repository Atlas/Ollama/ComfyUI containers.

No Atlas JupyterHub service is started. Its current image cannot prove 0.2.2 compatibility without
an Atlas-owned dependency change, so clean local and GitHub-hosted released-wheel execution is the
authoritative evidence.

## 12.19.14 Success criteria

Issue #61 succeeds when the repository has an evidence-backed decision on 0.2.2, the selected
version is represented by one exact production pin, the imported package is proven to be the
canonical released wheel, every affected consumer tier passes without downstream shims or broader
Torch changes, stale release-coupled documentation is reconciled across all three surfaces, the
advisory policy remains fail-closed, and the result completes the full GitFlow path with clean
local and remote state.
