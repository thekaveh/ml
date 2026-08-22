# Issue #72 NNx upstream-triage design

## 1. Objective

Turn the five findings in `docs/FINDINGS-NNX.md` into a durable, non-duplicative
triage record. Each finding must identify its upstream issue or an explicit
evidence-backed disposition, the local workaround and affected notebooks, the
released NNx status, and any remaining ml-eng-lab work.

This issue changes ml-eng-lab documentation and issue metadata only. It does
not modify the NNx repository, change the pinned NNx wheel, or execute
notebooks.

## 2. Evidence boundary

The triage uses these authoritative sources:

- open and closed issues in `thekaveh/NNx`;
- released NNx tags `v0.2.0` through `v0.2.3` and current `main` source;
- the published NNx changelog and release records;
- ml-eng-lab's exact `thekaveh-nnx[lm]==0.2.0` requirement;
- Issue #61's retained-version decision and Issue #69's loader migration;
- active notebook source, notebook READMEs, and the repository verifier.

Searches cover both titles and bodies. A matching closed issue is linked rather
than recreated. Source behavior without an issue is not treated as proof of a
maintainer decision unless the released documentation makes the contract
explicit.

## 3. Considered approaches

### 3.1 One upstream umbrella issue

One issue would minimize external records, but it would mix unrelated dataset,
callback, and persistence contracts. Individual fixes could not close
independently, and duplicate searches would be less precise.

### 3.2 Five new upstream issues

One issue per finding would be mechanically uniform, but it would duplicate the
closed regression-support issue and misrepresent the ReLU-only Net2DeeperNet
constraint as an unresolved bug even though released NNx documentation already
defines and enforces that mathematical boundary.

### 3.3 Evidence-based per-finding dispositions (selected)

Link the existing regression issue, record the released ReLU-only contract as
an explicit non-bug disposition, and open separate upstream issues only for the
three actionable unresolved gaps. This gives every finding a durable result
without manufacturing duplicate or misleading work.

## 4. Per-finding disposition

| Finding | Upstream result | Local result |
| --- | --- | --- |
| `NNDataset` defaults to one full-split batch | Open a focused upstream documentation/API-default issue. The behavior remains in `v0.2.3` and `main`; user-facing API docs expose the signature but do not explain the `None` semantics. | Resolved by Issue #69 for diffusion, MoE, and JEPA through explicit public `batch_sizes=` values. TinyShakespeare remains an intentional custom dataset. No new local follow-up. |
| `nnx.deepen` is ReLU-only | Explicit disposition: accepted, documented Net2DeeperNet constraint. `v0.2.0` and later explain the identity/ReLU equation, reject other activations, and document the error. Do not open a false enhancement suggesting a bias can make arbitrary non-idempotent activations exactly equivalent. | The model-surgery notebook deliberately uses ReLU and documents the contract. No local follow-up. |
| `NNTabularDataset` lacked regression targets | Link closed upstream issue `thekaveh/NNx#81`; NNx `v0.2.2` added `target_dtype` with release and test evidence. | Still unresolved under ml-eng-lab's Atlas-compatible `0.2.0` pin. Keep the manual float-target loader and create one local follow-up for a coordinated Atlas/root NNx upgrade and subsequent loader migration. |
| `EarlyStopping` defaults to `val_edp.error` | Open a focused upstream issue. `v0.2.3` and `main` still use the classification default; an absent monitored field silently disables stopping. The issue should ask for an explicit auto/fallback contract rather than construction-time loss guessing. | The diabetes notebook does not instantiate `EarlyStopping`; its README documents `monitor="val_edp.loss"` for future regression use. No immediate local code remains. |
| training prints an absolute run path | Open one upstream issue covering both `NNModel.train` and `Trainer.train`. All releases through `v0.2.3` and `main` build the message from `os.getcwd()`. | Keep verifier rule `E13.stale_active_notebook_path`; committed notebook output remains normalized. No new local follow-up until an upstream release changes the message. |

## 5. Upstream issue contract

Each new upstream issue must contain:

- a concise current-behavior summary;
- released-source evidence and a minimal reproduction or exact code path;
- downstream impact in ml-eng-lab;
- a backward-compatible proposed direction;
- verifiable acceptance criteria;
- a backlink to ml-eng-lab Issue #72.

The triage does not push branches, commits, or pull requests to NNx. Opening the
issues is the only NNx-side mutation.

## 6. Local follow-up contract

Create exactly one ml-eng-lab follow-up issue because concrete local work
remains for the released `NNTabularDataset.target_dtype` support. Its scope is a
coordinated root/Atlas NNx upgrade followed by migration of the diabetes
notebook away from its manual loader. It must depend on upstream `#81`, preserve
the current `0.2.0` contract until compatibility is requalified, and require the
normal dependency and notebook execution matrix.

Do not create speculative local issues for findings already resolved locally or
for upstream changes that have not shipped.

## 7. Documentation shape

`docs/FINDINGS-NNX.md` remains canonical. Add a compact status table before the
detailed findings and make each detailed section use the same labels:

- **Upstream disposition**
- **Release evidence**
- **Affected notebooks**
- **Local workaround/status**
- **Remaining work**

The document must distinguish `NNRun.save()` from the actual message emitters,
`NNModel.train()` and `Trainer.train()`. It must also distinguish upstream
resolution from availability in the retained ml-eng-lab runtime.

Update `docs/nnx-library.md` only where its summary would otherwise contradict
the canonical triage. Preserve the documentation manifest rather than adding a
new published page.

## 8. Guardrails and tests

Add a focused documentation contract test that fails if:

- any of the five canonical finding anchors disappears;
- the status table lacks one row per finding;
- an actionable row lacks its upstream URL;
- the resolved regression row lacks upstream `#81`, `v0.2.2`, or the local
  follow-up link;
- the ReLU row is incorrectly represented as an open bug;
- the portable-path row attributes the print to `NNRun.save()` rather than the
  two training entry points.

Then run:

- the focused documentation tests;
- the complete documentation-check test module;
- strict documentation build and wiki projection check;
- link validation through the repository documentation tooling;
- `scripts/verify_repo.py --check all --fast`;
- Ruff and the full repository test suite.

No Atlas service or notebook execution is required because no runtime code or
notebook is changed.

## 9. Rollback

The ml-eng-lab documentation/test commit and the local follow-up issue are
independently reversible. Upstream issues are durable coordination records and
must not be deleted during rollback; if superseded, close them with a link to
the replacement decision. No NNx source change exists to roll back.
