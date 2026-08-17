# 6.1 Dependency Contracts

This ledger records consumed dependency contracts that are intentionally pinned,
manual-only, or known to carry security/tooling constraints. It complements
`requirements.txt`, `torch-core-requirements.txt`, `torch-requirements.txt`,
`docs-requirements.in`, its compiled `docs-requirements.txt` lock, and the CI workflow; the manifests remain the source
of truth for installation.

## 6.1.1 Audit Snapshot

### 6.1.1.1 Reproducible four-surface audit

Last reviewed: 2026-08-12. The immutable capture metadata is:

- UTC capture timestamp: `2026-08-13T03:19:27Z`.
- Repository commit: `45105ca0410c7ea3170665d57567accc7be97461`.
- Platform: `Darwin` on `arm64`.
- Interpreter: `Python 3.11.0`.
- Auditor: `pip-audit 2.10.0`.

Manifest SHA-256 values:

| Manifest | SHA-256 |
| --- | --- |
| `requirements.txt` | `3f35f04f95bd1e293c844b41a2dcf96f7978b8c61ccd436e4813a604d9e528a7` |
| `torch-core-requirements.txt` | `2b99702ae89067c09abe10ddf3eb880eb854871feee7f64a8d51aaa4764578e5` |
| `torch-requirements.txt` | `771f07b281ee931f45372904da0472b293d9e64b1d0ec6ba11569a9b5a3925ec` |
| `docs-requirements.txt` | `9af475ff61cafc56f0edd75e28d9ca41463f87f0790523d5e077a1d71323b9cc` |
| `atlas-contract-requirements.txt` | `e786c8e7d940a97ae41ce880d5f5bbc62dc4f90ff03fd8c7718849e1c11412b0` |

The commands below are historical capture evidence for this dated snapshot; current enforcement
uses the selector-free projection described in [the enforcement boundary](#6114-enforcement-boundary).
They were run separately from the repository root. Exit `0` means no known vulnerabilities were
reported; exit `1` means the emitted findings form a complete observation. Any exit other than 0/1, missing output, or malformed JSON invalidates the observation.

```bash
AUDIT_DIR="$(mktemp -d /private/tmp/ml-eng-lab-issue59-audit.XXXXXX)"

python -m pip_audit -r requirements.txt -r torch-requirements.txt \
  --strict --vulnerability-service pypi --format json \
  --aliases on --desc off --progress-spinner off \
  --output "$AUDIT_DIR/runtime.json"

python -m pip_audit -r torch-requirements.txt \
  --strict --vulnerability-service pypi --format json \
  --aliases on --desc off --progress-spinner off \
  --output "$AUDIT_DIR/torch.json"

python -m pip_audit --disable-pip -r docs-requirements.txt \
  --strict --vulnerability-service pypi --format json \
  --aliases on --desc off --progress-spinner off \
  --output "$AUDIT_DIR/docs.json"

python -m pip_audit -r atlas-contract-requirements.txt \
  --strict --vulnerability-service pypi --format json \
  --aliases on --desc off --progress-spinner off \
  --output "$AUDIT_DIR/atlas-contract.json"
```

| Surface | Manifests | Exit | Resolved Dependencies | Vulnerable Packages | Raw Feed Records | Alias-Aware Unique Identities | JSON SHA-256 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Combined runtime | `requirements.txt`, `torch-requirements.txt` | 1 | 194 | 2 | 23 | 21 | `65db11cbf11f162241fc398674a5f91374a916ac43d4c984694ddb9e254c1ad5` |
| Torch | `torch-requirements.txt` | 1 | 39 | 2 | 23 | 21 | `faea4c874c75c7260064c96e26fad5e3105d2fd6c2b20d17ee4abbb57043c6b6` |
| Documentation | `docs-requirements.txt` | 0 | 42 | 0 | 0 | 0 | `c7fb014d9d45092476134bc78fe7e3fd81df93c66733b932c734d5fe27672afe` |
| Atlas contract | `atlas-contract-requirements.txt` | 0 | 5 | 0 | 0 | 0 | `025906bb0be0ae036140e484f0dcc2845e25e11e36c18a7aa23af5e05fd55db9` |

The runtime and Torch surfaces contain the same 21 alias-aware identities. Their 23 raw records
are preserved because `PYSEC-2025-191` and `PYSEC-2025-41` each occur twice with independently
emitted metadata. Counts across surfaces are observations, not additive vulnerability identities.
Several runtime requirements remain open ranges, so the resolver can select newer versions without
a committed manifest change. This is dated snapshot evidence, not a reproducible lock.

#### 6.1.1.1.1 Archived Issue #59/#61 accepted advisories

This subsection is immutable historical Issue #59/#61 audit evidence. Feed disappearance in a
later audit is reconciliation evidence only; it is not proof of remediation, non-reachability, or
an upstream fix.

Result: 23 known vulnerabilities across 194 resolved packages.

| Package | Manifest Constraint | Audited Resolved Version | Finding Count | Current Disposition |
| --- | --- | ---: | ---: | --- |
| `torch` | `torch==2.4.1` | `2.4.1` | 22 | Temporarily accepted for the qualified local/CI Torch stack. Upgrade only as a coordinated PyTorch, PyG, torchao, notebook, and CI compatibility change. Never load an untrusted pickle-backed checkpoint. |
| `pytorch-lightning` | `pytorch-lightning==2.4.0` | `2.4.0` | 1 | Temporarily accepted with the current Torch stack; the feed lists no fix version. Never load an untrusted pickle-backed checkpoint. Revisit with the coordinated stack upgrade. |

Each row below is one raw feed record. Duplicate primary IDs remain separate when the feed emits
different alias or fix metadata.

| Package | Advisory ID | Feed Records | Fix Versions | Audited Version | Aliases | Surface |
| --- | --- | ---: | --- | ---: | --- | --- |
| `torch` | `PYSEC-2025-191` | 1 | `2.7.1rc1` | `2.4.1` | `CVE-2025-2953`, `GHSA-3749-ghw9-m3mg`, `BIT-pytorch-2025-2953` | Combined runtime; Torch |
| `torch` | `PYSEC-2025-41` | 1 | `2.6.0` | `2.4.1` | `CVE-2025-32434`, `BIT-pytorch-2025-32434`, `GHSA-53q9-r3pm-6pq6` | Combined runtime; Torch |
| `torch` | `PYSEC-2025-41` | 1 | `2.6.0` | `2.4.1` | `CVE-2025-32434`, `GHSA-53q9-r3pm-6pq6` | Combined runtime; Torch |
| `torch` | `PYSEC-2024-259` | 1 | `2.5.0` | `2.4.1` | `CVE-2024-48063` | Combined runtime; Torch |
| `torch` | `PYSEC-2025-205` | 1 | `2.7.1` | `2.4.1` | `CVE-2025-55553`, `BIT-pytorch-2025-55553` | Combined runtime; Torch |
| `torch` | `PYSEC-2025-206` | 1 | `2.9.0` | `2.4.1` | `BIT-pytorch-2025-55554`, `CVE-2025-55554` | Combined runtime; Torch |
| `torch` | `PYSEC-2025-207` | 1 | `2.7.1` | `2.4.1` | `BIT-pytorch-2025-55557`, `CVE-2025-55557` | Combined runtime; Torch |
| `torch` | `PYSEC-2025-204` | 1 | `2.9.0` | `2.4.1` | `CVE-2025-55552`, `BIT-pytorch-2025-55552` | Combined runtime; Torch |
| `torch` | `PYSEC-2026-139` | 1 | None listed | `2.4.1` | `BIT-pytorch-2026-4538`, `CVE-2026-4538` | Combined runtime; Torch |
| `torch` | `PYSEC-2025-209` | 1 | `2.7.1` | `2.4.1` | `BIT-pytorch-2025-55560`, `CVE-2025-55560` | Combined runtime; Torch |
| `torch` | `PYSEC-2025-208` | 1 | `2.7.1` | `2.4.1` | `BIT-pytorch-2025-55558`, `CVE-2025-55558` | Combined runtime; Torch |
| `torch` | `PYSEC-2025-191` | 1 | None listed | `2.4.1` | `GHSA-3749-ghw9-m3mg`, `CVE-2025-2953`, `BIT-pytorch-2025-2953` | Combined runtime; Torch |
| `torch` | `PYSEC-2025-198` | 1 | `2.7.0` | `2.4.1` | `CVE-2025-46148`, `BIT-pytorch-2025-46148` | Combined runtime; Torch |
| `torch` | `PYSEC-2025-203` | 1 | `2.9.0` | `2.4.1` | `BIT-pytorch-2025-55551`, `CVE-2025-55551` | Combined runtime; Torch |
| `torch` | `PYSEC-2025-194` | 1 | `2.13.0` | `2.4.1` | `BIT-pytorch-2025-3000`, `CVE-2025-3000`, `GHSA-rrmf-rvhw-rf47` | Combined runtime; Torch |
| `torch` | `PYSEC-2026-1970` | 1 | `2.8.0` | `2.4.1` | `BIT-pytorch-2025-3730`, `GHSA-887c-mr87-cxwp`, `CVE-2025-3730` | Combined runtime; Torch |
| `torch` | `PYSEC-2026-2286` | 1 | `2.10.0` | `2.4.1` | `CVE-2026-24747`, `GHSA-63cw-57p8-fm3p` | Combined runtime; Torch |
| `torch` | `CVE-2025-2148` | 1 | None listed | `2.4.1` | `GHSA-c678-jfcj-6jmf` | Combined runtime; Torch |
| `torch` | `CVE-2025-2149` | 1 | None listed | `2.4.1` | `GHSA-x3gm-94wq-g975` | Combined runtime; Torch |
| `torch` | `CVE-2025-2998` | 1 | None listed | `2.4.1` | `GHSA-f4hp-rmr7-r7v8` | Combined runtime; Torch |
| `torch` | `CVE-2025-2999` | 1 | `2.9.1` | `2.4.1` | `GHSA-vgrw-7cvw-pwgx` | Combined runtime; Torch |
| `torch` | `CVE-2025-3001` | 1 | `2.10.0` | `2.4.1` | `GHSA-qfhq-4f3w-5fph` | Combined runtime; Torch |
| `pytorch-lightning` | `PYSEC-2026-3043` | 1 | None listed | `2.4.0` | `GHSA-75m9-98v2-hjpm`, `CVE-2026-31221` | Combined runtime; Torch |

The accepted risk is limited to trusted local and CI inputs on the pinned stack; it is not a
claim that the vulnerable paths are unreachable. Revisit the Torch and Lightning dispositions
when a manifest changes, the advisory feed changes, a fix becomes available, input trust or
reachability expands, or Issue #62 qualifies a coordinated replacement stack. Any upgrade must
validate Torch, PyG, torchao, notebook execution, and CI together rather than updating one pin.

### 6.1.1.2 Current Issue #63 locked four-surface audit

Last reviewed: 2026-08-16. `requirements/lock-policy.toml` defines the four logical surfaces and
their exact lock inputs. `make audit-advisories` first passes the offline lock verifier, derives six
temporary exact `package==version` projections from the validated locks, and invokes
`pip-audit==2.10.0` with `--disable-pip --no-deps`. It does not re-resolve human source manifests.
The projections contain 210 combined-runtime packages plus the two audited PyG extensions, 41
Torch packages plus the two extensions, 45 documentation packages, and 10 Atlas-contract packages.
The spaCy model and target-specific pyg-lib wheels are recorded as non-PyPI provenance evidence;
they are never mislabeled as clean PyPI audit results.

```text
make verify-dependency-locks  # offline structure, hashes, sources, consumers, and image coherence
make audit-advisories         # networked feed comparison over exact lock-derived projections
```

Result: 3 known vulnerabilities across 212 resolved packages.

| Package | Manifest Constraint | Audited Resolved Version | Finding Count | Current Disposition |
| --- | --- | ---: | ---: | --- |
| `setuptools` | `resolver-selected transitive dependency` | `81.0.0` | 2 | Temporarily accepted for the qualified resolver observation. The feed lists `83.0.0` as the fix. Revisit when the resolver selects that floor or the dependency contract can be advanced without breaking the qualified stack. |
| `torch` | `torch==2.11.0` | `2.11.0` | 1 | Temporarily accepted for the qualified Torch 2.11 matrix. The feed lists `2.13.0` as the fix, but Torch 2.13 lacks the complete approved PyG compiled-extension wheel surface. Never load untrusted pickle-backed checkpoints. |

Each row below is one raw feed record. The duplicate setuptools primary ID is preserved because
the resolver feed emitted it twice; policy comparison remains alias-aware and identity-based.

| Package | Advisory ID | Feed Records | Fix Versions | Audited Version | Aliases | Surface |
| --- | --- | ---: | --- | ---: | --- | --- |
| `setuptools` | `PYSEC-2026-3447` | 1 | `83.0.0` | `81.0.0` | `BIT-setuptools-2026-59890`, `CVE-2026-59890`, `GHSA-h35f-9h28-mq5c` | Combined runtime; Torch; Documentation; Atlas contract |
| `setuptools` | `PYSEC-2026-3447` | 1 | `83.0.0` | `81.0.0` | `BIT-setuptools-2026-59890`, `CVE-2026-59890`, `GHSA-h35f-9h28-mq5c` | Combined runtime; Torch; Documentation; Atlas contract |
| `torch` | `PYSEC-2025-194` | 1 | `2.13.0` | `2.11.0` | `BIT-pytorch-2025-3000`, `CVE-2025-3000`, `GHSA-rrmf-rvhw-rf47` | Combined runtime; Torch |

The input hashes enforced by D10 are:

| Input | SHA-256 |
| --- | --- |
| `vulnerability-audit-requirements.txt` | `889b9ef59073551e13c18fcb421ae6f35491db95081941eb7e05ae44af7a5918` |
| `requirements.txt` | `6e86caa5a287e9566e15bdffbb6628249397307783dee3b6e98e728ef06275b9` |
| `torch-core-requirements.txt` | `28b09abee07d1c3551b47f28938a546bf1dd712f18e34bea9b40e3d49410810b` |
| `torch-ecosystem-requirements.txt` | `9e0083918fd410e30aea337ba281fbfe05f89846eff067cf4997d296e1ce1dff` |
| `torch-requirements.txt` | `5ab5581cbaf6aefd0698b63e22279e799a1edd65ae003b52b65d6a3b5d64c5cb` |
| `torch-audit-requirements.txt` | `6d544b226c6e96f296c5105a20ea00704c3e1db4bf91946392df8f3ec5236d2a` |
| `pyg-extension-audit-requirements.txt` | `3bdf07aaf4dc3a02524d7f7e11f6127c68203403201dc32d36b356670bfff498` |
| `docs-requirements.txt` | `8cf8df4156b6537310a9e6ad151a9d3280138f1c23178a75064ed93129d9538c` |
| `atlas-contract-requirements.txt` | `ca9faca3b1769495e70e9cb317fc66cbcf747981e0932f3c9df8a817fccdd768` |
| `security/accepted-advisories.json` | `452ac2a787c5c13814bab63f54a97c742dd22fa8da6a6a550b8f4a416df18dbb` |

The complete generated-lock inventory is the 14 outputs listed by
`requirements/lock-policy.toml`: bootstrap, compiler, audit, Atlas contract, documentation, and
core/runtime/root for Darwin arm64, Linux x86_64, and Linux aarch64. Every package-changing
consumer installs the hash-required bootstrap lock first. The four-stage runtime installer then
uses core, runtime, and root in order; `python-louvain==0.16` is the sole approved sdist and
`en-core-web-sm==3.8.0` the sole approved direct URL. Linux Torch packages come from the exact CPU
index, compiled PyG wheels from the exact Torch 2.11 flat page, and Linux torchao from its exact CPU
project page. `uv==0.11.19` is regeneration tooling only and is not added to notebook runtime locks.

Offline `make verify-dependency-locks` proves syntax, hash presence, cross-file coherence, source
policy, consumer wiring, and committed image-ledger equality. It cannot prove that a remote tag or
artifact still has the reviewed bytes. `make lock-check` independently regenerates every lock;
`make image-lock-check` resolves each registry index and native child. A controlled update uses
`make lock-write`, reviews the complete diff, then runs both networked checks and clean installs.
Rollback reverts the human inputs, policy, all generated locks, image ledger/references,
installer/consumer wiring, advisory policy, and documentation as one coherent change.

Issue #64 retains external data/model content integrity beyond the locked spaCy wheel. Issue #65
retains Atlas runtime ownership; this parent contract audits only the Atlas validation role. Issue
#66 retains the complete quantization notebook, which remains manual-only despite the locked tiny
QAT surface.

The pre-resolved `pyg-extension-audit-requirements.txt` supplement contains exactly
`torch-scatter==2.1.2` and `torch-sparse==0.6.18`; it contains neither `torch-cluster` nor
`torch-spline-conv`.
pyg-lib is an exact external-index wheel outside ordinary PyPI audit coverage; its version and provenance are verified by `verify_torch_stack`.

`PYSEC-2025-194` is retained at the new Torch version and continues to alias
`CVE-2025-3000`, `GHSA-rrmf-rvhw-rf47`, and `BIT-pytorch-2025-3000`.
`PYSEC-2026-3447` is new for resolver-selected setuptools. The prior Lightning identity and every
prior Torch identity other than `PYSEC-2025-194` disappeared from this capture. That disappearance
is reconciliation evidence only, not proof of remediation, non-reachability, or an upstream fix;
the complete prior rows remain in the archived Issue #59/#61 subsection above. No current identity
was re-keyed. Lightning remains pinned to 2.6.1 because the upstream
`GHSA-w37p-236h-pfx3` supply-chain advisory covers the newer `>=2.6.2` release line; absence from
this pip-audit observation is not a zero-risk claim.

Revisit both accepted identities whenever manifests, resolved versions, feed identities or aliases,
fix availability, input trust, or path reachability changes. Torch, PyG, torchao, notebook, and CI
compatibility remain one coordinated upgrade boundary.

### 6.1.1.3 Alias-aware historical reconciliation

The following reconciliation compares the 2026-07-04 ledger identities with the current feed.
Aliases identify re-keyed records; an alias is not an additional vulnerability.

| Historical Identity | Current Identity | Classification | Review |
| --- | --- | --- | --- |
| `PYSEC-2025-191` | `PYSEC-2025-191` | Retained | Still emitted as two raw records with different fix metadata. |
| `PYSEC-2025-41` | `PYSEC-2025-41` | Retained | Still emitted as two raw records with different alias metadata. |
| `PYSEC-2024-259` | `PYSEC-2024-259` | Retained | Same primary identity. |
| `PYSEC-2025-205` | `PYSEC-2025-205` | Retained | Same primary identity. |
| `PYSEC-2025-206` | `PYSEC-2025-206` | Retained | Same primary identity. |
| `PYSEC-2025-207` | `PYSEC-2025-207` | Retained | Same primary identity. |
| `PYSEC-2025-204` | `PYSEC-2025-204` | Retained | Same primary identity. |
| `PYSEC-2026-139` | `PYSEC-2026-139` | Retained | Same primary identity. |
| `PYSEC-2025-209` | `PYSEC-2025-209` | Retained | Same primary identity. |
| `PYSEC-2025-208` | `PYSEC-2025-208` | Retained | Same primary identity. |
| `PYSEC-2025-198` | `PYSEC-2025-198` | Retained | Same primary identity. |
| `PYSEC-2025-203` | `PYSEC-2025-203` | Retained | Same primary identity. |
| `CVE-2025-3000` | `PYSEC-2025-194` | Re-keyed | The former primary is a current alias; the current record lists fix `2.13.0`. |
| `CVE-2025-3730` | `PYSEC-2026-1970` | Re-keyed | The former primary is a current alias; fix `2.8.0` remains listed. |
| `CVE-2025-2148` | `CVE-2025-2148` | Retained | Same primary identity. |
| `CVE-2025-2149` | `CVE-2025-2149` | Retained | Same primary identity. |
| `CVE-2025-2998` | `CVE-2025-2998` | Retained | Same primary identity. |
| `CVE-2025-2999` | `CVE-2025-2999` | Retained | Same primary identity. |
| `CVE-2025-3001` | `CVE-2025-3001` | Retained | Same primary identity. |
| `CVE-2026-31221` | `PYSEC-2026-3043` | Re-keyed | The former primary is a current alias; no fix version is listed. |
| `PYSEC-2026-597` (`nltk`) | — | Absent | Absent from the 2026-08-12 snapshot; archived audit provenance only, not proof of remediation, reachability, or an upstream fix. The open range resolved to `nltk==3.10.3`; this is resolver drift, not a committed pin change. |
| — | `PYSEC-2026-2286` / `CVE-2026-24747` | Genuinely new | First recorded in this snapshot; the feed lists fix `2.10.0`. |

### 6.1.1.4 Enforcement boundary

This manually reviewed ledger is the canonical record for the
[current accepted-advisories snapshot](#6112-current-issue-63-locked-four-surface-audit). The repository's
[security policy](../SECURITY.md) describes how new advisory uncertainty is triaged.
`security/accepted-advisories.json` is the policy artifact. `make audit-advisories` runs all four
audit surfaces without suppression: combined runtime, Torch, documentation, and the parent-owned
Atlas contract. New primary advisory IDs and accepted-version drift fail the gate. A disappeared
accepted primary ID is reconciliation evidence, not proof of remediation, reachability, or an
upstream fix.

Review changes the JSON policy and current Markdown ledger rows together through review, including
the associated aliases, risk language, and historical reconciliation. Do not treat feed absence as
permission for an automatic removal: rerun all four surfaces, verify the resolved version and
primary-ID/alias relationship, then make the reviewed JSON and current-ledger update together.
The audit does not initialize Atlas or start a service. Issue #62 owns the coordinated Torch-stack
upgrade; Issue #63 adds complete dependency locks and derives this comparison from those exact
locks. The direct `pip-audit` tool pin remains isolated in the audit-tool lock.

The audit projections are generated from the committed locks and installed with no dependency
resolution. Combined-runtime and Torch evidence includes explicit non-PyPI records for the PyG
extension wheels selected from the approved flat source; missing, extra, changed, duplicate, or
ambiguous records fail closed. Runtime installation continues to consume the complete selected
platform lock rather than an audit projection.

### 6.1.1.5 Removal and reconciliation runbook

An accepted ID absent from valid audit output is evidence only, not proof of remediation,
non-reachability, or an upstream fix. Remove it only through this reviewed sequence:

1. Run `make audit-advisories` across all four audit surfaces.
2. Confirm the resolved package and version and the primary-ID/alias relationship.
3. Change the JSON baseline and current Markdown ledger row together through review.
4. Refresh snapshot metadata, raw JSON hashes and counts, summary, and current tables whenever
   the audit observation changes. Metadata includes the as-of date, repository commit, auditor
   version, platform, and exact commands.
5. Preserve the removed record in historical reconciliation with the evidence-only disclaimer and
   update its risk language.
6. Run focused comparator tests, full `make test`, `make verify`, `make lint`, `make docs-check`,
   `make docs-wiki`, and live `make audit-advisories`.
7. Integrate through a feature-to-`develop` pull request, then a `develop`-to-`main` pull request.

## 6.1.2 Torch Stack Pin

The supported Python 3.11 CPU matrix is torch==2.11.0, torchvision==0.26.0,
torchaudio==2.11.0, pytorch-lightning==2.6.1, torchmetrics==1.9.0, torchao==0.18.0,
torch-geometric==2.8.0.post1, pyg-lib==0.8.0, torch-scatter==2.1.2, and
torch-sparse==0.6.18; thekaveh-nnx[lm]==0.2.0 remains the separately verified consumer pin.

torch-core-requirements.txt contains the Torch trio. torch-ecosystem-requirements.txt contains
Lightning, TorchMetrics, and torchao. torch-requirements.txt contains the ecosystem include, the
Torch 2.11 CPU PyG selector, pyg-lib, scatter, sparse, and PyG. torch-audit-requirements.txt
contains core plus ecosystem plus PyG. pyg-extension-audit-requirements.txt contains only scatter
and sparse; pyg-lib is an external-index artifact verified by WHEEL/RECORD, platform, ownership,
import, and sampler gates.

The temporary verifier-local import debts have literal keys Torch 2.11.0 with outer torch-geometric 2.8.0.post1 and Torch 2.11.0 with outer torch-sparse 0.6.18. Every record in a nonempty captured group must have category identity DeprecationWarning, message `torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`., and exact selected-Torch RECORD origin torch/jit/_script.py. Count and line number are not pinned. Pytest remains -W error, no global filter is allowed, and a warning-free fresh-interpreter probe retires the exception.

Separately, the immutable Torch 2.11.0 + torchao 0.18.0 + thekaveh-nnx 0.2.0 + qat_config="8da4w" QAT debt is asserted only around NNx model.train: exactly one identity-UserWarning must equal the complete Deprecation: TorchAODType is deprecated, please use the torch.intN dtype instead (e.g. TorchAODType.INT4 -> torch.int4) and originate from the selected torchao RECORD path torchao/quantization/quant_primitives.py. The three distribution versions are parsed fail-closed and compared by their PEP 440 public versions, so platform local tags such as Torch 2.11.0+cpu retain the 2.11.0 key while malformed versions fail validation. Pytest remains -W error; zero warnings or tuple drift stops qualification for debt retirement; Issue #66 or an earlier NNx/Atlas upgrade owns migration to the current torch.int4 API.

`make install-torch-stack` selects one complete hash-required platform lock and installs it through
the shared sanitized boundary; source provenance is retained in the lock policy for PyPI, the
Linux CPU index, the PyG flat wheel source, and Darwin's native index. NNx remains binary-only.
Acceptance requires pip-check, the ten-component stack
verifier, the NNx verifier, four-surface advisory reconciliation from six commands, full repository
tests, zero-skip focused graph/quantization tests, Tier A/B/C 18/6/4, Darwin arm64, native Linux
arm64 Docker, Linux x86_64 PR gates, and three-surface documentation parity. Any failure rejects
the matrix and rollback restores the complete prior contract in a fresh environment or rebuilt
image.

Issue #62 completed the runtime matrix; Issue #63 preserves that matrix while replacing routine
resolution with verified immutable locks.

## 6.1.3 Manual-Only Quantization Notebook

notebooks/quantization-mnist-ffnn-pytorch/notebook.ipynb remains manual-only under Issue #66.
Issue #62 qualifies only the tiny PTQ/QAT dependency surface on torch==2.11.0,
torchvision==0.26.0, torchao==0.18.0, and thekaveh-nnx[lm]==0.2.0. Do not add the complete notebook
to Tier A/B/C without Issue #66 acceptance; Atlas remains Issue #65 and is not a substitute.

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
the canonical contract for ml-eng-lab notebook verification and CI. Record canonical local
evidence with:

```bash
make verify-nnx-install
```

The fail-closed verifier reads the one exact `[lm]` manifest pin and requires one matching installed
`thekaveh-nnx` distribution at that version. Canonical mode rejects any `direct_url.json`, requires
the distribution file inventory to own `WHEEL`, `RECORD`, and `nnx/__init__.py`, and proves that the
resolved `nnx` module is the same distribution-owned import origin outside this repository.
Diagnostics are redacted to stable contract categories; local paths and direct URLs are not emitted.

CI obtains the NNx wheel from the selected hash-required Linux lock in both `pytest-repository` and
`pytest-nnx-surface`, then runs `make verify-nnx-install` after installation and immediately before
tests. The lock records the wheel hash while the verifier separately proves canonical distribution
ownership and rejects editable/direct-URL installs.

Issue #61 reviewed the latest stable 0.2.2 wheel without changing this final contract. A fresh
canonical trial passed `1,350` repository tests, Tier A `18/18`, Tier B `6/6`, and Tier C `4/4` on
Darwin arm64; `torch_sparse==0.6.18` imported and the graph tiers completed. The isolated QAT probe
also passed with Torch 2.11.0, torchvision 0.26.0, and torchao 0.18.0. These results establish that
the released wheel works in the trial environments, but they do not override the recommended
Atlas JupyterHub image's independent NNx 0.2.0 pin. Because 0.2.2-only `NNModel.train` identity
keywords are unsupported there, the repository retained 0.2.0 and removed the trial-only calls.

Every NNx release review must run the complete Tier A, Tier B, and Tier C matrix. Platform
assumptions do not waive a tier: the Issue #61 Darwin arm64 run disproved the former claim that
`torch_sparse` made Tier B/C impossible on macOS. Quantization remains a separate manual-only
probe until Issue #66 defines tiering; torchao 0.18 requires Torch >=2.11, so the accepted side
environment is the proven Torch 2.11.0 / torchvision 0.26.0 / torchao 0.18.0 combination rather
than a looser, unverified version floor.

Editable installs are allowed only for intentional upstream NNx development. After installing an
external checkout editable, run:

```bash
NNX_ALLOW_EDITABLE=1 make test-nnx-surface
```

Only the exact value `1` selects editable-development mode. The verifier still requires the exact
manifest and distribution versions, valid PEP 610 JSON with a local `file:` URL and
`dir_info.editable` set to true, and an `nnx` import beneath that declared source directory. It
rejects malformed metadata, non-editable or VCS direct references, and unrelated shadow imports.
An editable result is development-surface evidence, never released-wheel evidence; return to
canonical mode and rerun `make verify-nnx-install` before recording release compatibility.

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
| NNx + language extras | `thekaveh-nnx` / `nnx` 0.2.0; `datasets` 5.0.0; `tokenizers` 0.22.2 | Atlas-owned image evidence; matches notebook imports and the `[lm]` extra at the observed version |
| Torch | `torch` 2.11.0+cpu; `torchvision` 0.26.0+cpu; `torchaudio` 2.11.0+cpu | Atlas runtime is newer than local/CI; do not infer a local pin bump |
| Torch extensions | `torchao` 0.17.0; `torch-geometric` 2.6.1; `python-louvain` 0.16 | Required import surfaces are present |
| NLP | spaCy 3.8.14, `en_core_web_sm` 3.8.0; NLTK 3.10.0 with VADER | Both task assets resolve |
| Notebook imports | 62 imports across active notebooks; zero failures | Import-level compatibility evidence |

The same live check imported the NumPy MNIST sibling modules from the mounted
checkout. This validates the consumer mount separately from package metadata.
It does not turn a successful import into a completed training or performance
smoke.

## 6.1.9 Atlas Versus Local/CI Dependency Boundaries

Atlas is Atlas-owned infrastructure and remains Issue #65. The checked-in Torch 2.11 CPU manifests
are authoritative for make test, papermill CI, Dockerfile, and Codespaces; no Atlas package
observation changes that contract. The complete quantization notebook remains manual-only under
Issue #66 even though Issue #62 qualifies its tiny Torch 2.11.0 + torchao 0.18.0 PTQ/QAT dependency
surface.

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

## 6.1.11 Canonical Bootstrap Tooling

The canonical installer uses the hash-required bootstrap lock and then installs the complete
platform lock selected by `requirements/lock-policy.toml`. Docker, Codespaces, CI, and local setup
delegate to that shared boundary; none carries a second dependency algorithm. Compiler tooling is
isolated in its own lock and is never part of the runtime environment.

## 6.1.12 Qualified Reproducibility Boundary

Human-authored manifests may retain ranges, but routine installation consumes committed,
hash-required locks for the qualified Darwin arm64, Linux x86_64, or Linux aarch64 target. Docker
and devcontainer bases use exact tag-plus-index-digest references with verified child manifests.
Offline verification proves policy, input, marker, hash-shape, and cross-file coherence; networked
`make lock-check` and `make image-lock-check` prove byte regeneration and remote digest identity.
This is reproducible for the qualified platform lock, not one cross-platform binary environment.
