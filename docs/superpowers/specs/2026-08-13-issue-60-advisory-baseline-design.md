# 12.17 Issue 60 Machine-Readable Advisory Baseline Design

## 12.17.1 Purpose

Issue #59 established a current, human-reviewed four-surface vulnerability snapshot. Issue #60
turns the accepted portion of that snapshot into an enforceable policy without changing any
dependency version, suppressing the auditor, or pretending the repository has a complete lock.

The design must distinguish three states:

1. a known advisory for the exact reviewed package version;
2. a newly observed advisory or accepted-version drift that must block integration; and
3. a previously accepted advisory that is no longer emitted and therefore needs documented
   reconciliation but is not, by itself, proof of remediation.

The result is a deterministic policy artifact, a fail-closed live comparator, an offline
repository-consistency check, and an isolated required CI signal.

## 12.17.2 Selected approach

Use a canonical JSON policy at `security/accepted-advisories.json`, a standard-library comparison
module at `scripts/advisory_baseline.py`, an exact `pip-audit==2.10.0` tool manifest at
`vulnerability-audit-requirements.txt`, and an unconditional `dependency-audit` job that runs
`make audit-advisories`.

This is selected over embedding the live audit in `pytest-repository`. A separate job preserves
attribution, timeout isolation, and rollback isolation. It becomes genuinely merge-blocking only
after its live context is added to ruleset `18620095` alongside the existing
`pytest-repository` and `atlas-consumer-policy` contexts.

The selected architecture has three coupled but distinct surfaces:

```text
live pip-audit observations
            |
            v
security/accepted-advisories.json
            |
            v
docs/dependency-contracts.md current accepted-advisories table
```

The live comparator owns the first relationship. The offline repository verifier owns the second.
`make verify` remains deterministic and network-independent.

## 12.17.3 Alternatives considered

### 12.17.3.1 Embed live resolution in `pytest-repository`

This would reuse an already-required check and avoid a ruleset update. It was rejected because a
vulnerability-feed outage or policy mismatch would be reported as a generic pytest failure, share
the complete-suite timeout, and couple rollback of the security gate to the repository-test job.

### 12.17.3.2 TOML policy with verifier-integrated live mode

Repeated TOML tables are comment-friendly, but noisier for the accepted identity set and require
more custom canonical-writing behavior. More importantly, a verifier live mode would tempt callers
to make the normal `make verify` path network-dependent.

### 12.17.3.3 Commit raw `pip-audit` JSON

Raw output preserves every resolved dependency and feed field, but produces large resolver-noise
diffs and confuses observations with acceptance policy. The dated raw-output hashes and complete
human ledger from Issue #59 remain the audit evidence; the new JSON stays compact and reviewable.

Generating `--ignore-vuln` arguments from any format is prohibited. Such suppression cannot bind
an advisory to an accepted package version or audit surface.

## 12.17.4 Canonical policy schema

The JSON root has exactly three keys:

```json
{
  "schema_version": 1,
  "audited_surfaces": [
    "combined-runtime",
    "torch",
    "documentation",
    "atlas-contract"
  ],
  "accepted_advisories": [
    {
      "advisory_id": "CVE-2025-2148",
      "package": "torch",
      "accepted_version": "2.4.1",
      "surfaces": ["combined-runtime", "torch"]
    }
  ]
}
```

The current artifact contains 21 entries: one for each unique
`(package, advisory_id, accepted_version)` identity in the Issue #59 snapshot. The Markdown ledger
continues to preserve all 23 raw feed records, including the duplicated `PYSEC-2025-191` and
`PYSEC-2025-41` records. Raw-record count is evidence, not the policy key.

Schema rules are fail-closed:

- `schema_version` must be integer `1`.
- `audited_surfaces` must equal the four names above in that order, with no duplicates.
- Each accepted item has exactly `advisory_id`, `package`, `accepted_version`, and `surfaces`.
- Package names use PEP 503 normalization before comparison and are stored in normalized form.
- `advisory_id`, package, and version must be non-empty strings.
- Each item has at least one known surface; surfaces use canonical order and contain no duplicates.
- Accepted items are unique and ordered by package, accepted version, then advisory ID.
- Unknown keys, unknown surfaces, duplicate identities, unsupported schema versions, and
  non-canonical formatting are errors.
- Canonical serialization is UTF-8, two-space indented JSON, one trailing newline, and no automatic
  write-back to the accepted path.

Aliases, fix versions, risk rationale, reachability uncertainty, and historical reconciliation
remain in `docs/dependency-contracts.md`. Primary IDs are the policy identities. A feed re-key is
therefore reported as a new primary ID and a disappeared old ID, forcing explicit alias review.

## 12.17.5 Audit surfaces and execution

`scripts/advisory_baseline.py` declares the four audit surfaces as typed constants rather than
reading executable command strings from JSON:

| Surface | Inputs | Special behavior |
| --- | --- | --- |
| `combined-runtime` | `requirements.txt`, `torch-requirements.txt` | Normal strict resolution |
| `torch` | `torch-requirements.txt` | Normal strict resolution |
| `documentation` | `docs-requirements.txt` | `--disable-pip` against the hash-locked file |
| `atlas-contract` | `atlas-contract-requirements.txt` | Parent-owned focused contract only |

Every command uses `python -m pip_audit`, `--strict`, the PyPI vulnerability service, JSON output,
aliases on, descriptions off, and the progress spinner off. Audit exits `0` and `1` are completed
observations. Any other exit, missing output, malformed JSON, missing surface, or skipped dependency
resolution fails closed.

The Atlas surface does not initialize the `infra/` submodule, inspect Atlas image dependencies,
contact JupyterHub, or start a service.

## 12.17.6 Comparator interfaces

The module exposes small, separately testable boundaries:

```python
def load_baseline(path: Path) -> Baseline: ...
def normalize_pip_audit(surface: str, payload: object) -> Observation: ...
def compare_baseline(baseline: Baseline, observations: Sequence[Observation]) -> Comparison: ...
def run_audit_surfaces(repo: Path, runner: AuditRunner = subprocess.run) -> Sequence[Observation]: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

The CLI prints deterministic, path-safe diagnostics and returns nonzero on invalid policy,
invalid observation, new identity, version drift, unexpected package/surface association, or a
missing required surface. It never prints credentials from index URLs and never accepts a flag
that writes directly to the checked-in baseline.

## 12.17.7 Comparison semantics

The gate fails when:

- an observed primary advisory ID is absent from the baseline for its package and version;
- an accepted package resolves at a different version, whether or not that new version emits the
  old advisory;
- an advisory appears for an unaccepted package or surface;
- any required surface is missing, duplicated, malformed, or not fully audited;
- policy JSON is malformed, non-canonical, duplicated, incomplete, or uses unknown fields;
- the baseline and current Markdown accepted-advisory identities disagree; or
- `--ignore-vuln`, a broad failure mask, or an alternate command path weakens the CI contract.

A baseline identity that disappears from valid current output does not fail the live comparison.
It produces a deterministic reconciliation notice. This preserves OM-047's original boundary:
block new IDs and accepted-version drift, while allowing reviewed removal. Disappearance is not
proof of remediation, non-reachability, or an upstream fix.

The removal workflow is deliberately manual:

1. rerun all four surfaces;
2. confirm package resolution and primary-ID/alias relationships;
3. treat feed absence as evidence only;
4. remove the JSON entry and current Markdown row together;
5. update historical reconciliation and risk language;
6. refresh snapshot metadata when the underlying observation changed;
7. run focused, full, verifier, lint, site, wiki, and live-audit checks; and
8. integrate through feature to `develop`, then `develop` to `main`.

There is no `accept-current`, `update-in-place`, or suppression-generating command.

## 12.17.8 Offline repository verification

`scripts/verify_repo.py` extends D10 without running the network audit. It loads the JSON policy,
validates its schema and canonical serialization, parses the unique current accepted identities
from §6.1.1.2, and requires exact parity of package, primary ID, accepted version, and surfaces.

The existing Markdown structure and raw-record count checks remain. Historical rows remain outside
the current-snapshot parse boundary. This preserves the distinction between 21 accepted identities
and 23 feed records.

## 12.17.9 CI and ruleset contract

The `dependency-audit` job is unconditional on the workflow's existing pushes and pull requests to
`develop` and `main`, weekly schedule, and manual dispatch. It uses Ubuntu 24.04 and Python 3.11,
checks out without submodules or persisted credentials, caches against the focused tool manifest
and all audited manifests, installs only `vulnerability-audit-requirements.txt`, and runs exact
`make audit-advisories`.

Workflow-contract tests reject job conditions, `continue-on-error`, services, containers, altered
step order, masked shell failures, missing inputs, unpinned `pip-audit`, `--ignore-vuln`, service
lifecycle commands, and Atlas/Ollama/ComfyUI/localhost probes. The `verify-repo` job executes the
positive workflow-contract test by explicit node ID so deleting the test cannot make its guard
vacuously green.

After the feature PR establishes the live context, ruleset `18620095` is updated to require exactly:

- `pytest-repository`;
- `atlas-consumer-policy`; and
- `dependency-audit`.

All other ruleset parameters remain unchanged.

## 12.17.10 Documentation surfaces

Canonical documentation updates cover:

- the implemented gate and removal process in `docs/dependency-contracts.md`;
- the contributor command and required signal in `docs/conventions.md` and `CONTRIBUTING.md`;
- the exact security-policy boundary in `SECURITY.md`;
- `security/` in the README repository tree;
- the isolated CI signal in `docs/architecture.md`;
- OM-047 as resolved in the maintenance ledger; and
- durable history in `CHANGELOG.md`.

The design and implementation records become manifest entries 12.17 and 12.18. Generated site and
wiki output remains ignored and is regenerated only for verification.

## 12.17.11 Test strategy

TDD begins with focused failures for:

- canonical policy parsing and serialization;
- all 21 real accepted identities;
- malformed schema, unknown keys, unsupported versions, duplicate identities, and ordering drift;
- exact four-surface command construction and exit `0`/`1` acceptance;
- missing or malformed observations and all other exit codes;
- new advisory IDs and package/surface drift;
- accepted-version drift with and without a currently emitted advisory;
- removed advisories and removed packages producing non-failing reconciliation notices;
- duplicate raw records collapsing to one policy identity;
- JSON-to-Markdown unique-identity parity;
- the exact Make target, CI job, focused self-contract selector, and required path filters; and
- canonical projection of enforcement and update language to site and wiki.

Final acceptance includes `make audit-advisories`, the full pytest suite, repository verification,
Ruff, strict documentation build, wiki generation, diff checks, live feature and release PR jobs,
and verification that the three required contexts are present in the active ruleset.

## 12.17.12 Scope boundaries

- Issue #61 owns NNx release review; this issue does not change the NNx pin.
- Issue #62 owns the coordinated Torch/PyG/Lightning/torchao upgrade; this issue records but does
  not remediate its accepted advisories.
- Issue #63 owns comprehensive locks, bootstrap pins, hashes, and base-image digests. Pinning the
  direct `pip-audit` tool is the focused exception needed to reproduce this gate's behavior.
- The Atlas audit covers only the parent-owned focused requirement file.
- No Atlas, JupyterHub, Docker, Docker Compose, Ollama, ComfyUI, or container operation is required.

## 12.17.13 Success criteria

The design is complete when the checked policy is deterministic and human-reviewable, new primary
IDs and accepted-version drift fail live CI, removed identities have an explicit non-automatic
reconciliation path, the human ledger and JSON policy cannot drift, the audit is a required GitHub
context, and the full repository and three-surface documentation contracts remain green.
