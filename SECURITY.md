# 13 Security policy

This policy defines how to report security vulnerabilities in ml-eng-lab and how the maintainer
handles reports. It covers this repository as a personal machine-learning notebook lab, including
the consumer configuration that connects it to external runtimes and dependencies.

## 13.1 Supported versions

| Version or line | Security support |
| --- | --- |
| Current `main` | Maintained security line |
| Older commits and tags, including `v0.1.0` | Historical; no separate maintenance or backport line unless explicitly announced |
| `develop` and feature branches | Pre-release work; reports are welcome when the issue could reach `main` |

Security changes follow the repository's normal feature → `develop` → `main` workflow described
in the [repository conventions](docs/conventions.md). A report does not create a promise to support
an older environment, dependency set, tag, or commit.

## 13.2 Report a vulnerability privately

Use GitHub's private vulnerability-reporting form: open this repository on GitHub, select
**Security** → **Advisories** → **Report a vulnerability**, and submit the report there. Private
vulnerability reporting is enabled for the repository.

Do not open a public issue, discussion, or pull request for an undisclosed vulnerability. Do not
put an exploit, access token, Jupyter URL, private dataset, model credential, or other secret in a
public artifact. If the private form is temporarily unavailable, avoid public disclosure until a
private reporting route is available.

## 13.3 What to include

Provide enough evidence to reproduce and assess the issue without exposing unrelated data:

- the affected commit, tag, notebook, script, workflow, configuration, or dependency version;
- the runtime path involved: local venv, local Docker, Codespaces, CI, Atlas JupyterHub, host-native
  Ollama, or optional host-native ComfyUI;
- minimal reproduction steps or a narrowly scoped proof of concept;
- expected and observed behavior, security impact, prerequisites, and affected trust boundary;
- any known mitigation or safe configuration; and
- redacted logs or screenshots when they add evidence.

Do not access data you do not own, disrupt services, persist access, or broaden testing beyond the
minimum needed to demonstrate the issue.

## 13.4 Response and coordinated disclosure

The maintainer will make a best-effort attempt to acknowledge the report, reproduce it, determine
ownership, assess impact, and coordinate a correction and disclosure. Timing depends on severity,
reproducibility, upstream dependencies, and maintainer availability. There is no guaranteed
acknowledgement, remediation, disclosure, support, or backport SLA, and no promise of a bounty,
CVE assignment, or release cadence.

Keep the report and exploit details private while validation and remediation are active. The
maintainer and reporter should agree on a reasonable disclosure point; this request is not a demand
for indefinite confidentiality.

## 13.5 Scope and upstream ownership

In scope are vulnerabilities caused or exposed by this repository's notebooks, helper scripts,
verification and documentation tooling, GitHub Actions workflows, Docker or Codespaces
configuration, Atlas consumer overlays and lifecycle wrappers, dependency integration, committed
outputs, or credential handling.

NNx and Atlas are upstream projects. Report a vulnerability that exists only in an upstream
project through that project's private security route. Report it here as well when ml-eng-lab's
pin, consumer configuration, mount, wrapper, notebook, or documented workflow makes the issue
reachable or changes its impact. Atlas ownership and admission rules are defined by the
[Atlas pin-bump and service-admission runbook](docs/atlas-pin-bump-runbook.md).

## 13.6 Dependency advisories

Dependency reports are triaged against the exact version and execution surface that ml-eng-lab
actually consumes. The [dependency contract ledger](docs/dependency-contracts.md) records the
local/CI Torch stack, the independently pinned Atlas runtime, manual exceptions, audit evidence,
and coordinated upgrade criteria. A new advisory remains actionable even when the vulnerable code
path is not known to be exercised; the report should state the reachable path or uncertainty.

The broader development dependency graph is not fully locked, and the repository does not claim an
automated vulnerability-baseline gate. Dependency pin changes require the tests appropriate to the
affected local, CI, notebook, and Atlas surfaces rather than a version-only edit.

## 13.7 Notebook, model, data, and artifact safety

Notebooks and their saved outputs are executable content. Review code and output metadata before
running an untrusted notebook, use an isolated least-privilege environment, and do not expose host
credentials or sensitive mounts to code whose provenance is unknown.

Models and datasets should come from trusted or reviewed sources with provenance, applicable
license, source version, and a checksum when the provider supplies one or the task contract
requires one. The repository does not currently enforce hashes for every external model or dataset.

Python `pickle`, `torch.load`, `joblib`, and similar checkpoint formats can execute attacker-
controlled code during deserialization. Load them only from trusted sources. Prefer non-executable
formats such as `safetensors` where the workflow supports them, and prefer `weights_only=True` for
compatible `torch.load` calls; neither option removes the need to validate provenance and expected
tensor shape or content.

## 13.8 Secrets and committed outputs

Do not place credentials, private keys, access tokens, token-bearing Jupyter URLs, or sensitive
data in notebooks, cell outputs, `data/`, `runs/`, screenshots, logs, configuration examples, or
commits. Treat ignored paths as convenience boundaries, not secret stores. If a secret is exposed,
revoke or rotate it first, remove it from active use, then report the exposure privately; deleting
the visible file or commit alone does not invalidate the credential.

## 13.9 Runtime boundaries

Atlas JupyterHub can execute against a mounted checkout and persistent runtime volumes. That access
means code in the remote kernel may read or modify mounted project data; the container, bind mount,
and Jupyter volume are operational boundaries, not trust boundaries. Follow the
[JupyterHub integration guide](docs/jupyterhub-integration.md) and use the least privilege and
smallest mount appropriate to the task. The same caution applies to local Docker and Codespaces.

For this consumer, Ollama must be host-native. Optional ComfyUI, if a future task admits it, must
also use a reviewed host-native source. Never start either service containerized for ml-eng-lab.
Service availability in Atlas does not authorize notebook use; admission remains explicit in the
[Atlas runbook](docs/atlas-pin-bump-runbook.md).
