# 12.31 Issue 67 Atlas Required-Environment Contract Design

## 12.31.1 Purpose and observed state

Issue #67 closes a gap between the Atlas service-admission runbook and the
machine-readable notebook contract. Every active task currently declares an
`atlas.required_services` list, but `scripts/docs/notebook_infrastructure.py`
cannot express which injected environment variables a future service must
provide. The generated task matrix therefore cannot display that requirement
or prove that a newly admitted service has a usable notebook-facing binding.

All twenty-one active tasks currently authorize JupyterHub alone. This issue
does not admit another service, start Atlas, change the retained Atlas pin, or
change any notebook execution tier. It evolves the parent-owned schema so a
later task can request a service without relying on ambient terminal state,
notebook literals, undocumented endpoints, or patched files under `infra/`.

## 12.31.2 Decision and alternatives

The selected schema is an explicit list of environment requirements under the
existing `atlas` mapping:

```yaml
atlas:
  executor: jupyterhub
  default_mode: vscode-remote
  required_services: [jupyterhub, spark]
  required_env:
    - name: SPARK_REMOTE
      service: spark
  workspace_access: remote
  artifact_policy: atlas-jupyter-volume
  constraints: []
```

Each entry identifies an uppercase environment-variable name and the declared
service responsible for injecting it. It intentionally does not contain the
variable's value. Endpoint values, credentials, tokens, and host paths remain
runtime-owned and must never be copied into task metadata or generated docs.

Two alternatives are rejected:

1. **A list of variable names.** This is compact but cannot prove which
   admitted service supplies a variable, so service admission and environment
   injection can drift independently.
2. **A YAML mapping from variable name to service.** This is also compact, but
   ordinary YAML loading can overwrite duplicate keys before validation. A
   list preserves every entry so duplicate declarations fail explicitly.

The list form also leaves service identifiers forward-compatible. The parent
schema validates service syntax and relationships rather than maintaining a
closed enumeration of services that Atlas may add later.

## 12.31.3 Schema and parsing contract

`AtlasEnvironmentRequirement` is a frozen value object with `name` and
`service` strings. `AtlasTaskContract` gains
`required_env: tuple[AtlasEnvironmentRequirement, ...]` immediately after
`required_services`.

The parser applies these fail-closed rules:

1. `atlas.required_env` is required and must be a list; an empty list is valid.
2. Every entry is a mapping with exactly the keys `name` and `service`.
   Missing keys and unexpected keys, including a literal `value`, fail.
3. `name` must match `^[A-Z][A-Z0-9_]*$`.
4. `service` must match the existing service-ID grammar
   `^[a-z][a-z0-9-]*$`.
5. Environment names must be unique within a task contract.
6. Every environment requirement must reference a member of that task's
   `required_services` list.
7. Every required service other than the `jupyterhub` executor must have at
   least one environment requirement. This is the admission link between a
   service and its notebook-facing configuration.
8. Requirements are canonicalized by `(name, service)` before storage and
   rendering so semantically equivalent input order produces identical docs.

The schema does not inspect process environment variables, resolve values, or
claim that a binding is live. It validates declarative task intent only.

## 12.31.4 Migration and compatibility

All twenty-one existing task specifications are explicitly migrated with:

```yaml
required_services: [jupyterhub]
required_env: []
```

This is a deliberate schema migration rather than an implicit default. Missing
`required_env` fails, so future tasks cannot silently omit the environment
half of service admission. The empty lists preserve every existing
JupyterHub-only runtime, workspace, artifact, tier, and constraint contract.

Rollback remains clear: revert the parser/model/table column and remove the
twenty-one explicit empty fields together. No notebook or Atlas state is
involved.

## 12.31.5 Deterministic documentation projection

The canonical task matrix gains a `Required environment` column immediately
after `Required Atlas services`. Empty requirements render as an em dash.
Non-empty requirements render in canonical order as backticked variable names
with their supplying service, separated by `<br>`, for example:

```text
`MLFLOW_TRACKING_URI` (mlflow)<br>`SPARK_REMOTE` (spark)
```

The projection never prints values. Existing marker validation continues to
reject missing, duplicated, reversed, or stale generated-table content.

Canonical documentation explains the relationship among availability,
authorization, injection, and runtime proof:

- an Atlas track exposing a service does not authorize notebook use;
- a declared service without a required environment binding is invalid;
- an injected variable alone does not prove the service is enabled or healthy;
- admission additionally requires central source configuration, successful
  doctor/consumer validation, and a targeted JupyterHub smoke; and
- no environment value or secret is committed or rendered.

The design and implementation plan are registered in `docs/manifest.yaml` so
the repository, generated site, and native wiki remain one projection.

## 12.31.6 Regression strategy

Tests begin red against the current parser and cover:

- loading and canonicalizing valid multi-service requirements;
- explicit empty requirements for the repository's twenty-one current tasks;
- missing `required_env`, a non-list value, a non-mapping entry, missing keys,
  unexpected keys, malformed names, malformed services, duplicate names,
  undeclared service references, and non-JupyterHub services without bindings;
- deterministic table rendering with sorted bindings and escaped cells;
- drift detection for the updated repository matrix; and
- documentation assertions that availability is not authorization and that
  actual environment values are never part of the schema or table.

The focused suite is `tests/test_notebook_infrastructure.py`. Repository-level
coverage in `tests/test_verify_repo.py` and `tests/test_check_docs.py` is
updated where fixtures or canonical prose depend on the table schema.

## 12.31.7 Validation and release contract

Local qualification runs in a clean environment installed from the committed
locks:

```bash
make verify-torch-stack
make verify-nnx-install
python -m pytest tests/test_notebook_infrastructure.py -q
python -m pytest tests/test_check_docs.py tests/test_verify_repo.py -q
make test
make verify
make lint
make docs-check
make docs-wiki
```

No live Atlas startup is required because the issue changes declarative schema,
validation, and generated documentation only. The static Atlas consumer and
documentation CI gates still run remotely.

After independent review and exact-SHA qualification, publish through feature
to `develop`, `develop` to `main`, and content-neutral `main` back to
`develop`. Pages and the native wiki must publish successfully. Final cleanup
removes only Issue #67 refs, worktrees, environments, and generated temporary
artifacts; the retained Atlas volumes and pinned submodule remain untouched.
