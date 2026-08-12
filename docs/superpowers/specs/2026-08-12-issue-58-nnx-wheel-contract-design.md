# 12.13 Issue 58 canonical NNx wheel contract design

**Status:** Approved
**Date:** 2026-08-12
**Decision:** Add a fail-closed installation verifier that proves ml-eng-lab is testing the exact
`thekaveh-nnx[lm]==0.2.0` named-release wheel, while retaining a validated, explicit local
editable-development mode.

## 12.13.1 Purpose

Issue #58 closes a provenance gap in the repository's NNx compatibility evidence. The runtime
manifests pin `thekaveh-nnx[lm]==0.2.0`, but the existing CI smoke only imports `nnx` and prints its
path. A matching distribution version does not prove that Python imported files owned by that
distribution: an editable checkout, `PYTHONPATH`, `.pth` file, or shadow package can supply the
same import name.

The defect is reproducible in the current local development environment. Distribution metadata
reports version `0.2.0`, while PEP 610 metadata declares an editable sibling checkout and
`nnx.__file__` resolves under that checkout. Those tests are useful upstream-development evidence,
but they are not proof of compatibility with the released wheel.

The change makes canonical-wheel identity an executable local and CI contract without prohibiting
intentional NNx development. It does not change the NNx version, introduce a dependency lock or
wheel hash, start Atlas, or create another required GitHub context.

## 12.13.2 Considered approaches

### 12.13.2.1 Validate only the distribution version

Checking `importlib.metadata.version("thekaveh-nnx") == "0.2.0"` is inexpensive but insufficient.
Editable installations carry the same version, and import shadowing can redirect `nnx` after the
distribution is installed. This approach is rejected.

### 12.13.2.2 Validate only inside the focused NNx workflow

Replacing the print-only smoke in `pytest-nnx-surface` would improve CI but leave direct local
pytest, `make test`, and the required `pytest-repository` signal able to report release compatibility
against an editable checkout. This approach is rejected.

### 12.13.2.3 Validate installation identity in local tests and both relevant CI jobs

This is the selected approach. A reusable standard-library verifier derives the expected pin from
the manifest, validates named-release or explicitly editable installation evidence, and is called
by the NNx surface suite. Both `pytest-repository` and `pytest-nnx-surface` select the NNx wheel
explicitly and run the canonical verifier after dependency installation and before their test
workloads. The existing required contexts remain unchanged.

### 12.13.2.4 Add a wheel hash lock

A PyPI wheel hash would provide stronger artifact identity, but it would introduce a second NNx
pin surface and overlaps Issue #63's accepted dependency-lock and pinned-build-input scope. Issue
#58 proves named-release wheel provenance rather than cryptographic byte identity. This approach is
deferred to Issue #63.

## 12.13.3 Exact pin source of truth

`requirements.txt` remains the only NNx version source. The verifier must locate exactly one
active requirement whose normalized distribution name is `thekaveh-nnx`, whose extras are exactly
`[lm]`, whose operator is exactly `==`, and whose version is `0.2.0`. It rejects missing,
duplicate, ranged, editable, URL, VCS, or otherwise ambiguous declarations.

The implementation must not introduce a separately maintained version constant or NNx-specific
requirements file. Issue #61 owns a future version change; Issue #63 owns future hash locking. A
pin change in `requirements.txt` automatically changes the version the verifier expects, while
tests continue to enforce the exact requirement shape.

## 12.13.4 Canonical named-release mode

Canonical mode is the default. The verifier uses public `importlib.metadata.Distribution` APIs and
must prove all of the following:

1. exactly one installed distribution normalized as `thekaveh-nnx` is discoverable;
2. its normalized metadata name is `thekaveh-nnx`;
3. its installed version equals the exact manifest pin;
4. `direct_url.json` is absent, rejecting editable, local-directory, local-archive, and VCS direct
   installations as non-canonical;
5. distribution files include `WHEEL`, `RECORD`, and `nnx/__init__.py`;
6. the resolved `nnx` import specification points to the same resolved `nnx/__init__.py` path
   returned by `Distribution.locate_file()`; and
7. the owned package path is outside this repository checkout.

The verifier inspects the import specification before importing `nnx`, so a malicious or broken
shadow module is not executed merely to diagnose provenance. Symlinks are resolved before
comparison. Missing or malformed metadata fails closed.

Success output may name the public distribution, version, and mode. Failure output must not reveal
the PEP 610 URL, editable checkout path, repository path, site-packages path, credentials, or other
host-specific details.

## 12.13.5 Explicit editable-development mode

Intentional upstream NNx development uses one conspicuous opt-in:

```bash
NNX_ALLOW_EDITABLE=1 make test-nnx-surface
```

Only the exact value `1` enables editable mode; absent, empty, or any other value is canonical mode
or an invalid configuration. Editable mode is validation, not bypass. It still requires the exact
manifest and installed distribution versions, valid JSON PEP 610 metadata, a `file:` URL, and
`dir_info.editable == true`. The resolved `nnx` import must live beneath the declared editable
source directory. Non-editable direct URLs, VCS sources, malformed URLs, missing metadata, and
unrelated shadow packages remain failures.

The opt-in applies to local commands only. CI workflow, job, and step environments must not define
`NNX_ALLOW_EDITABLE` or `PYTHONPATH`. Documentation must label editable results as development
surface evidence and require canonical mode before recording released-wheel compatibility.

## 12.13.6 Test-suite integration

Add `scripts/verify_nnx_install.py` with pure validation helpers, a small evidence model, and a CLI
entry through `python -m scripts.verify_nnx_install`. Unit tests use fixture-owned fake
distributions and import specifications; they do not mutate the active Python installation.

The NNx surface suite invokes the verifier once through a session-scoped fixture before its API
tests run. This covers all entry points, including direct pytest, `make test-nnx-surface`, and the
complete `make test` collection. A local environment with an editable NNx checkout therefore fails
by default and passes only when the explicit editable mode validates that checkout.

Add `make verify-nnx-install` as the public canonical command. `make test-nnx-surface` continues to
run the focused tests; its session fixture supplies the installation gate rather than duplicating
the command in the target. The Make target, help text, and phony declaration are regression-tested.

## 12.13.7 CI installation and execution order

The existing `pytest-repository` and `pytest-nnx-surface` jobs retain their IDs, names, runners,
Python version, timeouts, cache manifests, dependency roles, and ordinary failure semantics. Their
runtime installation commands use `python -m pip` and add `--only-binary=thekaveh-nnx` when
installing `requirements.txt`. This forces pip to select a wheel for NNx without claiming that all
other dependencies are binary-only or hash-locked.

After every dependency-install command and immediately before the test workload, both jobs run:

```bash
make verify-nnx-install
```

The focused job then retains Ruff and `make test-nnx-surface`; the required repository job retains
`make test`. No later pip command may intervene between validation and the relevant tests. The
repository job makes provenance failures merge-blocking through its already-required
`pytest-repository` context, while `pytest-nnx-surface` remains the faster diagnostic signal.

The workflow contract must reject editable installs, direct NNx references, missing binary-only
selection, omitted or reordered validation, `NNX_ALLOW_EDITABLE`, `PYTHONPATH`, conditions,
services, containers, failure masking, extra steps, or a weakened test command. The verifier job's
narrow self-contract selector must run the new workflow regressions.

## 12.13.8 Regression coverage

Test-driven implementation must first demonstrate failures for the missing behavior and then cover:

- accepted canonical-wheel metadata with exact pin, owned files, and matching import origin;
- accepted editable metadata only under the exact local opt-in;
- wrong, missing, duplicate, ranged, URL, or incorrectly-extra'd manifest requirements;
- missing, duplicate, misnamed, or wrong-version installed distributions;
- any direct URL in canonical mode, and malformed or non-editable direct URLs in editable mode;
- missing `WHEEL`, `RECORD`, or `nnx/__init__.py` ownership;
- import mismatch, repository shadowing, `.pth`/`PYTHONPATH` leakage, and symlink escape;
- unknown override values and redaction of host paths and direct URLs from diagnostics; and
- mutation-resistant job step order, controls, commands, binary-only selection, and selector
  coverage.

Tests must not hard-code a repository-wide test count or inspect private `dist._path` state.

## 12.13.9 Documentation changes

Canonical documentation must describe the executable contract without duplicating ownership:

- `docs/dependency-contracts.md` replaces its private-API/manual snippet with the public Make
  command, exact evidence semantics, and editable override.
- `docs/nnx-library.md` distinguishes named-release evidence from upstream editable iteration and
  adds the canonical recovery command.
- `docs/conventions.md` and `CONTRIBUTING.md` add the local/CI verification expectations.
- `README.md` links the NNx section to the canonical contract and gives the explicit editable
  invocation without restating implementation details.
- `docs/maintenance/overnight-2026-07-04.md` marks OM-006 resolved only after live CI proves the
  contract.
- `CHANGELOG.md` records the durable verifier, workflow, and editable-mode behavior.

The design and implementation plan are manifest-declared canonical pages. Generated site and wiki
trees remain ignored derivatives. No architecture diagram change is needed because the existing
focused NNx and complete pytest nodes remain accurate.

## 12.13.10 GitFlow rollout and GitHub records

Implementation proceeds on `feature/issue-58-nnx-wheel-contract` from synchronized `develop`.
After local TDD, task reviews, broad review, and full verification, open a ready feature-to-`develop`
pull request. Require green `pytest-repository`, `atlas-consumer-policy`, `pytest-nnx-surface`, and
every applicable check; record the live canonical verifier evidence.

Merge through a separate `develop`-to-`main` pull request, then restore ancestry with a
content-neutral `main`-to-`develop` synchronization pull request if required. Do not modify the
existing required-context set: `pytest-repository` already makes this contract blocking, and
`atlas-consumer-policy` remains independently required.

After release, check all Issue #58 acceptance criteria, add PR/test/runtime evidence, close the
issue, move its project item to Done, update parent Issue #53, and verify one clean worktree, only
the two long-lived branches, no open PRs, content-equivalent synchronized branches, and no
ml-eng-lab, Atlas, JupyterHub, Ollama, or ComfyUI container. Unrelated containers remain untouched.

## 12.13.11 Acceptance criteria

The implementation is complete only when:

1. CI proves the installed distribution version equals the exact NNx manifest pin;
2. canonical mode proves the imported package is owned by a named-release wheel and rejects every
   direct/editable installation;
3. editable PEP 610 metadata fails canonical CI and succeeds locally only through the exact,
   validated opt-in;
4. tests demonstrate accepted and rejected requirement, distribution, metadata, ownership, import,
   redaction, and workflow cases with red-before-green evidence;
5. both the required complete-suite job and focused NNx job run the verifier after binary-selected
   installation and immediately before tests;
6. focused tests, complete tests, lint, repository verification, documentation checks, wiki
   generation, and live pull-request CI pass;
7. documentation is synchronized across repository, site, and wiki surfaces;
8. no NNx version, ruleset context, Atlas/runtime service, or unrelated container is changed; and
9. GitFlow promotion, issue/project evidence, branch synchronization, and temporary-state cleanup
   are complete.
