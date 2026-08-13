from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.advisory_baseline import (
    AcceptedAdvisory,
    AdvisoryBaselineError,
    Baseline,
    compare_baseline,
    load_baseline,
    main,
    normalize_pip_audit,
    run_audit_surfaces,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def _canonical_document(*, entries: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "audited_surfaces": ["combined-runtime", "torch", "documentation", "atlas-contract"],
        "accepted_advisories": entries
        if entries is not None
        else [
            {
                "advisory_id": "CVE-2025-2148",
                "package": "torch",
                "accepted_version": "2.4.1",
                "surfaces": ["combined-runtime", "torch"],
            }
        ],
    }


def _write_policy(path: Path, document: dict[str, object]) -> None:
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def test_load_baseline_accepts_canonical_policy(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    _write_policy(path, _canonical_document())

    baseline = load_baseline(path)

    assert baseline.schema_version == 1
    assert baseline.audited_surfaces == ("combined-runtime", "torch", "documentation", "atlas-contract")
    assert [(item.package, item.advisory_id, item.accepted_version, item.surfaces) for item in baseline.accepted_advisories] == [
        ("torch", "CVE-2025-2148", "2.4.1", ("combined-runtime", "torch"))
    ]


@pytest.mark.parametrize(
    "document",
    [
        {**_canonical_document(), "extra": True},
        _canonical_document(entries=[{**_canonical_document()["accepted_advisories"][0], "extra": True}]),
    ],
)
def test_load_baseline_rejects_unknown_root_and_item_keys(tmp_path: Path, document: dict[str, object]) -> None:
    path = tmp_path / "baseline.json"
    _write_policy(path, document)

    with pytest.raises(AdvisoryBaselineError, match="unknown"):
        load_baseline(path)


@pytest.mark.parametrize(
    "document",
    [
        {
            "audited_surfaces": _canonical_document()["audited_surfaces"],
            "schema_version": 1,
            "accepted_advisories": _canonical_document()["accepted_advisories"],
        },
        _canonical_document(
            entries=[
                {
                    "package": "torch",
                    "advisory_id": "CVE-2025-2148",
                    "accepted_version": "2.4.1",
                    "surfaces": ["combined-runtime", "torch"],
                }
            ]
        ),
    ],
)
def test_load_baseline_rejects_reordered_root_and_item_keys(tmp_path: Path, document: dict[str, object]) -> None:
    path = tmp_path / "baseline.json"
    _write_policy(path, document)

    with pytest.raises(AdvisoryBaselineError, match="key order"):
        load_baseline(path)


def test_load_baseline_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    document = _canonical_document()
    document["schema_version"] = 2
    _write_policy(path, document)

    with pytest.raises(AdvisoryBaselineError, match="schema version"):
        load_baseline(path)


@pytest.mark.parametrize(
    "surfaces",
    [
        ["combined-runtime", "torch", "documentation"],
        ["combined-runtime", "torch", "documentation", "atlas-contract", "extra"],
        ["combined-runtime", "torch", "torch", "atlas-contract"],
        ["torch", "combined-runtime", "documentation", "atlas-contract"],
    ],
)
def test_load_baseline_rejects_missing_extra_duplicate_or_reordered_surfaces(tmp_path: Path, surfaces: list[str]) -> None:
    path = tmp_path / "baseline.json"
    document = _canonical_document()
    document["audited_surfaces"] = surfaces
    _write_policy(path, document)

    with pytest.raises(AdvisoryBaselineError, match="audited surfaces"):
        load_baseline(path)


def test_load_baseline_rejects_duplicate_identity_after_pep503_normalization(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    entry = _canonical_document()["accepted_advisories"][0]
    assert isinstance(entry, dict)
    _write_policy(path, _canonical_document(entries=[entry, {**entry, "package": "Torch"}]))

    with pytest.raises(AdvisoryBaselineError, match="duplicate"):
        load_baseline(path)


def test_load_baseline_rejects_lone_noncanonical_package_spelling(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    entry = _canonical_document()["accepted_advisories"][0]
    assert isinstance(entry, dict)
    _write_policy(path, _canonical_document(entries=[{**entry, "package": "Torch"}]))

    with pytest.raises(AdvisoryBaselineError, match="canonical package"):
        load_baseline(path)


@pytest.mark.parametrize(
    "surfaces",
    [["combined-runtime", "unknown"], ["combined-runtime", "combined-runtime"]],
)
def test_load_baseline_rejects_unknown_or_duplicate_item_surfaces(tmp_path: Path, surfaces: list[str]) -> None:
    path = tmp_path / "baseline.json"
    entry = _canonical_document()["accepted_advisories"][0]
    assert isinstance(entry, dict)
    _write_policy(path, _canonical_document(entries=[{**entry, "surfaces": surfaces}]))

    with pytest.raises(AdvisoryBaselineError, match="surfaces"):
        load_baseline(path)


def test_load_baseline_rejects_unsorted_items_and_noncanonical_serialization(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    entries = [
        {
            "advisory_id": "CVE-2025-2149",
            "package": "torch",
            "accepted_version": "2.4.1",
            "surfaces": ["combined-runtime", "torch"],
        },
        {
            "advisory_id": "CVE-2025-2148",
            "package": "torch",
            "accepted_version": "2.4.1",
            "surfaces": ["combined-runtime", "torch"],
        },
    ]
    _write_policy(path, _canonical_document(entries=entries))

    with pytest.raises(AdvisoryBaselineError, match="sorted"):
        load_baseline(path)

    _write_policy(path, _canonical_document())
    path.write_text(path.read_text(encoding="utf-8").replace("\n", "\r\n"), encoding="utf-8", newline="")
    with pytest.raises(AdvisoryBaselineError, match="canonical"):
        load_baseline(path)


def test_real_baseline_contains_21_unique_reviewed_identities() -> None:
    baseline = load_baseline(REPO_ROOT / "security/accepted-advisories.json")

    assert {(item.package, item.advisory_id, item.accepted_version) for item in baseline.accepted_advisories} == {
        ("pytorch-lightning", "PYSEC-2026-3043", "2.4.0"),
        ("torch", "CVE-2025-2148", "2.4.1"),
        ("torch", "CVE-2025-2149", "2.4.1"),
        ("torch", "CVE-2025-2998", "2.4.1"),
        ("torch", "CVE-2025-2999", "2.4.1"),
        ("torch", "CVE-2025-3001", "2.4.1"),
        ("torch", "PYSEC-2024-259", "2.4.1"),
        ("torch", "PYSEC-2025-191", "2.4.1"),
        ("torch", "PYSEC-2025-194", "2.4.1"),
        ("torch", "PYSEC-2025-198", "2.4.1"),
        ("torch", "PYSEC-2025-203", "2.4.1"),
        ("torch", "PYSEC-2025-204", "2.4.1"),
        ("torch", "PYSEC-2025-205", "2.4.1"),
        ("torch", "PYSEC-2025-206", "2.4.1"),
        ("torch", "PYSEC-2025-207", "2.4.1"),
        ("torch", "PYSEC-2025-208", "2.4.1"),
        ("torch", "PYSEC-2025-209", "2.4.1"),
        ("torch", "PYSEC-2025-41", "2.4.1"),
        ("torch", "PYSEC-2026-139", "2.4.1"),
        ("torch", "PYSEC-2026-1970", "2.4.1"),
        ("torch", "PYSEC-2026-2286", "2.4.1"),
    }


def test_real_baseline_contains_exact_reviewed_policy_quadruples() -> None:
    baseline = load_baseline(REPO_ROOT / "security/accepted-advisories.json")

    assert {
        (item.package, item.advisory_id, item.accepted_version, item.surfaces)
        for item in baseline.accepted_advisories
    } == {
        ("pytorch-lightning", "PYSEC-2026-3043", "2.4.0", ("combined-runtime", "torch")),
        ("torch", "CVE-2025-2148", "2.4.1", ("combined-runtime", "torch")),
        ("torch", "CVE-2025-2149", "2.4.1", ("combined-runtime", "torch")),
        ("torch", "CVE-2025-2998", "2.4.1", ("combined-runtime", "torch")),
        ("torch", "CVE-2025-2999", "2.4.1", ("combined-runtime", "torch")),
        ("torch", "CVE-2025-3001", "2.4.1", ("combined-runtime", "torch")),
        ("torch", "PYSEC-2024-259", "2.4.1", ("combined-runtime", "torch")),
        ("torch", "PYSEC-2025-191", "2.4.1", ("combined-runtime", "torch")),
        ("torch", "PYSEC-2025-194", "2.4.1", ("combined-runtime", "torch")),
        ("torch", "PYSEC-2025-198", "2.4.1", ("combined-runtime", "torch")),
        ("torch", "PYSEC-2025-203", "2.4.1", ("combined-runtime", "torch")),
        ("torch", "PYSEC-2025-204", "2.4.1", ("combined-runtime", "torch")),
        ("torch", "PYSEC-2025-205", "2.4.1", ("combined-runtime", "torch")),
        ("torch", "PYSEC-2025-206", "2.4.1", ("combined-runtime", "torch")),
        ("torch", "PYSEC-2025-207", "2.4.1", ("combined-runtime", "torch")),
        ("torch", "PYSEC-2025-208", "2.4.1", ("combined-runtime", "torch")),
        ("torch", "PYSEC-2025-209", "2.4.1", ("combined-runtime", "torch")),
        ("torch", "PYSEC-2025-41", "2.4.1", ("combined-runtime", "torch")),
        ("torch", "PYSEC-2026-139", "2.4.1", ("combined-runtime", "torch")),
        ("torch", "PYSEC-2026-1970", "2.4.1", ("combined-runtime", "torch")),
        ("torch", "PYSEC-2026-2286", "2.4.1", ("combined-runtime", "torch")),
    }


def _payload(*dependencies: dict[str, object]) -> list[dict[str, object]]:
    return list(dependencies)


def _dependency(name: str, version: str, *vulnerability_ids: str) -> dict[str, object]:
    return {
        "name": name,
        "version": version,
        "vulns": [
            {"id": vulnerability_id, "aliases": ["CVE-elsewhere"], "fix_versions": ["9.9.9"]}
            for vulnerability_id in vulnerability_ids
        ],
    }


def _baseline() -> Baseline:
    return Baseline(
        schema_version=1,
        audited_surfaces=("combined-runtime", "torch", "documentation", "atlas-contract"),
        accepted_advisories=(
            AcceptedAdvisory("torch", "CVE-2025-2148", "2.4.1", ("combined-runtime", "torch")),
        ),
    )


def _observations(*surface_payloads: tuple[str, object]):
    return tuple(normalize_pip_audit(surface, payload) for surface, payload in surface_payloads)


def _complete_observations(*, combined: object, torch: object) -> tuple:
    return _observations(
        ("combined-runtime", combined),
        ("torch", torch),
        ("documentation", _payload()),
        ("atlas-contract", _payload()),
    )


def test_normalize_accepts_clean_and_vulnerable_pip_audit_payloads() -> None:
    clean = normalize_pip_audit("documentation", _payload(_dependency("mkdocs", "9.0")))
    vulnerable = normalize_pip_audit(
        "torch",
        _payload(_dependency("Torch", "2.4.1", "CVE-2025-2148")),
    )

    assert clean.resolved_versions == (("mkdocs", "9.0"),)
    assert clean.advisories == ()
    assert vulnerable.resolved_versions == (("torch", "2.4.1"),)
    assert vulnerable.advisories == (("torch", "2.4.1", "CVE-2025-2148"),)


def test_normalize_collapses_duplicate_raw_records_to_one_identity() -> None:
    observation = normalize_pip_audit(
        "torch",
        _payload(
            _dependency("torch", "2.4.1", "PYSEC-2025-191"),
            _dependency("torch", "2.4.1", "PYSEC-2025-191"),
        ),
    )

    assert observation.advisories == (("torch", "2.4.1", "PYSEC-2025-191"),)


@pytest.mark.parametrize(
    "surface,payload",
    [
        ("unknown", _payload()),
        ("torch", {"not": "a list"}),
        ("torch", _payload({"name": "torch", "version": "2.4.1"})),
        ("torch", _payload({"name": "torch", "version": "2.4.1", "vulns": [{"id": 1}]})),
        (
            "torch",
            _payload(
                {"name": "torch", "version": "2.4.1", "vulns": [{"id": "CVE-2025-2148", "aliases": "bad"}]}
            ),
        ),
        ("torch", _payload(_dependency("torch", "2.4.1"), _dependency("torch", "2.5.0"))),
    ],
)
def test_normalize_rejects_malformed_payload_and_dependency_records(surface: str, payload: object) -> None:
    with pytest.raises(AdvisoryBaselineError):
        normalize_pip_audit(surface, payload)


def test_compare_accepts_exact_observations() -> None:
    result = compare_baseline(
        _baseline(),
        _complete_observations(
            combined=_payload(_dependency("torch", "2.4.1", "CVE-2025-2148")),
            torch=_payload(_dependency("torch", "2.4.1", "CVE-2025-2148")),
        ),
    )

    assert result.errors == ()
    assert result.notices == ()


def test_compare_fails_on_new_primary_id() -> None:
    result = compare_baseline(
        _baseline(),
        _complete_observations(
            combined=_payload(_dependency("torch", "2.4.1", "CVE-2025-2148", "CVE-2025-9999")),
            torch=_payload(_dependency("torch", "2.4.1", "CVE-2025-2148")),
        ),
    )

    assert result.errors == ("new advisory: torch 2.4.1 CVE-2025-9999 on combined-runtime",)


@pytest.mark.parametrize(
    "combined,torch,expected",
    [
        (
            _payload(_dependency("other", "2.4.1", "CVE-2025-2148")),
            _payload(_dependency("torch", "2.4.1", "CVE-2025-2148")),
            "new advisory: other 2.4.1 CVE-2025-2148 on combined-runtime",
        ),
        (
            _payload(_dependency("torch", "2.4.1", "CVE-2025-2148")),
            _payload(),
            "surface drift: torch 2.4.1 CVE-2025-2148 expected [combined-runtime, torch]; observed [combined-runtime]",
        ),
    ],
)
def test_compare_fails_on_package_or_surface_drift(combined: object, torch: object, expected: str) -> None:
    result = compare_baseline(_baseline(), _complete_observations(combined=combined, torch=torch))

    assert expected in result.errors


def test_compare_fails_on_accepted_version_drift_with_advisory_present() -> None:
    result = compare_baseline(
        _baseline(),
        _complete_observations(
            combined=_payload(_dependency("torch", "2.5.0", "CVE-2025-2148")),
            torch=_payload(_dependency("torch", "2.5.0", "CVE-2025-2148")),
        ),
    )

    assert result.errors == ("accepted version drift: torch expected 2.4.1; observed 2.5.0",)


def test_compare_fails_on_accepted_version_drift_with_empty_vulnerability_list() -> None:
    result = compare_baseline(
        _baseline(),
        _complete_observations(
            combined=_payload(_dependency("torch", "2.5.0")),
            torch=_payload(_dependency("torch", "2.5.0")),
        ),
    )

    assert result.errors == ("accepted version drift: torch expected 2.4.1; observed 2.5.0",)


def test_compare_rejects_cross_product_advisory_versions_even_when_surfaces_match() -> None:
    baseline = Baseline(
        schema_version=1,
        audited_surfaces=("combined-runtime", "torch", "documentation", "atlas-contract"),
        accepted_advisories=(
            AcceptedAdvisory("torch", "ADVISORY-A", "1.0", ("combined-runtime",)),
            AcceptedAdvisory("torch", "ADVISORY-B", "2.0", ("torch",)),
        ),
    )
    observations = _complete_observations(
        combined=_payload(_dependency("torch", "2.0", "ADVISORY-A")),
        torch=_payload(_dependency("torch", "2.0", "ADVISORY-B")),
    )

    result = compare_baseline(baseline, observations)

    assert result.errors == ("accepted version drift: torch expected 1.0; observed 2.0",)
    assert result.notices == ("reconcile removed advisory: torch 1.0 ADVISORY-A",)


def _versioned_identity_baseline() -> Baseline:
    return Baseline(
        schema_version=1,
        audited_surfaces=("combined-runtime", "torch", "documentation", "atlas-contract"),
        accepted_advisories=(
            AcceptedAdvisory("torch", "ADVISORY", "1.0", ("combined-runtime",)),
            AcceptedAdvisory("torch", "ADVISORY", "2.0", ("torch",)),
        ),
    )


def test_compare_accepts_same_primary_id_at_two_exact_accepted_versions() -> None:
    result = compare_baseline(
        _versioned_identity_baseline(),
        _complete_observations(
            combined=_payload(_dependency("torch", "1.0", "ADVISORY")),
            torch=_payload(_dependency("torch", "2.0", "ADVISORY")),
        ),
    )

    assert result.errors == ()
    assert result.notices == ()


def test_compare_reports_only_the_removed_versioned_identity() -> None:
    result = compare_baseline(
        _versioned_identity_baseline(),
        _complete_observations(
            combined=_payload(),
            torch=_payload(_dependency("torch", "2.0", "ADVISORY")),
        ),
    )

    assert result.errors == ()
    assert result.notices == ("reconcile removed advisory: torch 1.0 ADVISORY",)


def test_compare_fails_when_a_versioned_identity_moves_to_the_wrong_surface() -> None:
    result = compare_baseline(
        _versioned_identity_baseline(),
        _observations(
            ("combined-runtime", _payload()),
            ("torch", _payload(_dependency("torch", "2.0", "ADVISORY"))),
            ("documentation", _payload(_dependency("torch", "1.0", "ADVISORY"))),
            ("atlas-contract", _payload()),
        ),
    )

    assert result.errors == (
        "surface drift: torch 1.0 ADVISORY expected [combined-runtime]; observed [documentation]",
    )
    assert result.notices == ()


def test_compare_reports_removed_identity_without_failing() -> None:
    result = compare_baseline(
        _baseline(),
        _complete_observations(
            combined=_payload(_dependency("torch", "2.4.1")),
            torch=_payload(_dependency("torch", "2.4.1")),
        ),
    )

    assert result.errors == ()
    assert result.notices == ("reconcile removed advisory: torch 2.4.1 CVE-2025-2148",)


def test_compare_reports_removed_package_without_failing() -> None:
    result = compare_baseline(
        _baseline(),
        _complete_observations(combined=_payload(), torch=_payload()),
    )

    assert result.errors == ()
    assert result.notices == ("reconcile removed advisory: torch 2.4.1 CVE-2025-2148",)


def test_alias_only_and_fix_version_changes_do_not_change_policy_identity() -> None:
    payload = _payload(
        {
            "name": "torch",
            "version": "2.4.1",
            "vulns": [
                {
                    "id": "CVE-2025-2148",
                    "aliases": ["updated-alias"],
                    "fix_versions": ["100.0.0"],
                }
            ],
        }
    )
    result = compare_baseline(
        _baseline(),
        _complete_observations(combined=payload, torch=payload),
    )

    assert result.errors == ()
    assert result.notices == ()


def _audit_runner(*, returncode: int = 0, payload: object = None, write_output: bool = True):
    commands: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append((tuple(command), kwargs))
        if write_output:
            output = Path(command[command.index("--output") + 1])
            output.write_text(json.dumps([] if payload is None else payload), encoding="utf-8")
        return SimpleNamespace(returncode=returncode, stdout="", stderr="")

    return commands, run


def _assert_exact_audit_commands(commands: list[tuple[tuple[str, ...], dict[str, object]]], repo: Path) -> None:
    expected_inputs = (
        ("combined-runtime.json", ("-r", "requirements.txt", "-r", "torch-requirements.txt")),
        ("torch.json", ("-r", "torch-requirements.txt")),
        ("documentation.json", ("--disable-pip", "-r", "docs-requirements.txt")),
        ("atlas-contract.json", ("-r", "atlas-contract-requirements.txt")),
    )
    fixed_flags = (
        "--strict",
        "--vulnerability-service",
        "pypi",
        "--format",
        "json",
        "--aliases",
        "on",
        "--desc",
        "off",
        "--progress-spinner",
        "off",
    )
    assert len(commands) == len(expected_inputs)
    output_directories = set()
    for (command, kwargs), (output_name, inputs) in zip(commands, expected_inputs, strict=True):
        output = Path(command[-1])
        assert command == (sys.executable, "-m", "pip_audit", *inputs, *fixed_flags, "--output", str(output))
        assert output.name == output_name
        assert output.parent != repo
        output_directories.add(output.parent)
        assert kwargs == {"cwd": repo, "check": False, "capture_output": True, "text": True}
    assert len(output_directories) == 1


def test_audit_commands_match_the_complete_four_surface_contract(tmp_path: Path) -> None:
    commands, runner = _audit_runner()

    run_audit_surfaces(tmp_path, runner=runner)

    _assert_exact_audit_commands(commands, tmp_path)


@pytest.mark.parametrize(
    "command_index,mutation",
    [
        (0, lambda command: command.__setitem__(command.index("off", command.index("--progress-spinner")), "on")),
        (1, lambda command: command.__setitem__(command.index("torch-requirements.txt"), "requirements.txt")),
        (3, lambda command: command.__setitem__(command.index("atlas-contract-requirements.txt"), "requirements.txt")),
        (0, lambda command: command.remove("--strict")),
        (0, lambda command: command.insert(command.index("--format"), "--ignore-vuln")),
        (
            0,
            lambda command: command.__setitem__(
                slice(command.index("--format"), command.index("--format") + 2),
                ["json", "--format"],
            ),
        ),
    ],
)
def test_exact_audit_command_contract_rejects_weakened_or_reordered_tokens(
    tmp_path: Path,
    command_index: int,
    mutation,
) -> None:
    commands, runner = _audit_runner()
    run_audit_surfaces(tmp_path, runner=runner)
    mutated = list(commands)
    command, kwargs = mutated[command_index]
    command_list = list(command)
    mutation(command_list)
    mutated[command_index] = (tuple(command_list), kwargs)

    with pytest.raises(AssertionError):
        _assert_exact_audit_commands(mutated, tmp_path)


def test_audit_commands_cover_exact_four_surfaces_and_flags(tmp_path: Path) -> None:
    commands, runner = _audit_runner()

    observations = run_audit_surfaces(tmp_path, runner=runner)

    assert [observation.surface for observation in observations] == [
        "combined-runtime",
        "torch",
        "documentation",
        "atlas-contract",
    ]
    assert [command[:3] for command, _ in commands] == [(sys.executable, "-m", "pip_audit")] * 4
    assert [("-r", "requirements.txt", "-r", "torch-requirements.txt") == command[3:7] for command, _ in commands] == [
        True,
        False,
        False,
        False,
    ]
    for command, kwargs in commands:
        assert kwargs["cwd"] == tmp_path
        assert kwargs["check"] is False
        assert {"--strict", "--vulnerability-service", "pypi", "--format", "json"} <= set(command)
        assert {"--aliases", "on", "--desc", "off", "--progress-spinner"} <= set(command)
        assert "--ignore-vuln" not in command


def test_documentation_audit_uses_disable_pip(tmp_path: Path) -> None:
    commands, runner = _audit_runner()

    run_audit_surfaces(tmp_path, runner=runner)

    documentation = commands[2][0]
    assert documentation[3:6] == ("--disable-pip", "-r", "docs-requirements.txt")


@pytest.mark.parametrize("returncode", [0, 1])
def test_audit_exit_zero_and_one_are_completed_observations(tmp_path: Path, returncode: int) -> None:
    _, runner = _audit_runner(returncode=returncode, payload=_payload(_dependency("torch", "2.4.1")))

    observations = run_audit_surfaces(tmp_path, runner=runner)

    assert len(observations) == 4
    assert all(observation.resolved_versions == (("torch", "2.4.1"),) for observation in observations)


@pytest.mark.parametrize(
    "runner",
    [
        _audit_runner(returncode=2)[1],
        _audit_runner(write_output=False)[1],
        _audit_runner(payload="not-json")[1],
    ],
)
def test_audit_other_exit_missing_output_and_malformed_json_fail_closed(tmp_path: Path, runner) -> None:
    with pytest.raises(AdvisoryBaselineError):
        run_audit_surfaces(tmp_path, runner=runner)


def test_cli_returns_nonzero_for_comparison_errors_and_zero_for_notices(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.advisory_baseline as module

    monkeypatch.setattr(module, "load_baseline", lambda path: _baseline())
    monkeypatch.setattr(
        module,
        "run_audit_surfaces",
        lambda repo: _complete_observations(
            combined=_payload(_dependency("torch", "2.4.1", "CVE-2025-9999")),
            torch=_payload(_dependency("torch", "2.4.1")),
        ),
    )
    assert main(["--repo-root", "."]) == 1

    monkeypatch.setattr(
        module,
        "run_audit_surfaces",
        lambda repo: _complete_observations(combined=_payload(), torch=_payload()),
    )
    assert main(["--repo-root", "."]) == 0


def test_cli_has_no_accept_or_in_place_write_option(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--help"])

    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--accept" not in help_text
    assert "--write" not in help_text


def test_cli_diagnostics_do_not_disclose_index_credentials_or_temp_paths(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import scripts.advisory_baseline as module

    monkeypatch.setattr(
        module,
        "load_baseline",
        lambda path: (_ for _ in ()).throw(
            AdvisoryBaselineError("https://user:secret@example.test /private/tmp/advisory.json")
        ),
    )

    assert main(["--repo-root", "."]) == 1
    output = capsys.readouterr().err
    assert "secret" not in output
    assert "/private/tmp" not in output
