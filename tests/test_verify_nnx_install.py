from __future__ import annotations

import json
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from scripts.verify_nnx_install import VerificationError, parse_nnx_pin, verify_nnx_install


CANONICAL_FILES = ("thekaveh_nnx-0.2.0.dist-info/WHEEL", "thekaveh_nnx-0.2.0.dist-info/RECORD", "nnx/__init__.py")


class FakeDistribution:
    def __init__(self, *, name="thekaveh-nnx", version="0.2.0", root, files, direct_url=None):
        self.metadata = {"Name": name}
        self.version = version
        self.files = [Path(value) for value in files]
        self._root = root
        self._direct_url = direct_url

    def read_text(self, filename):
        if filename == "direct_url.json":
            return self._direct_url
        return None

    def locate_file(self, path):
        return self._root / path


def _spec(origin: Path | None) -> ModuleSpec | None:
    if origin is None:
        return None
    return ModuleSpec("nnx", loader=None, origin=str(origin))


@pytest.fixture
def canonical_install(tmp_path: Path):
    repo_root = tmp_path / "repository"
    install_root = tmp_path / "site-packages"
    repo_root.mkdir()
    package_init = install_root / "nnx" / "__init__.py"
    package_init.parent.mkdir(parents=True)
    package_init.write_text("", encoding="utf-8")
    distribution = FakeDistribution(root=install_root, files=CANONICAL_FILES)
    return repo_root, install_root, package_init, distribution


def _verify(requirements_text: str, canonical_install, **overrides):
    repo_root, _, package_init, distribution = canonical_install
    values = {
        "environ": {},
        "distributions": [distribution],
        "find_spec": lambda name: _spec(package_init),
        "repo_root": repo_root,
    }
    values.update(overrides)
    requirements_path = repo_root / "requirements.txt"
    requirements_path.write_text(requirements_text, encoding="utf-8")
    return verify_nnx_install(requirements_path, **values)


def test_parse_nnx_pin_accepts_the_exact_lm_pin():
    pin = parse_nnx_pin("thekaveh-nnx[lm]==0.2.0\n")

    assert pin.version == "0.2.0"
    assert pin.extras == frozenset({"lm"})


@pytest.mark.parametrize(
    "requirements_text",
    (
        "pytest\n",
        "thekaveh-nnx[lm]==0.2.0\nthekaveh_nnx[lm]==0.2.0\n",
        "thekaveh-nnx[lm]>=0.2.0\n",
        "thekaveh-nnx[lm] @ https://packages.invalid/nnx.whl\n",
        "-e git+https://packages.invalid/NNx.git#egg=thekaveh-nnx[lm]\n",
        "thekaveh-nnx==0.2.0\n",
        "thekaveh-nnx[dev]==0.2.0\n",
        "thekaveh-nnx[lm,dev]==0.2.0\n",
        "thekaveh-nnx[lm]===0.2.0\n",
        "thekaveh-nnx[lm]==not-a-version\n",
    ),
    ids=("missing", "duplicate", "range", "url", "editable", "missing-extra", "wrong-extra", "extra-extra", "operator", "malformed"),
)
def test_parse_nnx_pin_rejects_noncanonical_requirements(requirements_text: str):
    with pytest.raises(VerificationError, match="^NNx requirement is not an exact pin$"):
        parse_nnx_pin(requirements_text)


def test_verify_nnx_install_accepts_canonical_wheel_evidence(canonical_install):
    evidence = _verify("thekaveh-nnx[lm]==0.2.0\n", canonical_install)

    assert evidence.mode == "canonical-wheel"
    assert evidence.distribution == "thekaveh-nnx"
    assert evidence.version == "0.2.0"


@pytest.mark.parametrize(
    ("distributions", "files"),
    (
        ([], CANONICAL_FILES),
        (None, CANONICAL_FILES),
        ("wrong-version", CANONICAL_FILES),
        ("wrong-name", CANONICAL_FILES),
        ("direct-url", CANONICAL_FILES),
        ("canonical", tuple(path for path in CANONICAL_FILES if not path.endswith("WHEEL"))),
        ("canonical", tuple(path for path in CANONICAL_FILES if not path.endswith("RECORD"))),
        ("canonical", tuple(path for path in CANONICAL_FILES if path != "nnx/__init__.py")),
    ),
    ids=("missing", "duplicate", "wrong-version", "wrong-name", "direct-url", "missing-wheel", "missing-record", "missing-package"),
)
def test_verify_nnx_install_rejects_noncanonical_distribution_metadata(
    canonical_install, distributions, files
):
    _, install_root, _, canonical = canonical_install
    if distributions is None:
        selected = [canonical, FakeDistribution(root=install_root, files=files)]
    elif distributions == "wrong-version":
        selected = [FakeDistribution(version="0.1.9", root=install_root, files=files)]
    elif distributions == "wrong-name":
        selected = [FakeDistribution(name="thekaveh-nnx-fork", root=install_root, files=files)]
    elif distributions == "direct-url":
        selected = [
            FakeDistribution(
                root=install_root,
                files=files,
                direct_url='{"url":"file:///secret/editable/NNx","dir_info":{"editable":true}}',
            )
        ]
    elif distributions == "canonical":
        selected = [FakeDistribution(root=install_root, files=files)]
    else:
        selected = distributions

    with pytest.raises(VerificationError, match="^NNx distribution metadata is not canonical$") as error:
        _verify("thekaveh-nnx[lm]==0.2.0\n", canonical_install, distributions=selected)

    assert str(install_root) not in str(error.value)
    assert "file:///secret/editable/NNx" not in str(error.value)


def test_verify_nnx_install_rejects_import_origin_mismatch(canonical_install, tmp_path: Path):
    foreign_init = tmp_path / "shadow" / "nnx" / "__init__.py"

    with pytest.raises(VerificationError, match="^NNx import is not owned by the installed distribution$") as error:
        _verify(
            "thekaveh-nnx[lm]==0.2.0\n",
            canonical_install,
            find_spec=lambda name: _spec(foreign_init),
        )

    assert str(foreign_init) not in str(error.value)


def test_verify_nnx_install_rejects_missing_import_spec(canonical_install):
    with pytest.raises(VerificationError, match="^NNx import is not owned by the installed distribution$"):
        _verify("thekaveh-nnx[lm]==0.2.0\n", canonical_install, find_spec=lambda name: None)


def test_verify_nnx_install_rejects_repository_owned_package(canonical_install):
    repo_root, _, _, _ = canonical_install
    repository_init = repo_root / "nnx" / "__init__.py"
    repository_init.parent.mkdir()
    repository_init.write_text("", encoding="utf-8")
    distribution = FakeDistribution(root=repo_root, files=CANONICAL_FILES)

    with pytest.raises(VerificationError, match="^NNx import is not owned by the installed distribution$") as error:
        _verify(
            "thekaveh-nnx[lm]==0.2.0\n",
            canonical_install,
            distributions=[distribution],
            find_spec=lambda name: _spec(repository_init),
        )

    assert str(repo_root) not in str(error.value)


def test_verify_nnx_install_resolves_symlinks_before_repository_ownership_check(canonical_install):
    repo_root, install_root, _, _ = canonical_install
    repository_package = repo_root / "shadowed-nnx"
    repository_package.mkdir()
    repository_init = repository_package / "__init__.py"
    repository_init.write_text("", encoding="utf-8")
    (install_root / "nnx").unlink() if (install_root / "nnx").is_symlink() else None
    if (install_root / "nnx").exists():
        (install_root / "nnx" / "__init__.py").unlink()
        (install_root / "nnx").rmdir()
    (install_root / "nnx").symlink_to(repository_package, target_is_directory=True)
    distribution = FakeDistribution(root=install_root, files=CANONICAL_FILES)

    with pytest.raises(VerificationError, match="^NNx import is not owned by the installed distribution$") as error:
        _verify(
            "thekaveh-nnx[lm]==0.2.0\n",
            canonical_install,
            distributions=[distribution],
            find_spec=lambda name: _spec(install_root / "nnx" / "__init__.py"),
        )

    assert str(repo_root) not in str(error.value)


def _editable_distribution(source_root: Path, *, version: str = "0.2.0", direct_url=None):
    metadata_text = direct_url
    if metadata_text is None:
        metadata_text = json.dumps({"dir_info": {"editable": True}, "url": source_root.as_uri()})
    return FakeDistribution(version=version, root=source_root, files=(), direct_url=metadata_text)


def test_verify_nnx_install_accepts_validated_editable_development_install(canonical_install, tmp_path: Path):
    source_root = tmp_path / "NNx source"
    source_init = source_root / "nnx" / "__init__.py"
    source_init.parent.mkdir(parents=True)
    source_init.write_text("", encoding="utf-8")

    evidence = _verify(
        "thekaveh-nnx[lm]==0.2.0\n",
        canonical_install,
        environ={"NNX_ALLOW_EDITABLE": "1"},
        distributions=[_editable_distribution(source_root)],
        find_spec=lambda name: _spec(source_init),
    )

    assert evidence.mode == "editable-development"
    assert evidence.distribution == "thekaveh-nnx"
    assert evidence.version == "0.2.0"


def test_editable_mode_ignores_noneditable_candidate_at_another_version(canonical_install, tmp_path: Path):
    _, install_root, _, _ = canonical_install
    source_root = tmp_path / "NNx"
    source_init = source_root / "nnx" / "__init__.py"
    source_init.parent.mkdir(parents=True)
    source_init.write_text("", encoding="utf-8")
    stale_wheel = FakeDistribution(version="0.2.2", root=install_root, files=CANONICAL_FILES)

    evidence = _verify(
        "thekaveh-nnx[lm]==0.2.0\n",
        canonical_install,
        environ={"NNX_ALLOW_EDITABLE": "1"},
        distributions=[_editable_distribution(source_root), stale_wheel],
        find_spec=lambda name: _spec(source_init),
    )

    assert evidence.mode == "editable-development"
    assert evidence.version == "0.2.0"


@pytest.mark.parametrize("competitor_kind", ("same-version-wheel", "second-editable"))
def test_editable_mode_rejects_competing_pinned_distribution(
    canonical_install, tmp_path: Path, competitor_kind: str
):
    _, install_root, _, _ = canonical_install
    source_root = tmp_path / "NNx"
    source_init = source_root / "nnx" / "__init__.py"
    source_init.parent.mkdir(parents=True)
    source_init.write_text("", encoding="utf-8")
    if competitor_kind == "second-editable":
        other_source = tmp_path / "other-NNx"
        other_source.mkdir()
        competitor = _editable_distribution(other_source)
    else:
        competitor = FakeDistribution(root=install_root, files=CANONICAL_FILES)

    with pytest.raises(VerificationError, match="^NNx distribution metadata is not canonical$"):
        _verify(
            "thekaveh-nnx[lm]==0.2.0\n",
            canonical_install,
            environ={"NNX_ALLOW_EDITABLE": "1"},
            distributions=[_editable_distribution(source_root), competitor],
            find_spec=lambda name: _spec(source_init),
        )


@pytest.mark.parametrize("override", ("0", "true", "yes", " 1", "1 "))
def test_verify_nnx_install_rejects_unknown_editable_override(canonical_install, override: str):
    with pytest.raises(VerificationError, match="^NNx editable override is invalid$"):
        _verify("thekaveh-nnx[lm]==0.2.0\n", canonical_install, environ={"NNX_ALLOW_EDITABLE": override})


@pytest.mark.parametrize(
    "direct_url",
    (
        "not-json",
        "[]",
        '{}',
        '{"dir_info":{},"url":"file:///secret/NNx"}',
        '{"dir_info":{"editable":"true"},"url":"file:///secret/NNx"}',
        '{"dir_info":{"editable":false},"url":"file:///secret/NNx"}',
        '{"dir_info":{"editable":true},"url":"https://packages.invalid/NNx"}',
        '{"archive_info":{},"url":"file:///secret/nnx.whl"}',
        '{"vcs_info":{"vcs":"git"},"url":"git+https://packages.invalid/NNx.git"}',
        '{"dir_info":{"editable":true},"vcs_info":{"vcs":"git"},"url":"file:///secret/NNx"}',
    ),
    ids=(
        "malformed-json",
        "non-object",
        "missing-fields",
        "missing-editable",
        "non-boolean-editable",
        "non-editable-directory",
        "non-file-url",
        "archive",
        "vcs",
        "mixed-vcs-directory",
    ),
)
def test_verify_nnx_install_rejects_invalid_editable_metadata(
    canonical_install, tmp_path: Path, direct_url: str
):
    source_root = tmp_path / "declared-source"
    source_init = source_root / "nnx" / "__init__.py"
    source_init.parent.mkdir(parents=True)
    source_init.write_text("", encoding="utf-8")

    with pytest.raises(VerificationError, match="^NNx editable metadata is not valid$") as error:
        _verify(
            "thekaveh-nnx[lm]==0.2.0\n",
            canonical_install,
            environ={"NNX_ALLOW_EDITABLE": "1"},
            distributions=[_editable_distribution(source_root, direct_url=direct_url)],
            find_spec=lambda name: _spec(source_init),
        )

    assert direct_url not in str(error.value)
    assert "/secret/" not in str(error.value)
    assert "packages.invalid" not in str(error.value)


def test_verify_nnx_install_rejects_editable_import_outside_declared_source(canonical_install, tmp_path: Path):
    source_root = tmp_path / "declared-source"
    source_root.mkdir()
    foreign_init = tmp_path / "shadow" / "nnx" / "__init__.py"
    foreign_init.parent.mkdir(parents=True)
    foreign_init.write_text("", encoding="utf-8")

    with pytest.raises(VerificationError, match="^NNx import is not owned by the installed distribution$") as error:
        _verify(
            "thekaveh-nnx[lm]==0.2.0\n",
            canonical_install,
            environ={"NNX_ALLOW_EDITABLE": "1"},
            distributions=[_editable_distribution(source_root)],
            find_spec=lambda name: _spec(foreign_init),
        )

    assert str(source_root) not in str(error.value)
    assert str(foreign_init) not in str(error.value)


@pytest.mark.parametrize(
    ("requirements_text", "version", "message"),
    (
        ("thekaveh-nnx[lm]>=0.2.0\n", "0.2.0", "NNx requirement is not an exact pin"),
        ("thekaveh-nnx[lm]==0.2.0\n", "0.1.9", "NNx distribution metadata is not canonical"),
    ),
    ids=("ranged-pin", "wrong-installed-version"),
)
def test_editable_mode_still_requires_exact_pin_and_installed_version(
    canonical_install, tmp_path: Path, requirements_text: str, version: str, message: str
):
    source_root = tmp_path / "NNx"
    source_init = source_root / "nnx" / "__init__.py"
    source_init.parent.mkdir(parents=True)
    source_init.write_text("", encoding="utf-8")

    with pytest.raises(VerificationError, match=f"^{message}$"):
        _verify(
            requirements_text,
            canonical_install,
            environ={"NNX_ALLOW_EDITABLE": "1"},
            distributions=[_editable_distribution(source_root, version=version)],
            find_spec=lambda name: _spec(source_init),
        )
