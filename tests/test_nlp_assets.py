from __future__ import annotations

import hashlib
import io
import os
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import pytest

from scripts import nlp_assets


REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "requirements" / "nlp-assets.toml"
EXPECTED = nlp_assets.VaderAsset(
    url=(
        "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/"
        "packages/sentiment/vader_lexicon.zip"
    ),
    sha256="8adba4294eef3964d820bf655e37e61bdc3a341994356af59b74fb3b4a36ce5c",
    size=90486,
    resource=PurePosixPath("sentiment/vader_lexicon.zip"),
    member=PurePosixPath("vader_lexicon/vader_lexicon.txt"),
    license="MIT License",
)


def _zip_bytes(
    *,
    member: str = "vader_lexicon/vader_lexicon.txt",
    content: bytes = b"good\t2.0\nbad\t-2.0",
    extra_member: str | None = None,
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, content)
        if extra_member is not None:
            archive.writestr(extra_member, b"rogue")
    return output.getvalue()


def _asset_for(data: bytes, *, member: str = "vader_lexicon/vader_lexicon.txt"):
    return nlp_assets.VaderAsset(
        url=EXPECTED.url,
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
        resource=EXPECTED.resource,
        member=PurePosixPath(member),
        license=EXPECTED.license,
    )


def _write_target(data_dir: Path, data: bytes) -> Path:
    target = data_dir / "sentiment" / "vader_lexicon.zip"
    target.parent.mkdir(parents=True)
    target.write_bytes(data)
    return target


def _manifest_text(**changes: object) -> str:
    values: dict[str, object] = {
        "schema_version": 1,
        "url": EXPECTED.url,
        "sha256": EXPECTED.sha256,
        "size": EXPECTED.size,
        "resource": str(EXPECTED.resource),
        "member": str(EXPECTED.member),
        "license": EXPECTED.license,
    }
    values.update(changes)
    return (
        f"schema_version = {values['schema_version']!r}\n\n"
        "[vader_lexicon]\n"
        f"url = {values['url']!r}\n"
        f"sha256 = {values['sha256']!r}\n"
        f"size = {values['size']!r}\n"
        f"resource = {values['resource']!r}\n"
        f"member = {values['member']!r}\n"
        f"license = {values['license']!r}\n"
    ).replace("'", '"')


def test_real_manifest_is_the_exact_reviewed_identity() -> None:
    assert nlp_assets.load_manifest(MANIFEST) == EXPECTED


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("url", "http://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/sentiment/vader_lexicon.zip"),
        ("url", "https://user:secret@raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/sentiment/vader_lexicon.zip"),
        ("url", "https://raw.githubusercontent.com:444/nltk/nltk_data/gh-pages/packages/sentiment/vader_lexicon.zip"),
        ("url", EXPECTED.url + "?token=secret"),
        ("url", EXPECTED.url + "#fragment"),
        ("url", "https://example.com/vader_lexicon.zip"),
        ("url", EXPECTED.url.replace("gh-pages", "main")),
        ("sha256", "A" * 64),
        ("sha256", "0" * 63),
        ("sha256", "z" * 64),
        ("size", True),
        ("size", 0),
        ("size", -1),
        ("resource", "/sentiment/vader_lexicon.zip"),
        ("resource", "sentiment/../vader_lexicon.zip"),
        ("resource", "sentiment\\vader_lexicon.zip"),
        ("member", "/vader_lexicon/vader_lexicon.txt"),
        ("member", "vader_lexicon/../vader_lexicon.txt"),
        ("member", "vader_lexicon\\vader_lexicon.txt"),
        ("license", "unknown"),
    ],
)
def test_manifest_rejects_wrong_identity_without_leaking_input(
    tmp_path: Path, key: str, value: object
) -> None:
    manifest = tmp_path / "asset.toml"
    manifest.write_text(_manifest_text(**{key: value}), encoding="utf-8")

    with pytest.raises(nlp_assets.NLPAssetError) as caught:
        nlp_assets.load_manifest(manifest)

    assert caught.value.category == "manifest"
    assert str(caught.value) == "manifest"
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize(
    "contents",
    [
        "schema_version = 1\n",
        _manifest_text() + "extra = 1\n",
        _manifest_text().replace("url =", "rogue = \"x\"\nurl =", 1),
        _manifest_text().replace("[vader_lexicon]", "[vader_lexicon]\n[vader_lexicon]"),
        "not = [valid",
    ],
)
def test_manifest_rejects_missing_extra_duplicate_or_malformed_data(
    tmp_path: Path, contents: str
) -> None:
    manifest = tmp_path / "asset.toml"
    manifest.write_text(contents, encoding="utf-8")
    with pytest.raises(nlp_assets.NLPAssetError, match="^manifest$"):
        nlp_assets.load_manifest(manifest)


def test_verify_accepts_exact_zip_and_runs_smoke(tmp_path: Path) -> None:
    data = _zip_bytes()
    asset = _asset_for(data)
    target = _write_target(tmp_path, data)
    assert nlp_assets.verify_vader(asset, tmp_path) == target


def test_valid_existing_asset_never_opens_network(tmp_path: Path) -> None:
    data = _zip_bytes()
    asset = _asset_for(data)
    target = _write_target(tmp_path, data)

    def forbidden(_: str):
        raise AssertionError("network opened")

    assert nlp_assets.install_vader(asset, tmp_path, offline=False, opener=forbidden) == target


def test_missing_offline_fails_without_opening_network(tmp_path: Path) -> None:
    data = _zip_bytes()
    asset = _asset_for(data)

    def forbidden(_: str):
        raise AssertionError("network opened")

    with pytest.raises(nlp_assets.NLPAssetError, match="^missing$"):
        nlp_assets.install_vader(asset, tmp_path, offline=True, opener=forbidden)


@pytest.mark.parametrize(
    ("data", "asset_change", "category"),
    [
        (b"short", {"size": 99}, "size"),
        (_zip_bytes(), {"sha256": "0" * 64}, "hash"),
        (b"x" * len(_zip_bytes()), {}, "archive"),
        (_zip_bytes(member="wrong/file.txt"), {}, "archive"),
        (_zip_bytes(extra_member="rogue.txt"), {}, "archive"),
        (_zip_bytes(content=b"neutral\t0.0"), {}, "smoke"),
    ],
)
def test_verify_rejects_wrong_asset_identity(
    tmp_path: Path, data: bytes, asset_change: dict[str, object], category: str
) -> None:
    asset = _asset_for(data)
    if asset_change:
        asset = nlp_assets.VaderAsset(
            url=asset.url,
            sha256=str(asset_change.get("sha256", asset.sha256)),
            size=int(asset_change.get("size", asset.size)),
            resource=asset.resource,
            member=asset.member,
            license=asset.license,
        )
    _write_target(tmp_path, data)
    with pytest.raises(nlp_assets.NLPAssetError) as caught:
        nlp_assets.verify_vader(asset, tmp_path)
    assert caught.value.category == category


def test_verify_rejects_missing_directory_and_symlink_targets(tmp_path: Path) -> None:
    data = _zip_bytes()
    asset = _asset_for(data)
    with pytest.raises(nlp_assets.NLPAssetError, match="^missing$"):
        nlp_assets.verify_vader(asset, tmp_path)

    target = tmp_path / "sentiment" / "vader_lexicon.zip"
    target.mkdir(parents=True)
    with pytest.raises(nlp_assets.NLPAssetError, match="^destination$"):
        nlp_assets.verify_vader(asset, tmp_path)

    target.rmdir()
    real = tmp_path / "real.zip"
    real.write_bytes(data)
    target.symlink_to(real)
    with pytest.raises(nlp_assets.NLPAssetError, match="^destination$"):
        nlp_assets.verify_vader(asset, tmp_path)


def test_verify_rejects_symlinked_resource_parent(tmp_path: Path) -> None:
    data = _zip_bytes()
    asset = _asset_for(data)
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "sentiment").symlink_to(real, target_is_directory=True)
    (real / "vader_lexicon.zip").write_bytes(data)
    with pytest.raises(nlp_assets.NLPAssetError, match="^destination$"):
        nlp_assets.verify_vader(asset, tmp_path)


@contextmanager
def _response(data: bytes):
    yield io.BytesIO(data)


def test_online_install_streams_verifies_and_atomically_creates_target(tmp_path: Path) -> None:
    data = _zip_bytes()
    asset = _asset_for(data)
    target = nlp_assets.install_vader(asset, tmp_path, offline=False, opener=lambda _: _response(data))
    assert target.read_bytes() == data
    assert nlp_assets.verify_vader(asset, tmp_path) == target
    assert list(target.parent.glob(".vader_lexicon.*")) == []


@pytest.mark.parametrize("payload", [b"too short", _zip_bytes() + b"too long"])
def test_online_install_rejects_wrong_stream_size_and_cleans_temp(
    tmp_path: Path, payload: bytes
) -> None:
    expected = _zip_bytes()
    asset = _asset_for(expected)
    with pytest.raises(nlp_assets.NLPAssetError, match="^size$"):
        nlp_assets.install_vader(asset, tmp_path, offline=False, opener=lambda _: _response(payload))
    assert not (tmp_path / asset.resource).exists()
    assert list((tmp_path / "sentiment").glob(".vader_lexicon.*")) == []


def test_online_install_maps_opener_failure_to_redacted_network_error(tmp_path: Path) -> None:
    asset = _asset_for(_zip_bytes())

    def fail(_: str):
        raise RuntimeError("https://secret.invalid/?token=secret")

    with pytest.raises(nlp_assets.NLPAssetError) as caught:
        nlp_assets.install_vader(asset, tmp_path, offline=False, opener=fail)
    assert caught.value.category == "network"
    assert str(caught.value) == "network"


def test_corrupt_existing_target_is_never_overwritten(tmp_path: Path) -> None:
    data = _zip_bytes()
    asset = _asset_for(data)
    target = _write_target(tmp_path, b"corrupt")
    before = (target.read_bytes(), target.stat().st_ino)
    with pytest.raises(nlp_assets.NLPAssetError, match="^size$"):
        nlp_assets.install_vader(asset, tmp_path, offline=False, opener=lambda _: _response(data))
    assert (target.read_bytes(), target.stat().st_ino) == before


def test_atomic_race_verifies_valid_winner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _zip_bytes()
    asset = _asset_for(data)
    real_link = nlp_assets.os.link

    def race(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        Path(target).write_bytes(data)
        raise FileExistsError

    monkeypatch.setattr(nlp_assets.os, "link", race)
    target = nlp_assets.install_vader(asset, tmp_path, offline=False, opener=lambda _: _response(data))
    assert target.read_bytes() == data
    monkeypatch.setattr(nlp_assets.os, "link", real_link)


def test_atomic_race_rejects_corrupt_winner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _zip_bytes()
    asset = _asset_for(data)

    def race(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        Path(target).write_bytes(b"corrupt")
        raise FileExistsError

    monkeypatch.setattr(nlp_assets.os, "link", race)
    with pytest.raises(nlp_assets.NLPAssetError, match="^size$"):
        nlp_assets.install_vader(asset, tmp_path, offline=False, opener=lambda _: _response(data))
    assert list((tmp_path / "sentiment").glob(".vader_lexicon.*")) == []


def test_cli_reports_stable_success_and_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = _zip_bytes()
    asset = _asset_for(data)
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_text(sha256=asset.sha256, size=asset.size), encoding="utf-8"
    )
    _write_target(tmp_path / "data", data)

    assert nlp_assets.main(["verify", "--manifest", str(manifest), "--data-dir", str(tmp_path / "data")]) == 0
    assert capsys.readouterr().out == "nlp asset verification passed\n"

    (tmp_path / "data" / "sentiment" / "vader_lexicon.zip").unlink()
    assert nlp_assets.main(["verify", "--manifest", str(manifest), "--data-dir", str(tmp_path / "data")]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "nlp asset operation failed: missing\n"
    assert str(tmp_path) not in output.err


def test_default_data_dir_is_absolute() -> None:
    assert nlp_assets.default_nltk_data_dir().is_absolute()


REQUIRED_GUARDS = (
    ('if size != asset.size:', 2),
    ('if digest != asset.sha256:', 2),
    ('if files != {asset.member}:', 1),
    ('if offline:', 1),
    ('try:\n        return verify_vader(asset, data_dir)\n    except NLPAssetError as error:', 1),
    ('or data_dir.is_symlink():', 1),
    ('if target.is_symlink()', 1),
    ('os.link(temporary, target)', 1),
    ('except FileExistsError:\n            return verify_vader(asset, data_dir)', 1),
    ('temporary.unlink(missing_ok=True)', 1),
    ('analyzer.polarity_scores("good")["compound"] <= 0', 1),
    ('analyzer.polarity_scores("bad")["compound"] >= 0', 1),
)


def _assert_required_guards(source: str) -> None:
    for guard, count in REQUIRED_GUARDS:
        assert source.count(guard) == count, guard


def test_source_retains_all_fail_closed_guards() -> None:
    _assert_required_guards((REPO / "scripts/nlp_assets.py").read_text(encoding="utf-8"))


@pytest.mark.parametrize("guard", [item[0] for item in REQUIRED_GUARDS])
def test_each_fail_closed_guard_has_a_non_vacuous_deletion_mutation(guard: str) -> None:
    source = (REPO / "scripts/nlp_assets.py").read_text(encoding="utf-8")
    mutated = source.replace(guard, "# deleted guard", 1)
    assert mutated != source
    with pytest.raises(AssertionError):
        _assert_required_guards(mutated)
