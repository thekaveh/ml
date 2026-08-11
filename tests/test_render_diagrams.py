# tests/test_render_diagrams.py
from __future__ import annotations

import html
import re
from pathlib import Path

import pytest

cairosvg = pytest.importorskip("cairosvg")  # skips the whole module if cairosvg absent

from scripts.docs.render_diagrams import extract_svg, render_all, svg_to_png  # noqa: E402
from scripts.docs.manifest import parse_manifest  # noqa: E402

SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 50"><rect width="100" height="50" fill="#00e5ff"/></svg>'
HTML = f"<html><body>{SVG}</body></html>"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _visible_svg_text(master: Path) -> str:
    source = master.read_text(encoding="utf-8")
    return " ".join(
        html.unescape(re.sub(r"<[^>]+>", "", value)).strip()
        for value in re.findall(r"<text\b[^>]*>(.*?)</text>", source, flags=re.DOTALL)
    )


def test_extract_svg_pulls_inline_svg():
    assert extract_svg(HTML) == SVG


def test_extract_svg_sanitizes_non_xml_entities():
    src = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><text>a &middot; b &amp; c</text></svg>'
    out = extract_svg(f"<html>{src}</html>")
    assert "·" in out and "&middot;" not in out  # named entity → unicode (cairosvg/browser-safe)
    assert "&amp;" in out  # XML-predefined entity preserved


def test_svg_to_png_writes_png(tmp_path):
    out = tmp_path / "d.png"
    svg_to_png(SVG, out, width=200)
    assert out.exists() and out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_all_writes_svg_and_png(tmp_path):
    masters = tmp_path / "docs/diagrams"
    masters.mkdir(parents=True)
    (masters / "ml-eng-lab-system.html").write_text(HTML, encoding="utf-8")
    site_img = tmp_path / "generated/site/assets/img"
    png_dir = tmp_path / "docs/diagrams/img"
    manifest = parse_manifest(
        """
surfaces: [repo, site, wiki]
numbering: baked
sections: []
notebooks: []
diagrams:
  - id: system
    master: docs/diagrams/ml-eng-lab-system.html
"""
    )
    written = render_all(manifest, tmp_path, site_img, png_dir, width=200)
    assert (site_img / "system.svg").exists()
    assert (png_dir / "system.png").exists()
    assert written == [tmp_path / "docs/diagrams/ml-eng-lab-system.html"]


def test_documentation_architecture_describes_root_governance_and_readme_boundary():
    for relative_path in ("docs/index.md", "docs/architecture.md"):
        content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "manifest-declared canonical source set" in content
        assert "`SECURITY.md`" in content
        assert "`README.md`" in content
        assert "parity-guarded" in content
        assert "not a manifest-generated page" in content


def test_documentation_pipeline_masters_show_root_governance_and_readme_boundary():
    for filename in (
        "ml-eng-lab-docs-publishing.html",
        "ml-eng-lab-docs-sync.html",
    ):
        visible_text = _visible_svg_text(REPO_ROOT / "docs/diagrams" / filename)
        assert "SECURITY.md" in visible_text
        assert "root governance markdown" in visible_text
        assert "README opener" in visible_text
        assert "parity-guarded" in visible_text
        assert "not a generated page" in visible_text


def test_documentation_pipeline_masters_avoid_unsupported_status_glyphs():
    unsupported_glyphs = set("→←◇✓✗")
    for filename in (
        "ml-eng-lab-docs-publishing.html",
        "ml-eng-lab-docs-sync.html",
    ):
        visible_text = _visible_svg_text(REPO_ROOT / "docs/diagrams" / filename)
        assert unsupported_glyphs.isdisjoint(visible_text)
