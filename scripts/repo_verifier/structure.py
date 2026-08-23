"""Repository structure validation rules."""
from __future__ import annotations

import ast
import importlib.util
import io
import json
import re
import subprocess
import tokenize
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import nbformat

from . import common as _common
from .models import CheckResult, Finding, VerifierConfig

NOTEBOOK_ROOT = Path("notebooks")
DEFAULT_SUBPROCESS_TIMEOUT = 120

@dataclass(frozen=True)
class ImportedModule:
    module: str
    line: int
    relative: bool = False


_MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
_INLINE_CODE_RE = re.compile(r"`+[^`]*`+")
_IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))")
_NON_PYTHON_CELL_MAGICS = frozenset({
    "bash",
    "html",
    "javascript",
    "js",
    "latex",
    "perl",
    "ruby",
    "script",
    "sh",
    "svg",
    "writefile",
})
_GITIGNORE_REQUIRED_PATTERNS = (
    "docs/superpowers/",
    "plan-*.md", "notes-*.md", "audit-*.md",
    ".mypy_cache/", ".trunk/", ".vscode/",
)
_TRACKED_SUPERPOWERS_DOC_PREFIXES = (
    "docs/superpowers/specs/",
    "docs/superpowers/plans/",
)
_BLOAT_PATTERNS = (
    "__pycache__", ".ipynb_checkpoints", ".DS_Store",
    ".mypy_cache", ".pytest_cache",
)
# Top-level dirs that should not exist at all (either tracked or untracked).
_FORBIDDEN_TOPLEVEL_DIRS = ("common",)

# Modules expected to be available in the Atlas JupyterHub runtime but
# not necessarily in the verifier's lightweight venv. S2 reports these as
# warnings rather than errors when missing locally.
_RUNTIME_ONLY_MODULES = frozenset({
    "numpy",
    "torch", "torchvision", "torch_geometric", "torch_sparse", "torch_scatter",
    "pyg_lib",
    "matplotlib", "seaborn", "pandas", "sklearn", "scipy",
    "networkx", "community",
    "nnx",
    "tqdm",
})


def _cell_magic_name(line: str) -> str:
    stripped = line.lstrip()
    if not stripped.startswith("%%"):
        return ""
    return stripped[2:].split(None, 1)[0].strip().lower()


def _literal_dynamic_import(
    node: ast.AST,
    importlib_aliases: set[str] | None = None,
    import_module_aliases: set[str] | None = None,
) -> str:
    if not isinstance(node, ast.Call) or not node.args:
        return ""
    importlib_aliases = importlib_aliases or {"importlib"}
    import_module_aliases = import_module_aliases or set()
    func = node.func
    if isinstance(func, ast.Name):
        is_import = func.id == "__import__" or func.id in import_module_aliases
    elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        is_import = func.value.id in importlib_aliases and func.attr == "import_module"
    else:
        is_import = False
    if not is_import:
        return ""
    first_arg = node.args[0]
    if not isinstance(first_arg, ast.Constant) or not isinstance(first_arg.value, str):
        return ""
    return first_arg.value


def _paren_balance_delta(line: str) -> int:
    try:
        tokens = tokenize.generate_tokens(io.StringIO(line).readline)
        return sum(
            1 if token.string == "(" else -1 if token.string == ")" else 0
            for token in tokens
            if token.type == tokenize.OP
        )
    except tokenize.TokenError:
        code_text = line.split("#", 1)[0]
        return code_text.count("(") - code_text.count(")")


def _fallback_statement(lines: list[str], start: int) -> tuple[str, int]:
    statement_lines = [lines[start]]
    balance = _paren_balance_delta(lines[start])
    end = start
    while (balance > 0 or statement_lines[-1].rstrip().endswith("\\")) and end + 1 < len(lines):
        end += 1
        next_line = lines[end]
        statement_lines.append(next_line)
        balance += _paren_balance_delta(next_line)
    return "\n".join(statement_lines), end


def _imported_modules_from_source(source: str) -> Iterator[ImportedModule]:
    """Yield top-level imported module names and one-based line numbers."""
    for line in source.splitlines():
        if not line.strip():
            continue
        if _cell_magic_name(line) in _NON_PYTHON_CELL_MAGICS:
            return
        break

    cleaned_lines: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            cleaned_lines.append("")
        else:
            cleaned_lines.append(line)
    cleaned_source = "\n".join(cleaned_lines)

    try:
        tree = ast.parse(cleaned_source)
    except SyntaxError:
        fallback_lines = _blank_multiline_string_lines(cleaned_source)
        importlib_aliases = {"importlib"}
        import_module_aliases: set[str] = set()
        idx = 0
        while idx < len(fallback_lines):
            li = idx + 1
            line = fallback_lines[idx]
            try:
                line_tree = ast.parse(line)
            except SyntaxError:
                statement, end_idx = _fallback_statement(fallback_lines, idx)
                if end_idx > idx:
                    try:
                        line_tree = ast.parse(statement)
                    except SyntaxError:
                        line_tree = None
                    if line_tree is not None:
                        line_importlib_aliases, line_import_module_aliases = _importlib_aliases(line_tree)
                        importlib_aliases.update(line_importlib_aliases)
                        import_module_aliases.update(line_import_module_aliases)
                        for node in ast.walk(line_tree):
                            location = li + getattr(node, "lineno", 1) - 1
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    module = alias.name
                                    if module:
                                        yield ImportedModule(module=module, line=location)
                            elif isinstance(node, ast.ImportFrom):
                                if node.level:
                                    module = node.module or ", ".join(alias.name for alias in node.names)
                                    yield ImportedModule(module=module or ".", line=location, relative=True)
                                elif node.module:
                                    module = node.module
                                    if module:
                                        yield ImportedModule(module=module, line=location)
                            elif module := _literal_dynamic_import(node, importlib_aliases, import_module_aliases):
                                yield ImportedModule(module=module, line=location)
                        idx = end_idx + 1
                        continue
                m = _IMPORT_RE.match(line)
                if not m:
                    idx += 1
                    continue
                module = m.group(1) or m.group(2) or ""
                if module:
                    yield ImportedModule(module=module, line=li)
                idx += 1
                continue
            line_importlib_aliases, line_import_module_aliases = _importlib_aliases(line_tree)
            importlib_aliases.update(line_importlib_aliases)
            import_module_aliases.update(line_import_module_aliases)
            for node in ast.walk(line_tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name
                        if module:
                            yield ImportedModule(module=module, line=li)
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        module = node.module or ", ".join(alias.name for alias in node.names)
                        yield ImportedModule(module=module or ".", line=li, relative=True)
                    elif node.module:
                        module = node.module
                        if module:
                            yield ImportedModule(module=module, line=li)
                elif module := _literal_dynamic_import(node, importlib_aliases, import_module_aliases):
                    yield ImportedModule(module=module, line=li)
            idx += 1
        return

    importlib_aliases, import_module_aliases = _importlib_aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                if module:
                    yield ImportedModule(module=module, line=node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                module = node.module or ", ".join(alias.name for alias in node.names)
                yield ImportedModule(
                    module=module or ".",
                    line=node.lineno,
                    relative=True,
                )
            elif node.module:
                module = node.module
                if module:
                    yield ImportedModule(module=module, line=node.lineno)
        elif module := _literal_dynamic_import(node, importlib_aliases, import_module_aliases):
            yield ImportedModule(module=module, line=node.lineno)


def _importlib_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    importlib_aliases = {"importlib"}
    import_module_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    import_module_aliases.add(alias.asname or alias.name)
    return importlib_aliases, import_module_aliases


def _blank_multiline_string_lines(source: str) -> list[str]:
    lines = source.splitlines()
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type != tokenize.STRING:
                continue
            start_line, _ = tok.start
            end_line, _ = tok.end
            if end_line <= start_line:
                continue
            for idx in range(start_line - 1, min(end_line, len(lines))):
                lines[idx] = ""
    except tokenize.TokenError:
        pass
    return lines


def _git_ls_files(repo: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=repo, capture_output=True, text=True, check=True,
        timeout=DEFAULT_SUBPROCESS_TIMEOUT,
    )
    return out.stdout.splitlines()


def _is_allowed_tracked_superpowers_doc(path: str) -> bool:
    if not path.endswith(".md"):
        return False
    for prefix in _TRACKED_SUPERPOWERS_DOC_PREFIXES:
        if path.startswith(prefix) and "/" not in path.removeprefix(prefix):
            return True
    return False


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError):
        return ""


def _strip_markdown_code(text: str, *, strip_inline: bool = True) -> str:
    def code_fragment(value: str, *, crosses_lines: bool = False) -> str:
        return " " * len(value) if strip_inline or crosses_lines else value

    def mask_html_comments(
        line: str, in_comment: bool, inline_marker: str | None
    ) -> tuple[str, bool, str | None]:
        masked: list[str] = []
        index = 0
        crosses_lines = inline_marker is not None
        while index < len(line):
            if in_comment:
                close = line.find("-->", index)
                if close == -1:
                    masked.append(" " * (len(line) - index))
                    return "".join(masked), True, inline_marker
                masked.append(" " * (close + 3 - index))
                index = close + 3
                in_comment = False
                continue
            if inline_marker is not None:
                code_span = re.search(r"`+", line[index:])
                if code_span is None:
                    masked.append(code_fragment(line[index:], crosses_lines=crosses_lines))
                    return "".join(masked), in_comment, inline_marker
                end = index + code_span.end()
                masked.append(code_fragment(line[index:end], crosses_lines=crosses_lines))
                index = end
                if len(code_span.group(0)) == len(inline_marker):
                    inline_marker = None
                continue
            code_span = re.match(r"`+", line[index:])
            if code_span:
                inline_marker = code_span.group(0)
                end = index + len(inline_marker)
                masked.append(code_fragment(line[index:end]))
                index = end
                continue
            if line.startswith("<!--", index):
                in_comment = True
                continue
            masked.append(line[index])
            index += 1
        return "".join(masked), in_comment, inline_marker

    stripped: list[str] = []
    fence: tuple[str, int] | None = None
    raw_html_block: str | None = None
    in_comment = False
    inline_marker: str | None = None
    for line in text.splitlines():
        opener = re.match(r"^ {0,3}(?P<marker>`{3,}|~{3,})", line)
        invalid_backtick_fence = bool(
            not in_comment
            and inline_marker is None
            and opener
            and opener["marker"].startswith("`")
            and "`" in line[opener.end():]
        )
        if (
            fence is None
            and not in_comment
            and inline_marker is None
            and opener
            and not invalid_backtick_fence
        ):
            marker = opener["marker"]
            fence = (marker[0], len(marker))
            stripped.append(" " * len(line))
            continue
        if fence is not None:
            marker, minimum_length = fence
            if re.fullmatch(rf" {{0,3}}{re.escape(marker)}{{{minimum_length},}}\s*", line):
                fence = None
            stripped.append(" " * len(line))
            continue
        if raw_html_block is not None:
            if re.search(rf"</{raw_html_block}\s*>", line, re.IGNORECASE):
                raw_html_block = None
            stripped.append(" " * len(line))
            continue
        if line.startswith("    "):
            stripped.append(" " * len(line))
            continue
        line, in_comment, inline_marker = mask_html_comments(
            line, in_comment, inline_marker
        )
        raw_html_opener = re.match(
            r"^ {0,3}<(script|style|pre|textarea)(?=\s|>|$)",
            line,
            re.IGNORECASE,
        )
        if raw_html_opener is not None:
            raw_html_block = raw_html_opener.group(1).lower()
            if re.search(rf"</{raw_html_block}\s*>", line[raw_html_opener.end():], re.IGNORECASE):
                raw_html_block = None
            stripped.append(" " * len(line))
            continue
        stripped.append(line)
    return "\n".join(stripped)


def _github_markdown_slug(heading: str) -> str:
    heading = re.sub(r"<[^>]+>", "", heading)
    heading = re.sub(r"[`*_~\[\]]", "", heading).strip().lower()
    heading = re.sub(r"[^a-z0-9\s-]", "", heading)
    heading = re.sub(r"\s+", "-", heading).strip("-")
    return heading


def _markdown_heading_slugs(text: str) -> set[str]:
    counts: dict[str, int] = {}
    slugs: set[str] = set()
    for line in text.splitlines():
        m = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if not m:
            continue
        base = _github_markdown_slug(m.group(1))
        if not base:
            continue
        count = counts.get(base, 0)
        counts[base] = count + 1
        slugs.add(base if count == 0 else f"{base}-{count}")
    return slugs


def _split_markdown_link_target(target: str) -> tuple[str, str]:
    target = target.strip().strip("<>")
    target = target.split()[0] if target else ""
    if "#" not in target:
        return target, ""
    path_part, fragment = target.split("#", 1)
    return path_part, unquote(fragment)


def _iter_notebook_schema_files(repo: Path) -> Iterator[Path]:
    notebook_root = repo / NOTEBOOK_ROOT
    if not notebook_root.exists():
        return
    for nb_path in sorted(notebook_root.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in nb_path.parts:
            continue
        yield nb_path


def _active_task_path(repo: Path, task: str) -> Path:
    return repo / NOTEBOOK_ROOT / task


def _notebook_rel(path: Path, repo: Path) -> str:
    return str(path.relative_to(repo))


def _required_shellcheck_targets(repo: Path) -> tuple[Path, ...]:
    return (
        repo / "scripts" / "atlas-up.sh",
        repo / "scripts" / "atlas-down.sh",
        repo / "scripts" / "atlas-connect.sh",
    )


def _shellcheck_targets(repo: Path) -> tuple[Path, ...]:
    local_scripts = tuple(sorted((repo / "scripts").glob("*.sh")))
    return tuple(path for path in (*local_scripts, *_required_shellcheck_targets(repo)) if path.exists())


def _required_submodule_paths() -> tuple[str, ...]:
    return ("infra",)


def check_structure(repo: Path, config: VerifierConfig) -> CheckResult:
    result = CheckResult(name="structure")
    tracked = set(_git_ls_files(repo))

    valid_types = {"code", "markdown", "raw"}
    schema_notebooks = list(_iter_notebook_schema_files(repo))
    for nb in schema_notebooks:
        try:
            raw_doc = json.loads(nb.read_text(encoding="utf-8"))
            for i, c in enumerate(raw_doc.get("cells", [])):
                if "id" not in c:
                    result.findings.append(Finding(
                        id="S1.cell_id", check="structure", severity="error",
                        location=f"{nb.relative_to(repo)}:cell[{i}]",
                        message="cell is missing required nbformat v4 id",
                    ))
            try:
                nbformat.validate(raw_doc)
            except Exception as e:
                result.findings.append(Finding(
                    id="S1.schema",
                    check="structure",
                    severity="error",
                    location=str(nb.relative_to(repo)),
                    message=f"notebook schema validation failed: {e}",
                ))
            doc = nbformat.read(nb, as_version=4)
            for i, c in enumerate(doc.cells):
                if c.cell_type not in valid_types:
                    result.findings.append(Finding(
                        id="S1.cell_type", check="structure", severity="error",
                        location=f"{nb.relative_to(repo)}:cell[{i}]",
                        message=f"unknown cell_type={c.cell_type!r}",
                    ))
        except Exception as e:
            result.findings.append(Finding(
                id="S1.parse", check="structure", severity="error",
                location=str(nb.relative_to(repo)),
                message=f"failed to parse: {e}",
            ))

    notebooks = list(_common.iter_notebooks(repo, config, NOTEBOOK_ROOT))
    for nb in notebooks:
        try:
            doc = nbformat.read(nb, as_version=4)
        except Exception:
            continue
        sibling_modules = {
            p.stem for p in nb.parent.glob("*.py") if p.stem != "__init__"
        }
        seen_in_notebook: set[str] = set()
        for ci, cell in enumerate(doc.cells):
            if cell.cell_type != "code":
                continue
            for imported in _imported_modules_from_source(cell.source):
                module = imported.module
                module_root = module.split(".", 1)[0]
                li = imported.line
                location = f"{nb.relative_to(repo)}:cell[{ci}]:line[{li}]"
                if imported.relative:
                    result.findings.append(Finding(
                        id="S2.relative_import",
                        check="structure",
                        severity="error",
                        location=location,
                        message=(
                            "notebook uses a relative import that will not resolve "
                            f"reliably in a top-to-bottom kernel run: {module!r}"
                        ),
                    ))
                    continue
                if not module or module in seen_in_notebook:
                    continue
                seen_in_notebook.add(module)
                if module_root in sibling_modules:
                    continue
                if module_root in _RUNTIME_ONLY_MODULES:
                    continue
                try:
                    spec = importlib.util.find_spec(module)
                except (ImportError, ValueError) as e:
                    result.findings.append(Finding(
                        id="S2.import_error", check="structure", severity="warning",
                        location=location,
                        message=f"find_spec({module!r}) raised {e!r}",
                    ))
                    continue
                if spec is None:
                    severity = "warning" if module_root in _RUNTIME_ONLY_MODULES else "error"
                    result.findings.append(Finding(
                        id="S2.unresolved_import", check="structure", severity=severity,
                        location=location,
                        message=(
                            f"module {module!r} not importable in verifier env"
                            + (" (expected only in runtime container)"
                               if module_root in _RUNTIME_ONLY_MODULES else "")
                        ),
                    ))

    for doc_path, base_dir, raw_text in _common.iter_in_scope_markdown_documents(
        repo, config, NOTEBOOK_ROOT, _read_text
    ):
        text = _strip_markdown_code(raw_text)
        for m in _MARKDOWN_LINK_RE.finditer(text):
            path_part, fragment = _split_markdown_link_target(m.group(1))
            target = path_part
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if not target and not fragment:
                continue
            target_path = (base_dir / target).resolve() if target else doc_path.resolve()
            try:
                target_path.relative_to(repo.resolve())
            except ValueError:
                result.findings.append(Finding(
                    id="S3.repo_escape_link", check="structure", severity="error",
                    location=f"{doc_path.relative_to(repo)}",
                    message=f"internal link escapes repository root: {target}",
                    detail={"link": m.group(0)},
                ))
                continue
            if not target_path.exists():
                result.findings.append(Finding(
                    id="S3.broken_link", check="structure", severity="error",
                    location=f"{doc_path.relative_to(repo)}",
                    message=f"internal link target missing: {target}",
                    detail={"link": m.group(0)},
                ))
                continue
            if fragment and target_path.suffix.lower() == ".md":
                slugs = _markdown_heading_slugs(_read_text(target_path))
                if fragment not in slugs:
                    result.findings.append(Finding(
                        id="S3.broken_anchor", check="structure", severity="error",
                        location=f"{doc_path.relative_to(repo)}",
                        message=f"internal link anchor missing: #{fragment}",
                        detail={"link": m.group(0), "target": str(target_path.relative_to(repo))},
                    ))

    def add_common_import_findings(source: str, location_for_line: Callable[[int], str]) -> None:
        for imported in _imported_modules_from_source(source):
            if imported.relative:
                continue
            if imported.module == "common" or imported.module.startswith("common."):
                result.findings.append(Finding(
                    id="S5.common_import",
                    check="structure",
                    severity="error",
                    location=location_for_line(imported.line),
                    message="forbidden import; use `from nnx.` instead",
                ))

    for path in tracked:
        if path.startswith(("tests/", "notebooks/archive/")):
            continue
        full = repo / path
        if not full.is_file():
            continue
        suffix = full.suffix.lower()
        if suffix == ".py":
            add_common_import_findings(_read_text(full), lambda line, path=path: f"{path}:{line}")
        elif suffix == ".ipynb":
            try:
                doc = nbformat.read(full, as_version=4)
            except Exception:
                continue
            for ci, cell in enumerate(doc.cells):
                if cell.cell_type != "code":
                    continue
                add_common_import_findings(
                    cell.source,
                    lambda line, path=path, ci=ci: f"{path}:cell[{ci}]:line[{line}]",
                )

    gitignore_lines = {
        line.strip() for line in _read_text(repo / ".gitignore").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for pat in _GITIGNORE_REQUIRED_PATTERNS:
        if pat not in gitignore_lines:
            result.findings.append(Finding(
                id="S6.gitignore_missing", check="structure", severity="error",
                location=".gitignore",
                message=f"required pattern absent: {pat}",
            ))
    for path in tracked:
        if path.startswith(("docs/superpowers/",)) and not _is_allowed_tracked_superpowers_doc(path):
            result.findings.append(Finding(
                id="S6.tracked_bloat", check="structure", severity="error",
                location=path,
                message="bloat directory tracked; should be gitignored",
            ))

    for path in tracked:
        for pat in _BLOAT_PATTERNS:
            if pat in path:
                result.findings.append(Finding(
                    id="S7.tracked_bloat", check="structure", severity="error",
                    location=path,
                    message=f"bloat artifact tracked: contains {pat!r}",
                ))

    for d in _FORBIDDEN_TOPLEVEL_DIRS:
        if (repo / d).exists():
            result.findings.append(Finding(
                id="S7.forbidden_toplevel", check="structure", severity="error",
                location=d,
                message=(
                    "forbidden top-level directory exists (tracked or not); "
                    "violates repo conventions — see CONTRIBUTING.md"
                ),
            ))

    for script in sorted((repo / "scripts").glob("*.py")):
        if not script.is_file():
            continue
        rel = str(script.relative_to(repo))
        text = _read_text(script)
        has_shebang = text.startswith("#!")
        executable = bool(script.stat().st_mode & 0o111)
        if has_shebang != executable:
            result.findings.append(Finding(
                id="S8.script_executable_mismatch",
                check="structure",
                severity="error",
                location=rel,
                message=(
                    "script shebang and executable bit disagree; keep both "
                    "present for direct CLI scripts or both absent for modules"
                ),
                detail={"has_shebang": has_shebang, "executable": executable},
            ))

    return result
