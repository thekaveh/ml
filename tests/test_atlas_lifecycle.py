from __future__ import annotations

import os
import pty
import re
import select
import shutil
import subprocess
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
DOTENV_HELPER = SCRIPTS / "lib" / "atlas-dotenv.sh"
TEST_SUBPROCESS_TIMEOUT = 10


@pytest.fixture
def atlas_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "ml-eng-lab"
    scripts = repo / "scripts"
    infra = repo / "infra"
    scripts.mkdir(parents=True)
    infra.mkdir()
    (scripts / "lib").mkdir()

    for name in ("atlas-up.sh", "atlas-down.sh"):
        shutil.copy2(SCRIPTS / name, scripts / name)
    shutil.copy2(DOTENV_HELPER, scripts / "lib" / "atlas-dotenv.sh")

    (repo / "atlas.consumer.yml").write_text("name: ml-eng-lab\n", encoding="utf-8")
    (repo / "atlas.env.user.example").write_text(
        "# Copy/create this as atlas.env.user; it is ignored and machine-local.\n"
        "ML_ENG_LAB_REPO_PATH=/absolute/path/to/ml-eng-lab\n",
        encoding="utf-8",
    )
    (infra / ".env.example").write_text(
        "PROJECT_NAME=atlas\nJUPYTERHUB_PORT=63094\nJUPYTERHUB_TOKEN=\n",
        encoding="utf-8",
    )
    for name in ("start.sh", "stop.sh"):
        path = infra / name
        path.write_text("#!/usr/bin/env bash\nexit 97\n", encoding="utf-8")
        path.chmod(0o755)
    return repo


def run_script(
    repo: Path,
    name: str,
    *args: str,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(repo / "scripts" / name), *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=check,
        env=env,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )


def atlas_start_commands(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith("./start.sh")]


def test_up_dry_run_prepares_missing_state_and_prints_exact_start_order(
    atlas_repo: Path,
) -> None:
    result = run_script(atlas_repo, "atlas-up.sh", "--dry-run")
    manifest = atlas_repo / "atlas.consumer.yml"

    assert "prepare" in result.stdout.lower()
    assert atlas_start_commands(result.stdout) == [
        "./start.sh env backfill",
        f"./start.sh --consumer {manifest} compose validate",
        f"./start.sh --consumer {manifest} doctor --format json",
        f"./start.sh --consumer {manifest} --track ml-eng --no-tui --detach",
    ]
    assert not (atlas_repo / "infra" / ".env").exists()
    assert not (atlas_repo / "atlas.env.user").exists()


def test_prepare_creates_local_files_once_without_starting_atlas(
    atlas_repo: Path,
) -> None:
    first = run_script(atlas_repo, "atlas-up.sh", "--prepare")

    assert first.returncode == 0
    assert (atlas_repo / "infra" / ".env").read_text(encoding="utf-8") == (
        "PROJECT_NAME=atlas\nJUPYTERHUB_PORT=63094\nJUPYTERHUB_TOKEN=\n"
    )
    assert (atlas_repo / "atlas.env.user").read_text(encoding="utf-8") == (
        "# Copy/create this as atlas.env.user; it is ignored and machine-local.\n"
        f'ML_ENG_LAB_REPO_PATH="{atlas_repo}"\n'
    )

    (atlas_repo / "infra" / ".env").write_text("KEEP=infra\n", encoding="utf-8")
    (atlas_repo / "atlas.env.user").write_text(
        f"ML_ENG_LAB_REPO_PATH={atlas_repo}\nKEEP=user\n", encoding="utf-8"
    )
    second = run_script(atlas_repo, "atlas-up.sh", "--prepare")

    assert second.returncode == 0
    assert (atlas_repo / "infra" / ".env").read_text(encoding="utf-8") == (
        "KEEP=infra\n"
    )
    assert (atlas_repo / "atlas.env.user").read_text(encoding="utf-8") == (
        f"ML_ENG_LAB_REPO_PATH={atlas_repo}\nKEEP=user\n"
    )


def test_prepare_quotes_parser_significant_repo_path(atlas_repo: Path) -> None:
    special_repo = atlas_repo.with_name("ml eng #lab")
    atlas_repo.rename(special_repo)

    first = run_script(special_repo, "atlas-up.sh", "--prepare")
    second = run_script(special_repo, "atlas-up.sh", "--prepare")

    assert first.returncode == 0
    assert second.returncode == 0
    assert (special_repo / "atlas.env.user").read_text(encoding="utf-8") == (
        "# Copy/create this as atlas.env.user; it is ignored and machine-local.\n"
        f'ML_ENG_LAB_REPO_PATH="{special_repo}"\n'
    )


def test_prepare_accepts_matching_quoted_path_with_inline_comment(
    atlas_repo: Path,
) -> None:
    (atlas_repo / "atlas.env.user").write_text(
        f'  ML_ENG_LAB_REPO_PATH = "{atlas_repo}"  # local checkout\n',
        encoding="utf-8",
    )

    result = run_script(atlas_repo, "atlas-up.sh", "--prepare")

    assert result.returncode == 0


def test_validate_prepares_then_runs_only_non_live_commands(atlas_repo: Path) -> None:
    result = run_script(atlas_repo, "atlas-up.sh", "--validate", "--dry-run")
    manifest = atlas_repo / "atlas.consumer.yml"

    assert atlas_start_commands(result.stdout) == [
        "./start.sh env backfill",
        f"./start.sh --consumer {manifest} compose validate",
        f"./start.sh --consumer {manifest} doctor --format json",
    ]
    assert "--detach" not in result.stdout


def initialize_mock_atlas_git_repo(atlas_repo: Path) -> None:
    infra = atlas_repo / "infra"
    (infra / ".gitignore").write_text(".env\n", encoding="utf-8")
    (infra / "tracked.txt").write_text("clean\n", encoding="utf-8")
    subprocess.run(
        ["git", "init", "-q"],
        cwd=infra,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    subprocess.run(
        ["git", "add", "."],
        cwd=infra,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Atlas Test",
            "-c",
            "user.email=atlas-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=infra,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )


def test_up_executes_exact_commands_from_infra(atlas_repo: Path, tmp_path: Path) -> None:
    special_repo = atlas_repo.with_name("ml eng lab")
    atlas_repo.rename(special_repo)
    atlas_repo = special_repo
    command_log = tmp_path / "atlas-commands.log"
    start = atlas_repo / "infra" / "start.sh"
    start.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\0' \"$PWD\" \"$#\" \"$@\" >> \"$ATLAS_TEST_COMMAND_LOG\"\n",
        encoding="utf-8",
    )
    start.chmod(0o755)
    initialize_mock_atlas_git_repo(atlas_repo)
    env = {**os.environ, "ATLAS_TEST_COMMAND_LOG": str(command_log)}

    result = run_script(atlas_repo, "atlas-up.sh", env=env)

    assert result.returncode == 0
    infra = atlas_repo / "infra"
    manifest = atlas_repo / "atlas.consumer.yml"
    assert command_log.read_bytes().split(b"\0")[:-1] == [
        str(infra).encode(),
        b"2",
        b"env",
        b"backfill",
        str(infra).encode(),
        b"4",
        b"--consumer",
        str(manifest).encode(),
        b"compose",
        b"validate",
        str(infra).encode(),
        b"5",
        b"--consumer",
        str(manifest).encode(),
        b"doctor",
        b"--format",
        b"json",
        str(infra).encode(),
        b"6",
        b"--consumer",
        str(manifest).encode(),
        b"--track",
        b"ml-eng",
        b"--no-tui",
        b"--detach",
    ]


def test_up_overrides_conflicting_parent_repo_path_for_every_atlas_command(
    atlas_repo: Path, tmp_path: Path
) -> None:
    environment_log = tmp_path / "atlas-environment.log"
    start = atlas_repo / "infra" / "start.sh"
    start.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\0' \"${ML_ENG_LAB_REPO_PATH-}\" \"$#\" \"$@\" "
        ">> \"$ATLAS_TEST_ENVIRONMENT_LOG\"\n",
        encoding="utf-8",
    )
    start.chmod(0o755)
    initialize_mock_atlas_git_repo(atlas_repo)
    env = {
        **os.environ,
        "ATLAS_TEST_ENVIRONMENT_LOG": str(environment_log),
        "ML_ENG_LAB_REPO_PATH": "/tmp/conflicting-checkout",
    }

    result = run_script(atlas_repo, "atlas-up.sh", env=env)

    assert result.returncode == 0
    fields = environment_log.read_bytes().split(b"\0")[:-1]
    expected_repo = str(atlas_repo).encode()
    assert fields[0] == expected_repo
    assert fields[4] == expected_repo
    assert fields[10] == expected_repo
    assert fields[17] == expected_repo


def test_up_preserves_special_repo_path_through_mocked_normal_start(
    atlas_repo: Path, tmp_path: Path
) -> None:
    special_repo = atlas_repo.with_name("ml eng #lab")
    atlas_repo.rename(special_repo)
    atlas_repo = special_repo
    environment_log = tmp_path / "atlas-environment.log"
    start = atlas_repo / "infra" / "start.sh"
    start.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\0' \"${ML_ENG_LAB_REPO_PATH-}\" "
        ">> \"$ATLAS_TEST_ENVIRONMENT_LOG\"\n"
        "if [[ \"$*\" == *'compose validate'* ]]; then\n"
        "    printf 'ML_ENG_LAB_REPO_PATH=%s\\n' \"$ATLAS_TEST_EXPECTED_REPO\" "
        "> \"$ATLAS_TEST_USER_ENV\"\n"
        "fi\n",
        encoding="utf-8",
    )
    start.chmod(0o755)
    initialize_mock_atlas_git_repo(atlas_repo)
    env = {
        **os.environ,
        "ATLAS_TEST_ENVIRONMENT_LOG": str(environment_log),
        "ATLAS_TEST_EXPECTED_REPO": str(atlas_repo),
        "ATLAS_TEST_USER_ENV": str(atlas_repo / "atlas.env.user"),
    }
    env.pop("ML_ENG_LAB_REPO_PATH", None)

    result = run_script(atlas_repo, "atlas-up.sh", env=env)

    assert result.returncode == 0
    assert environment_log.read_bytes().split(b"\0")[:-1] == [
        str(atlas_repo).encode()
    ] * 4
    assert (atlas_repo / "atlas.env.user").read_text(encoding="utf-8") == (
        f"ML_ENG_LAB_REPO_PATH={atlas_repo}\n"
    )


def test_up_fails_if_normal_start_dirties_atlas_checkout(
    atlas_repo: Path, tmp_path: Path
) -> None:
    command_log = tmp_path / "atlas-commands.log"
    start = atlas_repo / "infra" / "start.sh"
    start.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s|%s\\n' \"$PWD\" \"$*\" >> \"$ATLAS_TEST_COMMAND_LOG\"\n"
        "if [[ \"$*\" == *'--detach'* ]]; then\n"
        "    printf 'dirty\\n' >> tracked.txt\n"
        "fi\n",
        encoding="utf-8",
    )
    start.chmod(0o755)
    initialize_mock_atlas_git_repo(atlas_repo)
    env = {**os.environ, "ATLAS_TEST_COMMAND_LOG": str(command_log)}

    result = run_script(atlas_repo, "atlas-up.sh", check=False, env=env)

    assert result.returncode != 0
    assert "changed tracked or non-ignored files" in result.stderr


@pytest.mark.parametrize(
    "value",
    [
        "relative/path",
        "/a/different/checkout",
        "$(touch /tmp/atlas-command-substitution-must-not-run)",
    ],
)
def test_prepare_rejects_repo_path_that_is_not_the_exact_checkout(
    atlas_repo: Path, value: str
) -> None:
    (atlas_repo / "atlas.env.user").write_text(
        f"ML_ENG_LAB_REPO_PATH={value}\n", encoding="utf-8"
    )

    result = run_script(atlas_repo, "atlas-up.sh", "--prepare", check=False)

    assert result.returncode != 0
    assert "ML_ENG_LAB_REPO_PATH" in result.stderr


def test_up_reports_how_to_initialize_an_absent_submodule(atlas_repo: Path) -> None:
    (atlas_repo / "infra" / "start.sh").unlink()

    result = run_script(atlas_repo, "atlas-up.sh", "--dry-run", check=False)

    assert result.returncode != 0
    assert "git submodule update --init --recursive infra" in result.stderr


@pytest.mark.parametrize(
    "args",
    [
        ("--cold",),
        ("--bogus",),
        ("--prepare", "--validate"),
        ("--prepare", "--prepare"),
        ("--dry-run", "--dry-run"),
    ],
)
def test_up_rejects_unsupported_or_conflicting_arguments(
    atlas_repo: Path, args: tuple[str, ...]
) -> None:
    result = run_script(atlas_repo, "atlas-up.sh", *args, check=False)

    assert result.returncode != 0


def test_down_preserves_volumes_by_default_and_cold_is_explicit(
    atlas_repo: Path,
) -> None:
    warm = run_script(atlas_repo, "atlas-down.sh", "--dry-run")
    cold = run_script(atlas_repo, "atlas-down.sh", "--cold", "--dry-run")

    assert warm.stdout.splitlines() == ["./stop.sh"]
    assert "--cold" not in warm.stdout
    assert cold.stdout.splitlines() == ["./stop.sh --cold"]


def test_down_executes_from_infra_with_an_explicit_argument_array(
    atlas_repo: Path, tmp_path: Path
) -> None:
    stop = atlas_repo / "infra" / "stop.sh"
    stop.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\0' \"$PWD\" \"$#\" \"$@\" > \"$ATLAS_TEST_COMMAND_LOG\"\n",
        encoding="utf-8",
    )
    stop.chmod(0o755)
    command_log = tmp_path / "stop-commands.log"
    env = {**os.environ, "ATLAS_TEST_COMMAND_LOG": str(command_log)}

    warm = run_script(atlas_repo, "atlas-down.sh", env=env)
    warm_fields = command_log.read_bytes().split(b"\0")[:-1]
    cold = run_script(atlas_repo, "atlas-down.sh", "--cold", env=env)
    cold_fields = command_log.read_bytes().split(b"\0")[:-1]

    assert warm.returncode == 0
    assert warm_fields == [str(atlas_repo / "infra").encode(), b"0"]
    assert cold.returncode == 0
    assert cold_fields == [
        str(atlas_repo / "infra").encode(),
        b"1",
        b"--cold",
    ]


def test_down_rejects_unknown_arguments_and_describes_cold_data_loss(
    atlas_repo: Path,
) -> None:
    result = run_script(atlas_repo, "atlas-down.sh", "--bogus", check=False)

    assert result.returncode != 0
    assert "destroys persisted volumes" in result.stderr


@pytest.mark.parametrize(
    "args", [("--cold", "--cold"), ("--dry-run", "--dry-run"), ("--bogus",)]
)
def test_down_rejects_duplicate_or_unexpected_arguments(
    atlas_repo: Path, args: tuple[str, ...]
) -> None:
    result = run_script(atlas_repo, "atlas-down.sh", *args, check=False)

    assert result.returncode != 0


@pytest.mark.parametrize("local_name", ["infra/.env", "atlas.env.user"])
def test_up_rejects_symlinked_local_environment_paths(
    atlas_repo: Path, local_name: str
) -> None:
    target = atlas_repo / "outside.env"
    target.write_text(
        f"ML_ENG_LAB_REPO_PATH={atlas_repo}\n", encoding="utf-8"
    )
    local_path = atlas_repo / local_name
    if local_path.exists():
        local_path.unlink()
    local_path.symlink_to(target)

    result = run_script(atlas_repo, "atlas-up.sh", "--prepare", check=False)

    assert result.returncode != 0
    assert "refusing" in result.stderr


def test_lifecycle_scripts_enable_strict_bash_without_eval() -> None:
    for name in ("atlas-up.sh", "atlas-down.sh"):
        source = (SCRIPTS / name).read_text(encoding="utf-8")
        assert "set -euo pipefail" in source
        assert "eval" not in source


@pytest.fixture
def connect_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "ml-eng-lab"
    scripts = repo / "scripts"
    infra = repo / "infra"
    scripts.mkdir(parents=True)
    infra.mkdir()
    (scripts / "lib").mkdir()
    shutil.copy2(SCRIPTS / "atlas-connect.sh", scripts / "atlas-connect.sh")
    shutil.copy2(DOTENV_HELPER, scripts / "lib" / "atlas-dotenv.sh")
    return repo


def run_in_pty(
    repo: Path, *args: str, env: dict[str, str] | None = None
) -> tuple[int, str]:
    master_fd, slave_fd = pty.openpty()
    process = subprocess.Popen(
        [str(repo / "scripts" / "atlas-connect.sh"), *args],
        cwd=repo,
        stdin=subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
        env=env,
    )
    os.close(slave_fd)
    deadline = time.monotonic() + TEST_SUBPROCESS_TIMEOUT
    chunks: list[bytes] = []
    try:
        while True:
            if time.monotonic() >= deadline:
                process.kill()
                raise subprocess.TimeoutExpired(process.args, TEST_SUBPROCESS_TIMEOUT)
            readable, _, _ = select.select([master_fd], [], [], 0.1)
            if not readable:
                if process.poll() is not None:
                    break
                continue
            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(master_fd)
    returncode = process.wait(timeout=TEST_SUBPROCESS_TIMEOUT)
    return returncode, b"".join(chunks).decode(errors="replace").replace(
        "\r\n", "\n"
    )


def file_snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_connect_prefers_safe_configured_token_and_writes_no_artifacts(
    connect_repo: Path, tmp_path: Path
) -> None:
    token = "configured-token_123"
    (connect_repo / "infra" / ".env").write_text(
        f"PROJECT_NAME=atlas\nJUPYTERHUB_PORT=64123\nJUPYTERHUB_TOKEN={token}\n",
        encoding="utf-8",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text("#!/usr/bin/env bash\nexit 91\n", encoding="utf-8")
    docker.chmod(0o755)
    before = file_snapshot(connect_repo)
    env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}

    returncode, output = run_in_pty(connect_repo, env=env)

    assert returncode == 0
    assert "Jupyter: Specify Jupyter Server for Connections" in output
    assert f"http://localhost:64123/?token={token}" in output
    assert file_snapshot(connect_repo) == before


def test_connect_reads_fallback_token_only_from_expected_container_log(
    connect_repo: Path, tmp_path: Path
) -> None:
    token = "one-time-token.456"
    (connect_repo / "infra" / ".env").write_text(
        "PROJECT_NAME=ml-eng\nJUPYTERHUB_PORT=63094\nJUPYTERHUB_TOKEN=\n",
        encoding="utf-8",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        '[[ "$#" -eq 2 && "$1" == "logs" && "$2" == "ml-eng-jupyterhub" ]] '
        "|| exit 92\n"
        f"printf 'http://127.0.0.1:8888/lab?token={token}\\n'\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    before = file_snapshot(connect_repo)
    env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}

    returncode, output = run_in_pty(connect_repo, env=env)

    assert returncode == 0
    assert f"http://localhost:63094/?token={token}" in output
    assert file_snapshot(connect_repo) == before


def test_connect_accepts_atlas_dotenv_whitespace_quotes_and_comments(
    connect_repo: Path,
) -> None:
    token = "quoted-token_789"
    (connect_repo / "infra" / ".env").write_text(
        "  PROJECT_NAME = atlas  # local project\n"
        'JUPYTERHUB_PORT = "64123" # direct port\n'
        f"JUPYTERHUB_TOKEN='{token}'  # pinned token\n",
        encoding="utf-8",
    )
    before = file_snapshot(connect_repo)

    returncode, output = run_in_pty(connect_repo)

    assert returncode == 0
    assert f"http://localhost:64123/?token={token}" in output
    assert file_snapshot(connect_repo) == before


@pytest.mark.parametrize(
    ("env_text", "unsafe_name"),
    [
        (
            "PROJECT_NAME=atlas;touch-pwned\n"
            "JUPYTERHUB_PORT=63094\n"
            "JUPYTERHUB_TOKEN=token\n",
            "PROJECT_NAME",
        ),
        (
            "PROJECT_NAME=atlas\n"
            "JUPYTERHUB_PORT=63094;open\n"
            "JUPYTERHUB_TOKEN=token\n",
            "JUPYTERHUB_PORT",
        ),
        (
            "PROJECT_NAME=atlas\n"
            "JUPYTERHUB_PORT=63094\n"
            "JUPYTERHUB_TOKEN=$(touch should-not-exist)\n",
            "JUPYTERHUB_TOKEN",
        ),
    ],
)
def test_connect_rejects_unsafe_dotenv_values_without_evaluating_them(
    connect_repo: Path, env_text: str, unsafe_name: str
) -> None:
    (connect_repo / "infra" / ".env").write_text(env_text, encoding="utf-8")

    returncode, output = run_in_pty(connect_repo)

    assert returncode != 0
    assert unsafe_name in output
    assert not (connect_repo / "should-not-exist").exists()


def test_connect_refuses_to_print_token_to_a_noninteractive_stream(
    connect_repo: Path,
) -> None:
    token = "must-not-leak"
    (connect_repo / "infra" / ".env").write_text(
        f"PROJECT_NAME=atlas\nJUPYTERHUB_PORT=63094\nJUPYTERHUB_TOKEN={token}\n",
        encoding="utf-8",
    )

    result = run_script(connect_repo, "atlas-connect.sh", check=False)

    assert result.returncode != 0
    assert token not in result.stdout
    assert token not in result.stderr
    assert "interactive terminal" in result.stderr


def test_connect_rejects_unexpected_arguments_without_reading_env(
    connect_repo: Path,
) -> None:
    result = run_script(connect_repo, "atlas-connect.sh", "--bogus", check=False)

    assert result.returncode != 0
    assert "accepts no arguments" in result.stderr


def test_connect_rejects_symlinked_atlas_environment(connect_repo: Path) -> None:
    target = connect_repo / "outside.env"
    target.write_text(
        "PROJECT_NAME=atlas\n"
        "JUPYTERHUB_PORT=63094\n"
        "JUPYTERHUB_TOKEN=must-not-leak\n",
        encoding="utf-8",
    )
    (connect_repo / "infra" / ".env").symlink_to(target)

    returncode, output = run_in_pty(connect_repo)

    assert returncode != 0
    assert "must-not-leak" not in output
    assert "environment is missing" in output


def test_connect_source_has_no_eval_endpoint_dependency_or_secret_file_output() -> None:
    source = (SCRIPTS / "atlas-connect.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in source
    assert "eval" not in source
    assert not re.search(r"ATLAS_.*JUPYTER.*ENDPOINT", source)
    assert "token.txt" not in source
    assert "url.txt" not in source


def test_wrappers_share_parent_owned_non_evaluating_dotenv_parser() -> None:
    helper_source = DOTENV_HELPER.read_text(encoding="utf-8")

    assert not re.search(r"(^|[\s;&|])eval(?:\s|$)", helper_source, re.MULTILINE)
    assert not re.search(r"(^|[\s;&|])source(?:\s|$)", helper_source, re.MULTILINE)
    for name in ("atlas-up.sh", "atlas-connect.sh"):
        source = (SCRIPTS / name).read_text(encoding="utf-8")
        assert 'source "$SCRIPT_DIR/lib/atlas-dotenv.sh"' in source
        assert "parse_atlas_env_value()" not in source
