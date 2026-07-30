#!/usr/bin/env bash

set -euo pipefail

usage() {
    printf '%s\n' \
        "Usage: $(basename "$0") [--prepare | --validate] [--dry-run]" \
        "  --prepare   Create and verify local Atlas environment files only." \
        "  --validate  Prepare, backfill, validate compose, and run doctor without starting services." \
        "  --dry-run   Print Atlas commands without running them."
}

die() {
    printf 'atlas-up: %s\n' "$*" >&2
    exit 2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091 # Path is resolved from this wrapper at runtime.
source "$SCRIPT_DIR/lib/atlas-dotenv.sh"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
INFRA="$REPO_ROOT/infra"
MANIFEST="$REPO_ROOT/atlas.consumer.yml"
ATLAS_ENV="$INFRA/.env"
ATLAS_ENV_EXAMPLE="$INFRA/.env.example"
USER_ENV="$REPO_ROOT/atlas.env.user"
USER_ENV_EXAMPLE="$REPO_ROOT/atlas.env.user.example"

mode="start"
mode_was_set=false
dry_run=false

for arg in "$@"; do
    case "$arg" in
        --prepare | --validate)
            if [[ "$mode_was_set" == true ]]; then
                usage >&2
                die "--prepare and --validate are mutually exclusive"
            fi
            mode="${arg#--}"
            mode_was_set=true
            ;;
        --dry-run)
            if [[ "$dry_run" == true ]]; then
                usage >&2
                die "--dry-run may be specified only once"
            fi
            dry_run=true
            ;;
        *)
            usage >&2
            die "unsupported argument: $arg"
            ;;
    esac
done

# The Atlas child process and Docker Compose inherit this shell environment.
# Policy-critical values must therefore come only from the materialized
# consumer configuration, never an ambient export from the invoking terminal.
unset LLM_PROVIDER_SOURCE COMFYUI_SOURCE OLLAMA_LOCALHOST_PORT

[[ -f "$INFRA/start.sh" && -f "$ATLAS_ENV_EXAMPLE" ]] || die \
    "Atlas submodule is not initialized. Run: git submodule update --init --recursive infra"
[[ -f "$MANIFEST" ]] || die "consumer manifest is missing: $MANIFEST"
[[ -f "$USER_ENV_EXAMPLE" ]] || die "local environment template is missing: $USER_ENV_EXAMPLE"
[[ "$REPO_ROOT" != *$'\n'* && "$REPO_ROOT" != *$'\r'* ]] || die \
    "repository path contains an unsupported newline"

serialize_repo_path() {
    if [[ "$REPO_ROOT" != *'"'* ]]; then
        printf '"%s"' "$REPO_ROOT"
    elif [[ "$REPO_ROOT" != *"'"* ]]; then
        printf "'%s'" "$REPO_ROOT"
    else
        die "repository paths containing both quote characters are unsupported by Atlas dotenv parsing"
    fi
}

render_user_env() {
    local line
    local repo_path_count=0
    local serialized_repo_path

    serialized_repo_path="$(serialize_repo_path)"

    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%$'\r'}"
        if [[ "$line" == ML_ENG_LAB_REPO_PATH=* ]]; then
            printf 'ML_ENG_LAB_REPO_PATH=%s\n' "$serialized_repo_path"
            repo_path_count=$((repo_path_count + 1))
        else
            printf '%s\n' "$line"
        fi
    done < "$USER_ENV_EXAMPLE"

    [[ "$repo_path_count" -eq 1 ]] || die \
        "$USER_ENV_EXAMPLE must contain exactly one ML_ENG_LAB_REPO_PATH entry"
}

read_repo_path() {
    local line
    local value=""
    local value_count=0

    while IFS= read -r line || [[ -n "$line" ]]; do
        if atlas_dotenv_parse_line "$line" &&
            [[ "$ATLAS_DOTENV_KEY" == "ML_ENG_LAB_REPO_PATH" ]]; then
            value="$ATLAS_DOTENV_VALUE"
            value_count=$((value_count + 1))
        fi
    done < "$USER_ENV"

    [[ "$value_count" -eq 1 ]] || die \
        "$USER_ENV must contain exactly one ML_ENG_LAB_REPO_PATH entry"
    printf '%s' "$value"
}

prepare_local_state() {
    local rendered_user_env
    local configured_repo_path

    if [[ -e "$ATLAS_ENV" || -L "$ATLAS_ENV" ]]; then
        [[ -f "$ATLAS_ENV" && ! -L "$ATLAS_ENV" ]] || die \
            "refusing non-regular Atlas environment path: $ATLAS_ENV"
    elif [[ "$dry_run" == true ]]; then
        printf 'prepare: would copy %s to %s\n' "$ATLAS_ENV_EXAMPLE" "$ATLAS_ENV"
    else
        (
            set -o noclobber
            umask 077
            command cat "$ATLAS_ENV_EXAMPLE" > "$ATLAS_ENV"
        ) || die "refusing to overwrite Atlas environment: $ATLAS_ENV"
        printf 'prepare: created %s\n' "$ATLAS_ENV"
    fi

    if [[ -e "$USER_ENV" || -L "$USER_ENV" ]]; then
        [[ -f "$USER_ENV" && ! -L "$USER_ENV" ]] || die \
            "refusing non-regular local environment path: $USER_ENV"
        configured_repo_path="$(read_repo_path)"
    else
        rendered_user_env="$(render_user_env)"
        if [[ "$dry_run" == true ]]; then
            printf 'prepare: would create %s for %s\n' "$USER_ENV" "$REPO_ROOT"
            configured_repo_path="$REPO_ROOT"
        else
            (
                set -o noclobber
                umask 077
                printf '%s\n' "$rendered_user_env" > "$USER_ENV"
            ) || die "refusing to overwrite local environment: $USER_ENV"
            printf 'prepare: created %s\n' "$USER_ENV"
            configured_repo_path="$(read_repo_path)"
        fi
    fi

    [[ "$configured_repo_path" == /* ]] || die \
        "ML_ENG_LAB_REPO_PATH must be an absolute path matching $REPO_ROOT"
    [[ "$configured_repo_path" == "$REPO_ROOT" ]] || die \
        "ML_ENG_LAB_REPO_PATH must match this checkout exactly: $REPO_ROOT"
    export ML_ENG_LAB_REPO_PATH="$configured_repo_path"
}

run_start() {
    if [[ "$dry_run" == true ]]; then
        printf './start.sh'
        printf ' %q' "$@"
        printf '\n'
    else
        (cd "$INFRA" && ./start.sh "$@")
    fi
}

preflight_comfyui_override() {
    local source

    # The pinned Atlas default is a container source, but --track ml-eng
    # disables ComfyUI because it is off-track. Only a consumer-local source
    # declaration survives that track synthesis, so that is the policy seam.
    source="$(atlas_dotenv_last_value "$USER_ENV" "COMFYUI_SOURCE")" || return 0
    case "$source" in
        disabled | localhost | managed-localhost-mps)
            ;;
        *)
            die \
                "COMFYUI_SOURCE must be disabled, localhost, or managed-localhost-mps; auto and containerized sources are prohibited for this consumer"
            ;;
    esac
}

preflight_native_ollama() {
    local source
    local port
    local url

    source="$(atlas_dotenv_last_value "$ATLAS_ENV" "LLM_PROVIDER_SOURCE")" || die \
        "LLM_PROVIDER_SOURCE was not materialized; start native Ollama (for example, run 'ollama serve') and retry"
    [[ "$source" == "ollama-localhost" ]] || die \
        "LLM_PROVIDER_SOURCE must be ollama-localhost; start native Ollama (for example, run 'ollama serve') and retry"

    if ! port="$(atlas_dotenv_last_value "$ATLAS_ENV" "OLLAMA_LOCALHOST_PORT")"; then
        port=11434
    fi
    [[ "$port" =~ ^[0-9]{1,5}$ ]] && ((10#$port >= 1 && 10#$port <= 65535)) || die \
        "OLLAMA_LOCALHOST_PORT must be an integer from 1 through 65535; start native Ollama (for example, run 'ollama serve') and retry"

    command -v curl >/dev/null 2>&1 || die \
        "curl is required to check native Ollama; start native Ollama (for example, run 'ollama serve') and retry"
    url="http://127.0.0.1:$port/api/version"
    curl --disable --noproxy '*' --fail --silent --show-error --max-time 2 "$url" >/dev/null || die \
        "native Ollama did not respond at $url; start native Ollama (for example, run 'ollama serve') and retry"
}

prepare_local_state
if [[ "$mode" == "prepare" ]]; then
    exit 0
fi

run_start env backfill
run_start --consumer "$MANIFEST" compose validate
run_start --consumer "$MANIFEST" doctor --format json
if [[ "$mode" == "validate" ]]; then
    exit 0
fi

if [[ "$dry_run" == false ]]; then
    preflight_comfyui_override
    preflight_native_ollama
fi
run_start --consumer "$MANIFEST" --track ml-eng --no-tui --detach

if [[ "$dry_run" == false ]]; then
    infra_changes="$(git -C "$INFRA" status --porcelain)"
    [[ -z "$infra_changes" ]] || die \
        "Atlas startup changed tracked or non-ignored files under $INFRA; inspect git status before continuing"
fi
