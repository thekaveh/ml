#!/usr/bin/env bash

set -euo pipefail

die() {
    printf 'atlas-connect: %s\n' "$*" >&2
    exit 2
}

[[ "$#" -eq 0 ]] || die "this helper accepts no arguments"
[[ -t 1 ]] || die \
    "refusing to print a token-bearing connection URL without an interactive terminal"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091 # Path is resolved from this wrapper at runtime.
source "$SCRIPT_DIR/lib/atlas-dotenv.sh"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
INFRA="$REPO_ROOT/infra"
ATLAS_ENV="$INFRA/.env"

[[ -f "$ATLAS_ENV" && ! -L "$ATLAS_ENV" ]] || die \
    "Atlas environment is missing; run ./scripts/atlas-up.sh --prepare first"

project_name=""
jupyterhub_port=""
jupyterhub_token=""
project_name_count=0
jupyterhub_port_count=0
jupyterhub_token_count=0

while IFS= read -r line || [[ -n "$line" ]]; do
    if ! atlas_dotenv_parse_line "$line"; then
        continue
    fi
    case "$ATLAS_DOTENV_KEY" in
        PROJECT_NAME)
            project_name="$ATLAS_DOTENV_VALUE"
            project_name_count=$((project_name_count + 1))
            ;;
        JUPYTERHUB_PORT)
            jupyterhub_port="$ATLAS_DOTENV_VALUE"
            jupyterhub_port_count=$((jupyterhub_port_count + 1))
            ;;
        JUPYTERHUB_TOKEN)
            jupyterhub_token="$ATLAS_DOTENV_VALUE"
            jupyterhub_token_count=$((jupyterhub_token_count + 1))
            ;;
    esac
done < "$ATLAS_ENV"

[[ "$project_name_count" -eq 1 ]] || die \
    "$ATLAS_ENV must contain exactly one PROJECT_NAME entry"
[[ "$jupyterhub_port_count" -eq 1 ]] || die \
    "$ATLAS_ENV must contain exactly one JUPYTERHUB_PORT entry"
[[ "$jupyterhub_token_count" -le 1 ]] || die \
    "$ATLAS_ENV must not contain duplicate JUPYTERHUB_TOKEN entries"

[[ "$project_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die \
    "PROJECT_NAME contains an unsafe value"
[[ "$jupyterhub_port" =~ ^[0-9]+$ && "${#jupyterhub_port}" -le 5 ]] || die \
    "JUPYTERHUB_PORT must be a decimal TCP port"
port_number=$((10#$jupyterhub_port))
((port_number >= 1 && port_number <= 65535)) || die \
    "JUPYTERHUB_PORT must be between 1 and 65535"

if [[ -z "$jupyterhub_token" ]]; then
    command -v docker >/dev/null 2>&1 || die \
        "docker is required to read the running JupyterHub container log"
    container_name="${project_name}-jupyterhub"
    if ! jupyterhub_token="$(
        docker logs "$container_name" 2>&1 |
            grep -Eo '(^|[?&[:space:]])token=[^&[:space:]]+' |
            sed 's/.*token=//' |
            tail -n 1
    )"; then
        die "could not obtain a one-time Jupyter token from $container_name logs"
    fi
    [[ -n "$jupyterhub_token" ]] || die \
        "no one-time Jupyter token was found in $container_name logs"
fi

[[ "${#jupyterhub_token}" -le 4096 &&
    "$jupyterhub_token" =~ ^[A-Za-z0-9._~-]+$ ]] || die \
    "JUPYTERHUB_TOKEN contains an unsafe value"

printf '%s\n' \
    "Connect VS Code to the running Atlas Jupyter server:" \
    "  1. Open the Command Palette." \
    "  2. Run: Jupyter: Specify Jupyter Server for Connections" \
    "  3. Choose: Existing Jupyter Server" \
    "  4. Enter this connection URL (treat it like a password):" \
    "     http://localhost:${jupyterhub_port}/?token=${jupyterhub_token}" \
    "  5. Select the remote notebook kernel."
