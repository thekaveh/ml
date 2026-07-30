#!/usr/bin/env bash

set -euo pipefail

usage() {
    printf '%s\n' \
        "Usage: $(basename "$0") [--cold] [--dry-run]" \
        "  --cold      Stops Atlas and destroys persisted volumes." \
        "  --dry-run   Print the Atlas stop command without running it."
}

die() {
    printf 'atlas-down: %s\n' "$*" >&2
    exit 2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
INFRA="$REPO_ROOT/infra"

dry_run=false
stop_args=()

for arg in "$@"; do
    case "$arg" in
        --cold)
            if [[ "${#stop_args[@]}" -ne 0 ]]; then
                usage >&2
                die "--cold may be specified only once"
            fi
            stop_args+=(--cold)
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

[[ -f "$INFRA/stop.sh" ]] || die \
    "Atlas submodule is not initialized. Run: git submodule update --init --recursive infra"

if [[ "$dry_run" == true ]]; then
    printf './stop.sh'
    if [[ "${#stop_args[@]}" -ne 0 ]]; then
        printf ' %q' "${stop_args[@]}"
    fi
    printf '\n'
else
    (cd "$INFRA" && ./stop.sh "${stop_args[@]}")
fi
