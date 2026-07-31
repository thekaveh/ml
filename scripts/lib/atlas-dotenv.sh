#!/usr/bin/env bash

# Limited, non-evaluating dotenv parsing compatible with Atlas's pinned
# consumer overlay parser. Callers remain responsible for allowed-key and
# value validation.

atlas_dotenv_trim() {
    local value="$1"

    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

atlas_dotenv_parse_value() {
    local value
    local quote
    local remainder
    local index
    local previous

    value="$(atlas_dotenv_trim "$1")"
    if [[ "${value:0:1}" == '"' || "${value:0:1}" == "'" ]]; then
        quote="${value:0:1}"
        remainder="${value:1}"
        if [[ "$remainder" == *"$quote"* ]]; then
            value="${remainder%%"$quote"*}"
        else
            while [[ "${value:0:1}" == '"' ]]; do value="${value:1}"; done
            while [[ "${value: -1}" == '"' ]]; do value="${value:0:${#value}-1}"; done
            while [[ "${value:0:1}" == "'" ]]; do value="${value:1}"; done
            while [[ "${value: -1}" == "'" ]]; do value="${value:0:${#value}-1}"; done
        fi
    else
        for ((index = 0; index < ${#value}; index++)); do
            if [[ "${value:index:1}" == "#" ]]; then
                if [[ "$index" -eq 0 ]]; then
                    value=""
                    break
                fi
                previous="${value:index-1:1}"
                if [[ "$previous" == " " || "$previous" == $'\t' ]]; then
                    value="${value:0:index}"
                    break
                fi
            fi
        done
        value="$(atlas_dotenv_trim "$value")"
    fi
    printf '%s' "$value"
}

atlas_dotenv_parse_line() {
    local line="$1"
    local stripped

    ATLAS_DOTENV_KEY=""
    ATLAS_DOTENV_VALUE=""
    line="${line%$'\r'}"
    stripped="$(atlas_dotenv_trim "$line")"
    if [[ -z "$stripped" || "$stripped" == \#* || "$stripped" != *=* ]]; then
        return 1
    fi
    # shellcheck disable=SC2034 # Outputs consumed by the calling wrapper.
    ATLAS_DOTENV_KEY="$(atlas_dotenv_trim "${stripped%%=*}")"
    # shellcheck disable=SC2034 # Outputs consumed by the calling wrapper.
    ATLAS_DOTENV_VALUE="$(atlas_dotenv_parse_value "${stripped#*=}")"
}

atlas_dotenv_last_value() {
    local path="$1"
    local key="$2"
    local line
    local value=""
    local found=false

    while IFS= read -r line || [[ -n "$line" ]]; do
        if atlas_dotenv_parse_line "$line" && [[ "$ATLAS_DOTENV_KEY" == "$key" ]]; then
            value="$ATLAS_DOTENV_VALUE"
            found=true
        fi
    done < "$path"

    [[ "$found" == true ]] || return 1
    printf '%s' "$value"
}
