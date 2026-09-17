#!/usr/bin/env bash
# Validate a PR title against Conventional Commits (<type>[scope][!]: <description>).
# Compound titles split on " | ". Exit 0 valid, 1 invalid, 2 usage.

set -euo pipefail

TITLE="${1:-}"
[[ -n "$TITLE" ]] || { echo "usage: $0 \"<PR title>\""; exit 2; }

VALID_TYPES="feat|fix|docs|style|refactor|perf|test|chore|build|ci|revert"
PATTERN="^(${VALID_TYPES})(\([a-zA-Z0-9_, -]+\))?(!?): .{1,}$"

# Split only on the documented " | " separator — a bare "|" is description text.
SEP=$'\x1f'
IFS="$SEP" read -ra SEGMENTS <<< "${TITLE// | /$SEP}"

segment_count=0
errors=()

for segment in "${SEGMENTS[@]}"; do
  trimmed="$(echo "$segment" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  [[ -z "$trimmed" ]] && continue
  segment_count=$((segment_count + 1))

  if [[ ! "$trimmed" =~ $PATTERN ]]; then
    errors+=("$trimmed")
  fi
done

if [[ $segment_count -eq 0 ]]; then
  echo "FAIL: PR title is empty"
  exit 1
fi

if ((${#errors[@]})); then
  echo "FAIL: the following title segment(s) do not follow conventional commit format:"
  for e in "${errors[@]}"; do
    echo "  - \"$e\""
  done
  echo ""
  echo "Required format: <type>[optional scope][!]: <description>"
  echo "Valid types: ${VALID_TYPES//|/, }"
  exit 1
fi

echo "OK: PR title follows conventional commit format"
