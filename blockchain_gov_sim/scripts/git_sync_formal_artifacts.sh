#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <commit-message> <path1> [path2 ...]" >&2
  exit 1
fi

COMMIT_MESSAGE="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

git add -f -- "$@"

if git diff --cached --quiet; then
  echo "[git-sync] no staged changes for: ${COMMIT_MESSAGE}"
  exit 0
fi

git commit -m "${COMMIT_MESSAGE}"
git push origin HEAD
