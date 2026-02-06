#!/usr/bin/env bash
set -euo pipefail

# Watch a file (or directory) and rerun clangd_check.sh on changes.
# Requires one of: watchexec, entr, or inotifywait.
#
# Usage:
#   scripts/clangd_watch.sh /abs/path/to/file.cu
#   scripts/clangd_watch.sh /home/swank/projects/flashinfer/include/flashinfer/moe/*.cuh
#   scripts/clangd_watch.sh /home/swank/projects/flashinfer/include/flashinfer/gemm
#
# Tip:
#   export CLANGD_DIAG_LOG=/tmp/clangd-diag.log
#   Then the agent can read that file to see the latest diagnostics.

TARGET="${1:-}"
if [[ -z "${TARGET}" ]]; then
  echo "usage: $0 <file-or-dir-or-glob>" >&2
  exit 2
fi

CHECK_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/clangd_check.sh"

if command -v watchexec >/dev/null 2>&1; then
  # watchexec understands globs and dirs; it can run on each change.
  exec watchexec -r -- "${CHECK_SCRIPT}" "${TARGET}"
fi

if command -v entr >/dev/null 2>&1; then
  # entr wants a file list on stdin; for globs, let the shell expand.
  # shellcheck disable=SC2086
  ls -1 ${TARGET} 2>/dev/null | entr -r "${CHECK_SCRIPT}" /_
  exit 0
fi

if command -v inotifywait >/dev/null 2>&1; then
  # Simple watch loop. Best results come from passing a single file path.
  #
  # If you pass a directory, we re-check the most recently changed file (best-effort),
  # otherwise we re-check the provided file.
  if [[ -d "${TARGET}" ]]; then
    DIR="${TARGET}"
    FILE=""
  else
    DIR="$(dirname "${TARGET}")"
    FILE="$(basename "${TARGET}")"
  fi

  # Initial run (so the log exists immediately).
  if [[ -n "${FILE}" && -f "${DIR}/${FILE}" ]]; then
    "${CHECK_SCRIPT}" "${DIR}/${FILE}" || true
  fi

  RECURSIVE="${CLANGD_WATCH_RECURSIVE:-0}"

  while true; do
    # Print the changed path. In recursive mode, %w includes the subdir path.
    if [[ "${RECURSIVE}" == "1" ]]; then
      CHANGED_PATH="$(inotifywait -q -r -e close_write,move,create,delete --format '%w%f' "${DIR}")" || true
    else
      CHANGED_PATH="$(inotifywait -q -e close_write,move,create,delete --format '%w%f' "${DIR}")" || true
    fi

    if [[ -n "${FILE}" ]]; then
      # Single-file mode: always check that file.
      if [[ -f "${DIR}/${FILE}" ]]; then
        "${CHECK_SCRIPT}" "${DIR}/${FILE}" || true
      fi
    else
      # Directory mode: check the changed file if it looks like a source file.
      if [[ -n "${CHANGED_PATH}" && -f "${CHANGED_PATH}" ]]; then
        case "${CHANGED_PATH}" in
          *.cu|*.cuh|*.c|*.cc|*.cpp|*.h|*.hpp|*.inl)
            "${CHECK_SCRIPT}" "${CHANGED_PATH}" || true
            ;;
        esac
      fi
    fi
  done
  exit 0
fi

echo "No watcher found. Install one of: watchexec, entr, inotify-tools" >&2
exit 1

