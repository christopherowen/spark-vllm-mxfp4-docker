#!/usr/bin/env bash
set -euo pipefail

# Opinionated clangd watcher for *our* MXFP4 / SM12x fusion work in FlashInfer.
# Watches a curated set of directories and re-runs clangd_check.sh on any
# modified C++/CUDA file. This avoids the noise/cost of watching the entire repo
# (e.g. 3rdparty/).
#
# Usage:
#   export CLANGD_DIAG_LOG=/tmp/clangd-diag.log
#   scripts/clangd_watch_mxfp4.sh
#
# Optional:
#   FLASHINFER_DIR=/workspace/flashinfer scripts/clangd_watch_mxfp4.sh

FLASHINFER_DIR="${FLASHINFER_DIR:-/home/swank/projects/flashinfer}"
CLANGD_DIAG_LOG="${CLANGD_DIAG_LOG:-/tmp/clangd-diag.log}"

CHECK_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/clangd_check.sh"

if ! command -v inotifywait >/dev/null 2>&1; then
  echo "inotifywait not found; install inotify-tools" >&2
  exit 1
fi

declare -a WATCH_DIRS=(
  # Python JIT generators / module selection (where bringup flags often get wired).
  "${FLASHINFER_DIR}/flashinfer/jit"
  "${FLASHINFER_DIR}/flashinfer/fused_moe"

  # SM12x MoE CUTLASS integration (launcher + headers).
  "${FLASHINFER_DIR}/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm"
  "${FLASHINFER_DIR}/csrc/fused_moe"
  "${FLASHINFER_DIR}/include/flashinfer"
)

for d in "${WATCH_DIRS[@]}"; do
  if [[ ! -d "${d}" ]]; then
    echo "missing watch dir: ${d}" >&2
  fi
done

echo "Writing diagnostics to: ${CLANGD_DIAG_LOG}"
echo "Watching:"
for d in "${WATCH_DIRS[@]}"; do
  echo "  - ${d}"
done

export CLANGD_DIAG_LOG="${CLANGD_DIAG_LOG}"

while true; do
  # Wait for a write-close anywhere in watched dirs; emit full path.
  CHANGED_PATH="$(
    inotifywait -q -r \
      -e close_write,move,create,delete \
      --format '%w%f' \
      "${WATCH_DIRS[@]}"
  )" || true

  if [[ -z "${CHANGED_PATH}" || ! -f "${CHANGED_PATH}" ]]; then
    continue
  fi

  case "${CHANGED_PATH}" in
    *.cu|*.cuh|*.c|*.cc|*.cpp|*.h|*.hpp|*.inl)
      "${CHECK_SCRIPT}" "${CHANGED_PATH}" >/dev/null || true
      ;;
  esac
done

