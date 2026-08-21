#!/usr/bin/env bash
# preflight.sh - unified run-before-launch checks (issue 02).
#
# Single boundary for launching the offline vLLM service (issue 03): exit 0 =
# READY, exit 1 = NOT READY, every failure prints reason + fix. Aggregates:
#   - issue 01 prerequisites (WSL2, Ubuntu, Windows 10, RTX 5090, driver,
#     python) and the created runtime (venv, nvcc release 13, vLLM version)
#   - offline mode enforcement (HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE)
#   - model integrity of a user-supplied local ModelOpt NVFP4 directory
#     (exactly 3 safetensors shards + index + required configs)
#   - VRAM: other processes using the GPU + free-VRAM gate
#
# This command NEVER downloads, copies, converts or deletes model files, and
# makes no network calls. Run inside WSL2 Ubuntu, e.g.:
#   bash scripts/preflight.sh --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/preflight-lib.sh
. "$SCRIPT_DIR/lib/preflight-lib.sh"

usage() {
  cat <<'EOF'
Usage:
  preflight.sh --model-dir DIR [--prefix DIR]
  preflight.sh --help

Options:
  --model-dir DIR   Local folder with the ModelOpt NVFP4 weights (required).
  --prefix DIR      Runtime directory (default: $HOME/vllm).

Exit: 0 = READY to start the offline service; 1 = NOT READY
(each failure prints a reason and a fix). No model files are downloaded,
copied, converted or deleted; this check makes no network calls.
EOF
}

cmd_preflight() {
  local prefix="" model_dir=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --prefix) prefix="${2:-}"; [ -n "$prefix" ] || { fail "--prefix needs a value"; exit 2; }; shift 2 ;;
      --model-dir) model_dir="${2:-}"; [ -n "$model_dir" ] || { fail "--model-dir needs a value"; exit 2; }; shift 2 ;;
      --help|-h) usage; exit 0 ;;
      *) fail "unknown option: $1"; usage >&2; exit 2 ;;
    esac
  done
  [ -n "$prefix" ] || prefix="$WSL2_ENV_PREFIX"
  if [ -z "$model_dir" ]; then
    fail "missing required option: --model-dir"
    usage >&2
    exit 2
  fi
  # Canonicalize an existing model dir to an absolute path.
  if [ -n "$model_dir" ] && [ -d "$model_dir" ]; then
    model_dir="$(cd "$model_dir" && pwd)"
  fi
  run_preflight "$prefix" "$model_dir"
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    -h|--help|help) usage; exit 0 ;;
    ""|check) shift || true; cmd_preflight "$@" ;;
    --*|-*) cmd_preflight "$@" ;;
    *) fail "unknown command: $cmd"; usage >&2; exit 2 ;;
  esac
}
main "$@"
