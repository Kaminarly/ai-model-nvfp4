#!/usr/bin/env bash
# fullcontext.sh - controlled full 262144-token context launch (issue 04).
#
# The CONTROLLED path to the full configuration, after the short-context
# service (issue 03) exists:
#   1. unified preflight (issue 02) must be READY
#   2. full-config VRAM gate: the target utilization must fit right now
#   3. short-context smoke test (issue 03 config) must boot, answer and reject
#      over-boundary prompts; the full configuration will not start otherwise
#   4. the context length is ramped step by step (CONTEXT_LADDER), verifying
#      each boot's effective max_model_len via /v1/models
#   5. the full configuration (262144 tokens, 16 concurrent sequences, 0.97
#      GPU memory utilization) boots, is verified, and stays running
#
# Every boot reads ONLY the given local raw ModelOpt NVFP4 safetensors
# directory (no downloads, no GGUF/AWQ/GPTQ conversion, no re-quantization),
# enforces offline mode, and binds loopback only.
#
# Usage (inside WSL2 Ubuntu), e.g.:
#   bash scripts/fullcontext.sh start --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/fullcontext-lib.sh
. "$SCRIPT_DIR/lib/fullcontext-lib.sh"

usage() {
  cat <<'EOF'
Usage:
  fullcontext.sh start --model-dir DIR [--prefix DIR] [--host IP] [--port N] [--dry-run]
  fullcontext.sh help

Options:
  --model-dir DIR   Local folder with the raw ModelOpt NVFP4 safetensors (required).
  --prefix DIR      Runtime directory (default: $HOME/qwen3-nvfp4-rtx5090).
  --host IP         Bind address (default: 127.0.0.1; loopback only - other
                    addresses are refused until the spec's auth/access-control
                    design exists).
  --port N          Port (default: 8000).
  --dry-run         Print the exact validation plan (preflight + VRAM gate +
                    every planned boot) without launching anything.

The CONTROLLED path to the full configuration (issue 04): after the unified
preflight (issue 02) and a full-config VRAM headroom gate, it runs the
short-context smoke test (issue 03 config). Only if that passes does it ramp
the context length step by step (CONTEXT_LADDER), verifying each boot's
effective context via /v1/models, then starts the full configuration
(262144 tokens, 16 concurrent sequences, 0.97 GPU memory utilization) and
keeps it running in the foreground. It reads only the given local model
directory and never downloads, converts or re-quantizes model files.
Ctrl-C stops the service. Target values are overridable via
FULL_MAX_MODEL_LEN / FULL_MAX_NUM_SEQS / FULL_GPU_MEM_UTIL / CONTEXT_LADDER.
EOF
}

cmd_start() {
  parse_start_options "$@"; rc=$?
  case "$rc" in
    0) : ;;
    3) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  if ! util_percent "$FULL_GPU_MEM_UTIL" >/dev/null; then
    fail "invalid FULL_GPU_MEM_UTIL: $FULL_GPU_MEM_UTIL (must be a 0..1 decimal with up to 2 fraction digits, e.g. 0.97)"
    exit 2
  fi
  DRY_RUN="$START_DRY"

  section "Controlled full-context launch (issue 04)"
  info "Target: ${FULL_MAX_MODEL_LEN} tokens, ${FULL_MAX_NUM_SEQS} concurrent sequences, ${FULL_GPU_MEM_UTIL} GPU memory utilization"
  info "Model:  $START_MODEL_DIR (raw ModelOpt NVFP4 safetensors; offline mode enforced; nothing is downloaded)"

  # The unified preflight (issue 02) is the first gate: refuse to start with
  # every failure reason + fix printed when it does not pass.
  if ! run_preflight "$START_PREFIX" "$START_MODEL_DIR"; then
    fail "full-context service NOT started: preflight did not pass (see reasons above)."
    exit 1
  fi
  ok "preflight READY"

  # Second gate: the full configuration must fit the current VRAM headroom
  # (spec: re-check VRAM headroom before launch), with lower-config
  # suggestions when it does not. Dry-run only plans - it must not be blocked
  # by a dynamic VRAM snapshot.
  if [ "$DRY_RUN" -eq 1 ]; then
    info "dry-run: the full-config VRAM gate would require free >= $FULL_GPU_MEM_UTIL * GPU total (see the planned steps below)"
    info "every boot would run offline (HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1) with MAX_JOBS=1 for the FlashInfer JIT build"
  elif ! check_full_vram; then
    fail "full-context service NOT started: the full configuration does not fit the current VRAM headroom."
    exit 1
  fi

  run_fullcontext_validation "$START_PREFIX" "$START_MODEL_DIR" "$START_HOST" "$START_PORT"
  exit $?
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    help|-h|--help|"") usage; exit 0 ;;
    start) shift || true; cmd_start "$@" ;;
    --*|-*) cmd_start "$@" ;;
    *) fail "unknown command: $cmd"; usage >&2; exit 2 ;;
  esac
}
main "$@"
