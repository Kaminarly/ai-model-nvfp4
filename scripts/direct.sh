#!/usr/bin/env bash
# direct.sh - preflight + VRAM gate + direct service boot, no validation.
#
# The fastest path to the long-context service: run the unified preflight
# (issue 02), check the full-config VRAM gate (issue 04), then boot vLLM and
# leave the OpenAI API server running in the foreground. Unlike
# scripts/fullcontext.sh it performs NO validation steps - no short-context
# smoke test, no context ramp, no concurrency probe, no over-boundary probe,
# no VRAM sampling. If the gates pass it just starts the service.
#
# Defaults target the machine-verified full context configuration
# (200000 tokens / 16 sequences / 0.90 GPU memory utilization, port 8192);
# every value is overridable via the same environment variables as
# fullcontext.sh (FULL_MAX_MODEL_LEN / FULL_MAX_NUM_SEQS / FULL_GPU_MEM_UTIL)
# plus --port / SERVE_PORT for the port.
#
# Usage (inside WSL2 Ubuntu), e.g.:
#   bash scripts/direct.sh start --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090
set -u

# Machine-verified defaults, set BEFORE sourcing the libs so their
# ${VAR:-default} lines keep these values instead of replacing them.
SERVE_PORT="${SERVE_PORT:-8192}"
FULL_MAX_MODEL_LEN="${FULL_MAX_MODEL_LEN:-200000}"
FULL_MAX_NUM_SEQS="${FULL_MAX_NUM_SEQS:-16}"
FULL_GPU_MEM_UTIL="${FULL_GPU_MEM_UTIL:-0.90}"
export SERVE_PORT FULL_MAX_MODEL_LEN FULL_MAX_NUM_SEQS FULL_GPU_MEM_UTIL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/fullcontext-lib.sh
. "$SCRIPT_DIR/lib/fullcontext-lib.sh"

usage() {
  cat <<'EOF'
Usage:
  direct.sh start --model-dir DIR [--prefix DIR] [--host IP] [--port N] [--lan] [--dry-run]
  direct.sh help

Options:
  --model-dir DIR   Local folder with the raw ModelOpt NVFP4 safetensors (required).
  --prefix DIR      Runtime directory (default: $HOME/vllm).
  --host IP         Bind address (default: 127.0.0.1; loopback only - other
                    addresses are refused unless --lan is given).
  --port N          Port (default: 8192; overridable with SERVE_PORT).
  --lan             Bind 0.0.0.0 (all interfaces) so devices on the LAN can
                    reach the API. Windows still needs the portproxy + firewall
                    setup - scripts/start-api-server-lan.bat does it all
                    (see README section '局域网访问').
  --dry-run         Print the plan (preflight + VRAM gate note + exact vLLM
                    command) without launching anything.

Direct boot path: the unified preflight (issue 02) must pass, then the
full-config VRAM gate (issue 04) must fit the current headroom, then vLLM is
started and kept running in the foreground - no smoke test, no context ramp,
no concurrency or boundary probes. Defaults are the machine-verified
configuration: context 200000, max-num-seqs 16, GPU memory utilization 0.90.
Overridable via FULL_MAX_MODEL_LEN / FULL_MAX_NUM_SEQS / FULL_GPU_MEM_UTIL.
The service reads only the given local model directory (offline mode
enforced), and Ctrl-C stops it.
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
    fail "invalid FULL_GPU_MEM_UTIL: $FULL_GPU_MEM_UTIL (must be a 0..1 decimal with up to 2 fraction digits, e.g. 0.90)"
    exit 2
  fi
  DRY_RUN="$START_DRY"

  section "Direct boot (preflight + VRAM gate, no validation)"
  info "Target: ${FULL_MAX_MODEL_LEN} tokens, ${FULL_MAX_NUM_SEQS} concurrent sequences, ${FULL_GPU_MEM_UTIL} GPU memory utilization, port ${START_PORT}"
  info "Model:  $START_MODEL_DIR (raw ModelOpt NVFP4 safetensors; offline mode enforced; nothing is downloaded)"

  # Gate 1: the unified preflight (issue 02). Refuse to start with every
  # failure reason + fix printed when it does not pass.
  if ! run_preflight "$START_PREFIX" "$START_MODEL_DIR"; then
    fail "service NOT started: preflight did not pass (see reasons above)."
    exit 1
  fi
  ok "preflight READY"

  # Gate 2: the full-config VRAM gate (issue 04). Dry-run only plans - it must
  # not be blocked by a dynamic VRAM snapshot.
  if [ "$DRY_RUN" -eq 1 ]; then
    info "dry-run: the full-config VRAM gate would require free >= $FULL_GPU_MEM_UTIL * GPU total"
    info "no validation steps run in direct mode: the service just boots after the gates"
    lan_hint "$START_HOST" "$START_PORT"
    local venv mname
    venv="$START_PREFIX/venv"
    mname="$(served_model_name "$START_MODEL_DIR")"
    local -a ARGS=()
    while IFS= read -r arg; do ARGS+=("$arg"); done < <(serve_argv_full "$venv" "$START_MODEL_DIR" "$START_HOST" "$START_PORT" "$mname" "$FULL_MAX_MODEL_LEN" "$FULL_MAX_NUM_SEQS")
    run_or_dry "${ARGS[@]}"
    return 0
  fi

  if ! check_full_vram; then
    fail "service NOT started: the full configuration does not fit the current VRAM headroom."
    exit 1
  fi

  run_direct "$START_PREFIX" "$START_MODEL_DIR" "$START_HOST" "$START_PORT"
}

# run_direct <prefix> <model-dir> <host> <port>: boot the full configuration
# with no verification steps and keep the foreground service running. The
# verified-instance trap pattern from fullcontext.sh is reused so Ctrl-C stops
# the service cleanly (under Git Bash a fake vLLM may exec a native process, so
# `wait` can return early; the launcher then idles until SIGINT/SIGTERM).
run_direct() {
  local prefix="$1" model_dir="$2" host="$3" port="$4"
  local len="$FULL_MAX_MODEL_LEN" seqs="$FULL_MAX_NUM_SEQS"
  section "Starting vLLM + OpenAI API server (no validation steps)"
  boot_vllm_background "$prefix" "$model_dir" "$host" "$port" "$len" "$seqs" "" || return 1
  local pid="$BOOT_PID"
  ok "service booting: http://$host:$port/v1 (models list: http://$host:$port/v1/models)"
  lan_hint "$host" "$port"
  info "Settings: quantization=modelopt, kv-cache-dtype=fp8, enable-prefix-caching, context=$len, max-num-seqs=$seqs, gpu-memory-utilization=$FULL_GPU_MEM_UTIL"
  info "Ctrl-C to stop"
  trap 'kill -TERM "$pid" 2>/dev/null' INT TERM
  wait "$pid" 2>/dev/null || true
  while kill -0 "$pid" 2>/dev/null; do sleep 1; done
  trap - INT TERM
  return 0
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
