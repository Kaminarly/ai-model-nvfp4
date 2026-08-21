#!/usr/bin/env bash
# serve.sh - start the offline short-context vLLM API service (issue 03).
#
# Single gate: the unified preflight (issue 02). A start request is refused
# unless run_preflight ends READY, with every failure reason + fix printed.
# On READY the service runs in the foreground:
#   - reads ONLY the given local raw ModelOpt NVFP4 safetensors directory
#     (no downloads, no GGUF/AWQ/GPTQ conversion, no re-quantization)
#   - fixed vLLM settings: --quantization modelopt --kv-cache-dtype fp8
#     --enable-prefix-caching --trust-remote-code, short context,
#     single-request config
#   - binds to 127.0.0.1 by default (loopback only; non-loopback binds are
#     refused - the spec defers LAN/internet exposure until auth is designed)
#   - HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are enforced for the process
#
# Usage (inside WSL2 Ubuntu), e.g.:
#   bash scripts/serve.sh start --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/serve-lib.sh
. "$SCRIPT_DIR/lib/serve-lib.sh"

usage() {
  cat <<'EOF'
Usage:
  serve.sh start --model-dir DIR [--prefix DIR] [--host IP] [--port N] [--lan] [--dry-run]
  serve.sh help

Options:
  --model-dir DIR   Local folder with the raw ModelOpt NVFP4 safetensors (required).
  --prefix DIR      Runtime directory (default: $HOME/vllm).
  --host IP         Bind address (default: 127.0.0.1; loopback only - other
                    addresses are refused unless --lan is given).
  --port N          Port (default: 8000).
  --lan             Bind 0.0.0.0 (all interfaces) so devices on the LAN can
                    reach the API. Windows still needs the portproxy + firewall
                    setup (see README section '局域网访问').
  --dry-run         Print the exact vLLM command without launching it.

The service only starts after the unified preflight (issue 02) passes; every
failure prints a reason and a fix. It uses fixed vLLM settings
(quantization=modelopt, kv-cache-dtype=fp8, enable-prefix-caching,
trust-remote-code), reads only the given local model directory, and never
downloads, converts or re-quantizes model files. Ctrl-C stops the service.
EOF
}

cmd_start() {
  parse_start_options "$@"; rc=$?
  case "$rc" in
    0) : ;;
    3) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  DRY_RUN="$START_DRY"

  section "Starting offline service"
  # The unified preflight (issue 02) is the only gate: refuse to start with
  # every failure reason + fix printed when it does not pass.
  if ! run_preflight "$START_PREFIX" "$START_MODEL_DIR"; then
    fail "service NOT started: preflight did not pass (see reasons above)."
    exit 1
  fi
  ok "preflight READY - launching offline service"
  run_offline_serve "$START_PREFIX" "$START_MODEL_DIR" "$START_HOST" "$START_PORT"
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
