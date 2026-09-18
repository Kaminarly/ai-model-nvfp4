#!/usr/bin/env bash
# sglang-dspark.sh - DSpark-accelerated SGLang API service in Docker, foreground.
#
# Runs the model author's certified SGLang image ATTACHED in this terminal: no
# `-d` and no restart policy. This window is the service console - SGLang's
# load progress and request log stream into it - and Ctrl-C stops the
# container.
#
# Why attached instead of `docker run -d --restart unless-stopped`:
# WSL reclaims the distro as soon as its last session ends (the distro journal
# shows `InitTerminateInstanceInternal ... reboot(RB_POWER_OFF)`), which kills a
# detached container anyway; a restart policy then starts it again on the next
# boot, so a detached setup spends its life reloading and being killed instead
# of serving. A foreground `docker run` holds one session open for exactly as
# long as the service runs - the lifetime we want, with Ctrl-C as the stop
# button. (`vmIdleTimeout`, the knob that would stretch the idle window, exists
# on Windows 11 only.)
#
# Usage (inside WSL2 Ubuntu; needs root because user kami is not in the docker
# group on this machine):
#   sudo bash scripts/sglang-dspark.sh start \
#     --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090 \
#     --draft-dir /home/kami/models/Qwen3.8-27B-DSpark-NVFP4
#
# NOTE (2026-09-19): the image is no longer present on this machine - it was
# deleted to free 41.9 GB of docker disk (digest
# lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1,
# see result/sglang-image-removal-result.md). The preflight below therefore
# exits 1 with a "pull it (about 18 GB)" hint; this script never pulls by
# itself, so nothing is downloaded unless you run that docker pull yourself.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Output helpers only ([INFO]/[OK]/[WARN]/[FAIL]/=== ... ===): same format as
# the other launchers so the console reads the same everywhere.
# shellcheck source=lib/wsl2-env-lib.sh
. "$SCRIPT_DIR/lib/wsl2-env-lib.sh"

DEFAULT_MODEL_DIR="/home/kami/models/Qwen3.8-27B-NVFP4-RTX5090"
DEFAULT_DRAFT_DIR="/home/kami/models/Qwen3.8-27B-DSpark-NVFP4"
IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:qwen38-27b}"
NAME="${SGLANG_NAME:-qwen38-sglang}"
PORT="${SERVE_PORT:-8192}"
CTX_LEN=163840
BLOCK_SIZE=7

usage() {
  cat <<'EOF'
Usage:
  sglang-dspark.sh start [options]
  sglang-dspark.sh help

Options:
  --model-dir DIR      Local target model folder, mounted read-only at
                       /models/target (default below).
  --draft-dir DIR      Local DSpark draft folder, mounted read-only at
                       /models/dspark (default below).
  --port N             Port inside WSL / published to Windows (default 8192,
                       same fixed port as the vLLM launchers' endpoint).
  --context-length N   Context window (default 163840). Measured single-request
                       cap on this 32 GB card at mem-fraction 0.90 is 166793
                       KV-pool tokens; 200000 does not fit (rejected at boot
                       time, not OOM).
  --block-size N       DSpark draft block size (default 7).
  --no-spec            Serve without speculative decoding (benchmark control;
                       same image, same model, no --speculative-* flags).
  --image TAG          SGLang image (default lmsysorg/sglang:qwen38-27b).
  --name NAME          Container name (default qwen38-sglang).
  --lan                Accepted for interface parity with the other launchers.
                       The container already binds 0.0.0.0 inside WSL; reaching
                       it from the LAN is a Windows-side step (portproxy +
                       firewall), which start-api-server-dspark.bat sets up.
  --dry-run            Print the exact docker command without running it.

The service runs in the foreground: SGLang's log is this console, Ctrl-C stops
the container. Nothing is downloaded - both model folders are bind-mounted
read-only. An existing container with the same name is removed first.
EOF
}

fail_hint() { # fail_hint <reason> <fix...>
  fail "$1"
  shift
  for line in "$@"; do printf '        fix: %s\n' "$line"; done
}

cmd_start() {
  local model_dir="$DEFAULT_MODEL_DIR" draft_dir="$DEFAULT_DRAFT_DIR"
  local port="$PORT" ctx="$CTX_LEN" block="$BLOCK_SIZE"
  local spec=1 dry=0

  while [ $# -gt 0 ]; do
    case "$1" in
      --model-dir)       model_dir="${2:-}"; [ -n "$model_dir" ] || { fail "--model-dir needs a value"; exit 2; }; shift 2 ;;
      --draft-dir)       draft_dir="${2:-}"; [ -n "$draft_dir" ] || { fail "--draft-dir needs a value"; exit 2; }; shift 2 ;;
      --port)            port="${2:-}";      [ -n "$port" ]      || { fail "--port needs a value"; exit 2; }; shift 2 ;;
      --context-length)  ctx="${2:-}";       [ -n "$ctx" ]       || { fail "--context-length needs a value"; exit 2; }; shift 2 ;;
      --block-size)      block="${2:-}";     [ -n "$block" ]     || { fail "--block-size needs a value"; exit 2; }; shift 2 ;;
      --image)           IMAGE="${2:-}";     [ -n "$IMAGE" ]     || { fail "--image needs a value"; exit 2; }; shift 2 ;;
      --name)            NAME="${2:-}";      [ -n "$NAME" ]      || { fail "--name needs a value"; exit 2; }; shift 2 ;;
      --no-spec)         spec=0; shift ;;
      --lan)             shift ;;   # Windows-side concern; see usage
      --dry-run)         dry=1; shift ;;
      --help|-h)         usage; exit 0 ;;
      *) fail "unknown option: $1"; usage >&2; exit 2 ;;
    esac
  done

  for pair in "port:$port" "context-length:$ctx" "block-size:$block"; do
    case "${pair#*:}" in
      ''|*[!0-9]*) fail "--${pair%%:*} must be a positive integer (got '${pair#*:}')"; exit 2 ;;
    esac
  done

  section "Preflight"

  if ! command -v docker >/dev/null 2>&1; then
    fail_hint "docker is not installed in this distribution." \
      "run D:\\WSL\\install-docker.sh as root, then retry."
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    fail_hint "cannot talk to the docker daemon." \
      "start it: systemctl start docker" \
      "if it is not enabled yet: systemctl enable --now docker"
    exit 1
  fi
  ok "docker daemon reachable"

  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    fail_hint "image '$IMAGE' is not present locally." \
      "pull it (about 18 GB): docker pull $IMAGE"
    exit 1
  fi
  ok "image present: $IMAGE"

  if [ ! -d "$model_dir" ]; then
    fail_hint "target model folder not found: $model_dir" "pass the right path with --model-dir."
    exit 1
  fi
  shopt -s nullglob
  local -a shards=("$model_dir"/model-*.safetensors)
  shopt -u nullglob
  if [ "${#shards[@]}" -eq 0 ]; then
    fail_hint "no model-*.safetensors in $model_dir" "this must be the raw ModelOpt NVFP4 folder."
    exit 1
  fi
  ok "target model: $model_dir (${#shards[@]} shard(s))"

  if [ "$spec" -eq 1 ]; then
    if [ ! -f "$draft_dir/model.safetensors" ]; then
      fail_hint "DSpark draft not found: $draft_dir/model.safetensors" \
        "pass the right path with --draft-dir, or serve without it using --no-spec."
      exit 1
    fi
    ok "DSpark draft: $draft_dir"
  else
    info "speculative decoding disabled (--no-spec): benchmark control run"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  else
    warn "nvidia-smi not found; the container may not see the GPU"
  fi

  section "Starting SGLang (DSpark) service"

  local -a ARGS=(
    run
    --name "$NAME"
    --gpus all --ipc=host --shm-size 32g
    -p "${port}:${port}"
    -v "$model_dir:/models/target:ro"
  )
  if [ "$spec" -eq 1 ]; then
    ARGS+=(-v "$draft_dir:/models/dspark:ro")
  fi
  ARGS+=(
    "$IMAGE"
    sglang serve
      --model-path /models/target
      --served-model-name Qwen3.8-27B-NVFP4-DSpark
  )
  if [ "$spec" -eq 1 ]; then
    ARGS+=(
      --speculative-algorithm DSPARK
      --speculative-draft-model-path /models/dspark
      --speculative-draft-model-quantization modelopt_fp4
      --speculative-dspark-block-size "$block"
    )
  fi
  ARGS+=(
    --trust-remote-code --tp-size 1
    --context-length "$ctx"
    --kv-cache-dtype fp8_e4m3
    --attention-backend flashinfer
    --chunked-prefill-size 1024
    --mamba-radix-cache-strategy extra_buffer_lazy
    --mamba-ssm-dtype bfloat16
    --max-mamba-cache-size 8
    --mm-feature-transport cpu
    --cuda-graph-max-bs-decode 1
    --mem-fraction-static 0.90
    --max-running-requests 1
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    --host 0.0.0.0 --port "$port"
  )

  if [ "$dry" -eq 1 ]; then
    info "dry-run: the container would be launched with:"
    printf 'docker'
    for a in "${ARGS[@]}"; do printf ' %s' "$a"; done
    printf '\n'
    return 0
  fi

  # A stale container from an earlier run would block the name and the port.
  if docker container inspect "$NAME" >/dev/null 2>&1; then
    warn "removing the existing container '$NAME' (stale run)"
    docker rm -f "$NAME" >/dev/null 2>&1 || true
  fi

  info "model    : $model_dir"
  [ "$spec" -eq 1 ] && info "draft    : $draft_dir (block size $block)"
  info "context  : $ctx tokens, kv-cache fp8_e4m3"
  info "model ID : Qwen3.8-27B-NVFP4-DSpark (use as \"model\" in requests)"
  info "endpoint : http://127.0.0.1:${port}/v1 (OpenAI compatible)"
  info "no restart policy: the container lives only while this console is open"
  info "Ctrl-C to stop"
  echo

  # Foreground, no -d: this session IS the service lifetime. docker forwards
  # Ctrl-C to the container, SGLang drains requests and exits; the stopped
  # container is kept (not --rm) so `docker logs <name>` still works afterwards.
  exec docker "${ARGS[@]}"
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