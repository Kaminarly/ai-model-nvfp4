#!/usr/bin/env bash
# sparkinfer-serve.sh - SparkInfer API service in Docker, foreground, offline.
#
# Runs the model author's SparkInfer image ATTACHED in this terminal: no `-d`,
# no restart policy and no `--rm`, so this window is the service console (the
# engine's load progress and request log stream here), Ctrl-C stops the
# container, and the stopped container is kept for `docker logs`.
#
# Why attached instead of `docker run -d --restart unless-stopped`: WSL reclaims
# the distro as soon as its last session ends (the distro journal shows
# `InitTerminateInstanceInternal ... reboot(RB_POWER_OFF)`), which kills a
# detached container anyway; a restart policy then starts it again on the next
# boot. A foreground `docker run` holds one session open for exactly as long as
# the service runs - the lifetime we want, with Ctrl-C as the stop button.
#
# No weights are downloaded: both local model folders are bind-mounted
# read-only at the paths the image's entrypoint already expects, and the
# entrypoint's fetch() only downloads when `$DIR/config.json` is missing - so a
# preflight that proves both config.json files exist makes the download branch
# unreachable. HF_HUB_OFFLINE=1 is set as a second fence.
#
# Usage (inside WSL2 Ubuntu; needs root because user kami is not in the docker
# group on this machine):
#   sudo bash scripts/sparkinfer-serve.sh start \
#     --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090 \
#     --draft-dir /home/kami/models/Qwen3.8-27B-DSpark-NVFP4
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Output helpers only ([INFO]/[OK]/[WARN]/[FAIL]/=== ... ===): same format as
# the other launchers so the console reads the same everywhere.
# shellcheck source=lib/wsl2-env-lib.sh
. "$SCRIPT_DIR/lib/wsl2-env-lib.sh"

DEFAULT_MODEL_DIR="/home/kami/models/Qwen3.8-27B-NVFP4-RTX5090"
DEFAULT_DRAFT_DIR="/home/kami/models/Qwen3.8-27B-DSpark-NVFP4"
IMAGE="${SPARKINFER_IMAGE:-ghcr.io/gittensor-ai-lab/sparkinfer-qwen38:0.5.10}"
NAME="${SPARKINFER_NAME:-qwen38-sparkinfer}"
PORT="${SERVE_PORT:-8192}"
# Container-internal port is fixed by the image (EXPOSE 8080, PORT=8080); only
# the host side of the mapping is configurable.
CONTAINER_PORT=8080
TARGET_DIR=/models/qwen38-nvfp4
DRAFT_DIR_IN=/models/qwen38-dspark

usage() {
  cat <<'EOF'
Usage:
  sparkinfer-serve.sh start [options]
  sparkinfer-serve.sh help

Options:
  --model-dir DIR      Local target model folder, mounted read-only at
                       /models/qwen38-nvfp4 (default below).
  --draft-dir DIR      Local DSpark draft folder, mounted read-only at
                       /models/qwen38-dspark (default below).
  --port N             Host port published to Windows (default 8192, the same
                       fixed port as the other launchers' endpoint). The
                       container always listens on 8080 internally.
  --context-length N   Context window. Defaults per mode: 262144 autoregressive
                       (the model's native window, ~27.9 GB) and 153600 with
                       DSpark (the measured sweet spot on a 32 GB card: same
                       decode rate as 131072 with 17% more context, ~2.4 GB of
                       startup headroom left). Measured cliff above it: 158720
                       still runs clean but leaves only ~2.2 GB; 161280 degrades
                       intermittently and 163840 drops the engine onto a slow
                       path where DSpark ends up slower than AR. Both defaults
                       are the image's own for autoregressive; this option only
                       has to be passed to override them.
  --no-spec            Serve autoregressive only: no draft mount, no
                       serve-dspark, and the context default becomes 262144.
  --model-name ID      Model id reported by the API (default
                       Qwen3.8-27B-NVFP4-DSpark, or Qwen3.8-27B-NVFP4 with
                       --no-spec).
  --image TAG          SparkInfer image (default the pinned 0.5.10 tag).
  --name NAME          Container name (default qwen38-sparkinfer).
  --no-download        Hard offline: also pass SPARKINFER_NO_DOWNLOAD=1 and
                       MANIFEST_PATH=/nonexistent. The entrypoint then refuses
                       to serve when a folder is incomplete instead of trying
                       to download, while skipping the compute-pool manifest
                       sha256 gate (this machine's folders are the inference
                       files only, so they deliberately do not hash-match the
                       manifest that ships in the image).
  --lan                Accepted for interface parity with the other launchers.
                       The container already binds 0.0.0.0 inside WSL; reaching
                       it from the LAN is a Windows-side step (portproxy +
                       firewall), which start-api-server-sparkinfer.bat sets up.
  --dry-run            Print the exact docker command without running it.

Environment overrides: SPARKINFER_IMAGE, SPARKINFER_NAME, SPARKINFER_CTX,
SPARKINFER_MODEL_NAME, SPARKINFER_SAMPLING_DEFAULTS, SPARKINFER_KV_INT8,
SPARKINFER_DSPARK_MAX_CTX, SPARKINFER_EXTRA_ARGS, SERVE_PORT.

Sampling / DSpark: DSpark only engages for greedy, plain-text, single-request
traffic. The image's default sampling comes from the checkpoint's
generation_config.json (temperature 1.0 / top_k 20 / top_p 0.95), which is a
sampling request and therefore NOT speculated. Either set
SPARKINFER_SAMPLING_DEFAULTS=greedy here or send `"temperature": 0` per request,
then verify with the sparkinfer_speculative_runs_total counter on /metrics.

The service runs in the foreground: the engine's log is this console, Ctrl-C
stops the container. Nothing is downloaded - both model folders are bind-mounted
read-only. An existing container with the same name is removed first.
EOF
}

fail_hint() { # fail_hint <reason> <fix...>
  fail "$1"
  shift
  for line in "$@"; do printf '        fix: %s\n' "$line"; done
}

# port_in_use <port>: true when something already listens on it inside WSL.
port_in_use() {
  local port="$1"
  command -v ss >/dev/null 2>&1 || return 1
  ss -ltn 2>/dev/null | grep -qE "[:.]${port}([^0-9]|$)"
}

cmd_start() {
  local model_dir="$DEFAULT_MODEL_DIR" draft_dir="$DEFAULT_DRAFT_DIR"
  local port="$PORT" ctx="" model_name="" image="$IMAGE" name="$NAME"
  local spec=1 dry=0 hard_offline=0

  while [ $# -gt 0 ]; do
    case "$1" in
      --model-dir)       model_dir="${2:-}";  [ -n "$model_dir" ]  || { fail "--model-dir needs a value"; exit 2; }; shift 2 ;;
      --draft-dir)       draft_dir="${2:-}";  [ -n "$draft_dir" ]  || { fail "--draft-dir needs a value"; exit 2; }; shift 2 ;;
      --port)            port="${2:-}";       [ -n "$port" ]       || { fail "--port needs a value"; exit 2; }; shift 2 ;;
      --context-length)  ctx="${2:-}";        [ -n "$ctx" ]        || { fail "--context-length needs a value"; exit 2; }; shift 2 ;;
      --model-name)      model_name="${2:-}"; [ -n "$model_name" ] || { fail "--model-name needs a value"; exit 2; }; shift 2 ;;
      --image)           image="${2:-}";      [ -n "$image" ]      || { fail "--image needs a value"; exit 2; }; shift 2 ;;
      --name)            name="${2:-}";       [ -n "$name" ]       || { fail "--name needs a value"; exit 2; }; shift 2 ;;
      --no-spec)         spec=0; shift ;;
      --no-download)     hard_offline=1; shift ;;
      --lan)             shift ;;   # Windows-side concern; see usage
      --dry-run)         dry=1; shift ;;
      --help|-h)         usage; exit 0 ;;
      *) fail "unknown option: $1"; usage >&2; exit 2 ;;
    esac
  done

  for pair in "port:$port"; do
    case "${pair#*:}" in
      ''|*[!0-9]*) fail "--${pair%%:*} must be a positive integer (got '${pair#*:}')"; exit 2 ;;
    esac
  done
  if [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
    fail "--port must be between 1 and 65535 (got '$port')"; exit 2
  fi
  if [ -n "$ctx" ]; then
    case "$ctx" in
      ''|*[!0-9]*) fail "--context-length must be a positive integer (got '$ctx')"; exit 2 ;;
    esac
  fi

  # Mode-dependent defaults, applied after parsing so --no-spec still wins.
  local sampling="${SPARKINFER_SAMPLING_DEFAULTS:-}"
  local kv_int8="${SPARKINFER_KV_INT8:-}"
  local dspark_ctx="${SPARKINFER_DSPARK_MAX_CTX:-}"
  local extra="${SPARKINFER_EXTRA_ARGS:-}"
  if [ "$spec" -eq 1 ]; then
    [ -n "$ctx" ] || ctx="${SPARKINFER_CTX:-153600}"
    [ -n "$model_name" ] || model_name="${SPARKINFER_MODEL_NAME:-Qwen3.8-27B-NVFP4-DSpark}"
  else
    [ -n "$ctx" ] || ctx="${SPARKINFER_CTX:-262144}"
    [ -n "$model_name" ] || model_name="${SPARKINFER_MODEL_NAME:-Qwen3.8-27B-NVFP4}"
  fi
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

  if ! docker image inspect "$image" >/dev/null 2>&1; then
    fail_hint "image '$image' is not present locally." \
      "pull it (about 1.4 GB): docker pull $image" \
      "the engine image contains no model weights - the weights come from the mount."
    exit 1
  fi
  ok "image present: $image"
  case "$image" in
    *:latest|*:latest@*)
      warn "image tag 'latest' is floating: pin a version (e.g. :0.5.10) or a digest for reproducibility."
      ;;
  esac

  # --- the three-fence offline guarantee, fence 1: prove the mount is complete
  #     before docker ever runs, so the entrypoint's fetch() cannot fire.
  if [ ! -d "$model_dir" ]; then
    fail_hint "target model folder not found: $model_dir" \
      "pass the right path with --model-dir."
    exit 1
  fi
  if [ ! -f "$model_dir/config.json" ]; then
    fail_hint "no config.json in $model_dir" \
      "this must be the raw ModelOpt NVFP4 folder; without config.json the" \
      "container entrypoint would try to download the weights from Hugging Face."
    exit 1
  fi
  if [ ! -f "$model_dir/tokenizer.json" ]; then
    fail_hint "no tokenizer.json in $model_dir" \
      "the entrypoint always passes --tokenizer \$MODEL_DIR/tokenizer.json;" \
      "re-download just that file (it is not part of the safetensors shards)."
    exit 1
  fi
  ok "target model: $model_dir (config.json + tokenizer.json)"

  if [ "$spec" -eq 1 ]; then
    if [ ! -d "$draft_dir" ]; then
      fail_hint "DSpark draft folder not found: $draft_dir" \
        "pass the right path with --draft-dir, or serve without it using --no-spec."
      exit 1
    fi
    if [ ! -f "$draft_dir/config.json" ] || [ ! -f "$draft_dir/model.safetensors" ]; then
      fail_hint "DSpark draft is incomplete: $draft_dir" \
        "expected config.json + model.safetensors (single-file layout);" \
        "fix the path with --draft-dir, or serve without it using --no-spec."
      exit 1
    fi
    ok "DSpark draft: $draft_dir (config.json + model.safetensors)"
  else
    info "speculative decoding disabled (--no-spec): autoregressive control run"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  else
    warn "nvidia-smi not found; the container may not see the GPU"
  fi

  if port_in_use "$port"; then
    fail_hint "port $port is already in use inside WSL." \
      "another engine is probably running (vLLM / GGUF / SGLang): stop it first." \
      "only one engine can run at a time - they share port $port and the VRAM." \
      "to inspect: ss -ltnp | grep :$port"
    exit 1
  fi
  ok "port $port is free"

  section "Starting SparkInfer service"

  local -a ARGS=(
    run
    --name "$name"
    --gpus all
    -p "${port}:${CONTAINER_PORT}"
    -v "$model_dir:$TARGET_DIR:ro"
  )
  if [ "$spec" -eq 1 ]; then
    ARGS+=(-v "$draft_dir:$DRAFT_DIR_IN:ro")
  fi
  # fence 2: even if a folder were incomplete, hf download fails instead of
  # reaching the network. --no-download adds the entrypoint's own hard refusal.
  ARGS+=(-e HF_HUB_OFFLINE=1)
  ARGS+=(-e "CTX=$ctx")
  ARGS+=(-e "MODEL_NAME=$model_name")
  if [ "$hard_offline" -eq 1 ]; then
    ARGS+=(-e SPARKINFER_NO_DOWNLOAD=1 -e MANIFEST_PATH=/nonexistent)
  fi
  if [ -n "$sampling" ]; then
    ARGS+=(-e "SPARKINFER_SAMPLING_DEFAULTS=$sampling")
  fi
  if [ -n "$kv_int8" ]; then
    ARGS+=(-e "SPARKINFER_KV_INT8=$kv_int8")
  fi
  if [ -n "$dspark_ctx" ]; then
    ARGS+=(-e "SPARKINFER_DSPARK_MAX_CTX=$dspark_ctx")
  fi
  if [ -n "$extra" ]; then
    # Word-split on purpose: this is a free-form passthrough for engine knobs
    # that this launcher does not model (documented in README 4.9).
    local -a extra_args=()
    read -r -a extra_args <<<"$extra"
    local a
    for a in "${extra_args[@]}"; do ARGS+=(-e "$a"); done
  fi
  ARGS+=("$image")
  [ "$spec" -eq 1 ] && ARGS+=(serve-dspark)

  if [ "$dry" -eq 1 ]; then
    info "dry-run: the container would be launched with:"
    printf 'docker'
    local a2
    for a2 in "${ARGS[@]}"; do printf ' %s' "$a2"; done
    printf '\n'
    return 0
  fi

  # A stale container from an earlier run would block the name and the port.
  if docker container inspect "$name" >/dev/null 2>&1; then
    warn "removing the existing container '$name' (stale run)"
    docker rm -f "$name" >/dev/null 2>&1 || true
  fi

  info "model    : $model_dir"
  if [ "$spec" -eq 1 ]; then
    info "draft    : $draft_dir"
    info "mode     : DSpark (serve-dspark) - greedy, plain-text, single-request only"
  else
    info "mode     : autoregressive (no draft)"
  fi
  info "context  : $ctx tokens"
  info "model ID : $model_name (use as \"model\" in requests)"
  info "endpoint : http://127.0.0.1:${port}/v1 (OpenAI compatible)"
  info "health   : http://127.0.0.1:${port}/health"
  info "no restart policy: the container lives only while this console is open"
  info "Ctrl-C to stop (the container is kept: docker logs $name)"
  echo

  # Foreground, no -d: this session IS the service lifetime. docker forwards
  # Ctrl-C to the container, the server drains and exits; the stopped container
  # is kept (not --rm) so `docker logs <name>` still works afterwards.
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