# serve-lib.sh - offline vLLM service launcher (issues 03/04).
#
# Sourced by scripts/serve.sh (short context, issue 03) and by
# scripts/lib/fullcontext-lib.sh (full context, issue 04, which sources this
# file). The unified preflight (issue 02) is the single gate: the launch
# helpers here must not be reached unless run_preflight ended READY. The
# service reads ONLY the user-supplied local ModelOpt NVFP4 safetensors
# directory, uses fixed vLLM settings (quantization=modelopt, kv-cache-dtype
# fp8, trust-remote-code) and binds to 127.0.0.1 by default. It never
# downloads, converts or re-quantizes model files and makes no network calls.

# shellcheck source=preflight-lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/preflight-lib.sh"

# ---------------------------------------------------------------------------
# Service configuration (issue 03 short-context defaults; env-overridable).
# Issue 04 replaces the short-context values with the full 262144-token config.
# ---------------------------------------------------------------------------
SERVE_HOST="${SERVE_HOST:-127.0.0.1}"            # loopback only (enforced by serve.sh, issue requirement)
SERVE_PORT="${SERVE_PORT:-8000}"
SHORT_MAX_MODEL_LEN="${SHORT_MAX_MODEL_LEN:-8192}"   # short context for the smoke validation
SHORT_MAX_NUM_SEQS="${SHORT_MAX_NUM_SEQS:-1}"        # single-request config (spec)

served_model_name() { # served_model_name <model-dir> [override]
  if [ -n "${SERVED_MODEL_NAME:-}" ]; then
    printf '%s\n' "$SERVED_MODEL_NAME"
  else
    basename "$1"
  fi
}

# validate_bind <host> <port>: enforce the spec's loopback-only + port-range
# contract shared by serve.sh (issue 03) and fullcontext.sh (issue 04).
# Prints the failure reason (and fix for non-loopback binds) and returns 2;
# callers print usage and exit.
validate_bind() {
  local host="$1" port="$2" portnum
  case "$port" in
    ''|*[!0-9]*)
      fail "invalid port: $port"
      return 2 ;;
  esac
  # 10# forces decimal so leading-zero ports ("08") cannot be misread as octal
  # by the range test below.
  portnum=$((10#$port))
  if [ "$portnum" -lt 1 ] || [ "$portnum" -gt 65535 ]; then
    fail "invalid port: $port (must be 1-65535)"
    return 2
  fi
  case "$host" in
    127.0.0.1|localhost|::1) return 0 ;;
  esac
  fail "refusing non-loopback bind address: $host"
  info "fix: the spec keeps this service loopback-only; binding beyond loopback needs the auth/access-control design first (see spec.md Out of Scope). Use --host 127.0.0.1."
  return 2
}

# prepare_vllm_env <venv>: export the environment every vLLM launch needs.
# Offline mode must be visible to the vllm process itself, not only to the
# preflight that exported it. FlashInfer JIT / deep-gemm look up CUDA_HOME (or
# nvcc in PATH) in a subprocess that never sees the venv, so the venv CUDA tree
# is exported. FlashInfer's one-time ninja kernel build is serialized with
# MAX_JOBS=1 (parallel nvcc OOMs the WSL2 RAM cap), and its JIT link step
# needs -ltvm_ffi from the venv. Issue 04 boots vLLM repeatedly (smoke, ramp,
# full config) and reuses this per boot.
prepare_vllm_env() {
  local venv="$1" cuda_home tvm_ffi_lib
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  cuda_home="$(cuda_home_in_venv "$venv" 2>/dev/null)"
  if [ -n "$cuda_home" ] && [ -x "$cuda_home/bin/nvcc" ]; then
    export CUDA_HOME="$cuda_home"
    export PATH="$cuda_home/bin:$PATH"
  fi
  export MAX_JOBS=1
  tvm_ffi_lib="$venv/lib/python3.14/site-packages/tvm_ffi/lib"
  if [ -f "$tvm_ffi_lib/libtvm_ffi.so" ]; then
    export FLASHINFER_EXTRA_LDFLAGS="-L$tvm_ffi_lib -ltvm_ffi"
  fi
}

# serve_argv <venv> <model-dir> <host> <port> <model-name> <max-len> <max-seqs> [util]
# Prints the vLLM launch argv, one argument per line. Fixed settings: ModelOpt
# NVFP4 quantization, FP8 KV cache, trust-remote-code, loopback bind. The
# optional 8th argument (a GPU memory utilization float like 0.97) is emitted
# only when given - serve.sh (issue 03) calls without it and keeps the vLLM
# default, while the issue 04 ramp/full boots pass FULL_GPU_MEM_UTIL. No
# conversion/re-quantization flags ever appear here.
serve_argv() {
  printf '%s\n' "$1/bin/vllm" "serve" \
    "--model" "$2" \
    "--quantization" "modelopt" \
    "--kv-cache-dtype" "fp8" \
    "--host" "$3" "--port" "$4" \
    "--served-model-name" "$5" \
    "--max-model-len" "$6" "--max-num-seqs" "$7"
  if [ -n "${8:-}" ]; then
    printf '%s\n' "--gpu-memory-utilization" "$8"
  fi
  printf '%s\n' "--trust-remote-code"
}

# parse_start_options "$@": shared option parsing for the start commands of
# serve.sh (issue 03) and fullcontext.sh (issue 04) - both accept the same
# option set and both enforce the same loopback/port contract. On success the
# globals START_PREFIX / START_MODEL_DIR / START_HOST / START_PORT / START_DRY
# are set and 0 is returned; on a usage error a reason has been printed and 2
# is returned; --help/-h returns 3 so the caller shows its own usage and
# exits 0.
START_PREFIX=""
START_MODEL_DIR=""
START_HOST=""
START_PORT=""
START_DRY=0
parse_start_options() {
  local prefix="" model_dir="" host="" port="" dry=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --prefix) prefix="${2:-}"; [ -n "$prefix" ] || { fail "--prefix needs a value"; return 2; }; shift 2 ;;
      --model-dir) model_dir="${2:-}"; [ -n "$model_dir" ] || { fail "--model-dir needs a value"; return 2; }; shift 2 ;;
      --host) host="${2:-}"; [ -n "$host" ] || { fail "--host needs a value"; return 2; }; shift 2 ;;
      --port) port="${2:-}"; [ -n "$port" ] || { fail "--port needs a value"; return 2; }; shift 2 ;;
      --dry-run) dry=1; shift ;;
      --help|-h) return 3 ;;
      *) fail "unknown option: $1"; return 2 ;;
    esac
  done
  [ -n "$prefix" ] || prefix="$WSL2_ENV_PREFIX"
  host="${host:-$SERVE_HOST}"
  port="${port:-$SERVE_PORT}"
  if ! validate_bind "$host" "$port"; then
    return 2
  fi
  if [ -z "$model_dir" ]; then
    fail "missing required option: --model-dir"
    return 2
  fi
  # Canonicalize an existing model dir to an absolute path.
  if [ -d "$model_dir" ]; then
    model_dir="$(cd "$model_dir" && pwd)"
  fi
  START_PREFIX="$prefix"
  START_MODEL_DIR="$model_dir"
  START_HOST="$host"
  START_PORT="$port"
  START_DRY="$dry"
  return 0
}

# run_offline_serve <prefix> <model-dir> <host> <port>
# Launches the offline vLLM API server in the foreground. Callers must have
# already passed run_preflight for the same prefix + model-dir.
run_offline_serve() {
  local prefix="$1" model_dir="$2" host="$3" port="$4"
  local venv="$prefix/venv" name max_len max_seqs
  name="$(served_model_name "$model_dir")"
  max_len="$SHORT_MAX_MODEL_LEN"
  max_seqs="$SHORT_MAX_NUM_SEQS"

  # The real launch needs the vllm CLI; dry-run only prints the plan, so it
  # must not require the binary to exist yet.
  if [ "$DRY_RUN" -eq 0 ] && [ ! -x "$venv/bin/vllm" ]; then
    fail "vllm launcher not found at $venv/bin/vllm"
    info "fix: re-create the runtime: bash scripts/wsl2-env.sh create --prefix $prefix"
    return 1
  fi

  # Offline enforcement, the venv CUDA tree and the FlashInfer JIT guards
  # (MAX_JOBS=1, tvm_ffi link flag) must be visible to the vllm process.
  prepare_vllm_env "$venv"

  section "Offline vLLM service (issue 03)"
  info "Model:       $model_dir (raw ModelOpt NVFP4 safetensors)"
  info "OpenAI API:  http://$host:$port/v1"
  info "Models list: http://$host:$port/v1/models"
  info "Settings:    quantization=modelopt, kv-cache-dtype=fp8, context=$max_len, max-num-seqs=$max_seqs"
  info "Offline:     HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
  [ -z "${CUDA_HOME:-}" ] || info "CUDA_HOME:   $CUDA_HOME (venv CUDA tree for FlashInfer/deep-gemm JIT)"
  info "JIT build:  MAX_JOBS=$MAX_JOBS (serialized FlashInfer ninja; avoids WSL RAM OOM)"
  [ -z "${FLASHINFER_EXTRA_LDFLAGS:-}" ] || info "JIT link:    FLASHINFER_EXTRA_LDFLAGS=$FLASHINFER_EXTRA_LDFLAGS (tvm_ffi + cu13 lib64 for FlashInfer JIT link)"

  local -a ARGS=()
  while IFS= read -r arg; do ARGS+=("$arg"); done < <(serve_argv "$venv" "$model_dir" "$host" "$port" "$name" "$max_len" "$max_seqs")

  if [ "$DRY_RUN" -eq 1 ]; then
    run_or_dry "${ARGS[@]}"
    return 0
  fi

  info "Launching: vllm serve (Ctrl-C to stop)"
  exec "${ARGS[@]}"
}
