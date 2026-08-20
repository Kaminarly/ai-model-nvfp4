# fullcontext-lib.sh - controlled full 262144-token context launch (issue 04).
#
# Sourced by scripts/fullcontext.sh. The short-context service (issue 03) is
# the prerequisite: before any full-context boot this runs the unified
# preflight (issue 02) and a full-config VRAM headroom gate, then a live
# short-context smoke test booted with the same fixed vLLM settings. Only after
# the smoke test passes does it ramp the context length step by step
# (CONTEXT_LADDER), verifying each boot's effective max_model_len through
# /v1/models, and finally boots the full configuration (262144 tokens, 16
# concurrent sequences, 0.97 GPU memory utilization) and leaves it running.
# Every boot reads ONLY the user-supplied local ModelOpt NVFP4 safetensors
# directory in offline mode; nothing is downloaded, converted or re-quantized.

# shellcheck source=serve-lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/serve-lib.sh"

# ---------------------------------------------------------------------------
# Full-context configuration (issue 04 defaults; env-overridable).
# ---------------------------------------------------------------------------
FULL_MAX_MODEL_LEN="${FULL_MAX_MODEL_LEN:-262144}"   # target context (spec: 262144)
FULL_MAX_NUM_SEQS="${FULL_MAX_NUM_SEQS:-16}"         # target concurrency (spec: 16)
FULL_GPU_MEM_UTIL="${FULL_GPU_MEM_UTIL:-0.97}"       # target VRAM utilization (model card)
# The ramp ladder: after the short-context smoke test (SHORT_MAX_MODEL_LEN)
# passes, each entry is booted and verified in order, so a failure pinpoints
# the machine's actual context boundary. Override to probe other lengths.
CONTEXT_LADDER="${CONTEXT_LADDER:-32768 65536 131072 262144}"
# Boundary probe sizes: SMOKE_OVERFLOW_BYTES is the over-limit prompt for the
# smoke step; FULL_OVERFLOW_BYTES is the over-limit prompt for the full
# config. The probe text is a run of 'x', which this tokenizer compresses at
# exactly 8 bytes per token (measured on the real model), so the sizes were
# chosen to exceed the respective context limits: 65536 bytes = 8192 tokens
# (>= the 8192-token smoke context) and 2500000 bytes = 312500 tokens (> the
# 262144-token full-config target; the previous 1100000 bytes = 137500 tokens
# fell below 200000-token contexts and the probe wrongly returned 200).
# LONG_PROOF_BYTES is the in-bound long prompt sent after the full config
# boots (131072 bytes = 16384 tokens, well inside any full context).
SMOKE_OVERFLOW_BYTES="${SMOKE_OVERFLOW_BYTES:-65536}"
LONG_PROOF_BYTES="${LONG_PROOF_BYTES:-131072}"
FULL_OVERFLOW_BYTES="${FULL_OVERFLOW_BYTES:-2500000}"
# How many concurrent short requests the full-config confirmation sends to
# exercise the 16-sequence scheduler.
CONCURRENT_PROOF="${CONCURRENT_PROOF:-16}"
# VRAM utilization tolerance for the post-boot confirmation, in percentage
# points (informational warn only).
VRAM_UTIL_TOLERANCE_PCT="${VRAM_UTIL_TOLERANCE_PCT:-5}"
# How long to wait for each vLLM boot to answer /v1/models (seconds), and how
# long to wait for a killed boot to release its port before the next boot.
# 600s covers the one-time FlashInfer JIT kernel build (MAX_JOBS=1 serializes
# ~16 nvcc TUs; each takes minutes on the first boot, then is cached).
BOOT_WAIT_MAX="${BOOT_WAIT_MAX:-600}"
STEP_SETTLE_SECS="${STEP_SETTLE_SECS:-1}"

# ---------------------------------------------------------------------------
# Full-config VRAM accounting
# ---------------------------------------------------------------------------
# util_percent <0.xx>: print the integer percent (97 for 0.97, 90 for 0.9);
# return 1 when the value is not a 0..1 decimal with up to 2 fraction digits.
# More digits are rejected instead of silently truncated (0.975 would be
# misread as 0.97 and understate the VRAM need).
util_percent() {
  local u="$1" digits
  case "$u" in
    0) printf '0\n'; return 0 ;;
    0.*) : ;;
    *) return 1 ;;
  esac
  digits="${u#0.}"
  case "$digits" in ''|*[!0-9]*) return 1 ;; esac
  [ "${#digits}" -le 2 ] || return 1
  while [ "${#digits}" -lt 2 ]; do digits="${digits}0"; done
  printf '%s\n' "$((10#$digits))"
}

# full_util_need <total-mib>: print ceil(total * FULL_GPU_MEM_UTIL), the MiB
# the full configuration asks vLLM to pre-allocate at boot.
full_util_need() {
  local total="$1" pct
  pct="$(util_percent "$FULL_GPU_MEM_UTIL")" || return 1
  printf '%s\n' "$(( (total * pct + 99) / 100 ))"
}

# gpu_mem_tot_used: print "total used" (MiB, numeric) from nvidia-smi, or
# nothing when the query fails. Shared by the pre-boot gate and the post-boot
# confirmation.
gpu_mem_tot_used() {
  local line total used
  line="$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)"
  total="${line%%,*}"
  used="${line#*,}"
  used="$(printf '%s' "$used" | trim)"
  case "$total" in ''|*[!0-9]*) return 1 ;; esac
  case "$used" in ''|*[!0-9]*) used=0 ;; esac
  printf '%s %s\n' "$total" "$used"
}

# suggest_lower_configs [last-ok]: concrete lower configurations when the
# target does not fit, plus the generic close-programs / WSL restart fix.
# With a known last-good context length the suggestions start from that
# boundary instead of hardcoded values.
suggest_lower_configs() {
  local last_ok="${1:-}"
  info "fix: close GPU-using programs on Windows (browsers, video, other models), then 'wsl --shutdown' and re-run."
  info "fix: or run a lower full-configuration, e.g.:"
  if [ -n "$last_ok" ]; then
    info "     FULL_MAX_MODEL_LEN=${last_ok} FULL_MAX_NUM_SEQS=8 FULL_GPU_MEM_UTIL=0.90 bash scripts/fullcontext.sh start --model-dir DIR"
    info "     FULL_MAX_MODEL_LEN=$((last_ok / 2)) FULL_MAX_NUM_SEQS=4 FULL_GPU_MEM_UTIL=0.85 bash scripts/fullcontext.sh start --model-dir DIR"
  else
    info "     FULL_MAX_MODEL_LEN=131072 FULL_MAX_NUM_SEQS=8  FULL_GPU_MEM_UTIL=0.90 bash scripts/fullcontext.sh start --model-dir DIR"
    info "     FULL_MAX_MODEL_LEN=65536  FULL_MAX_NUM_SEQS=4  FULL_GPU_MEM_UTIL=0.85 bash scripts/fullcontext.sh start --model-dir DIR"
  fi
  info "     (shorten the ramp with CONTEXT_LADDER, e.g. CONTEXT_LADDER=\"65536 131072\"; the short service from issue 03 always remains available)"
}

# check_full_vram: gate before any full-context boot. The preflight already
# guarantees nvidia-smi works; this re-reads the current totals so the check
# reflects the moment of launch (spec: re-check VRAM headroom before launch).
# Returns 0 when the target utilization fits, 1 with a reason + lower-config
# suggestions otherwise.
check_full_vram() {
  local mem total used free need
  mem="$(gpu_mem_tot_used)" || {
    fail "full-config VRAM gate: cannot read VRAM totals"
    return 1
  }
  total="${mem%% *}"
  used="${mem#* }"
  free=$((total - used))
  need="$(full_util_need "$total")"
  if [ "$free" -ge "$need" ]; then
    ok "full-config VRAM gate: free ${free} MiB of ${total} MiB >= ${need} MiB (${FULL_GPU_MEM_UTIL} * ${total})"
    return 0
  fi
  fail "full-config VRAM gate: only ${free} MiB VRAM free; the full ${FULL_MAX_MODEL_LEN}-token config needs >= ${need} MiB (${FULL_GPU_MEM_UTIL} * ${total})"
  fail "reason: vLLM pre-allocates ${FULL_GPU_MEM_UTIL} of the GPU at boot (KV cache + weights); other programs or desktop usage reduce the free VRAM below the target."
  suggest_lower_configs
  return 1
}

# ---------------------------------------------------------------------------
# HTTP helpers for the per-step verification
# ---------------------------------------------------------------------------
# wait_http_ok <url> <max-seconds>: poll until the URL answers. A dead boot
# is only diagnosed by the caller via tail_log after this returns 1; the
# wait itself does not consult pids, because under Git Bash the fake vLLM
# execs a native Windows python whose bash wrapper PID dies immediately (a
# dead pid is not evidence the service is gone).
wait_http_ok() {
  local url="$1" max="$2" i
  for i in $(seq 1 "$((max * 5))"); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then return 0; fi
    sleep 0.2
  done
  return 1
}

# wait_port_down <host> <port> <max-seconds>: poll until nothing answers
# /v1/models on the port (a killed boot released its listener).
wait_port_down() {
  local host="$1" port="$2" max="${3:-30}" i
  for i in $(seq 1 "$((max * 5))"); do
    if ! curl -fsS --max-time 2 "http://$host:$port/v1/models" >/dev/null 2>&1; then return 0; fi
    sleep 0.2
  done
  return 1
}

# win_path <path>: translate an MSYS /tmp path to a Windows path so curl (a
# native Windows binary on Git Bash hosts) can open the file; on Linux the
# path is passed through unchanged.
win_path() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -w "$1" 2>/dev/null
  else
    printf '%s\n' "$1"
  fi
}

# http_status <url> [POST <json-body-file>]: print the HTTP status code (000
# when curl itself fails). POST bodies are read from a file (--data-binary
# @file): the long prompts used by the boundary probe exceed the Windows
# command-line length limit, so the body must never travel as an argument.
http_status() {
  local url="$1" body_file="${3:-}" body_file_win out rc
  if [ "${2:-}" = "POST" ]; then
    out="$(curl -sS --max-time 120 -o /dev/null -w '%{http_code}' -X POST "$url" -H 'Content-Type: application/json' --data-binary "@$(win_path "$body_file")" 2>/dev/null)"
  else
    out="$(curl -sS --max-time 10 -o /dev/null -w '%{http_code}' "$url" 2>/dev/null)"
  fi
  rc=$?
  if [ "$rc" -eq 0 ]; then printf '%s\n' "$out"; else printf '000\n'; fi
}

# chat_body_file <file> <model> <content>: write a minimal OpenAI chat
# completion JSON to a file (long content must not become a command-line
# argument). max_tokens keeps real-vLLM sampling short (the fake server
# ignores it).
chat_body_file() {
  printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":16}\n' "$2" "$3" > "$1"
}

# error_message <url> <json-body-file>: print the error text of a rejected
# completion (the boundary probe uses it to show WHY the request was
# rejected, e.g. "maximum context length exceeded").
error_message() {
  curl -sS --max-time 120 -X POST "$1" -H 'Content-Type: application/json' --data-binary "@$(win_path "$2")" 2>/dev/null \
    | grep -oiE 'maximum context length|context length|too (many|long)|exceed' | head -1
}

# models_max_len <url>: print max_model_len reported by /v1/models (empty when
# the endpoint does not answer or the field is absent).
models_max_len() {
  local url="$1" body
  body="$(curl -fsS --max-time 10 "$url" 2>/dev/null)" || return 1
  printf '%s\n' "$body" | grep -oE '"max_model_len"[[:space:]]*:[[:space:]]*[0-9]+' | grep -oE '[0-9]+' | head -1
}

# ---------------------------------------------------------------------------
# Per-boot helpers
# ---------------------------------------------------------------------------
# serve_argv_full <venv> <model-dir> <host> <port> <model-name> <max-len>
#                  <max-seqs>: the issue 04 argv - the shared serve_argv
# (fixed modelopt/FP8/trust-remote-code) plus the GPU memory utilization cap
# of the full configuration. Util is passed as the vLLM float, e.g. 0.97.
serve_argv_full() {
  serve_argv "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$FULL_GPU_MEM_UTIL"
}

# boot_vllm_background <prefix> <model-dir> <host> <port> <max-len> <max-seqs>
#                      <logfile>: launch vLLM in the background with the full
# environment (prepare_vllm_env). The PID is stored in the global BOOT_PID
# (NOT printed to stdout, which belongs to the console); callers read
# BOOT_PID. An empty logfile means the final instance: it inherits
# stdout/stderr (the service console), like serve.sh start. Returns 1 when
# the venv launcher is missing. The vLLM process stays a child of the current
# shell so that `wait "$BOOT_PID"` works later.
BOOT_PID=""
boot_vllm_background() {
  local prefix="$1" model_dir="$2" host="$3" port="$4" max_len="$5" max_seqs="$6" logfile="$7"
  local venv="$prefix/venv" mname
  if [ ! -x "$venv/bin/vllm" ]; then
    fail "vllm launcher not found at $venv/bin/vllm"
    info "fix: re-create the runtime: bash scripts/wsl2-env.sh create --prefix $prefix"
    return 1
  fi
  prepare_vllm_env "$venv"
  mname="$(served_model_name "$model_dir")"
  local -a ARGS=()
  while IFS= read -r arg; do ARGS+=("$arg"); done < <(serve_argv_full "$venv" "$model_dir" "$host" "$port" "$mname" "$max_len" "$max_seqs")
  if [ -n "$logfile" ]; then
    mkdir -p "$(dirname "$logfile")"
    "${ARGS[@]}" >>"$logfile" 2>&1 &
  else
    "${ARGS[@]}" &
  fi
  BOOT_PID="$!"
  return 0
}

# stop_boot <pid> <host> <port>: stop one validated boot and wait for its
# port to free up so the next step can bind it. Under Git Bash the fake vLLM
# test double execs a native Windows python that outlives the bash wrapper
# PID, so when the port stays busy after `kill` the listener is looked up by
# port and terminated (taskkill on Windows, fuser elsewhere). On real WSL2
# Linux vLLM is a normal child process and the first kill is enough.
stop_boot() {
  local pid="$1" host="$2" port="$3" wpid
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  if ! wait_port_down "$host" "$port" 3; then
    wpid="$(netstat -ano 2>/dev/null | awk -v p=":$port" '$1=="TCP" && $2 ~ (p"$") && $4=="LISTENING" {print $5; exit}')"
    if [ -n "$wpid" ]; then
      if command -v taskkill >/dev/null 2>&1; then
        taskkill //F //PID "$wpid" >/dev/null 2>&1 || true
      elif command -v fuser >/dev/null 2>&1; then
        fuser -k "${port}/tcp" >/dev/null 2>&1 || true
      else
        kill -9 "$wpid" 2>/dev/null || true
      fi
    fi
  fi
  wait_port_down "$host" "$port" 15
  sleep "$STEP_SETTLE_SECS"
}

# verify_step <step-name> <host> <port> <max-len> <model-name>: wait for the
# booted server, confirm the effective context via /v1/models, and run a short
# in-bound chat request. Prints OK/FAIL lines; returns 1 on any failure.
verify_step() {
  local name="$1" host="$2" port="$3" max_len="$4" mname="$5"
  local base="http://$host:$port/v1" mlen code body_file
  if ! wait_http_ok "$base/models" "$BOOT_WAIT_MAX"; then
    fail "$name: server did not answer $base/models within ${BOOT_WAIT_MAX}s"
    return 1
  fi
  mlen="$(models_max_len "$base/models")"
  if [ "$mlen" = "$max_len" ]; then
    ok "$name: context in effect (max_model_len=$mlen)"
  else
    fail "$name: /v1/models reports max_model_len='${mlen:-<none>}', expected $max_len"
    return 1
  fi
  body_file="$(mktemp)"
  chat_body_file "$body_file" "$mname" "hello at $name"
  code="$(http_status "$base/chat/completions" POST "$body_file")"
  rm -f "$body_file"
  if [ "$code" = "200" ]; then
    ok "$name: in-bound request returned HTTP 200"
  else
    fail "$name: in-bound request returned HTTP ${code}, expected 200"
    return 1
  fi
}

# probe_boundary <host> <port> <max-len> <model-name> <overflow-bytes>: send a
# prompt beyond max_model_len and require an understandable 4xx rejection
# (acceptance: over-boundary requests end with an error, not a silent exit or
# a 200). Returns 0 on the expected rejection, 1 otherwise.
probe_boundary() {
  local host="$1" port="$2" max_len="$3" mname="$4" bytes="$5"
  local base="http://$host:$port/v1" text body_file code msg
  text="$(head -c "$bytes" /dev/zero | tr '\0' 'x')"
  body_file="$(mktemp)"
  chat_body_file "$body_file" "$mname" "$text"
  code="$(http_status "$base/chat/completions" POST "$body_file")"
  case "$code" in
    4*)
      msg="$(error_message "$base/chat/completions" "$body_file")"
      if [ -n "$msg" ]; then
        ok "boundary probe: over-limit prompt (${bytes} bytes > ${max_len}-token context) rejected with HTTP ${code} ('$msg')"
      else
        ok "boundary probe: over-limit prompt (${bytes} bytes > ${max_len}-token context) rejected with HTTP ${code}"
      fi
      rm -f "$body_file"
      return 0
      ;;
  esac
  rm -f "$body_file"
  fail "boundary probe: over-limit prompt (${bytes} bytes > ${max_len}-token context) returned HTTP ${code}, expected a 4xx rejection"
  info "reason: vLLM must reject prompts beyond max_model_len with an understandable error (acceptance 04), not answer or exit silently."
  return 1
}

# tail_log <logfile>: print the last lines of a failed boot's log for the
# user to act on.
tail_log() {
  local logfile="$1"
  info "last lines of $logfile:"
  tail -n 6 "$logfile" 2>/dev/null | sed 's/^/       | /'
}

# ---------------------------------------------------------------------------
# The validation sequence
# ---------------------------------------------------------------------------
# step_section <title>: print the step banner with a dynamic counter (the
# ladder length is configurable, so the total is computed, not hardcoded).
STEP_CUR=0
STEP_TOTAL=0
step_section() {
  STEP_CUR=$((STEP_CUR + 1))
  section "Step ${STEP_CUR}/${STEP_TOTAL}: $1"
}

# plan_step <kind> <prefix> <model-dir> <host> <port> <max-len> <max-seqs>:
# print one planned boot (dry-run only).
plan_step() {
  local kind="$1" prefix="$2" model_dir="$3" host="$4" port="$5" max_len="$6" max_seqs="$7"
  local venv="$prefix/venv" mname
  mname="$(served_model_name "$model_dir")"
  local -a ARGS=()
  while IFS= read -r arg; do ARGS+=("$arg"); done < <(serve_argv_full "$venv" "$model_dir" "$host" "$port" "$mname" "$max_len" "$max_seqs")
  info "plan: $kind -- max-model-len $max_len, max-num-seqs $max_seqs, gpu-memory-utilization $FULL_GPU_MEM_UTIL"
  run_or_dry "${ARGS[@]}"
}

# run_smoke <prefix> <model-dir> <host> <port>: boot the short-context config
# (issue 03 values, target utilization), verify it, probe the over-boundary
# error, then stop. The full configuration must not proceed unless this
# passes (acceptance 04-1).
run_smoke() {
  local prefix="$1" model_dir="$2" host="$3" port="$4"
  local mname len seqs log pid
  mname="$(served_model_name "$model_dir")"
  len="$SHORT_MAX_MODEL_LEN"
  seqs="$SHORT_MAX_NUM_SEQS"
  log="$prefix/logs/fullcontext-smoke.log"
  step_section "short-context smoke test (issue 03 config, ${len} tokens)"
  info "Model: $model_dir (raw ModelOpt NVFP4 safetensors); offline enforced"
  boot_vllm_background "$prefix" "$model_dir" "$host" "$port" "$len" "$seqs" "$log" || return 1
  pid="$BOOT_PID"
  if ! verify_step "smoke-${len}" "$host" "$port" "$len" "$mname"; then
    tail_log "$log"
    stop_boot "$pid" "$host" "$port"
    return 1
  fi
  if ! probe_boundary "$host" "$port" "$len" "$mname" "$SMOKE_OVERFLOW_BYTES"; then
    tail_log "$log"
    stop_boot "$pid" "$host" "$port"
    return 1
  fi
  stop_boot "$pid" "$host" "$port"
  ok "short-context smoke test passed: boot + in-bound request + over-boundary rejection all behave as expected"
}

# run_ramp <prefix> <model-dir> <host> <port> <model-name>: boot every context
# length in CONTEXT_LADDER in order and verify each. On failure reports the
# machine's actual boundary (the last verified length) with lower-config
# suggestions, and returns 1.
run_ramp() {
  local prefix="$1" model_dir="$2" host="$3" port="$4" mname="$5"
  local len seqs log pid last_ok="$SHORT_MAX_MODEL_LEN"
  for len in $CONTEXT_LADDER; do
    seqs=1
    log="$prefix/logs/fullcontext-ramp-${len}.log"
    step_section "ramp to context ${len} (max-num-seqs ${seqs})"
    boot_vllm_background "$prefix" "$model_dir" "$host" "$port" "$len" "$seqs" "$log" || return 1
    pid="$BOOT_PID"
    if ! verify_step "ramp-${len}" "$host" "$port" "$len" "$mname"; then
      tail_log "$log"
      stop_boot "$pid" "$host" "$port"
      fail "ramp step ${len} failed to start; the machine's actual context boundary is at ${last_ok} tokens."
      info "reason: the boot at ${len} tokens could not come up with ${FULL_GPU_MEM_UTIL} GPU memory utilization (weights + KV cache) - either VRAM is short or the environment changed mid-ramp."
      suggest_lower_configs "$last_ok"
      return 1
    fi
    stop_boot "$pid" "$host" "$port"
    last_ok="$len"
  done
}

# confirm_vram: sample the current VRAM usage and report the utilization
# against the target (informational; warns when the sample is far off).
confirm_vram() {
  local mem total used ratio target tol
  mem="$(gpu_mem_tot_used)" || {
    warn "cannot confirm VRAM utilization (nvidia-smi totals unreadable)"
    return 0
  }
  total="${mem%% *}"
  used="${mem#* }"
  target="$(util_percent "$FULL_GPU_MEM_UTIL")"
  tol="$VRAM_UTIL_TOLERANCE_PCT"
  if [ "$total" -gt 0 ]; then
    ratio=$((used * 100 / total))
    if [ "$ratio" -ge $((target - tol)) ] && [ "$ratio" -le $((target + tol)) ]; then
      ok "VRAM utilization ${ratio}% of ${total} MiB (target ${target}%, tolerance ${tol} points)"
    else
      warn "VRAM utilization ${ratio}% of ${total} MiB (target ${target}%); the sample may predate the KV cache growing, or other programs hold memory."
    fi
  else
    warn "cannot confirm VRAM utilization (nvidia-smi totals unreadable)"
  fi
}

# confirm_concurrency <base-url> <model> <n>: send n concurrent short chat
# requests and require every one to complete with HTTP 200, exercising the
# scheduler's sequence slots (the full config targets 16 concurrent
# sequences). The fake server answers concurrently; a real vLLM with enough
# slots does the same. Each request runs in a tracked subshell and is waited
# on by PID: an argument-less `wait` would also block on unrelated children
# (e.g. a previously killed boot whose exec'd process lingers), which under
# Git Bash never exit.
confirm_concurrency() {
  local base="$1" mname="$2" n="$3" body_file i pid ok_n=0 fail_n=0
  local -a PIDS=()
  body_file="$(mktemp)"
  chat_body_file "$body_file" "$mname" "concurrency probe"
  for i in $(seq 1 "$n"); do
    (
      code="$(http_status "$base/chat/completions" POST "$body_file")"
      [ "$code" = "200" ] && exit 0 || exit 1
    ) &
    PIDS+=("$!")
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null
    if [ "$?" -eq 0 ]; then ok_n=$((ok_n + 1)); else fail_n=$((fail_n + 1)); fi
  done
  rm -f "$body_file"
  if [ "$fail_n" -eq 0 ] && [ "$ok_n" -eq "$n" ]; then
    ok "concurrency probe: ${n}/${n} concurrent requests returned HTTP 200 (max-num-seqs ${FULL_MAX_NUM_SEQS} in effect)"
    return 0
  fi
  fail "concurrency probe: only ${ok_n}/${n} concurrent requests returned HTTP 200 (expected all; max-num-seqs ${FULL_MAX_NUM_SEQS})"
  return 1
}

# run_full <prefix> <model-dir> <host> <port> <model-name>: boot the full
# configuration, verify context + concurrency + a long in-bound request + the
# over-boundary error + VRAM utilization, then keep the service running
# (Ctrl-C stops it).
run_full() {
  local prefix="$1" model_dir="$2" host="$3" port="$4" mname="$5"
  local len seqs log pid code base
  len="$FULL_MAX_MODEL_LEN"
  seqs="$FULL_MAX_NUM_SEQS"
  log="" # the final instance inherits stdout/stderr, like serve.sh start
  base="http://$host:$port/v1"
  step_section "full configuration (${len} tokens, ${seqs} concurrent sequences, ${FULL_GPU_MEM_UTIL} utilization)"
  boot_vllm_background "$prefix" "$model_dir" "$host" "$port" "$len" "$seqs" "$log" || return 1
  pid="$BOOT_PID"

  # First verify the boot (context + in-bound request); the "stays running"
  # trap is only set right before the wait, after verification succeeded.
  if ! verify_step "full-config" "$host" "$port" "$len" "$mname"; then
    stop_boot "$pid" "$host" "$port"
    fail "full-config boot failed; the full ${len}-token configuration was NOT started."
    # The ramp verified up to $len at 1 sequence; lower the concurrency and
    # utilization first, keeping the verified context boundary (a critical
    # machine may fit the context but not 16 sequences at 0.97).
    suggest_lower_configs "$len"
    return 1
  fi

  # Confirm the target concurrency with actual concurrent requests (not just
  # echoing the launch flag).
  if ! confirm_concurrency "$base" "$mname" "$CONCURRENT_PROOF"; then
    stop_boot "$pid" "$host" "$port"
    fail "full-config concurrency probe failed; the full ${len}-token configuration was NOT started."
    return 1
  fi

  # Confirm a long in-bound request (below the boundary) returns normally.
  local proof body_file
  proof="$(head -c "$LONG_PROOF_BYTES" /dev/zero | tr '\0' 'x')"
  body_file="$(mktemp)"
  chat_body_file "$body_file" "$mname" "$proof"
  code="$(http_status "$base/chat/completions" POST "$body_file")"
  rm -f "$body_file"
  if [ "$code" = "200" ]; then
    ok "full-config: long-context request (${LONG_PROOF_BYTES} bytes, well under the ${len}-token boundary) returned HTTP 200"
  else
    stop_boot "$pid" "$host" "$port"
    fail "full-config: long-context request returned HTTP ${code}, expected 200"
    return 1
  fi

  # Confirm the over-boundary error at the FULL context (not only at smoke):
  # beyond the boundary the request must end with an understandable 4xx.
  if ! probe_boundary "$host" "$port" "$len" "$mname" "$FULL_OVERFLOW_BYTES"; then
    stop_boot "$pid" "$host" "$port"
    fail "full-config: over-boundary prompt was not rejected; the full ${len}-token configuration was NOT started."
    return 1
  fi

  confirm_vram
  ok "full-context configuration verified and enabled"
  ok "full-context service is running: http://$host:$port/v1 (Ctrl-C to stop)"

  # The verified instance IS the foreground service now: wait until the user
  # stops it. (Under Git Bash a fake vLLM may have exec'd a native process, so
  # `wait` can return early; the launcher then idles until SIGINT/SIGTERM.)
  trap 'kill -TERM "$pid" 2>/dev/null' INT TERM
  wait "$pid" 2>/dev/null || true
  while kill -0 "$pid" 2>/dev/null; do sleep 1; done
  trap - INT TERM
  return 0
}

# run_fullcontext_validation <prefix> <model-dir> <host> <port>: the controlled
# start path of issue 04. Dry-run prints the whole plan instead of booting.
run_fullcontext_validation() {
  local prefix="$1" model_dir="$2" host="$3" port="$4"
  local mname rc n_ladder

  if [ "$DRY_RUN" -eq 1 ]; then
    section "Full-context validation plan (issue 04, dry-run)"
    info "After preflight READY and the full-config VRAM gate, the validation would boot, verify and stop:"
    plan_step "short-context smoke test (issue 03 config)" "$prefix" "$model_dir" "$host" "$port" "$SHORT_MAX_MODEL_LEN" "$SHORT_MAX_NUM_SEQS"
    info "plan: boundary probe - an over-limit prompt (${SMOKE_OVERFLOW_BYTES} bytes > ${SHORT_MAX_MODEL_LEN} tokens) must be rejected with a 4xx error"
    for len in $CONTEXT_LADDER; do
      plan_step "ramp step" "$prefix" "$model_dir" "$host" "$port" "$len" "1"
    done
    plan_step "full configuration (stays running)" "$prefix" "$model_dir" "$host" "$port" "$FULL_MAX_MODEL_LEN" "$FULL_MAX_NUM_SEQS"
    info "plan: verify effective context via /v1/models, ${CONCURRENT_PROOF} concurrent requests, a long in-bound request (${LONG_PROOF_BYTES} bytes), the over-boundary error (${FULL_OVERFLOW_BYTES} bytes), and the VRAM utilization target"
    return 0
  fi

  mname="$(served_model_name "$model_dir")"
  n_ladder="$(printf '%s\n' $CONTEXT_LADDER | grep -c . || true)"
  STEP_TOTAL=$((1 + n_ladder + 1))

  if ! run_smoke "$prefix" "$model_dir" "$host" "$port"; then
    fail "full-context NOT started: the short-context smoke test did not pass."
    info "fix: re-run the issue 03 short service (scripts/serve.sh start) to confirm the environment, then retry."
    return 1
  fi
  if ! run_ramp "$prefix" "$model_dir" "$host" "$port" "$mname"; then
    fail "full-context NOT started: the ramp did not reach ${FULL_MAX_MODEL_LEN} tokens."
    return 1
  fi
  run_full "$prefix" "$model_dir" "$host" "$port" "$mname"
  return $?
}
