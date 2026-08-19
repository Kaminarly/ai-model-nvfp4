#!/usr/bin/env bash
# Test runner for issue 04 (controlled full 262144-token context launch).
# Bash only; uses the same fake toolchain as the issue 01-03 suites, plus the
# extended fake vLLM API server (tests/fakebin/vllm) that reports
# max_model_len, rejects over-limit prompts and can fake boot failures. Runs
# without WSL2 or a GPU; the live validation-path tests need a real python3 +
# curl and are skipped when either is missing.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/scripts/fullcontext.sh"
FAKEBIN="$ROOT/tests/fakebin"
FIXTURES="$ROOT/tests/fixtures"
TMP_ROOT="$(mktemp -d)"
SERVE_PID=""
trap 'if [ -n "$SERVE_PID" ]; then kill "$SERVE_PID" 2>/dev/null; fi; rm -rf "$TMP_ROOT"' EXIT

chmod +x "$FAKEBIN"/* 2>/dev/null || true

PASS=0
FAIL=0
say() { printf '%s\n' "$*"; }

run_env() { # run_env [VAR=val ...] -- <script args...>
  local vars=()
  while [ "$1" != "--" ]; do vars+=("$1"); shift; done
  shift
  env "${vars[@]}" PATH="$FAKEBIN:$PATH" bash "$SCRIPT" "$@" 2>&1
}

run_green() { # run_green [extra VAR=val ...] -- <script args...>
  run_env OS_RELEASE_FILE="$FIXTURES/os-release-ubuntu" \
          WSL_INTEROP=/run/WSL/1 \
          WIN_VER_CMD="$FAKEBIN/fake-win-cmd" \
          FAKE_PY_INCLUDE="$FIXTURES/pyinclude" \
          "$@"
}

expect_exit() { # expect_exit <expected> <actual> <name>
  if [ "$2" -eq "$1" ]; then PASS=$((PASS + 1)); say "ok   - $3"
  else FAIL=$((FAIL + 1)); say "FAIL - $3 (expected exit $1, got $2)"; fi
}

expect_contains() { # expect_contains <output> <needle> <name>
  case "$1" in
    *"$2"*) PASS=$((PASS + 1)); say "ok   - $3" ;;
    *) FAIL=$((FAIL + 1)); say "FAIL - $3: output misses '$2'"; printf '%s\n' "$1" | sed 's/^/       | /' ;;
  esac
}

expect_not_contains() { # expect_not_contains <output> <needle> <name>
  case "$1" in
    *"$2"*) FAIL=$((FAIL + 1)); say "FAIL - $3: output unexpectedly contains '$2'"; printf '%s\n' "$1" | sed 's/^/       | /' ;;
    *) PASS=$((PASS + 1)); say "ok   - $3" ;;
  esac
}

section() { say ""; say "== $1 =="; }

# wait_for_url <url> : poll until the URL answers, max ~20s.
wait_for_url() {
  local url="$1" i
  for i in $(seq 1 100); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then return 0; fi
    sleep 0.2
  done
  return 1
}

# wait_for_max_len <url> <expected>: poll until /v1/models reports the
# expected max_model_len (max ~60s). The green path boots smoke + ramps +
# full in sequence, so this waits for the FINAL full-context instance.
wait_for_max_len() {
  local url="$1" expect="$2" i body
  for i in $(seq 1 300); do
    body="$(curl -fsS --max-time 2 "$url" 2>/dev/null)" || { sleep 0.2; continue; }
    case "$body" in
      *"\"max_model_len\": $expect"*) return 0 ;;
    esac
    sleep 0.2
  done
  return 1
}

# kill_port <port>: kill whatever listens on the port. Under Git Bash the
# fake vLLM execs a native Windows python that outlives the bash wrapper PID,
# so the test frees the port explicitly after killing the launcher script.
kill_port() {
  local wpid
  wpid="$(netstat -ano 2>/dev/null | awk -v p=":$1" '$1=="TCP" && $2 ~ (p"$") && $4=="LISTENING" {print $5; exit}')"
  if [ -n "$wpid" ]; then
    taskkill //F //PID "$wpid" >/dev/null 2>&1 || true
  fi
}

# ---------------------------------------------------------------------------
# Fixtures: a fake runtime venv (via issue 01 create) and fake model dirs.
# ---------------------------------------------------------------------------
section "setup: fake venv + model fixtures"
PFX="$TMP_ROOT/pfx"
out="$(env OS_RELEASE_FILE="$FIXTURES/os-release-ubuntu" WSL_INTEROP=/run/WSL/1 \
      WIN_VER_CMD="$FAKEBIN/fake-win-cmd" \
      FAKE_NVCC_SRC="$FIXTURES/venv/bin/nvcc" FAKE_NVCC_SITE_PACKAGES=1 \
      FAKE_PIP_LOG="$TMP_ROOT/pip.log" PATH="$FAKEBIN:$PATH" \
      bash "$ROOT/scripts/wsl2-env.sh" create --prefix "$PFX")"; code=$?
expect_exit 0 "$code" "create fake venv for fullcontext tests"

INDEX='{"metadata":{"total_size":0},"weight_map":{"layers.0.weight":"model-00001-of-00003.safetensors","layers.1.weight":"model-00002-of-00003.safetensors","layers.2.weight":"model-00003-of-00003.safetensors"}}'
# make_safetensors <file> <data_bytes>: minimal valid safetensors file
# (8-byte LE header length + header JSON + zero data).
make_safetensors() {
  local file="$1" data_bytes="${2:-1024}"
  local header n b1 b2 b3 b4 prefix
  header='{"t":{"dtype":"F32","shape":['"$data_bytes"'],"data_offsets":[0,'"$((data_bytes * 4))"']}}'
  n=${#header}
  b1=$((n & 255)); b2=$(((n >> 8) & 255)); b3=$(((n >> 16) & 255)); b4=$(((n >> 24) & 255))
  prefix=$(printf '\\x%02x\\x%02x\\x%02x\\x%02x\\x00\\x00\\x00\\x00' "$b1" "$b2" "$b3" "$b4")
  printf '%b%s' "$prefix" "$header" > "$file"
  head -c "$((data_bytes * 4))" /dev/zero >> "$file"
}
make_model() { # make_model <dir>
  local d="$1"
  mkdir -p "$d"
  for f in model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors; do
    make_safetensors "$d/$f" 1024
  done
  printf '%s\n' "$INDEX" > "$d/model.safetensors.index.json"
  printf '{"model_type":"qwen3_5"}\n' > "$d/config.json"
  printf '{"algorithm":"modelopt","quant_type":"NVFP4"}\n' > "$d/hf_quant_config.json"
  printf '{"model_max_length":262144}\n' > "$d/generation_config.json"
  printf '{"tokenizer_class":"Qwen2Tokenizer"}\n' > "$d/tokenizer_config.json"
  printf '{"version":"1.0"}\n' > "$d/tokenizer.json"
  printf '{{- message }}\n' > "$d/chat_template.jinja"
}

MODEL_OK="$TMP_ROOT/model-ok"; make_model "$MODEL_OK"
MODEL_NO_SHARD="$TMP_ROOT/model-no-shard"; make_model "$MODEL_NO_SHARD"; rm "$MODEL_NO_SHARD/model-00002-of-00003.safetensors"

REALPY="$(command -v python3 2>/dev/null || true)"
HAVE_LIVE=0
if [ -n "$REALPY" ] && command -v curl >/dev/null 2>&1; then HAVE_LIVE=1; fi

# ---------------------------------------------------------------------------
# 1. CLI guardrails
# ---------------------------------------------------------------------------
section "fullcontext: cli guardrails"
out="$(run_green -- start --prefix "$PFX")"; code=$?
expect_exit 2 "$code" "missing --model-dir exits 2"
expect_contains "$out" "missing required option: --model-dir" "usage error names the option"

out="$(run_green -- start --model-dir "$MODEL_OK" --port abc)"; code=$?
expect_exit 2 "$code" "invalid port exits 2"
expect_contains "$out" "invalid port" "port validated"

out="$(run_green -- bogus)"; code=$?
expect_exit 2 "$code" "unknown command exits 2"

out="$(run_green -- start --bogus --model-dir "$MODEL_OK")"; code=$?
expect_exit 2 "$code" "unknown option exits 2"

out="$(run_green -- help)"; code=$?
expect_exit 0 "$code" "help exits 0"

out="$(run_green FULL_GPU_MEM_UTIL=abc -- start --dry-run --prefix "$PFX" --model-dir "$MODEL_OK")"; code=$?
expect_exit 2 "$code" "invalid FULL_GPU_MEM_UTIL exits 2"
expect_contains "$out" "invalid FULL_GPU_MEM_UTIL" "names the invalid variable"

out="$(run_green -- start --dry-run --prefix "$PFX" --model-dir "$MODEL_OK" --host 0.0.0.0)"; code=$?
expect_exit 2 "$code" "non-loopback host refused"
expect_contains "$out" "refusing non-loopback bind address" "non-loopback bind rejected"

# ---------------------------------------------------------------------------
# 2. Preflight gate: full config refuses to start when preflight fails
# ---------------------------------------------------------------------------
section "fullcontext: refuses startup when preflight fails"
out="$(run_green -- start --prefix "$PFX" --model-dir "$MODEL_NO_SHARD")"; code=$?
expect_exit 1 "$code" "missing shard blocks startup"
expect_contains "$out" "missing required file: model-00002-of-00003.safetensors" "keeps the failure reason"
expect_contains "$out" "preflight result: NOT READY" "preflight NOT READY boundary shown"
expect_contains "$out" "full-context service NOT started" "startup refused"

out="$(run_green -- start --prefix "$PFX" --model-dir "$TMP_ROOT/no-such-dir")"; code=$?
expect_exit 1 "$code" "invalid model path blocks startup"
expect_contains "$out" "model directory not found" "clear reason for bad path"

# ---------------------------------------------------------------------------
# 3. Full-config VRAM gate + lower-config suggestions
# ---------------------------------------------------------------------------
section "fullcontext: full-config VRAM gate"
# total 32607 MiB, used 11000 MiB -> free 21607 MiB: enough for the preflight
# gate (>= 20480) but below the full-config need (31629 = 0.97 * 32607), so
# the start is blocked at the full-config VRAM gate with suggestions.
out="$(run_green FAKE_GPU_MEMORY_LINE="32607, 11000" -- start --prefix "$PFX" --model-dir "$MODEL_OK")"; code=$?
expect_exit 1 "$code" "insufficient VRAM for the full config exits 1"
expect_contains "$out" "full-config VRAM gate: only 21607 MiB VRAM free" "reports the actual headroom"
expect_contains "$out" "needs >= 31629 MiB (0.97 * 32607)" "states the target need"
expect_contains "$out" "FULL_MAX_MODEL_LEN=131072" "suggests a lower configuration"
expect_contains "$out" "FULL_MAX_MODEL_LEN=65536" "suggests a second lower configuration"
expect_contains "$out" "wsl --shutdown" "suggests closing GPU programs + WSL restart"
expect_contains "$out" "full-context service NOT started" "startup refused"

# ---------------------------------------------------------------------------
# 4. Dry-run: the controlled plan (smoke -> ramp -> full)
# ---------------------------------------------------------------------------
section "fullcontext: dry-run validation plan"
out="$(run_green -- start --dry-run --prefix "$PFX" --model-dir "$MODEL_OK")"; code=$?
expect_exit 0 "$code" "dry-run green exits 0"
expect_contains "$out" "preflight result: READY" "preflight gate passes"
expect_contains "$out" "plan: short-context smoke test" "smoke step planned"
expect_contains "$out" "--max-model-len 8192 --max-num-seqs 1" "smoke uses the issue 03 config"
expect_contains "$out" "--max-model-len 32768 --max-num-seqs 1" "first ramp step planned"
expect_contains "$out" "--max-model-len 65536 --max-num-seqs 1" "second ramp step planned"
expect_contains "$out" "--max-model-len 131072 --max-num-seqs 1" "third ramp step planned"
expect_contains "$out" "--max-model-len 262144 --max-num-seqs 1" "fourth ramp step planned"
expect_contains "$out" "--max-model-len 262144 --max-num-seqs 16" "full config planned with 16 sequences"
expect_contains "$out" "--gpu-memory-utilization 0.97" "full config plans 0.97 utilization"
expect_contains "$out" "boundary probe" "over-boundary probe planned"
expect_contains "$out" "--quantization modelopt" "fixed modelopt quantization"
expect_contains "$out" "--kv-cache-dtype fp8" "fixed FP8 KV cache"
expect_contains "$out" "--trust-remote-code" "trust remote code set"
expect_contains "$out" "HF_HUB_OFFLINE=1" "offline env reported"
expect_contains "$out" "MAX_JOBS=1" "FlashInfer ninja build serialized"
expect_not_contains "$out" "gguf" "no GGUF conversion"
expect_not_contains "$out" "awq" "no AWQ conversion"
expect_not_contains "$out" "gptq" "no GPTQ conversion"

# ---------------------------------------------------------------------------
# 5. Guardrails: no downloads / conversion in the full sourced chain.
#    The fullcontext chain legitimately uses curl to probe the LOCAL OpenAI
#    endpoint (wait for boot, /v1/models, chat completions); what is
#    forbidden is downloading anything (curl -o/-O/--output, wget, HF tools,
#    git clone). The issue 01/02/03 scripts keep their stricter no-curl-at-all
#    assertion (they have no reason to touch HTTP).
# ---------------------------------------------------------------------------
section "fullcontext: no network behavior in scripts (full sourced chain)"
for f in "$SCRIPT" "$ROOT/scripts/lib/fullcontext-lib.sh"; do
  for bad in snapshot_download huggingface-cli "wget" "git clone" "curl -o" "curl -O" "curl --output"; do
    if grep -q "$bad" "$f"; then FAIL=$((FAIL + 1)); say "FAIL - $f contains forbidden '$bad'"
    else PASS=$((PASS + 1)); say "ok   - $f has no '$bad'"; fi
  done
done
for f in "$ROOT/scripts/lib/serve-lib.sh" "$ROOT/scripts/lib/preflight-lib.sh" "$ROOT/scripts/lib/wsl2-env-lib.sh"; do
  for bad in snapshot_download huggingface-cli "wget" "curl " "git clone"; do
    if grep -q "$bad" "$f"; then FAIL=$((FAIL + 1)); say "FAIL - $f contains forbidden '$bad'"
    else PASS=$((PASS + 1)); say "ok   - $f has no '$bad'"; fi
  done
done

# ---------------------------------------------------------------------------
# 6. Live controlled validation (needs a real python3 + curl; skipped otherwise)
# ---------------------------------------------------------------------------
section "fullcontext: live controlled validation (fake vLLM)"
if [ "$HAVE_LIVE" -eq 0 ]; then
  say "skip - live validation-path tests need a real python3 and curl (found python3='${REALPY:-<none>}')"
else
  cp "$FAKEBIN/vllm" "$PFX/venv/bin/vllm"
  chmod +x "$PFX/venv/bin/vllm"

  # 6a. Full green path: smoke -> 3 ramp steps -> full config stays running.
  #     Each boot's argv is recorded per-port in $ARGV_DIR. VRAM 32768 MiB
  #     with nothing used satisfies the 0.97 gate (needs 31785 MiB).
  ARGV_DIR="$TMP_ROOT/argv-green"; mkdir -p "$ARGV_DIR"
  PORT=$((21000 + RANDOM % 5000))
  OUT="$TMP_ROOT/green.out"
  run_green FAKE_VLLM_LOG_DIR="$ARGV_DIR" FAKE_VLLM_REALPY="$REALPY" \
            FAKE_GPU_MEMORY_LINE="32768, 0" \
            CONTEXT_LADDER="32768 65536 131072" BOOT_WAIT_MAX=20 STEP_SETTLE_SECS=0 \
            CONCURRENT_PROOF=4 \
    -- start --prefix "$PFX" --model-dir "$MODEL_OK" --port "$PORT" \
    >"$OUT" 2>&1 &
  SERVE_PID=$!

  # Wait for the FINAL full-context instance (262144), i.e. the whole
  # smoke -> ramp -> full sequence to complete, then for the launcher to
  # print its verification OK lines (the launcher polls /v1/models and prints
  # after the server answers; acting on the first 262144 response would cut
  # the verification output short).
  if wait_for_max_len "http://127.0.0.1:$PORT/v1/models" 262144; then
    say "ok   - full-context instance (262144) answered on loopback port $PORT"
  else
    FAIL=$((FAIL + 1)); say "FAIL - full-context instance did not reach 262144 on 127.0.0.1:$PORT"
    sed 's/^/       | /' "$OUT"
  fi
  # Poll the launcher output for the "verified and enabled" marker. The full
  # config step also sends a 1.1 MB over-boundary probe (two POSTs) and the
  # concurrency probe, which can take tens of seconds under Git Bash; allow
  # up to ~120s.
  verified=0
  for i in $(seq 1 600); do
    if grep -q "full-context configuration verified and enabled" "$OUT" 2>/dev/null; then
      verified=1; break
    fi
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then break; fi
    sleep 0.2
  done
  if [ "$verified" -eq 1 ]; then
    say "ok   - launcher printed the verified-and-enabled marker"
  else
    FAIL=$((FAIL + 1)); say "FAIL - launcher did not print the verified-and-enabled marker"
    sed 's/^/       | /' "$OUT"
  fi

  # The final instance must report the full 262144-token context.
  models="$(curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)"
  if [ -n "$models" ]; then
    expect_contains "$models" "model-ok" "model list returns the loaded model"
    expect_contains "$models" '"max_model_len": 262144' "full context in effect (262144)"
  else
    FAIL=$((FAIL + 1)); say "FAIL - /v1/models returned nothing"
  fi

  # A long in-bound request (below the boundary) must return normally.
  LONG_PROOF="$(head -c 4096 /dev/zero | tr '\0' 'x')"
  chat="$(curl -fsS --max-time 5 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
          -H 'Content-Type: application/json' \
          -d "{\"model\":\"model-ok\",\"messages\":[{\"role\":\"user\",\"content\":\"$LONG_PROOF\"}],\"max_tokens\":16}" 2>/dev/null)"
  if [ -n "$chat" ]; then
    expect_contains "$chat" "offline reply to:" "long in-bound request returns a completion"
  else
    FAIL=$((FAIL + 1)); say "FAIL - long in-bound request returned nothing"
  fi

  # Stop the validation script; its trap forwards the signal to the instance.
  kill "$SERVE_PID" 2>/dev/null
  wait "$SERVE_PID" 2>/dev/null
  SERVE_PID=""
  # Git Bash: the fake vLLM execs a native python that survives the bash
  # wrapper; free the port so no residual listener stays behind.
  kill_port "$PORT"

  out="$(cat "$OUT")"
  expect_contains "$out" "preflight result: READY" "preflight gate passed before launch"
  expect_contains "$out" "full-config VRAM gate:" "VRAM gate ran"
  expect_contains "$out" "smoke-8192: context in effect (max_model_len=8192)" "smoke context verified"
  expect_contains "$out" "boundary probe: over-limit prompt (65536 bytes > 8192-token context) rejected" "over-boundary rejection verified"
  expect_contains "$out" "short-context smoke test passed" "smoke step passed"
  expect_contains "$out" "ramp-32768: context in effect" "ramp 32768 verified"
  expect_contains "$out" "ramp-65536: context in effect" "ramp 65536 verified"
  expect_contains "$out" "ramp-131072: context in effect" "ramp 131072 verified"
  expect_contains "$out" "full-config: context in effect (max_model_len=262144)" "full config context verified"
  expect_contains "$out" "concurrency probe: 4/4 concurrent requests returned HTTP 200" "concurrency confirmed with real requests"
  expect_contains "$out" "full-config: long-context request (131072 bytes" "long request inside the full config returned 200"
  expect_contains "$out" "boundary probe: over-limit prompt (1100000 bytes > 262144-token context) rejected" "full-config over-boundary rejection verified"
  expect_contains "$out" "full-context configuration verified and enabled" "full config enabled"
  expect_contains "$out" "full-context service is running" "service left running"

  # Per-port argv logs: smoke, each ramp step, full config.
  expect_contains "$(cat "$ARGV_DIR/argv-$PORT.log" 2>/dev/null)" "argv-line serve --model $MODEL_OK --quantization modelopt --kv-cache-dtype fp8 --host 127.0.0.1 --port $PORT --served-model-name model-ok --max-model-len 262144 --max-num-seqs 16 --gpu-memory-utilization 0.97 --trust-remote-code" "full instance got the exact fixed argv"
  expect_contains "$(cat "$ARGV_DIR/argv-$PORT.log" 2>/dev/null)" "HF_HUB_OFFLINE=1" "offline mode exported to the final process"

  # Diagnostics: if the green path failed, show the launcher output tail and
  # the per-port argv log so the failure is understandable without re-running.
  if ! case "$out" in *"full-context configuration verified and enabled"*) true ;; *) false ;; esac; then
    say "--- green path diagnostics (launcher output tail) ---"
    tail -n 20 "$OUT" | sed 's/^/       | /'
    say "--- green path diagnostics (argv-$PORT.log) ---"
    tail -n 20 "$ARGV_DIR/argv-$PORT.log" 2>/dev/null | sed 's/^/       | /'
  fi

  # 6b. Smoke fails to boot -> full config must NOT start.
  section "fullcontext: smoke boot failure blocks the full config"
  PORT=$((27000 + RANDOM % 5000))
  OUT="$TMP_ROOT/bootfail.out"
  out="$(run_green FAKE_VLLM_LOG_DIR="$TMP_ROOT/argv-bootfail" FAKE_VLLM_REALPY="$REALPY" FAKE_VLLM_BOOT_FAIL=1 \
        FAKE_GPU_MEMORY_LINE="32768, 0" \
        BOOT_WAIT_MAX=5 STEP_SETTLE_SECS=0 \
        -- start --prefix "$PFX" --model-dir "$MODEL_OK" --port "$PORT")"; code=$?
  expect_exit 1 "$code" "smoke boot failure exits 1"
  expect_contains "$out" "server did not answer" "reports the failed boot"
  expect_contains "$out" "short-context smoke test did not pass" "blocks at the smoke gate"
  expect_contains "$out" "full-context NOT started" "full config not started"
  expect_contains "$out" "scripts/serve.sh start" "points back to the issue 03 short service"

  # 6c. Ramp step fails -> the machine's actual boundary is reported.
  section "fullcontext: ramp failure reports the actual boundary"
  PORT=$((29000 + RANDOM % 5000))
  out="$(run_green FAKE_VLLM_LOG_DIR="$TMP_ROOT/argv-rampfail" FAKE_VLLM_REALPY="$REALPY" \
        FAKE_VLLM_FAIL_BOOT_AT=65536 CONTEXT_LADDER="32768 65536" \
        FAKE_GPU_MEMORY_LINE="32768, 0" \
        BOOT_WAIT_MAX=5 STEP_SETTLE_SECS=0 \
        -- start --prefix "$PFX" --model-dir "$MODEL_OK" --port "$PORT")"; code=$?
  expect_exit 1 "$code" "ramp failure exits 1"
  expect_contains "$out" "smoke-8192: context in effect" "smoke passed before the ramp"
  expect_contains "$out" "ramp-32768: context in effect" "32768 verified as the last good step"
  expect_contains "$out" "ramp step 65536 failed to start" "names the failing step"
  expect_contains "$out" "the machine's actual context boundary is at 32768" "reports the verified boundary"
  expect_contains "$out" "full-context NOT started" "full config not started"
  expect_contains "$out" "FULL_MAX_MODEL_LEN=32768" "suggests a configuration at the actual boundary"
  expect_contains "$out" "FULL_MAX_MODEL_LEN=16384" "suggests a lower configuration below the boundary"

  # 6d. Over-limit prompt not rejected -> boundary probe fails -> NOT started.
  section "fullcontext: boundary probe failure blocks the full config"
  PORT=$((31000 + RANDOM % 5000))
  out="$(run_green FAKE_VLLM_LOG_DIR="$TMP_ROOT/argv-noreject" FAKE_VLLM_REALPY="$REALPY" \
        FAKE_VLLM_NO_REJECT=1 FAKE_GPU_MEMORY_LINE="32768, 0" \
        BOOT_WAIT_MAX=10 STEP_SETTLE_SECS=0 \
        -- start --prefix "$PFX" --model-dir "$MODEL_OK" --port "$PORT")"; code=$?
  expect_exit 1 "$code" "missing over-boundary rejection exits 1"
  expect_contains "$out" "boundary probe: over-limit prompt (65536 bytes > 8192-token context) returned HTTP 200" "reports the missing 4xx"
  expect_contains "$out" "full-context NOT started" "full config not started"

  # 6e. Wrong effective context reported -> verification rejects the step.
  section "fullcontext: wrong effective context is rejected"
  PORT=$((33000 + RANDOM % 5000))
  out="$(run_green FAKE_VLLM_LOG_DIR="$TMP_ROOT/argv-wronglen" FAKE_VLLM_REALPY="$REALPY" \
        FAKE_VLLM_MAX_LEN_OFFSET=1000 FAKE_GPU_MEMORY_LINE="32768, 0" \
        BOOT_WAIT_MAX=10 STEP_SETTLE_SECS=0 \
        -- start --prefix "$PFX" --model-dir "$MODEL_OK" --port "$PORT")"; code=$?
  expect_exit 1 "$code" "wrong effective context exits 1"
  expect_contains "$out" "reports max_model_len='9192', expected 8192" "names the mismatch"
  expect_contains "$out" "full-context NOT started" "full config not started"
fi

say ""
say "== results: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
