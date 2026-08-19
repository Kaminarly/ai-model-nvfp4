#!/usr/bin/env bash
# Test runner for issue 03 (offline short-context service). Bash only; uses
# the same fake toolchain as tests/run-tests.sh and tests/preflight-tests.sh,
# plus a fake vLLM API server (tests/fakebin/vllm) that logs the exact launch
# argv and serves /v1/models + /v1/chat/completions. Runs without WSL2 or a
# GPU; the live endpoint tests need a real python3 + curl and are skipped when
# either is missing.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/scripts/serve.sh"
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
expect_exit 0 "$code" "create fake venv for serve tests"

INDEX='{"metadata":{"total_size":0},"weight_map":{"layers.0.weight":"model-00001-of-00003.safetensors","layers.1.weight":"model-00002-of-00003.safetensors","layers.2.weight":"model-00003-of-00003.safetensors"}}'
# make_safetensors <file> <data_bytes>: minimal valid safetensors file
# (8-byte LE header length + header JSON + zero data) for the shard-header
# integrity check that now runs in the preflight gate.
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

# ---------------------------------------------------------------------------
# 1. CLI guardrails
# ---------------------------------------------------------------------------
section "serve: cli guardrails"
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

# ---------------------------------------------------------------------------
# 2. Preflight gate: the service refuses to start when preflight fails
# ---------------------------------------------------------------------------
section "serve: refuses startup when preflight fails"
out="$(run_green -- start --prefix "$PFX" --model-dir "$MODEL_NO_SHARD")"; code=$?
expect_exit 1 "$code" "missing shard blocks startup"
expect_contains "$out" "missing required file: model-00002-of-00003.safetensors" "keeps the failure reason"
expect_contains "$out" "preflight result: NOT READY" "preflight NOT READY boundary shown"
expect_contains "$out" "service NOT started" "startup refused"

out="$(run_green -- start --prefix "$PFX" --model-dir "$TMP_ROOT/no-such-dir")"; code=$?
expect_exit 1 "$code" "invalid model path blocks startup"
expect_contains "$out" "model directory not found" "clear reason for bad path"
expect_contains "$out" "service NOT started" "startup refused"

out="$(run_green FAKE_GPU_MEMORY_LINE="32607, 12288" -- start --prefix "$PFX" --model-dir "$MODEL_OK")"; code=$?
expect_exit 1 "$code" "low free VRAM blocks startup"
expect_contains "$out" "only 20319 MiB VRAM free" "free VRAM gate applied"
expect_contains "$out" "service NOT started" "startup refused"

# ---------------------------------------------------------------------------
# 3. Dry-run: fixed vLLM settings, loopback default, offline, no conversion
# ---------------------------------------------------------------------------
section "serve: dry-run fixed vLLM command"
out="$(run_green -- start --dry-run --prefix "$PFX" --model-dir "$MODEL_OK")"; code=$?
expect_exit 0 "$code" "dry-run green exits 0"
expect_contains "$out" "preflight result: READY" "preflight gate passes"
expect_contains "$out" "vllm serve" "launches the vllm CLI"
expect_contains "$out" "--model $MODEL_OK" "reads the user model dir directly"
expect_contains "$out" "--quantization modelopt" "fixed modelopt quantization"
expect_contains "$out" "--kv-cache-dtype fp8" "fixed FP8 KV cache"
expect_contains "$out" "--host 127.0.0.1" "loopback bind by default"
expect_contains "$out" "--served-model-name model-ok" "served name from model dir basename"
expect_contains "$out" "--max-model-len 8192" "short context default"
expect_contains "$out" "--max-num-seqs 1" "single-request config"
expect_contains "$out" "--trust-remote-code" "trust remote code set"
expect_contains "$out" "http://127.0.0.1:8000/v1" "endpoint URL printed"
expect_contains "$out" "HF_HUB_OFFLINE=1" "offline env reported"
expect_contains "$out" "CUDA_HOME:" "venv CUDA tree exported for JIT"
expect_contains "$out" "MAX_JOBS=1" "FlashInfer ninja build serialized (WSL RAM OOM guard)"
if [ -f "$PFX/venv/lib/python3.14/site-packages/tvm_ffi/lib/libtvm_ffi.so" ]; then
  expect_contains "$out" "FLASHINFER_EXTRA_LDFLAGS" "FlashInfer JIT link gets tvm_ffi (link failure guard)"
else
  expect_not_contains "$out" "FLASHINFER_EXTRA_LDFLAGS" "no tvm_ffi in fake venv: link flag not set"
fi
expect_not_contains "$out" "gguf" "no GGUF conversion"
expect_not_contains "$out" "awq" "no AWQ conversion"
expect_not_contains "$out" "gptq" "no GPTQ conversion"

out="$(run_green SERVED_MODEL_NAME=custom-name -- start --dry-run --prefix "$PFX" --model-dir "$MODEL_OK")"; code=$?
expect_exit 0 "$code" "dry-run with served-name override exits 0"
expect_contains "$out" "--served-model-name custom-name" "served name overridable"

out="$(run_green -- start --dry-run --prefix "$PFX" --model-dir "$MODEL_OK" --port 18080)"; code=$?
expect_exit 0 "$code" "dry-run with port override exits 0"
expect_contains "$out" "--host 127.0.0.1 --port 18080" "port overridable, host stays loopback"

out="$(run_green -- start --dry-run --prefix "$PFX" --model-dir "$MODEL_OK" --host 0.0.0.0)"; code=$?
expect_exit 2 "$code" "non-loopback host refused"
expect_contains "$out" "refusing non-loopback bind address" "non-loopback bind rejected"

out="$(run_green SERVE_HOST=0.0.0.0 -- start --dry-run --prefix "$PFX" --model-dir "$MODEL_OK")"; code=$?
expect_exit 2 "$code" "SERVE_HOST env non-loopback refused"
expect_contains "$out" "refusing non-loopback bind address" "env override also validated"

out="$(run_green -- start --dry-run --prefix "$PFX" --model-dir "$MODEL_OK" --port 0)"; code=$?
expect_exit 2 "$code" "port 0 refused"
out="$(run_green -- start --dry-run --prefix "$PFX" --model-dir "$MODEL_OK" --port 99999)"; code=$?
expect_exit 2 "$code" "port 99999 refused"
out="$(run_green -- start --dry-run --prefix "$PFX" --model-dir "$MODEL_OK" --port 08)"; code=$?
expect_exit 0 "$code" "leading-zero port accepted as decimal"
expect_contains "$out" "--port 08" "leading-zero port passed through to vllm"
out="$(run_green -- start --dry-run --prefix "$PFX" --model-dir "$MODEL_OK" --host ::1)"; code=$?
expect_exit 0 "$code" "IPv6 loopback host accepted"

# ---------------------------------------------------------------------------
# 4. Guardrails: no downloads, no conversion or re-quantization in the scripts
# ---------------------------------------------------------------------------
section "serve: no network behavior in scripts (full sourced chain)"
for f in "$SCRIPT" "$ROOT/scripts/lib/serve-lib.sh" "$ROOT/scripts/lib/preflight-lib.sh" "$ROOT/scripts/lib/wsl2-env-lib.sh"; do
  for bad in snapshot_download huggingface-cli "wget" "curl " "git clone"; do
    if grep -q "$bad" "$f"; then FAIL=$((FAIL + 1)); say "FAIL - $f contains forbidden '$bad'"
    else PASS=$((PASS + 1)); say "ok   - $f has no '$bad'"; fi
  done
done

# ---------------------------------------------------------------------------
# 5. Live offline launch: model list + short text inference via the fake API
#    server (needs a real python3 + curl; skipped otherwise).
# ---------------------------------------------------------------------------
section "serve: offline launch and OpenAI endpoints"
REALPY="$(command -v python3 2>/dev/null || true)"
if [ -z "$REALPY" ] || ! command -v curl >/dev/null 2>&1; then
  say "skip - live endpoint tests need a real python3 and curl (found python3='${REALPY:-<none>}')"
else
  cp "$FAKEBIN/vllm" "$PFX/venv/bin/vllm"
  chmod +x "$PFX/venv/bin/vllm"
  PORT=$((20000 + RANDOM % 10000))
  VLLM_LOG="$TMP_ROOT/fake-vllm.log"
  run_green FAKE_VLLM_LOG="$VLLM_LOG" FAKE_VLLM_REALPY="$REALPY" \
    -- start --prefix "$PFX" --model-dir "$MODEL_OK" --port "$PORT" \
    >"$TMP_ROOT/serve.out" 2>&1 &
  SERVE_PID=$!

  if wait_for_url "http://127.0.0.1:$PORT/v1/models"; then
    say "ok   - fake vLLM API server answered on loopback port $PORT"
  else
    FAIL=$((FAIL + 1)); say "FAIL - fake vLLM API server did not answer on 127.0.0.1:$PORT"
    sed 's/^/       | /' "$TMP_ROOT/serve.out"
  fi

  models="$(curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)"
  if [ -n "$models" ]; then
    expect_contains "$models" "model-ok" "model list returns the loaded model"
  else
    FAIL=$((FAIL + 1)); say "FAIL - /v1/models returned nothing"
  fi

  chat="$(curl -fsS --max-time 5 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
          -H 'Content-Type: application/json' \
          -d '{"model":"model-ok","messages":[{"role":"user","content":"hello offline"}]}' 2>/dev/null)"
  if [ -n "$chat" ]; then
    expect_contains "$chat" "offline reply to: hello offline" "short text request returns a completion"
    expect_contains "$chat" "model-ok" "completion names the served model"
  else
    FAIL=$((FAIL + 1)); say "FAIL - /v1/chat/completions returned nothing"
  fi

  kill "$SERVE_PID" 2>/dev/null
  wait "$SERVE_PID" 2>/dev/null
  SERVE_PID=""

  out="$(cat "$TMP_ROOT/serve.out")"
  expect_contains "$out" "preflight result: READY" "preflight gate passed before launch"
  expect_contains "$out" "Uvicorn running on http://127.0.0.1:$PORT" "fake vLLM reports loopback bind"
  expect_contains "$out" "http://127.0.0.1:$PORT/v1" "serve.sh prints the endpoint URL"

  log="$(cat "$VLLM_LOG" 2>/dev/null)"
  expect_contains "$log" "argv-line serve --model $MODEL_OK --quantization modelopt --kv-cache-dtype fp8 --host 127.0.0.1 --port $PORT --served-model-name model-ok --max-model-len 8192 --max-num-seqs 1 --trust-remote-code" "vllm process got the exact fixed argv"
  expect_contains "$log" "HF_HUB_OFFLINE=1" "offline mode exported to the process"
  expect_contains "$log" "TRANSFORMERS_OFFLINE=1" "transformers offline exported to the process"
  expect_contains "$log" "server-listening 127.0.0.1:$PORT model=model-ok" "server registered the served model"
fi

say ""
say "== results: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
