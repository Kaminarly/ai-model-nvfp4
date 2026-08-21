#!/usr/bin/env bash
# Test runner for issue 02 (unified preflight). Bash only; uses the same fake
# toolchain as tests/run-tests.sh, plus fake model directories built on the
# fly. Runs without WSL2 or a GPU.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/scripts/preflight.sh"
FAKEBIN="$ROOT/tests/fakebin"
FIXTURES="$ROOT/tests/fixtures"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

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

# ---------------------------------------------------------------------------
# Fixtures: a fake runtime venv (via issue 01 create) and fake model dirs.
# ---------------------------------------------------------------------------
section "setup: fake venv + model fixtures"
PFX="$TMP_ROOT/pfx"
# Build the fake runtime with the issue 01 script (preflight only checks it).
out="$(env OS_RELEASE_FILE="$FIXTURES/os-release-ubuntu" WSL_INTEROP=/run/WSL/1 \
      WIN_VER_CMD="$FAKEBIN/fake-win-cmd" \
      FAKE_NVCC_SRC="$FIXTURES/venv/bin/nvcc" FAKE_NVCC_SITE_PACKAGES=1 \
      FAKE_PIP_LOG="$TMP_ROOT/pip.log" PATH="$FAKEBIN:$PATH" \
      bash "$ROOT/scripts/wsl2-env.sh" create --prefix "$PFX")"; code=$?
expect_exit 0 "$code" "create fake venv for preflight"

INDEX='{"metadata":{"total_size":0},"weight_map":{"layers.0.weight":"model-00001-of-00003.safetensors","layers.1.weight":"model-00002-of-00003.safetensors","layers.2.weight":"model-00003-of-00003.safetensors"}}'
# make_safetensors <file> <data_bytes>: write a minimal but valid safetensors
# file (8-byte LE header length + header JSON + zero data) so the shard-header
# integrity check has a real structure to verify.
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
# Heretic-Ara-style split: 2 shards + a separate MTP weight file. The preflight
# must accept any split the index declares, not a hardcoded 3-shard layout.
MODEL_MTP_SPLIT="$TMP_ROOT/model-mtp-split"
mkdir -p "$MODEL_MTP_SPLIT"
for f in model-00001-of-00002.safetensors model-00002-of-00002.safetensors model-mtp-fc.safetensors; do
  make_safetensors "$MODEL_MTP_SPLIT/$f" 1024
done
printf '%s\n' '{"metadata":{"total_size":0},"weight_map":{"layers.0.weight":"model-00001-of-00002.safetensors","layers.1.weight":"model-00002-of-00002.safetensors","mtp.0.weight":"model-mtp-fc.safetensors"}}' > "$MODEL_MTP_SPLIT/model.safetensors.index.json"
for f in config.json hf_quant_config.json generation_config.json tokenizer_config.json tokenizer.json chat_template.jinja; do
  cp "$MODEL_OK/$f" "$MODEL_MTP_SPLIT/$f"
done
MODEL_NO_CONFIG="$TMP_ROOT/model-no-config"; make_model "$MODEL_NO_CONFIG"; rm "$MODEL_NO_CONFIG/config.json"
MODEL_EMPTY_SHARD="$TMP_ROOT/model-empty-shard"; make_model "$MODEL_EMPTY_SHARD"; : > "$MODEL_EMPTY_SHARD/model-00003-of-00003.safetensors"
MODEL_BAD_INDEX="$TMP_ROOT/model-bad-index"; make_model "$MODEL_BAD_INDEX"
printf '%s\n' '{"metadata":{},"weight_map":{"x.weight":"model-00009-of-00003.safetensors"}}' > "$MODEL_BAD_INDEX/model.safetensors.index.json"
# Truncated shard: real-world failure mode (a partial copy passes existence
# and non-empty checks but fails safetensors header coverage).
MODEL_TRUNCATED="$TMP_ROOT/model-truncated"; make_model "$MODEL_TRUNCATED"
sz="$(stat -c %s "$MODEL_TRUNCATED/model-00002-of-00003.safetensors")"
head -c "$((sz / 2))" "$MODEL_TRUNCATED/model-00002-of-00003.safetensors" > "$MODEL_TRUNCATED/model-00002-of-00003.safetensors.tmp"
mv "$MODEL_TRUNCATED/model-00002-of-00003.safetensors.tmp" "$MODEL_TRUNCATED/model-00002-of-00003.safetensors"

# ---------------------------------------------------------------------------
# 1. green preflight
# ---------------------------------------------------------------------------
section "preflight: green environment + complete model"
out="$(run_green -- --prefix "$PFX" --model-dir "$MODEL_OK")"; code=$?
expect_exit 0 "$code" "green preflight exits 0"
expect_contains "$out" "overall result: OK" "summary OK"
expect_contains "$out" "preflight result: READY" "single READY boundary"
expect_contains "$out" "vllm 0.27.1" "vLLM version checked"
expect_contains "$out" "release 13" "nvcc release 13 checked"
expect_contains "$out" "required files present" "model file list complete"
expect_contains "$out" "index lists 3 shard(s), all present" "shard index consistent"
expect_contains "$out" "offline" "offline mode enforced"
expect_contains "$out" "no other compute processes" "no VRAM squatters"
expect_contains "$out" "VRAM free" "free VRAM reported"
expect_contains "$out" "C compiler:" "C compiler present"
expect_contains "$out" "CUDA_HOME=" "venv CUDA tree resolvable"
expect_contains "$out" "nvcc 13.0 matches toolkit headers" "nvcc release matches runtime headers"
expect_contains "$out" "ninja:" "ninja build tool present"
expect_contains "$out" "3 shard headers consistent" "shard structure intact"

out="$(run_green -- --prefix "$PFX" --model-dir "$MODEL_MTP_SPLIT")"; code=$?
expect_exit 0 "$code" "green preflight (2-shard + mtp split) exits 0"
expect_contains "$out" "preflight result: READY" "MTP-split model passes preflight"
expect_contains "$out" "index lists 3 shard(s), all present" "index cross-check covers the mtp file"
expect_contains "$out" "3 shard headers consistent" "mtp file header verified too"

# ---------------------------------------------------------------------------
# 2. model integrity failures
# ---------------------------------------------------------------------------
section "preflight: model integrity failures"
out="$(run_green -- --prefix "$PFX" --model-dir "$MODEL_NO_SHARD")"; code=$?
expect_exit 1 "$code" "missing shard exits 1"
expect_contains "$out" "index references missing shard: model-00002-of-00003.safetensors" "names missing shard (caught by the index cross-check)"
expect_contains "$out" "preflight result: NOT READY" "single NOT READY boundary"

out="$(run_green -- --prefix "$PFX" --model-dir "$MODEL_NO_CONFIG")"; code=$?
expect_exit 1 "$code" "missing config exits 1"
expect_contains "$out" "missing required file: config.json" "names missing config"

out="$(run_green -- --prefix "$PFX" --model-dir "$MODEL_EMPTY_SHARD")"; code=$?
expect_exit 1 "$code" "empty shard exits 1"
expect_contains "$out" "file is empty" "detects zero-byte shard"

out="$(run_green -- --prefix "$PFX" --model-dir "$MODEL_BAD_INDEX")"; code=$?
expect_exit 1 "$code" "bad index exits 1"
expect_contains "$out" "index references missing shard" "index cross-check fails"

out="$(run_green -- --prefix "$PFX" --model-dir "$TMP_ROOT/no-such-dir")"; code=$?
expect_exit 1 "$code" "missing dir exits 1"
expect_contains "$out" "model directory not found" "invalid model path detected"

# ---------------------------------------------------------------------------
# 2b. shard-header integrity + C compiler (issue 02 hardening)
# ---------------------------------------------------------------------------
section "preflight: truncated shard + C compiler"
out="$(run_green -- --prefix "$PFX" --model-dir "$MODEL_TRUNCATED")"; code=$?
expect_exit 1 "$code" "truncated shard exits 1"
expect_contains "$out" "shard data truncated: model-00002-of-00003.safetensors" "names the truncated shard"
expect_contains "$out" "incomplete metadata, file not fully covered" "matches the real safetensors error"
expect_contains "$out" "preflight result: NOT READY" "NOT READY boundary"

out="$(run_green CC_BIN=/nonexistent/gcc -- --prefix "$PFX" --model-dir "$MODEL_OK")"; code=$?
expect_exit 1 "$code" "missing C compiler exits 1"
expect_contains "$out" "no C compiler found" "clear reason for missing cc/gcc"
expect_contains "$out" "build-essential" "suggests the fix"

out="$(run_green CC_BIN="$FAKEBIN/cc" CC_FAIL_COMPILE=1 -- --prefix "$PFX" --model-dir "$MODEL_OK")"; code=$?
expect_exit 1 "$code" "missing Python.h (compile probe fails) exits 1"
expect_contains "$out" "Triton build probe failed" "clear reason for failed compile probe"
expect_contains "$out" "python3-dev" "suggests the dev-headers fix"

out="$(run_green CC_BIN=/bin/true -- --prefix "$PFX" --model-dir "$MODEL_OK")"; code=$?
expect_exit 0 "$code" "CC_BIN override passes"

out="$(run_green NINJA_BIN=/nonexistent/ninja -- --prefix "$PFX" --model-dir "$MODEL_OK")"; code=$?
expect_exit 1 "$code" "missing ninja exits 1"
expect_contains "$out" "ninja not found" "clear reason for missing ninja"
expect_contains "$out" "ninja-build" "suggests the fix"

# ---------------------------------------------------------------------------
# 2c. CUDA compiler/header compatibility (issue 03, real hardware: CCCL
# cuda_toolkit.h #error when nvcc 13.3 met runtime headers 13.0 in the venv)
# ---------------------------------------------------------------------------
section "preflight: CUDA toolkit version mismatch"
out="$(run_green FAKE_CUDA_RELEASE=13.3 FAKE_CUDA_RUNTIME_VERSION=13000 -- --prefix "$PFX" --model-dir "$MODEL_OK")"; code=$?
expect_exit 1 "$code" "nvcc/header mismatch exits 1"
expect_contains "$out" "nvcc 13.3 but runtime headers are 13.0" "names the mismatch"
expect_contains "$out" "cuda_toolkit.h" "matches the real CCCL error"
expect_contains "$out" "nvidia-cuda-runtime" "suggests the aligned upgrade"
expect_contains "$out" "preflight result: NOT READY" "NOT READY boundary"

# ---------------------------------------------------------------------------
# 3. offline mode + VRAM
# ---------------------------------------------------------------------------
section "preflight: offline enforcement and VRAM"
out="$(run_green HF_HUB_OFFLINE=0 -- --prefix "$PFX" --model-dir "$MODEL_OK")"; code=$?
expect_exit 1 "$code" "offline disabled exits 1"
expect_contains "$out" "offline mode disabled" "flags disabled offline mode"

out="$(run_green FAKE_GPU_APPS="1234, python3, 5120 MiB" -- --prefix "$PFX" --model-dir "$MODEL_OK")"; code=$?
expect_exit 0 "$code" "VRAM occupied is a warning, exits 0"
expect_contains "$out" "other programs hold 5120 MiB" "reports VRAM squatters"
expect_contains "$out" "fix:" "offers release suggestion"

out="$(run_green FAKE_GPU_MEMORY_LINE="32607, 12288" -- --prefix "$PFX" --model-dir "$MODEL_OK")"; code=$?
expect_exit 1 "$code" "low free VRAM exits 1"
expect_contains "$out" "only 20319 MiB VRAM free" "computes free VRAM correctly"
expect_contains "$out" "Close GPU-using programs" "suggests freeing VRAM"

# ---------------------------------------------------------------------------
# 4. guardrails: no downloads, CLI
# ---------------------------------------------------------------------------
section "preflight: no network behavior and CLI"
for f in "$SCRIPT" "$ROOT/scripts/lib/preflight-lib.sh"; do
  for bad in snapshot_download huggingface-cli "wget" "curl " "git clone"; do
    if grep -q "$bad" "$f"; then FAIL=$((FAIL + 1)); say "FAIL - $f contains forbidden '$bad'"
    else PASS=$((PASS + 1)); say "ok   - $f has no '$bad'"; fi
  done
done

out="$(run_green -- --prefix "$PFX")"; code=$?
expect_exit 2 "$code" "missing --model-dir exits 2"
expect_contains "$out" "missing required option: --model-dir" "usage error names the option"

out="$(run_green -- --bogus --model-dir "$MODEL_OK")"; code=$?
expect_exit 2 "$code" "unknown option exits 2"

say ""
say "== results: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
