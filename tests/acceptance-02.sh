#!/usr/bin/env bash
# Real-hardware acceptance for issue 02. Run INSIDE WSL2 Ubuntu:
#   bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/tests/acceptance-02.sh
#
# Exercises every failure path of scripts/preflight.sh against the real
# model directory by temporarily moving files away and restoring them.
# Network-independent: the model is never downloaded, only renamed locally.
set -u

PREFLIGHT="/mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/preflight.sh"
FAKEBIN="/mnt/d/Code/MJ-Project/ai-model-nvfp4/tests/fakebin"
MODEL="/home/kami/models/Qwen3.8-27B-NVFP4-RTX5090"
TMP="/tmp/accept02"

PASS=0
FAIL=0
say() { printf '%s\n' "$*"; }

# expect <desc> <want_rc> <got_rc> <needle>
expect_rc() {
  if [ "$3" -eq "$2" ]; then
    say "ok   - $1 (rc=$2)"
  else
    say "FAIL - $1: expected rc=$2, got rc=$3"
    FAIL=$((FAIL + 1))
  fi
}

expect_out() { # expect_out <desc> <output> <needle>
  case "$2" in
    *"$3"*) PASS=$((PASS + 1)); say "ok   - $1" ;;
    *) FAIL=$((FAIL + 1)); say "FAIL - $1: output misses '$3'"; printf '%s\n' "$2" | sed 's/^/       | /' | tail -3 ;;
  esac
}

run_pf() { # run_pf [VAR=val ...] -- [extra preflight args...]
  local vars=()
  while [ "$1" != "--" ]; do vars+=("$1"); shift; done
  shift
  env "${vars[@]}" bash "$PREFLIGHT" "$@"
}

mkdir -p "$TMP"

section() { say ""; say "== $1 =="; }

# ---------------------------------------------------------------------------
# 1. missing weight shard
# ---------------------------------------------------------------------------
section "1. missing weight shard -> NOT READY (rc=1)"
mv "$MODEL/model-00002-of-00003.safetensors" "$TMP/shard2.bak"
out="$(bash "$PREFLIGHT" --model-dir "$MODEL" 2>&1)"; rc=$?
mv "$TMP/shard2.bak" "$MODEL/model-00002-of-00003.safetensors"
expect_rc "missing shard exits 1" 1 "$rc"
expect_out "names missing shard" "$out" "missing required file: model-00002-of-00003.safetensors"
expect_out "NOT READY boundary" "$out" "preflight result: NOT READY"
expect_out "fix given" "$out" "fix:"

# ---------------------------------------------------------------------------
# 2. missing config
# ---------------------------------------------------------------------------
section "2. missing config.json -> NOT READY (rc=1)"
mv "$MODEL/config.json" "$TMP/config.bak"
out="$(bash "$PREFLIGHT" --model-dir "$MODEL" 2>&1)"; rc=$?
mv "$TMP/config.bak" "$MODEL/config.json"
expect_rc "missing config exits 1" 1 "$rc"
expect_out "names missing config" "$out" "missing required file: config.json"

# ---------------------------------------------------------------------------
# 3. invalid model path
# ---------------------------------------------------------------------------
section "3. invalid model path -> NOT READY (rc=1)"
out="$(bash "$PREFLIGHT" --model-dir "$MODEL/DOES-NOT-EXIST" 2>&1)"; rc=$?
expect_rc "invalid path exits 1" 1 "$rc"
expect_out "detects invalid path" "$out" "model directory not found"

# ---------------------------------------------------------------------------
# 4. missing --model-dir -> usage error (rc=2)
# ---------------------------------------------------------------------------
section "4. missing --model-dir -> rc=2"
out="$(bash "$PREFLIGHT" 2>&1)"; rc=$?
expect_rc "missing model-dir exits 2" 2 "$rc"
expect_out "usage names option" "$out" "missing required option: --model-dir"

# ---------------------------------------------------------------------------
# 5. offline mode disabled -> NOT READY (rc=1)
# ---------------------------------------------------------------------------
section "5. offline mode disabled -> NOT READY (rc=1)"
out="$(HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 bash "$PREFLIGHT" --model-dir "$MODEL" 2>&1)"; rc=$?
expect_rc "offline disabled exits 1" 1 "$rc"
expect_out "flags offline disabled" "$out" "offline mode disabled"

# ---------------------------------------------------------------------------
# 6. VRAM occupied by others -> WARN, still READY (rc=0)
# (real script, fake nvidia-smi feed; no real GPU process is running)
# ---------------------------------------------------------------------------
section "6. VRAM occupied by others -> WARN not block (rc=0)"
out="$(PATH="$FAKEBIN:$PATH" FAKE_GPU_APPS="1234, python3, 5120 MiB" bash "$PREFLIGHT" --model-dir "$MODEL" 2>&1)"; rc=$?
expect_rc "vram occupied exits 0" 0 "$rc"
expect_out "reports squatters" "$out" "other programs hold 5120 MiB"
expect_out "offers release fix" "$out" "fix:"

# ---------------------------------------------------------------------------
# 7. free VRAM below gate -> NOT READY (rc=1)
# ---------------------------------------------------------------------------
section "7. free VRAM below gate -> NOT READY (rc=1)"
out="$(PATH="$FAKEBIN:$PATH" FAKE_GPU_MEMORY_LINE="32607, 12288" bash "$PREFLIGHT" --model-dir "$MODEL" 2>&1)"; rc=$?
expect_rc "low vram exits 1" 1 "$rc"
expect_out "computes free VRAM" "$out" "only 20319 MiB VRAM free"
expect_out "suggests freeing" "$out" "Close GPU-using programs"

# ---------------------------------------------------------------------------
# 8. GPU not visible (fake nvidia-smi missing) -> NOT READY (rc=1)
# ---------------------------------------------------------------------------
section "8. GPU not visible -> NOT READY (rc=1)"
out="$(PATH="$FAKEBIN:$PATH" FAKE_GPU_MODE=missing bash "$PREFLIGHT" --model-dir "$MODEL" 2>&1)"; rc=$?
expect_rc "gpu missing exits 1" 1 "$rc"
expect_out "detects broken nvidia-smi" "$out" "nvidia-smi reports no GPU"
expect_out "never suggests linux driver" "$out" "Do NOT install a Linux driver inside WSL2"

# ---------------------------------------------------------------------------
# 9. GPU reports no device -> NOT READY (rc=1)
# ---------------------------------------------------------------------------
section "9. GPU no device -> NOT READY (rc=1)"
out="$(PATH="$FAKEBIN:$PATH" FAKE_GPU_MODE=no-gpu bash "$PREFLIGHT" --model-dir "$MODEL" 2>&1)"; rc=$?
expect_rc "gpu no-device exits 1" 1 "$rc"
expect_out "detects no device" "$out" "nvidia-smi reports no GPU"

# ---------------------------------------------------------------------------
# 10. restore check: full model -> READY (rc=0)
# ---------------------------------------------------------------------------
section "10. restored model -> READY (rc=0)"
out="$(bash "$PREFLIGHT" --model-dir "$MODEL" 2>&1)"; rc=$?
expect_rc "green exits 0" 0 "$rc"
expect_out "READY boundary" "$out" "preflight result: READY"

say ""
say "== acceptance-02: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
