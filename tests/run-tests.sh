#!/usr/bin/env bash
# Lightweight test runner for issue 01 (bash only; no external test framework).
# Uses fake executables in tests/fakebin so it runs without WSL2 or a GPU.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/scripts/wsl2-env.sh"
FAKEBIN="$ROOT/tests/fakebin"
FIXTURES="$ROOT/tests/fixtures"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

chmod +x "$FAKEBIN"/* "$FIXTURES"/venv/bin/* 2>/dev/null || true

PASS=0
FAIL=0
say() { printf '%s\n' "$*"; }

# run_env [VAR=val ...] -- <script args...>
run_env() {
  local vars=()
  while [ "$1" != "--" ]; do vars+=("$1"); shift; done
  shift
  env "${vars[@]}" PATH="$FAKEBIN:$PATH" bash "$SCRIPT" "$@" 2>&1
}

# run_green [extra VAR=val ...] -- <script args...>   (green base environment)
run_green() {
  run_env OS_RELEASE_FILE="$FIXTURES/os-release-ubuntu" \
          WSL_INTEROP=/run/WSL/1 \
          WIN_VER_CMD="$FAKEBIN/fake-win-cmd" \
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
# 1. prereqs
# ---------------------------------------------------------------------------
section "prereqs: green environment"
out="$(run_green -- prereqs)"; code=$?
expect_exit 0 "$code" "prereqs green exits 0"
expect_contains "$out" "WSL2 kernel" "reports WSL2"
expect_contains "$out" "Ubuntu" "reports Ubuntu"
expect_contains "$out" "Windows 10 build" "reports Windows 10"
expect_contains "$out" "RTX 5090" "reports RTX 5090"
expect_contains "$out" "overall result: OK" "summary OK"

section "prereqs: not WSL2"
out="$(run_env FAKE_KERNEL=5.15.0-1-generic OS_RELEASE_FILE="$FIXTURES/os-release-ubuntu" WIN_VER_CMD="$FAKEBIN/fake-win-cmd" -- prereqs)"; code=$?
expect_exit 1 "$code" "prereqs non-WSL2 exits 1"
expect_contains "$out" "wsl --install -d Ubuntu" "suggests wsl --install"

section "prereqs: not Ubuntu"
out="$(run_green OS_RELEASE_FILE="$FIXTURES/os-release-fedora" -- prereqs)"; code=$?
expect_exit 1 "$code" "prereqs non-Ubuntu exits 1"
expect_contains "$out" "distro is not Ubuntu" "clear reason for non-Ubuntu"

section "prereqs: Windows 11"
out="$(run_green FAKE_WIN_BUILD=26100 -- prereqs)"; code=$?
expect_exit 1 "$code" "prereqs Windows 11 exits 1"
expect_contains "$out" "Windows 11" "clear reason for Windows 11"

section "prereqs: no GPU"
out="$(run_green FAKE_GPU_MODE=missing -- prereqs)"; code=$?
expect_exit 1 "$code" "prereqs no GPU exits 1"
expect_contains "$out" "Do NOT install a Linux driver inside WSL2" "never suggests Linux driver in WSL"
expect_contains "$out" "wsl --shutdown" "suggests WSL restart"

section "prereqs: wrong GPU model"
out="$(run_green FAKE_GPU_LINE="NVIDIA GeForce RTX 4080, 610.88, 16384 MiB" -- prereqs)"; code=$?
expect_exit 1 "$code" "prereqs wrong GPU exits 1"
expect_contains "$out" "expected 'RTX 5090'" "clear reason for wrong GPU"

section "prereqs: old driver"
out="$(run_green FAKE_GPU_LINE="NVIDIA GeForce RTX 5090, 560.94, 32768 MiB" -- prereqs)"; code=$?
expect_exit 1 "$code" "prereqs old driver exits 1"
expect_contains "$out" "610.88" "mentions required driver"

# ---------------------------------------------------------------------------
# 2. create (dry-run + real)
# ---------------------------------------------------------------------------
section "create: dry-run green"
out="$(run_green -- create --dry-run --prefix "$TMP_ROOT/dry")"; code=$?
expect_exit 0 "$code" "create dry-run exits 0"
expect_contains "$out" "python3 -m venv" "plans venv creation"
expect_contains "$out" "cuda-toolkit[nvcc]==13.3.1" "plans CUDA 13 toolkit install"
expect_contains "$out" "vllm==0.27.1" "plans vLLM 0.27.1 install"
expect_not_contains "$out" "snapshot_download" "no HF download"
expect_not_contains "$out" "huggingface-cli" "no HF CLI"

section "create: dry-run blocked by failed prereqs"
out="$(run_green FAKE_GPU_MODE=missing -- create --dry-run --prefix "$TMP_ROOT/blocked")"; code=$?
expect_exit 1 "$code" "create dry-run red exits 1"
expect_contains "$out" "NOT created" "reports nothing was created"

section "create: full run on fake env"
P1="$TMP_ROOT/full"
out="$(run_green FAKE_NVCC_SRC="$FIXTURES/venv/bin/nvcc" FAKE_PIP_LOG="$TMP_ROOT/pip.log" -- create --prefix "$P1")"; code=$?
expect_exit 0 "$code" "create full exits 0"
expect_contains "$out" "runtime created at" "reports creation"
expect_contains "$(cat "$P1/env-info.txt" 2>/dev/null)" "vllm==0.27.1" "env record written"
expect_contains "$(cat "$TMP_ROOT/pip.log" 2>/dev/null)" "cuda-toolkit[nvcc]==13.3.1" "toolkit install recorded"
expect_contains "$(cat "$TMP_ROOT/pip.log" 2>/dev/null)" "vllm==0.27.1" "vllm install recorded"

section "create: refuses existing venv without --force"
out="$(run_green -- create --prefix "$P1")"; code=$?
expect_exit 1 "$code" "create existing exits 1"
expect_contains "$out" "--force" "suggests --force"

# ---------------------------------------------------------------------------
# 3. verify
# ---------------------------------------------------------------------------
section "verify: green env"
out="$(run_green FAKE_NVCC_SRC="$FIXTURES/venv/bin/nvcc" -- verify --prefix "$P1")"; code=$?
expect_exit 0 "$code" "verify green exits 0"
expect_contains "$out" "vllm 0.27.1" "reports vLLM version"
expect_contains "$out" "release 13" "reports CUDA 13 nvcc"
expect_contains "$out" "overall result: OK" "verify summary OK"

section "verify: wrong vLLM version"
out="$(run_green FAKE_NVCC_SRC="$FIXTURES/venv/bin/nvcc" FAKE_VLLM_VERSION=0.28.0 -- verify --prefix "$P1")"; code=$?
expect_exit 1 "$code" "verify bad vllm exits 1"
expect_contains "$out" "pip install vllm==0.27.1" "suggests exact pip fix"

section "verify: wrong CUDA release"
out="$(run_green FAKE_NVCC_SRC="$FIXTURES/venv/bin/nvcc" FAKE_CUDA_RELEASE=12.8 -- verify --prefix "$P1")"; code=$?
expect_exit 1 "$code" "verify bad nvcc exits 1"
expect_contains "$out" "pip install cuda-toolkit" "suggests toolkit fix"

section "create+verify: nvcc under site-packages (real NVIDIA cu13 layout)"
P2="$TMP_ROOT/sitepkg"
out="$(run_green FAKE_NVCC_SRC="$FIXTURES/venv/bin/nvcc" FAKE_NVCC_SITE_PACKAGES=1 FAKE_PIP_LOG="$TMP_ROOT/pip2.log" -- create --prefix "$P2")"; code=$?
expect_exit 0 "$code" "create with site-packages nvcc exits 0"
out="$(run_green FAKE_NVCC_SRC="$FIXTURES/venv/bin/nvcc" FAKE_NVCC_SITE_PACKAGES=1 -- verify --prefix "$P2")"; code=$?
expect_exit 0 "$code" "verify finds site-packages nvcc exits 0"
expect_contains "$out" "release 13" "finds nvcc under site-packages/nvidia"
expect_not_contains "$out" "nvcc not found" "no false nvcc failure"

section "verify: missing env"
out="$(run_green -- verify --prefix "$TMP_ROOT/none")"; code=$?
expect_exit 1 "$code" "verify missing env exits 1"
expect_contains "$out" "create --prefix" "suggests create"

# ---------------------------------------------------------------------------
# 4. CLI guardrails
# ---------------------------------------------------------------------------
section "cli: unknown command and model flags"
out="$(run_green -- bogus)"; code=$?
expect_exit 2 "$code" "unknown command exits 2"
out="$(run_green -- create --model-dir /tmp/models)"; code=$?
expect_exit 2 "$code" "model-dir rejected exits 2"
expect_contains "$out" "out of scope" "model flag rejected with reason"

say ""
say "== results: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
