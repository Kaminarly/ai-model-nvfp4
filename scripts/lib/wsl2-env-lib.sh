# wsl2-env-lib.sh - shared helpers for the fixed WSL2 inference runtime (issue 01).
#
# Sourced by scripts/wsl2-env.sh. ASCII-only output keeps results safe on
# Windows consoles and inside WSL. All versions follow
# .scratch/qwen3-8-27b-native-nvfp4-wsl2/spec.md and can be overridden
# through environment variables (see README.md).

set -u

# ---------------------------------------------------------------------------
# Configuration (spec-fixed defaults; override via environment)
# ---------------------------------------------------------------------------
VLLM_VERSION="${VLLM_VERSION:-0.27.1}"
CUDA_MAJOR_VERSION="${CUDA_MAJOR_VERSION:-13}"
# CUDA 13 toolkit: NVIDIA's official PyPI meta-package 'cuda-toolkit' (the old
# 'nvidia-cuda-toolkit-cu13' name does not exist on PyPI). The [nvcc] extra pulls
# in exactly the compiler chain (nvcc + crt + runtime + nvvm) used for vLLM's
# first-time Blackwell FP4 kernel compilation. vLLM 0.27.1 has no [cuda-13]
# extra: its CUDA 13 stack (torch 2.13, cutlass-dsl[cu13], ...) is bundled in
# the base wheel, so VLLM_PIP_SPEC is a plain pin.
CUDA_TOOLKIT_PIP_SPEC="${CUDA_TOOLKIT_PIP_SPEC:-cuda-toolkit[nvcc]==13.3.1}"
VLLM_PIP_SPEC="${VLLM_PIP_SPEC:-vllm==${VLLM_VERSION}}"
REQUIRED_GPU_NAME="${REQUIRED_GPU_NAME:-RTX 5090}"
REQUIRED_DRIVER_VERSION="${REQUIRED_DRIVER_VERSION:-610.88}"
MIN_PYTHON_VERSION="${MIN_PYTHON_VERSION:-3.10}"
WSL2_ENV_PREFIX="${WSL2_ENV_PREFIX:-$HOME/vllm}"

# Test seams (only needed by tests/run-tests.sh; ignored in normal use).
WIN_VER_CMD="${WIN_VER_CMD:-cmd.exe}"
OS_RELEASE_FILE="${OS_RELEASE_FILE:-/etc/os-release}"

# ---------------------------------------------------------------------------
# Output helpers
#
# All output functions write through EXTERNAL printf (env printf). Bash's own
# printf buffers stdout when it is redirected to a file or pipe, and the
# buffer is lost if the script is killed (or a test terminates it) before the
# buffer flushes - the full-context validation (issue 04) is such a case.
# External printf exits (and flushes) per call, so every line lands
# immediately. Line-level writes are atomic on regular files, so output stays
# in order.
# ---------------------------------------------------------------------------
info()   { env printf '[INFO]  %s\n' "$*"; }
ok()     { env printf '[OK]    %s\n' "$*"; }
warn()   { env printf '[WARN]  %s\n' "$*"; }
fail()   { env printf '[FAIL]  %s\n' "$*"; }
section(){ env printf '\n=== %s ===\n' "$*"; }

trim() { sed 's/^[[:space:]]*//; s/[[:space:]]*$//'; }

# ---------------------------------------------------------------------------
# Check helpers (shared counters; each check counts once)
# ---------------------------------------------------------------------------
CURRENT_CHECK=""
CURRENT_DESC=""
PASSED=0
FAILED=0
WARNED=0

check_start() {
  CURRENT_CHECK="$1"
  CURRENT_DESC="$2"
  env printf '[CHECK] %-16s %s\n' "$1" "$2"
}

check_ok() {
  env printf '[OK]    %-16s %s\n' "$CURRENT_CHECK" "${1:-$CURRENT_DESC}"
  PASSED=$((PASSED + 1))
}

check_warn() {
  env printf '[WARN]  %-16s %s\n' "$CURRENT_CHECK" "$1"
  WARNED=$((WARNED + 1))
}

check_fail() {
  env printf '[FAIL]  %-16s %s\n' "$CURRENT_CHECK" "$1"
  env printf '        reason: %s\n' "$2"
  env printf '%s\n' "$3" | sed 's/^/        fix:    /'
  FAILED=$((FAILED + 1))
}

checks_summary() {
  env printf '\n=== Result ===\n'
  env printf 'passed=%d failed=%d warnings=%d\n' "$PASSED" "$FAILED" "$WARNED"
  if [ "$FAILED" -gt 0 ]; then
    fail "overall result: NOT OK"
    return 1
  fi
  ok "overall result: OK"
  return 0
}

# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
# ver_ge A B: true if dotted-numeric A >= B (e.g. 610.88 >= 610.88, 3.12 >= 3.10).
ver_ge() {
  local a="$1" b="$2" pa pb i
  for i in 1 2 3 4 5; do
    [ "$a" = "${a#*.}" ] && a="${a}.0"
    [ "$b" = "${b#*.}" ] && b="${b}.0"
    pa="${a%%.*}"
    pb="${b%%.*}"
    case "$pa" in *[!0-9]*) return 1 ;; esac
    case "$pb" in *[!0-9]*) return 1 ;; esac
    if [ "$pa" -gt "$pb" ]; then return 0; fi
    if [ "$pa" -lt "$pb" ]; then return 1; fi
    a="${a#*.}"
    b="${b#*.}"
  done
  return 0
}

# run_or_dry: execute a command, or print it when DRY_RUN=1. Output via
# external printf so redirected stdout is not buffered away (same rationale
# as the output helpers above).
DRY_RUN=0
run_or_dry() {
  if [ "$DRY_RUN" -eq 1 ]; then
    env printf '[DRY-RUN]'
    env printf ' %q' "$@"
    env printf '\n'
    return 0
  fi
  env printf '$ %s\n' "$*"
  "$@"
}

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------
check_wsl2() {
  check_start wsl2 "WSL2 kernel detected"
  local kernel
  kernel="$(uname -r 2>/dev/null)"
  case "$kernel" in
    *microsoft-standard-WSL2*) : ;;
    *)
      check_fail "WSL2 kernel not detected" \
        "kernel='${kernel:-<unknown>}'; expected a 'microsoft-standard-WSL2' kernel. This script must run inside WSL2 Ubuntu, not on native Windows." \
        $'From PowerShell as admin:\n    wsl --install -d Ubuntu\n    wsl --set-version Ubuntu 2\nReboot, then run: wsl -d Ubuntu -- bash /mnt/<drive>/<path>/ai-model-nvfp4/scripts/wsl2-env.sh prereqs'
      return 1
      ;;
  esac
  if [ -z "${WSL_INTEROP:-}" ]; then
    check_fail "WSL interop not active" "WSL_INTEROP is empty; Windows interop is required." \
      $'Restart WSL and open a new WSL2 terminal:\n    wsl --shutdown'
    return 1
  fi
  check_ok "kernel '$kernel'"
}

check_ubuntu() {
  check_start ubuntu "WSL distro is Ubuntu"
  local osfile="$OS_RELEASE_FILE"
  if [ ! -r "$osfile" ]; then
    check_fail "cannot read $osfile" "os-release is not readable; cannot confirm the distro." \
      "Run this script inside an Ubuntu WSL2 distro: wsl --install -d Ubuntu"
    return 1
  fi
  local id id_like
  id="$(sed -n 's/^ID=//p' "$osfile" | head -1 | tr -d '"')"
  id_like="$(sed -n 's/^ID_LIKE=//p' "$osfile" | head -1 | tr -d '"')"
  case " $id $id_like " in
    *" ubuntu "*)
      check_ok "distro ID=$id"
      ;;
    *)
      check_fail "distro is not Ubuntu" "ID=$id ID_LIKE=$id_like; this spec is fixed to Ubuntu." \
        $'Install an Ubuntu WSL2 distro:\n    wsl --install -d Ubuntu\nThen set it as default: wsl --set-default Ubuntu'
      return 1
      ;;
  esac
}

check_windows10() {
  check_start windows "Windows 10 host (build >= 19041, < 22000)"
  local ver major rest build
  ver="$("$WIN_VER_CMD" /c ver 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
  if [ -z "$ver" ]; then
    check_fail "cannot read Windows version" "cmd.exe interop query returned nothing." \
      $'Restart WSL so Windows interop works:\n    wsl --shutdown\nThen open a new WSL2 terminal and re-run.'
    return 1
  fi
  major="${ver%%.*}"
  rest="${ver#*.}"
  build="${rest#*.}"
  case "$build" in
    ''|*[!0-9]*)
      check_fail "cannot parse Windows build from '$ver'" "unexpected 'ver' output." \
        "Confirm this runs inside WSL2 with interop, then re-run."
      return 1
      ;;
  esac
  if [ "$major" != "10" ]; then
    check_fail "unsupported Windows version ($ver)" "expected Windows 10 (10.0.19xxx)." \
      "This spec is fixed to Windows 10; see .scratch/qwen3-8-27b-native-nvfp4-wsl2/spec.md."
    return 1
  fi
  if [ "$build" -ge 22000 ]; then
    check_fail "Windows 11 detected (build $build)" "this spec is fixed to Windows 10 (22H2, build 19045)." \
      "Use the Windows 10 host specified by the spec, or update the spec before changing the target."
    return 1
  fi
  if [ "$build" -ge 19041 ]; then
    check_ok "Windows 10 build $build"
  else
    check_fail "Windows 10 build $build is too old for WSL2" "WSL2 requires build 19041+." \
      "Run Windows Update to reach at least 10.0.19041 (22H2 recommended)."
    return 1
  fi
}

check_nvidia_smi() {
  check_start gpu "NVIDIA GPU visible (nvidia-smi)"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    check_fail "nvidia-smi not found in WSL" "The Windows NVIDIA driver is not exposed to WSL2." \
      $'Install/update the NVIDIA driver on the WINDOWS host (NVIDIA App or Windows Update), then restart WSL:\n    wsl --shutdown\nDo NOT install a Linux NVIDIA driver inside WSL2: WSL2 uses the Windows driver.'
    return 1
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    check_fail "nvidia-smi reports no GPU" "No NVIDIA device is visible from WSL2." \
      $'Check the Windows driver, then restart WSL:\n    wsl --shutdown\nDo NOT install a Linux driver inside WSL2.'
    return 1
  fi
  check_ok
}

check_gpu_identity() {
  check_start gpu-model "GPU is ${REQUIRED_GPU_NAME} (32 GB)"
  local line gpu_name driver mem mem_mib
  line="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1)"
  if [ -z "$line" ]; then
    check_fail "no GPU query data" "nvidia-smi returned no rows." \
      "Re-run after 'wsl --shutdown'; if it persists, update the Windows driver."
    return 1
  fi
  IFS=',' read -r gpu_name driver mem <<<"$line"
  gpu_name="$(printf '%s' "$gpu_name" | trim)"
  driver="$(printf '%s' "$driver" | trim)"
  mem="$(printf '%s' "$mem" | trim)"
  mem_mib="${mem%% *}"
  case "$mem_mib" in ''|'N/A'|'n/a') mem_mib="0" ;; esac

  case "$gpu_name" in
    *"$REQUIRED_GPU_NAME"*) : ;;
    *)
      check_fail "GPU is '$gpu_name', expected '${REQUIRED_GPU_NAME}'" "this spec is fixed to the RTX 5090 32 GB." \
        "Use a machine with an RTX 5090, or update the spec before changing REQUIRED_GPU_NAME."
      return 1
      ;;
  esac

  if ver_ge "$driver" "$REQUIRED_DRIVER_VERSION"; then
    ok "driver $driver (>= $REQUIRED_DRIVER_VERSION)"
  else
    check_fail "driver $driver is older than required $REQUIRED_DRIVER_VERSION" "Blackwell NVFP4 needs a recent driver." \
      $'Update the NVIDIA driver on the WINDOWS host, then restart WSL:\n    wsl --shutdown\nDo NOT install a Linux driver inside WSL2.'
    return 1
  fi

  case "$mem_mib" in
    ''|*[!0-9]*)
      check_fail "cannot parse VRAM from '$mem'" "nvidia-smi memory field is not numeric." \
        "Check the Windows driver, then re-run after 'wsl --shutdown'."
      return 1
      ;;
  esac
  if [ "$mem_mib" -ge 30720 ]; then
    check_ok "RTX 5090, driver $driver, $((mem_mib / 1024)) GiB VRAM"
  elif [ "$mem_mib" -ge 24576 ]; then
    check_warn "only $((mem_mib / 1024)) GiB VRAM: short-context startup may work, but the full 262144-token config will not fit."
    check_ok "RTX 5090, driver $driver, $((mem_mib / 1024)) GiB VRAM (below 32 GB)"
  else
    check_fail "VRAM is ${mem_mib} MiB (~$((mem_mib / 1024)) GiB), expected ~32 GB" "an RTX 5090 reports 32768 MiB." \
      "Close programs using the GPU on Windows, then 'wsl --shutdown' and re-run."
    return 1
  fi
}

check_python3() {
  check_start python "python3 available (>= $MIN_PYTHON_VERSION)"
  if ! command -v python3 >/dev/null 2>&1; then
    check_fail "python3 not found" "no system python3 inside WSL2." \
      $'Install Python in Ubuntu:\n    sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip'
    return 1
  fi
  local ver
  ver="$(python3 --version 2>&1)"
  ver="${ver#Python }"
  if ver_ge "$ver" "$MIN_PYTHON_VERSION"; then
    check_ok "Python $ver"
  else
    check_fail "python3 $ver is older than $MIN_PYTHON_VERSION" "vLLM needs a modern Python." \
      $'Install a newer Python in Ubuntu:\n    sudo apt-get update && sudo apt-get install -y python3 python3-venv'
    return 1
  fi
}

check_ensurepip() {
  check_start venv "python3 can create venvs (venv + ensurepip)"
  if ! python3 -c 'import venv, ensurepip' >/dev/null 2>&1; then
    check_fail "python3-venv/ensurepip missing" "python3 cannot create isolated environments." \
      $'Install it in Ubuntu:\n    sudo apt-get update && sudo apt-get install -y python3-venv'
    return 1
  fi
  check_ok
}

run_prereq_checks() {
  PASSED=0
  FAILED=0
  WARNED=0
  section "Prerequisite checks"
  info "Target: Windows 10 + WSL2 Ubuntu + ${REQUIRED_GPU_NAME} 32 GB; CUDA ${CUDA_MAJOR_VERSION} toolkit; vLLM ${VLLM_VERSION}"
  check_wsl2
  check_ubuntu
  check_windows10
  if check_nvidia_smi; then
    check_gpu_identity
  fi
  check_python3
  check_ensurepip
}

run_prereqs() {
  run_prereq_checks
  checks_summary
}

# ---------------------------------------------------------------------------
# venv checks (shared by 'verify' and the issue 02 preflight)
# ---------------------------------------------------------------------------
# nvcc_in_venv <venv>: print the nvcc binary path if present. The
# 'cuda-toolkit' wheel installs nvcc under the site-packages tree
# (site-packages/nvidia/cu13/bin/nvcc), not $venv/bin; accept both layouts.
nvcc_in_venv() {
  local venv="$1" nvcc_site
  if [ -x "$venv/bin/nvcc" ]; then
    printf '%s\n' "$venv/bin/nvcc"
    return 0
  fi
  nvcc_site="$(ls -1 "$venv"/lib/python*/site-packages/nvidia/cu*/bin/nvcc 2>/dev/null | head -1)"
  if [ -n "$nvcc_site" ] && [ -x "$nvcc_site" ]; then
    printf '%s\n' "$nvcc_site"
    return 0
  fi
  return 1
}

# cuda_home_in_venv <venv>: print the CUDA root for the venv's CUDA toolkit
# (the nvcc binary's parent's parent, e.g. site-packages/nvidia/cu13).
# Empty when no nvcc is present. FlashInfer/deep-gemm JIT components look up
# CUDA_HOME in a subprocess that does not know about the venv, so the launcher
# exports this value.
cuda_home_in_venv() {
  local venv="$1" nvcc
  nvcc="$(nvcc_in_venv "$venv")" || return 1
  dirname "$(dirname "$nvcc")"
}

check_venv_python() { # check_venv_python <venv>
  local venv="$1"
  check_start python "venv python >= $MIN_PYTHON_VERSION"
  local pyver
  pyver="$("$venv/bin/python" --version 2>&1 | sed 's/^Python //')"
  if [ -n "$pyver" ] && ver_ge "$pyver" "$MIN_PYTHON_VERSION"; then
    check_ok "Python $pyver"
  else
    check_fail "python check failed (got '${pyver:-<empty>}')" "the venv python is missing or too old." \
      "Re-run: bash scripts/wsl2-env.sh create --force --prefix $(dirname "$venv")"
  fi
}

check_nvcc_venv() { # check_nvcc_venv <venv>
  local venv="$1" nvcc_bin rel major nvcc_lib
  check_start nvcc "CUDA compiler is release $CUDA_MAJOR_VERSION"
  nvcc_bin="$(nvcc_in_venv "$venv")"
  if [ -n "$nvcc_bin" ]; then
    # nvcc may need the sibling libs (crt/nvvm) from its own cuXX tree.
    nvcc_lib="$(dirname "$(dirname "$nvcc_bin")")/lib"
    rel="$(LD_LIBRARY_PATH="${nvcc_lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$nvcc_bin" --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | head -1)"
    major="${rel#release }"
    major="${major%%.*}"
    if [ "$major" = "$CUDA_MAJOR_VERSION" ]; then
      check_ok "nvcc $rel"
    else
      check_fail "nvcc reports '${rel:-<unknown>}', expected release $CUDA_MAJOR_VERSION" "CUDA compiler version mismatch." \
        "Re-run create, or install into the venv: $venv/bin/python -m pip install $CUDA_TOOLKIT_PIP_SPEC"
    fi
  else
    check_fail "nvcc not found in venv" "the CUDA ${CUDA_MAJOR_VERSION} toolkit is not installed (looked in $venv/bin and site-packages/nvidia)." \
      "Re-run: bash scripts/wsl2-env.sh create --force --prefix $(dirname "$venv")"
  fi
}

check_vllm_venv() { # check_vllm_venv <venv>
  local venv="$1" vv
  check_start vllm "vLLM version == $VLLM_VERSION"
  vv="$("$venv/bin/python" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
  if [ "$vv" = "$VLLM_VERSION" ]; then
    check_ok "vllm $vv"
  elif [ -n "$vv" ]; then
    check_fail "vLLM is $vv, expected $VLLM_VERSION" "version mismatch with the spec." \
      "Install into the venv: $venv/bin/python -m pip install vllm==$VLLM_VERSION"
  else
    check_fail "vllm import failed" "vLLM is not importable from the venv." \
      "Re-run: bash scripts/wsl2-env.sh create --force --prefix $(dirname "$venv")"
  fi
}
