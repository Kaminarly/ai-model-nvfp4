#!/usr/bin/env bash
# wsl2-env.sh - build and verify the fixed WSL2 inference runtime (issue 01).
#
# Commands:
#   prereqs                 confirm Windows 10 + WSL2 Ubuntu + RTX 5090 prerequisites
#   create [--prefix DIR] [--force] [--dry-run]
#                           create the isolated venv with CUDA 13 toolkit + vLLM 0.27.1
#   verify  [--prefix DIR]  verify python / nvcc / vLLM versions of the created env
#   help
#
# Run inside WSL2 Ubuntu (from Windows PowerShell, example):
#   wsl -d Ubuntu -- bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/wsl2-env.sh create
#
# This command NEVER installs a Linux NVIDIA driver inside WSL and NEVER
# downloads, copies, converts or deletes model files (issues 02/03 handle models).
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/wsl2-env-lib.sh
. "$SCRIPT_DIR/lib/wsl2-env-lib.sh"

usage() {
  cat <<'EOF'
Usage:
  wsl2-env.sh prereqs
  wsl2-env.sh create [--prefix DIR] [--force] [--dry-run]
  wsl2-env.sh verify [--prefix DIR]
  wsl2-env.sh help

Commands:
  prereqs   Verify Windows 10 + WSL2 Ubuntu + RTX 5090 prerequisites.
  create    Build the isolated runtime: python venv + CUDA 13 toolkit + vLLM 0.27.1.
  verify    Verify installed versions (python, nvcc, vLLM) from the command line.
  help      Show this help.

Options:
  --prefix DIR   Runtime directory (default: $HOME/qwen3-nvfp4-rtx5090).
  --force        Rebuild the venv if it already exists (create only).
  --dry-run      Print the commands that would run, without executing (create only).

Model files are intentionally out of scope for this command: no downloads,
no conversion, no copies. See issues 02/03 for preflight and the offline service.
EOF
}

die_install() { # die_install <message> <fix-text>
  fail "$1"
  printf '%s\n' "$2" | sed 's/^/        fix:    /'
  exit 1
}

cmd_create() {
  local prefix="" venv="" force=0 dry=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --prefix) prefix="${2:-}"; [ -n "$prefix" ] || { fail "--prefix needs a value"; exit 2; }; shift 2 ;;
      --force) force=1; shift ;;
      --dry-run) dry=1; shift ;;
      --help|-h) usage; exit 0 ;;
      --model-dir|--model) fail "model handling is out of scope for this command (issue 01)."; exit 2 ;;
      *) fail "unknown option: $1"; usage >&2; exit 2 ;;
    esac
  done
  [ -n "$prefix" ] || prefix="$WSL2_ENV_PREFIX"
  DRY_RUN="$dry"
  venv="$prefix/venv"

  section "Create runtime 1/4: prerequisites"
  run_prereqs || { info "Fix the [FAIL] items above; the environment was NOT created."; exit 1; }

  section "Create runtime 2/4: isolated Python environment"
  if [ -e "$venv" ]; then
    if [ "$force" -eq 1 ]; then
      warn "removing existing venv: $venv (--force)"
      run_or_dry rm -rf "$venv" || die_install "could not remove $venv" "Check permissions on $prefix."
    else
      fail "venv already exists at $venv"
      info "fix: re-run with --force to rebuild, or use 'verify' to inspect the existing env."
      exit 1
    fi
  fi
  run_or_dry mkdir -p "$prefix"
  run_or_dry python3 -m venv "$venv" \
    || die_install "python3 -m venv failed" "Install python3-venv in Ubuntu (sudo apt-get install -y python3-venv) and re-run."

  section "Create runtime 3/4: CUDA ${CUDA_MAJOR_VERSION} toolkit + vLLM ${VLLM_VERSION}"
  run_or_dry "$venv/bin/python" -m pip install --upgrade pip \
    || die_install "pip upgrade failed" "Check network access from WSL2, then re-run."
  run_or_dry "$venv/bin/python" -m pip install --no-warn-script-location "$CUDA_TOOLKIT_PIP_SPEC" \
    || die_install "CUDA ${CUDA_MAJOR_VERSION} toolkit install failed" \
      "Check network access; adjust CUDA_TOOLKIT_PIP_SPEC (default 'cuda-toolkit[nvcc]') if needed (see README.md)."
  run_or_dry "$venv/bin/python" -m pip install "$VLLM_PIP_SPEC" \
    || die_install "vLLM ${VLLM_VERSION} install failed" \
      "Check network access; adjust VLLM_PIP_SPEC (default 'vllm==${VLLM_VERSION}') if needed (see README.md)."

  # Real-hardware hardening (issue 03, start 6/8): the 'cuda-toolkit' meta
  # wheel pulls a mixed cu13 tree; FlashInfer JIT fails its CCCL header check
  # when nvcc 13.3 meets runtime headers 13.0. Pin the four toolkit packages
  # to 13.3.x so the compiler and headers agree, then create the lib64
  # symlinks FlashInfer's JIT link step expects (it hardcodes
  # "-L$cuda_home/lib64 -lcudart -lcuda"; the pip tree ships libcudart in
  # lib/ and WSL2 provides libcuda.so under /usr/lib/wsl/lib).
  run_or_dry "$venv/bin/python" -m pip install \
    "nvidia-cuda-runtime==13.3.29" \
    "nvidia-cuda-nvrtc==13.3.33" \
    "nvidia-cuda-cupti==13.3.75" \
    "nvidia-nvtx==13.3.29" \
    || die_install "CUDA 13.3 component alignment failed" \
      "Check network access from WSL2, then re-run."
  local cu13_dir tvm_ffi_lib
  cu13_dir="$(cuda_home_in_venv "$venv" 2>/dev/null)"
  if [ -n "$cu13_dir" ] && [ -d "$cu13_dir" ]; then
    run_or_dry mkdir -p "$cu13_dir/lib64"
    if [ -f "$cu13_dir/lib/libcudart.so.13" ]; then
      run_or_dry ln -sfn "$cu13_dir/lib/libcudart.so.13" "$cu13_dir/lib64/libcudart.so"
    fi
    if [ -e /usr/lib/wsl/lib/libcuda.so ]; then
      run_or_dry ln -sfn /usr/lib/wsl/lib/libcuda.so "$cu13_dir/lib64/libcuda.so"
    fi
  else
    warn "could not locate venv CUDA tree; skipping lib64 symlinks (FlashInfer JIT link may fail at first kernel build)"
  fi

  section "Create runtime 4/4: environment record"
  local info_txt
  info_txt="# Qwen3-8-27B NVFP4 RTX5090 - WSL2 runtime (issue 01)
prefix:  $prefix
venv:    $venv
created: $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)
python:  $(python3 --version 2>&1)
cuda:    ${CUDA_TOOLKIT_PIP_SPEC} (major ${CUDA_MAJOR_VERSION})
vllm:    ${VLLM_PIP_SPEC}
models:  NOT handled by this command (issues 02/03 use a user-specified local directory)"
  if [ "$dry" -eq 1 ]; then
    printf '[DRY-RUN] write %s with:\n%s\n' "$prefix/env-info.txt" "$info_txt"
  else
    printf '%s\n' "$info_txt" > "$prefix/env-info.txt" \
      || die_install "could not write $prefix/env-info.txt" "Check permissions on $prefix."
  fi

  section "Done"
  ok "runtime created at $prefix"
  info "Next: bash scripts/wsl2-env.sh verify --prefix $prefix"
  info "Then: issue 02 (unified preflight) and issue 03 (offline short-context service)."
}

cmd_verify() {
  local prefix="" venv=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --prefix) prefix="${2:-}"; [ -n "$prefix" ] || { fail "--prefix needs a value"; exit 2; }; shift 2 ;;
      --help|-h) usage; exit 0 ;;
      --model-dir|--model) fail "model handling is out of scope for this command (issue 01)."; exit 2 ;;
      *) fail "unknown option: $1"; usage >&2; exit 2 ;;
    esac
  done
  [ -n "$prefix" ] || prefix="$WSL2_ENV_PREFIX"
  venv="$prefix/venv"
  PASSED=0
  FAILED=0
  WARNED=0

  section "Verify runtime (issue 01): $prefix"

  check_start venv "venv exists at $prefix"
  if [ ! -d "$venv" ]; then
    check_fail "venv not found" "no Python environment at $venv." \
      "Run: bash scripts/wsl2-env.sh create --prefix $prefix"
    checks_summary
    exit 1
  fi
  check_ok

  check_venv_python "$venv"
  check_nvcc_venv "$venv"
  check_vllm_venv "$venv"

  check_start torch "PyTorch present (informational)"
  local tv
  tv="$("$venv/bin/python" -c 'import torch; print(torch.__version__)' 2>/dev/null)"
  if [ -n "$tv" ]; then
    check_ok "torch $tv"
  else
    check_warn "torch not importable; vLLM normally installs it (re-check pip logs)."
  fi

  check_start gpu "NVIDIA GPU visible"
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    check_ok
  else
    check_fail "nvidia-smi failed" "GPU not visible from WSL2." \
      $'Update the driver on the WINDOWS host and restart WSL:\n    wsl --shutdown\nDo NOT install a Linux driver inside WSL2.'
  fi

  if checks_summary; then
    info "Next: issue 02 (unified preflight) and issue 03 (offline short-context service)."
    exit 0
  fi
  exit 1
}

main() {
  local cmd="${1:-}"
  if [ $# -gt 0 ]; then shift; fi
  case "$cmd" in
    prereqs) run_prereqs; exit $? ;;
    create) cmd_create "$@" ;;
    verify) cmd_verify "$@" ;;
    help|-h|--help|"") usage; exit 0 ;;
    *) fail "unknown command: $cmd"; usage >&2; exit 2 ;;
  esac
}
main "$@"
