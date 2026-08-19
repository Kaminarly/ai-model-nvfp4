# preflight-lib.sh - unified run-before-launch checks (issue 02).
#
# Sourced by scripts/preflight.sh. Reuses the issue 01 helpers from
# wsl2-env-lib.sh and adds model-integrity, offline-mode and VRAM checks.
# ASCII-only output keeps results safe on Windows consoles and inside WSL.

# shellcheck source=wsl2-env-lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/wsl2-env-lib.sh"

# ---------------------------------------------------------------------------
# Model file contract (issue 02)
#
# File list from the model card gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090:
# exactly 3 ModelOpt NVFP4 safetensors shards, the shard index, and the
# configs vLLM needs to boot the checkpoint offline (quantization=modelopt,
# kv-cache-dtype=fp8, trust-remote-code).
# ---------------------------------------------------------------------------
REQUIRED_MODEL_FILES="
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
model.safetensors.index.json
config.json
hf_quant_config.json
tokenizer.json
tokenizer_config.json
generation_config.json
chat_template.jinja
"

# VRAM thresholds (MiB; overridable via environment).
MIN_FREE_VRAM_MIB="${MIN_FREE_VRAM_MIB:-20480}"   # ~20 GiB free needed to boot short context
VRAM_OCCUPIED_WARN_MIB="${VRAM_OCCUPIED_WARN_MIB:-4096}"  # used-by-others above this -> warn

check_model_path() { # check_model_path <model-dir>
  local dir="$1"
  check_start model-path "model directory valid: $dir"
  if [ -z "$dir" ]; then
    check_fail "no --model-dir given" "model integrity cannot be checked without a directory." \
      "Pass --model-dir <absolute path to the model folder>, e.g. --model-dir /home/kami/models/Qwen3.8-27B-NVFP4-RTX5090"
    return 1
  fi
  if [ ! -d "$dir" ]; then
    check_fail "model directory not found" "'$dir' is not a directory." \
      "Put the model there (user-managed; this project never downloads it), then re-run with the correct --model-dir."
    return 1
  fi
  if [ ! -r "$dir" ] || [ ! -x "$dir" ]; then
    check_fail "model directory not readable" "'$dir' exists but is not accessible." \
      "Fix permissions: chmod +rx \"$dir\" (and every parent directory), then re-run."
    return 1
  fi
  check_ok
}

check_model_files() { # check_model_files <model-dir>
  local dir="$1" f n=0
  check_start model-files "model files complete (3 shards + index + configs)"
  for f in $REQUIRED_MODEL_FILES; do
    n=$((n + 1))
    if [ ! -f "$dir/$f" ]; then
      check_fail "missing required file: $f" "the model card publishes $f; vLLM needs it to boot." \
        "Copy the complete model folder into place (never downloaded by this project); see the model card file list, e.g. rsync -a host:Qwen3.8-27B-NVFP4-RTX5090/ \"$dir/\""
      return 1
    fi
    if [ ! -s "$dir/$f" ]; then
      check_fail "file is empty: $f" "a zero-byte file means the copy is broken." \
        "Re-copy '$f' (e.g. rsync --partial or re-download just that file), then re-run."
      return 1
    fi
  done
  check_ok "$n required files present"
}

check_model_index() { # check_model_index <model-dir>
  local dir="$1" idx refs r n=0
  check_start model-index "shard index consistent with files"
  idx="$dir/model.safetensors.index.json"
  refs="$(grep -oE '"model-0000[0-9]-of-00003\.safetensors"' "$idx" 2>/dev/null | tr -d '"' | sort -u)"
  if [ -z "$refs" ]; then
    check_fail "index lists no shards" "'$idx' contains no model-0000N-of-00003.safetensors references." \
      "The index is wrong or the wrong file was copied; re-copy it from the model card repo."
    return 1
  fi
  while read -r r; do
    [ -z "$r" ] && continue
    n=$((n + 1))
    if [ ! -f "$dir/$r" ]; then
      check_fail "index references missing shard: $r" "the index maps weights to a file that is not in the folder." \
        "Copy the complete folder from the model card; never mix shards from different downloads."
      return 1
    fi
  done <<<"$refs"
  check_ok "index lists $n shard(s), all present"
}

check_offline_mode() {
  check_start offline "Hugging Face / Transformers offline mode"
  local bad=""
  [ "${HF_HUB_OFFLINE:-1}" = "1" ] || bad="$bad HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
  [ "${TRANSFORMERS_OFFLINE:-1}" = "1" ] || bad="$bad TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
  if [ -n "$bad" ]; then
    check_fail "offline mode disabled:$bad" "vLLM must never reach model hosts." \
      "Unset those variables or set them to 1, then re-run: export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
    return 1
  fi
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  check_ok "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; this preflight makes no network calls"
}

# ---------------------------------------------------------------------------
# Shard structure integrity (issue 02 hardening, found on real hardware)
#
# vLLM fails at load time with "safetensors: incomplete metadata, file not
# fully covered" when a shard was truncated mid-copy (a partial shard passes
# the existence/non-empty checks). Safetensors files start with an 8-byte
# little-endian header length, then the header JSON whose data_offsets bound
# the tensor data. Checking those numbers against the real file size catches
# a truncated shard in milliseconds without hashing 10 GB.
# ---------------------------------------------------------------------------
# shard_header_len <file>: print the header length from the leading 8
# little-endian bytes, or empty when unreadable / implausibly large.
shard_header_len() {
  local file="$1" line b1 b2 b3 b4 b5 b6 b7 b8
  line="$(od -An -tu1 -N8 "$file" 2>/dev/null | tr -s ' ' | sed 's/^ //')"
  [ -n "$line" ] || return 1
  # shellcheck disable=SC2086
  set -- $line
  [ $# -ge 8 ] || return 1
  b1=$1 b2=$2 b3=$3 b4=$4 b5=$5 b6=$6 b7=$7 b8=$8
  # A real header is at most a few MB; require the upper 4 bytes to be zero.
  if [ "$b5" -ne 0 ] || [ "$b6" -ne 0 ] || [ "$b7" -ne 0 ] || [ "$b8" -ne 0 ]; then
    return 1
  fi
  printf '%s\n' "$((b1 + b2 * 256 + b3 * 65536 + b4 * 16777216))"
}

# check_shard_integrity <model-dir> <shard-file>
check_shard_integrity() {
  local dir="$1" f="$2" path="$1/$2" hlen size header max_end
  # Missing/empty shards already fail in check_model_files; only audit here.
  [ -f "$path" ] && [ -s "$path" ] || return 0
  hlen="$(shard_header_len "$path")"
  size="$(stat -c %s "$path" 2>/dev/null || printf '0\n')"
  if [ -z "$hlen" ]; then
    check_fail "shard header unreadable: $f" "the safetensors header length cannot be read from the leading 8 bytes." \
      "Re-copy '$f' from the model card; the file is corrupt or not a safetensors file."
    return 1
  fi
  if [ "$((8 + hlen))" -gt "$size" ]; then
    check_fail "shard header longer than file: $f" "the header declares ${hlen} bytes but the file is only ${size} bytes." \
      "Re-copy '$f'; the copy is truncated."
    return 1
  fi
  header="$(dd if="$path" bs=1 skip=8 count="$hlen" 2>/dev/null)"
  max_end="$(printf '%s' "$header" | grep -oE '"data_offsets"[[:space:]]*:[[:space:]]*\[[0-9]+[[:space:]]*,[[:space:]]*[0-9]+\]' | sed -E 's/.*,[[:space:]]*([0-9]+)\]/\1/' | sort -n | tail -1)"
  [ -n "$max_end" ] || max_end=0
  if [ "$((8 + hlen + max_end))" -gt "$size" ]; then
    check_fail "shard data truncated: $f" "the header maps weights up to byte $((8 + hlen + max_end)) but the file has only ${size} bytes (safetensors: incomplete metadata, file not fully covered)." \
      "Re-copy '$f' completely (e.g. rsync -a --partial --append-verify); a truncated copy makes vLLM fail at load time."
    return 1
  fi
}

check_model_shard_headers() { # check_model_shard_headers <model-dir>
  local dir="$1" f
  check_start shard-headers "safetensors shard headers fully cover their data"
  for f in model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors; do
    check_shard_integrity "$dir" "$f" || return 1
  done
  check_ok "3 shard headers consistent with file sizes"
}

# check_cuda_home <venv>: FlashInfer JIT and deep-gemm look up CUDA_HOME /
# nvcc in a subprocess that never sees the venv, so the launcher must export
# the venv CUDA tree. Real hardware failed mid-profile with "Could not find
# nvcc and default cuda_home='/usr/local/cuda' doesn't exist" even though the
# venv nvcc check passed. Verify the tree is resolvable before launch.
check_cuda_home() {
  local venv="$1" home nvcc
  check_start cuda-home "venv CUDA tree resolvable (FlashInfer/deep-gemm JIT)"
  nvcc="$(nvcc_in_venv "$venv")"
  home="$(cuda_home_in_venv "$venv" 2>/dev/null)"
  if [ -z "$home" ] || [ ! -x "$home/bin/nvcc" ] || [ ! -d "$home/include" ]; then
    check_fail "CUDA_HOME not resolvable (nvcc='${nvcc:-<none>}', home='${home:-<none>}')" \
      "FlashInfer JIT needs CUDA_HOME pointing at the venv CUDA tree (nvcc's parent's parent) with bin/nvcc and include/." \
      "Re-run: bash scripts/wsl2-env.sh create --force --prefix $(dirname "$venv")"
    return 1
  fi
  check_ok "CUDA_HOME=$home (nvcc + include present)"
}

# check_cuda_toolkit_versions <venv>: FlashInfer JIT bundles CCCL, which
# #errors at compile time when the CUDA compiler and the toolkit headers it
# sees disagree (cuda_toolkit.h: "CUDA compiler and CUDA toolkit headers are
# incompatible"). Real hardware hit this when the venv cu13 tree mixed nvcc
# 13.3 (nvidia-cuda-nvcc) with runtime headers 13.0 (nvidia-cuda-runtime,
# CUDART_VERSION 13000). Compare the nvcc release against CUDART_VERSION from
# the same tree, exactly as CCCL does. FAKE_CUDA_RUNTIME_VERSION overrides the
# header value (tests).
check_cuda_toolkit_versions() {
  local venv="$1" nvcc_bin home nvcc_lib rel header version
  local rel_major rel_minor hdr_major hdr_minor
  check_start cuda-toolkit-version "venv CUDA compiler matches its runtime headers (CCCL)"
  nvcc_bin="$(nvcc_in_venv "$venv")"
  home="$(cuda_home_in_venv "$venv" 2>/dev/null)"
  if [ -z "$nvcc_bin" ] || [ -z "$home" ] || [ ! -d "$home/include" ]; then
    return 0 # check_cuda_home already reported a broken tree; don't double-fail.
  fi
  nvcc_lib="$(dirname "$(dirname "$nvcc_bin")")/lib"
  rel="$(LD_LIBRARY_PATH="${nvcc_lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$nvcc_bin" --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | head -1)"
  rel="${rel#release }"
  if [ -z "$rel" ]; then
    check_fail "nvcc version unreadable ($nvcc_bin)" "the CCCL check needs the nvcc release to compare against the headers." \
      "Re-run: bash scripts/wsl2-env.sh create --force --prefix $(dirname "$venv")"
    return 1
  fi
  header="$home/include/cuda_runtime_api.h"
  [ -f "$header" ] || header="$home/include/cuda_runtime.h"
  if [ ! -f "$header" ]; then
    check_fail "no CUDA runtime header in $home/include" "the CCCL check reads CUDART_VERSION from cuda_runtime_api.h; neither it nor cuda_runtime.h exists." \
      "Re-run: bash scripts/wsl2-env.sh create --force --prefix $(dirname "$venv")"
    return 1
  fi
  version="$(sed -n 's/^#define CUDART_VERSION[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$header" | head -1)"
  case "$version" in
    ''|*[!0-9]*)
      check_fail "CUDART_VERSION unreadable from $header" "the CCCL check needs CUDART_VERSION to compare against the nvcc release." \
        "Re-run: bash scripts/wsl2-env.sh create --force --prefix $(dirname "$venv")"
      return 1
      ;;
  esac
  rel_major="${rel%%.*}"
  rel_minor="${rel#*.}"
  rel_minor="${rel_minor%%.*}"
  hdr_major=$((version / 1000))
  hdr_minor=$(((version % 1000) / 10))
  if [ "$rel_major" = "$hdr_major" ] && [ "$rel_minor" = "$hdr_minor" ]; then
    check_ok "nvcc $rel matches toolkit headers $hdr_major.$hdr_minor (CUDART_VERSION $version)"
  else
    check_fail "nvcc $rel but runtime headers are $hdr_major.$hdr_minor (CUDART_VERSION $version)" \
      "FlashInfer JIT bundles CCCL, which rejects a compiler/header mix at build time: cuda_toolkit.h '#error CUDA compiler and CUDA toolkit headers are incompatible' (real failure on this machine)." \
      "Align the CUDA packages in the venv to one version (match nvcc $rel_major.$rel_minor.x): $venv/bin/pip install --upgrade nvidia-cuda-runtime nvidia-cuda-nvrtc nvidia-cuda-cupti nvidia-nvtx"
    return 1
  fi
}

# check_build_tools: FlashInfer JIT builds its CUDA kernels with ninja. Real
# hardware failed mid-profile with "FileNotFoundError: [Errno 2] No such file
# or directory: 'ninja'" after CUDA_HOME was exported. NINJA_BIN overrides
# the probe (tests).
check_build_tools() {
  local ninja=""
  check_start build-tools "ninja build tool for FlashInfer JIT"
  if [ -n "${NINJA_BIN:-}" ]; then
    ninja="$NINJA_BIN"
  else
    ninja="$(command -v ninja 2>/dev/null || true)"
  fi
  if [ -n "$ninja" ] && [ -x "$ninja" ]; then
    check_ok "ninja: $ninja"
  else
    check_fail "ninja not found" "FlashInfer JIT builds its CUDA kernels with ninja (real error: FileNotFoundError: 'ninja')." \
      "Install it in Ubuntu: sudo apt-get update && sudo apt-get install -y ninja-build"
    return 1
  fi
}

# check_c_compiler <venv>: Triton builds its CUDA driver C shim on the first
# kernel run and needs a system C compiler (cc/gcc) AND the Python C headers
# (Python.h) matching the venv python. Real hardware failed mid-startup with
# "Failed to find C compiler" (no cc) and later "fatal error: Python.h: No
# such file or directory" (build-essential without python3-dev). This check
# compiles a trivial Python.h TU with the same include dir Triton uses.
# CC_BIN overrides the compiler probe (tests); CC_FAIL_COMPILE=1 forces the
# compile probe to fail (tests).
check_c_compiler() {
  local venv="$1" cc="" incdir="" probe_c probe_o rc
  check_start c-compiler "system C compiler + Python.h for Triton"
  if [ -n "${CC_BIN:-}" ]; then
    cc="$CC_BIN"
  else
    for c in cc gcc; do
      if command -v "$c" >/dev/null 2>&1; then cc="$(command -v "$c")"; break; fi
    done
  fi
  if [ -z "$cc" ] || [ ! -x "$cc" ]; then
    check_fail "no C compiler found (cc/gcc)" "Triton builds its CUDA driver C shim on the first kernel run and needs a system C compiler." \
      "Install it in Ubuntu: sudo apt-get update && sudo apt-get install -y build-essential"
    return 1
  fi
  incdir="$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["include"])' 2>/dev/null)"
  if [ -z "$incdir" ] || [ ! -d "$incdir" ]; then
    check_fail "cannot resolve the venv python include dir (got '${incdir:-<empty>}')" "Triton needs the Python C headers for the venv python." \
      "Install the matching dev package: sudo apt-get install -y python3-dev"
    return 1
  fi
  probe_c="$(mktemp)"
  probe_o="${probe_c}.o"
  printf '#include <Python.h>\nint main(void){return 0;}\n' > "$probe_c"
  if [ "${CC_FAIL_COMPILE:-0}" = "1" ]; then
    rc=1
  else
    "$cc" -I"$incdir" -x c -c "$probe_c" -o "$probe_o" 2>/dev/null
    rc=$?
  fi
  rm -f "$probe_c" "$probe_o"
  if [ "$rc" -eq 0 ]; then
    check_ok "C compiler: $cc; Python.h compile probe passed"
  else
    check_fail "Triton build probe failed (Python.h not found?)" "the C compiler exists but cannot compile a Python.h TU with include dir '$incdir' (real error: fatal error: Python.h: No such file or directory)." \
      "Install the Python dev headers: sudo apt-get update && sudo apt-get install -y python3-dev"
    return 1
  fi
}

check_vram_occupied() { # other compute processes on the GPU
  local apps used
  check_start vram-others "no other processes significantly using VRAM"
  apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep -v '^$')"
  used="$(printf '%s\n' "$apps" | sed -n 's/.*, *\([0-9][0-9]*\) *MiB.*/\1/p' | awk '{s+=$1} END{print s+0}')"
  if [ -z "$apps" ]; then
    check_ok "no other compute processes on the GPU"
  elif [ "$used" -lt "$VRAM_OCCUPIED_WARN_MIB" ]; then
    check_ok "other processes use ${used} MiB (warn level ${VRAM_OCCUPIED_WARN_MIB} MiB)"
  else
    check_warn "other programs hold ${used} MiB of VRAM; the full 262144-token config leaves little headroom."
    printf '%s\n' "$apps" | head -6 | sed 's/^/        note:    /'
    printf '%s\n' "Close GPU-using programs on Windows (browsers, video, other models), then 'wsl --shutdown' before a full-context launch." | sed 's/^/        fix:    /'
  fi
}

check_vram_available() { # free VRAM gate for short-context startup
  local line total used free
  check_start vram "free VRAM >= $((MIN_FREE_VRAM_MIB / 1024)) GiB for startup"
  line="$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)"
  total="${line%%,*}"
  used="${line#*,}"
  used="$(printf '%s' "$used" | trim)"
  case "$total" in
    ''|*[!0-9]*)
      check_fail "cannot read VRAM totals ('$line')" "nvidia-smi memory query returned nothing usable." \
        "Check the Windows driver, then 'wsl --shutdown' and re-run."
      return 1
      ;;
  esac
  case "$used" in ''|*[!0-9]*) used=0 ;; esac
  free=$((total - used))
  if [ "$free" -ge "$MIN_FREE_VRAM_MIB" ]; then
    check_ok "VRAM free ${free} MiB (total ${total} MiB, used ${used} MiB)"
  else
    check_fail "only ${free} MiB VRAM free of ${total} MiB" "short-context startup needs >= ${MIN_FREE_VRAM_MIB} MiB free." \
      "Close GPU-using programs on Windows (browsers, video, other models), then 'wsl --shutdown' and re-run."
  fi
}

# run_preflight <prefix> <model-dir>: the single preflight boundary.
# Exit 0 = READY (all checks passed), 1 = NOT READY.
run_preflight() {
  local prefix="$1" model_dir="$2" venv
  venv="$prefix/venv"
  PASSED=0
  FAILED=0
  WARNED=0

  section "Unified preflight (issue 02)"
  info "Target: offline startup of ModelOpt NVFP4 weights; vLLM ${VLLM_VERSION}; CUDA ${CUDA_MAJOR_VERSION}"
  run_prereq_checks

  check_start venv "runtime venv present ($prefix)"
  if [ ! -d "$venv" ]; then
    check_fail "venv not found" "no Python environment at $venv." \
      "Run: bash scripts/wsl2-env.sh create --prefix $prefix"
  else
    check_ok
  fi
  check_venv_python "$venv"
  check_nvcc_venv "$venv"
  check_vllm_venv "$venv"
  check_c_compiler "$venv"
  check_cuda_home "$venv"
  check_cuda_toolkit_versions "$venv"
  check_build_tools
  check_offline_mode
  check_model_path "$model_dir"
  check_model_files "$model_dir"
  check_model_index "$model_dir"
  check_model_shard_headers "$model_dir"

  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    check_vram_occupied
    check_vram_available
  fi

  checks_summary
  if [ "$FAILED" -eq 0 ]; then
    ok "preflight result: READY"
    return 0
  fi
  fail "preflight result: NOT READY"
  return 1
}
