#!/usr/bin/env bash
# Test runner for the SparkInfer launcher (scripts/sparkinfer-serve.sh). Bash
# only; uses fake docker/ss/nvidia-smi from tests/fakebin so it runs without
# WSL2, Docker or a GPU.
#
# What it locks down:
#   * the exact docker argv (both :ro mounts, -p <port>:8080, serve-dspark)
#   * the three-fence offline guarantee (mount completeness checked up front,
#     HF_HUB_OFFLINE=1 always, --no-download opt-in)
#   * foreground lifetime: never -d, never --restart, never --rm
#   * every preflight failure exits 1 with an executable fix, and malformed
#     input exits 2
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/scripts/sparkinfer-serve.sh"
FAKEBIN="$ROOT/tests/fakebin"
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

# --- fixtures --------------------------------------------------------------
# A complete target folder and a complete draft folder, plus the broken
# variants the preflight must refuse.
make_target() { # make_target <dir>
  mkdir -p "$1"
  printf '{"text_config":{"model_type":"qwen3_5_text"}}\n' > "$1/config.json"
  printf '{"version":"1.0"}\n' > "$1/tokenizer.json"
}
make_draft() { # make_draft <dir>
  mkdir -p "$1"
  printf '{"dflash_config":{"block_size":7}}\n' > "$1/config.json"
  printf 'stub\n' > "$1/model.safetensors"
}

MODEL_OK="$TMP_ROOT/target-ok"; make_target "$MODEL_OK"
MODEL_NO_CONFIG="$TMP_ROOT/target-no-config"; make_target "$MODEL_NO_CONFIG"; rm "$MODEL_NO_CONFIG/config.json"
MODEL_NO_TOK="$TMP_ROOT/target-no-tok"; make_target "$MODEL_NO_TOK"; rm "$MODEL_NO_TOK/tokenizer.json"
DRAFT_OK="$TMP_ROOT/draft-ok"; make_draft "$DRAFT_OK"
DRAFT_NO_CFG="$TMP_ROOT/draft-no-config"; make_draft "$DRAFT_NO_CFG"; rm "$DRAFT_NO_CFG/config.json"
DRAFT_NO_ST="$TMP_ROOT/draft-no-st"; make_draft "$DRAFT_NO_ST"; rm "$DRAFT_NO_ST/model.safetensors"

# ---------------------------------------------------------------------------
# 1. Dry-run: the exact argv contract
# ---------------------------------------------------------------------------
section "sparkinfer: dry-run argv (DSpark default)"
DLOG="$TMP_ROOT/docker.log"
out="$(run_env -- start --dry-run --model-dir "$MODEL_OK" --draft-dir "$DRAFT_OK")"; code=$?
expect_exit 0 "$code" "dry-run exits 0"
expect_contains "$out" "-v $MODEL_OK:/models/qwen38-nvfp4:ro" "target mounted read-only at the image's own path"
expect_contains "$out" "-v $DRAFT_OK:/models/qwen38-dspark:ro" "draft mounted read-only at the image's own path"
expect_contains "$out" "-p 8192:8080" "host 8192 mapped to the container's fixed 8080"
expect_contains "$out" "serve-dspark" "DSpark mode selected via the image's own subcommand"
expect_contains "$out" "-e HF_HUB_OFFLINE=1" "offline fence set"
expect_contains "$out" "-e CTX=153600" "DSpark context default is the measured 153600 sweet spot"
expect_contains "$out" "-e MODEL_NAME=Qwen3.8-27B-NVFP4-DSpark" "DSpark model id default"
expect_contains "$out" "ghcr.io/gittensor-ai-lab/sparkinfer-qwen38:0.5.10" "pinned image tag, not latest"
expect_not_contains "$out" "SPARKINFER_NO_DOWNLOAD" "hard-offline vars are opt-in only"
expect_not_contains "$out" " --rm" "no --rm so docker logs survives"
expect_not_contains "$out" " --restart" "no restart policy"
expect_not_contains "$out" " -d " "never detached"

section "sparkinfer: dry-run argv (--no-spec)"
out="$(run_env -- start --dry-run --no-spec --model-dir "$MODEL_OK" --draft-dir "$DRAFT_OK")"; code=$?
expect_exit 0 "$code" "--no-spec dry-run exits 0"
expect_not_contains "$out" "/models/qwen38-dspark" "no draft mount without speculation"
expect_not_contains "$out" "serve-dspark" "no serve-dspark without speculation"
expect_contains "$out" "-e CTX=262144" "autoregressive context default is the native window"
expect_contains "$out" "-e MODEL_NAME=Qwen3.8-27B-NVFP4 " "autoregressive model id default"

section "sparkinfer: overrides"
out="$(run_env -- start --dry-run --port 9001 --context-length 65536 --model-name Custom \
        --name probe --image example/img:1.2.3 --model-dir "$MODEL_OK" --draft-dir "$DRAFT_OK")"; code=$?
expect_exit 0 "$code" "override dry-run exits 0"
expect_contains "$out" "-p 9001:8080" "port override keeps the container side at 8080"
expect_contains "$out" "-e CTX=65536" "context override honoured"
expect_contains "$out" "-e MODEL_NAME=Custom" "model id override honoured"
expect_contains "$out" "--name probe" "container name override honoured"
expect_contains "$out" "example/img:1.2.3" "image override honoured"

out="$(run_env SPARKINFER_CTX=4096 SPARKINFER_SAMPLING_DEFAULTS=greedy SPARKINFER_KV_INT8=0 \
        SPARKINFER_IMAGE=example/env:9 SPARKINFER_NAME=envname \
        -- start --dry-run --model-dir "$MODEL_OK" --draft-dir "$DRAFT_OK")"; code=$?
expect_exit 0 "$code" "env-override dry-run exits 0"
expect_contains "$out" "-e CTX=4096" "SPARKINFER_CTX honoured"
expect_contains "$out" "-e SPARKINFER_SAMPLING_DEFAULTS=greedy" "sampling default passthrough"
expect_contains "$out" "-e SPARKINFER_KV_INT8=0" "int8-KV lever passthrough"
expect_contains "$out" "example/env:9" "SPARKINFER_IMAGE honoured"
expect_contains "$out" "--name envname" "SPARKINFER_NAME honoured"

out="$(run_env -- start --dry-run --no-download --model-dir "$MODEL_OK" --draft-dir "$DRAFT_OK")"; code=$?
expect_exit 0 "$code" "--no-download dry-run exits 0"
expect_contains "$out" "-e SPARKINFER_NO_DOWNLOAD=1" "--no-download sets the entrypoint's hard refusal"
expect_contains "$out" "-e MANIFEST_PATH=/nonexistent" "--no-download skips the pool manifest hash gate"

out="$(run_env -- start --dry-run --image example/img:latest --model-dir "$MODEL_OK" --draft-dir "$DRAFT_OK")"; code=$?
expect_exit 0 "$code" "floating tag still runs"
expect_contains "$out" "floating" "floating tag is called out as not reproducible"

# ---------------------------------------------------------------------------
# 2. Preflight refusals: every one must exit 1 with an executable fix
# ---------------------------------------------------------------------------
section "sparkinfer: refuses to start on incomplete mounts"
out="$(run_env -- start --model-dir "$MODEL_NO_CONFIG" --draft-dir "$DRAFT_OK")"; code=$?
expect_exit 1 "$code" "target without config.json is refused"
expect_contains "$out" "no config.json in" "names the missing file"
expect_contains "$out" "download the weights from Hugging Face" "explains why it matters"

out="$(run_env -- start --model-dir "$MODEL_NO_TOK" --draft-dir "$DRAFT_OK")"; code=$?
expect_exit 1 "$code" "target without tokenizer.json is refused"
expect_contains "$out" "no tokenizer.json in" "names the missing tokenizer"

out="$(run_env -- start --model-dir "$MODEL_OK" --draft-dir "$DRAFT_NO_CFG")"; code=$?
expect_exit 1 "$code" "draft without config.json is refused"
expect_contains "$out" "DSpark draft is incomplete" "reports the incomplete draft"
expect_contains "$out" "--no-spec" "offers the no-spec escape"

out="$(run_env -- start --model-dir "$MODEL_OK" --draft-dir "$DRAFT_NO_ST")"; code=$?
expect_exit 1 "$code" "draft without model.safetensors is refused"

out="$(run_env -- start --model-dir "$TMP_ROOT/nope" --draft-dir "$DRAFT_OK")"; code=$?
expect_exit 1 "$code" "missing model dir is refused"
expect_contains "$out" "target model folder not found" "names the bad path"

out="$(run_env -- start --no-spec --model-dir "$MODEL_OK" --draft-dir "$TMP_ROOT/nope")"; code=$?
expect_exit 0 "$code" "--no-spec does not need the draft folder"

section "sparkinfer: refuses to start on environment problems"
out="$(run_env FAKE_DOCKER_INFO=fail -- start --model-dir "$MODEL_OK" --draft-dir "$DRAFT_OK")"; code=$?
expect_exit 1 "$code" "unreachable docker daemon is refused"
expect_contains "$out" "cannot talk to the docker daemon" "clear reason"
expect_contains "$out" "systemctl start docker" "executable fix"

out="$(run_env FAKE_DOCKER_IMAGES=missing -- start --model-dir "$MODEL_OK" --draft-dir "$DRAFT_OK")"; code=$?
expect_exit 1 "$code" "missing image is refused"
expect_contains "$out" "is not present locally" "clear reason"
expect_contains "$out" "docker pull ghcr.io/gittensor-ai-lab/sparkinfer-qwen38:0.5.10" "exact pull command in the fix"

out="$(run_env FAKE_SS_LISTEN=8192 -- start --model-dir "$MODEL_OK" --draft-dir "$DRAFT_OK")"; code=$?
expect_exit 1 "$code" "busy port is refused"
expect_contains "$out" "already in use" "clear reason"
expect_contains "$out" "only one engine can run at a time" "explains the mutual exclusion"

out="$(run_env -- start --dry-run --port 81920 --model-dir "$MODEL_OK" --draft-dir "$DRAFT_OK")"; code=$?
expect_exit 2 "$code" "out-of-range port exits 2"

# docker itself absent from PATH.
out="$(env PATH="/usr/bin:/bin" bash "$SCRIPT" start --model-dir "$MODEL_OK" --draft-dir "$DRAFT_OK" 2>&1)"; code=$?
if command -v docker >/dev/null 2>&1; then
  say "skip - a real docker is on PATH; cannot test the absent-docker branch here"
else
  expect_exit 1 "$code" "absent docker is refused"
  expect_contains "$out" "docker is not installed" "clear reason"
fi

# ---------------------------------------------------------------------------
# 3. CLI guardrails and a stale container
# ---------------------------------------------------------------------------
section "sparkinfer: cli guardrails"
out="$(run_env -- start --model-dir "$MODEL_OK" --draft-dir "$DRAFT_OK" --port abc)"; code=$?
expect_exit 2 "$code" "non-numeric port exits 2"
expect_contains "$out" "must be a positive integer" "port validated"

out="$(run_env -- start --model-dir "$MODEL_OK" --draft-dir "$DRAFT_OK" --context-length x)"; code=$?
expect_exit 2 "$code" "non-numeric context exits 2"

out="$(run_env -- start --model-dir "$MODEL_OK" --draft-dir "$DRAFT_OK" --bogus)"; code=$?
expect_exit 2 "$code" "unknown option exits 2"
expect_contains "$out" "unknown option: --bogus" "names the bad option"

out="$(run_env -- bogus)"; code=$?
expect_exit 2 "$code" "unknown command exits 2"

out="$(run_env -- help)"; code=$?
expect_exit 0 "$code" "help exits 0"
expect_contains "$out" "sparkinfer-serve.sh start" "help shows usage"

out="$(run_env -- )"; code=$?
expect_exit 0 "$code" "no arguments prints usage and exits 0"

section "sparkinfer: stale container is removed, then docker run happens"
DLOG="$TMP_ROOT/docker-run.log"; rm -f "$DLOG"
out="$(run_env FAKE_DOCKER_CONTAINER=present FAKE_DOCKER_LOG="$DLOG" \
        -- start --model-dir "$MODEL_OK" --draft-dir "$DRAFT_OK")"; code=$?
expect_exit 0 "$code" "start with a stale container exits 0 (fake docker run)"
expect_contains "$out" "removing the existing container" "stale container reported"
expect_contains "$(cat "$DLOG" 2>/dev/null)" "rm -f qwen38-sparkinfer" "stale container removed before run"
expect_contains "$(cat "$DLOG" 2>/dev/null)" "run --name qwen38-sparkinfer --gpus all" "foreground docker run reached"
expect_contains "$out" "Ctrl-C to stop" "stop instruction printed"
expect_contains "$out" "docker logs qwen38-sparkinfer" "log recovery path printed"

section "sparkinfer: --dry-run starts nothing"
DLOG="$TMP_ROOT/docker-dry.log"; rm -f "$DLOG"
out="$(run_env FAKE_DOCKER_LOG="$DLOG" -- start --dry-run --model-dir "$MODEL_OK" --draft-dir "$DRAFT_OK")"; code=$?
expect_exit 0 "$code" "dry-run exits 0"
expect_not_contains "$(cat "$DLOG" 2>/dev/null)" "run --name" "dry-run never calls docker run"

# ---------------------------------------------------------------------------
# 4. Static guardrails on the script itself
# ---------------------------------------------------------------------------
section "sparkinfer: script never downloads or converts"
for bad in snapshot_download huggingface-cli "wget" "curl " "git clone" "gguf" "awq" "gptq"; do
  if grep -q "$bad" "$SCRIPT"; then
    FAIL=$((FAIL + 1)); say "FAIL - $SCRIPT contains forbidden '$bad'"
  else
    PASS=$((PASS + 1)); say "ok   - $SCRIPT has no '$bad'"
  fi
done

# The plan's escape hatch does not exist in this build; the launcher must not
# pretend otherwise (see result/sparkinfer-prefill-attn-gqa-rqh-research.md).
if grep -q "SPARKINFER_PREFILL_ATTN_GQA_RQH" "$SCRIPT"; then
  FAIL=$((FAIL + 1)); say "FAIL - launcher passes an env var the 0.5.10 binary does not read"
else
  PASS=$((PASS + 1)); say "ok   - launcher does not pass SPARKINFER_PREFILL_ATTN_GQA_RQH"
fi

# LF-only and ASCII-only: a .sh with CRLF breaks under bash, and this project
# has been bitten by encoding before.
if grep -q $'\r' "$SCRIPT"; then
  FAIL=$((FAIL + 1)); say "FAIL - $SCRIPT contains CR (must be LF)"
else
  PASS=$((PASS + 1)); say "ok   - $SCRIPT is LF-only"
fi
if LC_ALL=C grep -qP '[^\x00-\x7F]' "$SCRIPT" 2>/dev/null; then
  FAIL=$((FAIL + 1)); say "FAIL - $SCRIPT contains non-ASCII bytes"
else
  PASS=$((PASS + 1)); say "ok   - $SCRIPT is pure ASCII"
fi

say ""
say "== results: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]