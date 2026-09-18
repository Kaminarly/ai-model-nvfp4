#!/usr/bin/env bash
# run-suites.sh - re-run every project test suite and print a one-line verdict
# per suite (plus that suite's own summary line). The suites use fake tools in
# tests/fakebin, so this needs bash but no GPU.
#
#   bash .scratch/vllm-launcher-cleanup/run-suites.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

for suite in run preflight serve fullcontext sparkinfer; do
  if [ "$suite" = run ]; then
    script="$ROOT/tests/run-tests.sh"
  else
    script="$ROOT/tests/${suite}-tests.sh"
  fi
  log="/tmp/dsh-${suite}-tests.log"
  if bash "$script" >"$log" 2>&1; then
    status="OK    "
  else
    status="FAILED"
  fi
  printf '%-11s %s  %s\n' "$suite" "$status" "$(tail -1 "$log")"
done