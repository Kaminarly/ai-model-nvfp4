#!/usr/bin/env bash
# run-all-suites.sh - run every test suite and print its result line.
cd /mnt/d/Code/MJ-Project/ai-model-nvfp4 || exit 1
for t in run-tests preflight-tests serve-tests fullcontext-tests sparkinfer-tests; do
  printf '%-20s ' "$t"
  out="$(bash "tests/$t.sh" 2>&1)"
  line="$(printf '%s\n' "$out" | grep -E '^== results' | tail -1)"
  if [ -n "$line" ]; then
    printf '%s\n' "$line"
  else
    printf 'NO-RESULT -> %s\n' "$(printf '%s\n' "$out" | tail -2 | tr '\n' ' ')"
  fi
done