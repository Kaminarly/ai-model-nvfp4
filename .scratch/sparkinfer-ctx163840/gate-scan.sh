#!/usr/bin/env bash
# gate-scan.sh - which max_tokens still leaves speculation enabled?
# Same short prompt every time (it stops at ~700 tokens), so the only variable
# is the per-request KV reservation (prompt + max_tokens).
cd /mnt/d/Code/MJ-Project/ai-model-nvfp4/.scratch/sparkinfer-ctx163840 || exit 1
for n in 4096 8192 16384 16385 20480 32768; do
  printf 'max_tokens=%-6s ' "$n"
  python3 probe.py --prompt code --repeats 1 --max-tokens "$n" 2>&1 | grep -E '^  #1' | sed 's/^  //'
done