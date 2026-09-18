#!/usr/bin/env bash
# dryrun-maxtok32k.sh - verify the 32K output-cap passthrough without launching.
# Uses --port 8193 so the port-in-use preflight does not abort before printing argv
# (the running engine owns 8192).
export SPARKINFER_EXTRA_ARGS="SPARKINFER_PREFIX_CACHE=0 SPARKINFER_MAX_OUTPUT_TOKENS=32768"
exec bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/sparkinfer-serve.sh start \
  --dry-run --port 8193 --context-length 153600