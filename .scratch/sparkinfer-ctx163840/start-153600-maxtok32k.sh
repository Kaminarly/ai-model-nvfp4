#!/usr/bin/env bash
# start-153600-maxtok32k.sh - DSpark at ctx 153600 with the output cap raised to 32K.
# SPARKINFER_EXTRA_ARGS is split on spaces into one `-e KEY=VALUE` each, so both
# variables go in the same quoted string.
export SPARKINFER_EXTRA_ARGS="SPARKINFER_PREFIX_CACHE=0 SPARKINFER_MAX_OUTPUT_TOKENS=32768"
exec bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/sparkinfer-serve.sh start --context-length 153600