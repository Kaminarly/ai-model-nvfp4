#!/usr/bin/env bash
# start-153600-32k-both.sh - 32K output cap AND a 32K drafter window.
# Rationale: the engine declines speculation when prompt + max_tokens exceeds
# SPARKINFER_DSPARK_MAX_CTX (measured: engaged at 8192, gone at 16384), so a 32K
# output cap disables DSpark unless the drafter window is raised with it.
export SPARKINFER_EXTRA_ARGS="SPARKINFER_PREFIX_CACHE=0 SPARKINFER_MAX_OUTPUT_TOKENS=32768 SPARKINFER_DSPARK_MAX_CTX=32768"
exec bash /mnt/d/Code/MJ-Project/ai-model-nvfp4/scripts/sparkinfer-serve.sh start --context-length 153600