#!/usr/bin/env bash
# Launch a vLLM server with LMCache as the KV connector.
#   scripts/launch_vllm_lmcache.sh <model> <lmcache_config.yaml> [port]
# Variants for E2: (a) no LMCache  (b) configs/lmcache_baseline.yaml  (c) tierquant serde (see docs/design.md)
# Flag names change between vLLM/LMCache releases; check `vllm serve --help` and the LMCache docs.
set -euo pipefail
MODEL=${1:?model}; CFG=${2:-}; PORT=${3:-8000}
if [[ -n "$CFG" ]]; then
  export LMCACHE_CONFIG_FILE="$CFG"
  exec vllm serve "$MODEL" --port "$PORT" --enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
else
  exec vllm serve "$MODEL" --port "$PORT" --enable-prefix-caching
fi
