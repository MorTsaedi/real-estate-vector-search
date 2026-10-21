#!/usr/bin/env bash
#
# GPU embedding server: EmbeddingGemma 2 (Q8_0 + mmproj) on the RX 5700 XT
# via the Vulkan llama.cpp build. Serves:
#   text   POST /embedding {"content": "title: none | text: ..."}
#   image  POST /embedding {"content": [{"prompt_string": "<MARKER>",
#                                        "multimodal_data": ["<base64>"]}]}
# where MARKER comes from GET /props -> media_marker (random per start).
#
# Usage:  ./embed-server.sh            # foreground, :8081, Ctrl+C to stop
#
set -euo pipefail
cd "$(dirname "$0")"

PORT="${EMBED_PORT:-8081}"
LS="bin/llama-b11512/llama-server"

[ -x "$LS" ] || { echo "ERROR: Vulkan llama-server missing at $LS"; exit 1; }
[ -f models/embeddinggemma-2-Q8_0.gguf ] || {
    echo "ERROR: models/embeddinggemma-2-Q8_0.gguf missing"; exit 1; }

if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "embedding server already up on :$PORT (marker: $(curl -s -m 5 http://127.0.0.1:$PORT/props | grep -o '"media_marker": *"[^"]*"' ))"
    exit 0
fi

echo "== EG2 embedding server on :$PORT (GPU via Vulkan, Ctrl+C to stop)"
exec "$LS" \
    -m models/embeddinggemma-2-Q8_0.gguf \
    --mmproj models/mmproj-embeddinggemma-2-Q8_0.gguf \
    --embedding --pooling mean \
    -ngl 99 -c 8192 -b 8192 -ub 8192 \
    --port "$PORT"
