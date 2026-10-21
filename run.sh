#!/usr/bin/env bash
#
# One-command demo launcher (sanitized, env-driven).
# Starts Postgres + the GPU embedding server if needed, verifies the index,
# then runs the site in the foreground. Ctrl+C stops the site; shared
# services stay up for fast relaunch.
#
# Config via env:
#   SUDO_PASSWORD    sudo password if sudo needs it non-interactively
#   PG_CLUSTER       debian-style pg cluster name (default: main)
#   SSL_MODE         embed server: Vulkan (default) | cpu

set -euo pipefail
cd "$(dirname "$0")"

PORT=8002
EMBED_PORT=8081
PY="${PY:-../.venv/bin/python}"

log() { printf "\033[1;32m==>\033[0m %s\n" "$*"; }

# --- sudo ---------------------------------------------------------------
if [ -n "${SUDO_PASSWORD:-}" ]; then
    echo "$SUDO_PASSWORD" | sudo -S true 2>/dev/null
fi

# --- Postgres -----------------------------------------------------------
if ! pg_isready -q 2>/dev/null; then
    log "starting Postgres"
    if sudo -n true 2>/dev/null; then
        sudo pg_ctlcluster all "${PG_CLUSTER:-main}" start || sudo systemctl start postgresql
    else
        sudo systemctl start postgresql
    fi
    sleep 1
fi
if ! pg_isready -q 2>/dev/null; then
    log "Postgres failed to start — start it manually, then re-run."
    exit 1
fi
log "Postgres up"

# --- GPU embedding server ------------------------------------------------
if ! curl -s -m 2 "http://127.0.0.1:${EMBED_PORT}/health" >/dev/null 2>&1; then
    log "starting GPU embedding server [:${EMBED_PORT}]"
    ./embed-server.sh &
fi
log "waiting for embed server"
for _ in $(seq 1 60); do
    curl -s -m 2 "http://127.0.0.1:${EMBED_PORT}/health" >/dev/null 2>&1 && break
    sleep 1
done
curl -s -m 2 "http://127.0.0.1:${EMBED_PORT}/health" >/dev/null 2>&1 \
    || log "WARN: embed server not up — site still works, AI search is degraded"

# --- site ----------------------------------------------------------------
log "site: http://127.0.0.1:${PORT}"
exec "$PY" site.py
