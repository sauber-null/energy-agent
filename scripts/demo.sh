#!/usr/bin/env bash
# Start everything the web demo needs and keep the Mac awake while it runs.
# The Cloudflare tunnel starts too once ~/.cloudflared/config.yml exists.
# Stop with Ctrl+C.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

if [ -z "${DEMO_PASSWORD:-}" ] && ! grep -q '^DEMO_PASSWORD=.' .env 2>/dev/null; then
  echo "Set DEMO_PASSWORD in .env first (the demo never runs without a password)." >&2
  exit 1
fi

docker compose up -d --wait >/dev/null                     # Postgres + pgvector
if ! curl -sf http://localhost:11434/api/version >/dev/null; then
  open -a Ollama                                           # embeddings + local models
  until curl -sf http://localhost:11434/api/version >/dev/null; do sleep 1; done
fi

tunnel_config="$HOME/.cloudflared/config.yml"
if command -v cloudflared >/dev/null && [ -f "$tunnel_config" ]; then
  cloudflared tunnel run energy-agent &
  trap 'kill $! 2>/dev/null' EXIT
  # The public hostname lives only in the local config, not in the repository.
  echo "Tunnel started: https://$(awk '/hostname:/ {print $3; exit}' "$tunnel_config")"
else
  echo "No Cloudflare tunnel configured, the demo runs on http://127.0.0.1:8000 only."
fi

# caffeinate -i keeps the Mac from sleeping as long as the server runs.
caffeinate -i uv run energy-agent serve
