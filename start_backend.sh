#!/usr/bin/env bash
# Starts the Wingo backend (serves the frontend too). Reads Backend-main/.env.
# PORT is set by Render/Railway/Fly; defaults to 8099 locally.
# --timeout-keep-alive 75: uvicorn's default (5 s) closes idle connections that the host's proxy (Render/Vercel) still believes are open,
# which produces random 502/504 "disconnected" errors. Keep it LONGER than the proxy's idle timeout.
cd "$(dirname "$0")" && exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8099}" \
  --timeout-keep-alive 75 --proxy-headers --forwarded-allow-ips='*'
