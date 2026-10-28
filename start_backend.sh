#!/usr/bin/env bash
# Starts the Wingo backend (serves the frontend too) on port 8099. Reads Backend-main/.env.
cd "$(dirname "$0")/Backend-main" && exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8099
