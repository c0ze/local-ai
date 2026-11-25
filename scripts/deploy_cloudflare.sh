#!/usr/bin/env bash
# Build and deploy the app to the local-ai-arda-tr Worker (local-ai.arda.tr).
# Same steps the old GitHub Pages workflow ran (npm ci, npm run build), then
# wrangler deploy with your local wrangler login (run `npx wrangler login`
# once if it asks). Builds the working tree, so it refuses to run unless the
# tree is clean: what is deployed is what is committed.
# Usage: npm run deploy  (or scripts/deploy_cloudflare.sh)
# Set SKIP_DEPLOY=1 to build without deploying.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree is not clean; commit or stash first:" >&2
  git status --short >&2
  exit 1
fi

# The pinned Node (.mise.toml), as the workflow pinned Node 24.
if command -v mise >/dev/null 2>&1; then
  eval "$(mise env -s bash)"
fi

# Node-side install scripts the browser build does not need: onnxruntime-node
# (via kokoro-js) fetches CUDA binaries on Linux x64, and its release asset is
# gone (404); sharp tries to build from source against a system libvips.
export ONNXRUNTIME_NODE_INSTALL_CUDA=skip
export SHARP_IGNORE_GLOBAL_LIBVIPS=1

npm ci
npm run build
echo "built dist/ from $(git rev-parse --short HEAD)"

if [ -n "${SKIP_DEPLOY:-}" ]; then
  exit 0
fi
npx --yes wrangler@4.168.0 deploy
