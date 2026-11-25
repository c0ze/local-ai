# AGENTS.md

Guidance for coding agents working in this repository.

## Project Summary

Local AI Chat is a static Vite/TypeScript browser app for local WebGPU chat inference with Transformers.js. The app has optional browser web search, local Kokoro TTS, local Moonshine STT, browser cache management, model compatibility tests, and a Node MCP bridge sidecar.

There is no backend for chat inference or persisted chat state. Browser storage is the source of truth.

## Key Files

- `src/main.ts`: main UI/controller. Most user-visible behavior is orchestrated here.
- `src/model.ts`: Transformers.js model manager, loading, streaming generation, prompt formatting, context trimming, WebGPU checks.
- `src/models.ts`: model registry, aliases, dtype support, approximate weights, context windows, loader selection.
- `src/storage.ts`: validated `localStorage` persistence for threads and settings.
- `src/webSearch.ts`: SearxNG/DuckDuckGo/Wikipedia/current-time search context.
- `src/mcp.ts`: browser MCP client, tool-selection prompts, tool-result prompts, tool-call JSON parser.
- `scripts/mcp-bridge.mjs`: local MCP HTTP bridge for stdio, Streamable HTTP, and SSE servers.
- `src/modelCache.ts`: browser Cache API inspection/deletion for Hugging Face model files.
- `src/tts.ts`: Kokoro TTS manager.
- `src/stt.ts`: Moonshine STT manager.
- `src/tester.ts`: in-app compatibility test runner.
- `tests/model-compatibility.spec.ts`: Playwright model compatibility suite.
- `docs/ARCHITECTURE.md`: deeper architecture notes.

## Commands

```bash
npm install
npm run dev
npm run typecheck
npm run typecheck:tests
npm run build
npm run mcp:bridge
```

Model tests are expensive because they download and load browser models:

```bash
npm run test:models:install
npm run test:models
MODEL_TEST_MODELS=qwen npm run test:models
MODEL_TEST_DTYPES=all npm run test:models
```

Prefer `npm run typecheck` for normal code edits. Run model tests only when changing model loading, dtype selection, registry metadata, or compatibility behavior.

## Deploy

Deploying is manual. Pushing to GitHub does **not** deploy (the workflow is only a build
check), and nothing deploys on its own. To publish:

1. Commit the change. `scripts/deploy_cloudflare.sh` refuses a dirty working tree, then runs `npm ci` and `npm run build` and uploads `dist/`.
2. Run `npm run deploy` on a machine with the owner's wrangler login (`npx wrangler whoami`
   shows the personal account). Never create or store a Cloudflare API token for this.
3. Check the live site: `curl -sI https://local-ai.arda.tr/` should return 200.

Workers static assets take files up to 25 MiB. The largest file today is the onnxruntime `.wasm` (about 23.6 MB); an onnxruntime or transformers upgrade can push it over the limit, and the deploy then fails.

The Worker is `local-ai-arda-tr` (personal Cloudflare account), custom domain `local-ai.arda.tr`, both in
[wrangler.jsonc](wrangler.jsonc). README "Deployment" has the details. Until 2026-11-25
this was GitHub Pages, which could not renew its certificate behind the Cloudflare proxy.

## Runtime Assumptions

- Chat model inference is WebGPU-only.
- `localhost` is acceptable as a secure context for development.
- Transformers.js remote model loading is enabled; local model loading is disabled in `src/model.ts`.
- Model files live in the browser Cache API, usually under `transformers-cache`.
- Threads/settings/test results live in `localStorage`.
- The MCP bridge is a local development/runtime sidecar, not an inference server.

## Implementation Rules

- Keep state validation in `src/storage.ts` when changing persisted shapes.
- Keep model metadata and model-specific compatibility decisions in `src/models.ts`.
- Do not hard-code model registry assumptions in `src/main.ts` if they belong in `ModelInfo`.
- Keep provider-specific search/network code in `src/webSearch.ts`.
- Keep MCP prompt and parsing behavior in `src/mcp.ts`; keep transport/server behavior in `scripts/mcp-bridge.mjs`.
- Preserve single-turn cancellation through `AbortController` when adding async work to generation.
- Preserve the distinction between persisted chat messages and temporary prompt augmentation. Web search and MCP prompts should not overwrite the saved user message.
- Be careful with browser-only APIs during module initialization. Most code should access `window`, `navigator`, `caches`, and audio APIs inside functions or guarded branches.
- Do not add server-side persistence unless the product direction explicitly changes.

## UI Notes

- The UI is vanilla TypeScript plus Tailwind classes in `index.html` and `src/main.ts`.
- There is no component framework.
- Keep controls compact and app-like; this is an operational chat tool, not a marketing page.
- If adding settings, wire default values, normalization, save/load, and UI rendering together.

## Tool And Privacy Boundaries

- Web search sends user queries to configured external providers when enabled.
- MCP tool calls can read, write, or exfiltrate data depending on the exposed server. Treat MCP server configuration as trusted-user territory.
- Gated/private Hugging Face models are unsupported because browser code cannot safely hold private tokens.
- TTS/STT are intended to run locally, though their model assets are downloaded like other public ONNX models.

## Documentation Expectations

When changing architecture, tools, model loading, storage keys, commands, or user-visible abilities, update:

- `README.md`
- `docs/ARCHITECTURE.md`
- `AGENTS.md` if future agent workflow changes

Keep the README user-facing. Put deeper control-flow and extension details in `docs/ARCHITECTURE.md`.
