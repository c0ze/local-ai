# Local AI Chat

Local AI Chat is a private browser chat app for running public ONNX language models directly in the browser with WebGPU. The app is a static Vite/TypeScript site; there is no inference server, and chat state lives in browser storage.

Optional tools can add outside context:

- Web search through a user-configured SearxNG instance, with DuckDuckGo Instant Answer and Wikipedia fallbacks.
- MCP tools through a small local Node bridge that can connect to stdio, Streamable HTTP, or SSE MCP servers.
- Local text-to-speech through Kokoro.
- Local speech-to-text through Moonshine.

## What It Can Do

- Run local chat inference with Transformers.js and WebGPU.
- Select from a curated registry of public Hugging Face ONNX models, or enter a custom compatible model ID.
- Pick or auto-select dtype/quantization based on model support and detected GPU capability.
- Show GPU details, shader-f16 support, approximate model memory, and context window usage.
- Keep multiple chat threads in `localStorage`, including rename, favorite, delete, and Markdown export.
- Compact older conversation turns into a local-model summary to reduce future prompt size.
- Inspect and delete downloaded Hugging Face model files from the browser Cache API.
- Run browser-side model compatibility checks and save per-GPU results in `localStorage`.
- Dictate prompts locally with Moonshine and play assistant responses locally with Kokoro.
- Inject web search context into prompts when search is enabled.
- Let the local model choose one MCP tool call, execute it through the bridge, then answer from the tool result.

## Requirements

- Node.js 24.14.0 or newer for local development.
- A secure browser context. `localhost` works for development.
- A WebGPU-capable browser and GPU for chat inference.

| Browser | WebGPU status |
| --- | --- |
| Chrome / Edge 113+ | Supported on capable hardware |
| Safari 17.4+ | Supported on macOS Sonoma or newer |
| Firefox | Nightly only, behind `dom.webgpu.enabled` |

If Settings reports `google - swiftshader`, a software fallback adapter, or a low 1 GB buffer limit on Linux/AMD, Chrome is probably not using the real GPU.

### Chrome on Linux / AMD

1. Open `chrome://flags`.
2. Enable `Unsafe WebGPU`.
3. Enable `Vulkan`.
4. Relaunch Chrome.
5. Check Local AI Settings for the real GPU, `shader-f16`, and a max buffer above 1 GB.

If video playback turns white after enabling Vulkan, try `Choose ANGLE graphics backend -> OpenGL`, then disable `Zero-copy rasterizer`. If needed, disable only `Hardware-accelerated video decode`. Do not disable Chrome's general graphics acceleration setting, because that disables the WebGPU path this app needs.

## Development

```bash
npm install
npm run dev
```

`npm run dev` starts both Vite and the local MCP bridge. The bridge uses `mcp.config.json` if present, otherwise `mcp.config.example.json`.

The app has no required environment variables. `VITE_BASE_URL` is optional when serving the built site from a subpath.

Useful commands:

| Command | Purpose |
| --- | --- |
| `npm run dev` | Start Vite and the MCP bridge together. |
| `npm run dev:vite` | Start only the Vite dev server. |
| `npm run mcp:bridge` | Start only the MCP bridge. |
| `npm run typecheck` | Type-check the application. |
| `npm run typecheck:tests` | Type-check the Playwright model tests. |
| `npm run build` | Type-check and build the static site into `dist/`. |
| `npm run preview` | Preview the production build. |
| `npm audit --audit-level=moderate` | Check dependencies for moderate or higher vulnerabilities. |

## Architecture

The application is intentionally thin:

- `index.html` contains the app shell and settings/modals.
- `src/main.ts` owns UI state, thread rendering, settings, model lifecycle, search/MCP orchestration, TTS/STT controls, and cache/test UI.
- `src/model.ts` wraps Transformers.js model loading, prompt formatting, streaming generation, dtype selection fallback, WebGPU checks, and context trimming.
- `src/models.ts` is the model registry with dtype support, approximate weights, buffer requirements, context windows, aliases, and loader selection.
- `src/storage.ts` validates and persists threads/settings in `localStorage`.
- `src/webSearch.ts` fetches SearxNG, DuckDuckGo, Wikipedia, and computed current-time facts, then formats prompt context.
- `src/mcp.ts` talks to the local bridge and formats the model prompts used for MCP tool selection and final answers.
- `scripts/mcp-bridge.mjs` exposes `/health`, `/tools`, and `/tools/call` and connects to configured MCP servers.
- `src/modelCache.ts` reads and deletes Transformers.js/Hugging Face entries in the browser Cache API.
- `src/tts.ts` and `src/stt.ts` implement local Kokoro speech playback and Moonshine dictation.
- `src/tester.ts` and `tests/model-compatibility.spec.ts` run model compatibility checks in the browser.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the runtime flow, tool behavior, storage model, and main extension points.

## Models

Models are loaded from Hugging Face by Transformers.js. First load downloads model assets into the browser cache; later loads reuse cached files when available.

The built-in registry currently includes small sanity-check models through larger chat/coding models such as Qwen, Llama, DeepSeek, Phi, and Gemma-family ONNX presets. Custom models must be public, browser-loadable, and compatible with Transformers.js ONNX loading.

Important constraints:

- Gated/private Hugging Face models cannot be loaded safely from client-side browser code.
- WebGPU is required for chat inference.
- Larger models can fail because of browser/GPU buffer limits, even when total VRAM is higher.
- `q4f16` and `fp16` require `shader-f16`; use `q4` or `q8` on GPUs without it.

Use the Settings compatibility tests to record what loads and generates on the current browser/GPU. Playwright can run the same checks:

```bash
npm run test:models:install
npm run test:models
MODEL_TEST_DTYPES=all npm run test:models
MODEL_TEST_MODELS=qwen npm run test:models
PW_HEADLESS=1 npm run test:models
```

Optional variables include `PORT`, `PLAYWRIGHT_BASE_URL`, `MODEL_TEST_LOAD_TIMEOUT_MS`, `MODEL_TEST_INFER_TIMEOUT_MS`, `MODEL_TEST_RESULTS_DIR`, `MODEL_TEST_QUESTION`, `MODEL_TEST_DTYPES`, and `MODEL_TEST_INCLUDE_TOO_LARGE`.

## Web Search

Enable web search in Settings -> Tools. When enabled, each user prompt is searched before generation. The app tries a configured SearxNG instance first, then fills missing results with DuckDuckGo Instant Answer and Wikipedia. It also computes current-time facts for simple time-zone queries through browser `Intl`.

The search context is injected into the latest user message, and the model is instructed to cite result snippets with `[1]`, `[2]`, or direct facts with `[F1]`.

For SearxNG, the browser must be allowed to call the instance cross-origin:

```yaml
search:
  formats:
    - html
    - json
```

Your reverse proxy or SearxNG instance must also return `Access-Control-Allow-Origin` for the app origin, or `*`.

Search queries leave the browser while search is enabled. Model inference still runs locally.

## MCP Tools

The static app cannot spawn stdio MCP servers directly, so it talks to a local bridge. The bridge can connect to:

- Servers configured in `mcp.config.json`, including stdio servers.
- Per-browser-session HTTP/SSE server URLs saved in Settings -> Tools.

Start the bridge:

```bash
npm run mcp:bridge
```

Default bridge URL:

```text
http://localhost:8787
```

Example `mcp.config.json`:

```json
{
  "port": 8787,
  "servers": {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/absolute/path/to/allowed/folder"]
    },
    "remote": {
      "transport": "http",
      "url": "http://localhost:3000/mcp"
    }
  }
}
```

At generation time, the app lists tools, asks the loaded local model whether one tool call is needed, parses a JSON tool call if the model returns one, calls the bridge, then asks the model to answer from the tool result. The current implementation supports one MCP tool call per user prompt.

Only expose MCP servers you trust. Tool capabilities depend entirely on the server; they may read files, write files, call APIs, or send data elsewhere.

## Speech

Text-to-speech uses `onnx-community/Kokoro-82M-v1.0-ONNX` and runs locally in the browser. It loads lazily, supports Kokoro American/British English voices, synthesizes long responses in sentence chunks, and can pause/resume playback.

Speech-to-text uses `onnx-community/moonshine-base-ONNX`. The microphone button records up to 30 seconds, decodes to mono 16 kHz audio, and transcribes locally. Moonshine uses WebGPU when available and falls back to WASM.

## Storage

- Threads: `localStorage` key `local-ai-chat-threads`.
- Settings: `localStorage` key `local-ai-chat-settings`.
- Compatibility results: `localStorage` key `local-ai-chat-test-results`.
- Model files: browser Cache API key used by Transformers.js, usually `transformers-cache`.

Clearing downloaded models does not delete chat history. Clearing site data will remove chats, settings, compatibility results, and cached models.

## Deployment

`local-ai.arda.tr` is a Cloudflare Worker with static assets (`local-ai-arda-tr`, see [wrangler.jsonc](wrangler.jsonc)), deployed **by hand**:

```bash
npm run deploy
```

[scripts/deploy_cloudflare.sh](scripts/deploy_cloudflare.sh) refuses a dirty working tree, runs `npm ci` and `npm run build`, and uploads `dist/` with `wrangler deploy` using your local wrangler login. The [build check](.github/workflows/build.yml) runs the same build on every push. Until 2026-11-25 the site was on GitHub Pages; GitHub could not renew its certificate behind the Cloudflare proxy.

The app can run from any static hosting. If you need MCP tools from a deployed HTTPS page, run the bridge on the same computer as the browser and keep Settings -> MCP bridge pointed at `http://localhost:8787`. The bridge sends CORS and Private Network Access headers for local companion access.
