# Architecture

Local AI Chat is a static browser application. The browser owns the UI, state, model runtime, speech pipelines, cache management, and optional web-search calls. A small Node sidecar is used only for MCP because a static page cannot spawn local stdio processes.

## Runtime Shape

```text
Browser UI
  |
  |-- localStorage
  |     |-- chat threads
  |     |-- settings
  |     `-- model compatibility results
  |
  |-- Cache API
  |     `-- Transformers.js Hugging Face model files
  |
  |-- Transformers.js
  |     |-- chat model on WebGPU
  |     |-- Moonshine STT on WebGPU or WASM
  |     `-- Kokoro TTS through kokoro-js
  |
  |-- optional web fetches
  |     |-- SearxNG
  |     |-- DuckDuckGo Instant Answer
  |     `-- Wikipedia
  |
  `-- optional MCP bridge on localhost
        |-- configured stdio MCP servers
        |-- Streamable HTTP MCP servers
        `-- SSE MCP servers
```

## Main Modules

| Area | Files | Notes |
| --- | --- | --- |
| UI orchestration | `src/main.ts`, `index.html` | Owns DOM wiring, rendering, settings, generation lifecycle, thread actions, modals, and user interactions. |
| Model runtime | `src/model.ts` | Wraps Transformers.js model loading, WebGPU validation, dtype fallback, prompt formatting, streaming, cancellation, and context trimming. |
| Model registry | `src/models.ts` | Defines presets, aliases, supported dtypes, memory estimates, context windows, external-data chunk hints, and Gemma4 loader selection. |
| Persistence | `src/storage.ts` | Validates and migrates thread/settings shapes before reading/writing `localStorage`. |
| GPU detection | `src/gpu.ts` | Reads WebGPU adapter info, `shader-f16`, fallback adapter status, and max buffer limits. |
| Web search | `src/webSearch.ts` | Aggregates SearxNG, DuckDuckGo, Wikipedia, and current-time facts into a prompt block. |
| MCP browser client | `src/mcp.ts` | Lists tools, calls tools, formats tool-selection/tool-result prompts, and parses model-produced tool-call JSON. |
| MCP bridge | `scripts/mcp-bridge.mjs` | Local HTTP server for `/health`, `/tools`, and `/tools/call`; connects to stdio, Streamable HTTP, or SSE servers. |
| Model cache UI | `src/modelCache.ts` | Groups Hugging Face cache entries by model ID and deletes selected/all downloaded models. |
| Speech | `src/tts.ts`, `src/stt.ts` | Kokoro playback and Moonshine microphone transcription. |
| Compatibility tests | `src/tester.ts`, `tests/model-compatibility.spec.ts` | Browser-driven model load/inference tests with saved results. |

## Chat Flow

1. `ChatApp.sendMessage()` ensures a model is loaded.
2. The user message is appended to the current thread and saved.
3. If web search is enabled, `webSearch()` gathers context and `formatWebSearchContext()` replaces the latest user message content in the generation history.
4. If MCP server URLs are configured, `listMcpTools()` fetches available tools from the bridge.
5. MCP-enabled turns first call `generateWithMcp()`:
   - The model receives `formatMcpToolSelectionPrompt()`.
   - If the model returns parseable tool-call JSON, `callMcpTool()` executes one tool call through the bridge.
   - The model receives `formatMcpToolResultPrompt()` and produces the final answer.
   - If no tool call is parsed, the app falls back to normal generation.
6. Normal generation calls `ModelManager.generate()` with the thread history and current generation settings.
7. Tokens stream into the assistant bubble, then the final cleaned text is rendered as Markdown with syntax-highlighted code.
8. Threads are saved to `localStorage`; optional TTS can play the assistant message.

The app does not use retrieval storage, vector indexes, background workers, or server-side chat state.

## Model Loading

`ModelManager.load()` checks WebGPU, resolves `auto` dtype, unloads incompatible existing sessions, and then loads either:

- `pipeline('text-generation', modelId, { device: 'webgpu', dtype })` for standard models.
- `AutoProcessor` plus `Gemma4ForCausalLM` for Gemma4 text-only models.

Transformers.js is configured with:

```ts
env.allowLocalModels = false;
env.useBrowserCache = true;
```

This means model assets are fetched from remote public model repositories and cached by the browser. Local filesystem model loading is not enabled.

For non-chat-template models, prompts use a simple transcript format:

```text
User: ...
Assistant: ...
Assistant:
```

For models with tokenizer chat templates, the model pipeline receives chat message objects directly. Gemma4 uses its processor's chat template with thinking disabled.

## Context Handling

The app estimates text tokens at roughly four characters per token. Before generation, `ModelManager.trimMessagesToBudget()` walks backward through the conversation and keeps the newest turns that fit the configured prompt-context token budget. If only one oversized message can fit, the middle is omitted and the head/tail are preserved.

Manual compaction is separate. `compactCurrentThread()` summarizes older messages with the loaded local model, stores the summary as an assistant message, and keeps the newer tail unchanged.

## Tools And Abilities

### Browser Web Search

`webSearch()` can return:

- Direct facts from SearxNG answers/infoboxes.
- Result snippets from SearxNG, DuckDuckGo Instant Answer, and Wikipedia.
- Computed current-time facts for simple time-zone questions.

The app does not crawl pages or run a search index. Result quality depends on the configured providers and CORS access.

### MCP

The browser client sends dynamic server URLs in query/body parameters. The bridge names those dynamic servers `mcp_1`, `mcp_2`, and so on. Bridge-side configured servers keep their explicit names.

Bridge endpoints:

- `GET /health?url=...` returns bridge status and known server names.
- `GET /tools?url=...` returns all available tools plus per-server errors.
- `POST /tools/call` expects `server`, `name`, optional `arguments`, and optional `serverUrls`.

The browser prompt contract asks the model to return JSON in this shape:

```json
{
  "mcp_tool_call": {
    "server": "exact-server",
    "name": "exact-tool",
    "arguments": {}
  }
}
```

`parseMcpToolCall()` also tolerates fenced JSON and some qualified-name variants, but the bridge call uses only a normalized exact tool match.

### Speech

Kokoro TTS streams text into sentence jobs, synthesizes up to three jobs concurrently, and plays resulting audio buffers in order. Moonshine STT records with `MediaRecorder`, decodes/resamples with Web Audio, and transcribes the final clip after recording stops.

## State And Persistence

`src/storage.ts` validates stored data rather than trusting raw JSON. It also normalizes legacy model IDs, dtypes, generation settings, SearxNG URLs, MCP bridge URLs, and TTS settings.

Settings are saved globally plus per model:

- Global `modelId`, `dtype`, and generation values represent the active selection.
- `modelSettings[modelId]` remembers dtype and generation values per model.

## Compatibility Testing

There are two compatibility paths:

- In-app Settings tests call `runCompatibilityTests()` from `src/tester.ts`.
- Playwright tests in `tests/model-compatibility.spec.ts` launch the app, load models in the browser, run a short prompt, collect console errors, and write JSON results under `test-results/`.

Both paths are intentionally expensive on first run because they download model files and initialize WebGPU sessions.

## Extension Points

To add a model preset:

1. Add a `ModelInfo` entry in `src/models.ts`.
2. Include supported dtypes, approximate weights, context window, defaults, and any external-data chunk counts.
3. Add aliases in `normalizeModelId()` if helpful.
4. Run at least `npm run typecheck` and a filtered model test such as `MODEL_TEST_MODELS=<name> npm run test:models`.

To add a browser-side tool:

1. Keep provider/network code outside `src/main.ts`.
2. Return structured context.
3. Format an explicit model prompt block.
4. Wire the orchestration in `sendMessage()` without changing persisted thread content except for final user/assistant messages.

To change MCP behavior:

1. Update `src/mcp.ts` for browser prompt/parsing/client behavior.
2. Update `scripts/mcp-bridge.mjs` for transport, server, or endpoint behavior.
3. Preserve CORS and Private Network Access headers if deployed pages need to call the local bridge.

## Security And Privacy Boundaries

- Chat inference runs locally, but model files are downloaded from Hugging Face.
- Web search sends the query to configured search providers while enabled.
- MCP calls can do anything the exposed MCP server allows.
- Browser code cannot safely use private Hugging Face tokens; gated/private models are unsupported.
- Site data is local to the browser profile and origin.
