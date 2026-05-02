# Local AI Chat

A privacy-first, browser-based chat interface that runs AI models locally using WebGPU. No server required; model inference happens in your browser.

## Features

- **Fully Local Inference**: AI models run directly in your browser via WebGPU. Chats stay in localStorage.
- **Optional Web Search Tool**: Settings can enable DuckDuckGo/Wikipedia/SearxNG context for every answer.
- **Optional MCP Tool Bridge**: Settings can register MCP server URLs that are used through a small local bridge.
- **Local Text-to-Speech**: Kokoro voice synthesis runs in the browser and can play assistant responses on demand.
- **Local Speech-to-Text**: The composer microphone button records a short dictation and transcribes it locally with Moonshine.
- **OpenWebUI-inspired Design**: Clean chat interface with multiple conversation threads.
- **Persistent Chats**: Conversations are saved to your browser's localStorage.
- **Chat Management**: Rename or favorite threads, export a chat as Markdown, copy messages, and compact old context.
- **Markdown Support**: Assistant responses render markdown with syntax-highlighted code blocks.
- **Model Registry**: Curated public ONNX presets with GPU-aware dtype selection.
- **Downloaded Model Manager**: View cached model files and delete them from the browser cache.
- **Compatibility Tests**: Settings can load each preset and record which dtype works on your GPU.
- **One-Click Deploy**: GitHub Actions workflow included for automatic GitHub Pages deployment.

## Browser Compatibility

| Feature | Chrome | Firefox | Safari | Edge |
|---------|--------|---------|--------|------|
| WebGPU | 113+ | Nightly* | 17.4+ | 113+ |
| Required | ✅ | ⚠️ | ✅ | ✅ |

\* Firefox WebGPU support is behind a flag in Nightly builds.

**Minimum**: Chrome/Edge 113+, or Safari 17.4+ on macOS Sonoma.

## Browser Setup

Most systems work without changing browser settings. If Settings shows `google - swiftshader`, `Software fallback adapter: yes`, or a 1 GB max buffer on Linux / AMD, Chrome is using its software WebGPU adapter instead of the real GPU.

### Chrome on Linux / AMD

1. Open `chrome://flags`.
2. Search for `Unsafe WebGPU` and set it to **Enabled**.
3. Search for `Vulkan` and set it to **Enabled**.
4. Relaunch Chrome.
5. Open Local AI Settings. Hardware should show `amd`, `shader-f16 ✓`, and a max buffer above 1 GB.

If YouTube or other video sites show a white video player after enabling Vulkan, try these in order:

1. Open `chrome://flags`.
2. Search for `Choose ANGLE graphics backend` and set it to **OpenGL**.
3. Relaunch Chrome and test Local AI plus YouTube.
4. If video is still white, search for `Zero-copy rasterizer` and set it to **Disabled**.
5. Relaunch Chrome and test again.
6. If video is still white, search for `Hardware-accelerated video decode` and set only that flag to **Disabled**.
7. Relaunch Chrome.

Do not disable Chrome Settings → System → **Use graphics acceleration when available**. Turning that off disables the GPU path Local AI needs.

## Development

### Prerequisites

- [Node.js 24.14.0](https://nodejs.org/) (recommended via [mise](https://mise.jdx.dev/))
- A WebGPU-capable browser

### Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

`npm run dev` starts both the Vite app and the local MCP bridge. The bridge uses `mcp.config.json` when present, otherwise it falls back to `mcp.config.example.json`.

There are no required application environment variables for local development. `VITE_BASE_URL` is optional and only needed when serving the built site from a subpath.

### Build

```bash
npm run typecheck
npm run typecheck:tests
npm run build
```

The static site will be output to the `dist/` directory.

### Verification Commands

| Command | Purpose |
|---------|---------|
| `npm run typecheck` | Type-check the Vite application code. |
| `npm run typecheck:tests` | Type-check the Playwright model compatibility suite. |
| `npm run build` | Build the production static site. |
| `npm run mcp:bridge` | Start the local MCP bridge with `mcp.config.json`. |
| `npm audit --audit-level=moderate` | Check installed packages for moderate or higher vulnerabilities. |

### Model Integration Tests

The browser integration suite downloads each configured model, loads it with WebGPU, asks a short prompt, records the answer and timings, and fails on severe console/runtime errors. Results are written to `test-results/model-compatibility-*.json` with suggested dtype and generation defaults.

```bash
# First time only
npm run test:models:install

# Test each model with its recommended dtype
npm run test:models

# Try every viable dtype for each model, then compare results
npm run test:models:all
```

Useful filters:

```bash
MODEL_TEST_MODELS=qwen npm run test:models
MODEL_TEST_DTYPES=q4 npm run test:models
PW_HEADLESS=1 npm run test:models
```

Other optional test variables include `PORT`, `PLAYWRIGHT_BASE_URL`, `MODEL_TEST_LOAD_TIMEOUT_MS`, `MODEL_TEST_INFER_TIMEOUT_MS`, and `MODEL_TEST_RESULTS_DIR`.

## Deployment

### GitHub Pages (Recommended)

1. Push this repository to GitHub.
2. Go to **Settings → Pages**.
3. Set **Source** to "GitHub Actions".
4. Push to the `main` branch — the workflow in `.github/workflows/deploy.yml` will build and deploy automatically.

### Manual Deploy

Build the project and upload the `dist/` folder to any static hosting provider:

```bash
npm run build
```

## Architecture

| Layer | Technology |
|-------|-----------|
| Framework | Vanilla TypeScript + Vite |
| Styling | Tailwind CSS |
| AI Inference | Transformers.js (WebGPU) |
| Tool Bridge | MCP TypeScript SDK + local Node sidecar |
| Markdown | marked + highlight.js |
| Storage | localStorage |

## Model Notes

The app loads AI models directly from Hugging Face into your browser's cache. On first use, the selected model is downloaded. Subsequent loads reuse the browser cache when available.

### Chat and Model Cache Management

Chats are stored in `localStorage` and model files are stored in the browser Cache API used by Transformers.js. Use **Models** in the chat header to inspect downloaded Hugging Face model files, load a cached model, delete a single model, or clear all downloaded model files. Clearing model files does not delete chat history.

Use **Export** to download the current thread as Markdown. Thread favorites are stored with the thread data and only affect sidebar ordering. Rename a thread from the sidebar with the pencil icon (or double-click its title); Enter saves, Escape cancels.

### Web Search Tool

Settings -> **Tools** can enable a small browser-side search tool. When enabled, every prompt runs through a configured SearxNG instance if one is set, then falls back to DuckDuckGo's Instant Answer API and Wikipedia. Direct facts and up to five source snippets are injected into the generation prompt, and the local model is asked to cite used results with bracketed source numbers.

The DuckDuckGo/Wikipedia fallback is intentionally basic: it does not crawl pages, does not run a full search index, and may return sparse results for niche queries or anything news-like. Point Settings → **Web search (SearxNG)** at your own SearxNG instance for real-web results.

SearxNG requirements, since the request is cross-origin from the browser:

- JSON format enabled in `settings.yml`:
  ```yaml
  search:
    formats:
      - html
      - json
  ```
- `Access-Control-Allow-Origin` on SearxNG responses that permits the app origin (either the exact origin or `*`). Add the header at your reverse proxy if SearxNG sits behind one.

Model inference still runs locally, but search queries leave the browser to whichever providers are active while the tool is enabled.

### MCP Tools

Settings -> **Tools** lets you add MCP server URLs. If any URLs are saved, MCP tools are available on every prompt. Because the app is a static browser app, it cannot spawn stdio MCP servers directly. Instead, it talks to a local HTTP bridge, and the bridge connects to configured HTTP/SSE MCP servers.

This also works from the deployed static site: run the bridge on the same computer as your browser, point Settings -> **MCP bridge** at `http://localhost:8787`, and add one MCP server URL per line under **MCP server URLs**. The bridge sends CORS and Private Network Access headers so an HTTPS-hosted page can call the local companion service.

Start the bridge:

```bash
npm run mcp:bridge
```

Then open Settings and confirm **MCP bridge** is `http://localhost:8787`. Add MCP server URLs such as `http://localhost:3000/mcp`. The app asks the model whether one MCP tool call is needed, calls the tool when the model returns the required JSON, then asks the model to answer with the tool result.

To connect your own servers, create `mcp.config.json`:

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

You can still add always-on bridge-side servers in `mcp.config.json`, which is useful for stdio MCP servers. Only configure MCP servers you trust; tool calls can read, write, or send data depending on the server you expose.

### Text-to-Speech

The speaker button in the composer toggles local Kokoro TTS. When it is enabled, speech starts after the assistant response finishes so model generation stays responsive. Each assistant response also shows a **Play** action next to **Copy** so previous responses can be spoken independently. Long responses are synthesized with up to three sentence jobs in flight, played in order, and the active response button changes to **Pause** / **Resume** during playback.

Settings -> **Text to speech** lets you choose from the Kokoro-provided American and British English voices and adjust speech speed. Kokoro runs entirely in the browser using its Transformers.js stack. The app loads it lazily the first time you press **Play** and caches model/voice files in browser storage where supported.

### Speech-to-Text

The microphone button in the composer starts local dictation. Click it once to record, click again to stop, and the Moonshine transcript is inserted into the message box without sending automatically. Audio capture, resampling, and transcription all run in the browser; no backend or speech API is used.

### Context and Speed

Every message sends a prompt plus a requested number of output tokens to the model. In Settings, **Prompt context tokens** is a snapped slider with common stops like 512, 1024, 2048, 4096, and larger multiples; it controls how much recent chat history is sent. **Max new tokens** is also a snapped slider for the response cap. Lower values are faster and reduce WebGPU memory pressure. The app trims old messages first when the prompt would exceed the configured context budget.

Generation settings and dtype are remembered per model after saving. For example, if you tune Gemma4, switch to Qwen, and later select Gemma4 again, the Settings modal restores the last saved Gemma4 values.

Settings also shows a context bar for the selected model: prompt context plus max output tokens against that model's context window.

Use **Compact** in the chat header to manually summarize older turns. Compaction keeps the recent tail of the conversation, asks the loaded local model to summarize the older span, and replaces that old span with one compact summary message. This reduces future prompt size without deleting the newest working context.

### Important Limitations

**Gated / Private Models:** Transformers.js **cannot load gated or private models in the browser** for security reasons (HF tokens cannot be sent from client-side code). If you see a **401 Unauthorized** error, the model is gated/private and will not work. You must use a **publicly available ONNX model**.

**WebGPU Required:** The app is intentionally WebGPU-only. If your browser cannot get a GPU adapter, model loading will fail.

**Browser Memory Reporting:** WebGPU does not expose total or free VRAM to JavaScript. The Settings memory bar uses model weight estimates plus the detected WebGPU buffer limit, which is the hard browser-side limit behind most oversized-model failures.

**SwiftShader / 1 GB Limit:** If Settings shows `google - swiftshader` or "Software fallback adapter: yes", Chrome is not using the real GPU. SwiftShader is a software WebGPU adapter and usually exposes a 1 GB buffer limit. On Linux / AMD, enable `Unsafe WebGPU` and `Vulkan` in `chrome://flags`, then relaunch Chrome.

### Recommended Models

| Model | Size | Context / Output Defaults | Notes |
|-------|------|---------------------------|-------|
| `onnx-community/Qwen2.5-0.5B-Instruct` | ~0.5B params | 1536 / 192 | Default, smaller chat model |
| `onnx-community/SmolLM2-360M-Instruct-ONNX` | ~360M params | 1280 / 160 | Smaller chat model |
| `onnx-community/Llama-3.2-1B-Instruct` | ~1B params | 1024 / 128 | Larger chat model |
| `onnx-community/deepseek-coder-1.3b-instruct-ONNX` | ~1.3B params | 1280 / 160 | Small DeepSeek coding model |
| `onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX` | ~1.5B params | 1536 / 192 | Reasoning-tuned DeepSeek distill; no public 4B WebGPU ONNX preset found |
| `onnx-community/Qwen2.5-Coder-3B-Instruct` | ~3.1B params | 1024 / 128 | Larger Qwen coding model; multi-GB download |
| `onnx-community/Qwen3-4B-ONNX` | ~4B params | 1024 / 128 | 4B Qwen model; q4f16 only, needs shader-f16 and ~2 GB WebGPU buffers |
| `microsoft/Phi-3-mini-4k-instruct-onnx-web` | ~3.8B params | 768 / 96 | Much larger; needs more GPU memory |
| `onnx-community/gemma-4-E4B-it-ONNX` | ~4.5B effective params | 512 / 64 | Experimental text-only Gemma4 path; q4f16 is ~4.6 GB of ONNX external data |
| `Xenova/distilgpt2` | ~82M params | 512 / 64 | Tiny, fast sanity-check model; not instruction-tuned |

Gemma 4 E4B uses the Gemma4 processor/model classes rather than the generic `text-generation` pipeline. The app loads it in text-only mode, so image/audio inputs are not wired into the chat UI yet. It also needs large single WebGPU buffers: q4f16 needs about 1.31 GB and q4 needs about 1.95 GB, so browsers/adapters capped at 1.00 GB cannot run it even when total VRAM is higher. If you see old `model_q4.onnx_data` or `table index is out of bounds` errors, clear the browser cache and reload after updating dependencies; those are typical of the old generic pipeline trying to load Gemma4.

Kimi is not included yet because the public Kimi A3B release is a vision-language/custom-code model, and there is not currently a public Transformers.js text-generation ONNX preset for this app to load directly.

### Requirements

- A WebGPU-capable browser with an available GPU adapter (Chrome 113+, Edge 113+, Safari 17.4+, or Firefox Nightly with the WebGPU flag)
- The model must be publicly available in **ONNX format** on Hugging Face
- ~4GB RAM recommended for 1B+ parameter models

## License

MIT
