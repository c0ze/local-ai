import {
  AutoProcessor,
  Gemma4ForCausalLM,
  InterruptableStoppingCriteria,
  pipeline,
  env,
  TextStreamer,
} from '@huggingface/transformers';
import { externalDataChunksFor, modelLoaderFor } from './models';

env.allowLocalModels = false;
env.useBrowserCache = true;

export type ProgressData = {
  status: 'init' | 'download' | 'loading' | 'ready' | 'error';
  progress?: number;
  file?: string;
  message?: string;
};

export type ProgressCallback = (data: ProgressData) => void;
export type TokenCallback = (token: string) => void;
type ChatMessage = { role: string; content: string };

export type GenerateOptions = {
  temperature?: number;
  topP?: number;
  maxNewTokens?: number;
  repetitionPenalty?: number;
  contextTokens?: number;
};

export type Dtype = 'auto' | 'q4f16' | 'q4' | 'q8' | 'fp16' | 'fp32';

export class ModelManager {
  private generator: any = null;
  private processor: any = null;
  private isLoading = false;
  private modelName = '';
  private loadedDtype: Exclude<Dtype, 'auto'> | null = null;
  private loader: 'text-generation' | 'gemma4-text' = 'text-generation';

  get loading() {
    return this.isLoading;
  }
  get loaded() {
    return this.generator !== null;
  }
  get name() {
    return this.modelName;
  }
  get dtype() {
    return this.loadedDtype;
  }

  async load(modelId: string, onProgress: ProgressCallback, dtype: Dtype = 'auto'): Promise<void> {
    if (this.isLoading) return;
    const resolvedDtype = dtype === 'auto' ? await this.pickDtype() : dtype;
    if (this.loaded && this.modelName === modelId && this.loadedDtype === resolvedDtype) return;
    if (this.loaded) {
      await this.unload();
    }

    this.isLoading = true;
    this.modelName = modelId;

    try {
      onProgress({ status: 'init', progress: 0 });

      const hasWebGPU = await this.checkWebGPU();
      if (!hasWebGPU) {
        throw new Error(
          'WebGPU is unavailable: this browser could not get a GPU adapter. Use a WebGPU-capable browser and hardware, then enable the required browser or GPU flags.'
        );
      }

      this.loader = modelLoaderFor(this.modelName) ?? 'text-generation';
      if (this.loader === 'gemma4-text') {
        await this.loadGemma4Text(resolvedDtype, onProgress);
      } else {
        const externalDataChunks = externalDataChunksFor(this.modelName, resolvedDtype);
        this.generator = await pipeline('text-generation', this.modelName, {
          device: 'webgpu',
          dtype: resolvedDtype,
          ...(externalDataChunks ? { use_external_data_format: externalDataChunks } : {}),
          progress_callback: (data: any) => this.handleProgress(data, onProgress),
        });
      }

      this.loadedDtype = resolvedDtype;
      onProgress({ status: 'ready', progress: 100 });
    } catch (error: any) {
      this.loadedDtype = null;
      const msg = error?.message || String(error);

      if (msg.includes('401') || msg.includes('Unauthorized')) {
        onProgress({ status: 'error', message: `Unauthorized: "${this.modelName}" is gated, private, or not publicly readable in the browser. Try a public Transformers.js ONNX model.` });
      } else if (msg.includes('404')) {
        onProgress({ status: 'error', message: `Model not found: "${this.modelName}" may not be available in ONNX format for Transformers.js.` });
      } else if (msg.includes('MountedFiles') || msg.includes('external data file')) {
        onProgress({ status: 'error', message: `Failed to mount ONNX external data for "${this.modelName}" (${resolvedDtype}). Clear this site’s browser cache and reload, then try again.` });
      } else if (msg.includes('shader-f16') || msg.includes('BindGroupLayout') || msg.includes('ComputePipeline')) {
        onProgress({ status: 'error', message: `WebGPU rejected the shader pipeline. Your GPU/driver likely doesn't support shader-f16 — switch dtype to "q4" (or "q8") in Settings.` });
      } else if (typeof error === 'number' || /^\d+$/.test(msg)) {
        onProgress({ status: 'error', message: `WebGPU failed while loading "${this.modelName}" with runtime code ${msg}. This usually means the model is too large for the available GPU/browser memory. Try a smaller WebGPU model.` });
      } else {
        onProgress({ status: 'error', message: `Failed to load "${this.modelName}": ${msg}` });
      }
      console.error('Model loading error:', error);
      throw error;
    } finally {
      this.isLoading = false;
    }
  }

  async unload(): Promise<void> {
    if (this.generator?.dispose) {
      await this.generator.dispose();
    }

    this.generator = null;
    this.processor = null;
    this.modelName = '';
    this.loadedDtype = null;
    this.loader = 'text-generation';
  }

  private handleProgress(data: any, onProgress: ProgressCallback) {
    if (data.status === 'progress' || data.status === 'progress_total') {
      onProgress({
        status: 'download',
        progress: Math.round(data.progress ?? 0),
        file: data.file,
      });
    } else if (data.status === 'done') {
      onProgress({ status: 'loading', progress: 100, file: data.file });
    } else {
      onProgress({
        status: data.status === 'initiate' ? 'download' : data.status,
        file: data.file,
      });
    }
  }

  private async loadGemma4Text(dtype: Exclude<Dtype, 'auto'>, onProgress: ProgressCallback) {
    onProgress({ status: 'loading', message: 'Loading Gemma4 processor...' });
    this.processor = await AutoProcessor.from_pretrained(this.modelName, {
      progress_callback: (data: any) => this.handleProgress(data, onProgress),
    });

    onProgress({ status: 'loading', message: 'Loading Gemma4 text sessions...' });
    this.generator = await Gemma4ForCausalLM.from_pretrained(this.modelName, {
      device: 'webgpu',
      dtype,
      progress_callback: (data: any) => this.handleProgress(data, onProgress),
    });
  }

  formatPrompt(messages: ChatMessage[]): string {
    let prompt = '';
    for (const msg of messages) {
      if (msg.role === 'user') {
        prompt += `User: ${msg.content}\n`;
      } else if (msg.role === 'assistant') {
        prompt += `Assistant: ${msg.content}\n`;
      }
    }
    prompt += 'Assistant:';
    return prompt;
  }

  private formatGemma4Messages(messages: ChatMessage[]) {
    return messages.map((message) => ({
      role: message.role,
      content: message.content,
    }));
  }

  private estimateTextTokens(text: string): number {
    return Math.max(1, Math.ceil(text.length / 4));
  }

  private estimateMessageTokens(message: ChatMessage): number {
    return 6 + this.estimateTextTokens(message.content);
  }

  private truncateContentToTokenBudget(content: string, budgetTokens: number): string {
    const maxChars = Math.max(256, budgetTokens * 4);
    if (content.length <= maxChars) return content;

    const marker = '\n\n[...middle omitted to fit context budget...]\n\n';
    const sideChars = Math.max(96, Math.floor((maxChars - marker.length) / 2));
    return `${content.slice(0, sideChars)}${marker}${content.slice(-sideChars)}`;
  }

  private trimMessagesToBudget(messages: ChatMessage[], budgetTokens: number): ChatMessage[] {
    if (messages.length === 0) return [];

    const budget = Math.max(64, Math.floor(budgetTokens));
    const kept: ChatMessage[] = [];
    let used = 8;

    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i];
      const cost = this.estimateMessageTokens(message);

      if (used + cost > budget && kept.length > 0) break;

      if (used + cost > budget) {
        const remaining = Math.max(32, budget - used - 6);
        kept.unshift({
          ...message,
          content: this.truncateContentToTokenBudget(message.content, remaining),
        });
        break;
      }

      kept.unshift(message);
      used += cost;
    }

    return kept;
  }

  async generate(
    messages: ChatMessage[],
    onToken: TokenCallback,
    abortSignal?: AbortSignal,
    options: GenerateOptions = {}
  ): Promise<string> {
    if (!this.generator) throw new Error('Model not loaded');

    if (this.loader === 'gemma4-text') {
      return this.generateGemma4(messages, onToken, abortSignal, options);
    }

    const supportsChatTemplate = Boolean(this.generator.tokenizer?.chat_template);
    const contextTokens = options.contextTokens ?? 1536;
    const contextMessages = this.trimMessagesToBudget(messages, contextTokens);
    const input = supportsChatTemplate ? contextMessages : this.formatPrompt(contextMessages);

    let buffer = '';
    const stoppingCriteria = new InterruptableStoppingCriteria();
    const abort = () => stoppingCriteria.interrupt();
    if (abortSignal?.aborted) abort();
    abortSignal?.addEventListener('abort', abort, { once: true });

    const streamer = new TextStreamer(this.generator.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (text: string) => {
        if (abortSignal?.aborted) return;
        buffer += text;
        onToken(text);
      },
    });

    try {
      await this.generator(input, {
        max_new_tokens: options.maxNewTokens ?? 512,
        do_sample: true,
        temperature: options.temperature ?? 0.7,
        top_p: options.topP ?? 0.95,
        repetition_penalty: options.repetitionPenalty ?? 1.1,
        return_full_text: false,
        streamer,
        stopping_criteria: stoppingCriteria,
      });
    } finally {
      abortSignal?.removeEventListener('abort', abort);
    }

    return buffer
      .replace(/<end_of_turn>.*$/s, '')
      .replace(/<\|im_end\|>.*$/s, '')
      .replace(/<turn\|>.*$/s, '')
      .trim();
  }

  private async generateGemma4(
    messages: ChatMessage[],
    onToken: TokenCallback,
    abortSignal?: AbortSignal,
    options: GenerateOptions = {},
  ): Promise<string> {
    if (!this.processor) throw new Error('Gemma4 processor not loaded');

    const contextTokens = options.contextTokens ?? 512;
    const contextMessages = this.trimMessagesToBudget(messages, contextTokens);
    const prompt = this.processor.apply_chat_template(this.formatGemma4Messages(contextMessages), {
      add_generation_prompt: true,
      enable_thinking: false,
    });
    const inputs = await this.processor(prompt, null, null, { add_special_tokens: false });

    let buffer = '';
    const stoppingCriteria = new InterruptableStoppingCriteria();
    const abort = () => stoppingCriteria.interrupt();
    if (abortSignal?.aborted) abort();
    abortSignal?.addEventListener('abort', abort, { once: true });

    const streamer = new TextStreamer(this.processor.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (text: string) => {
        if (abortSignal?.aborted) return;
        buffer += text;
        onToken(text);
      },
    });

    try {
      const outputs = await this.generator.generate({
        ...inputs,
        max_new_tokens: options.maxNewTokens ?? 64,
        do_sample: true,
        temperature: options.temperature ?? 0.7,
        top_p: options.topP ?? 0.95,
        repetition_penalty: options.repetitionPenalty ?? 1.1,
        streamer,
        stopping_criteria: stoppingCriteria,
      });

      if (!buffer) {
        const inputLength = inputs.input_ids?.dims?.at(-1) ?? 0;
        const generated = typeof outputs?.slice === 'function'
          ? outputs.slice(null, [inputLength, null])
          : outputs;
        buffer = this.processor.batch_decode(generated, { skip_special_tokens: true })?.[0] ?? '';
      }
    } finally {
      abortSignal?.removeEventListener('abort', abort);
    }

    return buffer
      .replace(/<\|channel>thought[\s\S]*?<channel\|>/g, '')
      .replace(/<turn\|>.*$/s, '')
      .replace(/<\|endoftext\|>/g, '')
      .trim();
  }

  async checkWebGPU(): Promise<boolean> {
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
      return false;
    }

    try {
      const gpu = (navigator as Navigator & {
        gpu?: { requestAdapter: (opts?: { powerPreference?: 'high-performance' | 'low-power' }) => Promise<unknown | null> };
      }).gpu;

      if (!gpu) return false;

      const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
      return adapter !== null;
    } catch {
      return false;
    }
  }

  private async pickDtype(): Promise<Exclude<Dtype, 'auto'>> {
    try {
      const gpu = (navigator as Navigator & {
        gpu?: {
          requestAdapter: (opts?: { powerPreference?: 'high-performance' | 'low-power' }) => Promise<{ features?: ReadonlySet<string> } | null>;
        };
      }).gpu;
      const adapter = await gpu?.requestAdapter({ powerPreference: 'high-performance' });
      if (adapter?.features?.has('shader-f16')) return 'q4f16';
    } catch {
      /* fall through */
    }
    return 'q4';
  }
}
