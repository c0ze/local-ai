import { normalizeModelId } from './models';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

export interface ChatThread {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

const STORAGE_KEY = 'local-ai-chat-threads';

export function generateId(): string {
  return crypto.randomUUID();
}

function isMessage(value: unknown): value is Message {
  if (!value || typeof value !== 'object') return false;
  const message = value as Message;
  return (
    typeof message.id === 'string' &&
    (message.role === 'user' || message.role === 'assistant') &&
    typeof message.content === 'string' &&
    typeof message.timestamp === 'number'
  );
}

function isThread(value: unknown): value is ChatThread {
  if (!value || typeof value !== 'object') return false;
  const thread = value as ChatThread;
  return (
    typeof thread.id === 'string' &&
    typeof thread.title === 'string' &&
    Array.isArray(thread.messages) &&
    thread.messages.every(isMessage) &&
    typeof thread.createdAt === 'number' &&
    typeof thread.updatedAt === 'number'
  );
}

export function loadThreads(): ChatThread[] {
  try {
    const data = localStorage.getItem(STORAGE_KEY);
    if (!data) return [];
    const threads = JSON.parse(data) as unknown;
    return Array.isArray(threads) ? threads.filter(isThread) : [];
  } catch {
    return [];
  }
}

export function saveThreads(threads: ChatThread[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(threads));
  } catch (e) {
    console.error('Failed to save threads:', e);
  }
}

export function createThread(): ChatThread {
  return {
    id: generateId(),
    title: 'New Chat',
    messages: [],
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

export function deleteThread(threads: ChatThread[], threadId: string): ChatThread[] {
  return threads.filter((t) => t.id !== threadId);
}

export function updateThreadTitle(thread: ChatThread): void {
  if (thread.title !== 'New Chat') return;
  const firstUserMsg = thread.messages.find(m => m.role === 'user');
  if (firstUserMsg) {
    const clean = firstUserMsg.content.trim().replace(/\s+/g, ' ');
    thread.title = clean.substring(0, 30) + (clean.length > 30 ? '...' : '');
  }
}

const SETTINGS_KEY = 'local-ai-chat-settings';
const LEGACY_MODEL_IDS: Record<string, string> = {
  'HauhauCS/Gemma-4-E4B-Uncensored': 'onnx-community/Qwen2.5-0.5B-Instruct',
  'onnx-community/Phi-3-mini-4k-instruct-onnx-web': 'microsoft/Phi-3-mini-4k-instruct-onnx-web',
  'onnx-community/gemma-4-E2B-it-ONNX': 'onnx-community/Qwen2.5-0.5B-Instruct',
};

export interface GenerationParams {
  temperature: number;
  topP: number;
  maxNewTokens: number;
  repetitionPenalty: number;
  contextTokens: number;
}

export type Dtype = 'auto' | 'q4f16' | 'q4' | 'q8' | 'fp16' | 'fp32';

export interface ModelSpecificSettings {
  dtype: Dtype;
  generation: GenerationParams;
}

export interface AppSettings {
  modelId: string;
  hfToken: string;
  generation: GenerationParams;
  dtype: Dtype;
  modelSettings: Record<string, ModelSpecificSettings>;
}

export const DEFAULT_GENERATION: GenerationParams = {
  temperature: 0.7,
  topP: 0.95,
  maxNewTokens: 192,
  repetitionPenalty: 1.1,
  contextTokens: 1536,
};

const VALID_DTYPES: Dtype[] = ['auto', 'q4f16', 'q4', 'q8', 'fp16', 'fp32'];

export const DEFAULT_SETTINGS: AppSettings = {
  modelId: 'onnx-community/Qwen2.5-0.5B-Instruct',
  hfToken: '',
  generation: { ...DEFAULT_GENERATION },
  dtype: 'auto',
  modelSettings: {},
};

function defaultSettings(): AppSettings {
  return {
    ...DEFAULT_SETTINGS,
    generation: { ...DEFAULT_GENERATION },
    modelSettings: { ...DEFAULT_SETTINGS.modelSettings },
  };
}

function clamp(n: number, min: number, max: number, fallback: number): number {
  if (typeof n !== 'number' || !Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

function normalizeDtype(dtype: unknown): Dtype {
  return VALID_DTYPES.includes(dtype as Dtype) ? dtype as Dtype : DEFAULT_SETTINGS.dtype;
}

function normalizeGeneration(generation: Partial<GenerationParams> | undefined): GenerationParams {
  const gen = generation ?? DEFAULT_GENERATION;
  return {
    temperature: clamp(gen.temperature ?? DEFAULT_GENERATION.temperature, 0, 2, DEFAULT_GENERATION.temperature),
    topP: clamp(gen.topP ?? DEFAULT_GENERATION.topP, 0, 1, DEFAULT_GENERATION.topP),
    maxNewTokens: Math.round(clamp(gen.maxNewTokens ?? DEFAULT_GENERATION.maxNewTokens, 16, 1024, DEFAULT_GENERATION.maxNewTokens)),
    repetitionPenalty: clamp(gen.repetitionPenalty ?? DEFAULT_GENERATION.repetitionPenalty, 1, 2, DEFAULT_GENERATION.repetitionPenalty),
    contextTokens: Math.round(clamp(gen.contextTokens ?? DEFAULT_GENERATION.contextTokens, 256, 131072, DEFAULT_GENERATION.contextTokens)),
  };
}

function normalizeModelSettings(value: unknown): Record<string, ModelSpecificSettings> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {};

  const normalized: Record<string, ModelSpecificSettings> = {};
  for (const [rawModelId, rawSettings] of Object.entries(value)) {
    if (!rawModelId || !rawSettings || typeof rawSettings !== 'object') continue;
    const settings = rawSettings as Partial<ModelSpecificSettings>;
    normalized[normalizeModelId(rawModelId)] = {
      dtype: normalizeDtype(settings.dtype),
      generation: normalizeGeneration(settings.generation),
    };
  }
  return normalized;
}

function normalizeSettings(settings: AppSettings): AppSettings {
  const modelId = normalizeModelId(LEGACY_MODEL_IDS[settings.modelId] || settings.modelId || DEFAULT_SETTINGS.modelId);
  const modelSettings = normalizeModelSettings(settings.modelSettings);
  const savedForModel = modelSettings[modelId];
  const dtype = normalizeDtype(settings.dtype ?? savedForModel?.dtype);

  return {
    ...settings,
    modelId,
    dtype,
    generation: normalizeGeneration(settings.generation ?? savedForModel?.generation),
    modelSettings,
  };
}

export function loadSettings(): AppSettings {
  try {
    const data = localStorage.getItem(SETTINGS_KEY);
    if (!data) return defaultSettings();
    const parsed = { ...defaultSettings(), ...JSON.parse(data) };
    const normalized = normalizeSettings(parsed);
    if (normalized.modelId !== parsed.modelId) {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(normalized));
    }
    return normalized;
  } catch {
    return defaultSettings();
  }
}

export function saveSettings(settings: AppSettings): void {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(normalizeSettings(settings)));
  } catch (e) {
    console.error('Failed to save settings:', e);
  }
}
