import { ModelManager, type Dtype as ModelDtype } from './model';
import { MODELS, evaluateDtype, pickRunnableDtype, type ModelInfo } from './models';
import type { GpuCapability } from './gpu';
import type { Dtype } from './storage';

export type TestStatus = 'ok' | 'load_failed' | 'inference_failed' | 'timeout' | 'skipped';

export interface TestResult {
  modelId: string;
  dtype: Exclude<Dtype, 'auto'>;
  status: TestStatus;
  loadMs?: number;
  inferMs?: number;
  response?: string;
  error?: string;
  timestamp: number;
}

const RESULTS_KEY = 'local-ai-chat-test-results';
const LOAD_TIMEOUT_MS = 10 * 60 * 1000;
const INFER_TIMEOUT_MS = 90 * 1000;
const VALID_DTYPES: Exclude<Dtype, 'auto'>[] = ['q4f16', 'q4', 'q8', 'fp16', 'fp32'];
const VALID_STATUSES: TestStatus[] = ['ok', 'load_failed', 'inference_failed', 'timeout', 'skipped'];

function isOptionalNumber(value: unknown): boolean {
  return value === undefined || (typeof value === 'number' && Number.isFinite(value));
}

function isOptionalString(value: unknown): boolean {
  return value === undefined || typeof value === 'string';
}

function isTestResult(value: unknown): value is TestResult {
  if (!value || typeof value !== 'object') return false;
  const result = value as Partial<TestResult>;
  return (
    typeof result.modelId === 'string' &&
    VALID_DTYPES.includes(result.dtype as Exclude<Dtype, 'auto'>) &&
    VALID_STATUSES.includes(result.status as TestStatus) &&
    typeof result.timestamp === 'number' &&
    Number.isFinite(result.timestamp) &&
    isOptionalNumber(result.loadMs) &&
    isOptionalNumber(result.inferMs) &&
    isOptionalString(result.response) &&
    isOptionalString(result.error)
  );
}

export function loadTestResults(): TestResult[] {
  try {
    const raw = localStorage.getItem(RESULTS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter(isTestResult) : [];
  } catch {
    return [];
  }
}

function saveTestResults(results: TestResult[]): void {
  try {
    localStorage.setItem(RESULTS_KEY, JSON.stringify(results));
  } catch (e) {
    console.error('Failed to save test results:', e);
  }
}

export function clearTestResults(): void {
  localStorage.removeItem(RESULTS_KEY);
}

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timeout after ${ms / 1000}s`)), ms);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (err) => {
        clearTimeout(timer);
        reject(err);
      },
    );
  });
}

function createTimeoutSignal(parent: AbortSignal | undefined, ms: number): { signal: AbortSignal; cleanup: () => void } {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), ms);
  const abort = () => controller.abort();
  parent?.addEventListener('abort', abort, { once: true });
  return {
    signal: controller.signal,
    cleanup: () => {
      window.clearTimeout(timer);
      parent?.removeEventListener('abort', abort);
    },
  };
}

export type TestProgress = (info: {
  index: number;
  total: number;
  modelId: string;
  dtype: Exclude<Dtype, 'auto'>;
  phase: 'load' | 'infer' | 'done';
  message?: string;
}) => void;

export interface RunOptions {
  signal?: AbortSignal;
  onProgress?: TestProgress;
  /** When true, test all supported dtypes per model. When false, only the recommended dtype. */
  exhaustive?: boolean;
}

function plannedDtypesFor(model: ModelInfo, gpu: GpuCapability, exhaustive: boolean): Exclude<Dtype, 'auto'>[] {
  if (!exhaustive) {
    const dtype = pickRunnableDtype(model, gpu);
    return dtype ? [dtype] : [];
  }
  return model.supportedDtypes.filter((d) => evaluateDtype(model, d, gpu) === 'ok');
}

export async function runCompatibilityTests(
  gpu: GpuCapability,
  options: RunOptions = {},
): Promise<TestResult[]> {
  const exhaustive = options.exhaustive ?? false;
  const onProgress = options.onProgress ?? (() => {});

  const plan: { model: ModelInfo; dtype: Exclude<Dtype, 'auto'> }[] = [];
  for (const model of MODELS) {
    for (const dtype of plannedDtypesFor(model, gpu, exhaustive)) {
      plan.push({ model, dtype });
    }
  }

  const results = loadTestResults();
  // Strip any prior results for the (model, dtype) combos we're about to re-run.
  const planKeys = new Set(plan.map(({ model, dtype }) => `${model.id}|${dtype}`));
  const fresh = results.filter((r) => !planKeys.has(`${r.modelId}|${r.dtype}`));

  for (let i = 0; i < plan.length; i++) {
    if (options.signal?.aborted) break;
    const { model, dtype } = plan[i];
    const ts = Date.now();
    onProgress({ index: i, total: plan.length, modelId: model.id, dtype, phase: 'load' });

    const manager = new ModelManager();
    const result: TestResult = { modelId: model.id, dtype, status: 'ok', timestamp: ts };

    try {
      const loadStart = performance.now();
      await withTimeout(
        manager.load(model.id, () => {}, dtype as ModelDtype),
        LOAD_TIMEOUT_MS,
        'Model load',
      );
      result.loadMs = Math.round(performance.now() - loadStart);

      if (options.signal?.aborted) {
        await manager.unload();
        break;
      }

      onProgress({ index: i, total: plan.length, modelId: model.id, dtype, phase: 'infer' });
      const inferStart = performance.now();
      const tokens: string[] = [];
      const chat = [{ role: 'user' as const, content: 'Say hello in one short sentence.' }];
      const inferTimeout = createTimeoutSignal(options.signal, INFER_TIMEOUT_MS);
      try {
        await withTimeout(
          manager.generate(chat, (t) => tokens.push(t), inferTimeout.signal, { maxNewTokens: 32, temperature: 0.3 }),
          INFER_TIMEOUT_MS,
          'Inference',
        );
      } finally {
        inferTimeout.cleanup();
      }
      result.inferMs = Math.round(performance.now() - inferStart);
      result.response = tokens.join('').slice(0, 200);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      if (message.includes('timeout')) result.status = 'timeout';
      else if (result.loadMs === undefined) result.status = 'load_failed';
      else result.status = 'inference_failed';
      result.error = message;
    } finally {
      await manager.unload().catch(() => {});
    }

    onProgress({ index: i, total: plan.length, modelId: model.id, dtype, phase: 'done', message: result.status });
    fresh.push(result);
    saveTestResults(fresh);
  }

  return fresh;
}

/**
 * Pick the smallest dtype that has a passing test for this model, falling back to the
 * registry recommendation when no test data is available.
 */
const DTYPE_ORDER: Exclude<Dtype, 'auto'>[] = ['q4f16', 'q4', 'q8', 'fp16', 'fp32'];

export function suggestedDtypeFromTests(
  modelId: string,
  fallback: Exclude<Dtype, 'auto'>,
  results: TestResult[],
): Exclude<Dtype, 'auto'> {
  const passing = results.filter((r) => r.modelId === modelId && r.status === 'ok');
  if (!passing.length) return fallback;
  for (const dtype of DTYPE_ORDER) {
    if (passing.some((r) => r.dtype === dtype)) return dtype;
  }
  return fallback;
}
