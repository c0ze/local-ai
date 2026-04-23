import { expect, test, type Page } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { MODELS, evaluateDtype, pickRecommendedDtype, pickRunnableDtype, type ModelInfo } from '../src/models';
import type { GpuCapability } from '../src/gpu';
import type { Dtype } from '../src/storage';

type ConcreteDtype = Exclude<Dtype, 'auto'>;
type RunStatus = 'ok' | 'load_failed' | 'inference_failed' | 'timeout' | 'skipped';

type ConsoleRecord = {
  type: string;
  text: string;
  url?: string;
  lineNumber?: number;
  columnNumber?: number;
};

type BrowserRunResult = {
  status: RunStatus;
  answer?: string;
  error?: string;
  loadMs?: number;
  inferMs?: number;
  progress: unknown[];
};

type ModelRunResult = BrowserRunResult & {
  modelId: string;
  label: string;
  dtype: ConcreteDtype;
  console: ConsoleRecord[];
  severeConsole: ConsoleRecord[];
};

const DTYPE_ORDER: ConcreteDtype[] = ['q4f16', 'q4', 'q8', 'fp16', 'fp32'];
const RESULTS_DIR = process.env.MODEL_TEST_RESULTS_DIR ?? 'test-results';
const LOAD_TIMEOUT_MS = Number(process.env.MODEL_TEST_LOAD_TIMEOUT_MS ?? 10 * 60 * 1000);
const INFER_TIMEOUT_MS = Number(process.env.MODEL_TEST_INFER_TIMEOUT_MS ?? 90 * 1000);
const QUESTION = process.env.MODEL_TEST_QUESTION ?? 'Say hello in one short sentence.';

function selectedModels(): ModelInfo[] {
  const raw = process.env.MODEL_TEST_MODELS;
  if (!raw) return MODELS;

  const filters = raw
    .split(',')
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean);

  return MODELS.filter((model) =>
    filters.some((filter) =>
      model.id.toLowerCase().includes(filter) || model.label.toLowerCase().includes(filter)
    )
  );
}

function requestedDtypes(model: ModelInfo, gpu: GpuCapability): ConcreteDtype[] {
  const raw = process.env.MODEL_TEST_DTYPES ?? 'recommended';
  if (raw === 'recommended') {
    const dtype = pickRunnableDtype(model, gpu);
    return dtype ? [dtype] : [];
  }

  const includeTooLarge = process.env.MODEL_TEST_INCLUDE_TOO_LARGE === '1';
  if (raw === 'all') {
    return model.supportedDtypes.filter((dtype) => {
      const status = evaluateDtype(model, dtype, gpu);
      return status === 'ok' || (includeTooLarge && status === 'too_large');
    });
  }

  const requested = raw
    .split(',')
    .map((dtype) => dtype.trim())
    .filter((dtype): dtype is ConcreteDtype => DTYPE_ORDER.includes(dtype as ConcreteDtype));

  return requested.filter((dtype) => {
    if (!model.supportedDtypes.includes(dtype)) return false;
    const status = evaluateDtype(model, dtype, gpu);
    return status === 'ok' || (includeTooLarge && status === 'too_large');
  });
}

function isKnownNoisyConsole(record: ConsoleRecord): boolean {
  return [
    /Unable to determine content-length/i,
    /Some nodes were not assigned to the preferred execution providers/i,
    /Rerunning with verbose output on a non-minimal build/i,
  ].some((pattern) => pattern.test(record.text));
}

function isSevereConsole(record: ConsoleRecord): boolean {
  if (isKnownNoisyConsole(record)) return false;
  if (record.type === 'pageerror' || record.type === 'requestfailed') return true;
  if (record.type === 'error') return true;
  return /RuntimeError|Unauthorized|401|404|memory access out of bounds|out of memory|no available backend|Failed to/i.test(record.text);
}

function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

function choosePassingDtype(results: ModelRunResult[], model: ModelInfo, gpu: GpuCapability): ConcreteDtype {
  const passing = results.filter((result) => result.modelId === model.id && result.status === 'ok');
  for (const dtype of DTYPE_ORDER) {
    if (passing.some((result) => result.dtype === dtype)) return dtype;
  }
  return pickRecommendedDtype(model, gpu);
}

function deriveDefaults(results: ModelRunResult[], gpu: GpuCapability) {
  return MODELS.map((model) => {
    const dtype = choosePassingDtype(results, model, gpu);
    const passing = results
      .filter((result) => result.modelId === model.id && result.status === 'ok')
      .sort((a, b) => (a.inferMs ?? Number.POSITIVE_INFINITY) - (b.inferMs ?? Number.POSITIVE_INFINITY));

    const defaults = { ...model.generationDefaults };
    let reason = passing.length
      ? `Selected ${dtype} from passing browser test.`
      : `No passing browser test in this run; using registry fallback ${dtype}.`;

    const fastest = passing[0];
    if (fastest?.inferMs && fastest.answer) {
      const tokensPerSecond = estimateTokens(fastest.answer) / (fastest.inferMs / 1000);
      if (tokensPerSecond < 0.5) {
        defaults.maxNewTokens = Math.min(defaults.maxNewTokens, 64);
        defaults.contextTokens = Math.min(defaults.contextTokens, 512);
        reason += ' Throughput was very low, so output/context were capped aggressively.';
      } else if (tokensPerSecond < 1.5) {
        defaults.maxNewTokens = Math.min(defaults.maxNewTokens, 96);
        defaults.contextTokens = Math.min(defaults.contextTokens, 768);
        reason += ' Throughput was low, so output/context were capped conservatively.';
      } else if (tokensPerSecond < 3) {
        defaults.maxNewTokens = Math.min(defaults.maxNewTokens, 128);
        defaults.contextTokens = Math.min(defaults.contextTokens, 1024);
        reason += ' Throughput was moderate, so output/context remain conservative.';
      }
    }

    return {
      modelId: model.id,
      label: model.label,
      dtype,
      generation: defaults,
      reason,
    };
  });
}

async function detectGpu(page: Page): Promise<GpuCapability> {
  return page.evaluate(async () => {
    const modulePath = '/src/gpu.ts';
    const { detectGpu: browserDetectGpu } = await import(modulePath);
    return browserDetectGpu();
  });
}

async function runModelInBrowser(
  page: Page,
  model: ModelInfo,
  dtype: ConcreteDtype,
): Promise<BrowserRunResult> {
  return page.evaluate(
    async ({ modelId, dtype, question, loadTimeoutMs, inferTimeoutMs, maxNewTokens, contextTokens }) => {
      const modulePath = '/src/model.ts';
      const { ModelManager } = await import(modulePath);
      const manager = new ModelManager();
      const progress: unknown[] = [];
      let loadMs: number | undefined;
      let inferMs: number | undefined;

      const withTimeout = async <T,>(promise: Promise<T>, ms: number, label: string): Promise<T> => {
        let timer: number | undefined;
        const timeout = new Promise<never>((_, reject) => {
          timer = window.setTimeout(() => reject(new Error(`${label} timeout after ${ms / 1000}s`)), ms);
        });
        try {
          return await Promise.race([promise, timeout]);
        } finally {
          if (timer !== undefined) window.clearTimeout(timer);
        }
      };

      try {
        const loadStarted = performance.now();
        await withTimeout(
          manager.load(modelId, (event: unknown) => progress.push(event), dtype),
          loadTimeoutMs,
          'Model load',
        );
        loadMs = Math.round(performance.now() - loadStarted);

        const abortController = new AbortController();
        const abortTimer = window.setTimeout(() => abortController.abort(), inferTimeoutMs);
        try {
          const inferStarted = performance.now();
          const tokens: string[] = [];
          const answer = await withTimeout<string>(
            manager.generate(
              [{ role: 'user', content: question }],
              (token: string) => tokens.push(token),
              abortController.signal,
              {
                maxNewTokens,
                contextTokens,
                temperature: 0.1,
                topP: 0.9,
                repetitionPenalty: 1.05,
              },
            ),
            inferTimeoutMs,
            'Inference',
          );
          inferMs = Math.round(performance.now() - inferStarted);
          return { status: 'ok', answer: String(answer || tokens.join('')), loadMs, inferMs, progress };
        } finally {
          window.clearTimeout(abortTimer);
        }
      } catch (error) {
        const message = error instanceof Error ? error.stack || error.message : String(error);
        const status = message.includes('timeout')
          ? 'timeout'
          : loadMs === undefined
            ? 'load_failed'
            : 'inference_failed';
        return { status, error: message, loadMs, inferMs, progress };
      } finally {
        await manager.unload().catch(() => {});
      }
    },
    {
      modelId: model.id,
      dtype,
      question: QUESTION,
      loadTimeoutMs: LOAD_TIMEOUT_MS,
      inferTimeoutMs: INFER_TIMEOUT_MS,
      maxNewTokens: model.generationDefaults.maxNewTokens,
      contextTokens: model.generationDefaults.contextTokens,
    },
  );
}

test('downloads, loads, and prompts every configured model', async ({ page }, testInfo) => {
  const consoleRecords: ConsoleRecord[] = [];
  page.on('console', (message) => {
    const location = message.location();
    consoleRecords.push({
      type: message.type(),
      text: message.text(),
      url: location.url,
      lineNumber: location.lineNumber,
      columnNumber: location.columnNumber,
    });
  });
  page.on('pageerror', (error) => {
    consoleRecords.push({
      type: 'pageerror',
      text: error.stack || error.message,
    });
  });
  page.on('requestfailed', (request) => {
    consoleRecords.push({
      type: 'requestfailed',
      text: `${request.method()} ${request.url()} ${request.failure()?.errorText ?? ''}`.trim(),
    });
  });

  await page.goto('/');
  const gpu = await detectGpu(page);
  expect(gpu.available, 'WebGPU must be available to run model integration tests').toBe(true);

  const models = selectedModels();
  expect(models.length, 'At least one model must be selected for compatibility testing').toBeGreaterThan(0);

  const results: ModelRunResult[] = [];
  for (const model of models) {
    const dtypes = requestedDtypes(model, gpu);
    if (dtypes.length === 0) {
      results.push({
        modelId: model.id,
        label: model.label,
        dtype: model.fallbackDtype,
        status: 'skipped',
        error: 'No dtype was viable or requested for this GPU.',
        progress: [],
        console: [],
        severeConsole: [],
      });
      continue;
    }

    for (const dtype of dtypes) {
      const consoleStart = consoleRecords.length;
      const run = await runModelInBrowser(page, model, dtype);
      const scopedConsole = consoleRecords.slice(consoleStart);
      const severeConsole = scopedConsole.filter(isSevereConsole);

      results.push({
        ...run,
        modelId: model.id,
        label: model.label,
        dtype,
        console: scopedConsole,
        severeConsole,
      });

      await page.reload({ waitUntil: 'domcontentloaded' });
    }
  }

  const defaults = deriveDefaults(results, gpu);
  const output = {
    createdAt: new Date().toISOString(),
    question: QUESTION,
    gpu,
    mode: process.env.MODEL_TEST_DTYPES ?? 'recommended',
    results,
    suggestedDefaults: defaults,
  };

  await mkdir(RESULTS_DIR, { recursive: true });
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const resultPath = path.join(RESULTS_DIR, `model-compatibility-${timestamp}.json`);
  await writeFile(resultPath, JSON.stringify(output, null, 2));
  await testInfo.attach('model-compatibility-results', {
    body: JSON.stringify(output, null, 2),
    contentType: 'application/json',
  });

  console.log(`Model compatibility results written to ${resultPath}`);
  for (const result of results) {
    console.log(
      `${result.status.toUpperCase()} ${result.label} ${result.dtype} load=${result.loadMs ?? '-'}ms infer=${result.inferMs ?? '-'}ms answer=${JSON.stringify(result.answer ?? '')}`,
    );
    for (const record of result.severeConsole) {
      console.log(`SEVERE ${result.label} ${result.dtype}: [${record.type}] ${record.text}`);
    }
  }

  const failures = results.filter((result) => result.status !== 'ok' || result.severeConsole.length > 0);
  expect(failures, 'Every tested model/dtype should load, answer, and avoid severe console errors').toEqual([]);
});
