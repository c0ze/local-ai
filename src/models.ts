import type { Dtype } from './storage';
import type { GpuCapability } from './gpu';

export interface ModelInfo {
  id: string;
  label: string;
  paramsB: number;
  architecture: string;
  // dtypes that ship for this model on HF and are runnable in transformers.js
  supportedDtypes: Exclude<Dtype, 'auto'>[];
  // approximate total weight footprint in MB by dtype (shown to users, not a WebGPU hard limit)
  approxWeightsMb: Partial<Record<Exclude<Dtype, 'auto'>, number>>;
  // largest single GPU buffer allocation observed/expected for this dtype
  maxBufferRequirementMb?: Partial<Record<Exclude<Dtype, 'auto'>, number>>;
  // number of companion ONNX external-data chunks that must be mounted with this dtype
  externalDataChunks?: Partial<Record<Exclude<Dtype, 'auto'>, number>>;
  // the dtype to default to when GPU has the necessary features
  recommendedDtype: Exclude<Dtype, 'auto'>;
  // cheaper fallback for GPUs without shader-f16
  fallbackDtype: Exclude<Dtype, 'auto'>;
  generationDefaults: {
    maxNewTokens: number;
    contextTokens: number;
  };
  contextWindowTokens: number;
  loader?: 'text-generation' | 'gemma4-text';
  notes?: string;
}

const mb = (paramsB: number, bitsPerWeight: number) =>
  Math.round(paramsB * 1000 * bitsPerWeight / 8);

const dtypeProfile = (paramsB: number) => ({
  q4: mb(paramsB, 4.5),
  q4f16: mb(paramsB, 4.5),
  q8: mb(paramsB, 8),
  fp16: mb(paramsB, 16),
  fp32: mb(paramsB, 32),
});

export const MODELS: ModelInfo[] = [
  {
    id: 'Xenova/distilgpt2',
    label: 'DistilGPT-2',
    paramsB: 0.082,
    architecture: 'gpt2',
    supportedDtypes: ['fp32', 'fp16', 'q8'],
    approxWeightsMb: dtypeProfile(0.082),
    recommendedDtype: 'q8',
    fallbackDtype: 'q8',
    generationDefaults: { maxNewTokens: 64, contextTokens: 512 },
    contextWindowTokens: 1024,
    notes: 'Tiny sanity-check model. Not instruction-tuned.',
  },
  {
    id: 'onnx-community/SmolLM2-360M-Instruct-ONNX',
    label: 'SmolLM2 360M Instruct',
    paramsB: 0.36,
    architecture: 'llama',
    supportedDtypes: ['q4f16', 'q4', 'q8', 'fp16'],
    approxWeightsMb: dtypeProfile(0.36),
    recommendedDtype: 'q4f16',
    fallbackDtype: 'q4',
    generationDefaults: { maxNewTokens: 160, contextTokens: 1280 },
    contextWindowTokens: 8192,
  },
  {
    id: 'onnx-community/Qwen2.5-0.5B-Instruct',
    label: 'Qwen 2.5 0.5B Instruct',
    paramsB: 0.5,
    architecture: 'qwen2',
    supportedDtypes: ['q4f16', 'q4', 'q8', 'fp16'],
    approxWeightsMb: dtypeProfile(0.5),
    recommendedDtype: 'q4f16',
    fallbackDtype: 'q4',
    generationDefaults: { maxNewTokens: 192, contextTokens: 1536 },
    contextWindowTokens: 32768,
  },
  {
    id: 'onnx-community/Llama-3.2-1B-Instruct',
    label: 'Llama 3.2 1B Instruct',
    paramsB: 1.24,
    architecture: 'llama',
    supportedDtypes: ['q4f16', 'q4', 'q8'],
    approxWeightsMb: dtypeProfile(1.24),
    externalDataChunks: { q4f16: 1, q4: 1 },
    recommendedDtype: 'q4f16',
    fallbackDtype: 'q4',
    generationDefaults: { maxNewTokens: 128, contextTokens: 1024 },
    contextWindowTokens: 131072,
  },
  {
    id: 'onnx-community/deepseek-coder-1.3b-instruct-ONNX',
    label: 'DeepSeek Coder 1.3B Instruct',
    paramsB: 1.3,
    architecture: 'llama',
    supportedDtypes: ['q4f16', 'q4', 'q8'],
    approxWeightsMb: {
      q4f16: 814,
      q4: 1016,
      q8: 1285,
    },
    maxBufferRequirementMb: {
      q4f16: 814,
      q4: 1016,
      q8: 1285,
    },
    recommendedDtype: 'q4f16',
    fallbackDtype: 'q4',
    generationDefaults: { maxNewTokens: 160, contextTokens: 1280 },
    contextWindowTokens: 16384,
    notes: 'Small DeepSeek coding model; public Transformers.js ONNX.',
  },
  {
    id: 'onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX',
    label: 'DeepSeek R1 Distill Qwen 1.5B',
    paramsB: 1.5,
    architecture: 'qwen2',
    supportedDtypes: ['q4f16', 'q4', 'q8'],
    approxWeightsMb: {
      q4f16: 1306,
      q4: 1875,
      q8: 1760,
    },
    maxBufferRequirementMb: {
      q4f16: 1306,
      q4: 1875,
      q8: 1760,
    },
    recommendedDtype: 'q4f16',
    fallbackDtype: 'q4',
    generationDefaults: { maxNewTokens: 192, contextTokens: 1536 },
    contextWindowTokens: 131072,
    notes: 'Reasoning-tuned DeepSeek distill. No public 4B DeepSeek WebGPU ONNX found.',
  },
  {
    id: 'onnx-community/Qwen2.5-Coder-3B-Instruct',
    label: 'Qwen 2.5 Coder 3B Instruct',
    paramsB: 3.1,
    architecture: 'qwen2',
    supportedDtypes: ['q4f16', 'q4', 'q8'],
    approxWeightsMb: {
      q4f16: 2258,
      q4: 3043,
      q8: 3257,
    },
    maxBufferRequirementMb: {
      q4f16: 1999,
      q4: 1996,
      q8: 3256,
    },
    externalDataChunks: { q4f16: 1, q4: 1, q8: 1 },
    recommendedDtype: 'q4f16',
    fallbackDtype: 'q4',
    generationDefaults: { maxNewTokens: 128, contextTokens: 1024 },
    contextWindowTokens: 32768,
    notes: 'Larger browser coding model; first load is a multi-GB download.',
  },
  {
    id: 'onnx-community/Qwen3-4B-ONNX',
    label: 'Qwen3 4B',
    paramsB: 4.0,
    architecture: 'qwen3',
    supportedDtypes: ['q4f16'],
    approxWeightsMb: {
      q4f16: 2702,
    },
    maxBufferRequirementMb: {
      q4f16: 1999,
    },
    externalDataChunks: { q4f16: 2 },
    recommendedDtype: 'q4f16',
    fallbackDtype: 'q4f16',
    generationDefaults: { maxNewTokens: 128, contextTokens: 1024 },
    contextWindowTokens: 40960,
    notes: '4B Qwen chat model. Requires shader-f16 and >2 GB WebGPU buffers.',
  },
  {
    id: 'microsoft/Phi-3-mini-4k-instruct-onnx-web',
    label: 'Phi-3 mini 4k Instruct',
    paramsB: 3.82,
    architecture: 'phi3',
    supportedDtypes: ['q4', 'q4f16'],
    approxWeightsMb: dtypeProfile(3.82),
    externalDataChunks: { q4: 1, q4f16: 1 },
    recommendedDtype: 'q4',
    fallbackDtype: 'q4',
    generationDefaults: { maxNewTokens: 96, contextTokens: 768 },
    contextWindowTokens: 4096,
    notes: 'Largest preset. Needs ~3 GB VRAM.',
  },
  {
    id: 'onnx-community/gemma-4-E4B-it-ONNX',
    label: 'Gemma 4 E4B Instruct',
    paramsB: 4.5,
    architecture: 'gemma4',
    supportedDtypes: ['q4f16', 'q4'],
    approxWeightsMb: {
      q4f16: 4678,
      q4: 5356,
    },
    maxBufferRequirementMb: {
      q4f16: 1344,
      q4: 1997,
    },
    recommendedDtype: 'q4f16',
    fallbackDtype: 'q4',
    generationDefaults: { maxNewTokens: 64, contextTokens: 512 },
    contextWindowTokens: 131072,
    loader: 'gemma4-text',
    notes: 'Experimental text-only Gemma4 loader. Large ONNX external-data download.',
  },
];

export function findModel(id: string): ModelInfo | undefined {
  return MODELS.find((m) => m.id === id);
}

export function normalizeModelId(id: string): string {
  const trimmed = id.trim();
  const alias = trimmed.toLowerCase();
  const aliases: Record<string, string> = {
    'gemma-4-e4b-it': 'onnx-community/gemma-4-E4B-it-ONNX',
    'gemma4-e4b-it': 'onnx-community/gemma-4-E4B-it-ONNX',
    'gemma-4-e4b': 'onnx-community/gemma-4-E4B-it-ONNX',
    'gemma4-e4b': 'onnx-community/gemma-4-E4B-it-ONNX',
    'qwen3-4b': 'onnx-community/Qwen3-4B-ONNX',
    'qwen-3-4b': 'onnx-community/Qwen3-4B-ONNX',
    'qwen2.5-coder-3b': 'onnx-community/Qwen2.5-Coder-3B-Instruct',
    'qwen-coder-3b': 'onnx-community/Qwen2.5-Coder-3B-Instruct',
    'deepseek-r1-1.5b': 'onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX',
    'deepseek-r1-distill-qwen-1.5b': 'onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX',
    'deepseek-coder-1.3b': 'onnx-community/deepseek-coder-1.3b-instruct-ONNX',
  };
  return aliases[alias] ?? trimmed;
}

export function modelLoaderFor(id: string): ModelInfo['loader'] {
  const normalized = normalizeModelId(id);
  const known = findModel(normalized);
  if (known?.loader) return known.loader;
  return /(^|\/)gemma-4-e4b-it/i.test(normalized) ? 'gemma4-text' : 'text-generation';
}

export function externalDataChunksFor(
  id: string,
  dtype: Exclude<Dtype, 'auto'>,
): number | undefined {
  return findModel(normalizeModelId(id))?.externalDataChunks?.[dtype];
}

export type DtypeStatus = 'ok' | 'unsupported_by_model' | 'needs_shader_f16' | 'too_large';

export function evaluateDtype(
  model: ModelInfo,
  dtype: Exclude<Dtype, 'auto'>,
  gpu: GpuCapability,
): DtypeStatus {
  if (!model.supportedDtypes.includes(dtype)) return 'unsupported_by_model';
  if ((dtype === 'q4f16' || dtype === 'fp16') && !gpu.shaderF16) return 'needs_shader_f16';
  const maxBufferRequirementMb = model.maxBufferRequirementMb?.[dtype];
  if (
    maxBufferRequirementMb &&
    gpu.maxBufferBytes &&
    maxBufferRequirementMb * 1024 * 1024 > gpu.maxBufferBytes
  ) {
    return 'too_large';
  }
  return 'ok';
}

/**
 * Pick the smallest dtype the model ships in that is also runnable on the GPU.
 * Order: q4f16, q4, q8, fp16, fp32 (smallest → largest).
 */
const DTYPE_ORDER: Exclude<Dtype, 'auto'>[] = ['q4f16', 'q4', 'q8', 'fp16', 'fp32'];

export function pickRunnableDtype(model: ModelInfo, gpu: GpuCapability): Exclude<Dtype, 'auto'> | null {
  for (const dtype of DTYPE_ORDER) {
    if (!model.supportedDtypes.includes(dtype)) continue;
    if (evaluateDtype(model, dtype, gpu) === 'ok') return dtype;
  }
  return null;
}

export function pickRecommendedDtype(model: ModelInfo, gpu: GpuCapability): Exclude<Dtype, 'auto'> {
  const runnable = pickRunnableDtype(model, gpu);
  if (runnable) return runnable;
  return model.fallbackDtype;
}
