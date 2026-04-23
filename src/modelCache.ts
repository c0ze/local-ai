import { env } from '@huggingface/transformers';

export interface CachedModelFile {
  request: Request;
  url: string;
  file: string;
  revision: string;
  sizeBytes?: number;
}

export interface CachedModel {
  modelId: string;
  files: CachedModelFile[];
  totalBytes: number;
  hasUnknownBytes: boolean;
}

export interface StorageEstimate {
  usageBytes?: number;
  quotaBytes?: number;
}

const CACHE_KEY = env.cacheKey || 'transformers-cache';
const HF_HOSTS = new Set(['huggingface.co', 'hf.co']);

function parseHuggingFaceModelUrl(url: string): { modelId: string; revision: string; file: string } | null {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return null;
  }

  if (!HF_HOSTS.has(parsed.hostname)) return null;

  const parts = parsed.pathname.split('/').filter(Boolean).map(decodeURIComponent);
  const resolveIndex = parts.indexOf('resolve');
  if (resolveIndex < 2 || resolveIndex + 2 > parts.length) return null;

  return {
    modelId: `${parts[0]}/${parts[1]}`,
    revision: parts[resolveIndex + 1] || 'main',
    file: parts.slice(resolveIndex + 2).join('/'),
  };
}

function responseSize(response: Response | undefined): number | undefined {
  if (!response) return undefined;
  const header = response.headers.get('content-length') || response.headers.get('x-linked-size');
  if (!header) return undefined;

  const parsed = Number(header);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
}

async function openTransformersCache(): Promise<Cache | null> {
  if (typeof caches === 'undefined') return null;
  return caches.open(CACHE_KEY);
}

export async function listDownloadedModels(): Promise<CachedModel[]> {
  const cache = await openTransformersCache();
  if (!cache) return [];

  const requests = await cache.keys();
  const groups = new Map<string, CachedModel>();

  for (const request of requests) {
    const parsed = parseHuggingFaceModelUrl(request.url);
    if (!parsed) continue;

    const response = await cache.match(request);
    const sizeBytes = responseSize(response);
    const group = groups.get(parsed.modelId) ?? {
      modelId: parsed.modelId,
      files: [],
      totalBytes: 0,
      hasUnknownBytes: false,
    };

    group.files.push({
      request,
      url: request.url,
      file: parsed.file,
      revision: parsed.revision,
      sizeBytes,
    });
    if (sizeBytes === undefined) {
      group.hasUnknownBytes = true;
    } else {
      group.totalBytes += sizeBytes;
    }
    groups.set(parsed.modelId, group);
  }

  return Array.from(groups.values())
    .map((model) => ({
      ...model,
      files: model.files.sort((a, b) => a.file.localeCompare(b.file)),
    }))
    .sort((a, b) => a.modelId.localeCompare(b.modelId));
}

export async function deleteDownloadedModel(modelId: string): Promise<number> {
  const cache = await openTransformersCache();
  if (!cache) return 0;

  const requests = await cache.keys();
  let deleted = 0;
  for (const request of requests) {
    const parsed = parseHuggingFaceModelUrl(request.url);
    if (parsed?.modelId !== modelId) continue;
    if (await cache.delete(request)) deleted += 1;
  }
  return deleted;
}

export async function clearDownloadedModels(): Promise<number> {
  const cache = await openTransformersCache();
  if (!cache) return 0;

  const requests = await cache.keys();
  let deleted = 0;
  for (const request of requests) {
    if (!parseHuggingFaceModelUrl(request.url)) continue;
    if (await cache.delete(request)) deleted += 1;
  }
  return deleted;
}

export async function estimateStorage(): Promise<StorageEstimate | null> {
  if (typeof navigator === 'undefined' || !navigator.storage?.estimate) return null;
  const estimate = await navigator.storage.estimate();
  return {
    usageBytes: estimate.usage,
    quotaBytes: estimate.quota,
  };
}
