export interface GpuCapability {
  available: boolean;
  vendor?: string;
  architecture?: string;
  description?: string;
  shaderF16: boolean;
  maxBufferBytes?: number;
  maxStorageBufferBindingBytes?: number;
  features: string[];
}

interface WGAdapter {
  features: ReadonlySet<string>;
  limits: Record<string, number>;
  info?: { vendor?: string; architecture?: string; description?: string; device?: string };
  requestAdapterInfo?: () => Promise<{ vendor?: string; architecture?: string; description?: string }>;
}

interface WGGpu {
  requestAdapter: (opts?: { powerPreference?: 'high-performance' | 'low-power' }) => Promise<WGAdapter | null>;
}

export async function detectGpu(): Promise<GpuCapability> {
  const empty: GpuCapability = { available: false, shaderF16: false, features: [] };
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) return empty;

  const gpu = (navigator as Navigator & { gpu?: WGGpu }).gpu;
  if (!gpu) return empty;

  let adapter: WGAdapter | null = null;
  try {
    adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch {
    return empty;
  }
  if (!adapter) return empty;

  let info = adapter.info;
  if (!info && adapter.requestAdapterInfo) {
    try {
      info = await adapter.requestAdapterInfo();
    } catch {
      /* ignore */
    }
  }

  return {
    available: true,
    vendor: info?.vendor,
    architecture: info?.architecture,
    description: info?.description,
    shaderF16: adapter.features.has('shader-f16'),
    maxBufferBytes: adapter.limits.maxBufferSize,
    maxStorageBufferBindingBytes: adapter.limits.maxStorageBufferBindingSize,
    features: Array.from(adapter.features),
  };
}

export function formatBytes(bytes?: number): string {
  if (!bytes) return '–';
  const gb = bytes / 1024 ** 3;
  if (gb >= 1) return `${gb.toFixed(2)} GB`;
  const mb = bytes / 1024 ** 2;
  return `${mb.toFixed(0)} MB`;
}
