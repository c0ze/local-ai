export interface GpuCapability {
  available: boolean;
  secureContext: boolean;
  fallbackAdapter: boolean;
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
  info?: GPUAdapterInfoLike;
  requestAdapterInfo?: () => Promise<GPUAdapterInfoLike>;
}

interface WGGpu {
  requestAdapter: (opts?: { powerPreference?: 'high-performance' | 'low-power' }) => Promise<WGAdapter | null>;
}

interface GPUAdapterInfoLike {
  vendor?: string;
  architecture?: string;
  description?: string;
  device?: string;
  isFallbackAdapter?: boolean;
}

export async function detectGpu(): Promise<GpuCapability> {
  const secureContext = typeof window === 'undefined' ? true : window.isSecureContext;
  const empty: GpuCapability = { available: false, secureContext, fallbackAdapter: false, shaderF16: false, features: [] };
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

  const vendor = info?.vendor;
  const architecture = info?.architecture;
  const fallbackAdapter = Boolean(
    info?.isFallbackAdapter ||
      (vendor?.toLowerCase() === 'google' && architecture?.toLowerCase() === 'swiftshader')
  );

  return {
    available: true,
    secureContext,
    fallbackAdapter,
    vendor,
    architecture,
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
