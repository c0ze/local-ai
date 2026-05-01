import { pipeline } from '@huggingface/transformers';

export const MOONSHINE_MODEL_ID = 'onnx-community/moonshine-base-ONNX';
const TARGET_SAMPLE_RATE = 16000;
const MAX_RECORDING_MS = 30_000;

export type SttProgress = {
  status: 'recording' | 'loading' | 'transcribing' | 'ready' | 'error';
  message: string;
  progress?: number;
  file?: string;
};

type ProgressCallback = (progress: SttProgress) => void;
type TranscriberOutput = { text: string };

function getAudioContextCtor(): typeof AudioContext {
  const ctor = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!ctor) throw new Error('Web Audio is not available in this browser.');
  return ctor;
}

async function supportsWebGPU(): Promise<boolean> {
  try {
    const gpu = (navigator as Navigator & { gpu?: { requestAdapter: () => Promise<unknown> } }).gpu;
    if (!gpu) return false;
    return Boolean(await gpu.requestAdapter());
  } catch {
    return false;
  }
}

async function decodeBlobToMono16k(blob: Blob): Promise<Float32Array> {
  const AudioContextCtor = getAudioContextCtor();
  const context = new AudioContextCtor();
  try {
    const arrayBuffer = await blob.arrayBuffer();
    const decoded = await context.decodeAudioData(arrayBuffer);
    const mono = new Float32Array(decoded.length);

    for (let channel = 0; channel < decoded.numberOfChannels; channel++) {
      const data = decoded.getChannelData(channel);
      for (let i = 0; i < data.length; i++) {
        mono[i] += data[i] / decoded.numberOfChannels;
      }
    }

    if (decoded.sampleRate === TARGET_SAMPLE_RATE) return mono;

    const frameCount = Math.max(1, Math.ceil(mono.length * TARGET_SAMPLE_RATE / decoded.sampleRate));
    const offline = new OfflineAudioContext(1, frameCount, TARGET_SAMPLE_RATE);
    const sourceBuffer = offline.createBuffer(1, mono.length, decoded.sampleRate);
    sourceBuffer.getChannelData(0).set(mono);

    const source = offline.createBufferSource();
    source.buffer = sourceBuffer;
    source.connect(offline.destination);
    source.start();

    const rendered = await offline.startRendering();
    return new Float32Array(rendered.getChannelData(0));
  } finally {
    await context.close();
  }
}

export class MoonshineSttManager {
  private transcriber: any = null;
  private loadingPromise: Promise<void> | null = null;
  private recorder: MediaRecorder | null = null;
  private stream: MediaStream | null = null;
  private chunks: Blob[] = [];
  private startedAt = 0;
  private stopTimer: number | null = null;

  get recording(): boolean {
    return this.recorder?.state === 'recording';
  }

  get loading(): boolean {
    return this.loadingPromise !== null;
  }

  async load(onProgress?: ProgressCallback): Promise<void> {
    if (this.transcriber) return;
    if (this.loadingPromise) return this.loadingPromise;

    this.loadingPromise = this.loadModel(onProgress);
    try {
      await this.loadingPromise;
    } finally {
      this.loadingPromise = null;
    }
  }

  async startRecording(onProgress?: ProgressCallback): Promise<void> {
    if (this.recording) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('Microphone capture is not available in this browser.');
    }
    if (typeof MediaRecorder === 'undefined') {
      throw new Error('MediaRecorder is not available in this browser.');
    }

    this.stopTimer && window.clearTimeout(this.stopTimer);
    this.chunks = [];
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        autoGainControl: true,
        noiseSuppression: true,
      },
    });

    this.recorder = new MediaRecorder(this.stream);
    this.recorder.addEventListener('dataavailable', (event) => {
      if (event.data.size > 0) this.chunks.push(event.data);
    });
    this.recorder.start();
    this.startedAt = Date.now();
    this.stopTimer = window.setTimeout(() => {
      if (this.recording) this.recorder?.stop();
    }, MAX_RECORDING_MS);
    onProgress?.({ status: 'recording', message: 'Listening... click the mic again to transcribe.' });
  }

  async stopAndTranscribe(onProgress?: ProgressCallback): Promise<string> {
    if (!this.recorder) return '';

    const recorder = this.recorder;
    if (recorder.state === 'recording') {
      await new Promise<void>((resolve) => {
        recorder.addEventListener('stop', () => resolve(), { once: true });
        recorder.stop();
      });
    }

    this.stopTimer && window.clearTimeout(this.stopTimer);
    this.stopTimer = null;
    this.stopStream();

    const durationMs = Date.now() - this.startedAt;
    if (durationMs < 250 || this.chunks.length === 0) {
      this.cleanupRecording();
      return '';
    }

    onProgress?.({ status: 'transcribing', message: 'Preparing microphone audio...' });
    const blob = new Blob(this.chunks, { type: recorder.mimeType || 'audio/webm' });
    this.cleanupRecording();

    const audio = await decodeBlobToMono16k(blob);
    if (audio.length < TARGET_SAMPLE_RATE * 0.15) return '';

    await this.load(onProgress);
    onProgress?.({ status: 'transcribing', message: 'Transcribing with Moonshine...' });
    const output = await this.transcriber(audio) as TranscriberOutput;
    return output.text.trim();
  }

  cancel(): void {
    if (this.recording) {
      this.recorder?.stop();
    }
    this.stopTimer && window.clearTimeout(this.stopTimer);
    this.stopTimer = null;
    this.stopStream();
    this.cleanupRecording();
  }

  private async loadModel(onProgress?: ProgressCallback): Promise<void> {
    try {
      onProgress?.({ status: 'loading', message: 'Loading Moonshine...' });
      const device = await supportsWebGPU() ? 'webgpu' : 'wasm';
      const dtype: Record<string, 'fp32' | 'q4' | 'q8'> = device === 'webgpu'
        ? { encoder_model: 'fp32', decoder_model_merged: 'q4' }
        : { encoder_model: 'fp32', decoder_model_merged: 'q8' };

      this.transcriber = await pipeline('automatic-speech-recognition', MOONSHINE_MODEL_ID, {
        device,
        dtype,
        progress_callback: (data: any) => {
          onProgress?.(this.normalizeProgress(data));
        },
      });

      await this.transcriber(new Float32Array(TARGET_SAMPLE_RATE));
      onProgress?.({ status: 'ready', message: `Moonshine ready (${device}).` });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      onProgress?.({ status: 'error', message });
      throw error;
    }
  }

  private normalizeProgress(data: any): SttProgress {
    if (data?.status === 'progress' || data?.status === 'progress_total') {
      const progress = Math.round(data.progress ?? 0);
      return {
        status: 'loading',
        progress,
        file: data.file,
        message: `Downloading Moonshine... ${progress}%`,
      };
    }

    if (data?.status === 'done') {
      return {
        status: 'loading',
        progress: 100,
        file: data.file,
        message: 'Loading Moonshine into memory...',
      };
    }

    return {
      status: 'loading',
      file: data?.file,
      message: 'Loading Moonshine...',
    };
  }

  private stopStream(): void {
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
  }

  private cleanupRecording(): void {
    this.recorder = null;
    this.chunks = [];
    this.startedAt = 0;
  }
}
