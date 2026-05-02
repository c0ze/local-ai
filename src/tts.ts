export const KOKORO_MODEL_ID = 'onnx-community/Kokoro-82M-v1.0-ONNX';

export const KOKORO_VOICES = [
  { id: 'af_heart', name: 'Heart', language: 'American English', gender: 'Female', grade: 'A' },
  { id: 'af_alloy', name: 'Alloy', language: 'American English', gender: 'Female', grade: 'C' },
  { id: 'af_aoede', name: 'Aoede', language: 'American English', gender: 'Female', grade: 'C+' },
  { id: 'af_bella', name: 'Bella', language: 'American English', gender: 'Female', grade: 'A-' },
  { id: 'af_jessica', name: 'Jessica', language: 'American English', gender: 'Female', grade: 'D' },
  { id: 'af_kore', name: 'Kore', language: 'American English', gender: 'Female', grade: 'C+' },
  { id: 'af_nicole', name: 'Nicole', language: 'American English', gender: 'Female', grade: 'B-' },
  { id: 'af_nova', name: 'Nova', language: 'American English', gender: 'Female', grade: 'C' },
  { id: 'af_river', name: 'River', language: 'American English', gender: 'Female', grade: 'D' },
  { id: 'af_sarah', name: 'Sarah', language: 'American English', gender: 'Female', grade: 'C+' },
  { id: 'af_sky', name: 'Sky', language: 'American English', gender: 'Female', grade: 'C-' },
  { id: 'am_adam', name: 'Adam', language: 'American English', gender: 'Male', grade: 'F+' },
  { id: 'am_echo', name: 'Echo', language: 'American English', gender: 'Male', grade: 'D' },
  { id: 'am_eric', name: 'Eric', language: 'American English', gender: 'Male', grade: 'D' },
  { id: 'am_fenrir', name: 'Fenrir', language: 'American English', gender: 'Male', grade: 'C+' },
  { id: 'am_liam', name: 'Liam', language: 'American English', gender: 'Male', grade: 'D' },
  { id: 'am_michael', name: 'Michael', language: 'American English', gender: 'Male', grade: 'C+' },
  { id: 'am_onyx', name: 'Onyx', language: 'American English', gender: 'Male', grade: 'D' },
  { id: 'am_puck', name: 'Puck', language: 'American English', gender: 'Male', grade: 'C+' },
  { id: 'am_santa', name: 'Santa', language: 'American English', gender: 'Male', grade: 'D-' },
  { id: 'bf_alice', name: 'Alice', language: 'British English', gender: 'Female', grade: 'D' },
  { id: 'bf_emma', name: 'Emma', language: 'British English', gender: 'Female', grade: 'B-' },
  { id: 'bf_isabella', name: 'Isabella', language: 'British English', gender: 'Female', grade: 'C' },
  { id: 'bf_lily', name: 'Lily', language: 'British English', gender: 'Female', grade: 'D' },
  { id: 'bm_daniel', name: 'Daniel', language: 'British English', gender: 'Male', grade: 'D' },
  { id: 'bm_fable', name: 'Fable', language: 'British English', gender: 'Male', grade: 'C' },
  { id: 'bm_george', name: 'George', language: 'British English', gender: 'Male', grade: 'C' },
  { id: 'bm_lewis', name: 'Lewis', language: 'British English', gender: 'Male', grade: 'D+' },
] as const;

export type KokoroVoiceId = typeof KOKORO_VOICES[number]['id'];

export type TtsProgress = {
  status: 'init' | 'download' | 'loading' | 'ready' | 'error';
  progress?: number;
  file?: string;
  message?: string;
  durationSeconds?: number;
  totalDurationSeconds?: number;
  peak?: number;
  chunkIndex?: number;
};

export type SpeakOptions = {
  voice: string;
  speed: number;
  onProgress?: (progress: TtsProgress) => void;
  onEnded?: () => void;
};

export type WarmOptions = Pick<SpeakOptions, 'voice' | 'speed' | 'onProgress'>;

type RawAudioLike = {
  audio?: Float32Array;
  sampling_rate?: number;
  toBlob?: () => Blob;
};

const VOICE_IDS = new Set<string>(KOKORO_VOICES.map((voice) => voice.id));

export function isKokoroVoiceId(value: string): value is KokoroVoiceId {
  return VOICE_IDS.has(value);
}

export function kokoroVoiceLabel(id: string): string {
  const voice = KOKORO_VOICES.find((item) => item.id === id);
  return voice ? `${voice.name} (${voice.language}, ${voice.gender})` : id;
}

export class KokoroTtsManager {
  private tts: any = null;
  private loadingPromise: Promise<void> | null = null;
  private audioContext: AudioContext | null = null;
  private source: AudioBufferSourceNode | null = null;
  private pendingText = '';
  private sentenceQueue: Array<{ index: number; text: string }> = [];
  private completed = new Map<number, AudioBuffer>();
  private activeSynths = 0;
  private inputClosed = false;
  private paused = false;
  private active = false;
  private options: SpeakOptions | null = null;
  private currentResolve: (() => void) | null = null;
  private currentReject: ((error: unknown) => void) | null = null;
  private currentOnEnded: (() => void) | undefined;
  private playbackToken = 0;
  private nextSentenceIndex = 0;
  private nextPlayIndex = 0;
  private synthesizedChunks = 0;
  private totalDurationSeconds = 0;
  private warmupPromise: Promise<void> | null = null;
  private warmedVoices = new Set<string>();

  private readonly maxConcurrentSynths = 3;

  get loaded(): boolean {
    return this.tts !== null;
  }

  get loading(): boolean {
    return this.loadingPromise !== null;
  }

  get playing(): boolean {
    return this.active && !this.paused;
  }

  get isActive(): boolean {
    return this.active;
  }

  get isPaused(): boolean {
    return this.active && this.paused;
  }

  async load(onProgress?: (progress: TtsProgress) => void): Promise<void> {
    if (this.tts) return;
    if (this.loadingPromise) return this.loadingPromise;

    this.loadingPromise = this.loadModel(onProgress);
    try {
      await this.loadingPromise;
    } finally {
      this.loadingPromise = null;
    }
  }

  async warm(options: WarmOptions): Promise<void> {
    const voice = isKokoroVoiceId(options.voice) ? options.voice : 'af_heart';
    if (this.warmedVoices.has(voice)) {
      options.onProgress?.({ status: 'ready', progress: 100, message: 'Kokoro ready.' });
      return;
    }
    if (this.warmupPromise) return this.warmupPromise;

    this.warmupPromise = this.warmVoice(voice, options);
    try {
      await this.warmupPromise;
    } finally {
      this.warmupPromise = null;
    }
  }

  async start(options: SpeakOptions): Promise<void> {
    this.stop();
    const token = ++this.playbackToken;

    this.pendingText = '';
    this.sentenceQueue = [];
    this.completed.clear();
    this.activeSynths = 0;
    this.inputClosed = false;
    this.paused = false;
    this.active = true;
    this.options = options;
    this.currentOnEnded = options.onEnded;
    this.nextSentenceIndex = 0;
    this.nextPlayIndex = 0;
    this.synthesizedChunks = 0;
    this.totalDurationSeconds = 0;

    options.onProgress?.({ status: 'loading', message: 'Starting speech pipeline...' });

    const done = new Promise<void>((resolve, reject) => {
      this.currentResolve = resolve;
      this.currentReject = reject;
    });

    void (async () => {
      try {
        await this.ensureAudioContext();
        await this.preparePipeline(token, options);
      } catch (error) {
        if (token === this.playbackToken) this.failCurrent(error);
      }
    })();
    return done;
  }

  async speak(text: string, options: SpeakOptions): Promise<void> {
    const trimmed = text.trim();
    if (!trimmed) throw new Error('Nothing to speak.');

    const done = this.start(options);
    this.pushText(trimmed);
    this.closeInput();
    return done;
  }

  pushText(text: string): void {
    if (!this.active || this.inputClosed || !text) return;
    this.pendingText += text;

    for (const sentence of this.takeCompleteSentences(false)) {
      this.enqueueSentence(sentence);
    }

    this.pumpSynths(this.playbackToken);
  }

  closeInput(): void {
    if (!this.active || this.inputClosed) return;
    for (const sentence of this.takeCompleteSentences(true)) {
      this.enqueueSentence(sentence);
    }
    this.inputClosed = true;
    this.pumpSynths(this.playbackToken);
    this.finishIfIdle(this.playbackToken);
  }

  stop(): void {
    this.playbackToken += 1;
    if (this.source) {
      try {
        this.source.stop();
      } catch {
        /* Source may already be stopped. */
      }
    }
    this.pendingText = '';
    this.sentenceQueue = [];
    this.completed.clear();
    this.activeSynths = 0;
    this.inputClosed = true;
    this.paused = false;
    this.active = false;
    this.options = null;
    this.currentOnEnded = undefined;
    this.clearAudio();
    this.resolveCurrent();
  }

  async pause(): Promise<void> {
    if (!this.active || this.paused) return;
    this.paused = true;
    if (this.audioContext?.state === 'running') {
      await this.audioContext.suspend();
    }
  }

  async resume(): Promise<void> {
    if (!this.active || !this.paused) return;
    this.paused = false;
    if (this.audioContext?.state === 'suspended') {
      await this.audioContext.resume();
    }
    this.playNextQueued(this.playbackToken);
  }

  private async ensureAudioContext(): Promise<AudioContext> {
    const AudioContextCtor = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioContextCtor) throw new Error('Web Audio is not available in this browser.');

    if (!this.audioContext) {
      this.audioContext = new AudioContextCtor();
    }

    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }

    return this.audioContext;
  }

  private async preparePipeline(token: number, options: SpeakOptions): Promise<void> {
    try {
      await this.load(options.onProgress);
      if (token !== this.playbackToken) return;
      options.onProgress?.({ status: 'loading', message: 'Speech pipeline ready.' });
      this.pumpSynths(token);
      this.finishIfIdle(token);
    } catch (error) {
      if (token !== this.playbackToken) return;
      this.failCurrent(error);
    }
  }

  private async warmVoice(voice: KokoroVoiceId, options: WarmOptions): Promise<void> {
    await this.load(options.onProgress);
    if (this.warmedVoices.has(voice)) return;

    options.onProgress?.({ status: 'loading', message: `Preparing ${kokoroVoiceLabel(voice)}...` });
    await this.tts.generate('Ready.', {
      voice,
      speed: options.speed,
    });
    this.warmedVoices.add(voice);
    options.onProgress?.({ status: 'ready', progress: 100, message: 'Kokoro ready.' });
  }

  private audioBufferFromRawAudio(audio: RawAudioLike): AudioBuffer {
    const waveform = audio.audio;
    const sampleRate = audio.sampling_rate;
    if (!waveform || !sampleRate) {
      throw new Error('Kokoro returned audio in an unsupported format.');
    }

    const peak = this.peak(waveform);
    if (waveform.length === 0 || peak <= 0.000001) {
      throw new Error('Kokoro generated silent audio.');
    }

    if (!this.audioContext) throw new Error('Audio context was not initialized.');

    const buffer = this.audioContext.createBuffer(1, waveform.length, sampleRate);
    buffer.getChannelData(0).set(waveform);
    return buffer;
  }

  private enqueueSentence(text: string): void {
    const clean = text.trim();
    if (!clean) return;
    this.sentenceQueue.push({
      index: this.nextSentenceIndex,
      text: clean,
    });
    this.nextSentenceIndex += 1;
    this.options?.onProgress?.({
      status: 'loading',
      message: `Queued sentence ${this.nextSentenceIndex}.`,
      chunkIndex: this.nextSentenceIndex,
    });
  }

  private pumpSynths(token: number): void {
    if (token !== this.playbackToken || !this.tts || !this.options) return;

    while (this.activeSynths < this.maxConcurrentSynths && this.sentenceQueue.length > 0) {
      const job = this.sentenceQueue.shift()!;
      this.activeSynths += 1;
      void this.synthesizeSentence(job, token, this.options);
    }
  }

  private async synthesizeSentence(
    job: { index: number; text: string },
    token: number,
    options: SpeakOptions,
  ): Promise<void> {
    try {
      const audio = await this.tts.generate(job.text, {
        voice: isKokoroVoiceId(options.voice) ? options.voice : 'af_heart',
        speed: options.speed,
      }) as RawAudioLike;

      if (token !== this.playbackToken) return;

      const buffer = this.audioBufferFromRawAudio(audio);
      this.completed.set(job.index, buffer);
      this.synthesizedChunks += 1;
      this.totalDurationSeconds += buffer.duration;
      options.onProgress?.({
        status: 'ready',
        message: this.source
          ? `Prepared sentence ${job.index + 1}.`
          : `Playing sentence ${job.index + 1}.`,
        chunkIndex: job.index + 1,
        durationSeconds: buffer.duration,
        totalDurationSeconds: this.totalDurationSeconds,
        peak: this.peak(audio.audio!),
      });
      this.playNextQueued(token);
    } catch (error) {
      if (token === this.playbackToken) this.failCurrent(error);
    } finally {
      this.activeSynths = Math.max(0, this.activeSynths - 1);
      this.pumpSynths(token);
      this.finishIfIdle(token);
    }
  }

  private playNextQueued(token: number): void {
    if (token !== this.playbackToken || this.paused || this.source) return;
    if (!this.audioContext) throw new Error('Audio context was not initialized.');

    const buffer = this.completed.get(this.nextPlayIndex);
    if (!buffer) {
      this.finishIfIdle(token);
      return;
    }
    this.completed.delete(this.nextPlayIndex);

    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(this.audioContext.destination);
    source.onended = () => {
      if (token === this.playbackToken) {
        this.clearAudio();
        this.nextPlayIndex += 1;
        this.playNextQueued(token);
      }
    };

    this.source = source;
    source.start();
  }

  private finishIfIdle(token: number): void {
    if (
      token !== this.playbackToken ||
      !this.inputClosed ||
      this.source ||
      this.sentenceQueue.length > 0 ||
      this.activeSynths > 0 ||
      this.completed.size > 0
    ) return;

    this.active = false;
    this.paused = false;
    this.currentOnEnded?.();
    this.currentOnEnded = undefined;
    this.resolveCurrent();
  }

  private peak(waveform: Float32Array): number {
    let peak = 0;
    for (const sample of waveform) {
      peak = Math.max(peak, Math.abs(sample));
    }
    return peak;
  }

  private async loadModel(onProgress?: (progress: TtsProgress) => void): Promise<void> {
    try {
      onProgress?.({ status: 'init', progress: 0, message: 'Initializing Kokoro...' });
      const { KokoroTTS } = await import('kokoro-js');
      this.tts = await KokoroTTS.from_pretrained(KOKORO_MODEL_ID, {
        dtype: 'q8',
        device: 'wasm',
        progress_callback: (data: any) => {
          onProgress?.(this.normalizeProgress(data));
        },
      });
      onProgress?.({ status: 'ready', progress: 100, message: 'Kokoro ready.' });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      onProgress?.({ status: 'error', message });
      throw error;
    }
  }

  private normalizeProgress(data: any): TtsProgress {
    if (data?.status === 'progress' || data?.status === 'progress_total') {
      return {
        status: 'download',
        progress: Math.round(data.progress ?? 0),
        file: data.file,
        message: `Downloading Kokoro... ${Math.round(data.progress ?? 0)}%`,
      };
    }

    if (data?.status === 'done') {
      return {
        status: 'loading',
        progress: 100,
        file: data.file,
        message: 'Loading Kokoro into memory...',
      };
    }

    return {
      status: data?.status === 'initiate' ? 'download' : 'loading',
      file: data?.file,
      message: 'Loading Kokoro...',
    };
  }

  private clearAudio(): void {
    this.source = null;
  }

  private takeCompleteSentences(flush: boolean): string[] {
    const sentences: string[] = [];
    let start = 0;

    for (let i = 0; i < this.pendingText.length; i++) {
      const char = this.pendingText[i];
      const isBoundary = /[.!?]/.test(char) || char === '\n';
      if (!isBoundary) continue;

      let end = i + 1;
      while (end < this.pendingText.length && /["')\]}]/.test(this.pendingText[end])) {
        end += 1;
      }

      if (!flush && end >= this.pendingText.length && char !== '\n') break;

      const sentence = this.pendingText.slice(start, end).trim();
      if (sentence) sentences.push(sentence);

      while (end < this.pendingText.length && /\s/.test(this.pendingText[end])) {
        end += 1;
      }
      start = end;
      i = end - 1;
    }

    this.pendingText = this.pendingText.slice(start);

    if (flush) {
      const tail = this.pendingText.trim();
      if (tail) sentences.push(tail);
      this.pendingText = '';
    } else if (this.pendingText.length > 360) {
      const splitAt = this.pendingText.lastIndexOf(' ', 260);
      const end = splitAt > 120 ? splitAt : 260;
      sentences.push(this.pendingText.slice(0, end).trim());
      this.pendingText = this.pendingText.slice(end).trimStart();
    }

    return sentences;
  }

  private resolveCurrent(): void {
    const resolve = this.currentResolve;
    this.currentResolve = null;
    this.currentReject = null;
    resolve?.();
  }

  private failCurrent(error: unknown): void {
    this.active = false;
    this.paused = false;
    this.currentOnEnded = undefined;
    this.pendingText = '';
    this.sentenceQueue = [];
    this.completed.clear();
    this.activeSynths = 0;
    this.clearAudio();

    const reject = this.currentReject;
    this.currentResolve = null;
    this.currentReject = null;
    reject?.(error);
  }
}
