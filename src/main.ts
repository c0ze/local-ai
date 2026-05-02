import './style.css';
import { marked } from 'marked';
import hljs from 'highlight.js';
import 'highlight.js/styles/atom-one-light.css';
import { ModelManager } from './model';
import {
  ChatThread,
  Message,
  loadThreads,
  saveThreads,
  createThread,
  deleteThread,
  updateThreadTitle,
  generateId,
  loadSettings,
  saveSettings,
  DEFAULT_GENERATION,
  type AppSettings,
  type Dtype,
  type GenerationParams,
  type ModelSpecificSettings,
} from './storage';
import { detectGpu, formatBytes, type GpuCapability } from './gpu';
import {
  MODELS,
  findModel,
  evaluateDtype,
  pickRunnableDtype,
  type DtypeStatus,
  type ModelInfo,
} from './models';
import {
  loadTestResults,
  clearTestResults,
  runCompatibilityTests,
  suggestedDtypeFromTests,
  type TestResult,
} from './tester';
import { formatWebSearchContext, webSearch, type WebSearchResponse } from './webSearch';
import {
  clearDownloadedModels,
  deleteDownloadedModel,
  estimateStorage,
  listDownloadedModels,
  type CachedModel,
} from './modelCache';
import {
  KokoroTtsManager,
  KOKORO_VOICES,
  isKokoroVoiceId,
  kokoroVoiceLabel,
  type TtsProgress,
} from './tts';
import { MoonshineSttManager, type SttProgress } from './stt';
import {
  callMcpTool,
  formatMcpToolResultPrompt,
  formatMcpToolSelectionPrompt,
  listMcpTools,
  parseMcpToolCall,
  type McpTool,
} from './mcp';

const CONTEXT_TOKEN_STOPS = [
  256,
  512,
  768,
  1024,
  1280,
  1536,
  2048,
  4096,
  8192,
  16384,
  32768,
  65536,
  131072,
];

const MAX_NEW_TOKEN_STOPS = [
  16,
  32,
  64,
  96,
  128,
  160,
  192,
  256,
  384,
  512,
  768,
  1024,
];

class ChatApp {
  private model = new ModelManager();
  private tts = new KokoroTtsManager();
  private stt = new MoonshineSttManager();
  private threads: ChatThread[] = [];
  private currentThreadId: string | null = null;
  private isGenerating = false;
  private isCompacting = false;
  private ttsMessageId: string | null = null;
  private abortController: AbortController | null = null;
  private promptHistoryIndex: number | null = null;
  private promptHistoryDraft = '';
  private ttsWarmupToken = 0;
  private settings: AppSettings = loadSettings();

  private sidebar!: HTMLElement;
  private threadList!: HTMLElement;
  private chatMessages!: HTMLElement;
  private chatContextMeter!: HTMLElement;
  private chatContextMeterFill!: HTMLElement;
  private messagesContainer: HTMLElement | null = null;
  private messageInput!: HTMLTextAreaElement;
  private sttBtn!: HTMLButtonElement;
  private ttsToggleBtn!: HTMLButtonElement;
  private sendBtn!: HTMLButtonElement;
  private newChatBtn!: HTMLButtonElement;
  private toggleSidebarBtn!: HTMLButtonElement;
  private modelLoader!: HTMLElement;
  private loadingModelDetails!: HTMLElement;
  private progressBar!: HTMLElement;
  private progressText!: HTMLElement;
  private progressFile!: HTMLElement;
  private compactContextBtn!: HTMLButtonElement;
  private exportChatBtn!: HTMLButtonElement;
  private downloadModelBtn!: HTMLButtonElement;
  private downloadModelLabel!: HTMLElement;
  private sidebarBackdrop!: HTMLElement;
  private modelCacheBtn!: HTMLButtonElement;
  private webgpuHelpBtn!: HTMLButtonElement;
  private modelStatus!: HTMLElement;
  private webgpuStatus!: HTMLElement;
  private currentThreadTitle!: HTMLElement;
  private settingsBtn!: HTMLButtonElement;
  private settingsModal!: HTMLElement;
  private webgpuHelpModal!: HTMLElement;
  private closeWebgpuHelpBtn!: HTMLButtonElement;
  private modelCacheModal!: HTMLElement;
  private closeModelCacheBtn!: HTMLButtonElement;
  private refreshModelCacheBtn!: HTMLButtonElement;
  private clearModelCacheBtn!: HTMLButtonElement;
  private modelCacheSummary!: HTMLElement;
  private modelCacheList!: HTMLElement;
  private closeSettingsBtn!: HTMLButtonElement;
  private saveSettingsBtn!: HTMLButtonElement;
  private settingsGeneralTab!: HTMLButtonElement;
  private settingsToolsTab!: HTMLButtonElement;
  private settingsPanelGeneral!: HTMLElement;
  private settingsPanelTools!: HTMLElement;
  private modelIdInput!: HTMLInputElement;
  private webSearchEnabledInput!: HTMLInputElement;
  private searxngUrlInput!: HTMLInputElement;
  private mcpBridgeUrlInput!: HTMLInputElement;
  private mcpServerUrlsInput!: HTMLTextAreaElement;
  private ttsEnabledInput!: HTMLInputElement;
  private ttsVoiceSelect!: HTMLSelectElement;
  private ttsSpeedInput!: HTMLInputElement;
  private ttsSpeedValue!: HTMLElement;
  private ttsStatus!: HTMLElement;
  private webgpuWarning!: HTMLElement;
  private webgpuWarningTitle!: HTMLElement;
  private webgpuWarningMessage!: HTMLElement;
  private webgpuWarningHelpBtn!: HTMLButtonElement;
  private dismissWebgpuWarningBtn!: HTMLButtonElement;
  private temperatureInput!: HTMLInputElement;
  private temperatureValue!: HTMLElement;
  private topPInput!: HTMLInputElement;
  private topPValue!: HTMLElement;
  private maxTokensInput!: HTMLInputElement;
  private maxTokensValue!: HTMLElement;
  private maxTokensMin!: HTMLElement;
  private maxTokensMax!: HTMLElement;
  private maxTokenStopsDataList!: HTMLDataListElement;
  private contextTokensInput!: HTMLInputElement;
  private contextTokensValue!: HTMLElement;
  private contextTokensMin!: HTMLElement;
  private contextTokensMax!: HTMLElement;
  private contextTokenStopsDataList!: HTMLDataListElement;
  private repPenaltyInput!: HTMLInputElement;
  private repPenaltyValue!: HTMLElement;
  private resetParamsBtn!: HTMLButtonElement;
  private dtypeSelect!: HTMLSelectElement;
  private modelSelect!: HTMLSelectElement;
  private modelMeta!: HTMLElement;
  private dtypeHint!: HTMLElement;
  private resourcePanel!: HTMLElement;
  private contextUsageBar!: HTMLElement;
  private contextUsageLabel!: HTMLElement;
  private contextUsageDetail!: HTMLElement;
  private vramUsageBar!: HTMLElement;
  private vramUsageLabel!: HTMLElement;
  private vramUsageDetail!: HTMLElement;
  private gpuInfoEl!: HTMLElement;
  private gpuShaderF16El!: HTMLElement;
  private runTestsBtn!: HTMLButtonElement;
  private cancelTestsBtn!: HTMLButtonElement;
  private clearTestsBtn!: HTMLButtonElement;
  private exhaustiveTestsCheckbox!: HTMLInputElement;
  private testProgressEl!: HTMLElement;
  private testResultsEl!: HTMLElement;

  private gpu: GpuCapability = {
    available: false,
    secureContext: typeof window === 'undefined' ? true : window.isSecureContext,
    fallbackAdapter: false,
    shaderF16: false,
    features: [],
  };
  private testResults: TestResult[] = loadTestResults();
  private testAbort: AbortController | null = null;
  private contextTokenStops = CONTEXT_TOKEN_STOPS.filter((value) => value <= 8192);
  private maxNewTokenStops = MAX_NEW_TOKEN_STOPS;

  private escapeHtml(value: string): string {
    return value
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  private renderMarkdown(content: string): string {
    return marked.parse(this.escapeHtml(content), { async: false }) as string;
  }

  constructor() {
    this.initDOM();
    this.initEvents();
    this.loadData();
    this.updateModelStatus();
    if (this.settings.tts.enabled) {
      void this.warmTts();
    }
    void this.checkWebGPU();
  }

  private initDOM() {
    this.sidebar = document.getElementById('sidebar')!;
    this.sidebarBackdrop = document.getElementById('sidebar-backdrop')!;
    this.threadList = document.getElementById('thread-list')!;
    this.chatMessages = document.getElementById('chat-messages')!;
    this.chatContextMeter = document.getElementById('chat-context-meter')!;
    this.chatContextMeterFill = document.getElementById('chat-context-meter-fill')!;
    this.messageInput = document.getElementById('message-input') as HTMLTextAreaElement;
    this.sendBtn = document.getElementById('send-btn') as HTMLButtonElement;
    this.newChatBtn = document.getElementById('new-chat-btn') as HTMLButtonElement;
    this.toggleSidebarBtn = document.getElementById('toggle-sidebar') as HTMLButtonElement;
    this.modelLoader = document.getElementById('model-loader')!;
    this.loadingModelDetails = document.getElementById('loading-model-details')!;
    this.progressBar = document.getElementById('progress-bar')!;
    this.progressText = document.getElementById('progress-text')!;
    this.progressFile = document.getElementById('progress-file')!;
    this.compactContextBtn = document.getElementById('compact-context-btn') as HTMLButtonElement;
    this.exportChatBtn = document.getElementById('export-chat-btn') as HTMLButtonElement;
    this.downloadModelBtn = document.getElementById('download-model-btn') as HTMLButtonElement;
    this.downloadModelLabel = document.getElementById('download-model-label')!;
    this.modelCacheBtn = document.getElementById('model-cache-btn') as HTMLButtonElement;
    this.webgpuHelpBtn = document.getElementById('webgpu-help-btn') as HTMLButtonElement;
    this.modelStatus = document.getElementById('model-status')!;
    this.webgpuStatus = document.getElementById('webgpu-status')!;
    this.currentThreadTitle = document.getElementById('current-thread-title')!;
    this.settingsBtn = document.getElementById('settings-btn') as HTMLButtonElement;
    this.settingsModal = document.getElementById('settings-modal')!;
    this.webgpuHelpModal = document.getElementById('webgpu-help-modal')!;
    this.closeWebgpuHelpBtn = document.getElementById('close-webgpu-help') as HTMLButtonElement;
    this.modelCacheModal = document.getElementById('model-cache-modal')!;
    this.closeModelCacheBtn = document.getElementById('close-model-cache') as HTMLButtonElement;
    this.refreshModelCacheBtn = document.getElementById('refresh-model-cache') as HTMLButtonElement;
    this.clearModelCacheBtn = document.getElementById('clear-model-cache') as HTMLButtonElement;
    this.modelCacheSummary = document.getElementById('model-cache-summary')!;
    this.modelCacheList = document.getElementById('model-cache-list')!;
    this.closeSettingsBtn = document.getElementById('close-settings') as HTMLButtonElement;
    this.saveSettingsBtn = document.getElementById('save-settings') as HTMLButtonElement;
    this.settingsGeneralTab = document.getElementById('settings-general-tab') as HTMLButtonElement;
    this.settingsToolsTab = document.getElementById('settings-tools-tab') as HTMLButtonElement;
    this.settingsPanelGeneral = document.getElementById('settings-panel-general')!;
    this.settingsPanelTools = document.getElementById('settings-panel-tools')!;
    this.modelIdInput = document.getElementById('model-id-input') as HTMLInputElement;
    this.webSearchEnabledInput = document.getElementById('web-search-enabled-input') as HTMLInputElement;
    this.searxngUrlInput = document.getElementById('searxng-url-input') as HTMLInputElement;
    this.mcpBridgeUrlInput = document.getElementById('mcp-bridge-url-input') as HTMLInputElement;
    this.mcpServerUrlsInput = document.getElementById('mcp-server-urls-input') as HTMLTextAreaElement;
    this.ttsEnabledInput = document.getElementById('tts-enabled-input') as HTMLInputElement;
    this.ttsVoiceSelect = document.getElementById('tts-voice-select') as HTMLSelectElement;
    this.ttsSpeedInput = document.getElementById('tts-speed-input') as HTMLInputElement;
    this.ttsSpeedValue = document.getElementById('tts-speed-value')!;
    this.ttsStatus = document.getElementById('tts-status')!;
    this.webgpuWarning = document.getElementById('webgpu-warning')!;
    this.webgpuWarningTitle = document.getElementById('webgpu-warning-title')!;
    this.webgpuWarningMessage = document.getElementById('webgpu-warning-message')!;
    this.webgpuWarningHelpBtn = document.getElementById('webgpu-warning-help-btn') as HTMLButtonElement;
    this.dismissWebgpuWarningBtn = document.getElementById('dismiss-webgpu-warning') as HTMLButtonElement;
    this.temperatureInput = document.getElementById('temperature-input') as HTMLInputElement;
    this.temperatureValue = document.getElementById('temperature-value')!;
    this.topPInput = document.getElementById('top-p-input') as HTMLInputElement;
    this.topPValue = document.getElementById('top-p-value')!;
    this.maxTokensInput = document.getElementById('max-tokens-input') as HTMLInputElement;
    this.maxTokensValue = document.getElementById('max-tokens-value')!;
    this.maxTokensMin = document.getElementById('max-tokens-min')!;
    this.maxTokensMax = document.getElementById('max-tokens-max')!;
    this.maxTokenStopsDataList = document.getElementById('max-token-stops') as HTMLDataListElement;
    this.contextTokensInput = document.getElementById('context-tokens-input') as HTMLInputElement;
    this.contextTokensValue = document.getElementById('context-tokens-value')!;
    this.contextTokensMin = document.getElementById('context-tokens-min')!;
    this.contextTokensMax = document.getElementById('context-tokens-max')!;
    this.contextTokenStopsDataList = document.getElementById('context-token-stops') as HTMLDataListElement;
    this.repPenaltyInput = document.getElementById('rep-penalty-input') as HTMLInputElement;
    this.repPenaltyValue = document.getElementById('rep-penalty-value')!;
    this.resetParamsBtn = document.getElementById('reset-params-btn') as HTMLButtonElement;
    this.dtypeSelect = document.getElementById('dtype-select') as HTMLSelectElement;
    this.modelSelect = document.getElementById('model-select') as HTMLSelectElement;
    this.modelMeta = document.getElementById('model-meta')!;
    this.dtypeHint = document.getElementById('dtype-hint')!;
    this.resourcePanel = document.getElementById('resource-panel')!;
    this.contextUsageBar = document.getElementById('context-usage-bar')!;
    this.contextUsageLabel = document.getElementById('context-usage-label')!;
    this.contextUsageDetail = document.getElementById('context-usage-detail')!;
    this.vramUsageBar = document.getElementById('vram-usage-bar')!;
    this.vramUsageLabel = document.getElementById('vram-usage-label')!;
    this.vramUsageDetail = document.getElementById('vram-usage-detail')!;
    this.gpuInfoEl = document.getElementById('gpu-info')!;
    this.gpuShaderF16El = document.getElementById('gpu-shader-f16')!;
    this.runTestsBtn = document.getElementById('run-tests-btn') as HTMLButtonElement;
    this.cancelTestsBtn = document.getElementById('cancel-tests-btn') as HTMLButtonElement;
    this.clearTestsBtn = document.getElementById('clear-tests-btn') as HTMLButtonElement;
    this.exhaustiveTestsCheckbox = document.getElementById('exhaustive-tests') as HTMLInputElement;
    this.testProgressEl = document.getElementById('test-progress')!;
    this.testResultsEl = document.getElementById('test-results')!;
    this.sttBtn = document.getElementById('stt-btn') as HTMLButtonElement;
    this.ttsToggleBtn = document.getElementById('tts-toggle-btn') as HTMLButtonElement;

    this.populateModelSelect();
    this.populateTtsVoiceSelect();
    this.updateTtsToggleButton();
    this.updateSttButton();
  }

  private populateModelSelect() {
    this.modelSelect.innerHTML = '';
    for (const m of MODELS) {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = `${m.label} — ${m.paramsB.toFixed(2)}B`;
      this.modelSelect.appendChild(opt);
    }
    const customOpt = document.createElement('option');
    customOpt.value = '__custom__';
    customOpt.textContent = '— Custom (use field below) —';
    this.modelSelect.appendChild(customOpt);
  }

  private populateTtsVoiceSelect() {
    this.ttsVoiceSelect.innerHTML = '';

    for (const voice of KOKORO_VOICES) {
      const option = document.createElement('option');
      option.value = voice.id;
      option.textContent = `${voice.name} — ${voice.language}, ${voice.gender} · ${voice.grade}`;
      this.ttsVoiceSelect.appendChild(option);
    }
  }

  private initEvents() {
    this.newChatBtn.addEventListener('click', () => this.startNewChat());
    this.toggleSidebarBtn.addEventListener('click', () => this.toggleSidebar());
    this.sidebarBackdrop.addEventListener('click', () => this.closeSidebarOnMobile());
    this.compactContextBtn.addEventListener('click', () => void this.compactCurrentThread());
    this.exportChatBtn.addEventListener('click', () => this.exportCurrentThread());
    this.downloadModelBtn.addEventListener('click', () => this.loadModel());
    this.modelCacheBtn.addEventListener('click', () => void this.openModelCache());
    this.webgpuHelpBtn.addEventListener('click', () => this.openWebgpuHelp());
    this.webgpuWarningHelpBtn.addEventListener('click', () => this.openWebgpuHelp());
    this.sttBtn.addEventListener('click', () => void this.toggleStt());
    this.ttsToggleBtn.addEventListener('click', () => this.toggleTts());
    this.sendBtn.addEventListener('click', () => this.sendMessage());
    this.settingsBtn.addEventListener('click', () => this.openSettings());
    this.closeSettingsBtn.addEventListener('click', () => this.closeSettings());
    this.closeWebgpuHelpBtn.addEventListener('click', () => this.closeWebgpuHelp());
    this.closeModelCacheBtn.addEventListener('click', () => this.closeModelCache());
    this.refreshModelCacheBtn.addEventListener('click', () => void this.refreshModelCache());
    this.clearModelCacheBtn.addEventListener('click', () => void this.clearModelCache());
    this.saveSettingsBtn.addEventListener('click', () => void this.saveSettingsHandler());
    this.settingsGeneralTab.addEventListener('click', () => this.showSettingsTab('general'));
    this.settingsToolsTab.addEventListener('click', () => this.showSettingsTab('tools'));
    this.searxngUrlInput.addEventListener('input', () => {
      if (this.searxngUrlInput.value.trim()) {
        this.webSearchEnabledInput.checked = true;
      }
    });

    this.modelSelect.addEventListener('change', () => this.onModelSelectionChanged());
    this.modelIdInput.addEventListener('input', () => this.onCustomModelIdInput());
    this.dtypeSelect.addEventListener('change', () => {
      this.refreshDtypeHint();
      this.refreshResourceBars();
    });

    this.runTestsBtn.addEventListener('click', () => void this.runTests());
    this.cancelTestsBtn.addEventListener('click', () => this.cancelTests());
    this.clearTestsBtn.addEventListener('click', () => this.clearTests());

    this.settingsModal.addEventListener('click', (e) => {
      if (e.target === this.settingsModal) this.closeSettings();
    });
    this.webgpuHelpModal.addEventListener('click', (e) => {
      if (e.target === this.webgpuHelpModal) this.closeWebgpuHelp();
    });
    this.modelCacheModal.addEventListener('click', (e) => {
      if (e.target === this.modelCacheModal) this.closeModelCache();
    });

    this.messageInput.addEventListener('keydown', (e) => {
      if (this.handlePromptHistoryNavigation(e)) return;

      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });

    this.dismissWebgpuWarningBtn.addEventListener('click', () => {
      this.webgpuWarning.classList.add('hidden');
    });

    this.messageInput.addEventListener('input', () => {
      this.promptHistoryIndex = null;
      this.promptHistoryDraft = '';
      this.resizeMessageInput();
    });

    this.temperatureInput.addEventListener('input', () => {
      this.temperatureValue.textContent = Number(this.temperatureInput.value).toFixed(2);
    });
    this.topPInput.addEventListener('input', () => {
      this.topPValue.textContent = Number(this.topPInput.value).toFixed(2);
    });
    this.repPenaltyInput.addEventListener('input', () => {
      this.repPenaltyValue.textContent = Number(this.repPenaltyInput.value).toFixed(2);
    });
    this.maxTokensInput.addEventListener('input', () => {
      this.updateMaxTokenSliderLabels();
      this.refreshResourceBars();
    });
    this.contextTokensInput.addEventListener('input', () => {
      this.updateContextSliderLabels();
      this.refreshResourceBars();
    });
    this.ttsSpeedInput.addEventListener('input', () => {
      this.ttsSpeedValue.textContent = Number(this.ttsSpeedInput.value).toFixed(2);
    });
    this.resetParamsBtn.addEventListener('click', () => {
      const info = findModel(this.modelSelect.value);
      this.applyParamsToInputs(info?.generationDefaults ?? DEFAULT_GENERATION);
    });
  }

  private contextStopsFor(info: ModelInfo | null): number[] {
    const maxContext = info?.contextWindowTokens ?? 8192;
    const stops = CONTEXT_TOKEN_STOPS.filter((value) => value <= maxContext);
    if (!stops.includes(maxContext)) stops.push(maxContext);
    return stops.sort((a, b) => a - b);
  }

  private maxNewTokenStopsFor(info: ModelInfo | null): number[] {
    const maxByContext = Math.max(16, Math.min(1024, (info?.contextWindowTokens ?? 8192) - 256));
    const stops = MAX_NEW_TOKEN_STOPS.filter((value) => value <= maxByContext);
    if (!stops.includes(maxByContext)) stops.push(maxByContext);
    return stops.sort((a, b) => a - b);
  }

  private nearestStopIndex(stops: number[], value: number): number {
    let nearest = 0;
    let nearestDistance = Number.POSITIVE_INFINITY;
    for (let i = 0; i < stops.length; i++) {
      const distance = Math.abs(stops[i] - value);
      if (distance < nearestDistance) {
        nearest = i;
        nearestDistance = distance;
      }
    }
    return nearest;
  }

  private nearestContextStopIndex(tokens: number): number {
    return this.nearestStopIndex(this.contextTokenStops, tokens);
  }

  private contextTokensFromSlider(): number {
    const index = Number.parseInt(this.contextTokensInput.value, 10);
    const safeIndex = Number.isFinite(index)
      ? Math.min(this.contextTokenStops.length - 1, Math.max(0, index))
      : 0;
    return this.contextTokenStops[safeIndex] ?? DEFAULT_GENERATION.contextTokens;
  }

  private maxNewTokensFromSlider(): number {
    const index = Number.parseInt(this.maxTokensInput.value, 10);
    const safeIndex = Number.isFinite(index)
      ? Math.min(this.maxNewTokenStops.length - 1, Math.max(0, index))
      : 0;
    return this.maxNewTokenStops[safeIndex] ?? DEFAULT_GENERATION.maxNewTokens;
  }

  private updateMaxTokenSliderLabels() {
    const value = this.maxNewTokensFromSlider();
    const first = this.maxNewTokenStops[0] ?? 16;
    const last = this.maxNewTokenStops[this.maxNewTokenStops.length - 1] ?? 1024;
    this.maxTokensValue.textContent = this.formatCount(value);
    this.maxTokensMin.textContent = this.formatCount(first);
    this.maxTokensMax.textContent = this.formatCount(last);
  }

  private updateContextSliderLabels() {
    const value = this.contextTokensFromSlider();
    const first = this.contextTokenStops[0] ?? 256;
    const last = this.contextTokenStops[this.contextTokenStops.length - 1] ?? 8192;
    this.contextTokensValue.textContent = this.formatCount(value);
    this.contextTokensMin.textContent = this.formatCount(first);
    this.contextTokensMax.textContent = this.formatCount(last);
  }

  private setMaxTokenSliderValue(tokens: number) {
    const info = findModel(this.modelSelect.value) ?? null;
    this.maxNewTokenStops = this.maxNewTokenStopsFor(info);
    this.maxTokensInput.min = '0';
    this.maxTokensInput.max = String(Math.max(0, this.maxNewTokenStops.length - 1));
    this.maxTokensInput.step = '1';
    this.maxTokensInput.value = String(this.nearestStopIndex(this.maxNewTokenStops, tokens));

    this.maxTokenStopsDataList.innerHTML = '';
    for (let i = 0; i < this.maxNewTokenStops.length; i++) {
      const option = document.createElement('option');
      option.value = String(i);
      option.label = this.formatCount(this.maxNewTokenStops[i]);
      this.maxTokenStopsDataList.appendChild(option);
    }

    this.updateMaxTokenSliderLabels();
  }

  private setContextSliderValue(tokens: number) {
    const info = findModel(this.modelSelect.value) ?? null;
    this.contextTokenStops = this.contextStopsFor(info);
    this.contextTokensInput.min = '0';
    this.contextTokensInput.max = String(Math.max(0, this.contextTokenStops.length - 1));
    this.contextTokensInput.step = '1';
    this.contextTokensInput.value = String(this.nearestContextStopIndex(tokens));

    this.contextTokenStopsDataList.innerHTML = '';
    for (let i = 0; i < this.contextTokenStops.length; i++) {
      const option = document.createElement('option');
      option.value = String(i);
      option.label = this.formatCount(this.contextTokenStops[i]);
      this.contextTokenStopsDataList.appendChild(option);
    }

    this.updateContextSliderLabels();
  }

  private applyParamsToInputs(params: Partial<GenerationParams>) {
    const merged = { ...DEFAULT_GENERATION, ...params };
    this.temperatureInput.value = String(merged.temperature);
    this.temperatureValue.textContent = merged.temperature.toFixed(2);
    this.topPInput.value = String(merged.topP);
    this.topPValue.textContent = merged.topP.toFixed(2);
    this.setMaxTokenSliderValue(merged.maxNewTokens);
    this.setContextSliderValue(merged.contextTokens);
    this.repPenaltyInput.value = String(merged.repetitionPenalty);
    this.repPenaltyValue.textContent = merged.repetitionPenalty.toFixed(2);
    this.refreshResourceBars();
  }

  private savedSettingsForModel(modelId: string): ModelSpecificSettings | null {
    return this.settings.modelSettings[modelId.trim()] ?? null;
  }

  private defaultSettingsForModel(info: ModelInfo): ModelSpecificSettings {
    return {
      dtype: pickRunnableDtype(info, this.gpu) ?? 'auto',
      generation: { ...DEFAULT_GENERATION, ...info.generationDefaults },
    };
  }

  private applyDtypeToSelect(dtype: Dtype, info: ModelInfo | null) {
    if (this.dtypeSelect.querySelector(`option[value="${dtype}"]:not([disabled])`)) {
      this.dtypeSelect.value = dtype;
    } else if (info) {
      const recommended = pickRunnableDtype(info, this.gpu);
      this.dtypeSelect.value = recommended && this.dtypeSelect.querySelector(`option[value="${recommended}"]:not([disabled])`)
        ? recommended
        : 'auto';
    } else {
      this.dtypeSelect.value = 'auto';
    }
    this.refreshDtypeHint();
  }

  private applyModelSettingsToInputs(modelId: string, info: ModelInfo | null, preferred?: ModelSpecificSettings) {
    const settings = preferred ?? this.savedSettingsForModel(modelId) ?? (info ? this.defaultSettingsForModel(info) : null);
    if (!settings) {
      this.applyParamsToInputs(DEFAULT_GENERATION);
      this.applyDtypeToSelect('auto', info);
      return;
    }

    this.applyParamsToInputs(settings.generation);
    this.applyDtypeToSelect(settings.dtype, info);
  }

  private updateSttButton(state: 'idle' | 'recording' | 'busy' = 'idle') {
    const disabled = this.isGenerating || this.isCompacting || state === 'busy';
    this.sttBtn.disabled = disabled;
    this.sttBtn.title = state === 'recording'
      ? 'Stop recording and transcribe'
      : state === 'busy'
        ? 'Transcribing microphone audio'
        : 'Dictate message';
    this.sttBtn.setAttribute('aria-label', this.sttBtn.title);
    this.sttBtn.className = state === 'recording'
      ? 'flex-shrink-0 p-1.5 rounded-md bg-cream-100 text-accent hover:text-accent-hover transition-colors'
      : 'flex-shrink-0 p-1.5 rounded-md text-ink-400 hover:text-accent disabled:text-ink-400/50 disabled:cursor-not-allowed transition-colors';
  }

  private formatSttProgress(progress: SttProgress): string {
    return progress.message;
  }

  private resizeMessageInput() {
    this.messageInput.style.height = 'auto';
    this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 160) + 'px';
  }

  private setMessageInputValue(value: string) {
    this.messageInput.value = value;
    this.messageInput.focus();
    this.resizeMessageInput();
    this.messageInput.setSelectionRange(value.length, value.length);
  }

  private insertTranscript(text: string) {
    const transcript = text.trim();
    if (!transcript) return;

    const current = this.messageInput.value;
    const needsSpace = current.trim().length > 0 && !/\s$/.test(current);
    this.promptHistoryIndex = null;
    this.promptHistoryDraft = '';
    this.setMessageInputValue(`${current}${needsSpace ? ' ' : ''}${transcript}`);
  }

  private async toggleStt() {
    if (this.isGenerating || this.isCompacting) return;

    if (this.stt.recording) {
      this.updateSttButton('busy');
      try {
        const transcript = await this.stt.stopAndTranscribe((progress) => {
          this.setTtsStatus(this.formatSttProgress(progress));
        });
        this.insertTranscript(transcript);
        this.setTtsStatus(transcript ? 'Dictation inserted.' : 'No speech detected.');
      } catch (error) {
        this.setTtsStatus(`Moonshine failed: ${this.getErrorMessage(error)}`);
      } finally {
        this.updateSttButton();
      }
      return;
    }

    try {
      await this.stt.startRecording((progress) => {
        this.setTtsStatus(this.formatSttProgress(progress));
      });
      this.updateSttButton('recording');
    } catch (error) {
      this.setTtsStatus(`Microphone failed: ${this.getErrorMessage(error)}`);
      this.updateSttButton();
    }
  }

  private toggleTts() {
    const nextEnabled = !this.settings.tts.enabled;
    this.settings = {
      ...this.settings,
      tts: {
        ...this.settings.tts,
        enabled: nextEnabled,
      },
    };
    saveSettings(this.settings);
    this.settings = loadSettings();
    if (!nextEnabled) {
      this.stopTts();
      this.setTtsStatus('Text to speech is off.');
    } else {
      this.setTtsStatus(`Text to speech on: ${kokoroVoiceLabel(this.settings.tts.voice)}.`);
      void this.warmTts();
    }
    this.updateTtsToggleButton();
    this.updateTtsPlaybackButtons();
  }

  private updateTtsToggleButton() {
    const active = this.settings.tts.enabled;
    this.ttsToggleBtn.setAttribute('aria-pressed', String(active));
    this.ttsToggleBtn.title = active
      ? 'Text to speech is on. Use Play on assistant responses.'
      : 'Turn on text to speech.';
    this.ttsToggleBtn.className = active
      ? 'flex-shrink-0 p-1.5 rounded-md bg-cream-100 text-accent hover:text-accent-hover transition-colors'
      : 'flex-shrink-0 p-1.5 rounded-md text-ink-400 hover:text-accent transition-colors';
  }

  private setTtsStatus(message: string) {
    this.ttsStatus.textContent = message || 'Runs locally via WebGPU. Chats stored in your browser.';
    this.ttsStatus.className = message.includes('failed') || message.includes('blocked') || message.includes('Error')
      ? 'min-h-4 text-[11px] text-accent text-center mt-3'
      : 'min-h-4 text-[11px] text-ink-400 text-center mt-3';
  }

  private formatTtsProgress(progress: TtsProgress): string {
    if (progress.message) return progress.message;
    if (progress.status === 'download') return `Downloading Kokoro... ${progress.progress ?? 0}%`;
    if (progress.status === 'loading') return 'Loading Kokoro...';
    if (progress.status === 'ready') return 'Kokoro ready.';
    if (progress.status === 'error') return `Kokoro failed: ${progress.message ?? 'unknown error'}`;
    return 'Initializing Kokoro...';
  }

  private async warmTts() {
    if (!this.settings.tts.enabled) return;

    const token = ++this.ttsWarmupToken;
    try {
      await this.tts.warm({
        voice: this.settings.tts.voice,
        speed: this.settings.tts.speed,
        onProgress: (progress) => {
          if (token !== this.ttsWarmupToken || !this.settings.tts.enabled) return;
          this.setTtsStatus(this.formatTtsProgress(progress));
        },
      });
      if (token === this.ttsWarmupToken && this.settings.tts.enabled && !this.tts.isActive) {
        this.setTtsStatus(`Kokoro ready: ${kokoroVoiceLabel(this.settings.tts.voice)}.`);
      }
    } catch (error) {
      if (token !== this.ttsWarmupToken || !this.settings.tts.enabled) return;
      this.setTtsStatus(`Kokoro failed: ${this.getErrorMessage(error)}`);
    }
  }

  private async checkWebGPU() {
    this.gpu = await detectGpu();
    if (this.gpu.available) {
      if (this.gpu.fallbackAdapter) {
        this.webgpuStatus.textContent = 'WebGPU: fallback adapter';
        this.webgpuStatus.className = 'text-accent';
        this.webgpuWarningTitle.textContent = 'Chrome is using the WebGPU software fallback';
        this.webgpuWarningMessage.textContent =
          'The app can run, but SwiftShader usually caps buffers at 1 GB and is much slower than your real GPU. On Linux / AMD, enable Unsafe WebGPU and Vulkan in chrome://flags. If video sites show a white player, try ANGLE OpenGL and Zero-copy rasterizer off before disabling only Hardware-accelerated video decode.';
        this.webgpuWarning.classList.remove('hidden');
      } else {
        this.webgpuStatus.textContent = 'WebGPU: ready';
        this.webgpuStatus.className = 'text-ink-500';
        this.webgpuWarning.classList.add('hidden');
      }
    } else {
      this.webgpuStatus.textContent = 'WebGPU: unavailable';
      this.webgpuStatus.className = 'text-accent';
      this.webgpuWarningTitle.textContent = 'WebGPU is not available in this browser';
      this.webgpuWarningMessage.textContent = this.gpu.secureContext
        ? "This app runs AI models locally via WebGPU. Without it, model inference won't work."
        : 'This page is not a secure context. Open the https:// URL and clear site data if Chrome still marks it insecure.';
      this.webgpuWarning.classList.remove('hidden');
    }
    this.renderHardware();
    if (!this.settingsModal.classList.contains('hidden')) {
      const info = findModel(this.modelSelect.value);
      this.populateDtypeOptions(info ?? null);
      this.refreshResourceBars();
    }
    this.updateModelStatus();
  }

  private renderHardware() {
    const {
      available,
      secureContext,
      fallbackAdapter,
      vendor,
      architecture,
      description,
      shaderF16,
      maxBufferBytes,
      maxStorageBufferBindingBytes,
    } = this.gpu;
    if (!available) {
      this.gpuShaderF16El.textContent = '';
      this.gpuInfoEl.textContent = secureContext
        ? 'WebGPU not available in this browser.'
        : 'This page is not a secure context. Open the https:// URL and clear site data if Chrome still marks it insecure.';
      return;
    }
    this.gpuShaderF16El.textContent = shaderF16 ? 'shader-f16 ✓' : 'shader-f16 ✗';
    this.gpuShaderF16El.className = `text-[11px] font-mono ${shaderF16 ? 'text-ink-500' : 'text-accent'}`;

    const labelParts = [vendor, architecture, description].filter(Boolean).join(' · ');
    const adapterLine = document.createElement('div');
    adapterLine.className = 'truncate';
    adapterLine.textContent = labelParts || 'Adapter info hidden by browser';

    const securityLine = document.createElement('div');
    securityLine.className = secureContext ? 'text-ink-500' : 'text-accent';
    securityLine.textContent = secureContext ? 'Secure context: yes' : 'Secure context: no';

    const fallbackLine = document.createElement('div');
    fallbackLine.className = fallbackAdapter ? 'text-accent' : 'text-ink-500';
    fallbackLine.textContent = fallbackAdapter
      ? 'Software fallback adapter: yes (SwiftShader)'
      : 'Software fallback adapter: no';

    const bufferLine = document.createElement('div');
    const bufferValue = document.createElement('span');
    bufferValue.className = 'font-mono text-ink-700';
    bufferValue.textContent = formatBytes(maxBufferBytes);
    bufferLine.append('Max buffer: ', bufferValue);

    const storageLine = document.createElement('div');
    const storageValue = document.createElement('span');
    storageValue.className = 'font-mono text-ink-700';
    storageValue.textContent = formatBytes(maxStorageBufferBindingBytes);
    storageLine.append('Max storage binding: ', storageValue);

    this.gpuInfoEl.replaceChildren(securityLine, adapterLine, fallbackLine, bufferLine, storageLine);
  }

  private openWebgpuHelp() {
    this.webgpuHelpModal.classList.remove('hidden');
  }

  private closeWebgpuHelp() {
    this.webgpuHelpModal.classList.add('hidden');
  }

  private async openModelCache() {
    this.modelCacheModal.classList.remove('hidden');
    await this.refreshModelCache();
  }

  private closeModelCache() {
    this.modelCacheModal.classList.add('hidden');
  }

  private formatStorageBytes(bytes?: number): string {
    if (bytes === undefined) return 'unknown';
    if (bytes === 0) return '0 MB';
    return formatBytes(bytes);
  }

  private formatModelCacheSize(model: CachedModel): string {
    if (model.totalBytes === 0 && model.hasUnknownBytes) return 'unknown';
    const knownSize = this.formatStorageBytes(model.totalBytes);
    return model.hasUnknownBytes ? `${knownSize}+` : knownSize;
  }

  private async refreshModelCache() {
    this.refreshModelCacheBtn.disabled = true;
    this.modelCacheSummary.textContent = 'Scanning browser cache...';
    this.modelCacheList.textContent = '';

    try {
      const [models, storage] = await Promise.all([
        listDownloadedModels(),
        estimateStorage(),
      ]);
      this.renderModelCacheList(models);

      const modelBytes = models.reduce((sum, model) => sum + model.totalBytes, 0);
      const modelCount = models.length === 1 ? '1 model' : `${models.length} models`;
      const modelSummary = `${modelCount} · ${this.formatStorageBytes(modelBytes)} cached model data`;
      const storageSummary = storage?.usageBytes
        ? ` · origin storage ${this.formatStorageBytes(storage.usageBytes)} / ${this.formatStorageBytes(storage.quotaBytes)}`
        : '';
      this.modelCacheSummary.textContent = `${modelSummary}${storageSummary}`;
      this.clearModelCacheBtn.disabled = models.length === 0;
      this.clearModelCacheBtn.className = models.length === 0
        ? 'text-[12px] text-ink-400 cursor-not-allowed transition-colors'
        : 'text-[12px] text-accent hover:text-accent-hover transition-colors';
    } catch (error) {
      console.error('Failed to inspect model cache:', error);
      this.modelCacheSummary.textContent = 'Could not read the browser model cache.';
      this.modelCacheList.textContent = '';
    } finally {
      this.refreshModelCacheBtn.disabled = false;
    }
  }

  private renderModelCacheList(models: CachedModel[]) {
    if (models.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'border border-cream-200 rounded-lg bg-cream-100/40 p-4 text-sm text-ink-500';
      empty.textContent = 'No downloaded Hugging Face model files found in the browser cache.';
      this.modelCacheList.replaceChildren(empty);
      return;
    }

    const cards = models.map((model) => {
      const info = findModel(model.modelId);
      const card = document.createElement('div');
      card.className = 'border border-cream-200 rounded-lg bg-cream-100/40 p-4';

      const header = document.createElement('div');
      header.className = 'flex items-start justify-between gap-4';

      const titleWrap = document.createElement('div');
      titleWrap.className = 'min-w-0';

      const title = document.createElement('div');
      title.className = 'text-sm font-medium text-ink-800 truncate';
      title.textContent = info?.label ?? model.modelId;

      const meta = document.createElement('div');
      meta.className = 'text-[12px] text-ink-500 mt-0.5 truncate';
      meta.textContent = `${model.modelId} · ${model.files.length} files · ${this.formatModelCacheSize(model)}`;

      titleWrap.append(title, meta);

      const deleteBtn = document.createElement('button');
      deleteBtn.type = 'button';
      deleteBtn.className = 'flex-shrink-0 text-[12px] text-accent hover:text-accent-hover transition-colors';
      deleteBtn.textContent = 'Delete';
      deleteBtn.addEventListener('click', () => void this.deleteCachedModel(model.modelId));

      const loadBtn = document.createElement('button');
      loadBtn.type = 'button';
      loadBtn.className = 'flex-shrink-0 text-[12px] text-ink-600 hover:text-ink-900 transition-colors';
      const loadedHere = this.model.loaded && this.model.name === model.modelId;
      loadBtn.textContent = loadedHere ? 'Loaded' : this.model.loading ? 'Loading' : 'Load';
      loadBtn.disabled = loadedHere || this.model.loading;
      loadBtn.addEventListener('click', () => void this.loadCachedModel(model.modelId));

      const actions = document.createElement('div');
      actions.className = 'flex flex-shrink-0 items-center gap-3';
      actions.append(loadBtn, deleteBtn);

      header.append(titleWrap, actions);

      const details = document.createElement('details');
      details.className = 'mt-3';

      const summary = document.createElement('summary');
      summary.className = 'text-[12px] text-ink-500 hover:text-ink-700 cursor-pointer select-none';
      summary.textContent = 'Files';

      const fileList = document.createElement('div');
      fileList.className = 'mt-2 space-y-1';

      for (const file of model.files) {
        const row = document.createElement('div');
        row.className = 'flex items-baseline justify-between gap-3 text-[11px] text-ink-500';

        const name = document.createElement('span');
        name.className = 'font-mono truncate';
        name.textContent = file.file;

        const size = document.createElement('span');
        size.className = 'font-mono flex-shrink-0 text-ink-400';
        size.textContent = this.formatStorageBytes(file.sizeBytes);

        row.append(name, size);
        fileList.appendChild(row);
      }

      details.append(summary, fileList);
      card.append(header, details);
      return card;
    });

    this.modelCacheList.replaceChildren(...cards);
  }

  private async deleteCachedModel(modelId: string) {
    const label = findModel(modelId)?.label ?? modelId;
    if (!window.confirm(`Delete downloaded files for ${label}? The model will download again next time you load it.`)) return;

    if (this.model.loaded && this.model.name === modelId) {
      await this.model.unload();
      this.updateModelStatus();
    }

    this.modelCacheSummary.textContent = `Deleting ${label}...`;
    await deleteDownloadedModel(modelId);
    await this.refreshModelCache();
  }

  private async loadCachedModel(modelId: string) {
    if (this.model.loading) return;
    const info = findModel(modelId);
    const previous = this.settings.modelSettings[modelId];
    const dtype = previous?.dtype ?? (info ? 'auto' : this.settings.dtype);
    const generation = previous?.generation ?? (info ? this.defaultSettingsForModel(info).generation : this.settings.generation);

    saveSettings({
      ...this.settings,
      modelId,
      dtype,
      generation,
      modelSettings: {
        ...this.settings.modelSettings,
        [modelId]: {
          dtype,
          generation,
        },
      },
    });
    this.settings = loadSettings();

    if (this.model.loaded || this.model.loading) {
      await this.model.unload();
    }

    this.updateModelStatus();
    this.closeModelCache();
    void this.loadModel();
  }

  private async clearModelCache() {
    if (!window.confirm('Delete all downloaded model files? Models will download again when loaded.')) return;

    if (this.model.loaded) {
      await this.model.unload();
      this.updateModelStatus();
    }

    this.clearModelCacheBtn.disabled = true;
    this.modelCacheSummary.textContent = 'Deleting downloaded models...';
    await clearDownloadedModels();
    await this.refreshModelCache();
  }

  private formatCount(value: number): string {
    return Math.round(value).toLocaleString('en-US');
  }

  private displayModelName(modelId: string): string {
    return findModel(modelId)?.label ?? modelId.split('/').pop() ?? modelId;
  }

  private estimateTextTokens(text: string): number {
    return Math.max(1, Math.ceil(text.length / 4));
  }

  private estimateMessageTokens(message: Message): number {
    return 6 + this.estimateTextTokens(message.content);
  }

  private estimateThreadTokens(messages: Message[]): number {
    if (messages.length === 0) return 0;
    return 8 + messages.reduce((total, message) => total + this.estimateMessageTokens(message), 0);
  }

  private splitMessagesForCompaction(messages: Message[]): { toCompact: Message[]; toKeep: Message[] } {
    if (messages.length <= 4) return { toCompact: [], toKeep: messages };

    const keepBudget = Math.max(512, Math.floor(this.settings.generation.contextTokens * 0.35));
    const toKeep: Message[] = [];
    let used = 8;

    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i];
      const cost = this.estimateMessageTokens(message);
      if (toKeep.length >= 4 && used + cost > keepBudget) break;
      toKeep.unshift(message);
      used += cost;
    }

    const keepCount = Math.min(messages.length - 2, Math.max(2, toKeep.length));
    return {
      toCompact: messages.slice(0, messages.length - keepCount),
      toKeep: messages.slice(messages.length - keepCount),
    };
  }

  private formatCompactionTranscript(messages: Message[]): string {
    const transcript = messages
      .map((message, index) => {
        const role = message.role === 'user' ? 'User' : 'Assistant';
        return `Turn ${index + 1} (${role}):\n${message.content.trim() || '[empty]'}`;
      })
      .join('\n\n---\n\n');

    const maxChars = Math.max(4096, this.settings.generation.contextTokens * 4);
    if (transcript.length <= maxChars) return transcript;

    const marker = '\n\n[...middle omitted before summarization...]\n\n';
    const sideChars = Math.max(1536, Math.floor((maxChars - marker.length) / 2));
    return `${transcript.slice(0, sideChars)}${marker}${transcript.slice(-sideChars)}`;
  }

  private updateCompactContextButton() {
    const thread = this.getCurrentThread();
    const canCompact = Boolean(thread && thread.messages.length > 4 && !this.isGenerating && !this.isCompacting && !this.model.loading);
    this.compactContextBtn.disabled = !canCompact;
    this.compactContextBtn.textContent = this.isCompacting ? 'Compacting...' : 'Compact';
  }

  private updateExportButton() {
    const thread = this.getCurrentThread();
    const canExport = Boolean(thread && thread.messages.length > 0);
    this.exportChatBtn.disabled = !canExport;
    this.exportChatBtn.title = canExport ? 'Export this chat as Markdown' : 'No messages to export yet';
  }

  private updateConversationContextMeter() {
    const thread = this.getCurrentThread();
    const maxTokens = Math.max(1, this.settings.generation.contextTokens);
    const usedTokens = thread ? this.estimateThreadTokens(thread.messages) : 0;
    const ratio = usedTokens / maxTokens;
    const percent = Math.min(100, Math.max(0, ratio * 100));
    const hue = Math.max(0, Math.round(132 - Math.min(1, ratio) * 132));
    const title = `Context estimate: ${this.formatCount(usedTokens)} / ${this.formatCount(maxTokens)} tokens`;

    this.chatContextMeterFill.style.height = `${percent}%`;
    this.chatContextMeterFill.style.backgroundColor = `hsl(${hue} 62% 40%)`;
    this.chatContextMeter.setAttribute('aria-valuemax', String(maxTokens));
    this.chatContextMeter.setAttribute('aria-valuenow', String(Math.min(usedTokens, maxTokens)));
    this.chatContextMeter.title = title;
    this.updateCompactContextButton();
    this.updateExportButton();
  }

  private setMeter(
    bar: HTMLElement,
    label: HTMLElement,
    current: number,
    max: number | undefined,
    currentText: string,
    maxText: string,
  ) {
    const hasMax = max !== undefined && max > 0;
    const percent = hasMax ? Math.min(100, Math.max(0, (current / max) * 100)) : 0;
    const overLimit = hasMax && current > max;

    bar.style.width = `${percent}%`;
    bar.className = `h-full rounded-full transition-all duration-300 ${overLimit ? 'bg-accent' : 'bg-ink-600'}`;
    label.textContent = hasMax ? `${currentText} / ${maxText}` : currentText;
    label.className = `text-[12px] font-mono ${overLimit ? 'text-accent' : 'text-ink-700'}`;
  }

  private selectedDtypeForResource(info: ModelInfo): Exclude<Dtype, 'auto'> | null {
    const selected = this.dtypeSelect.value as Dtype;
    if (selected !== 'auto' && info.supportedDtypes.includes(selected)) return selected;
    return pickRunnableDtype(info, this.gpu) ?? info.recommendedDtype;
  }

  private refreshResourceBars() {
    const info = findModel(this.modelSelect.value);
    if (!info) {
      this.resourcePanel.classList.add('hidden');
      return;
    }

    this.resourcePanel.classList.remove('hidden');

    const promptTokens = this.contextTokensFromSlider();
    const outputTokens = this.maxNewTokensFromSlider();
    const safePromptTokens = Number.isFinite(promptTokens) ? promptTokens : info.generationDefaults.contextTokens;
    const safeOutputTokens = Number.isFinite(outputTokens) ? outputTokens : info.generationDefaults.maxNewTokens;
    const totalContextTokens = Math.max(0, safePromptTokens + safeOutputTokens);

    this.setMeter(
      this.contextUsageBar,
      this.contextUsageLabel,
      totalContextTokens,
      info.contextWindowTokens,
      this.formatCount(totalContextTokens),
      this.formatCount(info.contextWindowTokens),
    );
    this.contextUsageDetail.textContent =
      `${this.formatCount(safePromptTokens)} prompt + ${this.formatCount(safeOutputTokens)} output tokens`;

    const dtype = this.selectedDtypeForResource(info);
    if (!dtype) {
      this.setMeter(this.vramUsageBar, this.vramUsageLabel, 0, undefined, 'unknown', '');
      this.vramUsageDetail.textContent = 'No compatible dtype is available for this model on the detected adapter.';
      return;
    }

    const weightsMb = info.approxWeightsMb[dtype];
    const bufferMb = info.maxBufferRequirementMb?.[dtype] ?? weightsMb;
    const maxBufferMb = this.gpu.maxBufferBytes ? this.gpu.maxBufferBytes / 1024 / 1024 : undefined;

    if (bufferMb) {
      this.setMeter(
        this.vramUsageBar,
        this.vramUsageLabel,
        bufferMb,
        maxBufferMb,
        this.formatMb(bufferMb),
        maxBufferMb ? this.formatMb(maxBufferMb) : '',
      );
    } else {
      this.setMeter(this.vramUsageBar, this.vramUsageLabel, 0, maxBufferMb, 'unknown', maxBufferMb ? this.formatMb(maxBufferMb) : '');
    }

    const weightsText = weightsMb ? `~${this.formatMb(weightsMb)} ${dtype} weights` : `${dtype} weights unknown`;
    const bufferText = bufferMb ? `largest buffer ~${this.formatMb(bufferMb)}` : 'largest buffer unknown';
    const maxText = maxBufferMb ? `${this.formatMb(maxBufferMb)} WebGPU max buffer` : 'WebGPU max buffer not reported';
    this.vramUsageDetail.textContent = `${weightsText}; ${bufferText} / ${maxText}`;
  }

  private onModelSelectionChanged() {
    const value = this.modelSelect.value;
    if (value === '__custom__') {
      this.modelIdInput.focus();
      this.modelMeta.textContent = 'Using the custom ID below.';
      this.populateDtypeOptions(null);
      this.refreshResourceBars();
      return;
    }
    this.modelIdInput.value = '';
    const info = findModel(value);
    this.populateDtypeOptions(info ?? null);
    if (info) {
      this.applyModelSettingsToInputs(value, info);
      this.renderModelMeta(info);
      this.refreshResourceBars();
    }
  }

  private onCustomModelIdInput() {
    const id = this.modelIdInput.value.trim();
    if (!id) return;
    this.modelSelect.value = '__custom__';
    this.modelMeta.textContent = 'Custom model — dtype auto-detected at load.';
    this.populateDtypeOptions(null);
    this.refreshResourceBars();
  }

  private renderModelMeta(info: ReturnType<typeof findModel>) {
    if (!info) {
      this.modelMeta.textContent = '';
      return;
    }
    const passing = this.testResults
      .filter((r) => r.modelId === info.id && r.status === 'ok')
      .map((r) => r.dtype);
    const tag = passing.length ? ` · tested ✓ ${passing.join(', ')}` : '';
    const defaults = `default ${info.generationDefaults.contextTokens} ctx / ${info.generationDefaults.maxNewTokens} out`;
    const window = `max ${this.formatCount(info.contextWindowTokens)} ctx`;
    this.modelMeta.textContent = `${info.architecture} · ${info.paramsB.toFixed(2)}B params · ${window} · ${defaults}${tag}${info.notes ? ` · ${info.notes}` : ''}`;
  }

  private formatMb(value?: number): string {
    if (!value) return '';
    return value < 1024 ? `${value} MB` : `${(value / 1024).toFixed(2)} GB`;
  }

  private dtypeStatusMessage(
    info: ModelInfo,
    dtype: Exclude<Dtype, 'auto'>,
    status: DtypeStatus,
  ): string {
    const sizeMb = info.approxWeightsMb[dtype];
    const sizeStr = this.formatMb(sizeMb);
    switch (status) {
      case 'ok':
        return sizeStr
          ? `${sizeStr} of model weights. Compatible with detected hard WebGPU limits.`
          : 'Compatible with detected hard WebGPU limits.';
      case 'unsupported_by_model':
        return 'This model does not ship in this dtype.';
      case 'needs_shader_f16':
        return 'Your GPU does not expose the shader-f16 feature.';
      case 'too_large': {
        const bufferMb = info.maxBufferRequirementMb?.[dtype];
        const bufferStr = this.formatMb(bufferMb);
        const maxBuffer = formatBytes(this.gpu.maxBufferBytes);
        if (bufferStr) {
          return `Needs a single GPU buffer around ${bufferStr}; this adapter allows ${maxBuffer}.`;
        }
        return `${sizeStr} exceeds this adapter's max buffer size (${maxBuffer}).`;
      }
    }
  }

  private populateDtypeOptions(info: ReturnType<typeof findModel> | null) {
    const previous = this.dtypeSelect.value;
    this.dtypeSelect.innerHTML = '';

    const auto = document.createElement('option');
    auto.value = 'auto';
    auto.textContent = 'Auto (recommended)';
    this.dtypeSelect.appendChild(auto);

    const dtypes: Exclude<Dtype, 'auto'>[] = ['q4f16', 'q4', 'q8', 'fp16', 'fp32'];
    for (const dtype of dtypes) {
      const opt = document.createElement('option');
      opt.value = dtype;

      let suffix = '';
      let disabled = false;
      if (info) {
        const status: DtypeStatus = evaluateDtype(info, dtype, this.gpu);
        const sizeMb = info.approxWeightsMb[dtype];
        const sizeStr = sizeMb ? ` · ~${sizeMb < 1024 ? `${sizeMb} MB` : `${(sizeMb / 1024).toFixed(2)} GB`}` : '';
        if (status === 'unsupported_by_model') {
          suffix = ' — not shipped for this model';
          disabled = true;
        } else if (status === 'needs_shader_f16') {
          suffix = ' — needs shader-f16 (your GPU lacks it)';
          disabled = true;
        } else if (status === 'too_large') {
          const bufferMb = info.maxBufferRequirementMb?.[dtype];
          const bufferStr = this.formatMb(bufferMb);
          const maxBuffer = formatBytes(this.gpu.maxBufferBytes);
          suffix = bufferStr
            ? `${sizeStr} — needs single buffer ~${bufferStr}; max ${maxBuffer}`
            : `${sizeStr} — exceeds max buffer ${maxBuffer}`;
          disabled = true;
        } else {
          suffix = sizeStr;
        }
      } else if ((dtype === 'q4f16' || dtype === 'fp16') && !this.gpu.shaderF16) {
        suffix = ' — needs shader-f16 (your GPU lacks it)';
        disabled = true;
      }

      opt.textContent = `${dtype}${suffix}`;
      opt.disabled = disabled;
      this.dtypeSelect.appendChild(opt);
    }

    if (this.dtypeSelect.querySelector(`option[value="${previous}"]:not([disabled])`)) {
      this.dtypeSelect.value = previous;
    } else {
      this.dtypeSelect.value = 'auto';
    }
    this.refreshDtypeHint();
  }

  private refreshDtypeHint() {
    const value = this.dtypeSelect.value as Dtype;
    const info = findModel(this.modelSelect.value);
    if (value === 'auto') {
      if (!info) {
        this.dtypeHint.textContent = 'Auto picks q4f16 if the GPU supports shader-f16, otherwise q4.';
        return;
      }
      const runnable = pickRunnableDtype(info, this.gpu);
      this.dtypeHint.textContent = runnable
        ? `Auto picks ${runnable} for this model on this GPU.`
        : `No shipped dtype for this model fits this GPU. ${this.dtypeStatusMessage(info, info.recommendedDtype, evaluateDtype(info, info.recommendedDtype, this.gpu))}`;
      return;
    }
    if (info) {
      const status = evaluateDtype(info, value as Exclude<Dtype, 'auto'>, this.gpu);
      this.dtypeHint.textContent = this.dtypeStatusMessage(info, value as Exclude<Dtype, 'auto'>, status);
    } else {
      this.dtypeHint.textContent = '';
    }
  }

  private async runTests() {
    if (this.testAbort) return;
    if (!this.gpu.available) {
      this.testProgressEl.textContent = 'WebGPU not available — cannot test.';
      this.testProgressEl.classList.remove('hidden');
      return;
    }
    if (this.model.loaded) {
      await this.model.unload();
      this.updateModelStatus();
    }
    this.testAbort = new AbortController();
    this.runTestsBtn.disabled = true;
    this.cancelTestsBtn.classList.remove('hidden');
    this.testProgressEl.classList.remove('hidden');
    const controller = this.testAbort;

    try {
      this.testResults = await runCompatibilityTests(this.gpu, {
        signal: controller.signal,
        exhaustive: this.exhaustiveTestsCheckbox.checked,
        onProgress: ({ index, total, modelId, dtype, phase }) => {
          this.testProgressEl.textContent = `${index + 1}/${total} · ${modelId} (${dtype}) · ${phase}…`;
          this.renderTestResults();
        },
      });
    } finally {
      const wasCanceled = controller.signal.aborted;
      this.testAbort = null;
      this.runTestsBtn.disabled = false;
      this.cancelTestsBtn.classList.add('hidden');
      this.testProgressEl.textContent = wasCanceled ? 'Tests canceled.' : 'Tests complete.';
      this.renderTestResults();
      this.populateDtypeOptions(findModel(this.modelSelect.value) ?? null);
    }
  }

  private cancelTests() {
    this.testAbort?.abort();
  }

  private clearTests() {
    clearTestResults();
    this.testResults = [];
    this.renderTestResults();
    this.populateDtypeOptions(findModel(this.modelSelect.value) ?? null);
  }

  private renderTestResults() {
    if (!this.testResults.length) {
      this.testResultsEl.innerHTML = '<p class="text-[11px] text-ink-400">No test results yet.</p>';
      return;
    }
    const groups = new Map<string, TestResult[]>();
    for (const r of this.testResults) {
      if (!groups.has(r.modelId)) groups.set(r.modelId, []);
      groups.get(r.modelId)!.push(r);
    }

    const rows: string[] = [];
    for (const [modelId, entries] of groups) {
      const info = findModel(modelId);
      const label = info ? info.label : modelId;
      const okRow = entries.find((e) => e.status === 'ok');
      const suggested = okRow
        ? suggestedDtypeFromTests(modelId, info?.recommendedDtype ?? 'q4', this.testResults)
        : null;
      rows.push(`<div class="font-medium text-ink-900 text-[12px] mt-2">${this.escapeHtml(label)}${suggested ? ` <span class="text-[11px] text-accent ml-1">→ ${suggested}</span>` : ''}</div>`);
      for (const r of entries) {
        const icon = r.status === 'ok' ? '✓' : r.status === 'timeout' ? '⏱' : '✗';
        const color = r.status === 'ok' ? 'text-ink-700' : 'text-accent';
        const timing = r.status === 'ok' ? `${((r.loadMs ?? 0) / 1000).toFixed(1)}s load · ${((r.inferMs ?? 0) / 1000).toFixed(1)}s infer` : (r.error ?? '');
        rows.push(`<div class="flex items-baseline gap-2 ${color}"><span class="font-mono w-6">${icon}</span><span class="font-mono w-12 text-[11px]">${r.dtype}</span><span class="text-[11px] truncate">${this.escapeHtml(timing)}</span></div>`);
      }
    }
    this.testResultsEl.innerHTML = rows.join('');
  }

  private loadData() {
    this.threads = loadThreads();
    if (this.threads.length === 0) {
      const thread = createThread();
      this.threads.push(thread);
      saveThreads(this.threads);
      this.switchThread(thread.id);
    } else {
      this.switchThread(this.orderedThreads()[0].id);
    }
    this.renderThreadList();
  }

  private orderedThreads(): ChatThread[] {
    return [...this.threads].sort((a, b) => {
      const favoriteDelta = Number(Boolean(b.favorite)) - Number(Boolean(a.favorite));
      if (favoriteDelta !== 0) return favoriteDelta;
      return b.updatedAt - a.updatedAt;
    });
  }

  private startNewChat() {
    const thread = createThread();
    this.threads.unshift(thread);
    saveThreads(this.threads);
    this.renderThreadList();
    this.switchThread(thread.id);

    this.closeSidebarOnMobile();
  }

  private switchThread(threadId: string) {
    this.currentThreadId = threadId;
    const thread = this.threads.find((t) => t.id === threadId);
    if (!thread) return;

    this.currentThreadTitle.textContent = thread.title;
    this.renderMessages(thread.messages);
    this.renderThreadList();
    this.closeSidebarOnMobile();
  }

  private renderThreadList() {
    this.threadList.innerHTML = '';

    for (const thread of this.orderedThreads()) {
      const row = document.createElement('div');
      row.className = `group flex items-center gap-1 rounded-md transition-colors ${
        thread.id === this.currentThreadId ? 'bg-cream-200' : 'hover:bg-cream-200/60'
      }`;

      const favorite = document.createElement('button');
      favorite.className = `flex-shrink-0 ml-1 p-1 rounded transition-colors ${
        thread.favorite
          ? 'text-accent hover:text-accent-hover'
          : 'text-ink-400 opacity-0 group-hover:opacity-100 hover:text-accent'
      }`;
      favorite.title = thread.favorite ? 'Unfavorite chat' : 'Favorite chat';
      favorite.setAttribute('aria-label', favorite.title);
      favorite.innerHTML =
        `<svg class="w-3.5 h-3.5" ${thread.favorite ? 'fill="currentColor"' : 'fill="none"'} stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M11.48 3.5c.2-.5.84-.5 1.04 0l2.25 5.42 5.85.47c.54.04.74.71.33 1.06l-4.46 3.82 1.36 5.72c.13.52-.42.93-.88.65L12 17.57l-4.97 3.07c-.46.28-1.01-.13-.88-.65l1.36-5.72-4.46-3.82c-.41-.35-.21-1.02.33-1.06l5.85-.47 2.25-5.42z"/></svg>`;
      favorite.onclick = (e) => {
        e.stopPropagation();
        this.toggleThreadFavorite(thread.id);
      };

      const title = document.createElement('button');
      title.className = `flex-1 min-w-0 text-left px-3 py-1.5 text-[13px] truncate ${
        thread.id === this.currentThreadId ? 'text-ink-900' : 'text-ink-500 group-hover:text-ink-900'
      }`;
      title.textContent = thread.title;
      title.onclick = () => this.switchThread(thread.id);
      title.ondblclick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.startRenameThread(thread.id, title);
      };

      const rename = document.createElement('button');
      rename.className =
        'flex-shrink-0 p-1 rounded text-ink-400 opacity-0 group-hover:opacity-100 hover:text-accent hover:bg-cream-100 transition-opacity';
      rename.title = 'Rename chat';
      rename.setAttribute('aria-label', 'Rename chat');
      rename.innerHTML =
        '<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"/></svg>';
      rename.onclick = (e) => {
        e.stopPropagation();
        this.startRenameThread(thread.id, title);
      };

      const del = document.createElement('button');
      del.className =
        'flex-shrink-0 p-1 mr-1 rounded text-ink-400 opacity-0 group-hover:opacity-100 hover:text-accent hover:bg-cream-100 transition-opacity';
      del.title = 'Delete chat';
      del.setAttribute('aria-label', 'Delete chat');
      del.innerHTML =
        '<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6M1 7h22M9 7V4a1 1 0 011-1h4a1 1 0 011 1v3"/></svg>';
      del.onclick = (e) => {
        e.stopPropagation();
        this.deleteThread(thread.id);
      };

      row.appendChild(favorite);
      row.appendChild(title);
      row.appendChild(rename);
      row.appendChild(del);
      this.threadList.appendChild(row);
    }
  }

  private startRenameThread(threadId: string, titleBtn: HTMLButtonElement) {
    const thread = this.threads.find((t) => t.id === threadId);
    if (!thread) return;

    const input = document.createElement('input');
    input.type = 'text';
    input.value = thread.title;
    input.className =
      'flex-1 min-w-0 mx-1 my-0.5 bg-cream-50 border border-cream-300 rounded px-2 py-1 text-[13px] text-ink-900 outline-none focus:border-accent';
    input.setAttribute('aria-label', 'Rename chat');

    let settled = false;
    const commit = () => {
      if (settled) return;
      settled = true;
      const trimmed = input.value.trim();
      if (trimmed && trimmed !== thread.title) {
        thread.title = trimmed;
        thread.updatedAt = Date.now();
        saveThreads(this.threads);
        if (threadId === this.currentThreadId) {
          this.currentThreadTitle.textContent = trimmed;
        }
      }
      this.renderThreadList();
    };
    const cancel = () => {
      if (settled) return;
      settled = true;
      this.renderThreadList();
    };

    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        commit();
      } else if (e.key === 'Escape') {
        e.preventDefault();
        cancel();
      }
    });
    input.addEventListener('blur', commit);

    titleBtn.replaceWith(input);
    input.focus();
    input.select();
  }

  private toggleThreadFavorite(threadId: string) {
    const thread = this.threads.find((t) => t.id === threadId);
    if (!thread) return;
    thread.favorite = !thread.favorite;
    saveThreads(this.threads);
    this.renderThreadList();
  }

  private deleteThread(threadId: string) {
    const thread = this.threads.find((t) => t.id === threadId);
    if (!thread) return;
    if (thread.messages.length > 0 && !confirm(`Delete "${thread.title}"?`)) return;

    this.threads = deleteThread(this.threads, threadId);

    if (this.threads.length === 0) {
      const fresh = createThread();
      this.threads.push(fresh);
      saveThreads(this.threads);
      this.switchThread(fresh.id);
    } else {
      saveThreads(this.threads);
      if (this.currentThreadId === threadId) {
        this.switchThread(this.orderedThreads()[0].id);
      }
    }
    this.renderThreadList();
  }

  private renderMessages(messages: Message[]) {
    this.chatMessages.innerHTML = '';
    this.messagesContainer = document.createElement('div');
    this.messagesContainer.className = 'max-w-3xl mx-auto px-6 py-10 space-y-8';
    this.chatMessages.appendChild(this.messagesContainer);

    if (messages.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'text-center mt-20';
      empty.innerHTML = `
        <h2 class="font-serif text-2xl text-ink-900 mb-2">Welcome to Local AI</h2>
        <p class="text-sm text-ink-500">${this.model.loaded ? 'Start typing to chat.' : 'Load a model to start chatting. Everything runs on your device.'}</p>
      `;
      this.messagesContainer.appendChild(empty);
      this.updateConversationContextMeter();
      return;
    }

    for (const msg of messages) {
      this.appendMessageElement(msg);
    }

    this.scrollToBottom();
    this.updateConversationContextMeter();
  }

  private appendMessageElement(msg: Message): HTMLElement {
    const isUser = msg.role === 'user';
    const wrapper = document.createElement('div');
    wrapper.className = isUser ? 'flex justify-end' : 'flex justify-start';
    wrapper.dataset.messageId = msg.id;

    const shell = document.createElement('div');
    shell.className = isUser ? 'flex max-w-[85%] flex-col items-end gap-1' : 'flex w-full flex-col items-start gap-1';

    const bubble = document.createElement('div');
    if (isUser) {
      bubble.className = 'bg-cream-100 border border-cream-200 text-ink-900 rounded-2xl px-4 py-2.5 text-[15px] leading-relaxed whitespace-pre-wrap';
      bubble.textContent = msg.content;
    } else {
      bubble.className = 'w-full prose-chat';
      bubble.innerHTML = this.renderMarkdown(msg.content);
      this.highlightCode(bubble);
      this.addCopyButtons(bubble);
    }

    const actions = document.createElement('div');
    actions.className = 'flex items-center gap-3';

    const copyBtn = document.createElement('button');
    copyBtn.type = 'button';
    copyBtn.className = 'text-[11px] text-ink-400 hover:text-accent transition-colors';
    copyBtn.textContent = 'Copy';
    copyBtn.addEventListener('click', () => void this.copyMessage(msg, copyBtn));
    actions.appendChild(copyBtn);

    if (!isUser) {
      const playBtn = document.createElement('button');
      playBtn.type = 'button';
      playBtn.dataset.ttsMessageId = msg.id;
      playBtn.addEventListener('click', () => void this.toggleMessageTts(msg, playBtn));
      actions.appendChild(playBtn);
      this.updateTtsPlaybackButton(playBtn, msg.id);
    }

    shell.append(bubble, actions);
    wrapper.appendChild(shell);
    (this.messagesContainer ?? this.chatMessages).appendChild(wrapper);
    return bubble;
  }

  private async copyText(text: string): Promise<void> {
    if (navigator.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(text);
        return;
      } catch {
        /* Fall back to the legacy copy path below. */
      }
    }

    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.setAttribute('readonly', '');
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    textarea.remove();
  }

  private async copyMessage(message: Message, button: HTMLButtonElement) {
    await this.copyText(message.content);
    button.textContent = 'Copied';
    setTimeout(() => {
      button.textContent = 'Copy';
    }, 1500);
  }

  private speechTextFromMessage(content: string): string {
    const container = document.createElement('div');
    container.innerHTML = this.renderMarkdown(content);
    container.querySelectorAll('pre').forEach((pre) => {
      pre.replaceWith(document.createTextNode('\nCode block omitted.\n'));
    });
    container.querySelectorAll('br').forEach((br) => {
      br.replaceWith(document.createTextNode('\n'));
    });

    const blockTags = new Set([
      'ADDRESS', 'ARTICLE', 'ASIDE', 'BLOCKQUOTE', 'DD', 'DIV', 'DL', 'DT',
      'FIGCAPTION', 'FIGURE', 'FOOTER', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6',
      'HEADER', 'HR', 'LI', 'MAIN', 'NAV', 'OL', 'P', 'PRE', 'SECTION', 'TABLE',
      'TBODY', 'TD', 'TFOOT', 'TH', 'THEAD', 'TR', 'UL',
    ]);

    const parts: string[] = [];
    const walk = (node: Node) => {
      if (node.nodeType === Node.TEXT_NODE) {
        parts.push(node.textContent ?? '');
        return;
      }

      if (!(node instanceof HTMLElement)) return;
      if (node.tagName === 'SCRIPT' || node.tagName === 'STYLE') return;

      if (node.tagName === 'LI') parts.push('\n');
      for (const child of node.childNodes) {
        walk(child);
      }
      if (blockTags.has(node.tagName)) parts.push('\n');
    };

    walk(container);

    return parts.join('')
      .replace(/[ \t]+\n/g, '\n')
      .replace(/\n[ \t]+/g, '\n')
      .replace(/\n{3,}/g, '\n\n')
      .replace(/[ \t]{2,}/g, ' ')
      .trim();
  }

  private async toggleMessageTts(message: Message, button: HTMLButtonElement) {
    if (!this.settings.tts.enabled) {
      this.setTtsStatus('Turn on text to speech first.');
      return;
    }

    if (this.ttsMessageId === message.id) {
      if (this.tts.isPaused) {
        await this.tts.resume();
        this.setTtsStatus('Speech resumed.');
      } else {
        await this.tts.pause();
        this.setTtsStatus('Speech paused.');
      }
      this.updateTtsPlaybackButtons();
      return;
    }

    const text = this.speechTextFromMessage(message.content);
    if (!text) {
      this.setTtsStatus('Nothing readable to speak in this response.');
      return;
    }

    this.ttsMessageId = message.id;
    this.updateTtsPlaybackButtons();

    try {
      await this.tts.speak(text, {
        voice: this.settings.tts.voice,
        speed: this.settings.tts.speed,
        onProgress: (progress) => {
          this.setTtsStatus(this.formatTtsProgress(progress));
        },
        onEnded: () => {
          if (this.ttsMessageId === message.id) {
            this.ttsMessageId = null;
            this.updateTtsPlaybackButtons();
            this.setTtsStatus('Speech finished.');
          }
        },
      });
    } catch (error) {
      const errorMessage = this.getErrorMessage(error);
      this.ttsMessageId = null;
      this.updateTtsPlaybackButtons();
      this.setTtsStatus(`Kokoro failed: ${errorMessage}`);
    }
  }

  private beginTtsForMessage(message: Message) {
    if (!this.settings.tts.enabled) return;

    const text = this.speechTextFromMessage(message.content);
    if (!text) {
      this.setTtsStatus('Nothing readable to speak in this response.');
      return;
    }

    this.ttsMessageId = message.id;
    this.updateTtsPlaybackButtons();
    void this.tts.start({
      voice: this.settings.tts.voice,
      speed: this.settings.tts.speed,
      onProgress: (progress) => {
        this.setTtsStatus(this.formatTtsProgress(progress));
      },
      onEnded: () => {
        if (this.ttsMessageId === message.id) {
          this.ttsMessageId = null;
          this.updateTtsPlaybackButtons();
          this.setTtsStatus('Speech finished.');
        }
      },
    }).catch((error) => {
      if (this.ttsMessageId !== message.id) return;
      const errorMessage = this.getErrorMessage(error);
      this.ttsMessageId = null;
      this.updateTtsPlaybackButtons();
      this.setTtsStatus(`Kokoro failed: ${errorMessage}`);
    });
    this.tts.pushText(text);
    this.tts.closeInput();
  }

  private stopTts() {
    this.tts.stop();
    this.ttsMessageId = null;
    this.updateTtsPlaybackButtons();
  }

  private updateTtsPlaybackButton(button: HTMLButtonElement, messageId: string) {
    const enabled = this.settings.tts.enabled;
    const active = this.ttsMessageId === messageId;
    const paused = active && this.tts.isPaused;
    const label = paused ? 'Resume speech' : active ? 'Pause speech' : 'Play speech';
    button.innerHTML = active && !paused
      ? '<svg class="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h3.5v14H7V5Zm6.5 0H17v14h-3.5V5Z"/></svg>'
      : '<svg class="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7L8 5Z"/></svg>';
    button.setAttribute('aria-label', label);
    button.disabled = !enabled;
    button.title = enabled
      ? `${label} with Kokoro (${kokoroVoiceLabel(this.settings.tts.voice)})`
      : 'Turn on text to speech to play this response.';
    button.className = enabled
      ? 'inline-flex h-5 w-5 items-center justify-center rounded text-ink-400 hover:text-accent hover:bg-cream-100 transition-colors disabled:text-ink-400/50 disabled:cursor-wait'
      : 'inline-flex h-5 w-5 items-center justify-center rounded text-ink-400/50 cursor-not-allowed transition-colors';
  }

  private updateTtsPlaybackButtons() {
    const buttons = this.chatMessages.querySelectorAll<HTMLButtonElement>('button[data-tts-message-id]');
    for (const button of buttons) {
      const id = button.dataset.ttsMessageId;
      if (id) this.updateTtsPlaybackButton(button, id);
    }
  }

  private safeFilename(value: string): string {
    const cleaned = value
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '');
    return cleaned || 'chat';
  }

  private formatThreadAsMarkdown(thread: ChatThread): string {
    const lines = [
      `# ${thread.title}`,
      '',
      `Exported: ${new Date().toISOString()}`,
      `Created: ${new Date(thread.createdAt).toISOString()}`,
      `Updated: ${new Date(thread.updatedAt).toISOString()}`,
      '',
    ];

    for (const message of thread.messages) {
      lines.push(`## ${message.role === 'user' ? 'User' : 'Assistant'}`);
      lines.push('');
      lines.push(message.content.trim() || '[empty]');
      lines.push('');
    }

    return lines.join('\n');
  }

  private exportCurrentThread() {
    const thread = this.getCurrentThread();
    if (!thread || thread.messages.length === 0) return;

    const markdown = this.formatThreadAsMarkdown(thread);
    const blob = new Blob([markdown], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    const date = new Date().toISOString().slice(0, 10);
    link.href = url;
    link.download = `${this.safeFilename(thread.title)}-${date}.md`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  private addCopyButtons(container: HTMLElement) {
    const codeBlocks = container.querySelectorAll('pre');
    for (const pre of codeBlocks) {
      const wrapper = document.createElement('div');
      wrapper.className = 'relative group';

      const copyBtn = document.createElement('button');
      copyBtn.className =
        'absolute top-2 right-2 p-1.5 rounded bg-gray-700/80 text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity text-xs';
      copyBtn.textContent = 'Copy';
      copyBtn.onclick = async () => {
        const code = pre.querySelector('code')?.textContent || '';
        await this.copyText(code);
        copyBtn.textContent = 'Copied!';
        setTimeout(() => (copyBtn.textContent = 'Copy'), 2000);
      };

      pre.parentNode?.insertBefore(wrapper, pre);
      wrapper.appendChild(pre);
      wrapper.appendChild(copyBtn);
    }
  }

  private scrollToBottom() {
    this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
  }

  private getErrorMessage(error: unknown): string {
    if (error instanceof Error) return error.message;
    return String(error);
  }

  private buildGenerationHistory(messages: Message[], search: WebSearchResponse | null) {
    const history = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    if (!search) return history;

    for (let i = history.length - 1; i >= 0; i--) {
      if (history[i].role === 'user') {
        history[i] = {
          role: 'user',
          content: formatWebSearchContext(search, history[i].content),
        };
        break;
      }
    }

    return history;
  }

  private buildPromptAugmentedHistory(messages: Message[], latestUserPrompt: string) {
    const history = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    for (let i = history.length - 1; i >= 0; i--) {
      if (history[i].role === 'user') {
        history[i] = {
          role: 'user',
          content: latestUserPrompt,
        };
        break;
      }
    }

    return history;
  }

  private async generateWithMcp(
    tools: McpTool[],
    threadMessages: Message[],
    content: string,
    assistantMsg: Message,
    bubble: HTMLElement,
  ): Promise<string | null> {
    if (!this.abortController) return null;

    bubble.textContent = tools.length
      ? `Checking ${tools.length} MCP tools...`
      : 'No MCP tools found. Answering from the model...';
    this.scrollToBottom();

    if (!tools.length) return null;

    const selectionHistory = this.buildPromptAugmentedHistory(
      threadMessages,
      formatMcpToolSelectionPrompt(tools, content),
    );
    let draft = '';
    const selection = await this.model.generate(
      selectionHistory,
      (token) => {
        draft += token;
        assistantMsg.content = draft;
        bubble.textContent = draft;
        this.scrollToBottom();
      },
      this.abortController.signal,
      {
        ...this.settings.generation,
        temperature: 0.1,
        topP: 0.9,
        maxNewTokens: Math.min(384, Math.max(128, this.settings.generation.maxNewTokens)),
      },
    );

    const toolCall = parseMcpToolCall(selection, tools);
    if (!toolCall) {
      return null;
    }

    bubble.textContent = `Calling ${toolCall.server}.${toolCall.name}...`;
    this.scrollToBottom();
    const result = await callMcpTool(
      this.settings.mcp.bridgeUrl,
      toolCall,
      this.settings.mcp.serverUrls,
      this.abortController.signal,
    );
    const finalHistory = this.buildPromptAugmentedHistory(
      threadMessages,
      formatMcpToolResultPrompt(toolCall, result, content),
    );

    let buffer = '';
    const cleaned = await this.model.generate(
      finalHistory,
      (token) => {
        buffer += token;
        assistantMsg.content = buffer;
        bubble.textContent = buffer;
        this.updateConversationContextMeter();
        this.scrollToBottom();
      },
      this.abortController.signal,
      this.settings.generation,
    );

    return cleaned;
  }

  private isRuntimeMemoryError(message: string): boolean {
    const normalized = message.toLowerCase();
    return (
      normalized.includes('memory access out of bounds') ||
      normalized.includes('out of memory') ||
      normalized.includes('max buffer size') ||
      normalized.includes('invalid buffer')
    );
  }

  private resolveDtypeForLoad(): Dtype {
    const configured = this.settings.dtype;
    const info = findModel(this.settings.modelId.trim());
    if (!info) return configured;

    if (configured !== 'auto') {
      const status = evaluateDtype(info, configured, this.gpu);
      if (status === 'ok') return configured;
    }

    return pickRunnableDtype(info, this.gpu) ?? info.recommendedDtype;
  }

  private getLoadBlockReason(modelId: string, dtype: Dtype): string | null {
    const info = findModel(modelId);
    if (!info || dtype === 'auto') return null;
    const status = evaluateDtype(info, dtype, this.gpu);
    if (status === 'ok') return null;
    return `${info.label} (${dtype}) cannot run on this GPU. ${this.dtypeStatusMessage(info, dtype, status)}`;
  }

  private setDownloadModelState(state: 'default' | 'loaded' | 'blocked' | 'loading', label: string) {
    this.downloadModelLabel.textContent = label;
    this.downloadModelBtn.setAttribute('aria-label', label);
    this.downloadModelBtn.title = label;
    this.downloadModelBtn.classList.remove(
      'text-ink-600', 'hover:text-ink-900', 'hover:bg-cream-100',
      'text-ink-400', 'cursor-default',
      'text-accent', 'cursor-not-allowed', 'cursor-wait',
    );
    if (state === 'loaded') {
      this.downloadModelBtn.classList.add('text-ink-400', 'cursor-default');
    } else if (state === 'blocked') {
      this.downloadModelBtn.classList.add('text-accent', 'cursor-not-allowed');
    } else if (state === 'loading') {
      this.downloadModelBtn.classList.add('text-accent', 'cursor-wait');
    } else {
      this.downloadModelBtn.classList.add('text-ink-600', 'hover:text-ink-900', 'hover:bg-cream-100');
    }
  }

  private updateModelStatus() {
    if (this.model.loaded) {
      const name = this.displayModelName(this.model.name);
      this.modelStatus.textContent = this.model.dtype ? `${name} (${this.model.dtype})` : name;
      this.modelStatus.className = 'text-ink-700 truncate';
      this.downloadModelBtn.disabled = true;
      this.setDownloadModelState('loaded', 'Loaded');
      return;
    }

    const dtype = this.resolveDtypeForLoad();
    const blocked = this.getLoadBlockReason(this.settings.modelId.trim(), dtype);
    if (blocked) {
      this.modelStatus.textContent = `Not compatible: ${this.displayModelName(this.settings.modelId)}`;
      this.modelStatus.className = 'text-accent truncate';
      this.downloadModelBtn.disabled = true;
      this.setDownloadModelState('blocked', 'Not compatible');
      return;
    }

    this.modelStatus.textContent = `Selected: ${this.displayModelName(this.settings.modelId)}`;
    this.modelStatus.className = 'text-ink-400 truncate';
    this.downloadModelBtn.disabled = false;
    this.setDownloadModelState('default', 'Load model');
  }

  private async loadModel() {
    if (this.model.loading || this.model.loaded) return;

    const modelId = this.settings.modelId.trim();
    if (!modelId) return;
    const dtype = this.resolveDtypeForLoad();
    const blocked = this.getLoadBlockReason(modelId, dtype);

    this.modelLoader.classList.remove('hidden');
    this.loadingModelDetails.textContent = `Loading ${this.displayModelName(modelId)} (${dtype}). This may take a few minutes on first load.`;
    this.progressBar.style.width = '0%';
    this.progressFile.textContent = '';
    this.progressText.textContent = 'Initializing...';
    this.modelStatus.textContent = `Loading: ${this.displayModelName(modelId)}`;
    this.modelStatus.className = 'text-accent truncate';
    this.downloadModelBtn.disabled = true;
    this.setDownloadModelState('loading', 'Loading');

    if (blocked) {
      this.progressText.textContent = blocked;
      this.loadingModelDetails.textContent = blocked;
      this.modelStatus.textContent = `Not compatible: ${this.displayModelName(modelId)}`;
      this.modelStatus.className = 'text-accent truncate';
      this.setDownloadModelState('blocked', 'Not compatible');
      this.downloadModelBtn.disabled = true;
      setTimeout(() => {
        this.modelLoader.classList.add('hidden');
      }, 3000);
      return;
    }

    try {
      await this.model.load(modelId, (data) => {
        if (data.progress !== undefined) {
          this.progressBar.style.width = `${data.progress}%`;
        }

        switch (data.status) {
          case 'init':
            this.progressText.textContent = 'Initializing...';
            break;
          case 'download':
            this.progressText.textContent = `Downloading... ${data.progress}%`;
            this.progressFile.textContent = data.file || '';
            break;
          case 'loading':
            this.progressText.textContent = data.message || 'Loading model into memory...';
            break;
          case 'ready':
            this.progressText.textContent = 'Model ready!';
            this.progressBar.style.width = '100%';
            break;
          case 'error':
            this.progressText.textContent = data.message || 'Error loading model.';
            break;
        }
      }, dtype);

      this.updateModelStatus();

      const thread = this.getCurrentThread();
      if (thread && thread.messages.length === 0) {
        this.renderMessages(thread.messages);
      }
    } catch (error) {
      console.error('Failed to load model:', error);
      this.modelStatus.textContent = `Load failed: ${this.displayModelName(modelId)}`;
      this.modelStatus.className = 'text-accent truncate';
      this.setDownloadModelState('default', 'Retry');
      this.downloadModelBtn.disabled = false;
    } finally {
      setTimeout(() => {
        this.modelLoader.classList.add('hidden');
      }, this.model.loaded ? 1000 : 3000);
    }
  }

  private openSettings() {
    this.showSettingsTab('general');
    const known = findModel(this.settings.modelId);
    if (known) {
      this.modelSelect.value = this.settings.modelId;
      this.modelIdInput.value = '';
      this.renderModelMeta(known);
    } else {
      this.modelSelect.value = '__custom__';
      this.modelIdInput.value = this.settings.modelId;
      this.modelMeta.textContent = 'Custom model — dtype auto-detected at load.';
    }
    this.populateDtypeOptions(known ?? null);
    this.applyModelSettingsToInputs(this.settings.modelId, known ?? null, {
      dtype: this.settings.dtype,
      generation: this.settings.generation,
    });
    this.webSearchEnabledInput.checked = this.settings.webSearch.enabled;
    this.searxngUrlInput.value = this.settings.webSearch.searxngUrl;
    this.mcpBridgeUrlInput.value = this.settings.mcp.bridgeUrl;
    this.mcpServerUrlsInput.value = this.settings.mcp.serverUrls.join('\n');
    this.ttsEnabledInput.checked = this.settings.tts.enabled;
    this.ttsVoiceSelect.value = isKokoroVoiceId(this.settings.tts.voice)
      ? this.settings.tts.voice
      : 'af_heart';
    this.ttsSpeedInput.value = String(this.settings.tts.speed);
    this.ttsSpeedValue.textContent = this.settings.tts.speed.toFixed(2);
    this.renderHardware();
    this.renderTestResults();
    this.settingsModal.classList.remove('hidden');
  }

  private showSettingsTab(tab: 'general' | 'tools') {
    const isGeneral = tab === 'general';
    this.settingsPanelGeneral.classList.toggle('hidden', !isGeneral);
    this.settingsPanelTools.classList.toggle('hidden', isGeneral);
    this.settingsGeneralTab.className = isGeneral
      ? 'flex-1 rounded-md bg-cream-50 px-3 py-1.5 text-[13px] text-ink-900 shadow-sm'
      : 'flex-1 rounded-md px-3 py-1.5 text-[13px] text-ink-500 hover:text-ink-900';
    this.settingsToolsTab.className = isGeneral
      ? 'flex-1 rounded-md px-3 py-1.5 text-[13px] text-ink-500 hover:text-ink-900'
      : 'flex-1 rounded-md bg-cream-50 px-3 py-1.5 text-[13px] text-ink-900 shadow-sm';
  }

  private closeSettings() {
    this.settingsModal.classList.add('hidden');
  }

  private async saveSettingsHandler() {
    const customId = this.modelIdInput.value.trim();
    const modelId = this.modelSelect.value === '__custom__'
      ? customId
      : (customId || this.modelSelect.value);
    if (!modelId) {
      this.modelIdInput.focus();
      return;
    }

    const maxNewTokens = this.maxNewTokensFromSlider();
    let contextTokens = this.contextTokensFromSlider();
    const selectedInfo = findModel(modelId);
    if (selectedInfo && Number.isFinite(contextTokens) && Number.isFinite(maxNewTokens)) {
      contextTokens = Math.min(contextTokens, Math.max(256, selectedInfo.contextWindowTokens - maxNewTokens));
    }

    const previousDtype = this.settings.dtype;
    const previousTtsEnabled = this.settings.tts.enabled;
    const dtype = this.dtypeSelect.value as AppSettings['dtype'];
    const generation: GenerationParams = {
      temperature: parseFloat(this.temperatureInput.value),
      topP: parseFloat(this.topPInput.value),
      maxNewTokens,
      contextTokens,
      repetitionPenalty: parseFloat(this.repPenaltyInput.value),
    };
    const ttsVoice = isKokoroVoiceId(this.ttsVoiceSelect.value)
      ? this.ttsVoiceSelect.value
      : 'af_heart';
    const searxngUrl = this.searxngUrlInput.value.trim();

    saveSettings({
      modelId,
      dtype,
      generation,
      modelSettings: {
        ...this.settings.modelSettings,
        [modelId]: {
          dtype,
          generation,
        },
      },
      webSearch: {
        enabled: this.webSearchEnabledInput.checked || Boolean(searxngUrl),
        searxngUrl,
      },
      mcp: {
        bridgeUrl: this.mcpBridgeUrlInput.value,
        serverUrls: this.mcpServerUrlsInput.value
          .split(/\r?\n/)
          .map((value) => value.trim())
          .filter(Boolean),
      },
      tts: {
        enabled: this.ttsEnabledInput.checked,
        voice: ttsVoice,
        speed: Number.parseFloat(this.ttsSpeedInput.value),
      },
    });
    this.settings = loadSettings();

    const shouldUnloadCurrentModel =
      this.model.loaded &&
      (this.model.name !== this.settings.modelId || previousDtype !== this.settings.dtype);

    if (shouldUnloadCurrentModel) {
      await this.model.unload();
      const thread = this.getCurrentThread();
      if (thread && thread.messages.length === 0) {
        this.renderMessages(thread.messages);
      }
    }

    if (previousTtsEnabled && !this.settings.tts.enabled) {
      this.stopTts();
    } else if (this.settings.tts.enabled) {
      void this.warmTts();
    }

    this.updateModelStatus();
    this.updateTtsToggleButton();
    this.updateTtsPlaybackButtons();
    this.updateConversationContextMeter();
    this.closeSettings();

    if (shouldUnloadCurrentModel) {
      void this.loadModel();
    }
  }

  private getCurrentThread(): ChatThread | null {
    return this.threads.find((t) => t.id === this.currentThreadId) || null;
  }

  private getUserPromptHistory(): string[] {
    const thread = this.getCurrentThread();
    if (!thread) return [];

    const prompts: string[] = [];
    for (const message of thread.messages) {
      const content = message.content.trim();
      if (message.role !== 'user' || !content) continue;
      if (prompts[prompts.length - 1] === content) continue;
      prompts.push(content);
    }
    return prompts;
  }

  private handlePromptHistoryNavigation(event: KeyboardEvent): boolean {
    if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return false;
    if (event.shiftKey || event.altKey || event.metaKey || event.ctrlKey || event.isComposing) return false;

    const input = this.messageInput;
    const atStart = input.selectionStart === 0 && input.selectionEnd === 0;

    if (event.key === 'ArrowUp' && !atStart) return false;
    if (event.key === 'ArrowDown' && this.promptHistoryIndex === null) return false;

    const history = this.getUserPromptHistory();
    if (history.length === 0) return false;

    event.preventDefault();

    if (this.promptHistoryIndex === null) {
      this.promptHistoryDraft = input.value;
      this.promptHistoryIndex = history.length;
    }

    if (event.key === 'ArrowUp') {
      this.promptHistoryIndex = Math.max(0, this.promptHistoryIndex - 1);
      this.setMessageInputValue(history[this.promptHistoryIndex]);
      return true;
    }

    this.promptHistoryIndex += 1;
    if (this.promptHistoryIndex >= history.length) {
      this.promptHistoryIndex = null;
      this.setMessageInputValue(this.promptHistoryDraft);
      this.promptHistoryDraft = '';
    } else {
      this.setMessageInputValue(history[this.promptHistoryIndex]);
    }

    return true;
  }

  private async compactCurrentThread() {
    if (this.isGenerating || this.isCompacting) return;

    const thread = this.getCurrentThread();
    if (!thread) return;

    const { toCompact, toKeep } = this.splitMessagesForCompaction(thread.messages);
    if (toCompact.length === 0) return;

    this.isCompacting = true;
    this.isGenerating = true;
    this.sendBtn.disabled = true;
    this.updateCompactContextButton();

    try {
      if (!this.model.loaded) {
        await this.loadModel();
        if (!this.model.loaded) return;
      }

      const transcript = this.formatCompactionTranscript(toCompact);
      const summaryPrompt = [
        {
          role: 'user',
          content: [
            'Compact the earlier chat transcript into a durable context summary for continuing this same conversation.',
            'Preserve user goals, technical facts, model/settings choices, decisions, unresolved issues, constraints, and any concrete data that future replies need.',
            'Do not answer the conversation. Return only the compact summary in concise markdown bullets.',
            '',
            'Earlier transcript:',
            transcript,
          ].join('\n'),
        },
      ];

      const summary = await this.model.generate(
        summaryPrompt,
        () => {},
        new AbortController().signal,
        {
          ...this.settings.generation,
          temperature: 0.2,
          topP: 0.9,
          maxNewTokens: Math.min(512, Math.max(192, this.settings.generation.maxNewTokens * 2)),
          contextTokens: this.settings.generation.contextTokens,
        },
      );

      const summaryMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: `Context compacted from ${toCompact.length} earlier messages:\n\n${summary.trim()}`,
        timestamp: Date.now(),
      };

      thread.messages = [summaryMessage, ...toKeep];
      thread.updatedAt = Date.now();
      saveThreads(this.threads);
      this.renderMessages(thread.messages);
      this.renderThreadList();
      this.currentThreadTitle.textContent = thread.title;
      this.updateConversationContextMeter();
    } catch (error) {
      console.error('Context compaction error:', error);
      const errorMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: `Context compaction failed: ${this.getErrorMessage(error)}`,
        timestamp: Date.now(),
      };
      thread.messages.push(errorMessage);
      thread.updatedAt = Date.now();
      saveThreads(this.threads);
      this.appendMessageElement(errorMessage);
      this.updateConversationContextMeter();
      this.scrollToBottom();
    } finally {
      this.isCompacting = false;
      this.isGenerating = false;
      this.sendBtn.disabled = false;
      this.updateCompactContextButton();
    }
  }

  private async sendMessage() {
    const content = this.messageInput.value.trim();
    if (!content || this.isGenerating) return;

    if (!this.model.loaded) {
      await this.loadModel();
      if (!this.model.loaded) {
        // Model failed to load — keep the user's text so they don't lose it.
        this.messageInput.value = content;
        this.messageInput.focus();
        return;
      }
    }

    const thread = this.getCurrentThread();
    if (!thread) return;

    const userMsg: Message = {
      id: generateId(),
      role: 'user',
      content,
      timestamp: Date.now(),
    };
    thread.messages.push(userMsg);
    thread.updatedAt = Date.now();
    updateThreadTitle(thread);
    saveThreads(this.threads);

    this.promptHistoryIndex = null;
    this.promptHistoryDraft = '';
    this.messageInput.value = '';
    this.resizeMessageInput();

    if (thread.messages.length === 1) {
      this.renderMessages([]);
      this.messagesContainer?.replaceChildren();
    }
    this.appendMessageElement(userMsg);
    this.renderThreadList();
    this.currentThreadTitle.textContent = thread.title;
    this.updateConversationContextMeter();
    this.scrollToBottom();

    this.isGenerating = true;
    this.sendBtn.disabled = true;
    this.abortController = new AbortController();

    const assistantMsg: Message = {
      id: generateId(),
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
    };
    thread.messages.push(assistantMsg);

    const bubble = this.appendMessageElement(assistantMsg);
    bubble.textContent = '…';
    this.updateConversationContextMeter();
    this.scrollToBottom();

    try {
      let searchResults: WebSearchResponse | null = null;
      const webSearchActive = this.settings.webSearch.enabled;
      const mcpActive = this.settings.mcp.serverUrls.length > 0;

      if (webSearchActive) {
        bubble.textContent = 'Searching web...';
        try {
          searchResults = await webSearch(content, {
            signal: this.abortController.signal,
            limit: 5,
            searxngUrl: this.settings.webSearch.searxngUrl,
          });
          const resultCount = searchResults.results.length + searchResults.facts.length;
          bubble.textContent = resultCount
            ? `Reading ${resultCount} web results...`
            : 'No web results found. Answering from the model...';
        } catch (error) {
          if (this.abortController.signal.aborted) throw error;
          const message = this.getErrorMessage(error);
          bubble.textContent = `Web search failed: ${message}. Answering from the model...`;
        }
        this.scrollToBottom();
      }

      let mcpTools: McpTool[] = [];
      if (mcpActive) {
        bubble.textContent = 'Loading MCP tools...';
        try {
          mcpTools = await listMcpTools(
            this.settings.mcp.bridgeUrl,
            this.settings.mcp.serverUrls,
            this.abortController.signal,
          );
        } catch (error) {
          if (this.abortController.signal.aborted) throw error;
          bubble.textContent = `MCP bridge failed: ${this.getErrorMessage(error)}. Answering from the model...`;
          this.scrollToBottom();
        }
      }

      const mcpAnswer = mcpActive
        ? await this.generateWithMcp(mcpTools, thread.messages.slice(0, -1), content, assistantMsg, bubble)
        : null;

      if (mcpAnswer !== null) {
        assistantMsg.content = mcpAnswer;
        bubble.innerHTML = this.renderMarkdown(mcpAnswer);
        this.highlightCode(bubble);
        this.addCopyButtons(bubble);
        if (this.settings.tts.enabled && this.ttsMessageId === null && !this.tts.isActive) {
          this.beginTtsForMessage(assistantMsg);
        }
        this.updateConversationContextMeter();
        return;
      }

      const chatHistory = this.buildGenerationHistory(thread.messages.slice(0, -1), searchResults);

      let buffer = '';
      const cleaned = await this.model.generate(
        chatHistory,
        (token) => {
          buffer += token;
          assistantMsg.content = buffer;
          bubble.textContent = buffer;
          this.updateConversationContextMeter();
          this.scrollToBottom();
        },
        this.abortController.signal,
        this.settings.generation
      );

      assistantMsg.content = cleaned;
      bubble.innerHTML = this.renderMarkdown(cleaned);
      this.highlightCode(bubble);
      this.addCopyButtons(bubble);
      if (this.settings.tts.enabled && this.ttsMessageId === null && !this.tts.isActive) {
        this.beginTtsForMessage(assistantMsg);
      }
      this.updateConversationContextMeter();
    } catch (error) {
      console.error('Generation error:', error);
      const message = this.getErrorMessage(error);
      const memoryHint = this.isRuntimeMemoryError(message)
        ? ' The model session was unloaded. Try a smaller model or lower Settings -> Generation -> Prompt context tokens / Max new tokens.'
        : '';
      if (memoryHint) {
        await this.model.unload();
        this.updateModelStatus();
      }
      if (this.ttsMessageId === assistantMsg.id) {
        this.stopTts();
      }
      assistantMsg.content = `Error generating response: ${message}${memoryHint}`;
      bubble.className = 'w-full text-sm text-accent';
      bubble.textContent = `Error: ${message}${memoryHint}`;
      this.updateConversationContextMeter();
    } finally {
      this.isGenerating = false;
      this.sendBtn.disabled = false;
      this.abortController = null;
      saveThreads(this.threads);
      this.updateConversationContextMeter();
      this.scrollToBottom();
    }
  }

  private highlightCode(container: HTMLElement) {
    container.querySelectorAll('pre code').forEach((block) => {
      hljs.highlightElement(block as HTMLElement);
    });
  }

  private toggleSidebar() {
    const hidden = this.sidebar.classList.toggle('-translate-x-full');
    this.sidebarBackdrop.classList.toggle('hidden', hidden);
  }

  private closeSidebarOnMobile() {
    if (window.innerWidth >= 1024) return;
    this.sidebar.classList.add('-translate-x-full');
    this.sidebarBackdrop.classList.add('hidden');
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new ChatApp();
});
