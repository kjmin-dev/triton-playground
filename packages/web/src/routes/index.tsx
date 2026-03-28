import { createFileRoute } from '@tanstack/react-router';
import type { LucideIcon } from 'lucide-react';
import {
  ArrowRight,
  Bot,
  Download,
  Globe2,
  Headphones,
  Languages,
  LoaderCircle,
  MicVocal,
  Play,
  Sparkles,
  Upload,
  Waves,
} from 'lucide-react';
import * as React from 'react';

import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { getWorkerBaseUrl } from '@/lib/runtime-config';

export const Route = createFileRoute('/')({
  component: Home,
});

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

type DemoMode = 'vad' | 'stt' | 'localize' | 'tts';
type TtsVoiceMode = 'reference_clone' | 'preset_voice';

type ReadyResponse = {
  status: string;
  profile: string;
  triton: {
    model_name: string;
    model_ready: boolean;
    server_live: boolean;
    server_ready: boolean;
    status: string;
    summary: string;
  };
};

type VadResponse = {
  duration_ms: number;
  filename: string;
  model: string;
  sample_rate: number;
  segment_count: number;
  segments: Array<{
    average_probability: number;
    duration_ms: number;
    end_ms: number;
    peak_probability: number;
    start_ms: number;
  }>;
  threshold: number;
  window_ms: number;
  window_scores: number[];
};

type SttResponse = {
  duration_ms: number;
  filename: string;
  language: string;
  model: string;
  repository_model_name: string;
  sample_rate: number;
  segment_count: number;
  segments: Array<{
    average_probability: number;
    duration_ms: number;
    end_ms: number;
    peak_probability: number;
    start_ms: number;
    text: string;
  }>;
  task: string;
  threshold: number;
  transcript: string;
};

type LocalizeResponse = {
  duration_ms: number;
  filename: string;
  message?: string;
  models: {
    stt: string;
    translation: string;
    tts: string;
  };
  sample_rate: number;
  source_language: string;
  stage?: string;
  stages: {
    stt: {
      elapsed_ms?: number;
      language?: string;
      message?: string;
      segment_count?: number;
      speaker_count?: number;
      status: string;
      task?: string;
      transcript?: string;
      segments?: Array<{
        start_ms: number;
        end_ms: number;
        duration_ms: number;
        text: string;
        average_probability: number;
        peak_probability: number;
        speaker_id?: string;
      }>;
    };
    translation: {
      elapsed_ms?: number;
      message?: string;
      reason?: string;
      source_language?: string;
      status: string;
      target_language?: string;
      text?: string;
    };
    tts: {
      audio_base64?: string;
      content_type?: string;
      duration_ms?: number;
      elapsed_ms?: number;
      language?: string;
      message?: string;
      reason?: string;
      sample_rate?: number;
      speaker_count?: number;
      speakers?: string[];
      status: string;
      voice_cloning?: boolean;
      voice_cloning_mode?: string;
    };
  };
  status: string;
  target_language: string;
  threshold: number;
  transcript: string;
  translated_text: string;
};

type TtsPreviewVariant = {
  emotion: string;
  is_default: boolean;
  label: string;
  preview_id: string;
  prompt: string;
  text: string;
};

type TtsActor = {
  actor_id: string;
  default_preview_id: string;
  description: string;
  is_default: boolean;
  label: string;
  language: string;
  preview_prompt: string;
  preview_text: string;
  preview_variants: TtsPreviewVariant[];
  speaker_name: string;
};

type TtsActorCatalogResponse = {
  default_voice_mode: TtsVoiceMode;
  voice_modes: Array<{
    available: boolean;
    description: string;
    label: string;
    mode: TtsVoiceMode;
  }>;
  status: string;
  preset_actor_message: string | null;
  preset_actor_model_id: string | null;
  supports_preset_actors: boolean;
  supports_reference_voice_clone: boolean;
  reference_audio: {
    accepted_formats: string[];
    optional: boolean;
    recommended_sample_rate: number;
  };
  default_actor_id_by_language: Record<string, string>;
  actors: TtsActor[];
};

type TtsResponse = {
  actor: string | null;
  actor_label: string | null;
  audio_base64: string;
  content_type: string;
  duration_ms: number;
  language: string;
  model: string;
  reference_audio_filename?: string | null;
  repository_model_name: string;
  sample_rate: number;
  status: string;
  text: string;
  voice_source_label: string;
  voice_mode: 'preset_actor' | 'reference_clone';
};

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const THRESHOLD_PRESETS = [0.3, 0.5, 0.65, 0.8];

const STT_LANGUAGE_OPTIONS = [
  { value: 'auto', label: 'Auto' },
  { value: 'ko', label: 'Korean' },
  { value: 'en', label: 'English' },
  { value: 'ja', label: 'Japanese' },
  { value: 'zh', label: 'Chinese' },
];

const TARGET_LANGUAGE_OPTIONS = STT_LANGUAGE_OPTIONS.filter((o) => o.value !== 'auto');
const TTS_LANGUAGE_OPTIONS = TARGET_LANGUAGE_OPTIONS;

const TTS_DELIVERY_PRESETS = [
  {
    id: 'neutral',
    label: 'Neutral',
    description: 'Balanced and direct delivery',
    prompt: 'clear natural delivery',
  },
  {
    id: 'warm',
    label: 'Warm',
    description: 'Gentle and reassuring pacing',
    prompt: 'warm reassuring delivery',
  },
  {
    id: 'energetic',
    label: 'Energetic',
    description: 'Bright and forward momentum',
    prompt: 'bright energetic performance',
  },
  {
    id: 'dramatic',
    label: 'Dramatic',
    description: 'High contrast and expressive emphasis',
    prompt: 'dramatic expressive delivery',
  },
] as const;

const TTS_TONE_PRESETS = [
  { id: 'natural', label: 'Natural', prompt: 'natural studio tone' },
  { id: 'soft', label: 'Soft', prompt: 'soft intimate tone' },
  { id: 'bright', label: 'Bright', prompt: 'bright crisp tone' },
  { id: 'calm', label: 'Calm', prompt: 'calm composed tone' },
  { id: 'urgent', label: 'Urgent', prompt: 'urgent high-focus tone' },
] as const;

const TTS_TIMBRE_HINTS = [
  { id: 'auto', label: 'Auto', prompt: '' },
  { id: 'feminine', label: 'Feminine', prompt: 'feminine timbre hint' },
  { id: 'masculine', label: 'Masculine', prompt: 'masculine timbre hint' },
  { id: 'androgynous', label: 'Androgynous', prompt: 'androgynous balanced timbre hint' },
  { id: 'youthful', label: 'Youthful', prompt: 'youthful timbre hint' },
  { id: 'mature', label: 'Mature', prompt: 'mature timbre hint' },
] as const;

const SCORE_LIMIT = 80;
const MAX_BYTES = 150 * 1024 * 1024;

const MODES: Array<{
  value: DemoMode;
  label: string;
  icon: LucideIcon;
  action: string;
  running: string;
}> = [
  { value: 'vad', label: 'VAD', icon: Waves, action: 'Run VAD', running: 'Running VAD\u2026' },
  { value: 'stt', label: 'STT', icon: MicVocal, action: 'Run STT', running: 'Running STT\u2026' },
  {
    value: 'localize',
    label: 'Voice Dub',
    icon: Sparkles,
    action: 'Run Voice Dub',
    running: 'Dubbing\u2026',
  },
  {
    value: 'tts',
    label: 'TTS Studio',
    icon: Headphones,
    action: 'Generate TTS',
    running: 'Generating TTS\u2026',
  },
];

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function fmtBytes(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KiB`;
  return `${(b / (1024 * 1024)).toFixed(1)} MiB`;
}

function fmtMs(ms: number | undefined) {
  if (!ms) return '0 ms';
  if (ms < 1000) return `${ms} ms`;
  const s = Math.round(ms / 100) / 10;
  if (s < 60) return `${s.toFixed(1)} s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

function fmtProb(p: number) {
  return `${Math.round(p * 100)}%`;
}

function langLabel(code: string) {
  return STT_LANGUAGE_OPTIONS.find((o) => o.value === code)?.label ?? code;
}

function buildAudioDataUrl(contentType?: string, audioBase64?: string) {
  if (!contentType || !audioBase64) return null;
  return `data:${contentType};base64,${audioBase64}`;
}

function downloadAudioDataUrl(filename: string, src: string) {
  const a = document.createElement('a');
  a.href = src;
  a.download = filename;
  a.click();
}

function resolveTtsPreviewVariant(actor: TtsActor | null | undefined, previewId?: string | null) {
  if (!actor) return null;
  return (
    actor.preview_variants.find((preview) => preview.preview_id === previewId) ??
    actor.preview_variants.find((preview) => preview.is_default) ??
    actor.preview_variants[0] ??
    null
  );
}

function ttsPreviewCacheKey(actorId: string, previewId: string) {
  return `${actorId}:${previewId}`;
}

function buildTtsDirection(options: {
  customPrompt: string;
  deliveryPresetId: (typeof TTS_DELIVERY_PRESETS)[number]['id'];
  timbreHintId: (typeof TTS_TIMBRE_HINTS)[number]['id'];
  tonePresetId: (typeof TTS_TONE_PRESETS)[number]['id'];
}) {
  const deliveryPrompt = TTS_DELIVERY_PRESETS.find((item) => item.id === options.deliveryPresetId)?.prompt ?? '';
  const tonePrompt = TTS_TONE_PRESETS.find((item) => item.id === options.tonePresetId)?.prompt ?? '';
  const timbrePrompt = TTS_TIMBRE_HINTS.find((item) => item.id === options.timbreHintId)?.prompt ?? '';
  const customPrompt = options.customPrompt.trim();

  return [deliveryPrompt, tonePrompt, timbrePrompt, customPrompt].filter(Boolean).join('; ');
}

async function downloadSubtitle(fmt: string, loc: LocalizeResponse, workerBaseUrl: string) {
  try {
    const res = await fetch(`${workerBaseUrl}/api/subtitles/${fmt}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(loc),
    });
    if (!res.ok) return;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `subtitles.${fmt}`;
    a.click();
    URL.revokeObjectURL(url);
  } catch {
    // silent fail
  }
}

/* ------------------------------------------------------------------ */
/*  Page                                                               */
/* ------------------------------------------------------------------ */

function Home() {
  const workerBaseUrl = getWorkerBaseUrl();

  const [mode, setMode] = React.useState<DemoMode>('localize');
  const [tab, setTab] = React.useState<DemoMode>('localize');
  const [threshold, setThreshold] = React.useState('0.50');
  const [srcLang, setSrcLang] = React.useState('auto');
  const [tgtLang, setTgtLang] = React.useState('en');
  const [file, setFile] = React.useState<File | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [busy, setBusy] = React.useState<DemoMode | null>(null);
  const [ready, setReady] = React.useState<ReadyResponse | null>(null);
  const [vad, setVad] = React.useState<VadResponse | null>(null);
  const [sttRes, setStt] = React.useState<SttResponse | null>(null);
  const [loc, setLoc] = React.useState<LocalizeResponse | null>(null);
  const [ttsRes, setTtsRes] = React.useState<TtsResponse | null>(null);
  const [ttsCatalog, setTtsCatalog] = React.useState<TtsActorCatalogResponse | null>(null);
  const [ttsCatalogError, setTtsCatalogError] = React.useState<string | null>(null);
  const [ttsText, setTtsText] = React.useState('안녕하세요. Triton Playground 음성 합성 테스트입니다.');
  const [ttsPrompt, setTtsPrompt] = React.useState('');
  const [ttsLanguage, setTtsLanguage] = React.useState('ko');
  const [ttsVoiceMode, setTtsVoiceMode] = React.useState<TtsVoiceMode>('reference_clone');
  const [ttsDeliveryPresetId, setTtsDeliveryPresetId] =
    React.useState<(typeof TTS_DELIVERY_PRESETS)[number]['id']>('neutral');
  const [ttsTonePresetId, setTtsTonePresetId] = React.useState<(typeof TTS_TONE_PRESETS)[number]['id']>('natural');
  const [ttsTimbreHintId, setTtsTimbreHintId] = React.useState<(typeof TTS_TIMBRE_HINTS)[number]['id']>('auto');
  const [ttsActorId, setTtsActorId] = React.useState('');
  const [ttsReferenceFile, setTtsReferenceFile] = React.useState<File | null>(null);
  const [ttsPreviewBusyKey, setTtsPreviewBusyKey] = React.useState<string | null>(null);
  const [ttsPreviewError, setTtsPreviewError] = React.useState<string | null>(null);
  const [ttsPreviewAudioByKey, setTtsPreviewAudioByKey] = React.useState<Record<string, string>>({});

  const thr = Number.parseFloat(threshold);
  const thrOk = Number.isFinite(thr) && thr >= 0.1 && thr <= 0.99;
  const live = Boolean(ready?.triton.server_ready && ready?.triton.model_ready);
  const cfg = MODES.find((m) => m.value === mode) ?? MODES[0];

  const [dragging, setDragging] = React.useState(false);

  const originalAudioUrl = React.useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  const ttsReferenceAudioUrl = React.useMemo(
    () => (ttsReferenceFile ? URL.createObjectURL(ttsReferenceFile) : null),
    [ttsReferenceFile]
  );
  React.useEffect(() => {
    return () => {
      if (originalAudioUrl) URL.revokeObjectURL(originalAudioUrl);
    };
  }, [originalAudioUrl]);
  React.useEffect(() => {
    return () => {
      if (ttsReferenceAudioUrl) URL.revokeObjectURL(ttsReferenceAudioUrl);
    };
  }, [ttsReferenceAudioUrl]);

  const audioSrc = buildAudioDataUrl(loc?.stages.tts.content_type, loc?.stages.tts.audio_base64);
  const ttsAudioSrc = buildAudioDataUrl(ttsRes?.content_type, ttsRes?.audio_base64);
  const scores = vad?.window_scores.slice(0, SCORE_LIMIT) ?? [];
  const speechWins = vad && thrOk ? vad.window_scores.filter((s) => s >= thr).length : 0;
  const ttsActors = React.useMemo(() => ttsCatalog?.actors ?? [], [ttsCatalog]);
  const selectedTtsActor = React.useMemo(
    () => ttsActors.find((actor) => actor.actor_id === ttsActorId) ?? null,
    [ttsActorId, ttsActors]
  );
  const presetVoiceAvailable = Boolean(ttsCatalog?.supports_preset_actors);
  const isReferenceVoiceMode = ttsVoiceMode === 'reference_clone';
  const isPresetVoiceMode = ttsVoiceMode === 'preset_voice';
  const effectiveTtsDirection = React.useMemo(
    () =>
      buildTtsDirection({
        customPrompt: ttsPrompt,
        deliveryPresetId: ttsDeliveryPresetId,
        timbreHintId: ttsTimbreHintId,
        tonePresetId: ttsTonePresetId,
      }),
    [ttsDeliveryPresetId, ttsPrompt, ttsTimbreHintId, ttsTonePresetId]
  );
  const canRunPipeline = Boolean(file) && thrOk && busy === null;
  const canRunTts =
    Boolean(ttsText.trim()) &&
    Boolean(ttsLanguage) &&
    (isReferenceVoiceMode ? Boolean(ttsReferenceFile) : Boolean(presetVoiceAvailable && selectedTtsActor)) &&
    busy === null;

  function handleFileDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) {
      setFile(dropped);
      setError(null);
    }
  }

  React.useEffect(() => {
    let off = false;
    (async () => {
      try {
        const res = await fetch(`${workerBaseUrl}/api/ready`);
        const body = (await res.json().catch(() => null)) as ReadyResponse | null;
        if (off) return;
        setReady(body);
        if (!res.ok) setError(body?.triton.summary ?? `worker check failed: ${res.status}`);
      } catch (err) {
        if (!off) setError(err instanceof Error ? err.message : 'failed to reach worker');
      }
    })();
    return () => {
      off = true;
    };
  }, [workerBaseUrl]);

  React.useEffect(() => {
    let off = false;
    (async () => {
      try {
        const res = await fetch(`${workerBaseUrl}/api/tts/actors`);
        const body = (await res.json().catch(() => null)) as TtsActorCatalogResponse | null;
        if (off) return;
        if (!res.ok || !body) {
          setTtsCatalogError(`failed to load TTS actors: ${res.status}`);
          return;
        }
        setTtsCatalog(body);
        setTtsCatalogError(null);
        setTtsVoiceMode((prev) => {
          if (prev === 'preset_voice' && !body.supports_preset_actors) return 'reference_clone';
          if (prev === 'reference_clone' && !ttsReferenceFile) return body.default_voice_mode;
          return prev;
        });
      } catch (err) {
        if (!off) setTtsCatalogError(err instanceof Error ? err.message : 'failed to load TTS actors');
      }
    })();
    return () => {
      off = true;
    };
  }, [ttsReferenceFile, workerBaseUrl]);

  React.useEffect(() => {
    if (ttsReferenceFile) {
      setTtsVoiceMode('reference_clone');
    }
  }, [ttsReferenceFile]);

  React.useEffect(() => {
    if (ttsActors.length === 0) return;
    if (selectedTtsActor && selectedTtsActor.language === ttsLanguage) return;

    const defaultActorId =
      ttsCatalog?.default_actor_id_by_language[ttsLanguage] ??
      ttsActors.find((actor) => actor.language === ttsLanguage)?.actor_id ??
      ttsActors[0]?.actor_id ??
      '';
    setTtsActorId(defaultActorId);
  }, [selectedTtsActor, ttsActors, ttsCatalog, ttsLanguage]);

  async function runPipeline(route: Exclude<DemoMode, 'tts'>) {
    if (!file) {
      setError('Upload a WAV file first.');
      return;
    }
    if (!thrOk) {
      setError('Threshold must be between 0.10 and 0.99.');
      return;
    }

    setBusy(route);
    setTab(route);
    setError(null);
    if (route === 'vad') setVad(null);
    else if (route === 'stt') setStt(null);
    else setLoc(null);

    try {
      const fd = new FormData();
      fd.append('file', file);
      const q = new URLSearchParams({ threshold });

      if (route === 'stt' && srcLang !== 'auto') {
        q.set('language', srcLang);
      } else if (route === 'localize') {
        q.set('target_language', tgtLang);
        if (srcLang !== 'auto') q.set('source_language', srcLang);
      }

      const res = await fetch(`${workerBaseUrl}/api/${route}?${q}`, {
        method: 'POST',
        body: fd,
      });
      const body = await res.json().catch(() => ({}) as Record<string, unknown>);

      if (!res.ok) {
        if (route === 'localize') setLoc(body as LocalizeResponse);
        throw new Error(
          (body as { detail?: string; message?: string }).detail ??
            (body as { detail?: string; message?: string }).message ??
            `request failed: ${res.status}`
        );
      }

      if (route === 'vad') setVad(body as VadResponse);
      else if (route === 'stt') setStt(body as SttResponse);
      else setLoc(body as LocalizeResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : `failed to run ${route}`);
    } finally {
      setBusy(null);
    }
  }

  async function runTts() {
    if (!ttsText.trim()) {
      setError('Enter dialogue text for TTS.');
      return;
    }
    if (isReferenceVoiceMode) {
      if (!ttsReferenceFile) {
        setError('Upload reference audio to use Reference Voice mode.');
        return;
      }
    } else {
      if (!ttsCatalog?.supports_preset_actors) {
        setError(ttsCatalog?.preset_actor_message ?? 'Preset voice mode is unavailable in the current runtime.');
        return;
      }
      if (!selectedTtsActor || selectedTtsActor.language !== ttsLanguage) {
        setError('Select a preset voice that matches the current output language.');
        return;
      }
    }

    setBusy('tts');
    setTab('tts');
    setError(null);
    setTtsRes(null);

    try {
      const fd = new FormData();
      fd.append('text', ttsText.trim());
      fd.append('language', ttsLanguage);
      fd.append('model', 'qwen3_tts_0_6b');
      if (effectiveTtsDirection) fd.append('prompt', effectiveTtsDirection);
      if (isPresetVoiceMode && selectedTtsActor) {
        fd.append('actor', selectedTtsActor.actor_id);
      }
      if (isReferenceVoiceMode && ttsReferenceFile) {
        fd.append('reference_audio', ttsReferenceFile);
      }

      const res = await fetch(`${workerBaseUrl}/api/tts`, {
        method: 'POST',
        body: fd,
      });
      const body = await res.json().catch(() => ({}) as Record<string, unknown>);
      if (!res.ok) {
        throw new Error(
          (body as { detail?: string; message?: string }).detail ??
            (body as { detail?: string; message?: string }).message ??
            `request failed: ${res.status}`
        );
      }

      setTtsRes(body as TtsResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'failed to generate TTS');
    } finally {
      setBusy(null);
    }
  }

  async function previewTtsActor(actorId: string) {
    if (!ttsCatalog?.supports_preset_actors) {
      setTtsPreviewError(
        ttsCatalog?.preset_actor_message ?? 'Preset voice preview is unavailable in the current runtime.'
      );
      return;
    }
    const actor = ttsCatalog?.actors.find((item) => item.actor_id === actorId);
    if (!actor) return;
    const previewVariant = resolveTtsPreviewVariant(actor, null);
    if (!previewVariant) return;
    const previewKey = ttsPreviewCacheKey(actor.actor_id, previewVariant.preview_id);

    setTtsPreviewBusyKey(previewKey);
    setTtsPreviewError(null);
    try {
      const fd = new FormData();
      fd.append('text', previewVariant.text);
      fd.append('language', actor.language);
      fd.append('actor', actor.actor_id);
      fd.append('model', 'qwen3_tts_0_6b');
      if (previewVariant.prompt.trim()) fd.append('prompt', previewVariant.prompt.trim());

      const res = await fetch(`${workerBaseUrl}/api/tts`, {
        method: 'POST',
        body: fd,
      });
      const body = await res.json().catch(() => ({}) as Record<string, unknown>);
      if (!res.ok) {
        throw new Error(
          (body as { detail?: string; message?: string }).detail ??
            (body as { detail?: string; message?: string }).message ??
            `request failed: ${res.status}`
        );
      }

      const ttsBody = body as TtsResponse;
      const previewSrc = buildAudioDataUrl(ttsBody.content_type, ttsBody.audio_base64);
      if (!previewSrc) {
        throw new Error('preview audio was empty');
      }
      setTtsPreviewAudioByKey((prev) => ({ ...prev, [previewKey]: previewSrc }));
    } catch (err) {
      setTtsPreviewError(err instanceof Error ? err.message : 'failed to preview preset voice');
    } finally {
      setTtsPreviewBusyKey(null);
    }
  }

  return (
    <main className='mx-auto max-w-5xl px-4 py-6 sm:px-6'>
      {/* ── Header ── */}
      <header className='flex items-center justify-between'>
        <h1 className='text-base font-semibold tracking-tight text-slate-950'>Triton Playground</h1>
        <div className='flex items-center gap-2 text-xs text-slate-500'>
          <span className={`h-1.5 w-1.5 rounded-full ${live ? 'bg-emerald-500' : 'bg-amber-400'}`} />
          {live ? 'Ready' : 'Connecting'}
        </div>
      </header>

      {/* ── Banner ── */}
      <div className='mt-5 rounded-xl bg-slate-950 px-5 py-4 text-white'>
        <p className='text-sm font-semibold tracking-tight'>Speech &amp; Audio AI on Triton 24.05</p>
        <p className='mt-1 text-sm leading-relaxed text-slate-400'>
          Switch between pipeline demos and a dedicated Qwen3-TTS studio. VAD detects speech, STT transcribes it, Voice
          Dub translates and re-voices, and TTS Studio lets you switch between reference cloning and a preset voice
          library without changing the rest of the workflow.
        </p>
      </div>

      {/* ── Controls ── */}
      <form
        className='mt-5 space-y-4'
        onSubmit={(e) => {
          e.preventDefault();
          if (mode === 'tts') {
            void runTts();
            return;
          }
          void runPipeline(mode);
        }}
      >
        <div className='space-y-4 rounded-xl border border-slate-200 bg-white/70 p-4 backdrop-blur'>
          <div className='flex gap-1 rounded-lg bg-slate-100 p-1'>
            {MODES.map((m) => {
              const Icon = m.icon;
              const active = mode === m.value;
              const done =
                (m.value === 'vad' && vad) ||
                (m.value === 'stt' && sttRes) ||
                (m.value === 'localize' && loc) ||
                (m.value === 'tts' && ttsRes);
              return (
                <button
                  key={m.value}
                  className={`flex flex-1 items-center justify-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition ${
                    active ? 'bg-white text-slate-950 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                  }`}
                  disabled={busy !== null}
                  onClick={() => {
                    setMode(m.value);
                    setError(null);
                  }}
                  type='button'
                >
                  <Icon className='h-3.5 w-3.5' />
                  {m.label}
                  {done ? <span className='h-1.5 w-1.5 rounded-full bg-emerald-500' /> : null}
                </button>
              );
            })}
          </div>
        </div>

        {mode !== 'tts' ? (
          <>
            <label
              className={`flex cursor-pointer items-center gap-3 rounded-xl border-2 border-dashed px-4 py-5 backdrop-blur transition ${
                dragging
                  ? 'border-cyan-400 bg-cyan-50/80'
                  : file
                    ? 'border-emerald-300 bg-emerald-50/50 hover:border-emerald-400'
                    : 'border-slate-300 bg-white/70 hover:border-slate-400 hover:bg-white/90'
              }`}
              onDragEnter={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={(e) => {
                e.preventDefault();
                setDragging(false);
              }}
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleFileDrop}
            >
              <div className={`rounded-lg p-2 ${dragging ? 'bg-cyan-100' : 'bg-slate-100'}`}>
                <Upload className={`h-4 w-4 ${dragging ? 'text-cyan-600' : 'text-slate-500'}`} />
              </div>
              <div className='min-w-0 flex-1'>
                <div className='truncate text-sm font-medium text-slate-900'>
                  {dragging ? 'Drop audio file here' : file ? file.name : 'Upload audio file'}
                </div>
                <div className='text-xs text-slate-500'>
                  {file ? fmtBytes(file.size) : `Drag & drop or click \u00b7 WAV \u00b7 max ${fmtBytes(MAX_BYTES)}`}
                </div>
              </div>
              {file && (
                <span className='shrink-0 rounded-full bg-emerald-50 px-2 py-0.5 text-xs font-medium text-emerald-700'>
                  Loaded
                </span>
              )}
              <input
                accept='.wav,audio/wav'
                className='sr-only'
                disabled={busy !== null}
                onChange={(e) => {
                  setFile(e.target.files?.[0] ?? null);
                  setError(null);
                }}
                type='file'
              />
            </label>

            <div className='space-y-4 rounded-xl border border-slate-200 bg-white/70 p-4 backdrop-blur'>
              <div className='space-y-2'>
                <div className='flex items-center gap-3'>
                  <span className='w-16 shrink-0 text-sm font-medium text-slate-700'>Threshold</span>
                  <input
                    className='flex-1 accent-slate-950'
                    disabled={busy !== null}
                    max='0.99'
                    min='0.10'
                    onChange={(e) => setThreshold(e.target.value)}
                    step='0.01'
                    type='range'
                    value={threshold}
                  />
                  <input
                    className='w-16 rounded-lg border border-slate-200 px-2 py-1.5 text-center text-sm'
                    disabled={busy !== null}
                    max='0.99'
                    min='0.10'
                    onChange={(e) => setThreshold(e.target.value)}
                    step='0.01'
                    type='number'
                    value={threshold}
                  />
                </div>
                <div className='flex gap-1.5 pl-[76px]'>
                  {THRESHOLD_PRESETS.map((p) => (
                    <button
                      key={p}
                      className={`rounded-md px-2.5 py-1 text-xs transition ${
                        threshold === p.toFixed(2)
                          ? 'bg-slate-900 text-white'
                          : 'border border-slate-200 text-slate-600 hover:bg-slate-50'
                      }`}
                      disabled={busy !== null}
                      onClick={() => setThreshold(p.toFixed(2))}
                      type='button'
                    >
                      {p.toFixed(2)}
                    </button>
                  ))}
                </div>
              </div>

              {mode !== 'vad' && (
                <div className='flex flex-wrap items-center gap-4'>
                  <label className='flex items-center gap-2'>
                    <Languages className='h-3.5 w-3.5 text-slate-400' />
                    <span className='text-sm font-medium text-slate-700'>Source</span>
                    <select
                      className='rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm'
                      disabled={busy !== null}
                      onChange={(e) => setSrcLang(e.target.value)}
                      value={srcLang}
                    >
                      {STT_LANGUAGE_OPTIONS.map((o) => (
                        <option key={o.value} value={o.value}>
                          {o.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  {mode === 'localize' && (
                    <label className='flex items-center gap-2'>
                      <Globe2 className='h-3.5 w-3.5 text-slate-400' />
                      <span className='text-sm font-medium text-slate-700'>Target</span>
                      <select
                        className='rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm'
                        disabled={busy !== null}
                        onChange={(e) => setTgtLang(e.target.value)}
                        value={tgtLang}
                      >
                        {TARGET_LANGUAGE_OPTIONS.map((o) => (
                          <option key={o.value} value={o.value}>
                            {o.label}
                          </option>
                        ))}
                      </select>
                    </label>
                  )}
                </div>
              )}
            </div>
          </>
        ) : (
          <div className='space-y-5 rounded-xl border border-slate-200 bg-white/80 p-5 backdrop-blur'>
            <div className='grid gap-4 sm:grid-cols-2'>
              <label className='space-y-2'>
                <span className='flex items-center gap-2 text-sm font-medium text-slate-700'>
                  <Globe2 className='h-3.5 w-3.5 text-slate-400' />
                  Language
                </span>
                <select
                  className='w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm'
                  disabled={busy !== null}
                  onChange={(e) => setTtsLanguage(e.target.value)}
                  value={ttsLanguage}
                >
                  {TTS_LANGUAGE_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>

              <div className='space-y-2'>
                <span className='flex items-center gap-2 text-sm font-medium text-slate-700'>
                  <Bot className='h-3.5 w-3.5 text-slate-400' />
                  Voice Source
                </span>
                <div className='grid gap-2'>
                  {(
                    ttsCatalog?.voice_modes ?? [
                      {
                        available: true,
                        description: 'Upload a WAV reference clip and clone its voice with the Base checkpoint.',
                        label: 'Reference Voice',
                        mode: 'reference_clone' as TtsVoiceMode,
                      },
                      {
                        available: presetVoiceAvailable,
                        description: presetVoiceAvailable
                          ? 'Use built-in preset voices from the optional CustomVoice checkpoint.'
                          : 'Preset voices require the optional CustomVoice checkpoint.',
                        label: 'Preset Voice Library',
                        mode: 'preset_voice' as TtsVoiceMode,
                      },
                    ]
                  ).map((voiceMode) => {
                    const active = ttsVoiceMode === voiceMode.mode;
                    return (
                      <button
                        className={`rounded-xl border px-3 py-3 text-left transition ${
                          active
                            ? 'border-cyan-400 bg-cyan-50'
                            : voiceMode.available
                              ? 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'
                              : 'cursor-not-allowed border-slate-200 bg-slate-50 text-slate-400'
                        }`}
                        disabled={!voiceMode.available || busy !== null}
                        key={voiceMode.mode}
                        onClick={() => setTtsVoiceMode(voiceMode.mode)}
                        type='button'
                      >
                        <div className='flex items-center justify-between gap-2'>
                          <span className='text-sm font-semibold text-slate-950'>{voiceMode.label}</span>
                          <span
                            className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.12em] ${
                              voiceMode.available ? 'bg-emerald-50 text-emerald-700' : 'bg-slate-100 text-slate-500'
                            }`}
                          >
                            {voiceMode.available ? 'Ready' : 'Unavailable'}
                          </span>
                        </div>
                        <p className='mt-1 text-xs leading-5 text-slate-500'>{voiceMode.description}</p>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            {isReferenceVoiceMode && (
              <label className='space-y-2'>
                <span className='flex items-center gap-2 text-sm font-medium text-slate-700'>
                  <Upload className='h-3.5 w-3.5 text-slate-400' />
                  Reference Voice
                </span>
                <div className='rounded-xl border border-dashed border-slate-300 bg-slate-50 px-3 py-3'>
                  <div className='flex items-center justify-between gap-3'>
                    <div className='min-w-0'>
                      <p className='truncate text-sm font-medium text-slate-900'>
                        {ttsReferenceFile ? ttsReferenceFile.name : 'Required WAV reference audio'}
                      </p>
                      <p className='text-xs text-slate-500'>
                        {ttsReferenceFile
                          ? fmtBytes(ttsReferenceFile.size)
                          : 'Upload a clean sample to clone the source voice'}
                      </p>
                    </div>
                    <div className='flex items-center gap-2'>
                      {ttsReferenceFile && (
                        <Button onClick={() => setTtsReferenceFile(null)} size='sm' type='button' variant='ghost'>
                          Clear
                        </Button>
                      )}
                      <label className='cursor-pointer'>
                        <span className='inline-flex h-7 items-center justify-center rounded-[min(var(--radius-md),12px)] border border-border bg-background px-2.5 text-[0.8rem] font-medium text-foreground transition hover:bg-muted hover:text-foreground'>
                          {ttsReferenceFile ? 'Replace' : 'Upload'}
                        </span>
                        <input
                          accept='.wav,audio/wav'
                          className='sr-only'
                          disabled={busy !== null}
                          onChange={(e) => setTtsReferenceFile(e.target.files?.[0] ?? null)}
                          type='file'
                        />
                      </label>
                    </div>
                  </div>
                  {ttsReferenceAudioUrl && <audio className='mt-3 w-full' controls src={ttsReferenceAudioUrl} />}
                </div>
              </label>
            )}

            <label className='space-y-2'>
              <span className='text-sm font-medium text-slate-700'>Dialogue</span>
              <textarea
                className='min-h-36 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm leading-6 text-slate-900 outline-none transition focus:border-cyan-300 focus:ring-4 focus:ring-cyan-100'
                disabled={busy !== null}
                onChange={(e) => setTtsText(e.target.value)}
                placeholder='Enter the line you want Qwen3-TTS to perform.'
                value={ttsText}
              />
            </label>

            <div className='grid gap-4 md:grid-cols-3'>
              <label className='space-y-2'>
                <span className='text-sm font-medium text-slate-700'>Delivery</span>
                <select
                  className='w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm'
                  disabled={busy !== null}
                  onChange={(e) =>
                    setTtsDeliveryPresetId(e.target.value as (typeof TTS_DELIVERY_PRESETS)[number]['id'])
                  }
                  value={ttsDeliveryPresetId}
                >
                  {TTS_DELIVERY_PRESETS.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <p className='text-xs text-slate-500'>
                  {TTS_DELIVERY_PRESETS.find((item) => item.id === ttsDeliveryPresetId)?.description}
                </p>
              </label>

              <label className='space-y-2'>
                <span className='text-sm font-medium text-slate-700'>Tone</span>
                <select
                  className='w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm'
                  disabled={busy !== null}
                  onChange={(e) => setTtsTonePresetId(e.target.value as (typeof TTS_TONE_PRESETS)[number]['id'])}
                  value={ttsTonePresetId}
                >
                  {TTS_TONE_PRESETS.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <p className='text-xs text-slate-500'>
                  {TTS_TONE_PRESETS.find((item) => item.id === ttsTonePresetId)?.prompt}
                </p>
              </label>

              <label className='space-y-2'>
                <span className='text-sm font-medium text-slate-700'>Voice Hint</span>
                <select
                  className='w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm'
                  disabled={busy !== null}
                  onChange={(e) => setTtsTimbreHintId(e.target.value as (typeof TTS_TIMBRE_HINTS)[number]['id'])}
                  value={ttsTimbreHintId}
                >
                  {TTS_TIMBRE_HINTS.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <p className='text-xs text-slate-500'>
                  {TTS_TIMBRE_HINTS.find((item) => item.id === ttsTimbreHintId)?.prompt || 'No timbre hint'}
                </p>
              </label>
            </div>

            <label className='space-y-2'>
              <span className='text-sm font-medium text-slate-700'>Custom Direction</span>
              <textarea
                className='min-h-24 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm leading-6 text-slate-900 outline-none transition focus:border-cyan-300 focus:ring-4 focus:ring-cyan-100'
                disabled={busy !== null}
                onChange={(e) => setTtsPrompt(e.target.value)}
                placeholder='Optional extra direction. Example: slightly playful, late-night radio calm, more restrained ending'
                value={ttsPrompt}
              />
              <div className='rounded-xl border border-slate-200 bg-slate-50 px-3 py-3'>
                <p className='text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500'>
                  Effective Direction
                </p>
                <p className='mt-2 text-sm leading-6 text-slate-700'>
                  {effectiveTtsDirection || 'No additional direction. The model will use plain generation defaults.'}
                </p>
              </div>
            </label>

            <div className='flex flex-wrap items-start gap-3 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm leading-6 text-slate-600'>
              <span className='rounded-full bg-white px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-500'>
                {isReferenceVoiceMode ? 'Reference Clone' : 'Preset Voice'}
              </span>
              <p className='flex-1 min-w-[220px]'>
                {isReferenceVoiceMode
                  ? ttsReferenceFile
                    ? `Using ${ttsReferenceFile.name} as the clone reference.`
                    : 'Upload a WAV clip to enable reference clone generation.'
                  : selectedTtsActor
                    ? `Using ${selectedTtsActor.label}. Only voices matching ${langLabel(ttsLanguage)} can be selected.`
                    : 'Select a preset voice that matches the current output language.'}
              </p>
              {ttsCatalog?.preset_actor_message && !presetVoiceAvailable && (
                <p className='basis-full text-amber-900'>{ttsCatalog.preset_actor_message}</p>
              )}
            </div>

            {isPresetVoiceMode ? (
              <div className='space-y-3'>
                <div className='flex items-center justify-between gap-3'>
                  <div>
                    <h2 className='text-sm font-medium text-slate-900'>Preset Voice Library</h2>
                    <p className='text-xs text-slate-500'>
                      All built-in voices are shown. Only voices matching the current output language can be selected.
                    </p>
                  </div>
                  {ttsCatalogError && <span className='text-xs text-red-600'>{ttsCatalogError}</span>}
                </div>

                {ttsPreviewError && (
                  <p className='rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700'>
                    {ttsPreviewError}
                  </p>
                )}

                <div className='grid gap-3 sm:grid-cols-2 xl:grid-cols-4'>
                  {ttsActors.map((actor) => {
                    const previewVariant = resolveTtsPreviewVariant(actor, null);
                    const previewKey = previewVariant
                      ? ttsPreviewCacheKey(actor.actor_id, previewVariant.preview_id)
                      : null;
                    const previewSrc = previewKey ? ttsPreviewAudioByKey[previewKey] : undefined;
                    const active = selectedTtsActor?.actor_id === actor.actor_id;
                    const matchesLanguage = actor.language === ttsLanguage;
                    return (
                      <div
                        className={`rounded-2xl border p-4 text-left transition ${
                          active
                            ? 'border-cyan-400 bg-cyan-50 shadow-sm'
                            : matchesLanguage
                              ? 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'
                              : 'border-slate-200 bg-slate-50 text-slate-500'
                        }`}
                        key={actor.actor_id}
                        onClick={() => {
                          if (matchesLanguage) setTtsActorId(actor.actor_id);
                        }}
                        onKeyDown={(e) => {
                          if ((e.key === 'Enter' || e.key === ' ') && matchesLanguage) {
                            e.preventDefault();
                            setTtsActorId(actor.actor_id);
                          }
                        }}
                        role='button'
                        tabIndex={0}
                      >
                        <div className='flex items-center justify-between gap-3'>
                          <div>
                            <div className='flex items-center gap-2'>
                              <span className='text-sm font-semibold text-slate-950'>{actor.label}</span>
                              {actor.is_default && (
                                <span className='rounded-full bg-slate-900 px-1.5 py-0.5 text-[10px] font-semibold tracking-wide text-white'>
                                  DEFAULT
                                </span>
                              )}
                            </div>
                            <p className='mt-1 text-xs leading-5 text-slate-500'>
                              {langLabel(actor.language)}
                              {matchesLanguage
                                ? ' ready for this output language'
                                : ` only available for ${langLabel(actor.language)}`}
                            </p>
                          </div>
                          <Button
                            disabled={!presetVoiceAvailable || !matchesLanguage}
                            onClick={(e) => {
                              e.stopPropagation();
                              void previewTtsActor(actor.actor_id);
                            }}
                            size='sm'
                            type='button'
                            variant='outline'
                          >
                            {ttsPreviewBusyKey === previewKey ? (
                              <>
                                <LoaderCircle className='h-3.5 w-3.5 animate-spin' />
                                Loading
                              </>
                            ) : (
                              <>
                                <Play className='h-3.5 w-3.5' />
                                Preview
                              </>
                            )}
                          </Button>
                        </div>

                        <div className='mt-3 text-xs font-medium text-slate-600'>
                          {active
                            ? 'Selected preset voice'
                            : matchesLanguage
                              ? 'Click to select this voice'
                              : 'Switch output language to select'}
                        </div>

                        {previewSrc && <audio className='mt-3 w-full' controls src={previewSrc} />}
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className='rounded-xl border border-slate-200 bg-slate-50 px-4 py-4 text-sm leading-6 text-slate-600'>
                Reference Voice mode ignores preset speakers completely. If you want to use built-in voices instead of
                cloning an uploaded sample, switch to Preset Voice and choose one from the library.
              </div>
            )}
          </div>
        )}

        {/* Error */}
        {error && <p className='rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700'>{error}</p>}

        {/* Run */}
        <Button className='w-full' disabled={mode === 'tts' ? !canRunTts : !canRunPipeline} size='lg' type='submit'>
          {busy === mode ? (
            <>
              <LoaderCircle className='h-4 w-4 animate-spin' />
              {cfg.running}
            </>
          ) : (
            <>
              {cfg.action}
              <ArrowRight className='h-4 w-4' />
            </>
          )}
        </Button>
      </form>

      {/* ── Results ── */}
      <section className='mt-8 border-t border-slate-200 pt-6'>
        <Tabs onValueChange={(v) => setTab(v as DemoMode)} value={tab}>
          <TabsList className='mb-6 grid w-full grid-cols-4'>
            <TabsTrigger value='vad'>
              VAD
              {vad ? <span className='ml-1.5 inline-block h-1.5 w-1.5 rounded-full bg-emerald-500' /> : null}
            </TabsTrigger>
            <TabsTrigger value='stt'>
              STT
              {sttRes ? <span className='ml-1.5 inline-block h-1.5 w-1.5 rounded-full bg-emerald-500' /> : null}
            </TabsTrigger>
            <TabsTrigger value='localize'>
              Voice Dub
              {loc ? <span className='ml-1.5 inline-block h-1.5 w-1.5 rounded-full bg-emerald-500' /> : null}
            </TabsTrigger>
            <TabsTrigger value='tts'>
              TTS Studio
              {ttsRes ? <span className='ml-1.5 inline-block h-1.5 w-1.5 rounded-full bg-emerald-500' /> : null}
            </TabsTrigger>
          </TabsList>

          {/* ─ VAD ─ */}
          <TabsContent value='vad'>
            {vad ? (
              <div className='space-y-6'>
                <div className='grid grid-cols-2 gap-3 sm:grid-cols-4'>
                  <Metric detail={fmtMs(vad.duration_ms)} label='File' value={vad.filename} />
                  <Metric
                    detail={`threshold ${vad.threshold.toFixed(2)}`}
                    label='Segments'
                    value={String(vad.segment_count)}
                  />
                  <Metric
                    detail={`${vad.sample_rate} Hz`}
                    label='Speech windows'
                    value={`${speechWins}/${vad.window_scores.length}`}
                  />
                  <Metric detail={vad.model} label='Window size' value={fmtMs(vad.window_ms)} />
                </div>

                {scores.length > 0 && (
                  <div>
                    <h3 className='mb-2 text-sm font-medium text-slate-900'>Window scores</h3>
                    <div
                      className='grid h-32 items-end gap-px rounded-lg border border-slate-200 bg-slate-50 p-3'
                      style={{
                        gridTemplateColumns: `repeat(${scores.length}, minmax(0, 1fr))`,
                      }}
                    >
                      {scores.map((s, i) => (
                        <div className='flex h-full items-end' key={`${i}-${s}`} title={`${i + 1}: ${s.toFixed(4)}`}>
                          <div
                            className={`w-full rounded-t-sm ${s >= vad.threshold ? 'bg-cyan-500' : 'bg-slate-300'}`}
                            style={{ height: `${Math.max(4, Math.min(100, s * 100))}%` }}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {vad.segments.length > 0 ? (
                  <div>
                    <h3 className='mb-2 text-sm font-medium text-slate-900'>Detected segments</h3>
                    <div className='space-y-2'>
                      {vad.segments.map((seg) => (
                        <div
                          className='rounded-lg border border-slate-200 bg-white px-4 py-3'
                          key={`${seg.start_ms}-${seg.end_ms}`}
                        >
                          <div className='flex items-center justify-between text-sm'>
                            <span className='font-medium text-slate-950'>
                              {fmtMs(seg.start_ms)} &ndash; {fmtMs(seg.end_ms)}
                            </span>
                            <span className='text-xs text-slate-500'>{fmtMs(seg.duration_ms)}</span>
                          </div>
                          <div className='mt-2 grid gap-2 sm:grid-cols-2'>
                            <Bar label='Avg' tone='cyan' value={seg.average_probability} />
                            <Bar label='Peak' tone='emerald' value={seg.peak_probability} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <p className='text-sm text-slate-500'>No speech crossed the threshold.</p>
                )}
              </div>
            ) : (
              <EmptyState>Run VAD to detect speech segments.</EmptyState>
            )}
          </TabsContent>

          {/* ─ STT ─ */}
          <TabsContent value='stt'>
            {sttRes ? (
              <div className='space-y-6'>
                <div className='grid grid-cols-2 gap-3 sm:grid-cols-4'>
                  <Metric detail={sttRes.task} label='Language' value={langLabel(sttRes.language)} />
                  <Metric
                    detail={`threshold ${sttRes.threshold.toFixed(2)}`}
                    label='Segments'
                    value={String(sttRes.segment_count)}
                  />
                  <Metric detail={`${sttRes.sample_rate} Hz`} label='Duration' value={fmtMs(sttRes.duration_ms)} />
                  <Metric detail={sttRes.model} label='Model' value={sttRes.repository_model_name} />
                </div>

                <div>
                  <h3 className='mb-2 text-sm font-medium text-slate-900'>Transcript</h3>
                  <div className='rounded-lg border border-slate-200 bg-white p-4'>
                    <p className='whitespace-pre-wrap text-sm leading-7 text-slate-800'>
                      {sttRes.transcript || 'No transcript returned.'}
                    </p>
                  </div>
                </div>

                {sttRes.segments.length > 0 && (
                  <div>
                    <h3 className='mb-2 text-sm font-medium text-slate-900'>Segments</h3>
                    <div className='space-y-2'>
                      {sttRes.segments.map((seg) => (
                        <div
                          className='rounded-lg border border-slate-200 bg-white px-4 py-3'
                          key={`${seg.start_ms}-${seg.end_ms}`}
                        >
                          <div className='flex items-center justify-between text-sm'>
                            <span className='font-medium text-slate-950'>
                              {fmtMs(seg.start_ms)} &ndash; {fmtMs(seg.end_ms)}
                            </span>
                            <span className='text-xs text-slate-500'>{fmtMs(seg.duration_ms)}</span>
                          </div>
                          <p className='mt-2 rounded-md bg-slate-50 px-3 py-2 text-sm leading-6 text-slate-700'>
                            {seg.text || '\u2014'}
                          </p>
                          <div className='mt-2 grid gap-2 sm:grid-cols-2'>
                            <Bar label='Avg' tone='cyan' value={seg.average_probability} />
                            <Bar label='Peak' tone='emerald' value={seg.peak_probability} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <EmptyState>Run STT to transcribe speech.</EmptyState>
            )}
          </TabsContent>

          {/* ─ Localize ─ */}
          <TabsContent value='localize'>
            {loc ? (
              <div className='space-y-6'>
                <div className='grid grid-cols-2 gap-3 sm:grid-cols-4'>
                  <Metric label='Source' value={langLabel(loc.source_language)} />
                  <Metric label='Target' value={langLabel(loc.target_language)} />
                  <Metric label='Status' tone={loc.status === 'ok' ? 'ready' : 'warning'} value={loc.status} />
                  <Metric
                    label='Total'
                    value={fmtMs((loc as Record<string, unknown>).elapsed_ms as number | undefined)}
                  />
                </div>

                <div className='grid gap-2 sm:grid-cols-3'>
                  <Stage
                    label='STT'
                    model={loc.models.stt}
                    status={loc.stages.stt.status}
                    elapsed={loc.stages.stt.elapsed_ms}
                    detail={
                      loc.stages.stt.speaker_count && loc.stages.stt.speaker_count > 1
                        ? `${loc.stages.stt.speaker_count} speakers`
                        : undefined
                    }
                  />
                  <Stage
                    label='Translation'
                    model={loc.models.translation}
                    status={loc.stages.translation.status}
                    elapsed={loc.stages.translation.elapsed_ms}
                  />
                  <Stage
                    label='TTS'
                    model={loc.models.tts}
                    status={loc.stages.tts.status}
                    elapsed={loc.stages.tts.elapsed_ms}
                    detail={
                      loc.stages.tts.voice_cloning
                        ? `VC: ${loc.stages.tts.voice_cloning_mode}${loc.stages.tts.speaker_count && loc.stages.tts.speaker_count > 1 ? ` (${loc.stages.tts.speaker_count})` : ''}`
                        : undefined
                    }
                  />
                </div>

                {loc.message && (
                  <p className='rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700'>
                    {loc.message}
                  </p>
                )}

                <div className='grid gap-4 lg:grid-cols-2'>
                  <div>
                    <h3 className='mb-2 text-sm font-medium text-slate-900'>Transcript</h3>
                    <div className='rounded-lg border border-slate-200 bg-white p-4'>
                      <p className='whitespace-pre-wrap text-sm leading-7 text-slate-800'>
                        {loc.transcript || '\u2014'}
                      </p>
                    </div>
                  </div>
                  <div>
                    <h3 className='mb-2 text-sm font-medium text-slate-900'>Translation</h3>
                    <div className='rounded-lg border border-slate-200 bg-white p-4'>
                      <p className='whitespace-pre-wrap text-sm leading-7 text-slate-800'>
                        {loc.translated_text || '\u2014'}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Audio comparison */}
                <div className='space-y-3'>
                  <h3 className='text-sm font-medium text-slate-900'>Audio Comparison</h3>
                  <div className='grid gap-3 lg:grid-cols-2'>
                    {/* Original */}
                    <div className='rounded-lg border border-slate-200 bg-white p-4'>
                      <div className='mb-2 flex items-center justify-between'>
                        <span className='text-sm font-medium text-slate-700'>Original</span>
                        <span className='text-xs text-slate-500'>
                          {langLabel(loc.source_language)} &middot; {fmtMs(loc.duration_ms)}
                        </span>
                      </div>
                      {originalAudioUrl ? (
                        <audio className='w-full' controls src={originalAudioUrl} />
                      ) : (
                        <div className='rounded-md border border-dashed border-slate-200 bg-slate-50 px-3 py-3 text-center text-xs text-slate-400'>
                          Upload another file to compare
                        </div>
                      )}
                    </div>
                    {/* Synthesized */}
                    <div
                      className={`rounded-lg border p-4 ${audioSrc ? 'border-cyan-200 bg-cyan-50/50' : 'border-slate-200 bg-white'}`}
                    >
                      <div className='mb-2 flex items-center justify-between'>
                        <span className='text-sm font-medium text-slate-700'>
                          Dubbed
                          {loc.stages.tts.voice_cloning && (
                            <span className='ml-1.5 rounded-full bg-cyan-100 px-1.5 py-0.5 text-xs text-cyan-700'>
                              VC
                            </span>
                          )}
                        </span>
                        <span className='text-xs text-slate-500'>
                          {langLabel(loc.target_language)}
                          {loc.stages.tts.sample_rate ? ` \u00b7 ${loc.stages.tts.sample_rate} Hz` : ''}
                          {loc.stages.tts.duration_ms ? ` \u00b7 ${fmtMs(loc.stages.tts.duration_ms)}` : ''}
                        </span>
                      </div>
                      {audioSrc ? (
                        <audio className='w-full' controls src={audioSrc} />
                      ) : (
                        <div className='rounded-md border border-dashed border-slate-200 bg-slate-50 px-3 py-3 text-center text-xs text-slate-400'>
                          {loc.stages.tts.message ?? loc.stages.tts.reason ?? 'No audio preview available'}
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Subtitle download */}
                {loc.stages.stt.segments && loc.stages.stt.segments.length > 0 && (
                  <div>
                    <h3 className='mb-2 text-sm font-medium text-slate-900'>Download Subtitles</h3>
                    <div className='flex gap-2'>
                      {(['srt', 'vtt', 'csv'] as const).map((fmt) => (
                        <button
                          key={fmt}
                          className='flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50'
                          onClick={() => downloadSubtitle(fmt, loc, workerBaseUrl)}
                          type='button'
                        >
                          <Download className='h-3.5 w-3.5' />
                          {fmt.toUpperCase()}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <EmptyState>Run Voice Dub for end-to-end speech translation with voice cloning.</EmptyState>
            )}
          </TabsContent>

          {/* ─ TTS ─ */}
          <TabsContent value='tts'>
            {ttsRes ? (
              <div className='space-y-6'>
                <div className='grid grid-cols-2 gap-3 sm:grid-cols-4'>
                  <Metric label='Language' value={langLabel(ttsRes.language)} />
                  <Metric detail={ttsRes.voice_mode} label='Voice source' value={ttsRes.voice_source_label} />
                  <Metric detail={`${ttsRes.sample_rate} Hz`} label='Duration' value={fmtMs(ttsRes.duration_ms)} />
                  <Metric detail={ttsRes.model} label='Model' value={ttsRes.repository_model_name} />
                </div>

                <div className='grid gap-4 lg:grid-cols-[minmax(0,1.15fr)_minmax(320px,0.85fr)]'>
                  <div className='space-y-4'>
                    <div>
                      <h3 className='mb-2 text-sm font-medium text-slate-900'>Generated Dialogue</h3>
                      <div className='rounded-lg border border-slate-200 bg-white p-4'>
                        <p className='whitespace-pre-wrap text-sm leading-7 text-slate-800'>{ttsRes.text}</p>
                      </div>
                    </div>

                    {ttsReferenceAudioUrl && (
                      <div>
                        <h3 className='mb-2 text-sm font-medium text-slate-900'>Reference Voice</h3>
                        <div className='rounded-lg border border-slate-200 bg-white p-4'>
                          <div className='mb-2 flex items-center justify-between text-xs text-slate-500'>
                            <span>{ttsRes.reference_audio_filename ?? ttsReferenceFile?.name ?? 'reference.wav'}</span>
                            <span>16 kHz clone input</span>
                          </div>
                          <audio className='w-full' controls src={ttsReferenceAudioUrl} />
                        </div>
                      </div>
                    )}
                  </div>

                  <div
                    className={`rounded-2xl border p-4 ${ttsAudioSrc ? 'border-cyan-200 bg-cyan-50/60' : 'border-slate-200 bg-white'}`}
                  >
                    <div className='flex items-center justify-between gap-3'>
                      <div>
                        <h3 className='text-sm font-medium text-slate-900'>Generated Audio</h3>
                        <p className='mt-1 text-xs text-slate-500'>{ttsRes.voice_source_label}</p>
                      </div>
                      {ttsAudioSrc && (
                        <Button
                          onClick={() =>
                            downloadAudioDataUrl(`tts-${ttsRes.language}-${ttsRes.voice_mode}.wav`, ttsAudioSrc)
                          }
                          size='sm'
                          type='button'
                          variant='outline'
                        >
                          <Download className='h-3.5 w-3.5' />
                          Download
                        </Button>
                      )}
                    </div>

                    {ttsAudioSrc ? (
                      <div className='mt-4 space-y-4'>
                        <audio className='w-full' controls src={ttsAudioSrc} />
                        <div className='grid grid-cols-2 gap-3'>
                          <Metric label='Voice mode' value={ttsRes.voice_mode} />
                          <Metric label='Voice source' value={ttsRes.voice_source_label} />
                        </div>
                      </div>
                    ) : (
                      <div className='mt-4 rounded-md border border-dashed border-slate-200 bg-slate-50 px-3 py-3 text-center text-xs text-slate-400'>
                        Generated audio was empty.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <EmptyState>
                Generate single-line TTS with either an uploaded reference voice or a preset voice library.
              </EmptyState>
            )}
          </TabsContent>
        </Tabs>
      </section>
    </main>
  );
}

/* ------------------------------------------------------------------ */
/*  Shared components                                                  */
/* ------------------------------------------------------------------ */

function Metric({
  detail,
  label,
  tone,
  value,
}: {
  detail?: string;
  label: string;
  tone?: 'ready' | 'warning';
  value: string;
}) {
  return (
    <div
      className={`rounded-lg border p-3 ${
        tone === 'ready'
          ? 'border-emerald-200 bg-emerald-50'
          : tone === 'warning'
            ? 'border-amber-200 bg-amber-50'
            : 'border-slate-200 bg-white'
      }`}
    >
      <div className='text-xs text-slate-500'>{label}</div>
      <div className='mt-1 truncate text-sm font-semibold text-slate-950'>{value}</div>
      {detail && <div className='mt-0.5 truncate text-xs text-slate-500'>{detail}</div>}
    </div>
  );
}

function Bar({ label, tone, value }: { label: string; tone: 'cyan' | 'emerald'; value: number }) {
  return (
    <div className='flex items-center gap-2'>
      <span className='w-8 shrink-0 text-xs text-slate-500'>{label}</span>
      <div className='h-1.5 flex-1 rounded-full bg-slate-200'>
        <div
          className={`h-1.5 rounded-full ${tone === 'emerald' ? 'bg-emerald-500' : 'bg-cyan-500'}`}
          style={{ width: `${Math.min(100, value * 100)}%` }}
        />
      </div>
      <span className='w-8 shrink-0 text-right text-xs tabular-nums text-slate-600'>{fmtProb(value)}</span>
    </div>
  );
}

function Stage({
  label,
  model,
  status,
  elapsed,
  detail,
}: {
  label: string;
  model: string;
  status: string;
  elapsed?: number;
  detail?: string;
}) {
  const ok = status === 'ok';
  return (
    <div
      className={`rounded-lg border p-3 ${ok ? 'border-emerald-200 bg-emerald-50' : 'border-slate-200 bg-slate-50'}`}
    >
      <div className='flex items-center justify-between'>
        <span className='text-sm font-medium text-slate-950'>{label}</span>
        <div className='flex items-center gap-2'>
          {elapsed != null && <span className='text-xs tabular-nums text-slate-400'>{fmtMs(elapsed)}</span>}
          <span className={`text-xs font-medium ${ok ? 'text-emerald-700' : 'text-slate-500'}`}>{status}</span>
        </div>
      </div>
      <div className='mt-1 flex items-center gap-2'>
        <span className='truncate text-xs text-slate-500'>{model}</span>
        {detail && (
          <span className='shrink-0 rounded-full bg-cyan-100 px-1.5 py-0.5 text-xs text-cyan-700'>{detail}</span>
        )}
      </div>
    </div>
  );
}

function EmptyState({ children }: { children: React.ReactNode }) {
  return (
    <div className='rounded-lg border border-dashed border-slate-300 bg-slate-50 px-4 py-12 text-center text-sm text-slate-500'>
      {children}
    </div>
  );
}
