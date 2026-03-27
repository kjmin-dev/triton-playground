import { createFileRoute } from '@tanstack/react-router';
import type { LucideIcon } from 'lucide-react';
import { ArrowRight, Globe2, Languages, LoaderCircle, MicVocal, Sparkles, Upload, Waves } from 'lucide-react';
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

type DemoMode = 'vad' | 'stt' | 'localize';

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
      language?: string;
      message?: string;
      segment_count?: number;
      status: string;
      task?: string;
      transcript?: string;
    };
    translation: {
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
      language?: string;
      message?: string;
      reason?: string;
      sample_rate?: number;
      status: string;
    };
  };
  status: string;
  target_language: string;
  threshold: number;
  transcript: string;
  translated_text: string;
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

  const thr = Number.parseFloat(threshold);
  const thrOk = Number.isFinite(thr) && thr >= 0.1 && thr <= 0.99;
  const live = Boolean(ready?.triton.server_ready && ready?.triton.model_ready);
  const cfg = MODES.find((m) => m.value === mode) ?? MODES[0];

  const [dragging, setDragging] = React.useState(false);

  const originalAudioUrl = React.useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  React.useEffect(() => {
    return () => {
      if (originalAudioUrl) URL.revokeObjectURL(originalAudioUrl);
    };
  }, [originalAudioUrl]);

  const audioSrc =
    loc?.stages.tts.audio_base64 && loc.stages.tts.content_type
      ? `data:${loc.stages.tts.content_type};base64,${loc.stages.tts.audio_base64}`
      : null;
  const scores = vad?.window_scores.slice(0, SCORE_LIMIT) ?? [];
  const speechWins = vad && thrOk ? vad.window_scores.filter((s) => s >= thr).length : 0;

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

  async function run(route: DemoMode) {
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
          Upload a WAV file, pick a mode, and run inference. VAD detects speech, STT transcribes it, Voice Dub
          translates and re-voices in one pass.
        </p>
      </div>

      {/* ── Controls ── */}
      <form
        className='mt-5 space-y-4'
        onSubmit={(e) => {
          e.preventDefault();
          void run(mode);
        }}
      >
        {/* Upload (click or drag & drop) */}
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

        {/* Mode + params */}
        <div className='space-y-4 rounded-xl border border-slate-200 bg-white/70 p-4 backdrop-blur'>
          {/* Mode segmented control */}
          <div className='flex gap-1 rounded-lg bg-slate-100 p-1'>
            {MODES.map((m) => {
              const Icon = m.icon;
              const active = mode === m.value;
              const done =
                (m.value === 'vad' && vad) || (m.value === 'stt' && sttRes) || (m.value === 'localize' && loc);
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

          {/* Threshold */}
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

          {/* Language selects */}
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

        {/* Error */}
        {error && <p className='rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700'>{error}</p>}

        {/* Run */}
        <Button className='w-full' disabled={!file || !thrOk || busy !== null} size='lg' type='submit'>
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
          <TabsList className='mb-6 grid w-full grid-cols-3'>
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
                    elapsed={(loc.stages.stt as Record<string, unknown>).elapsed_ms as number | undefined}
                  />
                  <Stage
                    label='Translation'
                    model={loc.models.translation}
                    status={loc.stages.translation.status}
                    elapsed={(loc.stages.translation as Record<string, unknown>).elapsed_ms as number | undefined}
                  />
                  <Stage
                    label='TTS'
                    model={loc.models.tts}
                    status={loc.stages.tts.status}
                    elapsed={(loc.stages.tts as Record<string, unknown>).elapsed_ms as number | undefined}
                    detail={
                      (loc.stages.tts as Record<string, unknown>).voice_cloning
                        ? `VC: ${(loc.stages.tts as Record<string, unknown>).voice_cloning_mode}`
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
                          {Boolean((loc.stages.tts as Record<string, unknown>).voice_cloning) && (
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
              </div>
            ) : (
              <EmptyState>Run Voice Dub for end-to-end speech translation with voice cloning.</EmptyState>
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
