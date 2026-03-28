import { createFileRoute } from '@tanstack/react-router';
import { ArrowRight, LoaderCircle } from 'lucide-react';
import { useState } from 'react';

import { AudioUpload } from '@/components/audio-upload';
import { ModeSelector } from '@/components/mode-selector';
import { PipelineControls } from '@/components/pipeline-controls';
import { ResultsPanel } from '@/components/results/results-panel';
import { TtsStudio } from '@/components/tts-studio';
import { Button } from '@/components/ui/button';
import { MODES } from '@/constants';
import { usePipeline } from '@/hooks/use-pipeline';
import { useTts } from '@/hooks/use-tts';
import { useWorkerReady } from '@/hooks/use-worker-ready';
import { buildAudioDataUrl } from '@/lib/audio';
import type { DemoMode } from '@/types/api';

export const Route = createFileRoute('/')({
  component: Home,
});

function Home() {
  const { live, error: readyError } = useWorkerReady();
  const pipeline = usePipeline();
  const tts = useTts();

  const [mode, setMode] = useState<DemoMode>('localize');
  const [tab, setTab] = useState<DemoMode>('localize');

  const busy = pipeline.busy ?? tts.busy;
  const error = pipeline.error ?? tts.error ?? readyError;
  const cfg = MODES.find((m) => m.value === mode) ?? MODES[0];
  const canRunPipeline = Boolean(pipeline.file) && pipeline.thrOk && busy === null;
  const canRunTts = tts.canRunTts && pipeline.busy === null;

  const audioSrc = buildAudioDataUrl(pipeline.loc?.stages.tts.content_type, pipeline.loc?.stages.tts.audio_base64);

  function clearError() {
    pipeline.setError(null);
    tts.setError(null);
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
            void tts.runTts();
            setTab('tts');
            return;
          }
          void pipeline.runPipeline(mode).then(setTab);
        }}
      >
        <div className='space-y-4 rounded-xl border border-slate-200 bg-white/70 p-4 backdrop-blur'>
          <ModeSelector
            mode={mode}
            onModeChange={(m) => {
              setMode(m);
              clearError();
            }}
            disabled={busy !== null}
            results={{
              vad: pipeline.vad,
              stt: pipeline.sttRes,
              localize: pipeline.loc,
              tts: tts.ttsRes,
            }}
          />
        </div>

        {mode !== 'tts' ? (
          <>
            <AudioUpload
              file={pipeline.file}
              onFileChange={(f) => {
                pipeline.setFile(f);
                clearError();
              }}
              disabled={busy !== null}
            />
            <PipelineControls
              mode={mode}
              threshold={pipeline.threshold}
              onThresholdChange={pipeline.setThreshold}
              srcLang={pipeline.srcLang}
              onSrcLangChange={pipeline.setSrcLang}
              tgtLang={pipeline.tgtLang}
              onTgtLangChange={pipeline.setTgtLang}
              disabled={busy !== null}
            />
          </>
        ) : (
          <TtsStudio
            disabled={busy !== null}
            ttsLanguage={tts.ttsLanguage}
            onTtsLanguageChange={tts.setTtsLanguage}
            ttsVoiceMode={tts.ttsVoiceMode}
            onTtsVoiceModeChange={tts.setTtsVoiceMode}
            ttsCatalog={tts.ttsCatalog}
            presetVoiceAvailable={tts.presetVoiceAvailable}
            isReferenceVoiceMode={tts.isReferenceVoiceMode}
            isPresetVoiceMode={tts.isPresetVoiceMode}
            ttsReferenceFile={tts.ttsReferenceFile}
            onTtsReferenceFileChange={tts.setTtsReferenceFile}
            ttsReferenceAudioUrl={tts.ttsReferenceAudioUrl}
            ttsText={tts.ttsText}
            onTtsTextChange={tts.setTtsText}
            ttsDeliveryPresetId={tts.ttsDeliveryPresetId}
            onTtsDeliveryPresetIdChange={tts.setTtsDeliveryPresetId}
            ttsTonePresetId={tts.ttsTonePresetId}
            onTtsTonePresetIdChange={tts.setTtsTonePresetId}
            ttsTimbreHintId={tts.ttsTimbreHintId}
            onTtsTimbreHintIdChange={tts.setTtsTimbreHintId}
            ttsPrompt={tts.ttsPrompt}
            onTtsPromptChange={tts.setTtsPrompt}
            effectiveTtsDirection={tts.effectiveTtsDirection}
            selectedTtsActor={tts.selectedTtsActor}
            ttsActorId={tts.ttsActorId}
            onTtsActorIdChange={tts.setTtsActorId}
            ttsActors={tts.ttsActors}
            ttsCatalogError={tts.ttsCatalogError}
          />
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
      <ResultsPanel
        tab={tab}
        onTabChange={setTab}
        vad={pipeline.vad}
        sttRes={pipeline.sttRes}
        loc={pipeline.loc}
        ttsRes={tts.ttsRes}
        thr={pipeline.thr}
        thrOk={pipeline.thrOk}
        originalAudioUrl={pipeline.originalAudioUrl}
        audioSrc={audioSrc}
        ttsAudioSrc={tts.ttsAudioSrc}
        ttsReferenceAudioUrl={tts.ttsReferenceAudioUrl}
        ttsReferenceFileName={tts.ttsReferenceFile?.name ?? null}
        workerBaseUrl={pipeline.workerBaseUrl}
      />
    </main>
  );
}
