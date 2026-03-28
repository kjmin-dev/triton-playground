import { Download } from 'lucide-react';

import { downloadSubtitle } from '@/lib/audio';
import { fmtMs, langLabel } from '@/lib/format';
import type { LocalizeResponse } from '@/types/api';

import { EmptyState } from '../shared/empty-state';
import { Metric } from '../shared/metric';
import { Stage } from '../shared/stage';

export function LocalizeResults({
  loc,
  originalAudioUrl,
  audioSrc,
  workerBaseUrl,
}: {
  loc: LocalizeResponse | null;
  originalAudioUrl: string | null;
  audioSrc: string | null;
  workerBaseUrl: string;
}) {
  if (!loc) return <EmptyState>Run Voice Dub for end-to-end speech translation with voice cloning.</EmptyState>;

  return (
    <div className='space-y-6'>
      <div className='grid grid-cols-2 gap-3 sm:grid-cols-4'>
        <Metric label='Source' value={langLabel(loc.source_language)} />
        <Metric label='Target' value={langLabel(loc.target_language)} />
        <Metric label='Status' tone={loc.status === 'ok' ? 'ready' : 'warning'} value={loc.status} />
        <Metric label='Total' value={fmtMs((loc as Record<string, unknown>).elapsed_ms as number | undefined)} />
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
        <p className='rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700'>{loc.message}</p>
      )}

      <div className='grid gap-4 lg:grid-cols-2'>
        <div>
          <h3 className='mb-2 text-sm font-medium text-slate-900'>Transcript</h3>
          <div className='rounded-lg border border-slate-200 bg-white p-4'>
            <p className='whitespace-pre-wrap text-sm leading-7 text-slate-800'>{loc.transcript || '\u2014'}</p>
          </div>
        </div>
        <div>
          <h3 className='mb-2 text-sm font-medium text-slate-900'>Translation</h3>
          <div className='rounded-lg border border-slate-200 bg-white p-4'>
            <p className='whitespace-pre-wrap text-sm leading-7 text-slate-800'>{loc.translated_text || '\u2014'}</p>
          </div>
        </div>
      </div>

      {/* Audio comparison */}
      <div className='space-y-3'>
        <h3 className='text-sm font-medium text-slate-900'>Audio Comparison</h3>
        <div className='grid gap-3 lg:grid-cols-2'>
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
          <div
            className={`rounded-lg border p-4 ${audioSrc ? 'border-cyan-200 bg-cyan-50/50' : 'border-slate-200 bg-white'}`}
          >
            <div className='mb-2 flex items-center justify-between'>
              <span className='text-sm font-medium text-slate-700'>
                Dubbed
                {loc.stages.tts.voice_cloning && (
                  <span className='ml-1.5 rounded-full bg-cyan-100 px-1.5 py-0.5 text-xs text-cyan-700'>VC</span>
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
  );
}
