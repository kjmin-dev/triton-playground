import { Download } from 'lucide-react';

import { downloadAudioDataUrl } from '@/lib/audio';
import { fmtMs, langLabel } from '@/lib/format';
import type { TtsResponse } from '@/types/api';

import { EmptyState } from '../shared/empty-state';
import { Metric } from '../shared/metric';
import { Button } from '../ui/button';

export function TtsResults({
  ttsRes,
  ttsAudioSrc,
  ttsReferenceAudioUrl,
  ttsReferenceFileName,
}: {
  ttsRes: TtsResponse | null;
  ttsAudioSrc: string | null;
  ttsReferenceAudioUrl: string | null;
  ttsReferenceFileName: string | null;
}) {
  if (!ttsRes) {
    return (
      <EmptyState>
        Generate single-line TTS with either an uploaded reference voice or a preset voice library.
      </EmptyState>
    );
  }

  return (
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
                  <span>{ttsRes.reference_audio_filename ?? ttsReferenceFileName ?? 'reference.wav'}</span>
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
                onClick={() => downloadAudioDataUrl(`tts-${ttsRes.language}-${ttsRes.voice_mode}.wav`, ttsAudioSrc)}
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
  );
}
