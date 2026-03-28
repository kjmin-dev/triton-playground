import { fmtMs, langLabel } from '@/lib/format';
import type { SttResponse } from '@/types/api';

import { Bar } from '../shared/bar';
import { EmptyState } from '../shared/empty-state';
import { Metric } from '../shared/metric';

export function SttResults({ sttRes }: { sttRes: SttResponse | null }) {
  if (!sttRes) return <EmptyState>Run STT to transcribe speech.</EmptyState>;

  return (
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
  );
}
