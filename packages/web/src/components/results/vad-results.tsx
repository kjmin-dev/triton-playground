import { SCORE_LIMIT } from '@/constants';
import { fmtMs } from '@/lib/format';
import type { VadResponse } from '@/types/api';

import { Bar } from '../shared/bar';
import { EmptyState } from '../shared/empty-state';
import { Metric } from '../shared/metric';

export function VadResults({ vad, thrOk, thr }: { vad: VadResponse | null; thrOk: boolean; thr: number }) {
  if (!vad) return <EmptyState>Run VAD to detect speech segments.</EmptyState>;

  const scores = vad.window_scores.slice(0, SCORE_LIMIT);
  const speechWins = thrOk ? vad.window_scores.filter((s) => s >= thr).length : 0;

  return (
    <div className='space-y-6'>
      <div className='grid grid-cols-2 gap-3 sm:grid-cols-4'>
        <Metric detail={fmtMs(vad.duration_ms)} label='File' value={vad.filename} />
        <Metric detail={`threshold ${vad.threshold.toFixed(2)}`} label='Segments' value={String(vad.segment_count)} />
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
  );
}
