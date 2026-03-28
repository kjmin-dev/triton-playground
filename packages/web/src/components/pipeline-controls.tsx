import { Globe2, Languages } from 'lucide-react';

import { STT_LANGUAGE_OPTIONS, TARGET_LANGUAGE_OPTIONS, THRESHOLD_PRESETS } from '@/constants';
import type { DemoMode } from '@/types/api';

export function PipelineControls({
  mode,
  threshold,
  onThresholdChange,
  srcLang,
  onSrcLangChange,
  tgtLang,
  onTgtLangChange,
  disabled,
}: {
  mode: Exclude<DemoMode, 'tts'>;
  threshold: string;
  onThresholdChange: (value: string) => void;
  srcLang: string;
  onSrcLangChange: (value: string) => void;
  tgtLang: string;
  onTgtLangChange: (value: string) => void;
  disabled: boolean;
}) {
  return (
    <div className='space-y-4 rounded-xl border border-slate-200 bg-white/70 p-4 backdrop-blur'>
      <div className='space-y-2'>
        <div className='flex items-center gap-3'>
          <span className='w-16 shrink-0 text-sm font-medium text-slate-700'>Threshold</span>
          <input
            className='flex-1 accent-slate-950'
            disabled={disabled}
            max='0.99'
            min='0.10'
            onChange={(e) => onThresholdChange(e.target.value)}
            step='0.01'
            type='range'
            value={threshold}
          />
          <input
            className='w-16 rounded-lg border border-slate-200 px-2 py-1.5 text-center text-sm'
            disabled={disabled}
            max='0.99'
            min='0.10'
            onChange={(e) => onThresholdChange(e.target.value)}
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
              disabled={disabled}
              onClick={() => onThresholdChange(p.toFixed(2))}
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
              disabled={disabled}
              onChange={(e) => onSrcLangChange(e.target.value)}
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
                disabled={disabled}
                onChange={(e) => onTgtLangChange(e.target.value)}
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
  );
}
