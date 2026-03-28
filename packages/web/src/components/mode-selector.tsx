import { MODES } from '@/constants';
import type { DemoMode, LocalizeResponse, SttResponse, TtsResponse, VadResponse } from '@/types/api';

export function ModeSelector({
  mode,
  onModeChange,
  disabled,
  results,
}: {
  mode: DemoMode;
  onModeChange: (mode: DemoMode) => void;
  disabled: boolean;
  results: {
    vad: VadResponse | null;
    stt: SttResponse | null;
    localize: LocalizeResponse | null;
    tts: TtsResponse | null;
  };
}) {
  return (
    <div className='flex gap-1 rounded-lg bg-slate-100 p-1'>
      {MODES.map((m) => {
        const Icon = m.icon;
        const active = mode === m.value;
        const done =
          (m.value === 'vad' && results.vad) ||
          (m.value === 'stt' && results.stt) ||
          (m.value === 'localize' && results.localize) ||
          (m.value === 'tts' && results.tts);
        return (
          <button
            key={m.value}
            className={`flex flex-1 items-center justify-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition ${
              active ? 'bg-white text-slate-950 shadow-sm' : 'text-slate-500 hover:text-slate-700'
            }`}
            disabled={disabled}
            onClick={() => onModeChange(m.value)}
            type='button'
          >
            <Icon className='h-3.5 w-3.5' />
            {m.label}
            {done ? <span className='h-1.5 w-1.5 rounded-full bg-emerald-500' /> : null}
          </button>
        );
      })}
    </div>
  );
}
