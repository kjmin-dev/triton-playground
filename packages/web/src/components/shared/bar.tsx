import { fmtProb } from '@/lib/format';

export function Bar({ label, tone, value }: { label: string; tone: 'cyan' | 'emerald'; value: number }) {
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
