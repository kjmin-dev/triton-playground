import { fmtMs } from '@/lib/format';

export function Stage({
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
