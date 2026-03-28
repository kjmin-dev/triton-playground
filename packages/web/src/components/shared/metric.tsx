export function Metric({
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
