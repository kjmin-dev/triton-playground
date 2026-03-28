import { Upload } from 'lucide-react';
import { useState } from 'react';

import { MAX_BYTES } from '@/constants';
import { fmtBytes } from '@/lib/format';

export function AudioUpload({
  file,
  onFileChange,
  disabled,
}: {
  file: File | null;
  onFileChange: (file: File | null) => void;
  disabled: boolean;
}) {
  const [dragging, setDragging] = useState(false);

  function handleFileDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) onFileChange(dropped);
  }

  return (
    <label
      className={`flex cursor-pointer items-center gap-3 rounded-xl border-2 border-dashed px-4 py-5 backdrop-blur transition ${
        dragging
          ? 'border-cyan-400 bg-cyan-50/80'
          : file
            ? 'border-emerald-300 bg-emerald-50/50 hover:border-emerald-400'
            : 'border-slate-300 bg-white/70 hover:border-slate-400 hover:bg-white/90'
      }`}
      onDragEnter={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(e) => {
        e.preventDefault();
        setDragging(false);
      }}
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleFileDrop}
    >
      <div className={`rounded-lg p-2 ${dragging ? 'bg-cyan-100' : 'bg-slate-100'}`}>
        <Upload className={`h-4 w-4 ${dragging ? 'text-cyan-600' : 'text-slate-500'}`} />
      </div>
      <div className='min-w-0 flex-1'>
        <div className='truncate text-sm font-medium text-slate-900'>
          {dragging ? 'Drop audio file here' : file ? file.name : 'Upload audio file'}
        </div>
        <div className='text-xs text-slate-500'>
          {file ? fmtBytes(file.size) : `Drag & drop or click \u00b7 WAV \u00b7 max ${fmtBytes(MAX_BYTES)}`}
        </div>
      </div>
      {file && (
        <span className='shrink-0 rounded-full bg-emerald-50 px-2 py-0.5 text-xs font-medium text-emerald-700'>
          Loaded
        </span>
      )}
      <input
        accept='.wav,audio/wav'
        className='sr-only'
        disabled={disabled}
        onChange={(e) => onFileChange(e.target.files?.[0] ?? null)}
        type='file'
      />
    </label>
  );
}
