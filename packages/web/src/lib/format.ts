import { STT_LANGUAGE_OPTIONS } from '@/constants';

export function fmtBytes(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KiB`;
  return `${(b / (1024 * 1024)).toFixed(1)} MiB`;
}

export function fmtMs(ms: number | undefined) {
  if (!ms) return '0 ms';
  if (ms < 1000) return `${ms} ms`;
  const s = Math.round(ms / 100) / 10;
  if (s < 60) return `${s.toFixed(1)} s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

export function fmtProb(p: number) {
  return `${Math.round(p * 100)}%`;
}

export function langLabel(code: string) {
  return STT_LANGUAGE_OPTIONS.find((o) => o.value === code)?.label ?? code;
}
