import { useCallback, useEffect, useMemo, useState } from 'react';

import { getWorkerBaseUrl } from '@/lib/runtime-config';
import type { DemoMode, LocalizeResponse, SttResponse, VadResponse } from '@/types/api';

export function usePipeline() {
  const workerBaseUrl = getWorkerBaseUrl();

  const [file, setFile] = useState<File | null>(null);
  const [threshold, setThreshold] = useState('0.50');
  const [srcLang, setSrcLang] = useState('auto');
  const [tgtLang, setTgtLang] = useState('en');
  const [busy, setBusy] = useState<DemoMode | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [vad, setVad] = useState<VadResponse | null>(null);
  const [sttRes, setStt] = useState<SttResponse | null>(null);
  const [loc, setLoc] = useState<LocalizeResponse | null>(null);

  const thr = Number.parseFloat(threshold);
  const thrOk = Number.isFinite(thr) && thr >= 0.1 && thr <= 0.99;

  const originalAudioUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  useEffect(() => {
    return () => {
      if (originalAudioUrl) URL.revokeObjectURL(originalAudioUrl);
    };
  }, [originalAudioUrl]);

  const runPipeline = useCallback(
    async (route: Exclude<DemoMode, 'tts'>) => {
      if (!file) {
        setError('Upload a WAV file first.');
        return route;
      }
      if (!thrOk) {
        setError('Threshold must be between 0.10 and 0.99.');
        return route;
      }

      setBusy(route);
      setError(null);
      if (route === 'vad') setVad(null);
      else if (route === 'stt') setStt(null);
      else setLoc(null);

      try {
        const fd = new FormData();
        fd.append('file', file);
        const q = new URLSearchParams({ threshold });

        if (route === 'stt' && srcLang !== 'auto') {
          q.set('language', srcLang);
        } else if (route === 'localize') {
          q.set('target_language', tgtLang);
          if (srcLang !== 'auto') q.set('source_language', srcLang);
        }

        const res = await fetch(`${workerBaseUrl}/api/${route}?${q}`, {
          method: 'POST',
          body: fd,
        });
        const body = await res.json().catch(() => ({}) as Record<string, unknown>);

        if (!res.ok) {
          if (route === 'localize') setLoc(body as LocalizeResponse);
          throw new Error(
            (body as { detail?: string; message?: string }).detail ??
              (body as { detail?: string; message?: string }).message ??
              `request failed: ${res.status}`
          );
        }

        if (route === 'vad') setVad(body as VadResponse);
        else if (route === 'stt') setStt(body as SttResponse);
        else setLoc(body as LocalizeResponse);
      } catch (err) {
        setError(err instanceof Error ? err.message : `failed to run ${route}`);
      } finally {
        setBusy(null);
      }
      return route;
    },
    [file, thrOk, threshold, srcLang, tgtLang, workerBaseUrl]
  );

  return {
    file,
    setFile,
    threshold,
    setThreshold,
    srcLang,
    setSrcLang,
    tgtLang,
    setTgtLang,
    busy,
    setBusy,
    error,
    setError,
    thr,
    thrOk,
    originalAudioUrl,
    vad,
    sttRes,
    loc,
    runPipeline,
    workerBaseUrl,
  };
}
