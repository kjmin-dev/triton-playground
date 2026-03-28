import { useEffect, useState } from 'react';

import { getWorkerBaseUrl } from '@/lib/runtime-config';
import type { TtsActorCatalogResponse } from '@/types/api';

export function useTtsCatalog() {
  const workerBaseUrl = getWorkerBaseUrl();
  const [ttsCatalog, setTtsCatalog] = useState<TtsActorCatalogResponse | null>(null);
  const [ttsCatalogError, setTtsCatalogError] = useState<string | null>(null);

  useEffect(() => {
    let off = false;
    (async () => {
      try {
        const res = await fetch(`${workerBaseUrl}/api/tts/actors`);
        const body = (await res.json().catch(() => null)) as TtsActorCatalogResponse | null;
        if (off) return;
        if (!res.ok || !body) {
          setTtsCatalogError(`failed to load TTS actors: ${res.status}`);
          return;
        }
        setTtsCatalog(body);
        setTtsCatalogError(null);
      } catch (err) {
        if (!off) setTtsCatalogError(err instanceof Error ? err.message : 'failed to load TTS actors');
      }
    })();
    return () => {
      off = true;
    };
  }, [workerBaseUrl]);

  return { ttsCatalog, ttsCatalogError };
}
