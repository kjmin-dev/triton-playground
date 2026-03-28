import { useEffect, useState } from 'react';

import { getWorkerBaseUrl } from '@/lib/runtime-config';
import type { ReadyResponse } from '@/types/api';

export function useWorkerReady() {
  const workerBaseUrl = getWorkerBaseUrl();
  const [ready, setReady] = useState<ReadyResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let off = false;
    (async () => {
      try {
        const res = await fetch(`${workerBaseUrl}/api/ready`);
        const body = (await res.json().catch(() => null)) as ReadyResponse | null;
        if (off) return;
        setReady(body);
        if (!res.ok) setError(body?.triton.summary ?? `worker check failed: ${res.status}`);
      } catch (err) {
        if (!off) setError(err instanceof Error ? err.message : 'failed to reach worker');
      }
    })();
    return () => {
      off = true;
    };
  }, [workerBaseUrl]);

  const live = Boolean(ready?.triton.server_ready && ready?.triton.model_ready);

  return { ready, live, error };
}
