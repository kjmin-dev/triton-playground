export type PublicRuntimeConfig = {
  workerApiUrl: string | null;
  workerPort: string;
  webPort: string;
};

declare global {
  interface Window {
    __TRITON_PLAYGROUND_CONFIG__?: PublicRuntimeConfig;
  }

  interface ImportMetaEnv {
    readonly WEB_PORT?: string;
    readonly WORKER_API_URL?: string;
    readonly WORKER_PORT?: string;
    readonly VITE_WORKER_API_URL?: string;
  }
}

const DEFAULT_WORKER_PORT = '8080';

export function getWorkerBaseUrl() {
  if (typeof window === 'undefined') {
    return (
      import.meta.env.WORKER_API_URL ?? import.meta.env.VITE_WORKER_API_URL ?? `http://localhost:${DEFAULT_WORKER_PORT}`
    );
  }

  const runtimeConfig = window.__TRITON_PLAYGROUND_CONFIG__;
  if (runtimeConfig?.workerApiUrl) {
    return runtimeConfig.workerApiUrl;
  }

  const configuredWorkerApiUrl = import.meta.env.WORKER_API_URL ?? import.meta.env.VITE_WORKER_API_URL;
  if (configuredWorkerApiUrl) {
    return configuredWorkerApiUrl;
  }

  const workerPort = runtimeConfig?.workerPort ?? import.meta.env.WORKER_PORT ?? DEFAULT_WORKER_PORT;

  return `${window.location.protocol}//${window.location.hostname}:${workerPort}`;
}
