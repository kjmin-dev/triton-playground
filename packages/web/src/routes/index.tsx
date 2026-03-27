import * as React from "react";
import { createFileRoute } from "@tanstack/react-router";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export const Route = createFileRoute("/")({
  component: Home,
});

type ReadyResponse = {
  status: string;
  profile: string;
  triton: {
    model_name: string;
    model_ready: boolean;
    server_live: boolean;
    server_ready: boolean;
  };
};

type ModelsResponse = {
  baseline_model_ids: string[];
  models: Array<{
    approved_for_auto_download: boolean;
    license_name: string;
    model_id: string;
    notes: string;
    serve_status: string;
    stage: string;
  }>;
  profile: string;
};

type VadResponse = {
  duration_ms: number;
  filename: string;
  model: string;
  sample_rate: number;
  segment_count: number;
  segments: Array<{
    average_probability: number;
    duration_ms: number;
    end_ms: number;
    peak_probability: number;
    start_ms: number;
  }>;
  threshold: number;
  window_ms: number;
};

function getWorkerBaseUrl() {
  if (typeof window === "undefined") {
    return import.meta.env.VITE_WORKER_API_URL ?? "http://localhost:8080";
  }

  return (
    import.meta.env.VITE_WORKER_API_URL ??
    `${window.location.protocol}//${window.location.hostname}:8080`
  );
}

function Home() {
  const workerBaseUrl = getWorkerBaseUrl();
  const [threshold, setThreshold] = React.useState("0.5");
  const [selectedFile, setSelectedFile] = React.useState<File | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [ready, setReady] = React.useState<ReadyResponse | null>(null);
  const [catalog, setCatalog] = React.useState<ModelsResponse | null>(null);
  const [result, setResult] = React.useState<VadResponse | null>(null);

  React.useEffect(() => {
    let cancelled = false;

    async function loadBootstrapData() {
      try {
        const [readyResponse, modelsResponse] = await Promise.all([
          fetch(`${workerBaseUrl}/api/ready`),
          fetch(`${workerBaseUrl}/api/models`),
        ]);

        if (!readyResponse.ok) {
          throw new Error(`worker ready check failed: ${readyResponse.status}`);
        }

        if (!modelsResponse.ok) {
          throw new Error(`model catalog check failed: ${modelsResponse.status}`);
        }

        const [readyPayload, modelsPayload] = await Promise.all([
          readyResponse.json() as Promise<ReadyResponse>,
          modelsResponse.json() as Promise<ModelsResponse>,
        ]);

        if (!cancelled) {
          setReady(readyPayload);
          setCatalog(modelsPayload);
          setError(null);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(
            loadError instanceof Error
              ? loadError.message
              : "failed to load worker bootstrap state",
          );
        }
      }
    }

    void loadBootstrapData();

    return () => {
      cancelled = true;
    };
  }, [workerBaseUrl]);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!selectedFile) {
      setError("Upload a PCM WAV file first.");
      return;
    }

    setIsSubmitting(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(
        `${workerBaseUrl}/api/vad?threshold=${encodeURIComponent(threshold)}`,
        {
          method: "POST",
          body: formData,
        },
      );

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail ?? `request failed with ${response.status}`);
      }

      setResult(payload as VadResponse);
    } catch (submitError) {
      setError(
        submitError instanceof Error ? submitError.message : "failed to run VAD",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,rgba(95,164,255,0.14),transparent_42%),linear-gradient(180deg,#f8fafc_0%,#eef4ff_100%)] px-6 py-10">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6">
        <section className="grid gap-4 lg:grid-cols-[1.3fr_0.7fr]">
          <Card className="border-slate-200/70 bg-white/90 shadow-sm">
            <CardHeader>
              <CardTitle className="text-3xl tracking-tight">
                Triton Speech Baseline
              </CardTitle>
              <CardDescription className="max-w-2xl text-sm leading-6">
                Compliance-first startup path for open models. The current happy
                path downloads an approved Silero VAD ONNX artifact, boots Triton,
                and returns speech segments for uploaded WAV audio.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form className="grid gap-4 md:grid-cols-[1fr_auto]" onSubmit={handleSubmit}>
                <div className="grid gap-3">
                  <label className="grid gap-2 text-sm font-medium text-slate-900">
                    WAV upload
                    <input
                      accept=".wav,audio/wav"
                      className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
                      onChange={(event) => {
                        setSelectedFile(event.target.files?.[0] ?? null);
                      }}
                      type="file"
                    />
                  </label>
                  <label className="grid gap-2 text-sm font-medium text-slate-900">
                    Threshold
                    <input
                      className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
                      max="0.99"
                      min="0.10"
                      onChange={(event) => setThreshold(event.target.value)}
                      step="0.01"
                      type="number"
                      value={threshold}
                    />
                  </label>
                </div>
                <div className="flex items-end">
                  <Button className="w-full md:w-auto" disabled={isSubmitting} type="submit">
                    {isSubmitting ? "Running VAD..." : "Run Happy Path"}
                  </Button>
                </div>
              </form>

              {error ? (
                <p className="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                  {error}
                </p>
              ) : null}
            </CardContent>
          </Card>

          <Card className="border-slate-200/70 bg-slate-950 text-white shadow-sm">
            <CardHeader>
              <CardTitle className="text-lg">Bootstrap State</CardTitle>
              <CardDescription className="text-slate-300">
                Worker and Triton readiness for the baseline profile.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3 text-sm">
              <div className="rounded-lg border border-white/10 bg-white/5 px-3 py-2">
                <div className="text-slate-400">Worker profile</div>
                <div className="font-medium text-white">
                  {ready?.profile ?? "loading"}
                </div>
              </div>
              <div className="rounded-lg border border-white/10 bg-white/5 px-3 py-2">
                <div className="text-slate-400">Triton model</div>
                <div className="font-medium text-white">
                  {ready?.triton.model_name ?? "silero_vad"}
                </div>
              </div>
              <div className="rounded-lg border border-white/10 bg-white/5 px-3 py-2">
                <div className="text-slate-400">Ready</div>
                <div className="font-medium text-white">
                  {ready?.triton.model_ready ? "yes" : "waiting"}
                </div>
              </div>
            </CardContent>
          </Card>
        </section>

        <section className="grid gap-6 lg:grid-cols-[0.8fr_1.2fr]">
          <Card className="border-slate-200/70 bg-white/90 shadow-sm">
            <CardHeader>
              <CardTitle>Approved Catalog</CardTitle>
              <CardDescription>
                Only explicit allowlist entries can enter the runtime download lane.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3">
              {catalog?.models.map((model) => (
                <div
                  className="rounded-xl border border-slate-200 px-3 py-3"
                  key={model.model_id}
                >
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <div className="font-medium text-slate-950">{model.model_id}</div>
                      <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                        {model.stage}
                      </div>
                    </div>
                    <div className="rounded-full bg-slate-100 px-2 py-1 text-xs font-medium text-slate-700">
                      {model.license_name}
                    </div>
                  </div>
                  <p className="mt-2 text-sm text-slate-600">{model.notes}</p>
                  <div className="mt-2 text-xs text-slate-500">
                    auto-download: {model.approved_for_auto_download ? "yes" : "no"} / serve:{" "}
                    {model.serve_status}
                  </div>
                </div>
              )) ?? <p className="text-sm text-slate-500">Loading catalog...</p>}
            </CardContent>
          </Card>

          <Card className="border-slate-200/70 bg-white/90 shadow-sm">
            <CardHeader>
              <CardTitle>Speech Segments</CardTitle>
              <CardDescription>
                Upload a PCM WAV file to get Silero VAD scores over Triton gRPC.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              {result ? (
                <>
                  <div className="grid gap-3 md:grid-cols-4">
                    <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                      <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                        file
                      </div>
                      <div className="mt-1 text-sm font-medium text-slate-950">
                        {result.filename}
                      </div>
                    </div>
                    <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                      <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                        duration
                      </div>
                      <div className="mt-1 text-sm font-medium text-slate-950">
                        {result.duration_ms} ms
                      </div>
                    </div>
                    <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                      <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                        sample rate
                      </div>
                      <div className="mt-1 text-sm font-medium text-slate-950">
                        {result.sample_rate} Hz
                      </div>
                    </div>
                    <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                      <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                        segments
                      </div>
                      <div className="mt-1 text-sm font-medium text-slate-950">
                        {result.segment_count}
                      </div>
                    </div>
                  </div>

                  {result.segments.length > 0 ? (
                    <div className="overflow-hidden rounded-xl border border-slate-200">
                      <table className="min-w-full divide-y divide-slate-200 text-sm">
                        <thead className="bg-slate-50 text-left text-slate-500">
                          <tr>
                            <th className="px-4 py-3 font-medium">start</th>
                            <th className="px-4 py-3 font-medium">end</th>
                            <th className="px-4 py-3 font-medium">duration</th>
                            <th className="px-4 py-3 font-medium">avg prob</th>
                            <th className="px-4 py-3 font-medium">peak prob</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100 bg-white">
                          {result.segments.map((segment) => (
                            <tr key={`${segment.start_ms}-${segment.end_ms}`}>
                              <td className="px-4 py-3">{segment.start_ms} ms</td>
                              <td className="px-4 py-3">{segment.end_ms} ms</td>
                              <td className="px-4 py-3">{segment.duration_ms} ms</td>
                              <td className="px-4 py-3">{segment.average_probability}</td>
                              <td className="px-4 py-3">{segment.peak_probability}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
                      No speech segments crossed the current threshold.
                    </p>
                  )}
                </>
              ) : (
                <div className="rounded-xl border border-dashed border-slate-300 px-4 py-10 text-center text-sm text-slate-500">
                  The result panel fills once the VAD happy path returns segment data.
                </div>
              )}
            </CardContent>
          </Card>
        </section>
      </div>
    </main>
  );
}
