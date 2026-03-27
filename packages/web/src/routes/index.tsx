import * as React from "react";
import { createFileRoute } from "@tanstack/react-router";
import { BadgeInfo, FileAudio2, Languages, SlidersHorizontal, Waves } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { getWorkerBaseUrl } from "@/lib/runtime-config";

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
    status: string;
    summary: string;
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
  window_scores: number[];
};

type SttResponse = {
  duration_ms: number;
  filename: string;
  language: string;
  model: string;
  repository_model_name: string;
  sample_rate: number;
  segment_count: number;
  segments: Array<{
    average_probability: number;
    duration_ms: number;
    end_ms: number;
    peak_probability: number;
    start_ms: number;
    text: string;
  }>;
  task: string;
  threshold: number;
  transcript: string;
};

type LocalizeResponse = {
  duration_ms: number;
  filename: string;
  message?: string;
  models: {
    stt: string;
    translation: string;
    tts: string;
  };
  sample_rate: number;
  source_language: string;
  stage?: string;
  stages: {
    stt: {
      language?: string;
      message?: string;
      segment_count?: number;
      status: string;
      task?: string;
      transcript?: string;
    };
    translation: {
      message?: string;
      reason?: string;
      source_language?: string;
      status: string;
      target_language?: string;
      text?: string;
    };
    tts: {
      audio_base64?: string;
      content_type?: string;
      duration_ms?: number;
      language?: string;
      message?: string;
      reason?: string;
      sample_rate?: number;
      status: string;
    };
  };
  status: string;
  target_language: string;
  threshold: number;
  transcript: string;
  translated_text: string;
};

const THRESHOLD_PRESETS = [0.3, 0.45, 0.5, 0.65, 0.8];
const STT_LANGUAGE_OPTIONS = [
  { value: "auto", label: "Auto detect" },
  { value: "ko", label: "Korean" },
  { value: "en", label: "English" },
  { value: "ja", label: "Japanese" },
  { value: "zh", label: "Chinese" },
];
const TARGET_LANGUAGE_OPTIONS = STT_LANGUAGE_OPTIONS.filter((option) => option.value !== "auto");
const WINDOW_SCORE_PREVIEW_LIMIT = 80;
const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;
const MAX_DURATION_SECONDS = 15 * 60;

function formatBytes(bytes: number) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KiB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

function formatThreshold(value: number) {
  return value.toFixed(2);
}

function Home() {
  const workerBaseUrl = getWorkerBaseUrl();
  const [threshold, setThreshold] = React.useState("0.5");
  const [language, setLanguage] = React.useState("auto");
  const [targetLanguage, setTargetLanguage] = React.useState("en");
  const [selectedFile, setSelectedFile] = React.useState<File | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [activeRun, setActiveRun] = React.useState<"vad" | "stt" | "localize" | null>(null);
  const [ready, setReady] = React.useState<ReadyResponse | null>(null);
  const [catalog, setCatalog] = React.useState<ModelsResponse | null>(null);
  const [result, setResult] = React.useState<VadResponse | null>(null);
  const [sttResult, setSttResult] = React.useState<SttResponse | null>(null);
  const [localizeResult, setLocalizeResult] = React.useState<LocalizeResponse | null>(null);

  const parsedThreshold = Number.parseFloat(threshold);
  const thresholdIsValid =
    Number.isFinite(parsedThreshold) && parsedThreshold >= 0.1 && parsedThreshold <= 0.99;
  const isRunningVad = activeRun === "vad";
  const isRunningStt = activeRun === "stt";
  const isRunningLocalization = activeRun === "localize";
  const selectedFileSummary = selectedFile
    ? `${selectedFile.name} · ${selectedFile.type || "unknown type"} · ${formatBytes(selectedFile.size)}`
    : "No file selected yet";
  const windowScoresPreview = result?.window_scores.slice(0, WINDOW_SCORE_PREVIEW_LIMIT) ?? [];
  const speechWindowCount =
    result && thresholdIsValid
      ? result.window_scores.filter((score) => score >= parsedThreshold).length
      : 0;
  const localizationAudioPreview =
    localizeResult?.stages.tts.audio_base64 && localizeResult.stages.tts.content_type
      ? `data:${localizeResult.stages.tts.content_type};base64,${localizeResult.stages.tts.audio_base64}`
      : null;

  React.useEffect(() => {
    let cancelled = false;

    async function loadBootstrapData() {
      try {
        const [readyResponse, modelsResponse] = await Promise.all([
          fetch(`${workerBaseUrl}/api/ready`),
          fetch(`${workerBaseUrl}/api/models`),
        ]);

        if (!modelsResponse.ok) {
          throw new Error(`model catalog check failed: ${modelsResponse.status}`);
        }

        const [readyPayload, modelsPayload] = await Promise.all([
          readyResponse.json().catch(() => null) as Promise<ReadyResponse | null>,
          modelsResponse.json() as Promise<ModelsResponse>,
        ]);

        if (!cancelled) {
          setReady(readyPayload);
          setCatalog(modelsPayload);
          setError(
            readyResponse.ok
              ? null
              : readyPayload?.triton.summary ??
                  `worker ready check failed: ${readyResponse.status}`,
          );
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

  async function runWorkerRoute(route: "vad" | "stt" | "localize") {
    if (!selectedFile) {
      setError("Upload a PCM WAV file first.");
      return;
    }

    if (!thresholdIsValid) {
      setError("Threshold must stay between 0.10 and 0.99.");
      return;
    }

    setActiveRun(route);
    setError(null);
    if (route === "vad") {
      setResult(null);
    } else if (route === "stt") {
      setSttResult(null);
    } else {
      setLocalizeResult(null);
    }

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const query = new URLSearchParams({ threshold });
      let response: Response;
      if (route === "vad") {
        response = await fetch(`${workerBaseUrl}/api/vad?${query.toString()}`, {
          method: "POST",
          body: formData,
        });
      } else if (route === "stt") {
        if (language !== "auto") {
          query.set("language", language);
        }
        response = await fetch(`${workerBaseUrl}/api/stt?${query.toString()}`, {
          method: "POST",
          body: formData,
        });
      } else {
        query.set("target_language", targetLanguage);
        if (language !== "auto") {
          query.set("source_language", language);
        }
        response = await fetch(`${workerBaseUrl}/api/localize?${query.toString()}`, {
          method: "POST",
          body: formData,
        });
      }

      const payload = await response.json().catch(() => ({} as { detail?: string; message?: string }));
      if (!response.ok) {
        if (route === "localize") {
          setLocalizeResult(payload as LocalizeResponse);
        }
        throw new Error(
          payload.detail ??
            payload.message ??
            `request failed with ${response.status}`,
        );
      }

      if (route === "vad") {
        setResult(payload as VadResponse);
      } else if (route === "stt") {
        setSttResult(payload as SttResponse);
      } else {
        setLocalizeResult(payload as LocalizeResponse);
      }
    } catch (submitError) {
      setError(
        submitError instanceof Error
          ? submitError.message
          : route === "vad"
            ? "failed to run VAD"
            : route === "stt"
              ? "failed to run STT"
              : "failed to run localization preview",
      );
    } finally {
      setActiveRun(null);
    }
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await runWorkerRoute("vad");
  }

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,rgba(95,164,255,0.16),transparent_42%),linear-gradient(180deg,#f8fafc_0%,#eef4ff_100%)] px-6 py-10">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6">
        <section className="grid gap-4 lg:grid-cols-[1.15fr_0.85fr]">
          <Card className="border-slate-200/70 bg-white/90 shadow-sm">
            <CardHeader>
              <div className="flex items-center gap-3 text-xs uppercase tracking-[0.24em] text-slate-500">
                <Waves className="h-4 w-4" />
                Baseline VAD workflow
              </div>
              <CardTitle className="text-3xl tracking-tight">
                Triton Speech Baseline
              </CardTitle>
              <CardDescription className="max-w-2xl text-sm leading-6">
                Upload a WAV file, tune the speech threshold, and inspect segment
                boundaries from the approved Silero VAD path. The worker now rejects
                malformed, oversized, and overlong uploads with clearer errors.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form className="grid gap-4" onSubmit={handleSubmit}>
                <div className="grid gap-4 md:grid-cols-[1.2fr_0.8fr]">
                  <label className="grid gap-2 text-sm font-medium text-slate-900">
                    WAV upload
                    <input
                      accept=".wav,audio/wav"
                      className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
                      onChange={(event) => {
                        setSelectedFile(event.target.files?.[0] ?? null);
                        setError(null);
                      }}
                      type="file"
                    />
                    <span className="text-xs font-normal leading-5 text-slate-500">
                      {selectedFileSummary}. Demo guardrails: up to {formatBytes(MAX_UPLOAD_BYTES)}
                      and {MAX_DURATION_SECONDS / 60} minutes.
                    </span>
                  </label>

                  <div className="grid gap-2 text-sm font-medium text-slate-900">
                    <div className="flex items-center gap-2">
                      <SlidersHorizontal className="h-4 w-4 text-slate-500" />
                      Threshold
                    </div>
                    <input
                      className="w-full accent-slate-900"
                      max="0.99"
                      min="0.10"
                      onChange={(event) => setThreshold(event.target.value)}
                      step="0.01"
                      type="range"
                      value={threshold}
                    />
                    <div className="grid grid-cols-[1fr_auto] items-center gap-3">
                      <input
                        className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
                        max="0.99"
                        min="0.10"
                        onChange={(event) => setThreshold(event.target.value)}
                        step="0.01"
                        type="number"
                        value={threshold}
                      />
                      <div className="rounded-full bg-slate-100 px-3 py-2 text-sm font-medium text-slate-700">
                        {thresholdIsValid ? formatThreshold(parsedThreshold) : "invalid"}
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {THRESHOLD_PRESETS.map((preset) => (
                        <button
                          className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-700 transition hover:border-slate-300 hover:bg-slate-50"
                          key={preset}
                          type="button"
                          onClick={() => setThreshold(formatThreshold(preset))}
                        >
                          {formatThreshold(preset)}
                        </button>
                      ))}
                    </div>
                    <label className="grid gap-2 pt-2 text-sm font-medium text-slate-900">
                      <span className="flex items-center gap-2">
                        <Languages className="h-4 w-4 text-slate-500" />
                        Source language hint
                      </span>
                      <select
                        className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
                        onChange={(event) => setLanguage(event.target.value)}
                        value={language}
                      >
                        {STT_LANGUAGE_OPTIONS.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                      <span className="text-xs font-normal leading-5 text-slate-500">
                        Optional for the Whisper lane and localization pipeline. `auto` keeps
                        language detection in the manual backend.
                      </span>
                    </label>
                    <label className="grid gap-2 pt-2 text-sm font-medium text-slate-900">
                      <span className="flex items-center gap-2">
                        <Languages className="h-4 w-4 text-slate-500" />
                        Target localization language
                      </span>
                      <select
                        className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
                        onChange={(event) => setTargetLanguage(event.target.value)}
                        value={targetLanguage}
                      >
                        {TARGET_LANGUAGE_OPTIONS.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                      <span className="text-xs font-normal leading-5 text-slate-500">
                        The first orchestration pair translates with MADLAD and synthesizes a
                        preview with Qwen3-TTS.
                      </span>
                    </label>
                  </div>
                </div>

                <div className="flex items-center justify-between gap-3 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
                  <div className="flex items-start gap-2">
                    <BadgeInfo className="mt-0.5 h-4 w-4 shrink-0 text-slate-500" />
                    <p>
                      Audio is resampled to 16 kHz before VAD. Non-PCM WAV, corrupt
                      containers, and oversized files are rejected with explicit reasons.
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Button disabled={activeRun !== null} type="submit">
                      {isRunningVad ? "Running VAD..." : "Run VAD Happy Path"}
                    </Button>
                    <Button
                      disabled={activeRun !== null}
                      onClick={() => {
                        void runWorkerRoute("stt");
                      }}
                      type="button"
                      variant="outline"
                    >
                      {isRunningStt ? "Running STT..." : "Run Audio -> VAD -> STT"}
                    </Button>
                    <Button
                      disabled={activeRun !== null}
                      onClick={() => {
                        void runWorkerRoute("localize");
                      }}
                      type="button"
                      variant="secondary"
                    >
                      {isRunningLocalization ? "Running Localization..." : "Run Localization Preview"}
                    </Button>
                  </div>
                </div>

                {error ? (
                  <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                    {error}
                  </p>
                ) : null}
              </form>
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
              <div className="rounded-lg border border-white/10 bg-white/5 px-3 py-2">
                <div className="text-slate-400">Triton status</div>
                <div className="font-medium text-white">
                  {ready?.triton.status ?? "loading"}
                </div>
              </div>
              <div className="rounded-lg border border-white/10 bg-white/5 px-3 py-2">
                <div className="text-slate-400">Summary</div>
                <div className="font-medium text-white">
                  {ready?.triton.summary ?? "waiting for bootstrap response"}
                </div>
              </div>
            </CardContent>
          </Card>
        </section>

        <section className="grid gap-6 lg:grid-cols-[0.86fr_1.14fr]">
          <Card className="border-slate-200/70 bg-white/90 shadow-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileAudio2 className="h-5 w-5 text-slate-600" />
                Approved Catalog
              </CardTitle>
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

          <div className="grid gap-6">
            <Card className="border-slate-200/70 bg-white/90 shadow-sm">
              <CardHeader>
                <CardTitle>Speech Segments</CardTitle>
                <CardDescription>
                  Inspect the segment list and the per-window score trace returned by the worker.
                </CardDescription>
              </CardHeader>
              <CardContent className="grid gap-5">
                {result ? (
                  <>
                    <div className="grid gap-3 md:grid-cols-5">
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
                      <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                        <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                          speech windows
                        </div>
                        <div className="mt-1 text-sm font-medium text-slate-950">
                          {speechWindowCount} / {result.window_scores.length}
                        </div>
                      </div>
                    </div>

                    <div className="grid gap-3 rounded-xl border border-slate-200 bg-slate-50 px-4 py-4">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <div className="text-sm font-medium text-slate-950">Window score trace</div>
                          <div className="text-xs text-slate-500">
                            First {windowScoresPreview.length} of {result.window_scores.length} windows.
                          </div>
                        </div>
                        <div className="rounded-full bg-white px-3 py-1 text-xs font-medium text-slate-700">
                          threshold {formatThreshold(result.threshold)}
                        </div>
                      </div>
                      {windowScoresPreview.length > 0 ? (
                        <div
                          className="grid h-36 items-end gap-[2px]"
                          style={{
                            gridTemplateColumns: `repeat(${windowScoresPreview.length}, minmax(0, 1fr))`,
                          }}
                        >
                          {windowScoresPreview.map((score, index) => {
                            const active = score >= result.threshold;
                            const barHeight = `${Math.max(6, Math.min(100, score * 100))}%`;

                            return (
                              <div
                                className="flex h-full items-end"
                                key={`${index}-${score}`}
                                title={`window ${index + 1}: ${score.toFixed(4)}`}
                              >
                                <div
                                  className={`w-full rounded-t-sm ${active ? "bg-emerald-500" : "bg-slate-300"}`}
                                  style={{ height: barHeight }}
                                />
                              </div>
                            );
                          })}
                        </div>
                      ) : (
                        <p className="text-sm text-slate-500">No window scores were returned.</p>
                      )}
                    </div>

                    {result.segments.length > 0 ? (
                      <div className="grid gap-3">
                        {result.segments.map((segment, index) => (
                          <div
                            className="rounded-xl border border-slate-200 bg-white px-4 py-4 shadow-sm"
                            key={`${segment.start_ms}-${segment.end_ms}`}
                          >
                            <div className="flex items-center justify-between gap-3">
                              <div>
                                <div className="text-sm font-medium text-slate-950">
                                  Segment {index + 1}
                                </div>
                                <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                                  {segment.start_ms} ms to {segment.end_ms} ms
                                </div>
                              </div>
                              <div className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-700">
                                {segment.duration_ms} ms
                              </div>
                            </div>

                            <div className="mt-3 grid gap-3 md:grid-cols-2">
                              <div>
                                <div className="flex items-center justify-between text-xs text-slate-500">
                                  <span>Average probability</span>
                                  <span>{segment.average_probability}</span>
                                </div>
                                <div className="mt-1 h-2 rounded-full bg-slate-100">
                                  <div
                                    className="h-2 rounded-full bg-emerald-500"
                                    style={{ width: `${Math.min(100, segment.average_probability * 100)}%` }}
                                  />
                                </div>
                              </div>
                              <div>
                                <div className="flex items-center justify-between text-xs text-slate-500">
                                  <span>Peak probability</span>
                                  <span>{segment.peak_probability}</span>
                                </div>
                                <div className="mt-1 h-2 rounded-full bg-slate-100">
                                  <div
                                    className="h-2 rounded-full bg-sky-500"
                                    style={{ width: `${Math.min(100, segment.peak_probability * 100)}%` }}
                                  />
                                </div>
                              </div>
                            </div>
                          </div>
                        ))}
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

            <Card className="border-slate-200/70 bg-white/90 shadow-sm">
              <CardHeader>
                <CardTitle>Whisper STT</CardTitle>
                <CardDescription>
                  The worker reuses Silero VAD, then sends each detected speech segment to the
                  manual Whisper Triton backend.
                </CardDescription>
              </CardHeader>
              <CardContent className="grid gap-5">
                {sttResult ? (
                  <>
                    <div className="grid gap-3 md:grid-cols-5">
                      <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                        <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                          model
                        </div>
                        <div className="mt-1 text-sm font-medium text-slate-950">
                          {sttResult.repository_model_name}
                        </div>
                      </div>
                      <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                        <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                          task
                        </div>
                        <div className="mt-1 text-sm font-medium text-slate-950">
                          {sttResult.task}
                        </div>
                      </div>
                      <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                        <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                          language
                        </div>
                        <div className="mt-1 text-sm font-medium text-slate-950">
                          {sttResult.language}
                        </div>
                      </div>
                      <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                        <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                          segments
                        </div>
                        <div className="mt-1 text-sm font-medium text-slate-950">
                          {sttResult.segment_count}
                        </div>
                      </div>
                      <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                        <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                          threshold
                        </div>
                        <div className="mt-1 text-sm font-medium text-slate-950">
                          {formatThreshold(sttResult.threshold)}
                        </div>
                      </div>
                    </div>

                    <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-4">
                      <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                        aggregated transcript
                      </div>
                      <p className="mt-2 whitespace-pre-wrap text-sm leading-7 text-slate-800">
                        {sttResult.transcript || "No transcript text was returned for the detected speech segments."}
                      </p>
                    </div>

                    {sttResult.segments.length > 0 ? (
                      <div className="grid gap-3">
                        {sttResult.segments.map((segment, index) => (
                          <div
                            className="rounded-xl border border-slate-200 bg-white px-4 py-4 shadow-sm"
                            key={`${segment.start_ms}-${segment.end_ms}-stt`}
                          >
                            <div className="flex items-center justify-between gap-3">
                              <div>
                                <div className="text-sm font-medium text-slate-950">
                                  Transcript segment {index + 1}
                                </div>
                                <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                                  {segment.start_ms} ms to {segment.end_ms} ms
                                </div>
                              </div>
                              <div className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-700">
                                {segment.duration_ms} ms
                              </div>
                            </div>
                            <p className="mt-3 rounded-lg bg-slate-50 px-3 py-3 text-sm leading-6 text-slate-800">
                              {segment.text || "No text returned for this segment."}
                            </p>
                            <div className="mt-3 grid gap-3 md:grid-cols-2">
                              <div>
                                <div className="flex items-center justify-between text-xs text-slate-500">
                                  <span>Average probability</span>
                                  <span>{segment.average_probability}</span>
                                </div>
                                <div className="mt-1 h-2 rounded-full bg-slate-100">
                                  <div
                                    className="h-2 rounded-full bg-emerald-500"
                                    style={{ width: `${Math.min(100, segment.average_probability * 100)}%` }}
                                  />
                                </div>
                              </div>
                              <div>
                                <div className="flex items-center justify-between text-xs text-slate-500">
                                  <span>Peak probability</span>
                                  <span>{segment.peak_probability}</span>
                                </div>
                                <div className="mt-1 h-2 rounded-full bg-slate-100">
                                  <div
                                    className="h-2 rounded-full bg-sky-500"
                                    style={{ width: `${Math.min(100, segment.peak_probability * 100)}%` }}
                                  />
                                </div>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
                        No speech segments were detected, so Whisper was not invoked.
                      </p>
                    )}
                  </>
                ) : (
                  <div className="rounded-xl border border-dashed border-slate-300 px-4 py-10 text-center text-sm text-slate-500">
                    Run the STT lane to inspect the aggregated transcript and per-segment text.
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="border-slate-200/70 bg-white/90 shadow-sm">
              <CardHeader>
                <CardTitle>Localization Preview</CardTitle>
                <CardDescription>
                  End-to-end orchestration: Whisper transcript to MADLAD translation to Qwen3-TTS
                  preview audio.
                </CardDescription>
              </CardHeader>
              <CardContent className="grid gap-5">
                {localizeResult ? (
                  <>
                    <div className="grid gap-3 md:grid-cols-4">
                      <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                        <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                          source
                        </div>
                        <div className="mt-1 text-sm font-medium text-slate-950">
                          {localizeResult.source_language}
                        </div>
                      </div>
                      <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                        <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                          target
                        </div>
                        <div className="mt-1 text-sm font-medium text-slate-950">
                          {localizeResult.target_language}
                        </div>
                      </div>
                      <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                        <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                          pipeline
                        </div>
                        <div className="mt-1 text-sm font-medium text-slate-950">
                          {localizeResult.models.translation} + {localizeResult.models.tts}
                        </div>
                      </div>
                      <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                        <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                          status
                        </div>
                        <div className="mt-1 text-sm font-medium text-slate-950">
                          {localizeResult.status}
                        </div>
                      </div>
                    </div>

                    <div className="grid gap-3 md:grid-cols-3">
                      <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-4">
                        <div className="flex items-center justify-between gap-3">
                          <div className="text-sm font-medium text-slate-950">STT stage</div>
                          <div className="rounded-full bg-white px-3 py-1 text-xs font-medium text-slate-700">
                            {localizeResult.stages.stt.status}
                          </div>
                        </div>
                        <div className="mt-2 text-xs uppercase tracking-[0.2em] text-slate-500">
                          {localizeResult.models.stt}
                        </div>
                        <p className="mt-3 text-sm leading-6 text-slate-700">
                          {localizeResult.stages.stt.message ??
                            localizeResult.stages.stt.transcript ??
                            "Waiting for transcript."}
                        </p>
                      </div>
                      <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-4">
                        <div className="flex items-center justify-between gap-3">
                          <div className="text-sm font-medium text-slate-950">Translation stage</div>
                          <div className="rounded-full bg-white px-3 py-1 text-xs font-medium text-slate-700">
                            {localizeResult.stages.translation.status}
                          </div>
                        </div>
                        <div className="mt-2 text-xs uppercase tracking-[0.2em] text-slate-500">
                          {localizeResult.models.translation}
                        </div>
                        <p className="mt-3 text-sm leading-6 text-slate-700">
                          {localizeResult.stages.translation.message ??
                            localizeResult.stages.translation.reason ??
                            localizeResult.stages.translation.text ??
                            "Waiting for translated text."}
                        </p>
                      </div>
                      <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-4">
                        <div className="flex items-center justify-between gap-3">
                          <div className="text-sm font-medium text-slate-950">TTS stage</div>
                          <div className="rounded-full bg-white px-3 py-1 text-xs font-medium text-slate-700">
                            {localizeResult.stages.tts.status}
                          </div>
                        </div>
                        <div className="mt-2 text-xs uppercase tracking-[0.2em] text-slate-500">
                          {localizeResult.models.tts}
                        </div>
                        <p className="mt-3 text-sm leading-6 text-slate-700">
                          {localizeResult.stages.tts.message ??
                            localizeResult.stages.tts.reason ??
                            (localizationAudioPreview
                              ? `${localizeResult.stages.tts.duration_ms} ms preview ready`
                              : "Waiting for synthesized preview.")}
                        </p>
                      </div>
                    </div>

                    {localizeResult.message ? (
                      <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                        {localizeResult.message}
                      </p>
                    ) : null}

                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-4">
                        <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                          transcript
                        </div>
                        <p className="mt-2 whitespace-pre-wrap text-sm leading-7 text-slate-800">
                          {localizeResult.transcript || "No transcript text is available."}
                        </p>
                      </div>
                      <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-4">
                        <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
                          translated text
                        </div>
                        <p className="mt-2 whitespace-pre-wrap text-sm leading-7 text-slate-800">
                          {localizeResult.translated_text || "No translated text is available."}
                        </p>
                      </div>
                    </div>

                    <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-4">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <div className="text-sm font-medium text-slate-950">Synthesized preview</div>
                          <div className="text-xs text-slate-500">
                            The worker wraps the TTS waveform as a WAV asset for browser playback.
                          </div>
                        </div>
                        {localizeResult.stages.tts.sample_rate ? (
                          <div className="rounded-full bg-white px-3 py-1 text-xs font-medium text-slate-700">
                            {localizeResult.stages.tts.sample_rate} Hz
                          </div>
                        ) : null}
                      </div>
                      {localizationAudioPreview ? (
                        <audio className="mt-4 w-full" controls src={localizationAudioPreview}>
                          <track kind="captions" />
                        </audio>
                      ) : (
                        <p className="mt-4 text-sm text-slate-500">
                          No preview audio is available for the current run.
                        </p>
                      )}
                    </div>
                  </>
                ) : (
                  <div className="rounded-xl border border-dashed border-slate-300 px-4 py-10 text-center text-sm text-slate-500">
                    Run the localization preview to inspect stage-by-stage transcript,
                    translation, and synthesized audio output.
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </section>
      </div>
    </main>
  );
}
