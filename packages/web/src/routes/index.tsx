import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  component: Home,
});

function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-6 p-8">
      <h1 className="text-4xl font-bold tracking-tight">Triton Playground</h1>
      <p className="text-lg text-muted-foreground">
        GPU model serving demo — TTS, STT, audio separation
      </p>
    </main>
  );
}
