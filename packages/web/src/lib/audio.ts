import { TTS_DELIVERY_PRESETS, TTS_TIMBRE_HINTS, TTS_TONE_PRESETS } from '@/constants';
import type { LocalizeResponse } from '@/types/api';

export function buildAudioDataUrl(contentType?: string, audioBase64?: string) {
  if (!contentType || !audioBase64) return null;
  return `data:${contentType};base64,${audioBase64}`;
}

export function downloadAudioDataUrl(filename: string, src: string) {
  const a = document.createElement('a');
  a.href = src;
  a.download = filename;
  a.click();
}

export function buildTtsDirection(options: {
  customPrompt: string;
  deliveryPresetId: (typeof TTS_DELIVERY_PRESETS)[number]['id'];
  timbreHintId: (typeof TTS_TIMBRE_HINTS)[number]['id'];
  tonePresetId: (typeof TTS_TONE_PRESETS)[number]['id'];
}) {
  const deliveryPrompt = TTS_DELIVERY_PRESETS.find((item) => item.id === options.deliveryPresetId)?.prompt ?? '';
  const tonePrompt = TTS_TONE_PRESETS.find((item) => item.id === options.tonePresetId)?.prompt ?? '';
  const timbrePrompt = TTS_TIMBRE_HINTS.find((item) => item.id === options.timbreHintId)?.prompt ?? '';
  const customPrompt = options.customPrompt.trim();

  return [deliveryPrompt, tonePrompt, timbrePrompt, customPrompt].filter(Boolean).join('; ');
}

export async function downloadSubtitle(fmt: string, loc: LocalizeResponse, workerBaseUrl: string) {
  try {
    const res = await fetch(`${workerBaseUrl}/api/subtitles/${fmt}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(loc),
    });
    if (!res.ok) return;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `subtitles.${fmt}`;
    a.click();
    URL.revokeObjectURL(url);
  } catch {
    // silent fail
  }
}
