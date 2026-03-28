import { useCallback, useEffect, useMemo, useState } from 'react';

import { TTS_DELIVERY_PRESETS, TTS_TIMBRE_HINTS, TTS_TONE_PRESETS } from '@/constants';
import { buildAudioDataUrl, buildTtsDirection } from '@/lib/audio';
import { getWorkerBaseUrl } from '@/lib/runtime-config';
import type { DemoMode, TtsActorCatalogResponse, TtsResponse, TtsVoiceMode } from '@/types/api';

import { useTtsCatalog } from './use-tts-catalog';

export function useTts() {
  const workerBaseUrl = getWorkerBaseUrl();
  const catalog = useTtsCatalog();

  const [busy, setBusy] = useState<DemoMode | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ttsRes, setTtsRes] = useState<TtsResponse | null>(null);

  const [ttsText, setTtsText] = useState('안녕하세요. Triton Playground 음성 합성 테스트입니다.');
  const [ttsPrompt, setTtsPrompt] = useState('');
  const [ttsLanguage, setTtsLanguage] = useState('ko');
  const [ttsVoiceMode, setTtsVoiceMode] = useState<TtsVoiceMode>('reference_clone');
  const [ttsDeliveryPresetId, setTtsDeliveryPresetId] =
    useState<(typeof TTS_DELIVERY_PRESETS)[number]['id']>('neutral');
  const [ttsTonePresetId, setTtsTonePresetId] = useState<(typeof TTS_TONE_PRESETS)[number]['id']>('natural');
  const [ttsTimbreHintId, setTtsTimbreHintId] = useState<(typeof TTS_TIMBRE_HINTS)[number]['id']>('auto');
  const [ttsActorId, setTtsActorId] = useState('');
  const [ttsReferenceFile, setTtsReferenceFile] = useState<File | null>(null);

  const ttsReferenceAudioUrl = useMemo(
    () => (ttsReferenceFile ? URL.createObjectURL(ttsReferenceFile) : null),
    [ttsReferenceFile]
  );
  useEffect(() => {
    return () => {
      if (ttsReferenceAudioUrl) URL.revokeObjectURL(ttsReferenceAudioUrl);
    };
  }, [ttsReferenceAudioUrl]);

  const { ttsCatalog, ttsCatalogError } = catalog;
  const ttsActors = useMemo(() => ttsCatalog?.actors ?? [], [ttsCatalog]);
  const selectedTtsActor = useMemo(
    () => ttsActors.find((actor) => actor.actor_id === ttsActorId) ?? null,
    [ttsActorId, ttsActors]
  );
  const presetVoiceAvailable = Boolean(ttsCatalog?.supports_preset_actors);
  const isReferenceVoiceMode = ttsVoiceMode === 'reference_clone';
  const isPresetVoiceMode = ttsVoiceMode === 'preset_voice';

  const effectiveTtsDirection = useMemo(
    () =>
      buildTtsDirection({
        customPrompt: ttsPrompt,
        deliveryPresetId: ttsDeliveryPresetId,
        timbreHintId: ttsTimbreHintId,
        tonePresetId: ttsTonePresetId,
      }),
    [ttsDeliveryPresetId, ttsPrompt, ttsTimbreHintId, ttsTonePresetId]
  );

  const ttsAudioSrc = buildAudioDataUrl(ttsRes?.content_type, ttsRes?.audio_base64);

  const canRunTts =
    Boolean(ttsText.trim()) &&
    Boolean(ttsLanguage) &&
    (isReferenceVoiceMode ? Boolean(ttsReferenceFile) : Boolean(presetVoiceAvailable && selectedTtsActor)) &&
    busy === null;

  // Sync voice mode on catalog load
  useEffect(() => {
    if (!ttsCatalog) return;
    setTtsVoiceMode((prev) => {
      if (prev === 'preset_voice' && !ttsCatalog.supports_preset_actors) return 'reference_clone';
      if (prev === 'reference_clone' && !ttsReferenceFile) return ttsCatalog.default_voice_mode;
      return prev;
    });
  }, [ttsCatalog, ttsReferenceFile]);

  // Switch to reference mode on file upload
  useEffect(() => {
    if (ttsReferenceFile) setTtsVoiceMode('reference_clone');
  }, [ttsReferenceFile]);

  // Reset actor when language changes
  useEffect(() => {
    if (ttsActors.length === 0) return;
    if (selectedTtsActor && selectedTtsActor.language === ttsLanguage) return;

    const defaultActorId =
      ttsCatalog?.default_actor_id_by_language[ttsLanguage] ??
      ttsActors.find((actor) => actor.language === ttsLanguage)?.actor_id ??
      ttsActors[0]?.actor_id ??
      '';
    setTtsActorId(defaultActorId);
  }, [selectedTtsActor, ttsActors, ttsCatalog, ttsLanguage]);

  const runTts = useCallback(async () => {
    if (!ttsText.trim()) {
      setError('Enter dialogue text for TTS.');
      return;
    }
    if (isReferenceVoiceMode) {
      if (!ttsReferenceFile) {
        setError('Upload reference audio to use Reference Voice mode.');
        return;
      }
    } else {
      if (!ttsCatalog?.supports_preset_actors) {
        setError(ttsCatalog?.preset_actor_message ?? 'Preset voice mode is unavailable in the current runtime.');
        return;
      }
      if (!selectedTtsActor) {
        setError('Select a preset voice from the library.');
        return;
      }
    }

    setBusy('tts');
    setError(null);
    setTtsRes(null);

    try {
      const fd = new FormData();
      fd.append('text', ttsText.trim());
      fd.append('language', ttsLanguage);
      fd.append('model', 'qwen3_tts_0_6b');
      if (effectiveTtsDirection) fd.append('prompt', effectiveTtsDirection);
      if (isPresetVoiceMode && selectedTtsActor) {
        fd.append('actor', selectedTtsActor.actor_id);
      }
      if (isReferenceVoiceMode && ttsReferenceFile) {
        fd.append('reference_audio', ttsReferenceFile);
      }

      const res = await fetch(`${workerBaseUrl}/api/tts`, {
        method: 'POST',
        body: fd,
      });
      const body = await res.json().catch(() => ({}) as Record<string, unknown>);
      if (!res.ok) {
        throw new Error(
          (body as { detail?: string; message?: string }).detail ??
            (body as { detail?: string; message?: string }).message ??
            `request failed: ${res.status}`
        );
      }

      setTtsRes(body as TtsResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'failed to generate TTS');
    } finally {
      setBusy(null);
    }
  }, [
    ttsText,
    isReferenceVoiceMode,
    ttsReferenceFile,
    ttsCatalog,
    selectedTtsActor,
    ttsLanguage,
    isPresetVoiceMode,
    effectiveTtsDirection,
    workerBaseUrl,
  ]);

  return {
    busy,
    setBusy,
    error,
    setError,
    ttsRes,
    ttsText,
    setTtsText,
    ttsPrompt,
    setTtsPrompt,
    ttsLanguage,
    setTtsLanguage,
    ttsVoiceMode,
    setTtsVoiceMode,
    ttsDeliveryPresetId,
    setTtsDeliveryPresetId,
    ttsTonePresetId,
    setTtsTonePresetId,
    ttsTimbreHintId,
    setTtsTimbreHintId,
    ttsActorId,
    setTtsActorId,
    ttsReferenceFile,
    setTtsReferenceFile,
    ttsReferenceAudioUrl,
    ttsCatalog: ttsCatalog as TtsActorCatalogResponse | null,
    ttsCatalogError,
    ttsActors,
    selectedTtsActor,
    presetVoiceAvailable,
    isReferenceVoiceMode,
    isPresetVoiceMode,
    effectiveTtsDirection,
    ttsAudioSrc,
    canRunTts,
    runTts,
    workerBaseUrl,
  };
}
