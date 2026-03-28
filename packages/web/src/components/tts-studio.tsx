import { Bot, Globe2, Upload } from 'lucide-react';

import { TTS_DELIVERY_PRESETS, TTS_LANGUAGE_OPTIONS, TTS_TIMBRE_HINTS, TTS_TONE_PRESETS } from '@/constants';
import { fmtBytes, langLabel } from '@/lib/format';
import type { TtsActor, TtsActorCatalogResponse, TtsVoiceMode } from '@/types/api';

import { Button } from './ui/button';
import { VoiceLibrary } from './voice-library';

export function TtsStudio({
  disabled,
  ttsLanguage,
  onTtsLanguageChange,
  ttsVoiceMode,
  onTtsVoiceModeChange,
  ttsCatalog,
  presetVoiceAvailable,
  isReferenceVoiceMode,
  isPresetVoiceMode,
  ttsReferenceFile,
  onTtsReferenceFileChange,
  ttsReferenceAudioUrl,
  ttsText,
  onTtsTextChange,
  ttsDeliveryPresetId,
  onTtsDeliveryPresetIdChange,
  ttsTonePresetId,
  onTtsTonePresetIdChange,
  ttsTimbreHintId,
  onTtsTimbreHintIdChange,
  ttsPrompt,
  onTtsPromptChange,
  effectiveTtsDirection,
  selectedTtsActor,
  ttsActorId,
  onTtsActorIdChange,
  ttsActors,
  ttsCatalogError,
}: {
  disabled: boolean;
  ttsLanguage: string;
  onTtsLanguageChange: (value: string) => void;
  ttsVoiceMode: TtsVoiceMode;
  onTtsVoiceModeChange: (mode: TtsVoiceMode) => void;
  ttsCatalog: TtsActorCatalogResponse | null;
  presetVoiceAvailable: boolean;
  isReferenceVoiceMode: boolean;
  isPresetVoiceMode: boolean;
  ttsReferenceFile: File | null;
  onTtsReferenceFileChange: (file: File | null) => void;
  ttsReferenceAudioUrl: string | null;
  ttsText: string;
  onTtsTextChange: (value: string) => void;
  ttsDeliveryPresetId: (typeof TTS_DELIVERY_PRESETS)[number]['id'];
  onTtsDeliveryPresetIdChange: (id: (typeof TTS_DELIVERY_PRESETS)[number]['id']) => void;
  ttsTonePresetId: (typeof TTS_TONE_PRESETS)[number]['id'];
  onTtsTonePresetIdChange: (id: (typeof TTS_TONE_PRESETS)[number]['id']) => void;
  ttsTimbreHintId: (typeof TTS_TIMBRE_HINTS)[number]['id'];
  onTtsTimbreHintIdChange: (id: (typeof TTS_TIMBRE_HINTS)[number]['id']) => void;
  ttsPrompt: string;
  onTtsPromptChange: (value: string) => void;
  effectiveTtsDirection: string;
  selectedTtsActor: TtsActor | null;
  ttsActorId: string;
  onTtsActorIdChange: (id: string) => void;
  ttsActors: TtsActor[];
  ttsCatalogError: string | null;
}) {
  return (
    <div className='space-y-5 rounded-xl border border-slate-200 bg-white/80 p-5 backdrop-blur'>
      {/* Language & Voice Source */}
      <div className='grid gap-4 sm:grid-cols-2'>
        <label className='space-y-2'>
          <span className='flex items-center gap-2 text-sm font-medium text-slate-700'>
            <Globe2 className='h-3.5 w-3.5 text-slate-400' />
            Language
          </span>
          <select
            className='w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm'
            disabled={disabled}
            onChange={(e) => onTtsLanguageChange(e.target.value)}
            value={ttsLanguage}
          >
            {TTS_LANGUAGE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>

        <div className='space-y-2'>
          <span className='flex items-center gap-2 text-sm font-medium text-slate-700'>
            <Bot className='h-3.5 w-3.5 text-slate-400' />
            Voice Source
          </span>
          <div className='grid gap-2'>
            {(
              ttsCatalog?.voice_modes ?? [
                {
                  available: true,
                  description: 'Upload a WAV reference clip and clone its voice with the Base checkpoint.',
                  label: 'Reference Voice',
                  mode: 'reference_clone' as TtsVoiceMode,
                },
                {
                  available: presetVoiceAvailable,
                  description: presetVoiceAvailable
                    ? 'Use built-in preset voices from the optional CustomVoice checkpoint.'
                    : 'Preset voices require the optional CustomVoice checkpoint.',
                  label: 'Preset Voice Library',
                  mode: 'preset_voice' as TtsVoiceMode,
                },
              ]
            ).map((voiceMode) => {
              const active = ttsVoiceMode === voiceMode.mode;
              return (
                <button
                  className={`rounded-xl border px-3 py-3 text-left transition ${
                    active
                      ? 'border-cyan-400 bg-cyan-50'
                      : voiceMode.available
                        ? 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'
                        : 'cursor-not-allowed border-slate-200 bg-slate-50 text-slate-400'
                  }`}
                  disabled={!voiceMode.available || disabled}
                  key={voiceMode.mode}
                  onClick={() => onTtsVoiceModeChange(voiceMode.mode)}
                  type='button'
                >
                  <div className='flex items-center justify-between gap-2'>
                    <span className='text-sm font-semibold text-slate-950'>{voiceMode.label}</span>
                    <span
                      className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.12em] ${
                        voiceMode.available ? 'bg-emerald-50 text-emerald-700' : 'bg-slate-100 text-slate-500'
                      }`}
                    >
                      {voiceMode.available ? 'Ready' : 'Unavailable'}
                    </span>
                  </div>
                  <p className='mt-1 text-xs leading-5 text-slate-500'>{voiceMode.description}</p>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Reference Voice Upload */}
      {isReferenceVoiceMode && (
        <label className='space-y-2'>
          <span className='flex items-center gap-2 text-sm font-medium text-slate-700'>
            <Upload className='h-3.5 w-3.5 text-slate-400' />
            Reference Voice
          </span>
          <div className='rounded-xl border border-dashed border-slate-300 bg-slate-50 px-3 py-3'>
            <div className='flex items-center justify-between gap-3'>
              <div className='min-w-0'>
                <p className='truncate text-sm font-medium text-slate-900'>
                  {ttsReferenceFile ? ttsReferenceFile.name : 'Required WAV reference audio'}
                </p>
                <p className='text-xs text-slate-500'>
                  {ttsReferenceFile
                    ? fmtBytes(ttsReferenceFile.size)
                    : 'Upload a clean sample to clone the source voice'}
                </p>
              </div>
              <div className='flex items-center gap-2'>
                {ttsReferenceFile && (
                  <Button onClick={() => onTtsReferenceFileChange(null)} size='sm' type='button' variant='ghost'>
                    Clear
                  </Button>
                )}
                <label className='cursor-pointer'>
                  <span className='inline-flex h-7 items-center justify-center rounded-[min(var(--radius-md),12px)] border border-border bg-background px-2.5 text-[0.8rem] font-medium text-foreground transition hover:bg-muted hover:text-foreground'>
                    {ttsReferenceFile ? 'Replace' : 'Upload'}
                  </span>
                  <input
                    accept='.wav,audio/wav'
                    className='sr-only'
                    disabled={disabled}
                    onChange={(e) => onTtsReferenceFileChange(e.target.files?.[0] ?? null)}
                    type='file'
                  />
                </label>
              </div>
            </div>
            {ttsReferenceAudioUrl && <audio className='mt-3 w-full' controls src={ttsReferenceAudioUrl} />}
          </div>
        </label>
      )}

      {/* Dialogue */}
      <label className='space-y-2'>
        <span className='text-sm font-medium text-slate-700'>Dialogue</span>
        <textarea
          className='min-h-36 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm leading-6 text-slate-900 outline-none transition focus:border-cyan-300 focus:ring-4 focus:ring-cyan-100'
          disabled={disabled}
          onChange={(e) => onTtsTextChange(e.target.value)}
          placeholder='Enter the line you want Qwen3-TTS to perform.'
          value={ttsText}
        />
      </label>

      {/* Direction Presets */}
      <div className='grid gap-4 md:grid-cols-3'>
        <label className='space-y-2'>
          <span className='text-sm font-medium text-slate-700'>Delivery</span>
          <select
            className='w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm'
            disabled={disabled}
            onChange={(e) => onTtsDeliveryPresetIdChange(e.target.value as (typeof TTS_DELIVERY_PRESETS)[number]['id'])}
            value={ttsDeliveryPresetId}
          >
            {TTS_DELIVERY_PRESETS.map((option) => (
              <option key={option.id} value={option.id}>
                {option.label}
              </option>
            ))}
          </select>
          <p className='text-xs text-slate-500'>
            {TTS_DELIVERY_PRESETS.find((item) => item.id === ttsDeliveryPresetId)?.description}
          </p>
        </label>

        <label className='space-y-2'>
          <span className='text-sm font-medium text-slate-700'>Tone</span>
          <select
            className='w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm'
            disabled={disabled}
            onChange={(e) => onTtsTonePresetIdChange(e.target.value as (typeof TTS_TONE_PRESETS)[number]['id'])}
            value={ttsTonePresetId}
          >
            {TTS_TONE_PRESETS.map((option) => (
              <option key={option.id} value={option.id}>
                {option.label}
              </option>
            ))}
          </select>
          <p className='text-xs text-slate-500'>
            {TTS_TONE_PRESETS.find((item) => item.id === ttsTonePresetId)?.prompt}
          </p>
        </label>

        <label className='space-y-2'>
          <span className='text-sm font-medium text-slate-700'>Voice Hint</span>
          <select
            className='w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm'
            disabled={disabled}
            onChange={(e) => onTtsTimbreHintIdChange(e.target.value as (typeof TTS_TIMBRE_HINTS)[number]['id'])}
            value={ttsTimbreHintId}
          >
            {TTS_TIMBRE_HINTS.map((option) => (
              <option key={option.id} value={option.id}>
                {option.label}
              </option>
            ))}
          </select>
          <p className='text-xs text-slate-500'>
            {TTS_TIMBRE_HINTS.find((item) => item.id === ttsTimbreHintId)?.prompt || 'No timbre hint'}
          </p>
        </label>
      </div>

      {/* Custom Direction */}
      <label className='space-y-2'>
        <span className='text-sm font-medium text-slate-700'>Custom Direction</span>
        <textarea
          className='min-h-24 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm leading-6 text-slate-900 outline-none transition focus:border-cyan-300 focus:ring-4 focus:ring-cyan-100'
          disabled={disabled}
          onChange={(e) => onTtsPromptChange(e.target.value)}
          placeholder='Optional extra direction. Example: slightly playful, late-night radio calm, more restrained ending'
          value={ttsPrompt}
        />
        <div className='rounded-xl border border-slate-200 bg-slate-50 px-3 py-3'>
          <p className='text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500'>Effective Direction</p>
          <p className='mt-2 text-sm leading-6 text-slate-700'>
            {effectiveTtsDirection || 'No additional direction. The model will use plain generation defaults.'}
          </p>
        </div>
      </label>

      {/* Voice Summary */}
      <div className='flex flex-wrap items-start gap-3 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm leading-6 text-slate-600'>
        <span className='rounded-full bg-white px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-500'>
          {isReferenceVoiceMode ? 'Reference Clone' : 'Preset Voice'}
        </span>
        <p className='min-w-[220px] flex-1'>
          {isReferenceVoiceMode
            ? ttsReferenceFile
              ? `Using ${ttsReferenceFile.name} as the clone reference.`
              : 'Upload a WAV clip to enable reference clone generation.'
            : selectedTtsActor
              ? `Using ${selectedTtsActor.label} (${langLabel(selectedTtsActor.language)}).`
              : 'Select a preset voice from the library.'}
        </p>
        {ttsCatalog?.preset_actor_message && !presetVoiceAvailable && (
          <p className='basis-full text-amber-900'>{ttsCatalog.preset_actor_message}</p>
        )}
      </div>

      {/* Voice Library or Reference Info */}
      {isPresetVoiceMode ? (
        <VoiceLibrary
          actors={ttsActors}
          selectedActorId={ttsActorId}
          language={ttsLanguage}
          onSelectActor={onTtsActorIdChange}
          onLanguageChange={onTtsLanguageChange}
          catalogError={ttsCatalogError}
        />
      ) : (
        <div className='rounded-xl border border-slate-200 bg-slate-50 px-4 py-4 text-sm leading-6 text-slate-600'>
          Reference Voice mode ignores preset speakers completely. If you want to use built-in voices instead of cloning
          an uploaded sample, switch to Preset Voice and choose one from the library.
        </div>
      )}
    </div>
  );
}
