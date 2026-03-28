import type { LucideIcon } from 'lucide-react';
import { Headphones, MicVocal, Sparkles, Waves } from 'lucide-react';

import type { DemoMode } from '@/types/api';

export const THRESHOLD_PRESETS = [0.3, 0.5, 0.65, 0.8];

export const STT_LANGUAGE_OPTIONS = [
  { value: 'auto', label: 'Auto' },
  { value: 'ko', label: 'Korean' },
  { value: 'en', label: 'English' },
  { value: 'ja', label: 'Japanese' },
  { value: 'zh', label: 'Chinese' },
];

export const TARGET_LANGUAGE_OPTIONS = STT_LANGUAGE_OPTIONS.filter((o) => o.value !== 'auto');
export const TTS_LANGUAGE_OPTIONS = TARGET_LANGUAGE_OPTIONS;

export const TTS_DELIVERY_PRESETS = [
  { id: 'neutral', label: 'Neutral', description: 'Balanced and direct delivery', prompt: 'clear natural delivery' },
  { id: 'warm', label: 'Warm', description: 'Gentle and reassuring pacing', prompt: 'warm reassuring delivery' },
  {
    id: 'energetic',
    label: 'Energetic',
    description: 'Bright and forward momentum',
    prompt: 'bright energetic performance',
  },
  {
    id: 'dramatic',
    label: 'Dramatic',
    description: 'High contrast and expressive emphasis',
    prompt: 'dramatic expressive delivery',
  },
] as const;

export const TTS_TONE_PRESETS = [
  { id: 'natural', label: 'Natural', prompt: 'natural studio tone' },
  { id: 'soft', label: 'Soft', prompt: 'soft intimate tone' },
  { id: 'bright', label: 'Bright', prompt: 'bright crisp tone' },
  { id: 'calm', label: 'Calm', prompt: 'calm composed tone' },
  { id: 'urgent', label: 'Urgent', prompt: 'urgent high-focus tone' },
] as const;

export const TTS_TIMBRE_HINTS = [
  { id: 'auto', label: 'Auto', prompt: '' },
  { id: 'feminine', label: 'Feminine', prompt: 'feminine timbre hint' },
  { id: 'masculine', label: 'Masculine', prompt: 'masculine timbre hint' },
  { id: 'androgynous', label: 'Androgynous', prompt: 'androgynous balanced timbre hint' },
  { id: 'youthful', label: 'Youthful', prompt: 'youthful timbre hint' },
  { id: 'mature', label: 'Mature', prompt: 'mature timbre hint' },
] as const;

export const SCORE_LIMIT = 80;
export const MAX_BYTES = 150 * 1024 * 1024;

export const MODES: Array<{
  value: DemoMode;
  label: string;
  icon: LucideIcon;
  action: string;
  running: string;
}> = [
  { value: 'vad', label: 'VAD', icon: Waves, action: 'Run VAD', running: 'Running VAD\u2026' },
  { value: 'stt', label: 'STT', icon: MicVocal, action: 'Run STT', running: 'Running STT\u2026' },
  { value: 'localize', label: 'Voice Dub', icon: Sparkles, action: 'Run Voice Dub', running: 'Dubbing\u2026' },
  { value: 'tts', label: 'TTS Studio', icon: Headphones, action: 'Generate TTS', running: 'Generating TTS\u2026' },
];
