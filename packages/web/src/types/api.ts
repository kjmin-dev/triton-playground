export type DemoMode = 'vad' | 'stt' | 'localize' | 'tts';
export type TtsVoiceMode = 'reference_clone' | 'preset_voice';

export type ReadyResponse = {
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

export type VadResponse = {
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

export type SttResponse = {
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

export type LocalizeResponse = {
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
      elapsed_ms?: number;
      language?: string;
      message?: string;
      segment_count?: number;
      speaker_count?: number;
      status: string;
      task?: string;
      transcript?: string;
      segments?: Array<{
        start_ms: number;
        end_ms: number;
        duration_ms: number;
        text: string;
        average_probability: number;
        peak_probability: number;
        speaker_id?: string;
      }>;
    };
    translation: {
      elapsed_ms?: number;
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
      elapsed_ms?: number;
      language?: string;
      message?: string;
      reason?: string;
      sample_rate?: number;
      speaker_count?: number;
      speakers?: string[];
      status: string;
      voice_cloning?: boolean;
      voice_cloning_mode?: string;
    };
  };
  status: string;
  target_language: string;
  threshold: number;
  transcript: string;
  translated_text: string;
};

export type TtsPreviewVariant = {
  emotion: string;
  is_default: boolean;
  label: string;
  preview_id: string;
  prompt: string;
  text: string;
};

export type TtsActor = {
  actor_id: string;
  default_preview_id: string;
  description: string;
  is_default: boolean;
  label: string;
  language: string;
  preview_prompt: string;
  preview_text: string;
  preview_variants: TtsPreviewVariant[];
  speaker_name: string;
};

export type TtsActorCatalogResponse = {
  default_voice_mode: TtsVoiceMode;
  voice_modes: Array<{
    available: boolean;
    description: string;
    label: string;
    mode: TtsVoiceMode;
  }>;
  status: string;
  preset_actor_message: string | null;
  preset_actor_model_id: string | null;
  supports_preset_actors: boolean;
  supports_reference_voice_clone: boolean;
  reference_audio: {
    accepted_formats: string[];
    optional: boolean;
    recommended_sample_rate: number;
  };
  default_actor_id_by_language: Record<string, string>;
  actors: TtsActor[];
};

export type TtsResponse = {
  actor: string | null;
  actor_label: string | null;
  audio_base64: string;
  content_type: string;
  duration_ms: number;
  language: string;
  model: string;
  reference_audio_filename?: string | null;
  repository_model_name: string;
  sample_rate: number;
  status: string;
  text: string;
  voice_source_label: string;
  voice_mode: 'preset_actor' | 'reference_clone';
};
