import { Pause, Play, Volume2 } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';

import { langLabel } from '@/lib/format';
import type { TtsActor, TtsPreviewVariant } from '@/types/api';

function previewStaticUrl(actorId: string, previewId: string) {
  return `/audio/previews/${actorId}/${previewId}.wav`;
}

export function VoiceLibrary({
  actors,
  selectedActorId,
  language,
  onSelectActor,
  onLanguageChange,
  catalogError,
}: {
  actors: TtsActor[];
  selectedActorId: string;
  language: string;
  onSelectActor: (actorId: string) => void;
  onLanguageChange: (language: string) => void;
  catalogError: string | null;
}) {
  const [playingKey, setPlayingKey] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const stopPlayback = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    setPlayingKey(null);
  }, []);

  useEffect(() => {
    return () => {
      if (audioRef.current) audioRef.current.pause();
    };
  }, []);

  function playPreview(actorId: string, variant: TtsPreviewVariant) {
    const key = `${actorId}:${variant.preview_id}`;

    if (playingKey === key) {
      stopPlayback();
      return;
    }

    stopPlayback();

    const audio = new Audio(previewStaticUrl(actorId, variant.preview_id));
    audioRef.current = audio;
    setPlayingKey(key);

    audio.addEventListener('ended', () => setPlayingKey(null));
    audio.addEventListener('error', () => setPlayingKey(null));
    audio.play().catch(() => setPlayingKey(null));
  }

  function handleSelectActor(actor: TtsActor) {
    if (actor.language !== language) {
      onLanguageChange(actor.language);
    }
    onSelectActor(actor.actor_id);
  }

  return (
    <div className='space-y-3'>
      <div className='flex items-center justify-between gap-3'>
        <div>
          <h2 className='text-sm font-medium text-slate-900'>Voice Library</h2>
          <p className='text-xs text-slate-500'>
            Select a voice to use. Clicking a voice from another language will switch the output language automatically.
          </p>
        </div>
        {catalogError && <span className='text-xs text-red-600'>{catalogError}</span>}
      </div>

      <div className='grid gap-3 sm:grid-cols-2'>
        {actors.map((actor) => {
          const active = selectedActorId === actor.actor_id;
          const matchesLanguage = actor.language === language;
          return (
            <div
              className={`cursor-pointer rounded-2xl border p-4 transition ${
                active
                  ? 'border-cyan-400 bg-cyan-50/80 shadow-sm ring-1 ring-cyan-200'
                  : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'
              }`}
              key={actor.actor_id}
              onClick={() => handleSelectActor(actor)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  handleSelectActor(actor);
                }
              }}
              role='button'
              tabIndex={0}
            >
              {/* Actor header */}
              <div className='flex items-center justify-between gap-3'>
                <div className='flex items-center gap-2.5'>
                  <div
                    className={`flex h-8 w-8 items-center justify-center rounded-full text-xs font-bold ${
                      active ? 'bg-cyan-500 text-white' : 'bg-slate-100 text-slate-600'
                    }`}
                  >
                    {actor.label[0]}
                  </div>
                  <div>
                    <div className='flex items-center gap-2'>
                      <span className='text-sm font-semibold text-slate-950'>{actor.label}</span>
                      {actor.is_default && (
                        <span className='rounded bg-slate-900 px-1 py-px text-[9px] font-bold tracking-wider text-white'>
                          DEFAULT
                        </span>
                      )}
                    </div>
                    <p className='text-xs text-slate-500'>
                      {langLabel(actor.language)}
                      {!matchesLanguage && (
                        <span className='ml-1 text-slate-400'>&middot; will switch from {langLabel(language)}</span>
                      )}
                    </p>
                  </div>
                </div>
                {active && <Volume2 className='h-4 w-4 shrink-0 text-cyan-600' />}
              </div>

              {/* Emotion preset buttons */}
              <div className='mt-3 flex flex-wrap gap-1.5'>
                {actor.preview_variants.map((variant) => {
                  const key = `${actor.actor_id}:${variant.preview_id}`;
                  const isPlaying = playingKey === key;
                  return (
                    <button
                      key={variant.preview_id}
                      className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-medium transition ${
                        isPlaying
                          ? 'border-cyan-400 bg-cyan-100 text-cyan-800'
                          : 'border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:bg-slate-50'
                      }`}
                      onClick={(e) => {
                        e.stopPropagation();
                        playPreview(actor.actor_id, variant);
                      }}
                      type='button'
                    >
                      {isPlaying ? <Pause className='h-3 w-3' /> : <Play className='h-3 w-3' />}
                      {variant.label}
                    </button>
                  );
                })}
              </div>

              {/* Selection hint */}
              <p className='mt-2.5 text-[11px] text-slate-400'>{active ? 'Currently selected' : 'Click to select'}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
