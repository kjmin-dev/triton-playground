import type { DemoMode, LocalizeResponse, SttResponse, TtsResponse, VadResponse } from '@/types/api';

import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { LocalizeResults } from './localize-results';
import { SttResults } from './stt-results';
import { TtsResults } from './tts-results';
import { VadResults } from './vad-results';

export function ResultsPanel({
  tab,
  onTabChange,
  vad,
  sttRes,
  loc,
  ttsRes,
  thr,
  thrOk,
  originalAudioUrl,
  audioSrc,
  ttsAudioSrc,
  ttsReferenceAudioUrl,
  ttsReferenceFileName,
  workerBaseUrl,
}: {
  tab: DemoMode;
  onTabChange: (tab: DemoMode) => void;
  vad: VadResponse | null;
  sttRes: SttResponse | null;
  loc: LocalizeResponse | null;
  ttsRes: TtsResponse | null;
  thr: number;
  thrOk: boolean;
  originalAudioUrl: string | null;
  audioSrc: string | null;
  ttsAudioSrc: string | null;
  ttsReferenceAudioUrl: string | null;
  ttsReferenceFileName: string | null;
  workerBaseUrl: string;
}) {
  return (
    <section className='mt-8 border-t border-slate-200 pt-6'>
      <Tabs onValueChange={(v) => onTabChange(v as DemoMode)} value={tab}>
        <TabsList className='mb-6 grid w-full grid-cols-4'>
          <TabsTrigger value='vad'>
            VAD
            {vad ? <span className='ml-1.5 inline-block h-1.5 w-1.5 rounded-full bg-emerald-500' /> : null}
          </TabsTrigger>
          <TabsTrigger value='stt'>
            STT
            {sttRes ? <span className='ml-1.5 inline-block h-1.5 w-1.5 rounded-full bg-emerald-500' /> : null}
          </TabsTrigger>
          <TabsTrigger value='localize'>
            Voice Dub
            {loc ? <span className='ml-1.5 inline-block h-1.5 w-1.5 rounded-full bg-emerald-500' /> : null}
          </TabsTrigger>
          <TabsTrigger value='tts'>
            TTS Studio
            {ttsRes ? <span className='ml-1.5 inline-block h-1.5 w-1.5 rounded-full bg-emerald-500' /> : null}
          </TabsTrigger>
        </TabsList>

        <TabsContent value='vad'>
          <VadResults vad={vad} thrOk={thrOk} thr={thr} />
        </TabsContent>

        <TabsContent value='stt'>
          <SttResults sttRes={sttRes} />
        </TabsContent>

        <TabsContent value='localize'>
          <LocalizeResults
            loc={loc}
            originalAudioUrl={originalAudioUrl}
            audioSrc={audioSrc}
            workerBaseUrl={workerBaseUrl}
          />
        </TabsContent>

        <TabsContent value='tts'>
          <TtsResults
            ttsRes={ttsRes}
            ttsAudioSrc={ttsAudioSrc}
            ttsReferenceAudioUrl={ttsReferenceAudioUrl}
            ttsReferenceFileName={ttsReferenceFileName}
          />
        </TabsContent>
      </Tabs>
    </section>
  );
}
