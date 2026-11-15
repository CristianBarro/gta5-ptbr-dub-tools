"""Transcribe candidate reference clips (English) with the local faster-whisper small model."""
import sys, json, pathlib, os
os.environ['HF_HUB_OFFLINE'] = '1'
folder = pathlib.Path(sys.argv[1])
from faster_whisper import WhisperModel
snap = next(pathlib.Path(r'C:\Users\<user>\.cache\huggingface\hub\models--Systran--faster-whisper-small\snapshots').iterdir())
m = WhisperModel(str(snap), device='cpu', compute_type='int8', cpu_threads=6, local_files_only=True)
out = {}
for f in sorted(folder.glob('*.wav')):
    segs, info = m.transcribe(str(f), language='en', beam_size=5, vad_filter=False)
    segs = list(segs)
    out[f.name] = dict(text=' '.join(s.text.strip() for s in segs),
                       min_avg_logprob=min([s.avg_logprob for s in segs] or [-9]),
                       max_no_speech=max([s.no_speech_prob for s in segs] or [1]))
    print(f.name, json.dumps(out[f.name], ensure_ascii=False), flush=True)
(folder.parent / 'reference_transcripts.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
