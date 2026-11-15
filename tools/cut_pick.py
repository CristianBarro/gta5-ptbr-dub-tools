"""ASR every generated take of a cutscene batch and pick one per line.   (D:\\JARVIS\\.venv python, faster-whisper)

usage: cut_pick.py <gen dir> [--model large-v3|medium]   -> <gen>/escolha.json {id: {file, wer, seconds, asr, all: [...]}}
Incremental: a take already transcribed (same file) is not transcribed again; progress is saved every 50 lines.
Pick: lowest word error rate (in steps of 0.1); ties -> shorter take. Takes over 3x the subtitle time or with
WER > 0.5 are only used when nothing better exists, and are then flagged.
"""
import sys, json, re, unicodedata, pathlib, wave, argparse, numpy as np
from faster_whisper import WhisperModel
ap = argparse.ArgumentParser(); ap.add_argument('gen'); ap.add_argument('--model', default='large-v3'); a = ap.parse_args()
gen = pathlib.Path(a.gen); man = json.loads((gen / 'manifest.json').read_text(encoding='utf-8'))
out_p = gen / 'escolha.json'
res = json.loads(out_p.read_text(encoding='utf-8')) if out_p.exists() else {}
done = {c['file']: c for v in res.values() for c in v.get('all', [])}
def words(s):
    s = unicodedata.normalize('NFKD', s.lower()); s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", s)
def dist(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1): p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[len(b)]
def load16(p):
    w = wave.open(str(p)); sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    if sr == 24000: x = np.interp(np.arange(0, len(x), 1.5), np.arange(len(x)), x).astype(np.float32)
    return x
m = WhisperModel(a.model, device='cpu', compute_type='int8')
n_new = 0
for lid, e in sorted(man.items()):
    ref = words(e['text_tts']); cands = []
    for k, t in e['takes'].items():
        if t.get('status') != 'generated': continue
        if t['file'] in done and done[t['file']].get('text') == e['text_tts']: cands.append(done[t['file']]); continue
        segs, _ = m.transcribe(load16(gen / t['file']), language='pt', beam_size=5, temperature=0)
        hyp = ' '.join(s.text.strip() for s in segs); wer = dist(ref, words(hyp)) / max(1, len(ref))
        cands.append(dict(take=k, file=t['file'], seconds=t['seconds'], wer=round(wer, 3), asr=hyp, text=e['text_tts'])); n_new += 1
    if not cands: continue
    ok = [c for c in cands if c['wer'] <= 0.5 and c['seconds'] <= 3 * max(0.8, e['target_seconds'])]
    best = min(ok or cands, key=lambda c: (round(c['wer'] / 0.1), c['seconds']))
    res[lid] = dict(best, character=e['character'], text=e['text_tts'], flagged=not ok, all=cands)
    if n_new and n_new % 50 == 0:
        out_p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8'); print('progress', len(res), 'lines', flush=True)
out_p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
bad = sum(1 for v in res.values() if v['wer'] > 0.3 or v['flagged'])
print('picked', len(res), 'lines | new transcriptions', n_new, '| bad (WER>0.3 or flagged)', bad)
