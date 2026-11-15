"""Check subtitle time vs. where the line is actually heard, on a sample of cutscenes.   (D:\\JARVIS\\.venv, faster-whisper)

usage: cut_align_check.py <lote dir> <voice root> [n_cuts=24] [seed=7]
For each sampled cut: Whisper (small.en, word timestamps) on the dialogue-channel voice stem; each subtitle with >= 3
words is matched to the ASR word stream (its first 3 words in order, within +-8 s of the subtitle time);
delta = heard_time - subtitle audio_t. Writes <lote>/alinhamento_amostra.csv; prints median / spread per cut.
"""
import csv, sys, re, random, pathlib, wave, collections, statistics, numpy as np
from faster_whisper import WhisperModel
lote, vroot = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
n_cuts = int(sys.argv[3]) if len(sys.argv) > 3 else 24; seed = int(sys.argv[4]) if len(sys.argv) > 4 else 7
cuts = {r['cut']: r for r in csv.DictReader(open(lote / 'cutscenes.csv', encoding='utf-8-sig'))}
ev = collections.defaultdict(list)
for e in csv.DictReader(open(lote / 'legendas_todas.csv', encoding='utf-8-sig')): ev[e['cut']].append(e)
norm = lambda s: re.findall(r"[a-z']+", s.lower().replace('’', "'"))
cand = sorted(c for c, v in ev.items() if cuts[c]['awc_path'] and cuts[c]['protagonists'] and len(v) >= 8 and float(cuts[c]['awc_seconds']) < 240)
random.Random(seed).shuffle(cand); sample = sorted(cand[:n_cuts])
m = WhisperModel('small.en', device='cpu', compute_type='int8')
rows = []
for cut in sample:
    d = vroot / pathlib.Path(cuts[cut]['awc_path'].replace('\\', '/')).stem
    main = (d / 'main.txt').read_text().strip() if (d / 'main.txt').exists() else None
    files = [d / f'ch_{main}_voice.wav'] if main else sorted(d.glob('ch_*_voice.wav'))
    best = None
    for f in files:
        w = wave.open(str(f)); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        if best is None or float((x ** 2).mean()) > best[0]: best = (float((x ** 2).mean()), x)
    x = best[1]; x = x[: len(x) // 3 * 3].reshape(-1, 3).mean(1).astype(np.float32)
    words = []
    for s in m.transcribe(x, language='en', word_timestamps=True, vad_filter=False)[0]:
        for wd in s.words or []:
            for t in norm(wd.word): words.append((t, wd.start))
    deltas = []
    for e in ev[cut]:
        ws = norm(e['english'])
        if len(ws) < 3: continue
        t0 = float(e['audio_t']); hit = None
        for i in range(len(words) - 2):
            if abs(words[i][1] - t0) > 8: continue
            if [words[i + j][0] for j in range(3)] == ws[:3]:
                if hit is None or abs(words[i][1] - t0) < abs(hit - t0): hit = words[i][1]
        if hit is not None: deltas.append(hit - t0)
        rows.append(dict(cut=cut, audio_t=e['audio_t'], english=e['english'], heard=round(hit, 2) if hit is not None else '',
                         delta=round(hit - t0, 2) if hit is not None else ''))
    n3 = sum(1 for e in ev[cut] if len(norm(e['english'])) >= 3)
    if deltas:
        print(f"{cut:28s} offset {float(cuts[cut]['audio_offset']):7.2f}  matched {len(deltas):3d}/{n3:3d}  median {statistics.median(deltas):+.2f}  "
              f"p10 {np.percentile(deltas, 10):+.2f}  p90 {np.percentile(deltas, 90):+.2f}", flush=True)
    else: print(f"{cut:28s} offset {float(cuts[cut]['audio_offset']):7.2f}  matched 0/{n3}", flush=True)
with open(lote / 'alinhamento_amostra.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
al = [r['delta'] for r in rows if r['delta'] != '']
print('ALL matched', len(al), 'of', sum(1 for r in rows), '| median', round(statistics.median(al), 2), '| within +-0.6 s', round(sum(1 for d in al if abs(d) <= 0.6) / len(al), 3))
