"""Build a dub_batch.py queue for the protagonist lines of one cutscene.

usage: cut_queue.py <falantes.csv> <speakers.json> <awc_path> <out_queue.csv> [--pt-override overrides.json] [--voice voice_stem.wav]
speakers.json: {"<n>": "MICHAEL", ...} — reviewed speaker per subtitle number (only protagonists need entries).
The PT text is the game's own pt-BR subtitle; overrides.json {"<n>": "text"} replaces it (e.g. numbers spelled out).
target_duration_seconds = subtitle on-screen duration (the mix step re-measures the English voice).

--voice: Demucs voice stem of the dialogue channel. Each line's English delivery is measured (loudness of its speech
relative to the cutscene's median line, plus '!' in the text) and intense lines get a stronger Chatterbox delivery:
  score >= 6 dB  -> exaggeration 0.9, cfg_weight 0.3   (shouting)
  score >= 2.5   -> exaggeration 0.7, cfg_weight 0.35  (raised / angry)
  otherwise      -> the approved profile unchanged
(user feedback 2026-11-14 on the family_3_int pilot: Michael sounded flat, not angry, shouting at Amanda)
"""
import csv, json, sys, pathlib, argparse, wave
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('falantes'); ap.add_argument('speakers'); ap.add_argument('awc_path'); ap.add_argument('out')
ap.add_argument('--pt-override'); ap.add_argument('--voice'); a = ap.parse_args()
spk = json.loads(pathlib.Path(a.speakers).read_text(encoding='utf-8'))
ovr = json.loads(pathlib.Path(a.pt_override).read_text(encoding='utf-8')) if a.pt_override else {}
cut = pathlib.Path(a.awc_path.replace('\\', '/')).stem.replace('_mastered_only', '')
subs = list(csv.DictReader(open(a.falantes, encoding='utf-8-sig')))
level = {}
if a.voice:
    w = wave.open(a.voice); sr = w.getframerate(); v = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    starts = sorted(float(s['audio_t']) for s in subs)
    for s in subs:
        t0 = float(s['audio_t']); nxt = next((x for x in starts if x > t0 + 0.05), len(v) / sr)
        seg = v[int(t0 * sr): int(min(nxt, t0 + max(float(s['dur']), 0.4)) * sr)]
        if len(seg) < sr // 10: continue
        fr = seg[: len(seg) // 480 * 480].reshape(-1, 480); db = 10 * np.log10((fr ** 2).mean(1) + 1e-12)
        level[s['n']] = float(np.mean(np.sort(db)[-max(3, len(db) // 3):]))   # loudest third of the 10 ms frames
    med = float(np.median(list(level.values()))) if level else 0.0
rows = []
for r in subs:
    who = spk.get(r['n'])
    if not who: continue
    ex = cfg = ''; score = ''
    if r['n'] in level:
        score = round(level[r['n']] - med + (3.0 if '!' in r['english'] else 0.0), 2)
        if score >= 6: ex, cfg = '0.9', '0.3'
        elif score >= 2.5: ex, cfg = '0.7', '0.35'
    rows.append(dict(id=f"{cut}:{int(r['n']):03d}", character=who, text_pt=ovr.get(r['n'], r['portuguese']), english=r['english'],
                     awc_path=a.awc_path, stream_hash='', sample_rate='48000', samples='', target_duration_seconds=r['dur'],
                     status='cutscene', audio_t=r['audio_t'], hash=r['hash'], intensity=score, exaggeration=ex, cfg_weight=cfg))
with open(a.out, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), 'lines ->', a.out, '| intense:', sum(1 for r in rows if r['exaggeration']))
for r in rows: print(r['id'], r['character'][:1], r['intensity'], r['exaggeration'] or '-', r['english'][:45])
