"""Attribute cutscene subtitles to the protagonists: cutscene FACIAL ANIMATION + vocative rule + voice check.
(D:\\JARVIS\\.venv-tts python: numpy)

usage: cut_speakers_face.py <lote dir> <faces dir> [--eval] [--min-face 0.3] [--min-voice 0.4]
  faces dir: YcdProbe bonesall output, <cut>__<ped>.csv (t, per facial channel frame-to-frame change, 30 fps)
  needs <lote>/seg_cam.npz (CAM++ embedding + loudness per subtitle, cut_speakers_cam.py)

1. face: per subtitle window [cut_t, cut_t + max(dur, 0.5)] and per ped in the scene, mean facial motion
   (tracks 25+26, each channel scaled by that ped's own median+p90 over the cut), divided by the mean over the peds.
   A ped addressed by name in the line ("Be very afraid, Franklin!") cannot be its speaker.
2. voice centroids per protagonist, from the cutscene lines whose face winner is that protagonist by a wide margin
   (>= 1.0) - same mastering and separation as the lines to classify.
3. a line is given to protagonist P when P's face wins by >= --min-face AND the line's voice is close to P's centroid
   (cosine >= --min-voice); or, with an unclear face, when P is among the two most active faces and the voice is
   clearly P's (cosine >= min-voice + 0.15 and 0.05 above the other protagonists present). Everything else stays English.
The conversation-metadata speakers are NOT used: they proved wrong in several cutscenes (e.g. bar_3_rcm).
--eval: precision / recall on <lote>/rotulos_manuais.json + the reviewed pilot, over a grid of thresholds.
Otherwise writes <lote>/falantes_todos.csv for cut_queue_batch.py.
"""
import csv, sys, json, re, pathlib, argparse, collections, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('lote'); ap.add_argument('faces'); ap.add_argument('--eval', action='store_true')
ap.add_argument('--min-face', type=float, default=0.6); ap.add_argument('--min-voice', type=float, default=0.30)
a = ap.parse_args()
lote, faces = pathlib.Path(a.lote), pathlib.Path(a.faces)
VOZES = pathlib.Path(r'C:\gta-dub\outputs\GTA_V_Dublagem_PTBR')
P3 = ('MICHAEL', 'FRANKLIN', 'TREVOR'); PLAYER = {'player_zero': 'MICHAEL', 'player_one': 'FRANKLIN', 'player_two': 'TREVOR'}
NAMES = {'MICHAEL': r'(michael|mike|mikey|m|de santa|desanta)', 'FRANKLIN': r'(franklin|frank|f|clinton)', 'TREVOR': r'(trevor|trev|t|philips)'}
def addressed(text, who):
    t = text.lower(); n = NAMES[who]
    return bool(re.search(r'(^|[.!?]\s*|\bhey,?\s+|\byo,?\s+)' + n + r'\s*[,!?.]', t) or re.search(r',\s*' + n + r'\s*[!?.]*$', t))
def who_of(ped):
    for k, w in PLAYER.items():
        if ped.lower().startswith(k): return w
    return None
cuts = {r['cut']: r for r in csv.DictReader(open(lote / 'cutscenes.csv', encoding='utf-8-sig'))}
ev = list(csv.DictReader(open(lote / 'legendas_todas.csv', encoding='utf-8-sig')))
z = np.load(lote / 'seg_cam.npz'); E, OK, LV = z['E'], z['OK'], z['LV']
by_cut = collections.defaultdict(list)
for i, e in enumerate(ev): by_cut[e['cut']].append(i)
def load_faces(cut):
    # facial tracks 24+25+26; each channel scaled by the ped's own (median + p90), but never by less than half the
    # scene's typical scale - otherwise a face that barely moves (a listener) gets its tiny twitches blown up.
    # (manual labels: precision 0.885 / recall 0.767 face-only, vs 0.767 / 0.903 with plain per-ped scaling)
    raw = {}
    for f in faces.glob(f'{cut}__*.csv'):
        rows = list(csv.reader(open(f)))
        if len(rows) < 3: continue
        cols = rows[0][1:]; x = np.array([[float(v) for v in r] for r in rows[1:]], np.float32)
        sel = [j for j, c in enumerate(cols) if c.split(':')[1] in ('24', '25', '26')]
        if sel: raw[f.stem.split('__', 1)[1]] = (x[:, 0], x[:, 1:][:, sel], [cols[j] for j in sel])
    if not raw: return {}
    # scene scale per channel, from every ped that has that channel (rigs differ: animals, some NPC heads)
    pool = collections.defaultdict(list)
    for _, x, c in raw.values():
        for j, k in enumerate(c): pool[k].append(x[:, j])
    g = {k: float(np.median(np.concatenate(v)) + np.percentile(np.concatenate(v), 90)) + 1e-6 for k, v in pool.items()}
    out = {}
    for p, (t, x, c) in raw.items():
        own = np.median(x, 0) + np.percentile(x, 90, 0)
        s = np.maximum(own, 0.5 * np.array([g[k] for k in c])) + 1e-6
        out[p] = (t, (x / s).mean(1))
    return out
# ---- face pass over every subtitle ----
face = [None] * len(ev)            # (ranked [(ped, score)], ) per line
for cut, idx in by_cut.items():
    if not cuts[cut]['protagonists']: continue
    F = load_faces(cut)
    if not F: continue
    for i in idx:
        e = ev[i]; t0 = float(e['cut_t']); t1 = t0 + max(float(e['dur']), 0.5)
        acts = {}
        for ped, (t, m) in F.items():
            w = (t >= t0) & (t < t1)
            if w.sum() >= 4: acts[ped] = float(m[w].mean())
        if not acts: continue
        mean = np.mean(list(acts.values())) + 1e-6
        sc = {p: v / mean for p, v in acts.items()}
        for p in list(sc):
            w = who_of(p)
            if w and addressed(e['english'], w): sc[p] = -1.0      # nobody calls himself by name
        face[i] = sorted(sc.items(), key=lambda kv: -kv[1])
def face_win(i):
    r = face[i]
    if not r: return None, 0.0, None
    second = r[1][1] if len(r) > 1 else 0.0
    return r[0][0], r[0][1] - second, (r[1][0] if len(r) > 1 else None)
# ---- voice centroids from face-confident lines ----
cent = {}
for w in P3:
    sel = [i for i in range(len(ev)) if face[i] and OK[i] and who_of(face_win(i)[0] or '') == w and face_win(i)[1] >= 1.0]
    c = E[sel].mean(0); cent[w] = c / np.linalg.norm(c)
    print(w, 'voice centroid from', len(sel), 'face-confident lines', flush=True)
def classify(i, min_face, min_voice):
    if not face[i] or not OK[i]: return ''
    present = set(filter(None, cuts[ev[i]['cut']]['protagonists'].split(',')))
    vc = {w: float(E[i] @ cent[w]) for w in P3 if w in present}
    ped, margin, ped2 = face_win(i); w = who_of(ped or '')
    # (a voice "rescue" for unclear faces was tried and only added false positives on the manual labels)
    if w and margin >= min_face and vc.get(w, -1) >= min_voice: return w
    return ''
truth = {}
man = json.loads((lote / 'rotulos_manuais.json').read_text(encoding='utf-8'))
for cut, labels in man.items():
    if cut.startswith('_'): continue
    for n, i in enumerate(by_cut[cut]):
        if str(n) in labels: truth[i] = labels[str(n)]
pil = json.loads((VOZES / 'cutscene_piloto' / 'family_3_int_speakers.json').read_text(encoding='utf-8'))
for n, i in enumerate(by_cut['family_3_int']): truth[i] = pil.get(str(n), 'NPC')
def score(fn):
    tp = fp = fn_ = 0
    for i, t in truth.items():
        p = fn(i)
        if p and p == t: tp += 1
        elif p: fp += 1
        elif t in P3: fn_ += 1
    return tp, fp, fn_
if a.eval:
    tp, fp, fn_ = score(lambda i: (who_of(face_win(i)[0] or '') or '') if face[i] else '')
    print(f'face only (winner)                   precision {tp / max(1, tp + fp):.3f} recall {tp / max(1, tp + fn_):.3f} (tp {tp} fp {fp} fn {fn_}) labelled {len(truth)}')
    for mf in (0.0, 0.3, 0.6, 1.0):
        for mv in (-1.0, 0.3, 0.4, 0.45, 0.5):
            tp, fp, fn_ = score(lambda i: classify(i, mf, mv))
            print(f'min-face {mf:.1f} min-voice {mv:5.2f}             precision {tp / max(1, tp + fp):.3f} recall {tp / max(1, tp + fn_):.3f} (tp {tp} fp {fp} fn {fn_})', flush=True)
    sys.exit()
rows = []; c = collections.Counter()
for i, e in enumerate(ev):
    sp = classify(i, a.min_face, a.min_voice) if cuts[e['cut']]['protagonists'] else ''
    ped, margin, _ = face_win(i) if face[i] else (None, 0.0, None)
    c[sp or '-'] += 1
    rows.append(dict(e, speaker=sp, speaker_source='face+voice' if sp else '', face_winner=ped or '', face_margin=round(margin, 2),
                     present=cuts[e['cut']]['protagonists'], level_db=round(float(LV[i]), 1)))
with open(lote / 'falantes_todos.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('attributed', dict(c), '| lines without face data', sum(1 for i in range(len(ev)) if face[i] is None))
