"""Attribute cutscene subtitles to Michael / Franklin / Trevor in bulk.   (D:\\JARVIS\\.venv-tts python)

usage: cut_speakers_batch.py <lote dir> <voice root>
  <lote>/cutscenes.csv, <lote>/legendas_todas.csv (cut_inventory.py); <voice root>/<awc>/ch_*_voice.wav (cut_separate.py)
writes <lote>/falantes_todos.csv (+ features cache <lote>/seg_features.npz) and <lote>/calibracao.json

Evidence per subtitle segment (voice stem of the loudest channel, from the subtitle time to the next one):
  * who is physically in the scene (cutfPedModelObject player_zero/one/two) - only those can be candidates
  * VoiceEncoder cosine to each protagonist. Pass 1 uses the English reference clips; pass 2 uses centroids rebuilt
    from the confident pass-1 cutscene segments (same mastering / separation domain), which separates far better
  * F0 median (yin): voices far above a male range are not protagonists
  * vocative: a line addressing a protagonist by name is not spoken by that protagonist
Lines with a speaker already resolved by conversation metadata keep it. Thresholds are calibrated on those known lines
(protagonist vs other) for precision >= 0.95; anything below threshold stays English (not dubbed).
"""
import csv, sys, json, re, pathlib, collections, numpy as np, librosa, torch
from chatterbox.models.voice_encoder import VoiceEncoder
lote, vroot = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
VOZES = pathlib.Path(r'C:\gta-dub\outputs\GTA_V_Dublagem_PTBR')
ckpt = pathlib.Path(r'D:\JARVIS\data\clone-models\hub\models--ResembleAI--chatterbox\snapshots\5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18')
ve = VoiceEncoder(); ve.load_state_dict(torch.load(ckpt / 've.pt', map_location='cpu', weights_only=True)); ve.eval()
P3 = ('MICHAEL', 'FRANKLIN', 'TREVOR')
NAMES = {'MICHAEL': r'\b(michael|mike|mikey|m|de santa|desanta)\b', 'FRANKLIN': r'\b(franklin|frank|f|clinton)\b',
         'TREVOR': r'\b(trevor|trev|t|philips)\b'}
def vocative(text, who):  # "Michael, calm down!" / "calm down, Michael" / "Hey T!"
    t = text.lower(); n = NAMES[who]
    return bool(re.search(r'(^|[.!?]\s*|\bhey,?\s+|\byo,?\s+)' + n + r'\s*[,!?.]', t) or re.search(r',\s*' + n + r'\s*[!?.]*$', t))
def emb(y):
    e = ve.embeds_from_wavs([y], sample_rate=16000, as_spk=True); return e / np.linalg.norm(e)
def centroid(es): c = np.mean(es, axis=0); return c / np.linalg.norm(c)
bank = VOZES / 'revisao_v4' / 'referencias_candidatas'
ref0 = {'MICHAEL': centroid([emb(librosa.load(str(p), sr=16000)[0]) for p in sorted(bank.glob('michael_*.wav'))]),
        'FRANKLIN': centroid([emb(librosa.load(str(p), sr=16000)[0]) for p in sorted(bank.glob('franklin_*.wav'))]),
        'TREVOR': emb(librosa.load(str(VOZES / 'revisao_v2' / 'referencias_preparadas' / 'trevor_reference.wav'), sr=16000)[0])}
cuts = {r['cut']: r for r in csv.DictReader(open(lote / 'cutscenes.csv', encoding='utf-8-sig'))}
ev = list(csv.DictReader(open(lote / 'legendas_todas.csv', encoding='utf-8-sig')))
cache = lote / 'seg_features.npz'
if cache.exists():
    z = np.load(cache); E, F0, LV, OK = z['E'], z['F0'], z['LV'], z['OK']
else:
    E = np.zeros((len(ev), 256), np.float32); F0 = np.zeros(len(ev), np.float32); LV = np.full(len(ev), -99, np.float32); OK = np.zeros(len(ev), bool)
    by_awc = collections.defaultdict(list)
    for i, e in enumerate(ev):
        c = cuts[e['cut']]
        if c['awc_path'] and c['protagonists']: by_awc[c['awc_path']].append(i)
    for k, (awc, idx) in enumerate(sorted(by_awc.items()), 1):
        d = vroot / pathlib.Path(awc.replace('\\', '/')).stem
        files = sorted(d.glob('ch_*_voice.wav'))
        if not files: print('no voice for', awc); continue
        stems = [librosa.load(str(f), sr=16000)[0] for f in files]
        v = max(stems, key=lambda s: float((s ** 2).mean()))
        starts = sorted({float(ev[i]['audio_t']) for i in idx})
        for i in idx:
            t0 = float(ev[i]['audio_t']); nxt = next((s for s in starts if s > t0 + 0.05), len(v) / 16000)
            seg = v[int(t0 * 16000): int(min(nxt, t0 + max(float(ev[i]['dur']), 0.5) + 0.3) * 16000)]
            if len(seg) < 4800 or float(np.abs(seg).max()) < 0.01: continue
            E[i] = emb(seg)
            fr = seg[: len(seg) // 160 * 160].reshape(-1, 160); db = 10 * np.log10((fr ** 2).mean(1) + 1e-12)
            LV[i] = float(np.mean(np.sort(db)[-max(3, len(db) // 3):]))
            f0 = librosa.yin(seg, fmin=60, fmax=400, sr=16000, frame_length=1024)
            loud = db[: len(f0)] if len(db) >= len(f0) else np.pad(db, (0, len(f0) - len(db)), constant_values=-99)
            F0[i] = float(np.median(f0[loud[: len(f0)] > np.percentile(loud, 60)])) if len(f0) else 0
            OK[i] = True
        if k % 20 == 0: print(f'features {k}/{len(by_awc)}', flush=True)
    np.savez(cache, E=E, F0=F0, LV=LV, OK=OK)
known = np.array([e['known_speaker'] for e in ev])
present = [set(filter(None, cuts[e['cut']]['protagonists'].split(','))) for e in ev]
voc = {w: np.array([vocative(e['english'], w) for e in ev]) for w in P3}
def scores(refs):
    S = np.stack([E @ refs[w] for w in P3], 1)
    for j, w in enumerate(P3):
        S[[w not in p for p in present], j] = -1; S[voc[w], j] = -1
    S[~OK] = -1; return S
def decide(S, thr, margin):
    best = S.argmax(1); top = S.max(1); s2 = np.sort(S, 1)[:, -2]
    lab = np.where((top >= thr) & (top - s2 >= margin) & (F0 < 185), best, -1); return lab
# pass 1 (English references) -> confident cutscene segments -> domain centroids
S1 = scores(ref0); lab1 = decide(S1, 0.78, 0.06)
refs = {}
for j, w in enumerate(P3):
    sel = (lab1 == j) | (known == w)
    refs[w] = centroid(E[sel & OK]) if (sel & OK).sum() >= 10 else ref0[w]
    print(w, 'pass-1 confident segments', int((lab1 == j).sum()), '+ known', int(((known == w) & OK).sum()))
S2 = scores(refs)
# calibrate on lines whose speaker is known from conversation metadata
kn = OK & (known != '') & np.array([bool(p) for p in present])
best_cfg, table = None, []
for thr in np.arange(0.60, 0.90, 0.01):
    for margin in (0.02, 0.04, 0.06, 0.08, 0.10):
        lab = decide(S2, thr, margin)
        pred = np.array([P3[l] if l >= 0 else '' for l in lab])
        tp = int(((pred == known) & (pred != '') & kn).sum()); fp = int(((pred != known) & (pred != '') & kn).sum())
        pos = int((np.isin(known, P3) & kn).sum())
        prec = tp / max(1, tp + fp); rec = tp / max(1, pos)
        table.append(dict(thr=round(float(thr), 2), margin=margin, precision=round(prec, 3), recall=round(rec, 3), tp=tp, fp=fp))
        if prec >= 0.95 and (best_cfg is None or rec > best_cfg['recall']): best_cfg = table[-1]
if best_cfg is None: best_cfg = max(table, key=lambda r: (r['precision'], r['recall']))
print('calibration', best_cfg)
lab = decide(S2, best_cfg['thr'], best_cfg['margin'])
rows = []
for i, e in enumerate(ev):
    sp = e['known_speaker'] if e['known_speaker'] else (P3[lab[i]] if lab[i] >= 0 else '')
    src = 'metadata' if e['known_speaker'] else ('voice' if sp else '')
    rows.append(dict(e, speaker=sp if sp in P3 else '', speaker_source=src if sp in P3 else '',
                     cos_M=round(float(S2[i, 0]), 3), cos_F=round(float(S2[i, 1]), 3), cos_T=round(float(S2[i, 2]), 3),
                     f0=round(float(F0[i]), 1), level_db=round(float(LV[i]), 1), present=cuts[e['cut']]['protagonists']))
with open(lote / 'falantes_todos.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
(lote / 'calibracao.json').write_text(json.dumps(dict(chosen=best_cfg, table=table), indent=1), encoding='utf-8')
c = collections.Counter(r['speaker'] for r in rows if r['speaker'])
print('attributed', dict(c), 'of', len(rows), '| in cuts with protagonists', sum(1 for p in present if p))
