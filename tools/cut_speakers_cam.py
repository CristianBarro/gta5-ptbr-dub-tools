"""Speaker features for cutscene subtitles with CAM++ (the x-vector model inside Chatterbox's S3Gen).  (D:\\JARVIS\\.venv-tts)

usage: cut_speakers_cam.py <lote dir> <voice root>
Computes one CAM++ embedding per subtitle segment (voice stem of the dialogue channel) -> <lote>/seg_cam.npz
and prints how well it separates the lines whose speaker is known (conversation metadata + the reviewed pilot cut).
The GE2E VoiceEncoder used before could not tell protagonists from other male characters on short lines
(known lines: own cosine 0.62-0.86 vs best NPC cosine 0.60-0.79).
"""
import csv, sys, json, pathlib, collections, statistics, numpy as np, librosa, torch
from chatterbox.models.s3gen.xvector import CAMPPlus
lote, vroot = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
VOZES = pathlib.Path(r'C:\gta-dub\outputs\GTA_V_Dublagem_PTBR')
ckpt = pathlib.Path(r'D:\JARVIS\data\clone-models\hub\models--ResembleAI--chatterbox\snapshots\5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18')
cam = CAMPPlus(memory_efficient=False)
sd = torch.load(ckpt / 's3gen.pt', map_location='cpu', weights_only=True)
cam.load_state_dict({k[len('speaker_encoder.'):]: v for k, v in sd.items() if k.startswith('speaker_encoder.')}); cam.eval(); del sd
P3 = ('MICHAEL', 'FRANKLIN', 'TREVOR')
def emb(y):
    with torch.no_grad(): e = cam.inference([torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))])[0].numpy()
    return e / (np.linalg.norm(e) + 1e-9)
def centroid(es): c = np.mean(es, axis=0); return c / np.linalg.norm(c)
cuts = {r['cut']: r for r in csv.DictReader(open(lote / 'cutscenes.csv', encoding='utf-8-sig'))}
ev = list(csv.DictReader(open(lote / 'legendas_todas.csv', encoding='utf-8-sig')))
cache = lote / 'seg_cam.npz'
if cache.exists():
    z = np.load(cache); E, OK, LV = z['E'], z['OK'], z['LV']
else:
    E = None; OK = np.zeros(len(ev), bool); LV = np.full(len(ev), -99, np.float32)
    by_awc = collections.defaultdict(list)
    for i, e in enumerate(ev):
        c = cuts[e['cut']]
        if c['awc_path']: by_awc[c['awc_path']].append(i)
    for k, (awc, idx) in enumerate(sorted(by_awc.items()), 1):
        d = vroot / pathlib.Path(awc.replace('\\', '/')).stem
        main = (d / 'main.txt').read_text().strip() if (d / 'main.txt').exists() else None
        files = [d / f'ch_{main}_voice.wav'] if main else sorted(d.glob('ch_*_voice.wav'))
        if not files or not files[0].exists(): print('no voice for', awc); continue
        stems = [librosa.load(str(f), sr=16000)[0] for f in files]
        v = max(stems, key=lambda s: float((s ** 2).mean()))
        starts = sorted({float(ev[i]['audio_t']) for i in idx})
        for i in idx:
            t0 = float(ev[i]['audio_t']); nxt = next((s for s in starts if s > t0 + 0.05), len(v) / 16000)
            seg = v[int(t0 * 16000): int(min(nxt, t0 + max(float(ev[i]['dur']), 0.5) + 0.3) * 16000)]
            if len(seg) < 6400 or float(np.abs(seg).max()) < 0.01: continue
            x = emb(seg)
            if E is None: E = np.zeros((len(ev), len(x)), np.float32)
            E[i] = x; OK[i] = True
            fr = seg[: len(seg) // 160 * 160].reshape(-1, 160); db = 10 * np.log10((fr ** 2).mean(1) + 1e-12)
            LV[i] = float(np.mean(np.sort(db)[-max(3, len(db) // 3):]))
        if k % 40 == 0: print(f'features {k}/{len(by_awc)}', flush=True)
    np.savez(cache, E=E, OK=OK, LV=LV)
print('segments with features', int(OK.sum()), 'of', len(ev), 'dim', E.shape[1])
bank = VOZES / 'revisao_v4' / 'referencias_candidatas'
ref0 = {'MICHAEL': centroid([emb(librosa.load(str(p), sr=16000)[0]) for p in sorted(bank.glob('michael_*.wav'))]),
        'FRANKLIN': centroid([emb(librosa.load(str(p), sr=16000)[0]) for p in sorted(bank.glob('franklin_*.wav'))]),
        'TREVOR': emb(librosa.load(str(VOZES / 'revisao_v2' / 'referencias_preparadas' / 'trevor_reference.wav'), sr=16000)[0])}
np.savez(lote / 'ref_cam.npz', **ref0)
# evaluation labels: conversation metadata + reviewed pilot (family_3_int: protagonists by number, the rest = NPC)
truth = {i: e['known_speaker'] for i, e in enumerate(ev) if e['known_speaker']}
pil = json.loads((VOZES / 'cutscene_piloto' / 'family_3_int_speakers.json').read_text(encoding='utf-8'))
for n, i in enumerate([i for i, e in enumerate(ev) if e['cut'] == 'family_3_int']): truth[i] = pil.get(str(n), 'NPC')
S = np.stack([E @ ref0[w] for w in P3], 1)
own, oth, npc = [], [], []
for i, t in truth.items():
    if not OK[i]: continue
    pres = set(filter(None, cuts[ev[i]['cut']]['protagonists'].split(',')))
    sc = {w: float(S[i, j]) for j, w in enumerate(P3) if w in pres}
    if t in P3 and t in sc: own.append(sc[t]); oth += [v for w, v in sc.items() if w != t]
    elif t not in P3 and sc: npc.append(max(sc.values()))
q = lambda v: [round(x, 2) for x in statistics.quantiles(v, n=10)]
print('labelled: protagonist lines', len(own), 'npc lines', len(npc))
print('own cos deciles          ', q(own)); print('other protagonist deciles', q(oth)); print('npc best-protag deciles  ', q(npc))
