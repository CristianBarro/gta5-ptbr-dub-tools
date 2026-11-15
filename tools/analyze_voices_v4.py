"""Acoustic diagnostics for v0.4: candidate reference clips and previous samples.

Speaker embeddings use the local Chatterbox VoiceEncoder (ve.pt). F0 uses librosa pyin.
These numbers are diagnostics only; they are not a substitute for listening approval.
usage: analyze_voices_v4.py <revisao_v4 folder> <revisao_v3 folder>
"""
import os, sys, json, pathlib
v4 = pathlib.Path(sys.argv[1]).resolve(); v3 = pathlib.Path(sys.argv[2]).resolve()
for key in ['TEMP', 'TMP', 'TMPDIR', 'NUMBA_CACHE_DIR']:
    p = v4 / 'cache' / ('numba' if key == 'NUMBA_CACHE_DIR' else 'tmp'); p.mkdir(parents=True, exist_ok=True); os.environ[key] = str(p)
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import numpy as np, librosa, torch
from chatterbox.models.voice_encoder import VoiceEncoder
CKPT = pathlib.Path(r'D:\JARVIS\data\clone-models\hub\models--ResembleAI--chatterbox\snapshots\5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18')
ve = VoiceEncoder(); ve.load_state_dict(torch.load(CKPT / 've.pt', map_location='cpu', weights_only=True)); ve.eval()

def describe(path):
    y, _ = librosa.load(str(path), sr=16000)
    emb = ve.embeds_from_wavs([y], sample_rate=16000, as_spk=True)
    emb = emb / np.linalg.norm(emb)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=16000, frame_length=1024)
    f0v = f0[voiced & np.isfinite(f0)]
    flat = float(np.median(librosa.feature.spectral_flatness(y=y)))
    return dict(seconds=round(len(y) / 16000, 3), f0_median=round(float(np.median(f0v)), 1) if len(f0v) else None,
                f0_p10=round(float(np.percentile(f0v, 10)), 1) if len(f0v) else None,
                f0_p90=round(float(np.percentile(f0v, 90)), 1) if len(f0v) else None,
                voiced_ratio=round(float(np.mean(voiced)), 3), spectral_flatness=round(flat, 4)), emb

rows, embs = {}, {}
for f in sorted((v4 / 'referencias_candidatas').glob('*.wav')):
    rows[f.name], embs[f.name] = describe(f)
for who in ['michael', 'franklin', 'trevor']:
    name = f'v3_{who}.wav'; rows[name], embs[name] = describe(v3 / 'amostras_ptbr' / f'{who}.wav')
    name = f'v3ref_{who}.wav'; rows[name], embs[name] = describe(v3 / 'referencias_preparadas' / f'{who}_reference.wav')
cent = {}
for who in ['michael', 'franklin']:
    keys = [k for k in embs if k.startswith(who + '_')]
    c = np.mean([embs[k] for k in keys], axis=0); cent[who] = c / np.linalg.norm(c)
for k in rows:
    rows[k]['cos_michael_centroid'] = round(float(embs[k] @ cent['michael']), 4)
    rows[k]['cos_franklin_centroid'] = round(float(embs[k] @ cent['franklin']), 4)
    rows[k]['cos_trevor_v3ref'] = round(float(embs[k] @ embs['v3ref_trevor.wav']), 4)
for k, r in rows.items(): print(k, json.dumps(r), flush=True)
(v4 / 'diagnostico_referencias.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
