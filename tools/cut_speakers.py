"""Guess who speaks each cutscene subtitle: VoiceEncoder cosine of the separated voice vs protagonist references.

usage: cut_speakers.py <legendas.csv> <voice.wav> <out.csv>     (D:\\JARVIS\\.venv-tts python)
voice.wav = the Demucs voice stem of the loudest (dialogue) channel. Writes legendas + cos_MICHAEL/FRANKLIN/TREVOR + guess.
guess is only a suggestion (non-protagonists have no reference); the speaker column must be reviewed before dubbing.
"""
import csv, sys, pathlib, numpy as np, librosa, torch
from chatterbox.models.voice_encoder import VoiceEncoder
VOZES = pathlib.Path(r'C:\gta-dub\outputs\GTA_V_Dublagem_PTBR')
ckpt = pathlib.Path(r'D:\JARVIS\data\clone-models\hub\models--ResembleAI--chatterbox\snapshots\5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18')
ve = VoiceEncoder(); ve.load_state_dict(torch.load(ckpt / 've.pt', map_location='cpu', weights_only=True)); ve.eval()
def emb(y):
    e = ve.embeds_from_wavs([y], sample_rate=16000, as_spk=True); return e / np.linalg.norm(e)
def centroid(paths):
    c = np.mean([emb(librosa.load(str(p), sr=16000)[0]) for p in paths], axis=0); return c / np.linalg.norm(c)
bank = VOZES / 'revisao_v4' / 'referencias_candidatas'
refs = {'MICHAEL': centroid(sorted(bank.glob('michael_*.wav'))), 'FRANKLIN': centroid(sorted(bank.glob('franklin_*.wav'))),
        'TREVOR': centroid([VOZES / 'revisao_v2' / 'referencias_preparadas' / 'trevor_reference.wav'])}
leg, voice_p, out = sys.argv[1], sys.argv[2], sys.argv[3]
v, _ = librosa.load(voice_p, sr=16000)
rows = list(csv.DictReader(open(leg, encoding='utf-8-sig')))
for r in rows:
    a, b = float(r['audio_t']), float(r['audio_t']) + max(float(r['dur']), 0.6)
    seg = v[int(a * 16000): int(b * 16000)]
    cos = {w: float(emb(seg) @ c) for w, c in refs.items()} if len(seg) > 4000 else {w: 0.0 for w in refs}
    for w, c in cos.items(): r['cos_' + w] = round(c, 3)
    best = max(cos, key=cos.get)
    r['guess'] = best if cos[best] >= 0.72 and cos[best] - sorted(cos.values())[-2] >= 0.04 else 'OUTRO?'
    print(r['n'], r['audio_t'], r['guess'], cos, r['english'][:50])
with open(out, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
