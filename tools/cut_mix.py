"""Remix a cutscene: replace the English voice of the dubbed protagonist lines by the chosen PT-BR takes.   (D:\\JARVIS\\.venv-tts)

usage: cut_mix.py <piloto dir> <cut name> <chan hash> [<chan hash> ...]
  <piloto>/<cut>_falantes.csv, <piloto>/<cut>_speakers.json, <piloto>/gen/escolha.json,
  <piloto>/orig/ch_<h>.wav (exact channel decode), <piloto>/sep/ch_<h>_voice.wav / _bg.wav (Demucs)
writes <piloto>/mix/ch_<h>.wav (same length as the original channel) and <piloto>/mix/mix_report.csv

Per dubbed line: the English speech region is found in the loudest channel's voice stem (energy, starting near the
subtitle time, ending at silence or the next subtitle). That region of the voice stem is faded out in every channel
and the PT-BR take is placed at the speech onset, level-matched per channel to the English voice it replaces (keeps the
original panning). A take longer than the free slot (until the next subtitle) is sped up, at most 1.3x; anything still
longer runs over. Output channel = background + edited voice.
"""
import sys, csv, json, pathlib, numpy as np, soundfile as sf, librosa
P, cut, hashes = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
SR = 48000
subs = list(csv.DictReader(open(P / f'{cut}_falantes.csv', encoding='utf-8-sig')))
spk = json.loads((P / f'{cut}_speakers.json').read_text(encoding='utf-8'))
pick = json.loads((P / 'gen' / 'escolha.json').read_text(encoding='utf-8'))
idp = next(iter(pick)).rsplit(":", 1)[0]  # queue ids use the audio stream name (e.g. family_3_int_seq)
orig = {h: sf.read(P / 'orig' / f'ch_{h}.wav', dtype='float32')[0] for h in hashes}
voice = {h: sf.read(P / 'sep' / f'ch_{h}_voice.wav', dtype='float32')[0] for h in hashes}
bg = {h: sf.read(P / 'sep' / f'ch_{h}_bg.wav', dtype='float32')[0] for h in hashes}
N = len(orig[hashes[0]])
main = max(hashes, key=lambda h: float(np.sqrt((voice[h] ** 2).mean())))
hop = 480  # 10 ms
env = np.sqrt(np.convolve(voice[main] ** 2, np.ones(hop) / hop, 'same')[::hop] + 1e-12)
thr = max(10 ** (-40 / 20), np.percentile(env, 90) * 10 ** (-24 / 20))
starts = sorted(float(s['audio_t']) for s in subs)
def next_sub(t): return next((s for s in starts if s > t + 0.05), N / SR)
# the English voice is muted through one mask built from the ORIGINAL stem; PT takes go to a separate track,
# so a later line's mute never cuts an earlier PT take
keep = np.ones(N, np.float32)
pt_track = {h: np.zeros(N, np.float32) for h in hashes}
report = []
for s in sorted(subs, key=lambda s: float(s['audio_t'])):
    who = spk.get(s['n'])
    if not who: continue
    lid = f"{idp}:{int(s['n']):03d}"
    t0, nxt = float(s['audio_t']), next_sub(float(s['audio_t']))
    # English speech region: first active frame from t0-0.12 s (earlier picks up the previous speaker's tail),
    # last active frame before the next subtitle
    a, b = int(max(0, t0 - 0.12) * 100), int(min(nxt, N / SR) * 100)
    act = np.where(env[a:b] > thr)[0]
    if len(act) == 0: on, off = t0, min(nxt, t0 + float(s['dur']))
    else: on, off = (a + act[0]) / 100, (a + act[-1] + 1) / 100 + 0.08
    off = min(max(off, on + 0.2), nxt + 0.15)
    pt, sr = sf.read(P / 'gen' / pick[lid]['file'], dtype='float32')
    pt = librosa.resample(pt, orig_sr=sr, target_sr=SR)
    pt, _ = librosa.effects.trim(pt, top_db=40, frame_length=1024, hop_length=128)
    slot = max(nxt - on - 0.04, off - on)
    rate = 1.0
    if len(pt) / SR > slot: rate = min(1.3, (len(pt) / SR) / slot); pt = librosa.effects.time_stretch(pt, rate=rate)
    i0, i1 = int(on * SR), int(off * SR)
    fade = int(0.02 * SR)
    keep[i0:i1] = 0
    keep[max(0, i0 - fade):i0] = np.minimum(keep[max(0, i0 - fade):i0], np.linspace(1, 0, i0 - max(0, i0 - fade)))
    keep[i1:i1 + fade] = np.minimum(keep[i1:i1 + fade], np.linspace(0, 1, len(keep[i1:i1 + fade])))
    j1 = min(N, i0 + len(pt)); seg = pt[: j1 - i0].copy()
    seg[:64] *= np.linspace(0, 1, min(64, len(seg))); seg[-256:] *= np.linspace(1, 0, len(seg[-256:]))
    prms = float(np.sqrt((seg ** 2).mean()) + 1e-9); ppk = float(np.abs(seg).max() + 1e-9)
    gains = {}
    for h in hashes:
        e = voice[h][i0:i1]
        # loudness of the English line in this channel, but never a higher peak than the English line had
        gains[h] = min(float(np.sqrt((e ** 2).mean()) + 1e-9) / prms, float(np.abs(e).max() + 1e-9) / ppk)
        pt_track[h][i0:j1] += seg * gains[h]
    report.append(dict(id=lid, who=who, sub_t=t0, en_on=round(on, 3), en_off=round(off, 3), next_sub=round(nxt, 3),
                       pt_seconds=round(len(pt) / SR, 3), speedup=round(rate, 3), overrun=round(max(0, on + len(pt) / SR - nxt), 3),
                       take=pick[lid]['take'], wer=pick[lid]['wer'], **{f'gain_{h}': round(g, 3) for h, g in gains.items()}))
(P / 'mix').mkdir(exist_ok=True)
for h in hashes:
    y = bg[h] + voice[h] * keep + pt_track[h]
    pk = float(np.abs(y).max()); opk = float(np.abs(orig[h]).max())
    over = np.abs(y) > 0.98                      # soft-limit only the samples that would clip; no global gain change
    y[over] = np.sign(y[over]) * (0.98 + 0.019 * np.tanh((np.abs(y[over]) - 0.98) / 0.019))
    sf.write(P / 'mix' / f'ch_{h}.wav', y, SR, subtype='PCM_16')
    print(h, 'peak', round(pk, 3), 'orig peak', round(opk, 3), 'limited samples', int(over.sum()), 'len ok', len(y) == len(orig[h]))
with open(P / 'mix' / 'mix_report.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(report[0])); w.writeheader(); w.writerows(report)
for r in report: print(r['id'], r['who'][:1], 'en', r['en_on'], '-', r['en_off'], 'pt', r['pt_seconds'], 'x', r['speedup'], 'over', r['overrun'])
