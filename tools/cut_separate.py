"""Split each channel of a cutscene mix into voice and background with Demucs (htdemucs_ft, GPU).

usage:
  cut_separate.py <out_dir> <chan0.wav> [<chan1.wav> ...]     writes <name>_voice.wav and <name>_bg.wav
  cut_separate.py --batch <orig_root> <voice_root>             every <orig_root>/<awc>/ch_*.wav -> <voice_root>/<awc>/ch_*_voice.wav
(D:\\JARVIS\\.venv-qwen python)
Each mono channel is run on its own (duplicated to stereo, model rate 44.1 kHz, resampled back).
bg = mix - voice, so voice + bg == mix exactly (batch mode stores only the voice; the mix step derives bg).
htdemucs_ft is a bag of 4 source-specialised models with one-hot weights: only the vocals specialist is run
(same vocals output as the full bag, ~4x faster).
"""
import sys, wave, pathlib, time, numpy as np, torch, julius
from demucs.pretrained import get_model
from demucs.apply import apply_model, BagOfModels
bag = get_model('htdemucs_ft'); bag.eval()
vi = bag.sources.index('vocals')
model = bag
if isinstance(bag, BagOfModels):
    w = [list(map(float, x)) for x in bag.weights]
    spec = [i for i, x in enumerate(w) if x[vi] > 0]
    if len(spec) == 1 and all(x[vi] == 0 for j, x in enumerate(w) if j != spec[0]):
        model = bag.models[spec[0]]; model.eval()
import os
# CUT_SEP_DEVICE=cpu frees the GPU (e.g. while the user plays); CUT_SEP_THREADS limits CPU threads
dev = os.environ.get('CUT_SEP_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')
if os.environ.get('CUT_SEP_THREADS'): torch.set_num_threads(int(os.environ['CUT_SEP_THREADS']))
def read(p):
    w = wave.open(str(p)); assert w.getnchannels() == 1 and w.getsampwidth() == 2
    return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768, w.getframerate()
def write(p, x, sr):  # temp file + rename: an interrupted run never leaves a truncated stem that looks finished
    p = pathlib.Path(p); tmp = p.with_suffix('.part')
    w = wave.open(str(tmp), 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes((np.clip(x, -1, 1) * 32767).round().astype(np.int16).tobytes()); w.close()
    tmp.replace(p)
def voice_of(x, sr):
    t = torch.from_numpy(x)[None]
    st = julius.resample_frac(t, sr, model.samplerate).expand(2, -1)[None]
    ref = st.mean(1, keepdim=True); m, s = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        src = apply_model(model, (st - m) / s, device=dev, shifts=1, split=True, overlap=0.25, progress=False)[0] * s + m
    v = julius.resample_frac(src[model.sources.index('vocals')].mean(0, keepdim=True).cpu(), model.samplerate, sr)[0].numpy()
    n = min(len(v), len(x)); return np.pad(v[:n], (0, len(x) - n))
rms = lambda x: float(np.sqrt((x ** 2).mean()) + 1e-12)
if sys.argv[1] == '--batch-main':
    # dialogue channel only (enough for speaker attribution): loudest original channel; if little of it is voice
    # (music-heavy channel; 1 of 56 checked AWCs), go on to the next loudest. main.txt names the channel kept.
    orig, vroot = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    dirs = sorted(d for d in orig.iterdir() if (d / 'done.txt').exists())
    t0 = time.time(); secs = 0
    for k, d in enumerate(dirs, 1):
        od = vroot / d.name
        if (od / 'done.txt').exists() or (od / 'main.txt').exists(): continue
        od.mkdir(parents=True, exist_ok=True)
        chans = {f: read(f) for f in sorted(d.glob('ch_*.wav'))}
        best = None
        for f in sorted(chans, key=lambda f: -rms(chans[f][0])):
            x, sr = chans[f]; vp = od / (f.stem + '_voice.wav')
            if vp.exists(): v = read(vp)[0]
            else: v = voice_of(x, sr); write(vp, v, sr); secs += len(x) / sr
            if best is None or rms(v) > best[1]: best = (f.stem[3:], rms(v))
            if rms(v) / rms(x) >= 0.5: break
        (od / 'main.txt').write_text(best[0] + '\n')
        if len(list(od.glob('ch_*_voice.wav'))) == len(chans): (od / 'done.txt').write_text('ok\n')
        print(f'[{k}/{len(dirs)}] {d.name} main {best[0]} audio {secs / 60:.1f} min in {(time.time() - t0) / 60:.1f} min', flush=True)
elif sys.argv[1] == '--batch-rest':
    # remaining channels, only for the AWCs listed (those that have dubbed lines to remix)
    orig, vroot = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    stems = [l.strip() for l in open(sys.argv[4], encoding='utf-8') if l.strip()]
    t0 = time.time(); secs = 0
    for k, stem in enumerate(stems, 1):
        d, od = orig / stem, vroot / stem
        if (od / 'done.txt').exists(): continue
        od.mkdir(parents=True, exist_ok=True)
        for f in sorted(d.glob('ch_*.wav')):
            vp = od / (f.stem + '_voice.wav')
            if vp.exists(): continue
            x, sr = read(f); write(vp, voice_of(x, sr), sr); secs += len(x) / sr
        (od / 'done.txt').write_text('ok\n')
        print(f'[{k}/{len(stems)}] {stem} audio {secs / 60:.1f} min in {(time.time() - t0) / 60:.1f} min', flush=True)
elif sys.argv[1] == '--batch':
    orig, vroot = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    dirs = sorted(d for d in orig.iterdir() if (d / 'done.txt').exists())
    t0 = time.time(); secs = 0
    for k, d in enumerate(dirs, 1):
        od = vroot / d.name
        if (od / 'done.txt').exists(): continue
        od.mkdir(parents=True, exist_ok=True)
        for f in sorted(d.glob('ch_*.wav')):
            x, sr = read(f); write(od / (f.stem + '_voice.wav'), voice_of(x, sr), sr); secs += len(x) / sr
        (od / 'done.txt').write_text('ok\n')
        print(f'[{k}/{len(dirs)}] {d.name} audio {secs / 60:.1f} min in {(time.time() - t0) / 60:.1f} min', flush=True)
else:
    out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    for f in map(pathlib.Path, sys.argv[2:]):
        x, sr = read(f); voice = voice_of(x, sr)
        write(out / (f.stem + '_voice.wav'), voice, sr); write(out / (f.stem + '_bg.wav'), x - voice, sr)
        print(f.name, 'voice rms', float(np.sqrt((voice ** 2).mean())), 'bg rms', float(np.sqrt(((x - voice) ** 2).mean())), dev)
