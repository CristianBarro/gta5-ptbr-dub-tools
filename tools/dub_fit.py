"""Fit the selected takes to the duration of the original stream.

Per line (selected take from qa_report.csv, original duration from the queue):
  1. within tolerance (<= target * TOL)  -> unchanged
  2. shorten long internal pauses (> 0.18 s) down to a 0.12 s floor, 10 ms crossfades
  3. tempo up to MAX_TEMPO with ffmpeg atempo (keeps pitch), aiming at target * AIM
  4. still too long -> status 'encurtar_texto' (Portuguese line must be adapted; never
     sped up beyond MAX_TEMPO, never cut)
Shorter-than-original takes are kept as they are.

usage: dub_fit.py <pilot dir> [--ffmpeg path]   (D:\\JARVIS\\.venv-tts python)
"""
import sys, csv, json, argparse, pathlib, subprocess, tempfile
import numpy as np, soundfile as sf
ap = argparse.ArgumentParser()
ap.add_argument('pilot', type=pathlib.Path)
ap.add_argument('--ffmpeg', default=r'C:\Program Files\Shotcut\ffmpeg.exe')
args = ap.parse_args()
out = args.pilot.resolve()
TOL, AIM, MAX_TEMPO = 1.10, 1.05, 1.25
PAUSE_MIN, PAUSE_FLOOR, XF = 0.18, 0.12, 0.010

def silent_runs(y, sr, hop=0.01):
    n = int(sr * hop); frames = len(y) // n
    rms = np.sqrt(np.mean(y[:frames * n].reshape(frames, n) ** 2, axis=1))
    thr = max(1e-4, float(np.max(rms)) * 10 ** (-38 / 20))
    quiet = rms < thr; runs, start = [], None
    for i, q in enumerate(quiet):
        if q and start is None: start = i
        if (not q or i == frames - 1) and start is not None:
            end = i if not q else i + 1
            if start > 0 and end < frames: runs.append((start * n, end * n))  # internal only
            start = None
    return [(a, b) for a, b in runs if (b - a) / sr > PAUSE_MIN]

def compress_pauses(y, sr, need):
    runs = silent_runs(y, sr)
    spare = [(a, b, (b - a) - int(PAUSE_FLOOR * sr)) for a, b in runs]
    total = sum(s for *_, s in spare)
    if total <= 0 or need <= 0: return y, 0.0
    frac = min(1.0, need * sr / total); xf = int(XF * sr)
    pieces, pos = [], 0
    for a, b, s in spare:
        cut = int(s * frac)
        if cut < 2 * xf: continue
        mid = (a + b) // 2; c0, c1 = mid - cut // 2, mid - cut // 2 + cut
        pieces.append(y[pos:c0]); pos = c1
    pieces.append(y[pos:])
    res = pieces[0]
    for p in pieces[1:]:  # equal-power crossfade at each join (joins are inside silence)
        k = min(xf, len(res), len(p)); w = np.linspace(0, 1, k)
        res = np.concatenate([res[:-k], res[-k:] * np.cos(w * np.pi / 2) + p[:k] * np.sin(w * np.pi / 2), p[k:]])
    return res.astype(np.float32), (len(y) - len(res)) / sr

def atempo(y, sr, factor):
    with tempfile.TemporaryDirectory(dir=out / 'cache') as td:
        a, b = pathlib.Path(td) / 'in.wav', pathlib.Path(td) / 'out.wav'
        sf.write(a, y, sr, subtype='FLOAT')
        subprocess.run([args.ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(a), '-filter:a', f'atempo={factor:.4f}',
                        '-ar', str(sr), '-ac', '1', '-c:a', 'pcm_f32le', str(b)], check=True)
        z, _ = sf.read(b, dtype='float32')
    return z

rows = list(csv.DictReader(open(out / 'qa_report.csv', encoding='utf-8-sig')))
dest_root = out / 'sincronizadas'; dest_root.mkdir(exist_ok=True); (out / 'cache').mkdir(exist_ok=True)
report = []
for r in rows:
    if not r.get('file'): continue
    y, sr = sf.read(out / r['file'], dtype='float32')
    target = float(r['target_seconds']); gen = len(y) / sr; steps = []
    if gen > target * TOL:
        y, removed = compress_pauses(y, sr, gen - target * AIM)
        if removed > 0.01: steps.append(f'pausas -{removed:.2f}s')
        cur = len(y) / sr
        if cur > target * TOL:
            f = min(MAX_TEMPO, cur / (target * AIM))
            if f > 1.01: y = atempo(y, sr, f); steps.append(f'tempo x{f:.2f}')
    fit = len(y) / sr; ratio = fit / target
    status = 'ok' if ratio <= TOL else 'encurtar_texto'
    peak = float(np.max(np.abs(y)));  y = y * min(1.0, 0.94 / max(peak, 1e-8))
    dest = dest_root / pathlib.Path(r['file']).relative_to('selecionadas'); dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest, y, sr, subtype='PCM_16')
    report.append(dict(id=r['id'], character=r['character'], fit_status=status, qa_status=r['status'], steps='; '.join(steps) or 'nenhum',
                       target_seconds=round(target, 2), generated_seconds=round(gen, 2), fitted_seconds=round(fit, 2),
                       ratio_before=round(gen / target, 2), ratio_after=round(ratio, 2), text_pt=r.get('text_tts') or r['text_pt'], adaptado=r.get('adaptado', 'False'), file=str(dest.relative_to(out))))
with open(out / 'sincronizacao.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(report[0])); w.writeheader(); w.writerows(report)
summary = {}
for x in report: summary.setdefault(x['character'], {}).setdefault(x['fit_status'], 0); summary[x['character']][x['fit_status']] += 1
(out / 'sincronizacao_resumo.json').write_text(json.dumps(dict(tolerance=TOL, max_tempo=MAX_TEMPO, summary=summary), ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False))
for x in report: print(f"{x['character']:8s} {x['fit_status']:14s} {x['ratio_before']:.2f}->{x['ratio_after']:.2f} [{x['steps']}] {x['text_pt']}")
