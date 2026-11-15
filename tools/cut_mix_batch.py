"""Remix every cutscene AWC that has dubbed lines; write the AwcPatcher plan.   (D:\\JARVIS\\.venv-tts)

usage: cut_mix_batch.py <lote dir> <work root> [--only-awc stem,stem]
  <lote>/cutscenes.csv, <lote>/falantes_todos.csv, <lote>/fila_cutscenes.csv, <lote>/gen/escolha.json
  <work>/orig/<stem>/ch_<H>.wav (exact decode), <work>/voice/<stem>/ch_<H>_voice.wav (Demucs)
writes <work>/mix/<stem>/ch_<H>.wav, <lote>/plan_cutscenes.tsv, <lote>/mix_report.csv

Same method as the approved-for-testing pilot (cut_mix.py), per AWC:
  * English speech region of each dubbed line: energy of the loudest channel's voice stem, from 0.12 s before the
    subtitle to the last active frame before the next subtitle (any speaker)
  * that region is faded out of the ORIGINAL voice stem (one keep-mask), the PT take goes to a separate track
  * PT take level = English line's RMS in that channel, capped at its peak (keeps the original panning)
  * take longer than the free slot -> sped up to at most 1.3x; still longer -> runs over (reported)
  * channel = (mix - voice) + voice*keep + pt ; soft limit only above 0.98 (no global gain change)
"""
import sys, csv, json, pathlib, argparse, collections, numpy as np, soundfile as sf, librosa
ap = argparse.ArgumentParser(); ap.add_argument('lote'); ap.add_argument('work'); ap.add_argument('--only-awc', default=''); a = ap.parse_args()
lote, work = pathlib.Path(a.lote), pathlib.Path(a.work)
SR = 48000
cuts = {r['cut']: r for r in csv.DictReader(open(lote / 'cutscenes.csv', encoding='utf-8-sig'))}
subs = list(csv.DictReader(open(lote / 'falantes_todos.csv', encoding='utf-8-sig')))
queue = {r['id']: r for r in csv.DictReader(open(lote / 'fila_cutscenes.csv', encoding='utf-8-sig'))}
pick = json.loads((lote / 'gen' / 'escolha.json').read_text(encoding='utf-8'))
only = set(filter(None, a.only_awc.split(',')))
starts_by = collections.defaultdict(set)
for s in subs:
    p = cuts[s['cut']]['awc_path']
    if p: starts_by[p].add(float(s['audio_t']))
lines_by = collections.defaultdict(list)
for lid, q in queue.items():
    q['gen_id'] = q.get('gen_id') or lid          # identical lines share one generated take
    if q['gen_id'] in pick and pick[q['gen_id']].get('file'): lines_by[q['awc_path']].append(q)
plan, report = [], []
for k, (awc, lines) in enumerate(sorted(lines_by.items()), 1):
    stem = pathlib.Path(awc.replace('\\', '/')).stem
    if only and stem not in only: continue
    od, vd, md = work / 'orig' / stem, work / 'voice' / stem, work / 'mix' / stem
    hs = [f.stem[3:] for f in sorted(od.glob('ch_*.wav'))]
    orig = {h: sf.read(od / f'ch_{h}.wav', dtype='float32')[0] for h in hs}
    voice = {h: sf.read(vd / f'ch_{h}_voice.wav', dtype='float32')[0] for h in hs}
    N = len(orig[hs[0]])
    main = max(hs, key=lambda h: float((voice[h] ** 2).mean()))
    hop = 480
    env = np.sqrt(np.convolve(voice[main] ** 2, np.ones(hop) / hop, 'same')[::hop] + 1e-12)
    thr = max(10 ** (-40 / 20), np.percentile(env, 90) * 10 ** (-24 / 20))
    starts = sorted(starts_by[awc])
    keep = np.ones(N, np.float32); pt_track = {h: np.zeros(N, np.float32) for h in hs}
    for q in sorted(lines, key=lambda q: float(q['audio_t'])):
        t0 = float(q['audio_t']); nxt = next((s for s in starts if s > t0 + 0.05), N / SR)
        lo, hi = int(max(0, t0 - 0.12) * 100), int(min(nxt, N / SR) * 100)
        act = np.where(env[lo:hi] > thr)[0]
        if len(act) == 0: on, off = t0, min(nxt, t0 + float(q['target_duration_seconds']))
        else: on, off = (lo + act[0]) / 100, (lo + act[-1] + 1) / 100 + 0.08
        if len(act) and act[0] == 0:
            # speech is already running 0.12 s before the subtitle (alignment sample: some cuts lead by 0.3-0.8 s).
            # Walk back to the silence that starts this line, at most 0.8 s and never into the previous subtitle's
            # first 0.25 s; if there is no silence there (previous speaker runs straight in), keep t0 - 0.12.
            prev = max([s for s in starts if s < t0 - 0.05] or [0.0])
            floor = int(max(0, t0 - 0.8, prev + 0.25) * 100); f = lo
            while f > floor and env[f - 1] > thr: f -= 1
            if f > floor: on = f / 100
        off = min(max(off, on + 0.2), nxt + 0.15, N / SR)
        pk = pick[q['gen_id']]
        pt, sr = sf.read(lote / 'gen' / pk['file'], dtype='float32')
        pt = librosa.resample(pt, orig_sr=sr, target_sr=SR)
        pt, _ = librosa.effects.trim(pt, top_db=40, frame_length=1024, hop_length=128)
        slot = max(nxt - on - 0.04, off - on, 0.2)
        rate = 1.0
        if len(pt) / SR > slot: rate = min(1.3, (len(pt) / SR) / slot); pt = librosa.effects.time_stretch(pt, rate=rate)
        i0, i1 = int(on * SR), int(off * SR); fade = int(0.02 * SR)
        keep[i0:i1] = 0
        keep[max(0, i0 - fade):i0] = np.minimum(keep[max(0, i0 - fade):i0], np.linspace(1, 0, i0 - max(0, i0 - fade)))
        keep[i1:i1 + fade] = np.minimum(keep[i1:i1 + fade], np.linspace(0, 1, len(keep[i1:i1 + fade])))
        j1 = min(N, i0 + len(pt)); seg = pt[: j1 - i0].copy()
        if len(seg) < 64: continue
        seg[:64] *= np.linspace(0, 1, 64); seg[-256:] *= np.linspace(1, 0, len(seg[-256:]))
        prms = float(np.sqrt((seg ** 2).mean()) + 1e-9); ppk = float(np.abs(seg).max() + 1e-9)
        for h in hs:
            e = voice[h][i0:i1]
            g = min(float(np.sqrt((e ** 2).mean()) + 1e-9) / prms, float(np.abs(e).max() + 1e-9) / ppk) if len(e) else 0.0
            pt_track[h][i0:j1] += seg * g
        report.append(dict(awc=stem, id=q['id'], who=q['character'], sub_t=t0, en_on=round(on, 3), en_off=round(off, 3),
                           next_sub=round(nxt, 3), pt_seconds=round(len(pt) / SR, 3), speedup=round(rate, 3),
                           overrun=round(max(0, on + len(pt) / SR - nxt), 3), wer=pk['wer'], take=pk['take'], gen_id=q['gen_id']))
    md.mkdir(parents=True, exist_ok=True); limited = 0
    for h in hs:
        y = (orig[h] - voice[h]) + voice[h] * keep + pt_track[h]
        over = np.abs(y) > 0.98; limited += int(over.sum())
        y[over] = np.sign(y[over]) * (0.98 + 0.019 * np.tanh((np.abs(y[over]) - 0.98) / 0.019))
        sf.write(md / f'ch_{h}.wav', y, SR, subtype='PCM_16')
        plan.append((awc, h, str(md / f'ch_{h}.wav')))
    print(f'[{k}/{len(lines_by)}] {stem} lines {len(lines)} channels {len(hs)} limited {limited}', flush=True)
with open(lote / 'plan_cutscenes.tsv', 'w', encoding='utf-8', newline='') as f:
    f.write('awc_path\tstream_hash\twav\n')
    for r in plan: f.write('\t'.join(r) + '\n')
with open(lote / 'mix_report.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(report[0])); w.writeheader(); w.writerows(report)
print('AWCs', len({r[0] for r in plan}), 'channels', len(plan), 'lines', len(report),
      '| sped up', sum(1 for r in report if r['speedup'] > 1), '| overrun > 0.3 s', sum(1 for r in report if r['overrun'] > 0.3))
