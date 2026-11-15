"""Prepare the AwcPatcher plan: resample fitted takes to the stream rate (PCM16 mono) and write plan.tsv.

usage: dub_awc_plan.py <lote dir> <pacote dir> [--character MICHAEL] [--ids id1,id2] [--source sincronizacao|manifest]
  sincronizacao: uses sincronizadas/ files from dub_fit.py (default)
  manifest:      uses the raw generated take (for quick pipeline tests)
Writes <pacote>/wav32/*.wav, <pacote>/plan.tsv, <pacote>/plan_lines.csv  (D:\\JARVIS\\.venv-tts python)
"""
import csv, json, argparse, pathlib, re
import numpy as np, soundfile as sf, librosa
ap = argparse.ArgumentParser()
ap.add_argument('lote', type=pathlib.Path); ap.add_argument('pacote', type=pathlib.Path)
ap.add_argument('--character', default='MICHAEL'); ap.add_argument('--ids', default='')
ap.add_argument('--source', choices=['sincronizacao', 'manifest'], default='sincronizacao')
ap.add_argument('--max-wer', type=float, default=9.0); ap.add_argument('--max-ratio', type=float, default=99.0)
a = ap.parse_args()
lote, pac = a.lote.resolve(), a.pacote.resolve()
(pac / 'wav32').mkdir(parents=True, exist_ok=True)
manifest = json.loads((lote / 'manifest.json').read_text(encoding='utf-8'))
queue = {r['id']: r for r in csv.DictReader(open(lote.parents[1] / 'GTA_V_Dialogue_Map_v0.1' / 'fila_dublagem.csv', encoding='utf-8-sig'))}
ids = set(filter(None, a.ids.split(',')))
items = []
if a.source == 'sincronizacao':
    wer = {r['id']: float(r['wer'] or 9) for r in csv.DictReader(open(lote / 'qa_report.csv', encoding='utf-8-sig')) if r.get('wer')}
    skipped = []
    for r in csv.DictReader(open(lote / 'sincronizacao.csv', encoding='utf-8-sig')):
        if r['character'] == a.character and (not ids or r['id'] in ids):
            if wer.get(r['id'], 9) > a.max_wer or float(r['ratio_after']) > a.max_ratio:
                skipped.append(dict(id=r['id'], wer=wer.get(r['id']), ratio_after=r['ratio_after'], text_pt=r['text_pt'])); continue
            items.append((r['id'], lote / r['file'], r['fit_status'], r['qa_status']))
else:
    for lid, e in manifest.items():
        if e['character'] != a.character or (ids and lid not in ids): continue
        t = next((t for t in e['takes'].values() if t.get('status') == 'generated'), None)
        if t: items.append((lid, lote / t['file'], 'raw', 'raw'))
rows, lines = [], []
for lid, src, fit, qa in items:
    q = queue[lid]
    rate = int(q['sample_rate'])  # 32000, some streams 32001/32002; AwcPatcher refuses any mismatch
    y, sr = sf.read(src, dtype='float32')
    if y.ndim > 1: y = y.mean(axis=1)
    if sr != rate: y = librosa.resample(y, orig_sr=sr, target_sr=rate, res_type='soxr_hq')
    y = np.clip(y, -1, 1)
    out = pac / 'wav32' / (re.sub(r'[^A-Za-z0-9_]+', '_', lid) + '.wav')
    sf.write(out, y, rate, subtype='PCM_16')
    rows.append((q['awc_path'], q['stream_hash'], str(out)))
    lines.append(dict(id=lid, awc_path=q['awc_path'], stream_hash=q['stream_hash'], seconds=round(len(y) / rate, 3),
                      original_seconds=q['target_duration_seconds'], fit_status=fit, qa_status=qa, text_pt=manifest[lid]['text_tts']))
with open(pac / 'plan.tsv', 'w', encoding='utf-8', newline='') as f:
    f.write('awc_path\tstream_hash\twav\n')
    for r in sorted(rows): f.write('\t'.join(r) + '\n')
with open(pac / 'plan_lines.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(lines[0])); w.writeheader(); w.writerows(lines)
if a.source == 'sincronizacao' and skipped:
    with open(pac / 'fora_do_teste.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(skipped[0])); w.writeheader(); w.writerows(skipped)
    print('skipped', len(skipped))
print('plan', len(rows), 'lines,', len({r[0] for r in rows}), 'awc,', len({r[0].split('.rpf')[0] for r in rows}), 'rpf')
