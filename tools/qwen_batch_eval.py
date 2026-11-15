"""Automatic gate for batched Trevor (Qwen3-TTS) generation.   (plain python: json/csv only)

  qwen_batch_eval.py queue <lote dir> <test dir>
      picks 24 Trevor lines of <lote>/fila_gen.csv (spread over lengths, no digits) -> <test>/fila_teste.csv
  qwen_batch_eval.py eval <test dir> <batch sizes...>
      compares <test>/single (one line per call) with <test>/batch<N> (dub_batch.py --batch-size N), each already
      transcribed by cut_pick.py (escolha.json). Writes <test>/qwen_batch_verdict.json:
      pass = no failed takes, mean WER not worse than one-by-one by more than 0.05, median duration ratio
      (batch / single, same line) within 0.85..1.15, at most 3 lines regenerated one by one, and >= 1.5x faster.
      The fastest passing batch size is chosen; run_cutscenes.ps1 uses it for Trevor, otherwise batch 1.
This is an automatic check, not an approval: the takes stay in <test> for listening.
"""
import sys, csv, json, re, random, pathlib, statistics as st
mode = sys.argv[1]
if mode == 'queue':
    lote, test = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]); test.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(lote / 'fila_gen.csv', encoding='utf-8-sig'))
            if r['character'] == 'TREVOR' and not re.search(r'\d', r['text_pt']) and r['text_pt'].strip()]
    rows.sort(key=lambda r: len(r['text_pt']))
    rng = random.Random(20261115); k = 24; n = len(rows)
    pick = [rows[min(n - 1, int((j + rng.random()) * n / k))] for j in range(k)]
    with open(test / 'fila_teste.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(pick)
    print('test queue', len(pick), 'Trevor lines, text length', len(pick[0]['text_pt']), '..', len(pick[-1]['text_pt']))
    sys.exit()
test = pathlib.Path(sys.argv[2]); sizes = [int(x) for x in sys.argv[3:]]
def load(d):
    man = json.loads((d / 'manifest.json').read_text(encoding='utf-8'))
    pick = json.loads((d / 'escolha.json').read_text(encoding='utf-8')) if (d / 'escolha.json').exists() else {}
    takes = {lid: e['takes'].get('take1', {}) for lid, e in man.items()}
    return takes, pick
s_takes, s_pick = load(test / 'single')
s_time = sum(t.get('generation_seconds', 0) for t in s_takes.values())
s_wer = st.mean(v['wer'] for v in s_pick.values()) if s_pick else None
res = dict(single=dict(seconds=round(s_time, 1), mean_wer=s_wer, lines=len(s_takes)), runs={}, pass_=False, batch_size=1)
best = None
for bs in sizes:
    d = test / f'batch{bs}'
    if not (d / 'manifest.json').exists(): res['runs'][bs] = dict(error='no manifest'); continue
    b_takes, b_pick = load(d)
    failed = sum(1 for t in b_takes.values() if t.get('status') != 'generated')
    one_by_one = sum(1 for t in b_takes.values() if t.get('status') == 'generated' and not t.get('batch'))
    b_time = sum(t.get('generation_seconds', 0) for t in b_takes.values())
    b_wer = st.mean(v['wer'] for v in b_pick.values()) if b_pick else None
    ratios = [b_takes[l]['seconds'] / s_takes[l]['seconds'] for l in b_takes
              if b_takes[l].get('seconds') and s_takes.get(l, {}).get('seconds')]
    ratio = st.median(ratios) if ratios else None
    speed = s_time / max(1e-6, b_time)
    ok = (failed == 0 and b_wer is not None and s_wer is not None and b_wer <= s_wer + 0.05
          and ratio is not None and 0.85 <= ratio <= 1.15 and one_by_one <= 3 and speed >= 1.5)
    res['runs'][bs] = dict(seconds=round(b_time, 1), speedup=round(speed, 2), mean_wer=b_wer, median_duration_ratio=ratio,
                           failed=failed, regenerated_one_by_one=one_by_one, pass_=ok)
    if ok and (best is None or speed > res['runs'][best]['speedup']): best = bs
if best:
    res['pass_'] = True; res['batch_size'] = best
out = dict(res, **{'pass': res.pop('pass_')})
for r in out['runs'].values():
    if 'pass_' in r: r['pass'] = r.pop('pass_')
out['note'] = 'automatic check only, not an approval; takes kept in this folder for listening'
(test / 'qwen_batch_verdict.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False))
