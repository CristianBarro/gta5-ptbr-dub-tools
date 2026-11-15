"""Build the dub_batch.py queue for every attributed protagonist line of every cutscene.

usage: cut_queue_batch.py <lote dir> [--only-awc a,b] [--retake escolha.json]
  reads <lote>/falantes_todos.csv (cut_speakers_batch.py), writes <lote>/fila_cutscenes.csv
  --retake: only the lines whose picked take is bad (WER > 0.3, or > 1.6x the free slot) -> <lote>/fila_retake.csv

id = <awc stem>:<subtitle hash>:<audio_t ms>  (unique per AWC position; the mix step groups by AWC)
Digits in the pt-BR text are spelled out (integers, money, percentages); lines that still contain digits are skipped
by dub_batch and stay in English.
Delivery (user feedback on the pilot: Michael sounded flat when angry): intensity = line loudness minus that
protagonist's median cutscene line loudness, +3 if the English line has '!'.
  >= 6 -> exaggeration 0.9 / cfg 0.3 ; >= 2.5 -> 0.7 / 0.35 ; otherwise the approved profile.
"""
import csv, json, re, sys, pathlib, argparse, statistics, collections
ap = argparse.ArgumentParser(); ap.add_argument('lote'); ap.add_argument('--only-awc', default=''); ap.add_argument('--retake'); a = ap.parse_args()
lote = pathlib.Path(a.lote)
UN = 'zero um dois três quatro cinco seis sete oito nove dez onze doze treze catorze quinze dezesseis dezessete dezoito dezenove'.split()
DEZ = ['', '', 'vinte', 'trinta', 'quarenta', 'cinquenta', 'sessenta', 'setenta', 'oitenta', 'noventa']
CEN = ['', 'cento', 'duzentos', 'trezentos', 'quatrocentos', 'quinhentos', 'seiscentos', 'setecentos', 'oitocentos', 'novecentos']
def ext(n):
    if n < 20: return UN[n]
    if n < 100: return DEZ[n // 10] + ('' if n % 10 == 0 else ' e ' + UN[n % 10])
    if n == 100: return 'cem'
    if n < 1000: return CEN[n // 100] + ('' if n % 100 == 0 else ' e ' + ext(n % 100))
    if n < 1_000_000:
        k, r = divmod(n, 1000); head = 'mil' if k == 1 else ext(k) + ' mil'
        return head + ('' if r == 0 else (' e ' if r < 100 or r % 100 == 0 else ' ') + ext(r))
    if n < 1_000_000_000:
        m, r = divmod(n, 1_000_000); head = 'um milhão' if m == 1 else ext(m) + ' milhões'
        return head + ('' if r == 0 else (' e ' if r < 100 else ' ') + ext(r))
    return str(n)
def spell(t):
    t = re.sub(r'(\d+)\s*%', lambda m: ext(int(m.group(1))) + ' por cento', t)
    t = re.sub(r'(?:US)?\$\s*(\d[\d.,]*)\s*(mil|milhões|milhão|bilhões|bilhão)?',
               lambda m: (ext(int(re.sub(r'[.,]', '', m.group(1)))) + (' ' + m.group(2) if m.group(2) else '') + ' dólares'), t)
    t = re.sub(r'\b(\d{1,3}(?:\.\d{3})+|\d+)\b', lambda m: ext(int(m.group(1).replace('.', ''))), t)
    return t
rows = list(csv.DictReader(open(lote / 'falantes_todos.csv', encoding='utf-8-sig')))
cuts = {r['cut']: r for r in csv.DictReader(open(lote / 'cutscenes.csv', encoding='utf-8-sig'))}
only = set(filter(None, a.only_awc.split(',')))
P3 = ('MICHAEL', 'FRANKLIN', 'TREVOR')
lines = [r for r in rows if r['speaker'] in P3 and r['portuguese'] and cuts[r['cut']]['awc_path']]
med = {w: statistics.median([float(r['level_db']) for r in lines if r['speaker'] == w and float(r['level_db']) > -90] or [-20]) for w in P3}
print('median line loudness dB', {w: round(v, 1) for w, v in med.items()})
seen, out = set(), []
for r in lines:
    awc = cuts[r['cut']]['awc_path']; stem = pathlib.Path(awc.replace('\\', '/')).stem
    if only and stem not in only: continue
    lid = f"{stem}:{r['hash']}:{int(round(float(r['audio_t']) * 1000))}"
    if lid in seen: continue          # same audio reused by two .cut files
    seen.add(lid)
    lv = float(r['level_db']); score = round(lv - med[r['speaker']] + (3.0 if '!' in r['english'] else 0.0), 2) if lv > -90 else ''
    ex = cfg = ''
    if score != '' and score >= 6: ex, cfg = '0.9', '0.3'
    elif score != '' and score >= 2.5: ex, cfg = '0.7', '0.35'
    out.append(dict(id=lid, character=r['speaker'], text_pt=spell(r['portuguese']), english=r['english'], awc_path=awc,
                    stream_hash='', sample_rate='48000', samples='', target_duration_seconds=r['dur'], status='cutscene',
                    audio_t=r['audio_t'], hash=r['hash'], cut=r['cut'], intensity=score, exaggeration=ex, cfg_weight=cfg,
                    speaker_source=r['speaker_source']))
# identical line (same character, text and delivery) is generated once: gen_id = id of the first occurrence.
# fila_cutscenes.csv keeps every line (the mix places each one); fila_gen.csv holds only the lines to generate.
first = {}
for r in out:
    r['gen_id'] = first.setdefault((r['character'], r['text_pt'].strip().lower(), r['exaggeration'], r['cfg_weight']), r['id'])
def dump(name, rows):
    with open(lote / name, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(rows)
gen = [r for r in out if r['gen_id'] == r['id']]
name = 'fila_gen.csv'
if a.retake:
    pick = json.loads(pathlib.Path(a.retake).read_text(encoding='utf-8'))
    bad = {k for k, v in pick.items() if v['wer'] > 0.3 or v.get('flagged')}
    gen = [r for r in gen if r['id'] in bad or r['id'] not in pick]; name = 'fila_retake.csv'
else:
    dump('fila_cutscenes.csv', out)
    # AWCs that will be remixed: only these need their remaining channels separated
    (lote / 'awc_com_falas.txt').write_text('\n'.join(sorted({pathlib.Path(r['awc_path'].replace('\\', '/')).stem for r in out})) + '\n', encoding='utf-8')
dump(name, gen)
print('all lines', len(out), '| to generate', len(gen), '| saved by reuse', len(out) - len([r for r in out if r['gen_id'] == r['id']]))
out = gen
c = collections.Counter(r['character'] for r in out)
print(name, len(out), 'lines', dict(c), '| intensified', sum(1 for r in out if r['exaggeration']),
      '| still with digits', sum(1 for r in out if re.search(r'\d', r['text_pt'])))
