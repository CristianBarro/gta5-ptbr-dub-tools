"""Inventory every cutscene: audio stream, offset, subtitles (EN/PT), known speakers, AWC location.

usage: cut_inventory.py <out dir>
Reads work/gta-dialogue/cutscenes/*.cut.xml, text-data/localized_text.csv, the dialogue map (dialogos.csv: subtitle
hash -> character when the conversation metadata resolved it) and audio_streams.csv (AWC per audio name).
Writes <out>/cutscenes.csv (one row per cut file) and <out>/legendas_todas.csv (one row per subtitle event).
"""
import csv, sys, re, pathlib, collections
import xml.etree.ElementTree as ET
HERE = pathlib.Path(__file__).resolve().parent
MAP = HERE.parents[1] / 'outputs' / 'GTA_V_Dialogue_Map_v0.1'
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
val = lambda e, tag: e.find(tag).get('value')
def ctrl_ref(m):
    v = m.group(1); c = int(v[1:], 16) if v[0] in 'xX' else int(v)
    return '' if c < 32 and c not in (9, 10, 13) else m.group(0)
def parse(x):  # some exported names hold control-char references (&#0;) that XML 1.0 forbids
    return ET.fromstring(re.sub(r'&#([xX][0-9A-Fa-f]+|\d+);', ctrl_ref, x.read_text(encoding='utf-8')))
def joaat(s):
    h = 0
    for c in s.lower().encode():
        h = (h + c) & 0xFFFFFFFF; h = (h + (h << 10)) & 0xFFFFFFFF; h ^= h >> 6
    h = (h + (h << 3)) & 0xFFFFFFFF; h ^= h >> 11; return (h + (h << 15)) & 0xFFFFFFFF
# ped model (StreamingName) of each protagonist: who is physically in the scene
PLAYER = {'MICHAEL': '%08X' % joaat('player_zero'), 'FRANKLIN': '%08X' % joaat('player_one'), 'TREVOR': '%08X' % joaat('player_two')}
cuts, events = [], []
for x in sorted((HERE / 'cutscenes').glob('*.cut.xml')):
    root = parse(x)
    objs = root.find('pCutsceneObjects')
    audio = [i for i in objs if i.get('type') == 'rage__cutfAudioObject']
    sub_ids = {int(val(i, 'iObjectId')) for i in objs if i.get('type') == 'rage__cutfSubtitleObject'}
    args = list(root.find('pCutsceneEventArgsList'))
    name = x.name.split('.rpf_')[-1].replace('.cut.xml', '')
    subs = []
    for ev in root.find('pCutsceneEventList'):
        if ev.find('iObjectId') is None or int(val(ev, 'iObjectId')) not in sub_ids: continue
        a = args[int(val(ev, 'iEventArgsIndex'))]
        if a.get('type') != 'rage__cutfSubtitleEventArgs': continue
        subs.append((float(val(ev, 'fTime')), a.find('cName').text.replace('hash_', '').upper(), float(val(a, 'fSubtitleDuration'))))
    an = audio[0].find('cName').text if audio else ''
    off = float(val(audio[0], 'fOffset')) if audio else 0.0
    peds = [i for i in objs if i.get('type') == 'rage__cutfPedModelObject']
    models = {(i.find('StreamingName').text or '').replace('hash_', '').upper() for i in peds if i.find('StreamingName') is not None}
    present = [w for w, h in PLAYER.items() if h in models]
    cuts.append(dict(cut=name, xml=x.name, audio_name=an, audio_offset=off, n_audio_objects=len(audio),
                     duration=float(val(root, 'fTotalDuration')), subtitles=len(subs),
                     protagonists=','.join(present), other_peds=len(peds) - len(present)))
    for t, h, d in sorted(subs):
        events.append(dict(cut=name, audio_name=an, cut_t=round(t, 3), audio_t=round(t + off, 3), dur=d, hash=h))
need = {e['hash'] for e in events}
text = {}
for r in csv.DictReader(open(HERE / 'text-data' / 'localized_text.csv', encoding='utf-8-sig')):
    if r['hash'] in need and r['language'] in ('en-US', 'pt-BR'):
        k = (r['language'], r['hash'])
        if k not in text or 'update' in r['path'].lower(): text[k] = (r['table'], r['text'])
known = {}
for r in csv.DictReader(open(MAP / 'dialogos.csv', encoding='utf-8-sig')):
    if r['subtitle_hash'] in need and r['character']: known[r['subtitle_hash']] = (r['character'], r['label'])
awc = collections.defaultdict(list)
for r in csv.DictReader(open(MAP / 'audio_streams.csv', encoding='utf-8-sig')):
    p = r['awc_path'].lower()
    if 'mastered_only' in p and r['stream_hash_29bit'] != '00000000': awc[p.split('\\')[-1].replace('_mastered_only.awc', '')].append(r)
clean = lambda s: re.sub(r'~[a-zA-Z]~', '', s or '').strip()
for e in events:
    en, pt = text.get(('en-US', e['hash']), ('', '')), text.get(('pt-BR', e['hash']), ('', ''))
    e.update(table=en[0] or pt[0], english=clean(en[1]), portuguese=clean(pt[1]),
             known_speaker=known.get(e['hash'], ('', ''))[0], label=known.get(e['hash'], ('', ''))[1])
for c in cuts:
    key = c['audio_name'].lower().replace('.wav', '')
    rows = awc.get(key, [])
    c['awc_path'] = rows[0]['awc_path'] if rows else ''
    c['channels'] = ','.join(r['stream_hash_29bit'] for r in rows)
    c['awc_seconds'] = rows[0]['duration_seconds'] if rows else ''
    ev = [e for e in events if e['cut'] == c['cut']]
    c['known_protag'] = sum(1 for e in ev if e['known_speaker'] in ('MICHAEL', 'FRANKLIN', 'TREVOR'))
    c['known_other'] = sum(1 for e in ev if e['known_speaker'] and e['known_speaker'] not in ('MICHAEL', 'FRANKLIN', 'TREVOR'))
    c['missing_text'] = sum(1 for e in ev if not e['english'])
for name, rows in (('cutscenes.csv', cuts), ('legendas_todas.csv', events)):
    with open(out / name, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('cuts', len(cuts), 'with subtitles', sum(1 for c in cuts if c['subtitles']), 'subtitle events', len(events))
print('cuts with audio found', sum(1 for c in cuts if c['awc_path']), 'audio names missing', sorted({c['audio_name'] for c in cuts if c['subtitles'] and not c['awc_path']})[:20])
print('events known protag', sum(1 for e in events if e['known_speaker'] in ('MICHAEL', 'FRANKLIN', 'TREVOR')),
      'known other', sum(1 for e in events if e['known_speaker'] and e['known_speaker'] not in ('MICHAEL', 'FRANKLIN', 'TREVOR')),
      'unknown', sum(1 for e in events if not e['known_speaker']), 'missing text', sum(1 for e in events if not e['english']))
print('audio streams shared by >1 cut', sum(1 for k, v in collections.Counter(c['audio_name'] for c in cuts if c['audio_name']).items() if v > 1))
