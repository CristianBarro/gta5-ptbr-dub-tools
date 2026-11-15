"""List the subtitles of a cutscene with their time in the cutscene audio stream.

usage: cut_subtitles.py <cut.xml> <out.csv>
The .cut XML (from CutScanner) has a cutfAudioObject (cName = audio stream, fOffset = audio start) and
cutfObjectIdEvents on the cutfSubtitleObject whose args (cutfSubtitleEventArgs) carry the GXT2 hash of the line
and its on-screen duration. audio_t = event fTime + fOffset (seconds into the *_mastered_only.awc channels).
Text comes from work/gta-dialogue/text-data/localized_text.csv (en-US and pt-BR, update.rpf wins when present).
"""
import csv, sys, pathlib, re
import xml.etree.ElementTree as ET
HERE = pathlib.Path(__file__).resolve().parent
xml, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
root = ET.parse(xml).getroot()
val = lambda e, tag: e.find(tag).get('value')
audio = next(i for i in root.find('pCutsceneObjects') if i.get('type') == 'rage__cutfAudioObject')
audio_name, offset = audio.find('cName').text, float(val(audio, 'fOffset'))
sub_ids = {int(val(i, 'iObjectId')) for i in root.find('pCutsceneObjects') if i.get('type') == 'rage__cutfSubtitleObject'}
args = list(root.find('pCutsceneEventArgsList'))
subs = []
for ev in root.find('pCutsceneEventList'):
    if ev.find('iObjectId') is None or int(val(ev, 'iObjectId')) not in sub_ids: continue
    a = args[int(val(ev, 'iEventArgsIndex'))]
    if a.get('type') != 'rage__cutfSubtitleEventArgs': continue
    h = a.find('cName').text.replace('hash_', '').upper()
    subs.append(dict(cut_t=float(val(ev, 'fTime')), hash=h, dur=float(val(a, 'fSubtitleDuration'))))
need = {s['hash'] for s in subs}
text = {}
for r in csv.DictReader(open(HERE / 'text-data' / 'localized_text.csv', encoding='utf-8-sig')):
    if r['hash'] in need and r['language'] in ('en-US', 'pt-BR'):
        k = (r['language'], r['hash'])
        if k not in text or 'update' in r['path'].lower(): text[k] = (r['table'], r['text'])
clean = lambda s: re.sub(r'~[a-zA-Z]~', '', s or '').strip()
with open(out, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['n', 'audio_name', 'cut_t', 'audio_t', 'dur', 'hash', 'table', 'english', 'portuguese'])
    for n, s in enumerate(sorted(subs, key=lambda s: s['cut_t'])):
        en, pt = text.get(('en-US', s['hash']), ('', '')), text.get(('pt-BR', s['hash']), ('', ''))
        w.writerow([n, audio_name, round(s['cut_t'], 3), round(s['cut_t'] + offset, 3), s['dur'], s['hash'], en[0] or pt[0], clean(en[1]), clean(pt[1])])
print(audio_name, 'offset', offset, 'subtitles', len(subs), 'missing text', sum(1 for s in subs if ('en-US', s['hash']) not in text))
