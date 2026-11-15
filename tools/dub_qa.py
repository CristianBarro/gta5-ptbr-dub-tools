"""Automatic QA for dub_batch.py output. Three passes (separate environments):

  asr     D:\\JARVIS\\.venv      faster-whisper small, Portuguese: transcript, word error rate
  speaker D:\\JARVIS\\.venv-tts  VoiceEncoder cosine vs the character's references
  report  any python            picks the best take per line, flags lines for review,
                                writes qa_report.csv, copies selected takes and builds one
                                listening file per character

The checks only screen. Human listening is still the approval criterion.
usage: dub_qa.py {asr|speaker|report} <pilot dir>
"""
import os, sys, re, csv, json, math, shutil, pathlib, unicodedata
mode, out = sys.argv[1], pathlib.Path(sys.argv[2]).resolve()
VOZES = pathlib.Path(r'C:\gta-dub\outputs\GTA_V_Dublagem_PTBR')
for key in ['TEMP', 'TMP', 'TMPDIR', 'NUMBA_CACHE_DIR']:
    p = out / 'cache' / ('numba' if key == 'NUMBA_CACHE_DIR' else 'tmp'); p.mkdir(parents=True, exist_ok=True); os.environ[key] = str(p)
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
takes = [(lid, k, t) for lid, e in manifest.items() for k, t in e['takes'].items() if t.get('status') == 'generated']

def words(text):
    v = ''.join(c for c in unicodedata.normalize('NFD', text.lower()) if unicodedata.category(c) != 'Mn')
    v = re.sub(r'\bvoce\b', 'ce', v); v = re.sub(r'\bpara\b', 'pra', v); v = re.sub(r'\besta\b', 'ta', v)
    return re.findall(r'\w+', v)
def distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1): cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

if mode == 'asr':
    from faster_whisper import WhisperModel
    snap = next(pathlib.Path(r'C:\Users\<user>\.cache\huggingface\hub\models--Systran--faster-whisper-small\snapshots').iterdir())
    m = WhisperModel(str(snap), device='cpu', compute_type='int8', cpu_threads=6, local_files_only=True)
    path = out / 'qa_asr.json'; res = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    for lid, k, t in takes:
        if t['file'] in res: continue
        segs, _ = m.transcribe(str(out / t['file']), language='pt', beam_size=5, temperature=0)
        hyp = ' '.join(s.text.strip() for s in segs)
        ref = words(manifest[lid]['text_tts']); err = distance(ref, words(hyp))
        res[t['file']] = dict(asr=hyp, word_errors=err, wer=round(err / max(1, len(ref)), 3))
        if len(res) % 25 == 0: path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
        print(t['file'], res[t['file']]['wer'], hyp, flush=True)

    path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')

elif mode == 'speaker':
    import numpy as np, librosa, torch
    from chatterbox.models.voice_encoder import VoiceEncoder
    ckpt = pathlib.Path(r'D:\JARVIS\data\clone-models\hub\models--ResembleAI--chatterbox\snapshots\5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18')
    ve = VoiceEncoder(); ve.load_state_dict(torch.load(ckpt / 've.pt', map_location='cpu', weights_only=True)); ve.eval()
    def emb(p):
        y, _ = librosa.load(str(p), sr=16000); e = ve.embeds_from_wavs([y], sample_rate=16000, as_spk=True); return e / np.linalg.norm(e)
    def centroid(paths):
        c = np.mean([emb(p) for p in paths], axis=0); return c / np.linalg.norm(c)
    bank = VOZES / 'revisao_v4' / 'referencias_candidatas'
    refs = {'MICHAEL': centroid(sorted(bank.glob('michael_*.wav'))), 'FRANKLIN': centroid(sorted(bank.glob('franklin_*.wav'))),
            'TREVOR': centroid([VOZES / 'revisao_v2' / 'referencias_preparadas' / 'trevor_reference.wav'])}
    approved = {'MICHAEL': VOZES / 'revisao_v5' / 'amostras_ptbr' / 'michael_A_nativo_fala_real.wav',
                'FRANKLIN': VOZES / 'revisao_v4' / 'amostras_ptbr' / 'franklin_fala_real.wav',
                'TREVOR': VOZES / 'revisao_v3' / 'amostras_ptbr' / 'trevor.wav'}
    res = {'_approved_baseline': {w: round(float(emb(p) @ refs[w]), 4) for w, p in approved.items()}}
    for lid, k, t in takes:
        who = manifest[lid]['character']; e = emb(out / t['file'])
        res[t['file']] = dict(cos_own=round(float(e @ refs[who]), 4),
                              cos_others={w: round(float(e @ c), 4) for w, c in refs.items() if w != who})
    (out / 'qa_speaker.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
    print(json.dumps(res['_approved_baseline']))

elif mode == 'report':
    import wave, array
    asr = json.loads((out / 'qa_asr.json').read_text(encoding='utf-8'))
    spk = json.loads((out / 'qa_speaker.json').read_text(encoding='utf-8')); base = spk['_approved_baseline']
    sel_dir = out / 'selecionadas'; shutil.rmtree(sel_dir, ignore_errors=True); sel_dir.mkdir()
    rows, listen = [], {}
    for lid, e in manifest.items():
        cands = []
        for k, t in e['takes'].items():
            if t.get('status') != 'generated' or t['file'] not in asr: continue
            a, s = asr[t['file']], spk[t['file']]
            cands.append(dict(take=k, **t, **a, cos_own=s['cos_own']))
        if not cands:
            rows.append(dict(id=lid, character=e['character'], status='falhou', reasons='sem take válido')); continue
        # 1) intelligible, 2) fittable by dub_fit.py (<= 1.25 tempo x 1.10 tolerance, before pause cuts),
        # 3) fewer word errors, 4) closest duration, 5) timbre
        best = min(cands, key=lambda c: (c['wer'] > 0.15, c['duration_ratio'] > 1.375, c['wer'],
                                         abs(math.log(max(1e-3, c['duration_ratio']))), -c['cos_own']))
        reasons = []
        if best['wer'] > 0.15: reasons.append(f"texto (WER {best['wer']:.0%})")
        if not 0.7 <= best['duration_ratio'] <= 1.5: reasons.append(f"duração {best['duration_ratio']:.2f}x do original")
        if best['cos_own'] < base[e['character']] - 0.06: reasons.append(f"timbre {best['cos_own']:.2f}")
        status = 'ok' if not reasons else 'revisar'
        n = sum(1 for r in rows if r['character'] == e['character']) + 1
        dest = sel_dir / e['character'].lower() / f"{n:02d}_{re.sub(r'[^A-Za-z0-9_]+', '_', lid)}.wav"
        dest.parent.mkdir(exist_ok=True); shutil.copy2(out / best['file'], dest)
        listen.setdefault(e['character'], []).append(dest)
        rows.append(dict(id=lid, character=e['character'], status=status, reasons='; '.join(reasons), take=best['take'],
                         file=str(dest.relative_to(out)), text_pt=e['text_pt'], text_tts=e['text_tts'], adaptado=bool(e.get('text_adapted')), asr=best['asr'], wer=best['wer'],
                         seconds=best['seconds'], target_seconds=round(e['target_seconds'], 2), duration_ratio=best['duration_ratio'],
                         cos_own=best['cos_own'], english=e['english'], awc_path=e['awc_path'], stream_hash=e['stream_hash']))
    with open(out / 'qa_report.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list({k: 0 for r in rows for k in r})); w.writeheader(); w.writerows(rows)
    for who, files in listen.items():  # one file per character: lines in order, 0.7 s gaps (pilots only)
        if len(files) > 60: continue
        data = array.array('h')
        for p in files:
            with wave.open(str(p), 'rb') as h: data.frombytes(h.readframes(h.getnframes()))
            data.extend([0] * int(24000 * 0.7))
        with wave.open(str(out / f'ouvir_{who.lower()}.wav'), 'wb') as h:
            h.setnchannels(1); h.setsampwidth(2); h.setframerate(24000); h.writeframes(data.tobytes())
    summary = {}
    for r in rows: summary.setdefault(r['character'], {}).setdefault(r['status'], 0); summary[r['character']][r['status']] += 1
    (out / 'qa_summary.json').write_text(json.dumps(dict(summary=summary, approved_baseline_cos=base), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))
    for r in rows:
        print(f"{r['character']:8s} {r['status']:7s} {r.get('take','')} wer={r.get('wer')} dur={r.get('seconds')}/{r.get('target_seconds')} cos={r.get('cos_own')} | {r.get('text_pt','')} || {r.get('asr','')} {('<< ' + r['reasons']) if r['reasons'] else ''}")
