"""Automatic screening of v0.4 candidates. Two passes, one per environment:

  asr     (D:\\JARVIS\\.venv)      faster-whisper small: Portuguese transcript, word errors,
                                    and language-ID probabilities (pt vs en) as an accent proxy
  speaker (D:\\JARVIS\\.venv-tts)  Chatterbox VoiceEncoder cosine against the dedicated-bank
                                    centroid of each protagonist, plus pyin F0 statistics
  rank    (any python)              merges both and ranks candidates

All numbers are screening heuristics only. They do not replace listening approval.
usage: score_candidates_v4.py {asr|speaker|rank} <revisao_v4>
"""
import os, sys, json, re, pathlib, unicodedata
mode, out = sys.argv[1], pathlib.Path(sys.argv[2]).resolve()
cand = out / 'candidatos'
for key in ['TEMP', 'TMP', 'TMPDIR', 'NUMBA_CACHE_DIR']:
    p = out / 'cache' / ('numba' if key == 'NUMBA_CACHE_DIR' else 'tmp'); p.mkdir(parents=True, exist_ok=True); os.environ[key] = str(p)
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
rows = json.loads((cand / 'candidates.json').read_text(encoding='utf-8'))

def words(text):
    value = ''.join(c for c in unicodedata.normalize('NFD', text.lower()) if unicodedata.category(c) != 'Mn')
    value = value.replace('voce', 'ce').replace('ta ', 'esta ')  # colloquial forms count as equal
    return re.findall(r'\w+', value)

def distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1): cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

if mode == 'asr':
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    snap = next(pathlib.Path(r'C:\Users\<user>\.cache\huggingface\hub\models--Systran--faster-whisper-small\snapshots').iterdir())
    m = WhisperModel(str(snap), device='cpu', compute_type='int8', cpu_threads=6, local_files_only=True)
    res = {}
    for r in rows:
        path = str(cand / r['file'])
        segs, _ = m.transcribe(path, language='pt', beam_size=5, temperature=0)
        segs = list(segs); hyp = ' '.join(s.text.strip() for s in segs)
        logprob = sum(s.avg_logprob * max(1, len(s.tokens)) for s in segs) / max(1, sum(max(1, len(s.tokens)) for s in segs))
        _, _, probs = m.detect_language(decode_audio(path, sampling_rate=16000))
        probs = dict(probs)
        ref_w, hyp_w = words(r['text']), words(hyp)
        res[r['file']] = dict(asr=hyp, word_errors=distance(ref_w, hyp_w), ref_words=len(ref_w), pt_logprob=round(logprob, 4),
                              p_pt=round(probs.get('pt', 0.0), 4), p_en=round(probs.get('en', 0.0), 4))
        print(r['file'], json.dumps(res[r['file']], ensure_ascii=False), flush=True)
    (cand / 'scores_asr.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')

elif mode == 'speaker':
    import numpy as np, librosa, torch
    from chatterbox.models.voice_encoder import VoiceEncoder
    ckpt = pathlib.Path(r'D:\JARVIS\data\clone-models\hub\models--ResembleAI--chatterbox\snapshots\5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18')
    ve = VoiceEncoder(); ve.load_state_dict(torch.load(ckpt / 've.pt', map_location='cpu', weights_only=True)); ve.eval()
    def emb(path):
        y, _ = librosa.load(str(path), sr=16000)
        e = ve.embeds_from_wavs([y], sample_rate=16000, as_spk=True); return e / np.linalg.norm(e), y
    cent = {}
    for who in ['michael', 'franklin']:
        bank = out / 'referencias_candidatas'
        if not bank.exists(): bank = out.parent / 'revisao_v4' / 'referencias_candidatas'
        es = [emb(p)[0] for p in sorted(bank.glob(who + '_*.wav'))]
        c = np.mean(es, axis=0); cent[who.upper()] = c / np.linalg.norm(c)
    trevor = emb(out.parent / 'revisao_v3' / 'amostras_ptbr' / 'trevor.wav')[0]
    res = {}
    for r in rows:
        e, y = emb(cand / r['file'])
        f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=16000, frame_length=1024)
        f0v = f0[voiced & np.isfinite(f0)]
        other = 'FRANKLIN' if r['character'] == 'MICHAEL' else 'MICHAEL'  # protagonist timbre margin
        res[r['file']] = dict(cos_own=round(float(e @ cent[r['character']]), 4), cos_other=round(float(e @ cent[other]), 4),
                              cos_trevor_pt=round(float(e @ trevor), 4),
                              f0_median=round(float(np.median(f0v)), 1) if len(f0v) else None)
        print(r['file'], json.dumps(res[r['file']]), flush=True)
    (cand / 'scores_speaker.json').write_text(json.dumps(res, indent=2), encoding='utf-8')

elif mode == 'rank':
    asr = json.loads((cand / 'scores_asr.json').read_text(encoding='utf-8'))
    spk = json.loads((cand / 'scores_speaker.json').read_text(encoding='utf-8'))
    merged = []
    for r in rows:
        x = dict(r, **asr[r['file']], **spk[r['file']])
        # screening score: intelligibility gate, then timbre margin and Portuguese-ness
        x['score'] = round((x['cos_own'] - x['cos_other']) * 2 + x['cos_own'] + 0.3 * x['p_pt'] - 0.35 * x['word_errors'], 4)
        merged.append(x)
    merged.sort(key=lambda x: (x['character'], x['phrase'], -x['score']))
    (cand / 'ranking.json').write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding='utf-8')
    for x in merged:
        print(f"{x['file']:40s} err={x['word_errors']} p_pt={x['p_pt']:.2f} lp={x.get('pt_logprob', 0):.3f} own={x['cos_own']:.3f} other={x['cos_other']:.3f} f0={x['f0_median']} score={x['score']:.3f} | {x['asr']}")
