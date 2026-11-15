"""Assemble the Michael v0.5 package (two variants) from screened candidates.
usage: finalize_michael_v5.py <revisao_v5>   (D:\JARVIS\.venv-tts python)"""
import sys, json, shutil, hashlib, pathlib
import numpy as np, soundfile as sf
out = pathlib.Path(sys.argv[1]).resolve(); v4 = out.parent / 'revisao_v4'; v3 = out.parent / 'revisao_v3'
APPROVED_TREVOR = '6254b53efbc07232ec0b6ff78962d99505b7c466fa6b4fdc1b8534864ce48dc0'
CHOSEN = {('A_nativo', 'comparacao'): 'michael_comparacao_daniel_full_s11.wav',
          ('A_nativo', 'fala_real'): 'michael_fala_real_daniel_full_s23.wav',
          ('B_misto', 'comparacao'): 'michael_comparacao_daniel_mix_s23.wav',
          ('B_misto', 'fala_real'): 'michael_fala_real_daniel_mix_s37.wav'}
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
rank = {r['file']: r for r in json.loads((out / 'candidatos' / 'ranking.json').read_text(encoding='utf-8'))}
dest = out / 'amostras_ptbr'; dest.mkdir(exist_ok=True)
shutil.copy2(v3 / 'amostras_ptbr' / 'trevor.wav', dest / 'trevor.wav'); assert sha(dest / 'trevor.wav') == APPROVED_TREVOR
report = []
for (variant, phrase), name in CHOSEN.items():
    t = dest / f'michael_{variant}_{phrase}.wav'; shutil.copy2(out / 'candidatos' / name, t); r = rank[name]
    report.append(dict(character='MICHAEL', variant=variant, phrase=phrase, text=r['text'], file=str(t), candidate=name,
                       method=r['method'], t3_source=r.get('t3_source'), seed=r['seed'], seconds=r['seconds'],
                       asr_pt=r['asr'], word_errors=r['word_errors'], pt_logprob=r['pt_logprob'], cos_own_bank=r['cos_own'],
                       cos_franklin_bank=r['cos_other'], f0_median_pyin=r['f0_median'], sha256=sha(t), approved=False))
(out / 'preview_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
ab = out / 'comparacao_ab'; ab.mkdir(exist_ok=True)
gap = np.zeros(int(24000 * 0.8), np.float32)
seq = [v4 / 'amostras_ptbr' / 'michael_comparacao.wav', dest / 'michael_A_nativo_comparacao.wav', dest / 'michael_B_misto_comparacao.wav']
parts = []
for p in seq:
    a, sr = sf.read(p, dtype='float32'); assert sr == 24000; parts += [a, gap]
sf.write(ab / 'michael_v04_depois_A_depois_B.wav', np.concatenate(parts[:-1]), 24000, subtype='PCM_16')
approval = dict(revision=5, state='awaiting_Michael_variant_choice', batch_generation_authorized=False,
                characters={'TREVOR': {'approved': True, 'approved_revision': 2, 'sha256': APPROVED_TREVOR},
                            'MICHAEL': {'approved': False, 'feedback_v4': 'still sounds anglicised',
                                        'variants': {'A_nativo': [str(x) for x in ['amostras_ptbr/michael_A_nativo_comparacao.wav', 'amostras_ptbr/michael_A_nativo_fala_real.wav']],
                                                     'B_misto': ['amostras_ptbr/michael_B_misto_comparacao.wav', 'amostras_ptbr/michael_B_misto_fala_real.wav']}},
                            'FRANKLIN': {'approved': False, 'current_revision': 4, 'note': 'no feedback on v0.4 yet'}})
(out / 'approval.json').write_text(json.dumps(approval, ensure_ascii=False, indent=2), encoding='utf-8')
for r in report: print(r['variant'], r['phrase'], r['seconds'], r['pt_logprob'], r['cos_own_bank'])
