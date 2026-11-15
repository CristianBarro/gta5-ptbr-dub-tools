"""Assemble the v0.4 approval package from the screened candidates.

Copies the chosen candidates to amostras_ptbr/, copies the approved Trevor sample
byte-for-byte, builds A/B files (v0.3 then v0.4) and writes the state files.
usage: finalize_v4.py <revisao_v4>   (run with D:\\JARVIS\\.venv-tts python)
"""
import sys, json, shutil, hashlib, pathlib
import numpy as np, soundfile as sf
out = pathlib.Path(sys.argv[1]).resolve(); v3 = out.parent / 'revisao_v3'
APPROVED_TREVOR = '6254b53efbc07232ec0b6ff78962d99505b7c466fa6b4fdc1b8534864ce48dc0'
CHOSEN = {
    ('MICHAEL', 'comparacao'): 'michael_comparacao_ptprompt_s37.wav',
    ('MICHAEL', 'fala_real'): 'michael_fala_real_cfg0_s11.wav',
    ('FRANKLIN', 'comparacao'): 'franklin_comparacao_ptprompt_s23.wav',
    ('FRANKLIN', 'fala_real'): 'franklin_fala_real_cfg0_s11.wav',
}
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
ranking = {r['file']: r for r in json.loads((out / 'candidatos' / 'ranking.json').read_text(encoding='utf-8'))}
dest = out / 'amostras_ptbr'; dest.mkdir(exist_ok=True)
shutil.copy2(v3 / 'amostras_ptbr' / 'trevor.wav', dest / 'trevor.wav')
assert sha(dest / 'trevor.wav') == APPROVED_TREVOR
report = []
for (who, phrase), name in CHOSEN.items():
    target = dest / f'{who.lower()}_{phrase}.wav'; shutil.copy2(out / 'candidatos' / name, target)
    r = ranking[name]
    report.append(dict(character=who, phrase=phrase, text=r['text'], file=str(target), candidate=name,
                       source_id=r.get('source_id'), seconds=r['seconds'], engine='Chatterbox Multilingual (pt) on CUDA',
                       method=r['method'], cfg_weight=r['cfg_weight'], exaggeration=r['exaggeration'],
                       temperature=r['temperature'], seed=r['seed'], asr_pt=r['asr'], word_errors=r['word_errors'],
                       whisper_p_pt=r['p_pt'], whisper_p_en=r['p_en'], cos_own_bank=r['cos_own'],
                       cos_other_protagonist=r['cos_other'], f0_median_pyin=r['f0_median'],
                       sha256=sha(target), approved=False))
report.append(dict(character='TREVOR', phrase='comparacao', text='Você tá de brincadeira comigo? Vamos acabar logo com essa bagunça!',
                   file=str(dest / 'trevor.wav'), engine='Qwen3-TTS 0.6B (revision 2)', sha256=APPROVED_TREVOR,
                   approved=True, approved_revision=2))
(out / 'preview_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

# A/B: v0.3 sample, 0.8 s silence, v0.4 sample (same sentence)
ab = out / 'comparacao_ab'; ab.mkdir(exist_ok=True)
for who in ['michael', 'franklin']:
    a, sr = sf.read(v3 / 'amostras_ptbr' / f'{who}.wav', dtype='float32')
    b, sr2 = sf.read(dest / f'{who}_comparacao.wav', dtype='float32'); assert sr == sr2 == 24000
    sf.write(ab / f'{who}_v03_depois_v04.wav', np.concatenate([a, np.zeros(int(sr * 0.8), np.float32), b]), sr, subtype='PCM_16')

approval = dict(revision=4, state='awaiting_Michael_and_Franklin_approval', batch_generation_authorized=False,
                characters={'TREVOR': {'approved': True, 'approved_revision': 2, 'sha256': APPROVED_TREVOR},
                            'MICHAEL': {'approved': False, 'candidate_files': ['amostras_ptbr/michael_comparacao.wav', 'amostras_ptbr/michael_fala_real.wav']},
                            'FRANKLIN': {'approved': False, 'candidate_files': ['amostras_ptbr/franklin_comparacao.wav', 'amostras_ptbr/franklin_fala_real.wav']}})
(out / 'approval.json').write_text(json.dumps(approval, ensure_ascii=False, indent=2), encoding='utf-8')
refs = json.loads((out / 'referencias_preparadas' / 'references_v4.json').read_text(encoding='utf-8'))
profiles = {who: dict(reference_file=refs[who]['file'], reference_seconds=refs[who]['seconds'], clips=refs[who]['clips'],
                      origin=f"dedicated ambient bank s_misc.rpf\\{who.lower()}_*_normal.awc (see refs_v4.tsv / reference_transcripts.json)",
                      state='awaiting_user_voice_approval') for who in ['MICHAEL', 'FRANKLIN']}
profiles['TREVOR'] = json.loads((v3 / 'voice_profiles.json').read_text(encoding='utf-8'))['TREVOR']
(out / 'voice_profiles.json').write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=1))
