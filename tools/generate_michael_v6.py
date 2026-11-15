"""Revision v0.6 for Michael: natural (not robotic) native Portuguese T3 conditioning.

Feedback on pilot v1: Michael "um pouco estranho ainda"; Franklin and Trevor great.
Hypothesis: v0.5 A_nativo conditioned T3 on the Windows OneCore "Daniel" voice, which is
a robotic concatenative/parametric TTS; T3 imitates that flat, mechanical rhythm.
v0.6 conditions T3 on natural Portuguese speech that the user already approved:
selected pilot v1 takes of Trevor (Qwen) and Franklin (Chatterbox). Michael stays only
in S3Gen (timbre), same reference as before.

Variants:
  trevor_nat   T3 embedding + prompt tokens from approved Trevor pilot takes
  franklin_nat T3 embedding + prompt tokens from approved Franklin pilot takes
  franklin_mix franklin_nat tokens; embedding 0.7 franklin_nat + 0.3 Michael

usage: generate_michael_v6.py --pilot <piloto_v1> --output <revisao_v6>
"""
import os, sys, json, time, argparse, pathlib, csv
ap = argparse.ArgumentParser()
ap.add_argument('--pilot', type=pathlib.Path, required=True)
ap.add_argument('--output', type=pathlib.Path, required=True)
ap.add_argument('--seeds', type=int, nargs='+', default=[1017, 1034])
args = ap.parse_args()
out = args.output.resolve(); pilot = args.pilot.resolve()
for key, folder in {'HF_HOME': 'hf', 'NUMBA_CACHE_DIR': 'numba', 'TEMP': 'tmp', 'TMP': 'tmp', 'TMPDIR': 'tmp', 'TORCH_HOME': 'torch'}.items():
    p = out / 'cache' / folder; p.mkdir(parents=True, exist_ok=True); os.environ[key] = str(p)
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.path.append(r'D:\JARVIS\.venv-tts\Lib\site-packages')
import numpy as np, librosa, soundfile as sf, torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.s3tokenizer import S3_SR

VOZES = out.parent
CKPT = r'D:\JARVIS\data\clone-models\hub\models--ResembleAI--chatterbox\snapshots\5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18'
SOURCES = {'trevor_nat': ['trevor/08_chatcau_CHATC_CHG2_3.wav', 'trevor/09_m3aud_M3_ESC1_1.wav', 'trevor/11_nig2aud_NIG2_CHASE03_3.wav'],
           'franklin_nat': ['franklin/06_ojasaud_OJASBS_NOPE_1.wav', 'franklin/09_fam3aud_FAM3_FST1_9.wav', 'franklin/10_lem2aud_LM2_BANT1_11.wav']}
TEST_IDS = ['fh1aud:FH1_MCS2_LO_1', 'meflaud:MEFL_C2N_1', 'arm3aud:ARM3_SLAM_2', 'met1aud:MET1_GOHOM1_15', 'csflaud:CSFL_CALL0_15']
manifest = json.loads((pilot / 'manifest.json').read_text(encoding='utf-8'))
SR = 24000

def load16(paths):
    parts = []
    for p in paths:
        y, _ = librosa.load(str(p), sr=S3_SR); y, _ = librosa.effects.trim(y, top_db=38)
        parts += [y / max(1e-8, float(np.sqrt(np.mean(y ** 2)))) * 0.08, np.zeros(int(S3_SR * 0.2), np.float32)]
    return np.concatenate(parts[:-1]).astype(np.float32)
def finish(audio, ref_rms):
    audio, _ = librosa.effects.trim(audio, top_db=42, frame_length=1024, hop_length=128)
    pad = np.zeros(int(SR * 0.06), np.float32); audio = np.concatenate([pad, audio, pad])
    rms = float(np.sqrt(np.mean(audio ** 2))); peak = float(np.max(np.abs(audio)))
    gain = min(ref_rms / max(1e-8, rms), 0.94 / max(1e-8, peak)); audio = audio * gain
    fade = int(SR * 0.008); audio[:fade] *= np.linspace(0, 1, fade); audio[-fade:] *= np.linspace(1, 0, fade)
    return audio.astype(np.float32), gain

model = ChatterboxMultilingualTTS.from_local(CKPT, 'cuda'); model.t3.tfmr.config._attn_implementation = 'eager'
plen = model.t3.hp.speech_cond_prompt_len
def t3_parts(w16):
    toks, _ = model.s3gen.tokenizer.forward([w16[:model.ENC_COND_LEN]], max_len=plen)
    emb = torch.from_numpy(model.ve.embeds_from_wavs([w16], sample_rate=S3_SR)).mean(0, keepdim=True)
    return torch.atleast_2d(toks).to('cuda'), emb.to('cuda')
parts = {k: t3_parts(load16([pilot / 'selecionadas' / f for f in v])) for k, v in SOURCES.items()}
michael_ref = VOZES / 'revisao_v4' / 'referencias_preparadas' / 'michael_reference.wav'
_, michael_emb = t3_parts(librosa.load(str(michael_ref), sr=S3_SR)[0].astype(np.float32))
def mix(a, b, wa):
    m = wa * a / a.norm() + (1 - wa) * b / b.norm(); return m / m.norm() * (wa * a.norm() + (1 - wa) * b.norm())
VARIANTS = {'trevor_nat': ('trevor_nat', lambda e: e), 'franklin_nat': ('franklin_nat', lambda e: e),
            'franklin_mix': ('franklin_nat', lambda e: mix(e, michael_emb, 0.7))}
g, _ = sf.read(str(VOZES / 'revisao_v3' / 'amostras_ptbr' / 'trevor.wav'), dtype='float32'); ref_rms = float(np.sqrt(np.mean(g ** 2)))
(out / 'candidatos').mkdir(exist_ok=True)
rows = []
EXAG = 0.4
for variant, (src, fn) in VARIANTS.items():
    model.prepare_conditionals(str(michael_ref), exaggeration=EXAG)
    toks, emb = parts[src]
    model.conds.t3 = T3Cond(speaker_emb=fn(emb), cond_prompt_speech_tokens=toks, emotion_adv=EXAG * torch.ones(1, 1, 1)).to(device='cuda')
    for lid in TEST_IDS:
        e = manifest[lid]
        for seed in args.seeds:
            torch.manual_seed(seed); np.random.seed(seed); tick = time.monotonic()
            w = model.generate(e['text_tts'], 'pt', exaggeration=EXAG, cfg_weight=0.5, temperature=0.75)
            audio, gain = finish(w.squeeze(0).numpy().astype(np.float32), ref_rms)
            name = f"{variant}__{lid.replace(':', '_')}__s{seed}.wav"
            sf.write(out / 'candidatos' / name, audio, SR, subtype='PCM_16')
            rows.append(dict(file=name, variant=variant, id=lid, character='MICHAEL', text=e['text_tts'], seed=seed,
                             seconds=round(len(audio) / SR, 3), target_seconds=e['target_seconds'],
                             generation_seconds=round(time.monotonic() - tick, 2)))
            print('DONE', name, rows[-1]['seconds'], flush=True)
(out / 'candidatos' / 'candidates.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
(out / 'condicionamento_sources.json').write_text(json.dumps(SOURCES, indent=2), encoding='utf-8')
print('STAGE finished', flush=True)
