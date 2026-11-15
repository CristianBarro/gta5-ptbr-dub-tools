"""Revision v0.5 for Michael only: remove the remaining English accent.

Feedback on v0.4: Michael still sounds anglicised. In Chatterbox the T3 model decides
phonetics/rhythm (speech tokens) and is conditioned on a speaker embedding plus prompt
speech tokens; S3Gen renders those tokens with the timbre of its own reference. In v0.4
T3 still received Michael's English voice (embedding, and prompt tokens for cfg0), which
carries the accent. v0.5 conditions T3 on *native Portuguese* speech only (or a blend of
embeddings) and keeps Michael solely in S3Gen, for timbre.

Portuguese conditioning sources (local only):
  trevor_pt : approved Portuguese Trevor sample (revision 2, hash-checked)
  daniel    : Windows OneCore "Microsoft Daniel - Portuguese (Brazil)" clips

Methods (T3 embedding / T3 prompt tokens; S3Gen always Michael):
  trevor_full  : trevor_pt / trevor_pt
  daniel_full  : daniel / daniel
  daniel_mix   : 0.5 Michael + 0.5 daniel (renormalised) / daniel
  trevor_mix   : 0.5 Michael + 0.5 trevor_pt / trevor_pt

usage: generate_michael_v5.py --output <revisao_v5> --v4 <revisao_v4> --guide <approved trevor.wav>
"""
import os, sys, json, time, hashlib, argparse, pathlib
ap = argparse.ArgumentParser()
ap.add_argument('--output', type=pathlib.Path, required=True)
ap.add_argument('--v4', type=pathlib.Path, required=True)
ap.add_argument('--guide', type=pathlib.Path, required=True)
ap.add_argument('--seeds', type=int, nargs='+', default=[11, 23, 37])
ap.add_argument('--methods', nargs='+', default=['trevor_full', 'daniel_full', 'daniel_mix', 'trevor_mix'])
ap.add_argument('--exaggeration', type=float, default=0.4)
args = ap.parse_args()
out = args.output.resolve()
for key, folder in {'HF_HOME': 'hf', 'NUMBA_CACHE_DIR': 'numba', 'TEMP': 'tmp', 'TMP': 'tmp', 'TMPDIR': 'tmp', 'TORCH_HOME': 'torch'}.items():
    p = out / 'cache' / folder; p.mkdir(parents=True, exist_ok=True); os.environ[key] = str(p)
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.path.append(r'D:\JARVIS\.venv-tts\Lib\site-packages')
import numpy as np, librosa, soundfile as sf, torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.s3tokenizer import S3_SR

CKPT = r'D:\JARVIS\data\clone-models\hub\models--ResembleAI--chatterbox\snapshots\5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18'
APPROVED_TREVOR = '6254b53efbc07232ec0b6ff78962d99505b7c466fa6b4fdc1b8534864ce48dc0'
assert hashlib.sha256(args.guide.read_bytes()).hexdigest() == APPROVED_TREVOR
PHRASES = {'comparacao': 'Você tá de brincadeira comigo? Vamos acabar logo com essa bagunça!',
           'fala_real': 'Agora você vai dar uma de psicólogo. Igual o resto da cidade.'}
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

print('STAGE loading Chatterbox Multilingual on CUDA', flush=True)
model = ChatterboxMultilingualTTS.from_local(CKPT, 'cuda')
model.t3.tfmr.config._attn_implementation = 'eager'
plen = model.t3.hp.speech_cond_prompt_len
michael_ref = args.v4.resolve() / 'referencias_preparadas' / 'michael_reference.wav'
sources = {'trevor_pt': load16([args.guide]),
           'daniel': load16(sorted((out / 'condicionamento_pt').glob('daniel_*.wav')))}
def t3_parts(wav16):
    toks, _ = model.s3gen.tokenizer.forward([wav16[:model.ENC_COND_LEN]], max_len=plen)
    emb = torch.from_numpy(model.ve.embeds_from_wavs([wav16], sample_rate=S3_SR)).mean(0, keepdim=True)
    return torch.atleast_2d(toks).to('cuda'), emb.to('cuda')
parts = {k: t3_parts(v) for k, v in sources.items()}
_, michael_emb = t3_parts(librosa.load(str(michael_ref), sr=S3_SR)[0].astype(np.float32))
def mix(a, b):
    m = 0.5 * a / a.norm() + 0.5 * b / b.norm(); return m / m.norm() * (0.5 * a.norm() + 0.5 * b.norm())
CONF = {'trevor_full': ('trevor_pt', lambda e: e), 'daniel_full': ('daniel', lambda e: e),
        'daniel_mix': ('daniel', lambda e: mix(michael_emb, e)), 'trevor_mix': ('trevor_pt', lambda e: mix(michael_emb, e))}
guide_audio, _ = sf.read(str(args.guide), dtype='float32'); ref_rms = float(np.sqrt(np.mean(guide_audio ** 2)))

(out / 'candidatos').mkdir(exist_ok=True)
manifest_path = out / 'candidatos' / 'candidates.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else []
for method in args.methods:
    src, emb_fn = CONF[method]
    model.prepare_conditionals(str(michael_ref), exaggeration=args.exaggeration)  # S3Gen <- Michael timbre
    toks, emb = parts[src]
    model.conds.t3 = T3Cond(speaker_emb=emb_fn(emb), cond_prompt_speech_tokens=toks,
                            emotion_adv=args.exaggeration * torch.ones(1, 1, 1)).to(device='cuda')
    for phrase, text in PHRASES.items():
        for seed in args.seeds:
            tick = time.monotonic(); torch.manual_seed(seed); np.random.seed(seed)
            wav = model.generate(text, 'pt', exaggeration=args.exaggeration, cfg_weight=0.5, temperature=0.75)
            audio, gain = finish(wav.squeeze(0).numpy().astype(np.float32), ref_rms)
            name = f'michael_{phrase}_{method}_s{seed}.wav'
            sf.write(out / 'candidatos' / name, audio, SR, subtype='PCM_16')
            row = dict(file=name, character='MICHAEL', phrase=phrase, text=text, method=method, t3_source=src,
                       cfg_weight=0.5, exaggeration=args.exaggeration, temperature=0.75, seed=seed,
                       seconds=round(len(audio) / SR, 3), gain=round(gain, 4), generation_seconds=round(time.monotonic() - tick, 2),
                       source_id='fam5aud:FAM5_SON4_21' if phrase == 'fala_real' else None)
            manifest = [m for m in manifest if m['file'] != name] + [row]
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
            print('DONE', name, row['seconds'], flush=True)
print('STAGE finished', flush=True)
