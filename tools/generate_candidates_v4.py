"""Revision v0.4 candidate generator for Michael and Franklin (Trevor untouched).

Diagnosis that motivated v0.4: in v0.3 the S3Gen voice conversion kept the pitch
contour of the Trevor guide (F0 median identical in all three files), so Michael and
Franklin inherited Trevor's high, manic melody. v0.4 synthesises Portuguese directly
per character with Chatterbox Multilingual (language 'pt'), on the GPU, using new
references taken from each protagonist's dedicated ambient bank (s_misc.rpf).

Methods per character:
  cfg0      - character prompt, cfg_weight=0 (recommended against accent transfer)
  cfg03     - character prompt, cfg_weight=0.3
  ptprompt  - T3 speech prompt tokens from the approved Portuguese Trevor sample
              (Portuguese prosody/phonetics context), speaker embedding and S3Gen
              timbre from the character reference, cfg_weight=0.5

Runs in D:\\JARVIS\\.venv-qwen (CUDA torch) with the chatterbox package from
D:\\JARVIS\\.venv-tts appended to sys.path. Attention is switched to eager because
the alignment analyser needs attention outputs. No model or package files are changed.

usage: generate_candidates_v4.py --output <revisao_v4> --guide <approved trevor.wav>
"""
import os, sys, json, time, hashlib, argparse, pathlib
ap = argparse.ArgumentParser()
ap.add_argument('--output', type=pathlib.Path, required=True)
ap.add_argument('--guide', type=pathlib.Path, required=True)
ap.add_argument('--seeds', type=int, nargs='+', default=[11, 23, 37])
ap.add_argument('--only', choices=['MICHAEL', 'FRANKLIN'])
ap.add_argument('--methods', nargs='+', default=['cfg0', 'cfg03', 'ptprompt'])
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
guide_hash = hashlib.sha256(args.guide.read_bytes()).hexdigest()
assert guide_hash == APPROVED_TREVOR, 'guide is not the approved Trevor sample'

REFS = {  # order matters: S3Gen uses the first 10 s, T3 prompt the first 6 s
    'MICHAEL': ['michael_d4.wav', 'michael_d7.wav', 'michael_d6.wav', 'michael_d5.wav'],
    'FRANKLIN': ['franklin_d4.wav', 'franklin_d2.wav', 'franklin_d1.wav', 'franklin_d6.wav'],
}
PHRASES = {
    'comparacao': {'MICHAEL': 'Você tá de brincadeira comigo? Vamos acabar logo com essa bagunça!',
                   'FRANKLIN': 'Você tá de brincadeira comigo? Vamos acabar logo com essa bagunça!'},
    'fala_real': {'MICHAEL': 'Agora você vai dar uma de psicólogo. Igual o resto da cidade.',
                  'FRANKLIN': 'O Stretch entregou cê pros Ballas e eu que sou o Sr. Desleal? Sério mesmo?'},
}
REAL_IDS = {'MICHAEL': 'fam5aud:FAM5_SON4_21', 'FRANKLIN': 'chatcau:CHATC_LAM13_3'}
SR = 24000

def build_reference(who):
    target = out / 'referencias_preparadas' / f'{who.lower()}_reference.wav'
    target.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for name in REFS[who]:
        y, _ = librosa.load(str(out / 'referencias_candidatas' / name), sr=SR)
        y, _ = librosa.effects.trim(y, top_db=38)
        y = y / max(1e-8, float(np.sqrt(np.mean(y ** 2)))) * 0.08
        parts += [y, np.zeros(int(SR * 0.2), np.float32)]
    y = np.concatenate(parts[:-1])
    y *= min(1.0, 0.95 / float(np.max(np.abs(y))))
    sf.write(target, y, SR, subtype='PCM_16')
    return target, len(y) / SR

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
guide_audio, _ = sf.read(str(args.guide), dtype='float32')
ref_rms = float(np.sqrt(np.mean(guide_audio ** 2)))
guide16 = librosa.load(str(args.guide), sr=S3_SR)[0][:model.ENC_COND_LEN]
pt_tokens, _ = model.s3gen.tokenizer.forward([guide16], max_len=model.t3.hp.speech_cond_prompt_len)
pt_tokens = torch.atleast_2d(pt_tokens).to('cuda')

(out / 'candidatos').mkdir(exist_ok=True)
manifest_path = out / 'candidatos' / 'candidates.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else []
refs_info = {}
for who in ['MICHAEL', 'FRANKLIN']:
    if args.only and who != args.only: continue
    ref, ref_seconds = build_reference(who)
    refs_info[who] = dict(file=str(ref), seconds=round(ref_seconds, 3), clips=REFS[who])
    for method in args.methods:
        model.prepare_conditionals(str(ref), exaggeration=0.5)
        cfg = {'cfg0': 0.0, 'cfg03': 0.3, 'ptprompt': 0.5}[method]
        if method == 'ptprompt':
            c = model.conds.t3
            model.conds.t3 = T3Cond(speaker_emb=c.speaker_emb, cond_prompt_speech_tokens=pt_tokens,
                                    emotion_adv=c.emotion_adv).to(device='cuda')
        for phrase, texts in PHRASES.items():
            for seed in args.seeds:
                text = texts[who]; tick = time.monotonic()
                torch.manual_seed(seed); np.random.seed(seed)
                wav = model.generate(text, 'pt', exaggeration=0.5, cfg_weight=cfg, temperature=0.75)
                audio = wav.squeeze(0).numpy().astype(np.float32)
                assert np.isfinite(audio).all()
                audio, gain = finish(audio, ref_rms)
                name = f'{who.lower()}_{phrase}_{method}_s{seed}.wav'
                sf.write(out / 'candidatos' / name, audio, SR, subtype='PCM_16')
                row = dict(file=name, character=who, phrase=phrase, text=text, method=method, cfg_weight=cfg,
                           exaggeration=0.5, temperature=0.75, seed=seed, seconds=round(len(audio) / SR, 3),
                           gain=round(gain, 4), generation_seconds=round(time.monotonic() - tick, 2),
                           source_id=REAL_IDS[who] if phrase == 'fala_real' else None)
                manifest = [m for m in manifest if m['file'] != name] + [row]
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
                print('DONE', json.dumps(row, ensure_ascii=False), flush=True)
(out / 'referencias_preparadas' / 'references_v4.json').write_text(json.dumps(refs_info, indent=2), encoding='utf-8')
print('STAGE finished', flush=True)
