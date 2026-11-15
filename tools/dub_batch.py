"""Resumable PT-BR dubbing generator (pilot first, campaign later).

Reads the dubbing queue (fila_dublagem.csv), normalises the Portuguese text and
generates N takes per line with the profile approved for each character:

  MICHAEL  Chatterbox Multilingual pt; T3 conditioned on natural approved pt-BR speech
           (Franklin pilot takes, hash-pinned), S3Gen timbre = Michael
           dedicated-bank reference. Approved revision 6, variant franklin_nat.
  TREVOR   Qwen3-TTS 0.6B Base, x_vector_only_mode, BF16 talker + FP32 decoder,
           Trevor reference of revision 2. Approved revision 2.
  FRANKLIN Chatterbox Multilingual pt, Franklin dedicated-bank reference; take1 cfg0,
           take2 ptprompt. Approved revision 4.

One model is loaded per process (6 GB VRAM): run once per character.
Every finished take is recorded immediately in <out>/manifest.json, so an interrupted
run continues where it stopped. Nothing in the game installation is touched.

usage:
  dub_batch.py --queue fila_dublagem.csv --approval revisao_v5/approval.json --out <dir>
               --character MICHAEL --select pilot --per-character 12 --takes 2
  (--select all is refused while batch_generation_authorized is false)
"""
import os, sys, re, csv, json, time, random, hashlib, argparse, pathlib
ap = argparse.ArgumentParser()
ap.add_argument('--queue', type=pathlib.Path, required=True)
ap.add_argument('--approval', type=pathlib.Path, required=True)
ap.add_argument('--out', type=pathlib.Path, required=True)
ap.add_argument('--character', choices=['MICHAEL', 'TREVOR', 'FRANKLIN'], required=True)
ap.add_argument('--select', choices=['pilot', 'all'], default='pilot')
ap.add_argument('--per-character', type=int, default=12)
ap.add_argument('--takes', type=int, default=2)
ap.add_argument('--adaptations', type=pathlib.Path, help='JSON {id: shorter PT text}; only these ids are (re)generated')
ap.add_argument('--batch-size', type=int, default=1,
                help='TREVOR only: generate this many lines per Qwen call (lines sorted by length; a failed batch falls back to one by one)')
args = ap.parse_args()
out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
for key, folder in {'HF_HOME': 'hf', 'NUMBA_CACHE_DIR': 'numba', 'TEMP': 'tmp', 'TMP': 'tmp', 'TMPDIR': 'tmp', 'TORCH_HOME': 'torch'}.items():
    p = out / 'cache' / folder; p.mkdir(parents=True, exist_ok=True); os.environ[key] = str(p)
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

PROJECT = pathlib.Path(r'C:\gta-dub')
VOZES = PROJECT / 'outputs' / 'GTA_V_Dublagem_PTBR'
approval = json.loads(args.approval.read_text(encoding='utf-8'))
who = args.character
if not approval['characters'].get(who, {}).get('approved'):
    sys.exit(f'{who} is not approved in {args.approval}; refusing to generate.')
if args.select == 'all' and not approval.get('batch_generation_authorized'):
    sys.exit('batch_generation_authorized is false; only --select pilot is allowed.')

ABBREV = {r'\bSr\.': 'Senhor', r'\bSra\.': 'Senhora', r'\bSrta\.': 'Senhorita', r'\bDr\.': 'Doutor', r'\bDra\.': 'Doutora'}
def normalise(text):
    t = text.strip()
    for pat, rep in ABBREV.items(): t = re.sub(pat, rep, t)
    t = t.replace('%', ' por cento')
    t = re.sub(r'\s+', ' ', t)
    if t and t[-1] not in '.!?…': t += '.'
    return t
def needs_text_review(text):
    return bool(re.search(r'\d', text))

rows = [r for r in csv.DictReader(open(args.queue, encoding='utf-8-sig')) if r['character'] == who]
if args.select == 'pilot':
    rng = random.Random(20261114)
    usable = [r for r in rows if not needs_text_review(r['text_pt'])]
    bins = [(0.0, 2.0, 4), (2.0, 4.0, 5), (4.0, 8.5, 3)]
    scale = args.per_character / 12
    chosen = []
    for lo, hi, n in bins:
        pool = [r for r in usable if lo <= float(r['target_duration_seconds']) < hi]
        chosen += rng.sample(pool, max(1, round(n * scale)))
    rows = chosen
adapt = json.loads(args.adaptations.read_text(encoding='utf-8')) if args.adaptations else {}
if adapt: rows = [r for r in rows if r['id'] in adapt]
print(f'STAGE {who}: {len(rows)} lines x {args.takes} takes', flush=True)

manifest_path = out / 'manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
def save_manifest():
    tmp = manifest_path.with_suffix('.tmp'); tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    for attempt in range(40):  # another process (cut_pick.py) may be reading the manifest: Windows refuses the replace meanwhile
        try: tmp.replace(manifest_path); return
        except PermissionError: time.sleep(0.5)
    tmp.replace(manifest_path)
safe = lambda s: re.sub(r'[^A-Za-z0-9_]+', '_', s)

import numpy as np, librosa, soundfile as sf, torch
SR_OUT = 24000
def finish(audio, sr, ref_rms):
    if sr != SR_OUT: audio = librosa.resample(audio, orig_sr=sr, target_sr=SR_OUT)
    audio, _ = librosa.effects.trim(audio, top_db=42, frame_length=1024, hop_length=128)
    pad = np.zeros(int(SR_OUT * 0.06), np.float32); audio = np.concatenate([pad, audio, pad]).astype(np.float32)
    rms = float(np.sqrt(np.mean(audio ** 2))); peak = float(np.max(np.abs(audio)))
    gain = min(ref_rms / max(1e-8, rms), 0.94 / max(1e-8, peak)); audio = audio * gain
    fade = int(SR_OUT * 0.008); audio[:fade] *= np.linspace(0, 1, fade); audio[-fade:] *= np.linspace(1, 0, fade)
    return audio, gain
trevor_approved = VOZES / 'revisao_v3' / 'amostras_ptbr' / 'trevor.wav'
assert hashlib.sha256(trevor_approved.read_bytes()).hexdigest() == approval['characters']['TREVOR']['sha256']
_g, _ = sf.read(str(trevor_approved), dtype='float32'); REF_RMS = float(np.sqrt(np.mean(_g ** 2)))

def clear_alignment_hooks(m):
    # chatterbox AlignmentStreamAnalyzer registers a forward hook on every generate() and never removes it;
    # the hooks pile up and each line gets slower (lote_v1 Michael: RTF 1.9 -> 7.7 over 1694 lines)
    for layer in m.t3.tfmr.layers: layer.self_attn._forward_hooks.clear()

# per-line delivery override (cutscenes): queue columns exaggeration / cfg_weight, set for each row before synth()
EMO = {'ex': None, 'cfg': None}

if who == 'MICHAEL':
    sys.path.append(r'D:\JARVIS\.venv-tts\Lib\site-packages')
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.models.s3tokenizer import S3_SR
    prof = approval['characters']['MICHAEL']['generation_profile']
    CKPT = r'D:\JARVIS\data\clone-models\hub\models--ResembleAI--chatterbox\snapshots\5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18'
    model = ChatterboxMultilingualTTS.from_local(CKPT, 'cuda'); model.t3.tfmr.config._attn_implementation = 'eager'
    native = []
    # rev6: T3 conditioned on natural approved Portuguese (Franklin pilot takes), pinned by hash
    for rel, digest in sorted(prof['t3_conditioning_files'].items()):
        p = VOZES / rel
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, f'conditioning file changed: {p}'
        y, _ = librosa.load(str(p), sr=S3_SR); y, _ = librosa.effects.trim(y, top_db=38)
        native += [y / max(1e-8, float(np.sqrt(np.mean(y ** 2)))) * 0.08, np.zeros(int(S3_SR * 0.2), np.float32)]
    native = np.concatenate(native[:-1]).astype(np.float32)
    toks, _ = model.s3gen.tokenizer.forward([native[:model.ENC_COND_LEN]], max_len=model.t3.hp.speech_cond_prompt_len)
    emb = torch.from_numpy(model.ve.embeds_from_wavs([native], sample_rate=S3_SR)).mean(0, keepdim=True)
    model.prepare_conditionals(str(VOZES / 'revisao_v4' / 'referencias_preparadas' / 'michael_reference.wav'), exaggeration=prof['exaggeration'])
    model.conds.t3 = T3Cond(speaker_emb=emb.to('cuda'), cond_prompt_speech_tokens=torch.atleast_2d(toks).to('cuda'),
                            emotion_adv=prof['exaggeration'] * torch.ones(1, 1, 1)).to(device='cuda')
    ENGINE = 'chatterbox_mtl_pt_franklin_nat_t3__michael_s3gen_rev6'
    def synth(text, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        clear_alignment_hooks(model)
        ex = EMO['ex'] if EMO['ex'] is not None else prof['exaggeration']
        cfg = EMO['cfg'] if EMO['cfg'] is not None else prof['cfg_weight']
        w = model.generate(text, 'pt', exaggeration=ex, cfg_weight=cfg, temperature=prof['temperature'])
        return w.squeeze(0).numpy().astype(np.float32), model.sr
elif who == 'TREVOR':
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(r'D:\JARVIS\data\qwen-models\Qwen3-TTS-12Hz-0.6B-Base',
                                          device_map=os.environ.get('DUB_QWEN_DEVICE', 'cuda:0'),  # cpu only for code tests
                                          torch_dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
    model.model.speech_tokenizer.model.to(dtype=torch.float32)
    TREVOR_REF = str(VOZES / 'revisao_v2' / 'referencias_preparadas' / 'trevor_reference.wav')
    ENGINE = 'qwen3_tts_0.6b_xvector__trevor_rev2'
    # x-vector prompt built once from the same reference (it used to be rebuilt from the file on every call;
    # same embedding, so the voice is unchanged)
    PROMPT = model.create_voice_clone_prompt(ref_audio=TREVOR_REF, x_vector_only_mode=True)
    QWEN_KW = dict(temperature=0.55, top_k=40, repetition_penalty=1.08, non_streaming_mode=True)
    def qwen_max_tokens(text): return min(600, 60 + 14 * len(text.split()) * 2)
    def synth(text, seed):
        torch.manual_seed(seed)
        wavs, sr = model.generate_voice_clone(text=text, language='Portuguese', voice_clone_prompt=PROMPT,
                                              max_new_tokens=qwen_max_tokens(text), **QWEN_KW)
        return np.asarray(wavs[0], dtype=np.float32), sr
    def synth_batch(texts, seed):
        torch.manual_seed(seed)
        wavs, sr = model.generate_voice_clone(text=list(texts), language=['Portuguese'] * len(texts), voice_clone_prompt=PROMPT * len(texts),
                                              max_new_tokens=max(qwen_max_tokens(t) for t in texts), **QWEN_KW)
        return [np.asarray(w, dtype=np.float32) for w in wavs], sr
elif who == 'FRANKLIN':
    sys.path.append(r'D:\JARVIS\.venv-tts\Lib\site-packages')
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.models.s3tokenizer import S3_SR
    prof = approval['characters']['FRANKLIN']['generation_profile']
    CKPT = r'D:\JARVIS\data\clone-models\hub\models--ResembleAI--chatterbox\snapshots\5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18'
    model = ChatterboxMultilingualTTS.from_local(CKPT, 'cuda'); model.t3.tfmr.config._attn_implementation = 'eager'
    model.prepare_conditionals(str(VOZES / 'revisao_v4' / 'referencias_preparadas' / 'franklin_reference.wav'), exaggeration=prof['exaggeration'])
    base_t3 = model.conds.t3
    g16 = librosa.load(str(trevor_approved), sr=S3_SR)[0][:model.ENC_COND_LEN]
    pt_toks, _ = model.s3gen.tokenizer.forward([g16], max_len=model.t3.hp.speech_cond_prompt_len)
    pt_t3 = T3Cond(speaker_emb=base_t3.speaker_emb, cond_prompt_speech_tokens=torch.atleast_2d(pt_toks).to('cuda'),
                   emotion_adv=base_t3.emotion_adv).to(device='cuda')
    ENGINE = 'chatterbox_mtl_pt__franklin_rev4'
    # take1 = cfg0 method, take2 = ptprompt method (both present in the approved v0.4 files)
    def synth(text, seed):
        method = 'cfg0' if seed == 1017 else 'ptprompt'
        model.conds.t3 = base_t3 if method == 'cfg0' else pt_t3
        torch.manual_seed(seed); np.random.seed(seed)
        clear_alignment_hooks(model)
        ex = EMO['ex'] if EMO['ex'] is not None else prof['exaggeration']
        cfg = prof['takes'][method]['cfg_weight'] if EMO['cfg'] is None else min(EMO['cfg'], prof['takes'][method]['cfg_weight'])
        w = model.generate(text, 'pt', exaggeration=ex, cfg_weight=cfg, temperature=prof['temperature'])
        return w.squeeze(0).numpy().astype(np.float32), model.sr
else:
    sys.exit('No generation profile for ' + who)

(out / 'linhas').mkdir(exist_ok=True)
BATCH = args.batch_size if who == 'TREVOR' else 1
jobs = []      # batch mode: (i, r, entry, key, seed, text) collected first, generated grouped by text length
def row_emo(r):
    EMO['ex'] = float(r['exaggeration']) if r.get('exaggeration') else None
    EMO['cfg'] = float(r['cfg_weight']) if r.get('cfg_weight') else None
def record(i, r, entry, key, seed, audio, sr, gen_seconds, batch_n=None, error=None):
    row_emo(r)
    try:
        if error is not None: raise error
        assert np.isfinite(audio).all() and len(audio) > 0.2 * sr
        audio, gain = finish(audio, sr, REF_RMS)
        rel = f'linhas/{safe(r["id"])}_{key}.wav'
        sf.write(out / rel, audio, SR_OUT, subtype='PCM_16')
        entry['takes'][key] = dict(file=rel, seed=seed, engine=ENGINE, seconds=round(len(audio) / SR_OUT, 3),
                                   duration_ratio=round(len(audio) / SR_OUT / max(0.2, entry['target_seconds']), 3),
                                   gain=round(gain, 4), generation_seconds=round(gen_seconds, 2), status='generated',
                                   exaggeration=EMO['ex'], cfg_override=EMO['cfg'])
        if batch_n: entry['takes'][key]['batch'] = batch_n
    except Exception as exc:  # recorded, not fatal: the line goes to review
        entry['takes'][key] = dict(file='', seed=seed, engine=ENGINE, status='failed', error=repr(exc)[:300])
    done_count = globals().get('done_count', 0) + 1; globals()['done_count'] = done_count
    if done_count % 10 == 0: save_manifest()
    print(f'[{i}/{len(rows)}] {r["id"]} {key} {entry["takes"][key].get("seconds")}s target {entry["target_seconds"]:.2f}s'
          + (f' (batch {batch_n})' if batch_n else ''), flush=True)
for i, r in enumerate(rows, 1):
    text = normalise(r['text_pt'])
    entry = manifest.setdefault(r['id'], dict(id=r['id'], character=who, text_pt=r['text_pt'], text_tts=text, english=r['english'],
                                              awc_path=r['awc_path'], stream_hash=r['stream_hash'],
                                              target_seconds=float(r['target_duration_seconds']), takes={}))
    if r['id'] in adapt and entry['text_tts'] != normalise(adapt[r['id']]):
        # shorter Portuguese adaptation: keep the earlier takes for the record, start new ones
        entry.setdefault('takes_before_adaptation', entry['takes']); entry['takes'] = {}
        entry['text_tts'] = normalise(adapt[r['id']]); entry['text_adapted'] = True
    text = entry['text_tts']; prefix = 'adapt' if entry.get('text_adapted') else 'take'
    EMO['ex'] = float(r['exaggeration']) if r.get('exaggeration') else None
    EMO['cfg'] = float(r['cfg_weight']) if r.get('cfg_weight') else None
    if EMO['ex'] is not None and who != 'TREVOR':  # intensified delivery gets its own takes (keeps the neutral ones)
        prefix = f"emo{int(round(EMO['ex'] * 100))}_"
    if needs_text_review(text):
        entry['status'] = 'texto_revisar_numeros'; continue
    entry.pop('status', None)  # cleared once an adaptation spells the numbers out
    for t in range(1, args.takes + 1):
        key = f'{prefix}{t}'
        if key in entry['takes'] and (out / entry['takes'][key]['file']).exists(): continue
        seed = 1000 + 17 * t
        if BATCH > 1: jobs.append((i, r, entry, key, seed, text)); continue
        tick = time.monotonic()
        err = None
        try: audio, sr = synth(text, seed)
        except Exception as exc: audio, sr, err = None, SR_OUT, exc
        record(i, r, entry, key, seed, audio, sr, time.monotonic() - tick, error=err)
if jobs:
    # Trevor in batches: similar lengths together (less padding); seed per batch, so takes differ from the one-by-one run
    # but come from the same model, prompt and sampling settings. A batch item longer than its own token budget allows
    # (it would have been cut there one by one) or a batch that fails is regenerated one by one.
    jobs.sort(key=lambda j: len(j[5]))
    print(f'BATCH {who}: {len(jobs)} takes in batches of {BATCH}', flush=True)
    for b in range(0, len(jobs), BATCH):
        group = jobs[b:b + BATCH]; tick = time.monotonic()
        try:
            wavs, sr = synth_batch([j[5] for j in group], group[0][4] + b)
        except Exception as exc:
            print('batch failed, one by one:', repr(exc)[:200], flush=True); wavs, sr = [None] * len(group), SR_OUT
        per = (time.monotonic() - tick) / len(group)
        for j, w in zip(group, wavs):
            i, r, entry, key, seed, text = j
            if w is None or len(w) / sr > qwen_max_tokens(text) / 12.0 + 0.5:
                t1 = time.monotonic(); err = None
                try: w, sr1 = synth(text, seed)
                except Exception as exc: w, sr1, err = None, SR_OUT, exc
                record(i, r, entry, key, seed, w, sr1, time.monotonic() - t1, error=err)
            else:
                record(i, r, entry, key, group[0][4] + b, w, sr, per, batch_n=len(group))
save_manifest()
print('STAGE finished', who, flush=True)
