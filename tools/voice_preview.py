"""Generate only three approval samples using local weights and original voices."""
import os, sys, pathlib, json, time, logging, faulthandler
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
sys.stdout.reconfigure(encoding="utf-8",line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8",line_buffering=True)
faulthandler.enable()
faulthandler.dump_traceback_later(45, repeat=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
script=pathlib.Path(__file__).resolve()
out=script.parents[1] if script.parent.name=='tools' else script.parents[2]/'outputs'/'GTA_V_Dublagem_PTBR'
out.mkdir(parents=True,exist_ok=True)
(out/'preview_worker.pid').write_text(str(os.getpid()),encoding='ascii')
os.environ['HF_HOME']=str(out/'cache')
for key,folder in {'NUMBA_CACHE_DIR':'numba','TEMP':'tmp','TMP':'tmp','TMPDIR':'tmp','MPLCONFIGDIR':'matplotlib','TRITON_CACHE_DIR':'triton','LIBROSA_CACHE_DIR':'librosa'}.items():
    cache=out/'cache'/folder
    cache.mkdir(parents=True,exist_ok=True)
    os.environ[key]=str(cache)
print('STAGE importing audio runtime',flush=True)
import torch, librosa, soundfile as sf, numpy as np
torch.set_num_threads(4)
profiles=json.loads((out/'voice_profiles.json').read_text(encoding='utf-8'))
for character,profile in profiles.items():
    pieces=[]
    for clip in profile['clips']:
        audio,sr=librosa.load(out/'referencias_originais'/clip['filename'],sr=24000,mono=True)
        pieces.extend([audio,np.zeros(2400,dtype=np.float32)])
    reference=out/'referencias_originais'/(character.lower()+'_reference.wav')
    sf.write(reference,np.concatenate(pieces),24000,subtype='PCM_16')
    profile['reference_file']=str(reference)
(out/'voice_profiles.json').write_text(json.dumps(profiles,ensure_ascii=False,indent=2),encoding='utf-8')
print('STAGE loading local model',torch.__version__,torch.cuda.mem_get_info(),flush=True)
from qwen_tts import Qwen3TTSModel
model=Qwen3TTSModel.from_pretrained(r'D:\JARVIS\data\qwen-models\Qwen3-TTS-12Hz-0.6B-Base',device_map='cuda:0',torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
print('STAGE model ready',flush=True)
texts={
 'MICHAEL':'Eu já fiz a minha parte. Agora eu só quero voltar para a minha família.',
 'FRANKLIN':'A gente precisa sair daqui, mano. Entra no carro e deixa que eu dirijo.',
 'TREVOR':'Você tá de brincadeira comigo? Vamos acabar logo com essa bagunça!'
}
report=[]
for character,text in texts.items():
    profile=profiles[character]
    tick=time.monotonic()
    print('STAGE generating',character,flush=True)
    torch.manual_seed(20261114)
    waveforms,sr=model.generate_voice_clone(text=text,language='Portuguese',ref_audio=profile['reference_file'],ref_text=profile['reference_transcript'],temperature=0.65,top_k=40,repetition_penalty=1.1,non_streaming_mode=True,max_new_tokens=250)
    target=out/'amostras_ptbr'/(character.lower()+'.wav')
    target.parent.mkdir(exist_ok=True)
    sf.write(target,waveforms[0],sr,subtype='PCM_16')
    report.append(dict(character=character,text=text,file=str(target),duration=len(waveforms[0])/sr,generation_seconds=time.monotonic()-tick,engine='Qwen3-TTS 0.6B',reference_seconds=sum(float(x['duration']) for x in profile['clips'])))
    (out/'preview_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report[-1],ensure_ascii=False),flush=True)
faulthandler.cancel_dump_traceback_later()
