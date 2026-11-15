"""Revise the three approval samples: separate speaker identity from English prosody.

Keep the talker in BF16 for VRAM, decode the waveform in FP32, and never start a batch.
"""
import os,sys,pathlib,json,time,logging,faulthandler,argparse
parser=argparse.ArgumentParser()
parser.add_argument('--source',type=pathlib.Path,required=True)
parser.add_argument('--output',type=pathlib.Path,required=True)
parser.add_argument('--only',choices=['MICHAEL','FRANKLIN','TREVOR'])
args=parser.parse_args()
source=args.source.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
for key,folder in {'HF_HOME':'hf','NUMBA_CACHE_DIR':'numba','TEMP':'tmp','TMP':'tmp','TMPDIR':'tmp','MPLCONFIGDIR':'matplotlib','TRITON_CACHE_DIR':'triton','LIBROSA_CACHE_DIR':'librosa'}.items():
    path=out/'cache'/folder;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8',line_buffering=True)
faulthandler.enable();faulthandler.dump_traceback_later(60,repeat=True)
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
print('STAGE importing local runtime',flush=True)
import torch,librosa,numpy as np,soundfile as sf
from scipy.signal import butter,sosfiltfilt
torch.set_num_threads(4)
profiles=json.loads((source/'voice_profiles.json').read_text(encoding='utf-8'))
(out/'referencias_preparadas').mkdir(exist_ok=True)
for who,profile in profiles.items():
    audio,sr=librosa.load(profile['reference_file'],sr=24000,mono=True)
    if who=='MICHAEL':
        # Remove DC/rumble and high-frequency reference hiss without changing pitch.
        audio=sosfiltfilt(butter(3,[55,7500],btype='bandpass',fs=sr,output='sos'),audio).astype(np.float32)
    path=out/'referencias_preparadas'/(who.lower()+'_reference.wav')
    sf.write(path,audio,sr,subtype='PCM_16')
    profile['reference_file']=str(path)
    profile['conditioning']='speaker_embedding_only'
    profile['state']='awaiting_user_voice_approval'
(out/'voice_profiles.json').write_text(json.dumps(profiles,ensure_ascii=False,indent=2),encoding='utf-8')
(out/'approval.json').write_text(json.dumps({'state':'awaiting_user_voice_approval','batch_generation_authorized':False,'revision':2,'feedback_addressed':['foreign_accent','Michael_hiss']},indent=2),encoding='utf-8')
print('STAGE loading model',torch.cuda.mem_get_info(),flush=True)
from qwen_tts import Qwen3TTSModel
model=Qwen3TTSModel.from_pretrained(r'D:\JARVIS\data\qwen-models\Qwen3-TTS-12Hz-0.6B-Base',device_map='cuda:0',torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
model.model.speech_tokenizer.model.to(dtype=torch.float32)
assert next(model.model.speech_tokenizer.model.parameters()).dtype==torch.float32
print('STAGE ready, waveform decoder FP32',flush=True)
texts={row['character']:row['text'] for row in json.loads((source/'preview_report.json').read_text(encoding='utf-8'))}
texts['FRANKLIN']='A gente precisa sair daqui, cara. Entra no carro e deixa que eu dirijo.'
report=json.loads((out/'preview_report.json').read_text(encoding='utf-8')) if args.only and (out/'preview_report.json').exists() else []
for who,text in texts.items():
    if args.only and who!=args.only: continue
    tick=time.monotonic();torch.manual_seed(20261115)
    print('STAGE synthesizing',who,'speaker embedding only',flush=True)
    wavs,sr=model.generate_voice_clone(text=text,language='Portuguese',ref_audio=profiles[who]['reference_file'],x_vector_only_mode=True,
                                     temperature=0.55,top_k=40,repetition_penalty=1.08,non_streaming_mode=True,max_new_tokens=250)
    raw=np.asarray(wavs[0],dtype=np.float32)
    assert np.isfinite(raw).all() and len(raw)>sr
    peak=float(np.max(np.abs(raw)))
    gain=min(1.0,0.94/max(peak,1e-8));audio=raw*gain
    # Short edge fades prevent artificial clicks, without rescaling time or pitch.
    fade=min(int(sr*0.008),len(audio)//8)
    audio[:fade]*=np.linspace(0,1,fade);audio[-fade:]*=np.linspace(1,0,fade)
    path=out/'amostras_ptbr'/(who.lower()+'.wav');path.parent.mkdir(exist_ok=True)
    sf.write(path,audio,sr,subtype='PCM_16')
    decoded,_=sf.read(path,dtype='float32');active=decoded[np.abs(decoded)>0.05]
    grid=float(np.mean((active.view(np.uint32)&0xffff)==0))
    result=dict(character=who,text=text,file=str(path),duration=len(audio)/sr,generation_seconds=time.monotonic()-tick,engine='Qwen3-TTS 0.6B',
                conditioning='speaker_embedding_only',waveform_decoder_dtype='float32',gain=gain,bfloat16_grid_fraction=grid,
                reference_seconds=sum(float(x['duration']) for x in profiles[who]['clips']))
    report=[row for row in report if row['character']!=who]+[result]
    report.sort(key=lambda row:list(texts).index(row['character']))
    (out/'preview_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
faulthandler.cancel_dump_traceback_later()
