"""Use the approved Portuguese performance as VC content; change only target voices.

The installed S3Gen .pt checkpoint is loaded directly using the same layout as
the installed multilingual loader. No package or model files are modified.
"""
import os,sys,pathlib,json,time,shutil,hashlib,logging,faulthandler,argparse
parser=argparse.ArgumentParser()
parser.add_argument('--source',type=pathlib.Path,required=True)
parser.add_argument('--output',type=pathlib.Path,required=True)
parser.add_argument('--only',choices=['MICHAEL','FRANKLIN'])
args=parser.parse_args();source=args.source.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
for key,folder in {'HF_HOME':'hf','TRANSFORMERS_CACHE':'hf/transformers','NUMBA_CACHE_DIR':'numba','TEMP':'tmp','TMP':'tmp','TMPDIR':'tmp','MPLCONFIGDIR':'mpl','LIBROSA_CACHE_DIR':'librosa','TORCH_HOME':'torch'}.items():
    path=out/'cache'/folder;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
sys.stdout.reconfigure(encoding='utf-8',line_buffering=True);sys.stderr.reconfigure(encoding='utf-8',line_buffering=True)
faulthandler.enable();faulthandler.dump_traceback_later(60,repeat=True)
logging.basicConfig(level=logging.INFO)
profiles=json.loads((source/'voice_profiles.json').read_text(encoding='utf-8'))
source_report={row['character']:row for row in json.loads((source/'preview_report.json').read_text(encoding='utf-8'))}
guide=source/'amostras_ptbr'/'trevor.wav';guide_hash=hashlib.sha256(guide.read_bytes()).hexdigest()
(out/'amostras_ptbr').mkdir(exist_ok=True);(out/'referencias_preparadas').mkdir(exist_ok=True)
if guide.resolve()!=(out/'amostras_ptbr'/'trevor.wav').resolve():shutil.copy2(guide,out/'amostras_ptbr'/'trevor.wav')
assert hashlib.sha256((out/'amostras_ptbr'/'trevor.wav').read_bytes()).hexdigest()==guide_hash
for who,profile in profiles.items():
    target=out/'referencias_preparadas'/(who.lower()+'_reference.wav')
    if pathlib.Path(profile['reference_file']).resolve()!=target.resolve():shutil.copy2(profile['reference_file'],target)
    profile['reference_file']=str(target)
    profile['state']='approved_revision_2' if who=='TREVOR' else 'awaiting_user_voice_approval'
    profile['conditioning']='approved_Portuguese_guide_plus_original_target_voice'
(out/'voice_profiles.json').write_text(json.dumps(profiles,ensure_ascii=False,indent=2),encoding='utf-8')
approval=dict(revision=3,state='awaiting_Michael_and_Franklin_approval',batch_generation_authorized=False,
              characters={'TREVOR':{'approved':True,'approved_revision':2,'sha256':guide_hash},'MICHAEL':{'approved':False},'FRANKLIN':{'approved':False}})
(out/'approval.json').write_text(json.dumps(approval,ensure_ascii=False,indent=2),encoding='utf-8')
print('STAGE importing local conversion runtime',flush=True)
import torch,librosa,numpy as np,soundfile as sf
from chatterbox.models.s3gen import S3Gen
from chatterbox.vc import ChatterboxVC
torch.set_num_threads(6)
checkpoint=pathlib.Path(r'D:\JARVIS\data\clone-models\hub\models--ResembleAI--chatterbox\snapshots\5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18')
print('STAGE loading local S3Gen conversion weights on CPU, FP32',flush=True)
s3gen=S3Gen()
weights=torch.load(checkpoint/'s3gen.pt',map_location='cpu',weights_only=True)
s3gen.load_state_dict(weights);del weights
s3gen.to(device='cpu',dtype=torch.float32).eval()
model=ChatterboxVC(s3gen,device='cpu')
guide_audio,_=sf.read(guide,dtype='float32')
guide_rms=float(np.sqrt(np.mean(guide_audio**2)))
report=json.loads((out/'preview_report.json').read_text(encoding='utf-8')) if args.only and (out/'preview_report.json').exists() else [dict(source_report['TREVOR'],file=str(out/'amostras_ptbr'/'trevor.wav'),approved=True,unchanged_sha256=guide_hash)]
for who in ['MICHAEL','FRANKLIN']:
    if args.only and who!=args.only:continue
    print('STAGE converting approved Portuguese performance to',who,flush=True)
    tick=time.monotonic();torch.manual_seed(20261116)
    wav=model.generate(audio=str(guide),target_voice_path=profiles[who]['reference_file'])
    audio=wav.squeeze(0).detach().cpu().float().numpy().copy();sr=model.sr
    assert np.isfinite(audio).all() and sr<len(audio)<sr*20
    peak=float(np.max(np.abs(audio)));rms=float(np.sqrt(np.mean(audio**2)))
    gain=min(guide_rms/max(1e-8,rms),0.94/max(1e-8,peak));audio*=gain
    fade=min(int(sr*.008),len(audio)//8);audio[:fade]*=np.linspace(0,1,fade);audio[-fade:]*=np.linspace(1,0,fade)
    target=out/'amostras_ptbr'/(who.lower()+'.wav');sf.write(target,audio,sr,subtype='PCM_16')
    row=dict(character=who,text=source_report['TREVOR']['text'],file=str(target),duration=len(audio)/sr,generation_seconds=time.monotonic()-tick,
             engine='Chatterbox S3Gen voice conversion',conditioning='approved Portuguese content + original English target timbre',
             waveform_decoder_dtype='float32',gain=gain,guide_sha256=guide_hash,approved=False)
    report=[x for x in report if x['character']!=who]+[row]
    report.sort(key=lambda row:['MICHAEL','FRANKLIN','TREVOR'].index(row['character']))
    (out/'preview_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(row,ensure_ascii=False),flush=True)
faulthandler.cancel_dump_traceback_later()
