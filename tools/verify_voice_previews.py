"""Offline speech and audio checks for the three user-approval samples."""
import os, pathlib, json, re, unicodedata, wave, sys
script=pathlib.Path(__file__).resolve()
out=script.parents[1] if script.parent.name=='tools' else script.parents[2]/'outputs'/'GTA_V_Dublagem_PTBR'
if len(sys.argv)>1: out=pathlib.Path(sys.argv[1]).resolve()
for key in ['TEMP','TMP','TMPDIR','NUMBA_CACHE_DIR']:
    folder=out/'cache'/('numba_asr' if key=='NUMBA_CACHE_DIR' else 'tmp')
    folder.mkdir(parents=True,exist_ok=True)
    os.environ[key]=str(folder)
os.environ['HF_HUB_OFFLINE']='1'
from faster_whisper import WhisperModel
import numpy as np

def tokens(text):
    value=''.join(c for c in unicodedata.normalize('NFD',text.lower()) if unicodedata.category(c)!='Mn')
    return re.findall(r'\w+',value)

def distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]

reports=json.loads((out/'preview_report.json').read_text(encoding='utf-8'))
assert len(reports)==3,'The three samples are required before approval.'
print('Loading local small speech recognizer',flush=True)
snapshot=next((pathlib.Path(r'C:\Users\<user>\.cache\huggingface\hub\models--Systran--faster-whisper-small\snapshots')).iterdir())
model=WhisperModel(str(snapshot),device='cpu',compute_type='int8',cpu_threads=4,local_files_only=True)
checks=[]
for row in reports:
    path=pathlib.Path(row['file'])
    with wave.open(str(path),'rb') as handle:
        assert handle.getnchannels()==1 and handle.getsampwidth()==2
        sr=handle.getframerate()
        audio=np.frombuffer(handle.readframes(handle.getnframes()),dtype='<i2').astype(np.float32)/32768
    assert audio.ndim==1 and np.isfinite(audio).all() and len(audio)>sr
    peak=float(np.max(np.abs(audio)));rms=float(np.sqrt(np.mean(audio**2)))
    clipped=float(np.mean(np.abs(audio)>=0.9999))
    segments,info=model.transcribe(str(path),language='pt',beam_size=5,vad_filter=False,condition_on_previous_text=False)
    transcript=' '.join(s.text.strip() for s in segments)
    expected=tokens(row['text']);recognized=tokens(transcript)
    wer=distance(expected,recognized)/max(1,len(expected))
    # Recognizers can split words ("brinca-deira", "a cabar"); retain WER and
    # also compare characters without word boundaries to distinguish that case.
    cer=distance(list(''.join(expected)),list(''.join(recognized)))/max(1,len(''.join(expected)))
    slang_anchors=set(expected).intersection({'mano','cara'})
    slang_anchor_pass=slang_anchors.issubset(set(recognized))
    result=dict(character=row['character'],expected_text=row['text'],recognized_text=transcript,word_error_rate=round(wer,4),
                duration=len(audio)/sr,sample_rate=sr,rms=rms,peak=peak,clipped_fraction=clipped,
                technical_pass=bool(rms>0.005 and clipped<0.005 and 1<len(audio)/sr<20),
                character_error_rate=round(cer,4),slang_anchor_pass=slang_anchor_pass,
                content_check_pass=(wer<=0.25 or cer<=0.05) and slang_anchor_pass)
    checks.append(result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
(out/'preview_verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
