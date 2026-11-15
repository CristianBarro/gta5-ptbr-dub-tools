"""Record pitch against original references; pitch is not a gender classifier."""
import os,sys,pathlib,json
root=pathlib.Path(sys.argv[1]).resolve()
for key in ['TEMP','TMP','TMPDIR','NUMBA_CACHE_DIR']:
    path=root/'cache'/('pitch_numba' if key=='NUMBA_CACHE_DIR' else 'tmp');path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
import librosa,numpy as np
profiles=json.loads((root/'voice_profiles.json').read_text(encoding='utf-8'))
result=[]
def estimate(path):
    wav,sr=librosa.load(path,sr=16000,mono=True)
    f0,voiced,prob=librosa.pyin(wav,fmin=65,fmax=350,sr=sr,frame_length=1024,hop_length=256)
    usable=f0[np.isfinite(f0)&voiced&(prob>.65)]
    if len(usable)<5:usable=f0[np.isfinite(f0)&voiced]
    return {'median_hz':float(np.median(usable)),'p10_hz':float(np.quantile(usable,.1)),'p90_hz':float(np.quantile(usable,.9)),'voiced_frames':len(usable)}
for who,profile in profiles.items():
    reference=estimate(profile['reference_file']);render=estimate(root/'amostras_ptbr'/(who.lower()+'.wav'))
    row={'character':who,'reference':reference,'render':render,'median_difference_semitones':float(12*np.log2(render['median_hz']/reference['median_hz']))}
    result.append(row);print(json.dumps(row),flush=True)
(root/'pitch_diagnostics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
