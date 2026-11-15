"""Compare previews to the three source voices with a separate local voice encoder.

Cosine scores are a diagnostic, not a perceptual guarantee or an identity probability.
"""
import os,sys,pathlib,json
script=pathlib.Path(__file__).resolve()
out=script.parents[1] if script.parent.name=='tools' else script.parents[2]/'outputs'/'GTA_V_Dublagem_PTBR'
if len(sys.argv)>1: out=pathlib.Path(sys.argv[1]).resolve()
for key in ['TEMP','TMP','TMPDIR','NUMBA_CACHE_DIR']:
    folder=out/'cache'/('numba_speaker' if key=='NUMBA_CACHE_DIR' else 'tmp')
    folder.mkdir(parents=True,exist_ok=True)
    os.environ[key]=str(folder)
import torch,numpy as np,librosa
sys.path.insert(0,r'D:\JARVIS\.venv-tts\Lib\site-packages\chatterbox\models')
from voice_encoder import VoiceEncoder
torch.set_num_threads(4)
encoder=VoiceEncoder()
encoder.load_state_dict(torch.load(r'D:\JARVIS\data\clone-models\hub\models--ResembleAI--chatterbox\snapshots\5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18\ve.pt',map_location='cpu',weights_only=True))
encoder.eval()
profiles=json.loads((out/'voice_profiles.json').read_text(encoding='utf-8'))
def embed(path):
    wav,sr=librosa.load(path,sr=16000,mono=True)
    with torch.inference_mode():
        vector=encoder.embeds_from_wavs([wav],sample_rate=sr)[0]
    return vector/(np.linalg.norm(vector)+1e-8)
references={key:embed(value['reference_file']) for key,value in profiles.items()}
checks=[]
for who in profiles:
    vector=embed(out/'amostras_ptbr'/(who.lower()+'.wav'))
    similarity={key:round(float(np.dot(vector,reference)),5) for key,reference in references.items()}
    result=dict(character=who,cosine_by_reference=similarity,closest_reference=max(similarity,key=similarity.get),
                caveat='Diagnostic cosine similarity, not a perceptual guarantee or probability.')
    checks.append(result)
    print(json.dumps(result),flush=True)
(out/'speaker_verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
