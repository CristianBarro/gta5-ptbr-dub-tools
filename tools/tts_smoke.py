"""Local GPU synthesis proof using the already installed Qwen runtime and weights."""
import os
import pathlib
import csv
import json
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
ROOT = pathlib.Path(__file__).resolve().parents[2]
TARGET = ROOT / "outputs" / "GTA_V_Dublagem_PTBR"
TARGET.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(TARGET / "cache")

import torch
import soundfile as sf
from faster_qwen3_tts import FasterQwen3TTS

torch.set_num_threads(4)
print(json.dumps({"stage": "gpu", "torch": torch.__version__, "cuda": torch.cuda.is_available(), "free_bytes": torch.cuda.mem_get_info()[0]}, ensure_ascii=False), flush=True)
started = time.monotonic()
model = FasterQwen3TTS.from_pretrained(r"D:\JARVIS\data\qwen-models\Qwen3-TTS-12Hz-0.6B-Base", device="cuda", dtype=torch.float16, local_files_only=True)
print(json.dumps({"stage": "loaded", "seconds": time.monotonic()-started}), flush=True)
catalog = ROOT / "outputs" / "GTA_V_Dialogue_Map_v0.1"
samples = list(csv.DictReader((catalog / "amostras_originais.csv").open(encoding="utf-8-sig")))
report = []
for item in samples:
    name = item["character"].lower()
    output = TARGET / "amostras_ptbr" / (name + ".wav")
    output.parent.mkdir(exist_ok=True)
    reference = catalog / "amostras" / item["filename"]
    torch.manual_seed(2401)
    tick = time.monotonic()
    wavs, sr = model.generate_voice_clone(text=item["portuguese"], language="Portuguese", ref_audio=str(reference), ref_text=item["english"],
                                         temperature=0.7, top_k=50, repetition_penalty=1.05, non_streaming_mode=True, max_new_tokens=360)
    sf.write(output, wavs[0], sr, subtype="PCM_16")
    result = dict(character=item["character"], text_pt=item["portuguese"], source_text=item["english"], output=str(output), sample_rate=sr,
                  seconds=time.monotonic()-tick, generated_duration=len(wavs[0])/sr, source_duration=float(item["duration_seconds"]))
    report.append(result)
    (TARGET / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False),flush=True)
