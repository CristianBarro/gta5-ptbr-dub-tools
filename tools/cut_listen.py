"""Transcribe a remixed cutscene channel with timestamps, language auto per segment.  (D:\\JARVIS\\.venv)
usage: cut_listen.py <wav 48k mono> [start_s end_s]"""
import sys, wave, numpy as np
from faster_whisper import WhisperModel
w = wave.open(sys.argv[1]); sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
a, b = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (0, len(x) / sr)
seg = x[int(a * sr): int(b * sr)]; seg = seg[: len(seg) // 3 * 3].reshape(-1, 3).mean(1).astype(np.float32)
m = WhisperModel('large-v3', device='cpu', compute_type='int8')
for s in m.transcribe(seg, beam_size=5, word_timestamps=False, vad_filter=False, multilingual=True)[0]:
    print(f'{a + s.start:6.2f}-{a + s.end:6.2f}  {s.text.strip()}', flush=True)
