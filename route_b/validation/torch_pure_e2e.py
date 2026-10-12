"""纯 torch TTS.infer 端到端（无 C++ runner），输出给 ASR 裁判。"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
import torch
import logging
logging.disable(logging.INFO)
from gsv_tts import TTS

tts = TTS(device="xpu", dtype="float32")
out = tts.infer(
    spk_audio_path="examples/laffey.mp3",
    prompt_audio_path="examples/AnAn.ogg",
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
    text="こんにちは。",
    top_k=15, top_p=1.0, temperature=1.0, repetition_penalty=1.35,
)
import soundfile as sf
import numpy as np
a = np.asarray(out.audio_data, dtype=np.float32).reshape(-1)
print("sr:", out.samplerate, flush=True)
sf.write("/tmp/opencode/torch_pure.wav", a, out.samplerate)
print("wrote torch_pure.wav", len(a.reshape(-1)) / 32000, "s", flush=True)
