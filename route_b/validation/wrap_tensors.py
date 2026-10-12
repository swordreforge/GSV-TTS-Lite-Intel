"""录 wrapper 传给 t2s.infer 的真实张量。"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
import torch
import numpy as np
from gsv_tts import TTS
import logging
logging.disable(logging.INFO)

tts = TTS(device="xpu", dtype="float32")
tts.load_gpt_model()
tts.load_sovits_model()
tts.init_language_module("ja")
t2s = list(tts.gpt_models.values())[0].t2s_model.eval()
orig = t2s.infer


def hooked(x, y, bert, **kw):
    print("infer kwargs:", {k: v for k, v in kw.items()}, flush=True)
    x.cpu().numpy().astype(np.int64).tofile("/tmp/opencode/wrap_x.bin")
    y.cpu().numpy().astype(np.int64).tofile("/tmp/opencode/wrap_y.bin")
    bert.cpu().numpy().tofile("/tmp/opencode/wrap_bert.bin")
    print("saved wrap tensors", tuple(x.shape), tuple(y.shape), tuple(bert.shape), flush=True)
    return orig(x, y, bert, **kw)


t2s.infer = hooked
out = tts.infer(
    spk_audio_path="examples/laffey.mp3",
    prompt_audio_path="examples/AnAn.ogg",
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
    text="こんにちは。",
    top_k=15, top_p=1.0, temperature=1.0, repetition_penalty=1.35,
)
print("done", flush=True)
