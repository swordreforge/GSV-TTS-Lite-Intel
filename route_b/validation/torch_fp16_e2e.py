"""纯 torch fp16 自由生成对照：fp16 权重是否导致 rambling。"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
import torch
import logging
logging.disable(logging.INFO)
import math
from torch.nn import functional as _F


def _eager_sdpa(q, k, v, attn_mask=None, scale=None, **kw):
    d = q.shape[-1]
    s = (q.float() @ k.float().transpose(-1, -2)) / (scale or math.sqrt(d))
    if attn_mask is not None:
        s = s.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else s + attn_mask.float()
    return (torch.softmax(s, dim=-1) @ v.float()).to(q.dtype)
_F.scaled_dot_product_attention = _eager_sdpa

from gsv_tts import TTS

tts = TTS(device="cpu", dtype="float32")
tts.load_gpt_model()
tts.load_sovits_model()
tts.init_language_module("ja")
gpt = list(tts.gpt_models.values())[0]
gpt.t2s_model = gpt.t2s_model.half().float()  # 权重舍入到 fp16（模拟 gguf），计算保持 fp32
print("t2s weights rounded to fp16", flush=True)
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
sf.write("/tmp/opencode/torch_fp16.wav", a, out.samplerate)
print("wrote torch_fp16.wav", round(len(a) / out.samplerate, 1), "s", flush=True)
