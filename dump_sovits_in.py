"""CPU 抓 SoVITS decode 输入存盘，供 XPU/OV 对比复用。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
from gsv_tts import TTS
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

OUT = "/tmp/opencode/sovits_in"
os.makedirs(OUT, exist_ok=True)
tts = TTS(device="cpu", dtype="float32")
tts.load_gpt_model(); tts.load_sovits_model(); tts.init_language_module("ja")
sv = list(tts.sovits_models.values())[0].vq_model.eval()
KW = dict(spk_audio_path="examples/laffey.mp3", prompt_audio_path="examples/AnAn.ogg",
          prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
          text="へぇー、ここまでしてくれるんですね。")
cap = {}
orig_dec = sv.decode
def wdec(codes, text, ge, **kw):
    cap["codes"] = codes.detach().cpu().clone()
    cap["text"] = text.detach().cpu().clone()
    cap["ge"] = ge.detach().cpu().clone()
    return orig_dec(codes, text, ge, **{**kw, "cuda_graph": False, "noise_scale": 0})
sv.decode = wdec
tts.infer(**KW)
for k, v in cap.items():
    np.ascontiguousarray(v.numpy()).tofile(f"{OUT}/{k}.bin")
    print(k, tuple(v.shape), v.dtype, flush=True)
import json
json.dump({k: list(v.shape) for k, v in cap.items()}, open(f"{OUT}/shapes.json", "w"))
print("SOVITS-IN-DUMP DONE", flush=True)
