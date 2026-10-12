"""终极仲裁：同输入 torch greedy vs runner greedy，逐 token 对齐（中文输入）。"""
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

import math
from torch.nn import functional as _F


def _eager_sdpa(q, k, v, attn_mask=None, scale=None, **kw):
    d = q.shape[-1]
    s = (q.float() @ k.float().transpose(-1, -2)) / (scale or math.sqrt(d))
    if attn_mask is not None:
        s = s.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else s + attn_mask.float()
    return (torch.softmax(s, dim=-1) @ v.float()).to(q.dtype)
_F.scaled_dot_product_attention = _eager_sdpa

D = "/tmp/opencode/zh_an"
x = torch.from_numpy(np.fromfile(f"{D}/x_ids.bin", np.int64).reshape(1, -1))
y = torch.fromfile(f"{D}/y.bin", np.int64) if False else torch.from_numpy(np.fromfile(f"{D}/y.bin", np.int64).reshape(1, -1))
bert = torch.from_numpy(np.fromfile(f"{D}/bert.bin", np.float32).reshape(1, -1, 1024))
print("shapes:", tuple(x.shape), tuple(y.shape), tuple(bert.shape), flush=True)

tts = TTS(device="cpu", dtype="float32")
tts.load_gpt_model()
t2s = list(tts.gpt_models.values())[0].t2s_model.eval()
with torch.inference_mode():
    ref = t2s.infer(x, y, bert, top_k=1, top_p=1.0, temperature=1.0, repetition_penalty=1.35)
ref = ref[0, 0].cpu().numpy() if ref.dim() == 3 else ref.cpu().numpy().reshape(-1)
print("torch greedy len:", len(ref), "head:", ref[:10].tolist(), flush=True)
ref.tofile(f"{D}/torch_greedy.bin")
print("TORCH DONE", flush=True)
