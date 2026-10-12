"""同进程：冷提 y → GPU 计算 → 新实例再提。测顺序效应。"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
import math
import torch
from torch.nn import functional as _F
_F.scaled_dot_product_attention = lambda q, k, v, attn_mask=None, scale=None, **kw: (
    torch.softmax((q.float() @ k.float().transpose(-1, -2)) / (scale or math.sqrt(q.shape[-1])), dim=-1)
    @ v.float()).to(q.dtype)
import logging
logging.disable(logging.INFO)
import numpy as np
from gsv_tts import TTS
from gsv_tts.GPT_SoVITS.Featurizer import CNHubert

tts = TTS(device="xpu", dtype="float32")
tts.load_gpt_model(); tts.load_sovits_model()
sv = list(tts.sovits_models.values())[0]
hub1 = CNHubert(tts.cnhubert_path, tts.tts_config)
y1 = tts._get_prompt(hub1, sv, "examples/AnAn.ogg")
a1 = y1.cpu().numpy().astype(np.int64).reshape(-1)
print("cold y done", flush=True)
# GPU 计算：大 matmul + SDPA
for _ in range(5):
    q = torch.randn(4, 16, 512, 32, device="xpu")
    _F.scaled_dot_product_attention(q, q, q)
torch.xpu.synchronize()
print("warmed", flush=True)
hub2 = CNHubert(tts.cnhubert_path, tts.tts_config)
y2 = tts._get_prompt(hub2, sv, "examples/AnAn.ogg")
a2 = y2.cpu().numpy().astype(np.int64).reshape(-1)
n = min(len(a1), len(a2))
print("cold-vs-warm diff:", int((a1[:n] != a2[:n]).sum()), "/", n, flush=True)
import collections
print("cold top3:", collections.Counter(a1.tolist()).most_common(3), flush=True)
print("warm top3:", collections.Counter(a2.tolist()).most_common(3), flush=True)
