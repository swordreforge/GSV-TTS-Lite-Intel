"""B-y 截短实验：专用提取（必 B）→ 全/头半/尾半 → runner 同 seed 比长度。"""
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
import collections

tts = TTS(device="xpu", dtype="float32")
tts.load_gpt_model(); tts.load_sovits_model()
hub = CNHubert(tts.cnhubert_path, tts.tts_config)
sv = list(tts.sovits_models.values())[0]
y = tts._get_prompt(hub, sv, "examples/AnAn.ogg").cpu().numpy().astype(np.int64).reshape(-1)
c = collections.Counter(y.tolist())
print("fresh y: uniq", len(c), "sil239", c.get(239, 0), flush=True)
y.tofile("/tmp/opencode/btrunc/y_full.bin")
n = len(y)
y[:n // 2].tofile("/tmp/opencode/btrunc/y_first.bin")
y[n // 2:].tofile("/tmp/opencode/btrunc/y_last.bin")
print("saved", flush=True)
