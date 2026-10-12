"""子进程提 y（供优选：多进程抽签，进程内重复无意义）。输出 y .bin + 统计。"""
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
import collections
from gsv_tts import TTS
from gsv_tts.GPT_SoVITS.Featurizer import CNHubert

# 用法：extract_y.py <audio> <out_bin> [device]
audio = sys.argv[1]
out = sys.argv[2]
dev = sys.argv[3] if len(sys.argv) > 3 else "xpu"
tts = TTS(device=dev, dtype="float32")
tts.load_gpt_model()
tts.load_sovits_model()
hub = CNHubert(tts.cnhubert_path, tts.tts_config)
sv = list(tts.sovits_models.values())[0]
y = tts._get_prompt(hub, sv, audio).cpu().numpy().astype(np.int64).reshape(-1)
y.tofile(out)
c = collections.Counter(y.tolist())
import json
print(json.dumps({"len": len(y), "uniq": len(c),
                  "sil239_frac": round(c.get(239, 0) / len(y), 4),
                  "top3": c.most_common(3)}), flush=True)
