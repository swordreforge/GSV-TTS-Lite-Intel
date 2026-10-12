"""同进程变线程数提 y，看多样性（免子进程）。"""
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

tts = TTS(device="cpu", dtype="float32")
tts.load_gpt_model(); tts.load_sovits_model()
sv = list(tts.sovits_models.values())[0]
base = None
for nt in [1, 2, 4, 8, 16]:
    torch.set_num_threads(nt)
    hub = CNHubert(tts.cnhubert_path, tts.tts_config)
    y = tts._get_prompt(hub, sv, "examples/AnAn.ogg").cpu().numpy().astype(np.int64).reshape(-1)
    y.tofile(f"/tmp/opencode/y_thr{nt}.bin")
    if base is None:
        base = y
        print(f"threads={nt}: baseline len {len(y)}", flush=True)
    else:
        n = min(len(base), len(y))
        print(f"threads={nt}: diff {(base[:n] != y[:n]).sum()}/{n}", flush=True)
    del hub
print("THREADVAR DONE", flush=True)
