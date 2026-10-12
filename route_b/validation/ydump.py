"""抓 Hubert 中间量：wav16k + feats 统计 + y，判分叉点。tag 用于区分进程。"""
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

TAG = os.environ.get("YTAG", "a")
tts = TTS(device="xpu", dtype="float32")
tts.load_gpt_model(); tts.load_sovits_model()
sv = list(tts.sovits_models.values())[0]
hub = CNHubert(tts.cnhubert_path, tts.tts_config)

wav, sr = tts._load_audio("examples/AnAn.ogg")
wav = wav.to(tts.tts_config.device)
wav16k = tts._resample(wav, sr, 16000).mean(dim=0)
print(f"{TAG} wav16k: len {len(wav16k)} mean {wav16k.float().mean():.6f} max {wav16k.float().abs().max():.6f}", flush=True)
w16 = wav16k.detach().cpu().float().numpy()
w16.tofile(f"/tmp/opencode/wav16k_{TAG}.bin")

silence = torch.zeros(int(16000 * 0.3), device=wav16k.device, dtype=wav16k.dtype)
wcat = torch.cat([wav16k, silence]).to(tts.tts_config.dtype)
feats = hub.model(wcat.unsqueeze(0))["last_hidden_state"]
f = feats.detach().cpu().float().numpy()
print(f"{TAG} feats: {f.shape} mean {f.mean():.6f} std {f.std():.6f}", flush=True)
f.tofile(f"/tmp/opencode/feats_{TAG}.bin")

y = tts._get_prompt(hub, sv, "examples/AnAn.ogg")
yy = y.cpu().numpy().astype(np.int64).reshape(-1)
yy.tofile(f"/tmp/opencode/y_{TAG}.bin")
import collections
print(f"{TAG} y uniq {len(set(yy.tolist()))} top3 {collections.Counter(yy.tolist()).most_common(3)}", flush=True)
print("YDUMP DONE", flush=True)
