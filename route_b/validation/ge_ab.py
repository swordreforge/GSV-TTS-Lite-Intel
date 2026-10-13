"""A/B：ge 来源 CPU vs XPU，其它全同。定位崩溃变量。"""
import sys, os
sys.path.insert(0, ".")
sys.path.insert(0, "route_b/runner")
import faulthandler
faulthandler.enable()
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
from gsv_tts.TTS import get_phones_and_bert as _gpb

MODE = sys.argv[1]  # "cpu" or "xpu"
UP = "/home/swordreforge/.cache/gsv/uploads/prompt_1791908424222_32uwgy.wav"

tts = TTS(device="cpu", dtype="float32")
tts.load_gpt_model(); tts.load_sovits_model()
tts.init_language_module("ja", "ja")
tts_x = TTS(device="xpu", dtype="float32")
tts_x.load_sovits_model()

if MODE == "cpu":
    _, ge = tts._prepare_sovits_resources(str(tts.default_sovits_path), UP)
else:
    _, ge = tts_x._prepare_sovits_resources(str(tts_x.default_sovits_path), UP)

svx = list(tts_x.sovits_models.values())[0].vq_model.eval()
print(f"[{MODE}] ge device={ge.device}", flush=True)
gex = ge.to("xpu")

Y = np.fromfile("/home/swordreforge/.cache/gsv/prompt_y/00c056656a6a8261.ybin", np.int64)
phones2, _, _, _ = _gpb("こんにちは。", tts.tts_config, "ja")

for k in range(3):
    seq = Y[:10 + k * 5]
    codes = torch.from_numpy(np.ascontiguousarray(seq)).unsqueeze(0).unsqueeze(0).to("xpu")
    ph2 = torch.LongTensor(phones2).unsqueeze(0).to("xpu")
    with torch.inference_mode():
        torch.xpu.synchronize()
        audio, _ = svx.decode(codes, ph2, gex, cuda_graph=False, noise_scale=0.5)
        torch.xpu.synchronize()
    print(f"[{MODE}] DEC{k} OK {tuple(audio.shape)}", flush=True)
print(f"[{MODE}] CLEAN", flush=True)
