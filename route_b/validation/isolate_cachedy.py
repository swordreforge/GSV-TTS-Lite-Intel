"""镜像服务端：tts(cpu) 出 phones，tts_x(xpu) 出 svx+ge；cached y + 真 phones2 → decode ×5。"""
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

UP = "/home/swordreforge/.cache/gsv/uploads/prompt_1791908424222_32uwgy.wav"
print("A: load cpu tts", flush=True)
tts = TTS(device="cpu", dtype="float32")
tts.load_gpt_model(); tts.load_sovits_model()
tts.init_language_module("ja", "ja")
print("B: load xpu tts", flush=True)
tts_x = TTS(device="xpu", dtype="float32")
tts_x.load_sovits_model()
print("C: prepare sovits (xpu)", flush=True)
sovits_m, ge = tts_x._prepare_sovits_resources(str(tts_x.default_sovits_path), UP)
svx = sovits_m.vq_model.eval()
print("D: ge device", ge.device, tuple(ge.shape), flush=True)
gex = ge.to("xpu")
print("E: gex ready", flush=True)

Y = np.fromfile("/home/swordreforge/.cache/gsv/prompt_y/00c056656a6a8261.ybin", np.int64)
print("F: cached y len", len(Y), flush=True)

phones2, _, _, _ = _gpb("こんにちは。", tts.tts_config, "ja")
print("G: phones2 n", len(phones2), flush=True)

for k in range(5):
    seq = Y[: 10 + k * 3]
    codes = torch.from_numpy(np.ascontiguousarray(seq)).unsqueeze(0).unsqueeze(0).to("xpu")
    ph2 = torch.LongTensor(phones2).unsqueeze(0).to("xpu")
    with torch.inference_mode():
        torch.xpu.synchronize()
        audio, _ = svx.decode(codes, ph2, gex, cuda_graph=False, noise_scale=0.5)
        torch.xpu.synchronize()
    print(f"H{k}: OK codes={len(seq)} audio={tuple(audio.shape)}", flush=True)
print("ISOLATION-CLEAN", flush=True)
