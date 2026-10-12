"""同 token 不同 noise_scale A/B（电音感对照）。"""
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
from gsv_tts.TTS import get_phones_and_bert as _gpb

tts = TTS(device="cpu", dtype="float32")
tts.load_gpt_model(); tts.load_sovits_model(); tts.init_language_module("ja")
_, ge = tts._prepare_sovits_resources(str(tts.default_sovits_path), "examples/laffey.mp3")
phones2, _, _, _ = _gpb("こんにちは。", tts.tts_config, "ja")
tts_x = TTS(device="xpu", dtype="float32")
tts_x.load_sovits_model()
svx = list(tts_x.sovits_models.values())[0].vq_model.eval()
toks = np.fromfile("/tmp/opencode/rate5/toks_1.bin", np.int64)[1:]
eos = np.where(toks == 1024)[0]
toks = toks[:eos[0]] if len(eos) else toks
import soundfile as sf
with torch.inference_mode():
    c = torch.from_numpy(toks).unsqueeze(0).unsqueeze(0).to("xpu")
    p = torch.LongTensor(phones2).unsqueeze(0).to("xpu")
    g = ge.to("xpu")
    a0, _ = svx.decode(c, p, g, cuda_graph=False, noise_scale=0)
    a5, _ = svx.decode(c, p, g, cuda_graph=False, noise_scale=0.5)
w0 = a0.detach().cpu().numpy().reshape(-1)
w5 = a5.detach().cpu().numpy().reshape(-1)
sf.write("/tmp/opencode/ab_noise0.wav", w0, 32000)
sf.write("/tmp/opencode/ab_noise05.wav", w5, 32000)
d = np.abs(w0[:len(w5)] - w5)
print("len:", len(w0), len(w5), "maxdiff=%.4f" % d.max(), flush=True)
print("AB DONE", flush=True)
