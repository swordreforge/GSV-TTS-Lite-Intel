"""导出 SoVITS enc_p 为 OV 静态 IR（精确 shape），GPU-f32 验证数值+耗时。"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# 仓库根（route_b/ 下脚本需回退两级）
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
import torch, time, copy
import numpy as np
import openvino as ov
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

TAG = sys.argv[1] if len(sys.argv) > 1 else "f772_t36"
OUT = os.path.expanduser(f"~/.cache/gsv/ov_models/sovits_encp_{TAG}")
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

with torch.inference_mode():
    codes, text, ge = cap["codes"], cap["text"], cap["ge"]
    q = torch.nn.functional.interpolate(sv.quantizer.decode(codes).float(), size=sv.quantizer.decode(codes).shape[-1]*2, mode="nearest")
    ge512 = sv.ge_to512(ge.transpose(2, 1)).transpose(2, 1)
    Fq, Tt = q.shape[-1], text.shape[1]
    print(f"enc_p in: q={tuple(q.shape)} text={tuple(text.shape)} ge512={tuple(ge512.shape)}", flush=True)

    class EncPW(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, y, t, g):
            m, logs, ymask = self.m.infer(y, t, g, 1)
            return m, logs, ymask
    w = EncPW(copy.deepcopy(sv.enc_p)).eval()
    with torch.inference_mode():
        rm, rlogs, rym = w(q, text, ge512)
    print("torch ref done", flush=True)

    t0 = time.time()
    ov_model = ov.convert_model(w, example_input=(q, text, ge512))
    print(f"convert: {(time.time()-t0):.1f}s", flush=True)
    ov.save_model(ov_model, os.path.join(OUT, "model.xml"))
    core = ov.Core()
    for dev, cfg in [("GPU", {"INFERENCE_PRECISION_HINT": "f32"}), ("CPU", {})]:
        cm = core.compile_model(ov_model, dev, cfg)
        inp = (q.numpy(), text.numpy(), ge512.numpy())
        for _ in range(2):
            cm(inp)
        t0 = time.time(); N = 5
        for _ in range(N):
            out = cm(inp)
        ms = (time.time() - t0) / N * 1000
        dm = np.abs(out[0] - rm.numpy()).max()
        dl = np.abs(out[1] - rlogs.numpy()).max()
        print(f"OV {dev}: m_maxdiff={dm:.6f} logs_maxdiff={dl:.6f} time={ms:.0f}ms (torch cpu enc_p ~140ms)", flush=True)
print("ENCP-EXPORT DONE", flush=True)
