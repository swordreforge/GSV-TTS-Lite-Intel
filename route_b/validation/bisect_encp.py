"""enc_p 三段定位：ssl分支 / text分支 / mrte+后级，pad shape 下 OV vs torch。"""
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

core = ov.Core()
with torch.inference_mode():
    codes, text, ge = cap["codes"], cap["text"], cap["ge"]
    q = torch.nn.functional.interpolate(sv.quantizer.decode(codes).float(), size=sv.quantizer.decode(codes).shape[-1]*2, mode="nearest")
    ge512 = sv.ge_to512(ge.transpose(2, 1)).transpose(2, 1)
    Fq, Tt = q.shape[-1], text.shape[1]
    FB, TB = 1024, 64
    qp = torch.zeros(1, q.shape[1], FB); qp[:, :, :Fq] = q
    tp = torch.zeros(1, TB, dtype=text.dtype); tp[:, :Tt] = text
    ymp = torch.zeros(1, 1, FB); ymp[:, :, :Fq] = 1
    tmp = torch.zeros(1, 1, TB); tmp[:, :, :Tt] = 1
    m2 = copy.deepcopy(sv.enc_p)

    class SSLW(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, y, ym):
            y = self.m.ssl_proj(y * ym) * ym
            return self.m.encoder_ssl(y * ym, ym)
    class TEXTW(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, t, tm):
            x = self.m.text_embedding(t).transpose(1, 2)
            return self.m.encoder_text(x * tm, tm)
    class MRTEW(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, y, ym, t, tm, g):
            return self.m.mrte(y, ym, t, tm, g, None)
    class ENC2W(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, y, ym):
            y = self.m.encoder2(y * ym, ym)
            stats = self.m.proj(y) * ym
            m, logs = torch.split(stats, self.m.out_channels, dim=1)
            return m, logs
    class TAILW(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, y, ym, t, tm, g):
            y = self.m.mrte(y, ym, t, tm, g, None)
            y = self.m.encoder2(y * ym, ym)
            stats = self.m.proj(y) * ym
            m, logs = torch.split(stats, self.m.out_channels, dim=1)
            return m, logs

    # torch refs（pad 输入）
    ys = SSLW(m2)(qp * ymp, ymp)
    ts = TEXTW(m2)(tp, tmp)
    mrte_r = MRTEW(m2)(ys, ymp, ts, tmp, ge512)
    tm_r, log_r = ENC2W(m2)(mrte_r * ymp, ymp)

    for name, mod, inp in [
        ("ssl", SSLW(m2), (qp * ymp, ymp)),
        ("mrte", MRTEW(m2), (ys, ymp, ts, tmp, ge512)),
        ("enc2proj", ENC2W(m2), (mrte_r * ymp, ymp)),
    ]:
        t0 = time.time()
        om = ov.convert_model(mod.eval(), example_input=inp)
        cm = core.compile_model(om, "GPU", {"INFERENCE_PRECISION_HINT": "f32"})
        r = cm([a.numpy() if torch.is_tensor(a) else a for a in inp])
        ms = (time.time() - t0)
        outs = [r[i] for i in range(len(r))]
        refs = {"ssl": [ys.numpy()], "mrte": [mrte_r.numpy()], "enc2proj": [tm_r.numpy(), log_r.numpy()]}[name]
        ds = [f"{np.abs(o - rf).max():.6f}" for o, rf in zip(outs, refs)]
        print(f"{name}: maxdiff={ds} convert+compile+run={ms:.0f}s", flush=True)
print("BISECT DONE", flush=True)
