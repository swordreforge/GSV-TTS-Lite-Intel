"""Bucket 概念验证：导出 Fb=1024/Tb=64 的 IR，pad 输入跑完 crop，对比精确 shape torch ref。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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

FB, TB = 1024, 64
OUT = os.path.expanduser("~/.cache/gsv/ov_models/sovits_bucket")
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
    print(f"exact: F={Fq} T={Tt}, bucket: Fb={FB} Tb={TB}", flush=True)
    assert Fq <= FB and Tt <= TB

    # 精确 ref（torch）
    rm, rlogs, rym = sv.enc_p.infer(q, text, ge512, 1)
    ref_audio = sv.flow_dec(rm, rym, ge)

    # pad 输入
    qp = torch.zeros(1, q.shape[1], FB); qp[:, :, :Fq] = q
    tp = torch.zeros(1, TB, dtype=text.dtype); tp[:, :Tt] = text
    # enc_p wrapper（deepcopy 去 weight_norm 残留）
    m2 = copy.deepcopy(sv)
    try: m2.dec.remove_weight_norm()
    except Exception: pass
    for fl in m2.flow.flows:
        if hasattr(fl, "enc") and hasattr(fl.enc, "in_layers"):
            for l in list(fl.enc.in_layers) + list(fl.enc.res_skip_layers) + ([fl.enc.cond_layer] if hasattr(fl.enc, "cond_layer") else []):
                try: torch.nn.utils.remove_weight_norm(l)
                except Exception: pass
    class EncPW(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, y, t, g, ym, tm):
            m, logs, _ = self.m.enc_p.infer(y, t, g, 1, y_mask=ym, text_mask=tm)
            return m, logs
    class FlowDecW(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.flow = m.flow; self.dec = m.dec
        def forward(self, zp, ymask, ge):
            return self.dec(self.flow(zp, ymask, ge) * ymask, g=ge)
    we, wf = EncPW(m2).eval(), FlowDecW(m2).eval()
    g512p = ge512  # [1,512,1] 无需 pad
    ymp = torch.zeros(1, 1, FB); ymp[:, :, :Fq] = 1
    tmp = torch.zeros(1, 1, TB); tmp[:, :, :Tt] = 1
    ov_enc = ov.convert_model(we, example_input=(qp, tp, g512p, ymp, tmp))
    ov.save_model(ov_enc, os.path.join(OUT, "enc_b1024x64.xml"))
    ov_fd = ov.convert_model(wf, example_input=(torch.zeros(1, 192, FB), torch.ones(1, 1, FB), ge))
    ov.save_model(ov_fd, os.path.join(OUT, "fd_b1024.xml"))
    core = ov.Core()
    ce = core.compile_model(ov_enc, "GPU", {"INFERENCE_PRECISION_HINT": "f32"})
    cf = core.compile_model(ov_fd, "GPU", {"INFERENCE_PRECISION_HINT": "f32"})
    r = ce((qp.numpy(), tp.numpy(), g512p.numpy(), ymp.numpy(), tmp.numpy()))
    me, logse = r[0], r[1]
    yme = ymp.numpy()
    # torch 同 pad 输入 parallelism（定位 OV 导出 vs 概念）
    with torch.inference_mode():
        pm, plogs, _ = m2.enc_p.infer(qp, tp, g512p, 1, y_mask=ymp, text_mask=tmp)
    print(f"OVpad-vs-torchpad: m_maxdiff={np.abs(me - pm.numpy()).max():.6f}", flush=True)
    print(f"enc pad-run: m shape {me.shape}", flush=True)
    F = Fq
    dm = np.abs(me[:, :, :F] - rm.numpy()).max()
    print(f"enc crop-vs-exact: m_maxdiff={dm:.6f}", flush=True)
    # flow-dec：用 enc 的 pad 输出直接喂（真 bucket 链路），crop 对比
    Audio = 640
    out = cf((me, yme, ge.numpy()))[0]
    # 注意 yme pad 区为 0（text/ymask 在 pad 区？yme 由 enc pad-run 产生，pad 区 ymask？
    d = np.abs(out[:, :, :F * Audio] - ref_audio.numpy())
    print(f"e2e pad-chain crop-vs-exact: maxdiff={d.max():.6f} mean={d.mean():.2e} n_gt1e-3={(d > 1e-3).sum()}", flush=True)
print("BUCKET-PROOF DONE", flush=True)
