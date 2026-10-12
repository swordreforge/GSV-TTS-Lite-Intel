"""SoVITS 结构探针：config + decode 输入输出形状 + 分阶段耗时 + 参数量。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, time
import numpy as np
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
print("version:", sv.version, "is_v2pro:", sv.is_v2pro, flush=True)
print("inter/hidden/filter:", sv.inter_channels, sv.hidden_channels, sv.filter_channels, flush=True)
print("n_heads/n_layers/kernel:", sv.n_heads, sv.n_layers, sv.kernel_size, flush=True)
print("upsample_rates:", sv.upsample_rates, "samples_per_frame:", sv.samples_per_frame, flush=True)
print("resblock:", sv.resblock, "kernels:", sv.resblock_kernel_sizes, flush=True)
print("gin_channels:", sv.gin_channels, flush=True)
n = sum(p.numel() for p in sv.parameters())
print(f"params: {n/1e6:.1f}M", flush=True)

KW = dict(spk_audio_path="examples/laffey.mp3", prompt_audio_path="examples/AnAn.ogg",
          prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
          text="へぇー、ここまでしてくれるんですね。")

cap = {}
orig_dec = sv.decode
def wdec(codes, text, ge, **kw):
    cap["codes"] = codes.detach().cpu().clone()
    cap["text"] = text.detach().cpu().clone()
    cap["ge"] = ge.detach().cpu().clone()
    cap["kw"] = kw
    with torch.inference_mode():
        t0 = time.time()
        o, attn = orig_dec(codes, text, ge, **{**kw, "cuda_graph": False, "noise_scale": 0})
        t1 = time.time()
    cap["o"] = o.detach().cpu().clone()
    print(f"decode(no-graph,noise=0): codes={tuple(codes.shape)} text={tuple(text.shape)} ge={tuple(ge.shape)} -> audio={tuple(o.shape)} attn={tuple(attn.shape)} time={(t1-t0)*1000:.0f}ms", flush=True)
    return o, attn
sv.decode = wdec
tts.infer(**KW)

# 分阶段计时（enc_p / flow / dec）
with torch.inference_mode():
    codes, text, ge = cap["codes"], cap["text"], cap["ge"]
    for name, fn in [
        ("quant+interp", lambda: torch.nn.functional.interpolate(sv.quantizer.decode(codes).float(), size=sv.quantizer.decode(codes).shape[-1]*2, mode="nearest")),
    ]:
        t0 = time.time(); r = fn(); print(f"  {name}: {tuple(r.shape)} {(time.time()-t0)*1000:.0f}ms", flush=True)
    q = torch.nn.functional.interpolate(sv.quantizer.decode(codes).float(), size=sv.quantizer.decode(codes).shape[-1]*2, mode="nearest")
    ge512 = sv.ge_to512(ge.transpose(2, 1)).transpose(2, 1) if sv.is_v2pro else ge
    t0 = time.time(); m, logs, ym = sv.enc_p.infer(q, text, ge512, 1); t1 = time.time()
    print(f"  enc_p: m={tuple(m.shape)} {(t1-t0)*1000:.0f}ms", flush=True)
    zp = m
    t0 = time.time(); z = sv.flow(zp, ym, ge); t1 = time.time()
    print(f"  flow: {tuple(z.shape)} {(t1-t0)*1000:.0f}ms", flush=True)
    t0 = time.time(); o = sv.dec(z * ym, g=ge); t1 = time.time()
    print(f"  dec: {tuple(o.shape)} {(t1-t0)*1000:.0f}ms", flush=True)
print("SOVITS-PROBE DONE", flush=True)
