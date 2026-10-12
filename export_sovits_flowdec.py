"""导出 SoVITS flow+dec 为 OV 静态 IR（F=772 精确 shape），GPU 验证数值+耗时。"""
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

OUT = os.path.expanduser("~/.cache/gsv/ov_models/sovits_flowdec_f772")
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
    m, logs, ym = sv.enc_p.infer(q, text, ge512, 1)
    F = m.shape[-1]
    print(f"frames F={F}", flush=True)
    zp, ymask = m.clone(), ym.clone()

    m2 = copy.deepcopy(sv)
    try:
        m2.dec.remove_weight_norm()
    except Exception as e:
        print("dec remove_wn skipped:", str(e)[:80], flush=True)
    for fl in m2.flow.flows:
        if hasattr(fl, "enc") and hasattr(fl.enc, "in_layers"):
            for l in list(fl.enc.in_layers) + list(fl.enc.res_skip_layers):
                try: torch.nn.utils.remove_weight_norm(l)
                except Exception: pass
            if hasattr(fl.enc, "cond_layer"):
                try: torch.nn.utils.remove_weight_norm(fl.enc.cond_layer)
                except Exception: pass
    m2.eval()
    with torch.inference_mode():
        ref = m2.flow_dec(zp, ymask, ge)
    print("torch ref done", flush=True)

    t0 = time.time()
    class FlowDecW(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.flow = m.flow
            self.dec = m.dec
        def forward(self, zp, ymask, ge):
            z = self.flow(zp, ymask, ge)
            return self.dec(z * ymask, g=ge)
    w = FlowDecW(m2).eval()
    ov_model = ov.convert_model(w, example_input=(zp, ymask, ge))
    print(f"convert: {(time.time()-t0):.1f}s", flush=True)
    ov.save_model(ov_model, os.path.join(OUT, "model.xml"))
    core = ov.Core()
    print("devices:", core.available_devices, flush=True)
    for dev in ["GPU", "CPU"]:
        try:
            cm = core.compile_model(ov_model, dev)
            # warmup
            for _ in range(2):
                cm((zp.numpy(), ymask.numpy(), ge.numpy()))
            t0 = time.time()
            N = 5
            for _ in range(N):
                out = cm((zp.numpy(), ymask.numpy(), ge.numpy()))
            ms = (time.time() - t0) / N * 1000
            got = out[0]
            refn = ref.detach().cpu().numpy() if torch.is_tensor(ref) else ref
            d = np.abs(got - refn).max()
            print(f"OV {dev}: maxdiff={d:.6f} time={ms:.0f}ms (torch cpu flow+dec ~5746ms)", flush=True)
        except Exception as e:
            print(f"OV {dev} FAIL: {str(e)[:200]}", flush=True)
print("FLOWDEC-EXPORT DONE", flush=True)
