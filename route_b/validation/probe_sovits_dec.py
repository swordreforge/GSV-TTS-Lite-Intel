"""SoVITS dec 优化探针：fp32稳态 / fp16数值+速度 / 变shape JIT代价。"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
import math, time
import torch
from torch.nn import functional as _F
_F.scaled_dot_product_attention = lambda q, k, v, attn_mask=None, scale=None, **kw: (
    torch.softmax((q.float() @ k.float().transpose(-1, -2)) / (scale or math.sqrt(q.shape[-1])), dim=-1)
    @ v.float()).to(q.dtype)
import logging
logging.disable(logging.INFO)
from gsv_tts import TTS

tts = TTS(device="xpu", dtype="float32")
tts.load_sovits_model()
sv = list(tts.sovits_models.values())[0].vq_model.eval()
print("dec class:", type(sv.dec).__name__, flush=True)


def sync():
    torch.xpu.synchronize()


def make(T):
    with torch.inference_mode():
        codes = torch.randint(0, 1000, (1, 1, T)).to("xpu")
        text = torch.randint(0, 700, (1, 11)).to("xpu")
        ge = torch.randn(1, 1024, 1).to("xpu")
        quant = torch.nn.functional.interpolate(
            sv.quantizer.decode(codes).float(), size=T * 2, mode="nearest")
        m_p, logs_p, y_mask = sv.enc_p.infer(
            quant, text, sv.ge_to512(ge.transpose(2, 1)).transpose(2, 1), 1, False, None, None, None)
        z = sv.flow(m_p, y_mask, ge)
        return z * y_mask, ge


def t(fn, n=3):
    sync(); t0 = time.time()
    for _ in range(n):
        fn()
    sync()
    return (time.time() - t0) / n * 1000


with torch.inference_mode():
    z640, ge640 = make(640)
    # warmup
    ref = sv.dec(z640, g=ge640)
    sync()
    print("dec-fp32-steady-T640: %.0fms" % t(lambda: sv.dec(z640, g=ge640)), flush=True)
    # fp16
    sv.dec.half()
    z16, ge16 = z640.half(), ge640.half()
    o16 = sv.dec(z16, g=ge16)
    sync()
    print("dec-fp16-steady-T640: %.0fms" % t(lambda: sv.dec(z16, g=ge16)), flush=True)
    d = (o16.float().cpu() - ref.cpu()).abs()
    print("fp16-vs-fp32: maxdiff=%.6f mean=%.2e" % (d.max(), d.mean()), flush=True)
    sv.dec.float()
    # 变 shape：新 T 首次耗时（JIT 代价）
    for T in (320, 480, 700):
        zT, geT = make(T)
        sync(); t0 = time.time()
        sv.dec(zT, g=geT)
        sync(); first = (time.time() - t0) * 1000
        steady = t(lambda z=zT, g=geT: sv.dec(z, g=g), n=2)
        print(f"dec-T{T}: first={first:.0f}ms steady={steady:.0f}ms", flush=True)
print("PROBE DONE", flush=True)
