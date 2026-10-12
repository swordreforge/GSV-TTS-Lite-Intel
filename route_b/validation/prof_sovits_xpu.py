"""SoVITS XPU profile：quant/enc_p/flow/dec 分段计时（dummy 输入，只看耗时分布）。"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# 仓库根（route_b/ 下脚本需回退两级）
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

T = int(os.environ.get("PROF_T", "640"))
tts = TTS(device="xpu", dtype="float32")
tts.load_sovits_model()
sv = list(tts.sovits_models.values())[0].vq_model.eval()

def sync():
    torch.xpu.synchronize()

with torch.inference_mode():
    codes = torch.randint(0, 1000, (1, 1, T)).to("xpu")
    text = torch.randint(0, 700, (1, 11)).to("xpu")
    ge = torch.randn(1, 1024, 1).to("xpu")
    # warmup
    o, _ = sv.decode(codes, text, ge, cuda_graph=False, noise_scale=0)
    sync()

    def t(fn, n=3):
        sync(); t0 = time.time()
        for _ in range(n):
            fn()
        sync()
        return (time.time() - t0) / n * 1000

    print("decode-full: %.0fms" % t(lambda: sv.decode(codes, text, ge, cuda_graph=False, noise_scale=0)), flush=True)
    q = lambda: sv.quantizer.decode(codes)
    print("quant: %.0fms" % t(q), flush=True)
    quant = torch.nn.functional.interpolate(q(), size=T * 2, mode="nearest")
    print("enc_p: %.0fms" % t(lambda: sv.enc_p.infer(quant, text, sv.ge_to512(ge.transpose(2, 1)).transpose(2, 1), 1, False, None, None, None)), flush=True)
    m_p, logs_p, y_mask = sv.enc_p.infer(quant, text, sv.ge_to512(ge.transpose(2, 1)).transpose(2, 1), 1, False, None, None, None)
    z_p = m_p
    print("flow: %.0fms" % t(lambda: sv.flow(z_p, y_mask, ge)), flush=True)
    z = sv.flow(z_p, y_mask, ge)
    print("dec-vocoder: %.0fms" % t(lambda: sv.dec(z * y_mask, g=ge)), flush=True)
print("PROF DONE", flush=True)
