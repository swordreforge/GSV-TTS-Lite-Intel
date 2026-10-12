"""SoVITS XPU 跑分：sovits-only 加载到 XPU，跑 enc_p+flow_dec，对比 CPU 数值+耗时。"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# 仓库根（route_b/ 下脚本需回退两级）
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
import torch, time
import numpy as np
import json
from gsv_tts import TTS
import logging
logging.disable(logging.INFO)

D = "/tmp/opencode/sovits_in"
shapes = json.load(open(f"{D}/shapes.json"))
codes = torch.from_numpy(np.fromfile(f"{D}/codes.bin", np.int64).reshape(shapes["codes"]))
text = torch.from_numpy(np.fromfile(f"{D}/text.bin", np.int64).reshape(shapes["text"]))
ge = torch.from_numpy(np.fromfile(f"{D}/ge.bin", np.float32).reshape(shapes["ge"]))
print("inputs:", shapes, flush=True)

tts = TTS(device="xpu", dtype="float32")
tts.load_sovits_model()
svx = list(tts.sovits_models.values())[0].vq_model.eval()
ttsc = TTS(device="cpu", dtype="float32")
ttsc.load_sovits_model()
svc = list(ttsc.sovits_models.values())[0].vq_model.eval()

def run(sv, dev, tag):
    c, t, g = codes.to(dev), text.to(dev), ge.to(dev)
    with torch.inference_mode():
        if dev == "xpu":
            torch.xpu.synchronize()
        t0 = time.time()
        q = torch.nn.functional.interpolate(sv.quantizer.decode(c).float(), size=sv.quantizer.decode(c).shape[-1]*2, mode="nearest")
        g512 = sv.ge_to512(g.transpose(2, 1)).transpose(2, 1)
        m, logs, ym = sv.enc_p.infer(q, t, g512, 1)
        z = sv.flow(m, ym, g)
        o = sv.dec(z * ym, g=g)
        if dev == "xpu":
            torch.xpu.synchronize()
        ms = (time.time() - t0) * 1000
    print(f"{tag}: m={tuple(m.shape)} audio={tuple(o.shape)} time={ms:.0f}ms", flush=True)
    return o.detach().cpu().float().numpy()

ref = run(svc, "cpu", "CPU ")
for _ in range(2):
    got = run(svx, "xpu", "XPU ")
d = np.abs(got - ref)
print(f"XPU-vs-CPU: maxdiff={d.max():.6f} mean={d.mean():.2e}", flush=True)
print("SOVITS-XPU DONE", flush=True)
