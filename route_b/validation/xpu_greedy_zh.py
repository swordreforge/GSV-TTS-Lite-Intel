"""XPU torch greedy vs CPU torch greedy（同输入）：XPU 后端是否移位。"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
import torch
import numpy as np
from gsv_tts import TTS
import logging
logging.disable(logging.INFO)

D = "/tmp/opencode/zh_an"
x = torch.from_numpy(np.fromfile(f"{D}/x_ids.bin", np.int64).reshape(1, -1)).to("xpu")
y = torch.from_numpy(np.fromfile(f"{D}/y.bin", np.int64).reshape(1, -1)).to("xpu")
bert = torch.from_numpy(np.fromfile(f"{D}/bert.bin", np.float32).reshape(1, -1, 1024)).to("xpu")
tts = TTS(device="xpu", dtype="float32")
tts.load_gpt_model()
t2s = list(tts.gpt_models.values())[0].t2s_model.eval()
with torch.inference_mode():
    ref = t2s.infer(x, y, bert, top_k=1, top_p=1.0, temperature=1.0, repetition_penalty=1.35)
o = ref[0, 0].cpu().numpy() if ref.dim() == 3 else ref.cpu().numpy().reshape(-1)
o.astype(np.int64).tofile("/tmp/opencode/xpu_greedy.bin")
cpu = np.fromfile(f"{D}/torch_greedy.bin", np.int64)
n = min(len(o), len(cpu))
d = np.where(o[:n] != cpu[:n])[0]
print("xpu len:", len(o), "cpu len:", len(cpu), flush=True)
print("XPU-vs-CPU greedy:", "IDENTICAL" if len(d) == 0 and len(o) == len(cpu) else f"first diff at {d[0] if len(d) else 'len-only'}", flush=True)
