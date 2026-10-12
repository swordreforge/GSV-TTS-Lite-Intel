"""分段解码：runner长轨迹头120/尾30 + torch干净 восстановите，转写定位泄漏段。"""
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

def piece(toks, name):
    with torch.inference_mode():
        audio, _ = svx.decode(torch.from_numpy(toks).unsqueeze(0).unsqueeze(0).to("xpu"),
                              torch.LongTensor(phones2).unsqueeze(0).to("xpu"),
                              ge.to("xpu"), cuda_graph=False, noise_scale=0.5)
    import soundfile as sf
    a = audio.detach().cpu().numpy().reshape(-1)
    sf.write(f"/tmp/opencode/seg_{name}.wav", a, 32000)
    print(name, round(len(a) / 32000, 1), flush=True)

r = np.fromfile("/tmp/opencode/rate5/d21.bin", np.int64)[1:]
eos = np.where(r == 1024)[0]
r = r[:eos[0]] if len(eos) else r
print("runner len:", len(r), flush=True)
piece(r[:120], "head120")
piece(r[-30:], "tail30")
v = np.fromfile("/tmp/opencode/torch_toks_v0.bin", np.int64)
piece(v, "torch23")
print("SEG DONE", flush=True)
