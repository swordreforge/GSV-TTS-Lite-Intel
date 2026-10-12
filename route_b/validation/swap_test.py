"""拆分测试：CPU-GPT + XPU-y。短=跟y走（y定行为），长=跟计算走。"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
import math
import torch
from torch.nn import functional as _F


def _eager_sdpa(q, k, v, attn_mask=None, scale=None, **kw):
    d = q.shape[-1]
    s = (q.float() @ k.float().transpose(-1, -2)) / (scale or math.sqrt(d))
    if attn_mask is not None:
        s = s.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else s + attn_mask.float()
    return (torch.softmax(s, dim=-1) @ v.float()).to(q.dtype)
_F.scaled_dot_product_attention = _eager_sdpa

import numpy as np
from gsv_tts import TTS
import logging
logging.disable(logging.INFO)
from gsv_tts.TTS import get_phones_and_bert as _gpb

import os as _os
tts = TTS(device=_os.environ.get("SWAP_DEV", "cpu"), dtype="float32")
tts.load_gpt_model()
tts.load_sovits_model()
tts.init_language_module("ja")
t2s = list(tts.gpt_models.values())[0].t2s_model.eval()
phones2, _, bert2, _ = _gpb("こんにちは。", tts.tts_config, "ja")
gpt_m, prompt_cpu, phones1, bert1 = tts._prepare_gpt_resources(
    str(tts.default_gpt_path), "examples/AnAn.ogg",
    "ちが……ちがう。レイア、貴様は間違っている。", "ja")
# 换上 XPU-y（同长度 184，同文本对应）
ypath = _os.environ.get("SWAP_Y", "/tmp/opencode/y_xpu.bin")
prompt_xpu = torch.from_numpy(np.fromfile(ypath, np.int64)).reshape(1, -1).to(tts.tts_config.device)
print("cpu-y head:", prompt_cpu.reshape(-1)[:6].tolist(), flush=True)
print("xpu-y head:", prompt_xpu.reshape(-1)[:6].tolist(), flush=True)
dev = tts.tts_config.device
x = torch.LongTensor(phones1 + phones2).unsqueeze(0).to(dev)
bert = torch.cat([bert1, bert2]).unsqueeze(0).to(dev)
with torch.inference_mode():
    out = t2s.infer(x, prompt_xpu, bert, top_k=15, top_p=1.0, temperature=1.0,
                    repetition_penalty=1.35)
o = out[0, 0].cpu().numpy() if out.dim() == 3 else out.cpu().numpy().reshape(-1)
print("CPU-GPT + XPU-y len:", len(o), "head:", o[:8].tolist(), flush=True)
o.astype(np.int64).tofile("/tmp/opencode/swap_toks.bin")
print("SWAP DONE", flush=True)
