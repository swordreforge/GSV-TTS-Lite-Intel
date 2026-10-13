"""seed 生效证明：同输入×3 seed，轨迹不同、长度都短。"""
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

OUT = "/tmp/opencode/seedproof"
os.makedirs(OUT, exist_ok=True)
tts = TTS(device="cpu", dtype="float32")
tts.load_gpt_model(); tts.load_sovits_model(); tts.init_language_module("ja")
gpt_m, prompt, phones1, bert1 = tts._prepare_gpt_resources(
    str(tts.default_gpt_path), "examples/AnAn.ogg",
    "ちが……ちがう。レイア、貴様は間違っている。", "ja")
# 好 y 资产覆盖（和 serving 一致）
goody = np.fromfile(os.path.expanduser("~/.cache/gsv/prompt_y/AnAn.good.ybin"), np.int64)
prompt = torch.from_numpy(goody.reshape(1, -1))
phones2, _, bert2, _ = _gpb("こんにちは。", tts.tts_config, "ja")
x = torch.LongTensor(phones1 + phones2).unsqueeze(0)
bert = torch.cat([bert1, bert2]).unsqueeze(0)
np.ascontiguousarray(x.numpy().astype(np.int64).reshape(-1)).tofile(f"{OUT}/x_ids.bin")
np.ascontiguousarray(prompt.numpy().astype(np.int64).reshape(-1)).tofile(f"{OUT}/y.bin")
np.ascontiguousarray(bert.numpy()).tofile(f"{OUT}/bert.bin")
np.array([x.shape[1], prompt.shape[1]], dtype=np.int64).tofile(f"{OUT}/xlen_ylen.bin")
print("prep done", tuple(x.shape), flush=True)
