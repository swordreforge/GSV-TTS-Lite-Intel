"""多句对照：短句多条，同 prompt 同 seed，量回声+取前 60 token 备解码试听。"""
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
from gsv_tts.TTS import get_phones_and_bert

CASES = [
    ("zh1", "谢谢。", "zh"),
    ("zh2", "再见。", "zh"),
    ("zh3", "今天天气不错。", "zh"),
    ("ja1", "ありがとう。", "ja"),
]
OUT = "/tmp/opencode/sent_comp"
os.makedirs(OUT, exist_ok=True)

tts = TTS(device="cpu", dtype="float32")
tts.load_gpt_model(); tts.load_sovits_model()
tts.init_language_module("zh", "ja")
sovits_m, ge = tts._prepare_sovits_resources(str(tts.default_sovits_path), "examples/laffey.mp3")
gpt_m, prompt, phones1, bert1 = tts._prepare_gpt_resources(
    str(tts.default_gpt_path), "examples/AnAn.ogg",
    "ちが……ちがう。レイア、貴様は間違っている。", "ja")
for name, text, lang in CASES:
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    phones2, _, bert2, _ = get_phones_and_bert(text, tts.tts_config, lang)
    x = torch.LongTensor(phones1 + phones2).unsqueeze(0)
    bert = torch.cat([bert1, bert2]).unsqueeze(0)
    np.ascontiguousarray(x.numpy().astype(np.int64).reshape(-1)).tofile(f"{d}/x_ids.bin")
    np.ascontiguousarray(prompt.numpy().astype(np.int64).reshape(-1)).tofile(f"{d}/y.bin")
    np.ascontiguousarray(bert.numpy()).tofile(f"{d}/bert.bin")
    np.array([x.shape[1], prompt.shape[1]], dtype=np.int64).tofile(f"{d}/xlen_ylen.bin")
    np.ascontiguousarray(np.array(phones2, dtype=np.int64)).tofile(f"{d}/phones2.bin")
    print(name, text, "phones2:", phones2, flush=True)
ge.cpu().numpy().tofile(f"{OUT}/ge.bin")
print("PREP DONE", flush=True)
