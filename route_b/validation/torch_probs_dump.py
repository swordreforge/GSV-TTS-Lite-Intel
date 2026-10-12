"""torch 侧 probs dump：teacher-forcing runner 轨迹，只存 loop 步（与 runner DUMP_PROBS 对齐）。"""
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

import math
from torch.nn import functional as _F


def _eager_sdpa(q, k, v, attn_mask=None, scale=None, **kw):
    d = q.shape[-1]
    s = (q.float() @ k.float().transpose(-1, -2)) / (scale or math.sqrt(d))
    if attn_mask is not None:
        s = s.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else s + attn_mask.float()
    return (torch.softmax(s, dim=-1) @ v.float()).to(q.dtype)
_F.scaled_dot_product_attention = _eager_sdpa

import gsv_tts.GPT_SoVITS.GPT.t2s_model as TM
from gsv_tts.GPT_SoVITS.GPT.utils import logits_to_probs as _ltp
from gsv_tts.TTS import get_phones_and_bert as _gpb

TRAJ = np.fromfile(os.environ.get("FORCE_TRAJ", "/tmp/opencode/rate5/toks_1.bin"), np.int64)
OUT = open(os.environ.get("FORCE_PROBS_OUT", "/tmp/opencode/torch_probs.bin"), "wb")
rec = {"step": 0}


def forced(logits, pre_tokens, **kw):
    i = rec["step"]
    probs = _ltp(logits=logits, previous_tokens=pre_tokens, **kw)
    pr = probs.detach().cpu().float().reshape(-1).numpy()
    if i == 0:
        open(os.environ.get("FORCE_PROBS0_OUT", "/tmp/opencode/torch_probs0.bin"), "wb").write(pr.astype(np.float32).tobytes())
    else:
        OUT.write(pr.astype(np.float32).tobytes())
    want = int(TRAJ[i]) if i < len(TRAJ) else int(np.argmax(pr))
    rec["step"] += 1
    return torch.tensor([[want]], device=logits.device), probs


TM.sample = forced

tts = TTS(device=os.environ.get("FORCE_DEV", "cpu"), dtype="float32")
tts.load_gpt_model()
tts.load_sovits_model()
tts.init_language_module("ja")
t2s = list(tts.gpt_models.values())[0].t2s_model.eval()
phones2, _, bert2, _ = _gpb("こんにちは。", tts.tts_config, "ja")
gpt_m, prompt, phones1, bert1 = tts._prepare_gpt_resources(
    str(tts.default_gpt_path), "examples/AnAn.ogg",
    "ちが……ちがう。レイア、貴様は間違っている。", "ja")
dev = os.environ.get("FORCE_DEV", "cpu")
x = torch.LongTensor(phones1 + phones2).unsqueeze(0).to(dev)
bert = torch.cat([bert1, bert2]).unsqueeze(0).to(dev)
prompt = prompt.to(dev)
with torch.inference_mode():
    out = t2s.infer(x, prompt, bert, top_k=15, top_p=1.0, temperature=1.0,
                    repetition_penalty=1.35)
OUT.close()
print("forced steps:", rec["step"], flush=True)
print("TORCH-PROBS DONE", flush=True)
