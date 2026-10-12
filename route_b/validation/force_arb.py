"""Teacher-forcing 仲裁：runner 轨迹喂给 torch，逐 yo步查 pick 是否在 torch top-15 内。"""
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

TRAJ = np.fromfile("/tmp/opencode/rate5/toks_1.bin", np.int64)  # 完整含t0
print("forcing len:", len(TRAJ), flush=True)

rec = {"step": 0, "rows": []}
orig_sample = TM.sample


def forced(logits, pre_tokens, **kw):
    i = rec["step"]
    probs = _ltp(logits=logits, previous_tokens=pre_tokens, **kw)
    pr = probs.detach().cpu().float().reshape(-1)
    topv, topi = torch.topk(pr, min(15, pr.numel()))
    want = int(TRAJ[i]) if i < len(TRAJ) else int(torch.argmax(pr))
    pe = float(pr[1024]) if pr.numel() > 1024 else 0.0
    rec["rows"].append((want, int(topi[0]), bool((topi == want).any()),
                        pe, float(pr[want]) if want < pr.numel() else 0.0, int(torch.argmax(pr))))
    rec["step"] += 1
    return torch.tensor([[want]]), probs


TM.sample = forced

tts = TTS(device="cpu", dtype="float32")
tts.load_gpt_model()
tts.load_sovits_model()
tts.init_language_module("ja")
# 只关心 GPT 部分：直接调 t2s infer（会连带跑 SoVITS？不——t2s.infer 只跑 GPT）
t2s = list(tts.gpt_models.values())[0].t2s_model.eval()
import gsv_tts.TTS as TTSM
from gsv_tts.TTS import get_phones_and_bert as _gpb
phones2, _, bert2, _ = _gpb("こんにちは。", tts.tts_config, "ja")
gpt_m, prompt, phones1, bert1 = tts._prepare_gpt_resources(
    str(tts.default_gpt_path), "examples/AnAn.ogg",
    "ちが……ちがう。レイア、貴様は間違っている。", "ja")
x = torch.LongTensor(phones1 + phones2).unsqueeze(0)
bert = torch.cat([bert1, bert2]).unsqueeze(0)
with torch.inference_mode():
    out = t2s.infer(x, prompt, bert, top_k=15, top_p=1.0, temperature=1.0,
                    repetition_penalty=1.35)
print("forced steps:", rec["step"], flush=True)
outside = [(i, w, float(p)) for i, (w, a, inside, pe, p, am) in enumerate(rec["rows"]) if not inside]
print("runner-pick-outside-torch-top15:", len(outside), outside[:10], flush=True)
agree = sum(1 for (w, a, _, _, _, am) in rec["rows"] if w == am)
print(f"argmax-agree: {agree}/{len(rec['rows'])}", flush=True)
pe = [pe for (_, _, _, pe, _, _) in rec["rows"]]
print("torch P(EOS) max:", max(pe), "at step:", int(np.argmax(pe)), flush=True)
np.array(pe).tofile("/tmp/opencode/peos.bin")
print("FORCE-ARB DONE", flush=True)
