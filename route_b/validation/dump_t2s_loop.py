"""Dump 真实循环前 5 步的 (xy_pos, logits)，C++ 回放验证 KV 跨步管理。"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# 仓库根（route_b/ 下脚本需回退两级）
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
from pathlib import Path
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

OUT = Path("/tmp/opencode/t2s_loop")
OUT.mkdir(parents=True, exist_ok=True)

tts = TTS(device="cpu", dtype="float32")
tts.load_gpt_model(); tts.load_sovits_model(); tts.init_language_module("ja")
t2s = list(tts.gpt_models.values())[0].t2s_model.eval()

KW = dict(spk_audio_path="examples/laffey.mp3", prompt_audio_path="examples/AnAn.ogg",
          prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
          text="へぇー、ここまでしてくれるんですね。")

steps = []
tr = t2s.t2s_transformer.decode_next_token
def w(x, kc, vc, ln, mk, bi):
    h = tr(x, kc, vc, ln, mk, bi)
    if len(steps) < 5:
        with torch.inference_mode():
            lg = t2s.ar_predict_layer(h[:, -1])
        steps.append((x.detach().cpu().clone(), mk.detach().cpu().clone(),
                      lg.detach().cpu().clone()))
    return h
t2s.t2s_transformer.decode_next_token = w

# 同时抓 prefill 后的 cache（第一次 decode 调用前 kc/vc/ln 状态）
pp = t2s.t2s_transformer.process_prompt
state = {}
def wpp(x, kc, vc, ln, mk):
    r = orig_pp_inner(x, kc, vc, ln, mk)
    if "done" not in state:
        state["done"] = True
        state["kc"] = [t.detach().cpu().clone() for t in kc]
        state["vc"] = [t.detach().cpu().clone() for t in vc]
        state["ln"] = ln.detach().cpu().clone()
    return r
orig_pp_inner = pp
t2s.t2s_transformer.process_prompt = wpp

tts.infer(**KW)
print(f"captured {len(steps)} steps, ln0={state['ln'].tolist()}", flush=True)

def sv(name, t):
    a = np.ascontiguousarray(t.detach().numpy() if isinstance(t, torch.Tensor) else t)
    a.tofile(OUT / (name + ".bin"))

for i, t in enumerate(state["kc"]): sv(f"kc{i}", t)
for i, t in enumerate(state["vc"]): sv(f"vc{i}", t)
sv("ln0", state["ln"].numpy())
for s, (x, mk, lg) in enumerate(steps):
    sv(f"step{s}_x", x)
    sv(f"step{s}_mask", mk.numpy().astype(np.uint8))
    sv(f"step{s}_logits", lg)
print("LOOP-DUMP DONE", flush=True)
