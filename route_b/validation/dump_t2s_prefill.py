"""Dump prefill (process_prompt) 的真实输入/输出，供 C++ 对照。"""
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

# 必须在 TTS 构造之前 patch（warmup 即跑 SDPA；CPU 原生不支持 bool mask）
import math
from torch.nn import functional as _F
def _eager_sdpa(q, k, v, attn_mask=None, scale=None, **kw):
    d = q.shape[-1]
    s = (q.float() @ k.float().transpose(-1, -2)) / (scale or math.sqrt(d))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            s = s.masked_fill(~attn_mask, float("-inf"))
        else:
            s = s + attn_mask.float()
    return (torch.softmax(s, dim=-1) @ v.float()).to(q.dtype)
_F.scaled_dot_product_attention = _eager_sdpa

OUT = Path("/tmp/opencode/t2s_prefill")
OUT.mkdir(parents=True, exist_ok=True)

tts = TTS(device="cpu", dtype="float32")
tts.load_gpt_model(); tts.load_sovits_model(); tts.init_language_module("ja")
t2s = list(tts.gpt_models.values())[0].t2s_model.eval()

KW = dict(spk_audio_path="examples/laffey.mp3", prompt_audio_path="examples/AnAn.ogg",
          prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
          text="へぇー、ここまでしてくれるんですね。")

# 抓 process_prompt 的输入：hook 住，存 xy_pos / prompt mask / 初始 cache
cap = {}
orig_pp = t2s.t2s_transformer.process_prompt
def wpp(x, kc, vc, ln, mk):
    if not cap:
        cap.update(x=x.detach().cpu().clone(),
                   kc0=[t.detach().cpu().clone() for t in kc],
                   vc0=[t.detach().cpu().clone() for t in vc],
                   ln=ln.detach().cpu().clone(), mk=mk.detach().cpu().clone())
    return orig_pp(x, kc, vc, ln, mk)
t2s.t2s_transformer.process_prompt = wpp
tts.infer(**KW)
L = cap["x"].shape[1]
K = cap["kc0"][0].shape[2]
print(f"prefill L={L} K={K}", flush=True)

def sv(name, t):
    a = np.ascontiguousarray(t.detach().numpy() if isinstance(t, torch.Tensor) else t)
    a.tofile(OUT / (name + ".bin"))

sv("x", cap["x"]); sv("ln", cap["ln"].numpy()); sv("mask", cap["mk"].numpy().astype(np.uint8))
for i, t in enumerate(cap["kc0"]): sv(f"kc{i}", t)
for i, t in enumerate(cap["vc0"]): sv(f"vc{i}", t)
# torch 参照：跑一遍 prefill 得 hidden + 末位 logits
with torch.inference_mode():
    h = t2s.t2s_transformer.process_prompt(
        cap["x"], [t.clone() for t in cap["kc0"]], [t.clone() for t in cap["vc0"]],
        cap["ln"].clone(), cap["mk"])
    logits = t2s.ar_predict_layer(h[:, -1])
sv("ref_h", h); sv("ref_logits", logits)
# 逐层参照（C++ 二分用）：单 block 的 process_prompt 只碰自己那层 cache
lbs = t2s.t2s_transformer.blocks
x_ = cap["x"]
for li, blk in enumerate(lbs):
    x_ = blk.process_prompt(
        x_, cap["kc0"][li].clone(), cap["vc0"][li].clone(), cap["mk"])
    sv(f"ly{li}", x_)
print("dumped prefill", tuple(h.shape), flush=True)
