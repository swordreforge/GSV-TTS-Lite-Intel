"""Dump runner 入口：x_ids, bert_feat, y_ids + prefill xy/mask + text/audio pos 表。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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

OUT = Path("/tmp/opencode/t2s_run")
OUT.mkdir(parents=True, exist_ok=True)

tts = TTS(device="cpu", dtype="float32")
tts.load_gpt_model(); tts.load_sovits_model(); tts.init_language_module("ja")
t2s = list(tts.gpt_models.values())[0].t2s_model.eval()

KW = dict(spk_audio_path="examples/laffey.mp3", prompt_audio_path="examples/AnAn.ogg",
          prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
          text=os.environ.get("GSV_UTT_TEXT", "へぇー、ここまでしてくれるんですね。"))

cap = {}
orig_psd = t2s.process_single_data
def wpsd(x, y, bert):
    xy, mk = orig_psd(x, y, bert)
    cap.update(x=x.detach().cpu().clone(), y=y.detach().cpu().clone(),
               bert=bert.detach().cpu().clone(),
               xy=xy.detach().cpu().clone(), mk=mk.detach().cpu().clone())
    return xy, mk
t2s.process_single_data = wpsd
tts.infer(**KW)
print(f"x={tuple(cap['x'].shape)} y={tuple(cap['y'].shape)} bert={tuple(cap['bert'].shape)} xy={tuple(cap['xy'].shape)} mask={tuple(cap['mk'].shape)}", flush=True)

def sv(name, t, dtype=None):
    a = np.ascontiguousarray(t.detach().numpy() if isinstance(t, torch.Tensor) else t)
    if dtype is not None:
        a = a.astype(dtype)
    a.tofile(OUT / (name + ".bin"))

sv("x_ids", cap["x"], np.int64)
sv("y_ids", cap["y"], np.int64)
sv("bert", cap["bert"].numpy())
sv("xy_ref", cap["xy"].numpy())
sv("prompt_mask", cap["mk"].numpy().astype(np.uint8))
# pos 表 + alpha + bert_proj（runner 入口需要）
sv("text_alpha", t2s.ar_text_position.alpha.detach().cpu().numpy())
sv("audio_alpha", t2s.ar_audio_position.alpha.detach().cpu().numpy())
sv("xscale_text", np.array([t2s.ar_text_position.x_scale], np.float32))
sv("xscale_audio", np.array([t2s.ar_audio_position.x_scale], np.float32))
W = t2s.bert_proj.weight.detach().cpu().numpy(); B = t2s.bert_proj.bias.detach().cpu().numpy()
sv("bert_proj_w", W); sv("bert_proj_b", B)
print("RUNNER-IN-DUMP DONE", flush=True)
