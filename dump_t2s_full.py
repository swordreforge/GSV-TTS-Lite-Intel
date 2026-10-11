"""Dump 全模型单步 decode 所需：24 层权重 + 真实抓到的单步输入/输出。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import torch
import numpy as np
from gsv_tts import TTS
import logging
logging.disable(logging.INFO)

# 必须在 TTS 构造之前 patch：load 时 initialize_runtime 就会 warmup 跑 SDPA
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

OUT = Path("/tmp/opencode/t2s_full")
OUT.mkdir(parents=True, exist_ok=True)

tts = TTS(device="cpu", dtype="float32")
tts.load_gpt_model(); tts.load_sovits_model(); tts.init_language_module("ja")
t2s = list(tts.gpt_models.values())[0].t2s_model.eval()

# 抓真实单步输入（ attention mask 语义照 XPU 管线：True=attend）
cap = {}
tr = t2s.t2s_transformer.decode_next_token
def w(x, kc, vc, ln, mk, bi):
    if not cap:
        cap.update(x=x.detach().cpu().clone(), kc=[t.detach().cpu().clone() for t in kc],
                   vc=[t.detach().cpu().clone() for t in vc], ln=ln.detach().cpu().clone(),
                   mk=mk.detach().cpu().clone(), bi=bi.detach().cpu().clone())
    return tr(x, kc, vc, ln, mk, bi)
t2s.t2s_transformer.decode_next_token = w
KW = dict(spk_audio_path="examples/laffey.mp3", prompt_audio_path="examples/AnAn.ogg",
          prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
          text="へぇー、ここまでしてくれるんですね。")
tts.infer(**KW)
K = cap["kc"][0].shape[2]
print("captured K =", K, flush=True)

with torch.inference_mode():
    ref_h = t2s.t2s_transformer.decode_next_token(
        cap["x"], cap["kc"], cap["vc"], cap["ln"], cap["mk"], cap["bi"])
    ref_logits = t2s.ar_predict_layer(ref_h[:, -1])
    # 逐层输出，供 C++ 二分定位（每层只碰自己的 cache 切片）
    layer_outs = []
    x_ = cap["x"]
    for li, blk in enumerate(t2s.t2s_transformer.blocks):
        x_ = blk.decode_next_token(
            x_, cap["kc"][li].clone(), cap["vc"][li].clone(),
            cap["mk"], cap["ln"], cap["bi"])
        layer_outs.append(x_.clone())

def sv(name, t):
    a = np.ascontiguousarray(t.detach().numpy() if isinstance(t, torch.Tensor) else t)
    a.tofile(OUT / (name + ".bin"))
    (OUT / (name + ".shape")).write_text(",".join(map(str, a.shape)))

sv("x", cap["x"]); sv("ln", cap["ln"].numpy())
for i, t in enumerate(cap["kc"]): sv(f"kc{i}", t)
for i, t in enumerate(cap["vc"]): sv(f"vc{i}", t)
sv("mask", cap["mk"].numpy().astype(np.uint8))
sv("ref_logits", ref_logits)
for i, o in enumerate(layer_outs):
    sv(f"ly{i}", o)
for i, blk in enumerate(t2s.t2s_transformer.blocks):
    sv(f"b{i}.qkv_w", blk.qkv.weight)
    sv(f"b{i}.qkv_b", blk.qkv.bias)
    sv(f"b{i}.out_w", blk.out_proj.weight)
    sv(f"b{i}.out_b", blk.out_proj.bias)
    sv(f"b{i}.n1_w", blk.norm1.weight); sv(f"b{i}.n1_b", blk.norm1.bias)
    sv(f"b{i}.mlp1_w", blk.mlp[0].weight); sv(f"b{i}.mlp1_b", blk.mlp[0].bias)
    sv(f"b{i}.mlp2_w", blk.mlp[2].weight); sv(f"b{i}.mlp2_b", blk.mlp[2].bias)
    sv(f"b{i}.n2_w", blk.norm2.weight); sv(f"b{i}.n2_b", blk.norm2.bias)
pw = t2s.ar_predict_layer.weight
sv("predict_w", pw)
(OUT / "K.txt").write_text(str(K))
print("dumped", flush=True)
