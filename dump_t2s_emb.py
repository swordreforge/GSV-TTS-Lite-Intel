"""Dump embedding+position 入口：emb 表 + pe_cache + x_scale + 前 5 步 (token, pe_idx, xy_ref)。"""
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

OUT = Path("/tmp/opencode/t2s_emb")
OUT.mkdir(parents=True, exist_ok=True)

tts = TTS(device="cpu", dtype="float32")
tts.load_gpt_model(); tts.load_sovits_model(); tts.init_language_module("ja")
t2s = list(tts.gpt_models.values())[0].t2s_model.eval()

KW = dict(spk_audio_path="examples/laffey.mp3", prompt_audio_path="examples/AnAn.ogg",
          prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
          text="へぇー、ここまでしてくれるんですね。")

# 抓 x_len（infer 的 x 参数即文本 token）
x_len_box = {}
orig_infer = t2s.infer
def w_infer(x, y, bert_feature, **kw):
    x_len_box["x_len"] = x.shape[1]
    return orig_infer(x, y, bert_feature, **kw)
t2s.infer = w_infer

# 抓 embedding 输入 token（顺序即调用序）+ 每次调用时的 kv_len
emb_toks = []
kv_at_emb = []
orig_emb_fw = t2s.ar_audio_embedding.forward
def wemb(y):
    emb_toks.append(y.detach().cpu().clone())
    # 找 active bucket 的 kv_len
    kv = None
    for b in t2s.cuda_graph_buckets.get(1, []):
        try:
            v = b.kv_cache_len.item() if torch.is_tensor(b.kv_cache_len) else int(b.kv_cache_len)
        except Exception:
            continue
        if v > 0:
            kv = v
            break
    kv_at_emb.append(kv)
    return orig_emb_fw(y)
t2s.ar_audio_embedding.forward = wemb

tts.infer(**KW)
print(f"emb calls={len(emb_toks)}, x_len={x_len_box.get('x_len')}, kv_at_emb={kv_at_emb[:7]}", flush=True)

def sv(name, a):
    np.ascontiguousarray(a).tofile(OUT / (name + ".bin"))

with torch.inference_mode():
    W = t2s.ar_audio_embedding.word_embeddings.weight.detach().cpu().numpy()
    sv("emb", W)
    print("emb", W.shape, flush=True)
    pe = t2s.ar_audio_position.pe.detach().cpu()          # [1,4000,512]
    alpha = float(t2s.ar_audio_position.alpha.detach().cpu().item())
    xs = float(t2s.ar_audio_position.x_scale)
    sv("alpha", np.array([alpha], dtype=np.float32))
    sv("xscale", np.array([xs], dtype=np.float32))
    pec = (alpha * pe).transpose(0, 1).detach().cpu().numpy()  # [4000,1,512]
    sv("pe_cache", np.ascontiguousarray(pec.reshape(4000, 512)))
    print("pe_cache", pec.shape, "alpha=", alpha, "x_scale=", xs, flush=True)
    x_len = x_len_box["x_len"]
    sv("x_len", np.array([x_len], dtype=np.int64))
    toks = [int(t.flatten()[0]) for t in emb_toks[1:6]]
    sv("tokens", np.array(toks, dtype=np.int64))
    # emb call 0 是 prefill；decode 从 call 1 开始，kv=[263..267]
    idxs = [k - x_len for k in kv_at_emb[1:6]]
    sv("pe_idx", np.array(idxs, dtype=np.int64))
    print("tokens=", toks, "pe_idx=", idxs, flush=True)
    # 用公式重算 xy，与 loop dump 的 step{s}_x 对照
    for s in range(5):
        t = toks[s]
        ye = W[t] * xs + pec[idxs[s], 0]
        ref = np.fromfile(f"/tmp/opencode/t2s_loop/step{s}_x.bin", np.float32)
        d = np.abs(ye - ref).max()
        print(f"  step{s} recompute-vs-loopdump maxdiff={d:.6f}", flush=True)
        sv(f"xy{s}", ye.astype(np.float32))
print("EMB-DUMP DONE", flush=True)
