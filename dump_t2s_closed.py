"""Dump closed-loop greedy (top_k=1) 全程：y + 每步(token, pe_idx, xy, logits, V, suppressed)。"""
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

import gsv_tts.GPT_SoVITS.GPT.t2s_model as TM

OUT = Path("/tmp/opencode/t2s_closed")
OUT.mkdir(parents=True, exist_ok=True)

tts = TTS(device="cpu", dtype="float32")
tts.load_gpt_model(); tts.load_sovits_model(); tts.init_language_module("ja")
t2s = list(tts.gpt_models.values())[0].t2s_model.eval()

KW = dict(spk_audio_path="examples/laffey.mp3", prompt_audio_path="examples/AnAn.ogg",
          prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
          text=os.environ.get("GSV_UTT_TEXT", "へぇー、ここまでしてくれるんですね。"),
          top_k=1, top_p=1.0, temperature=1.0, repetition_penalty=1.35, noise_scale=0)

box = {}
orig_infer = t2s.infer
def w_infer(x, y, bert_feature, **kw):
    box["x_len"] = x.shape[1]
    box["y"] = y.detach().cpu().clone()
    return orig_infer(x, y, bert_feature, **kw)
t2s.infer = w_infer

# sample 调用序：call0=prefill(V=1024恒suppress)，call_s=decode s-1 之后
sample_calls = []
orig_sample = TM.sample
def wsample(logits, prev=None, **kw):
    with torch.inference_mode():
        r = orig_sample(logits, prev, **kw)
    sample_calls.append((logits.detach().cpu().clone(),
                         prev.detach().cpu().clone() if prev is not None else None,
                         int(r[0].flatten()[0])))
    return r
TM.sample = wsample

# decode 输入
dec_inputs = []
tr = t2s.t2s_transformer.decode_next_token
def wdec(x, kc, vc, ln, mk, bi):
    dec_inputs.append((x.detach().cpu().clone(), mk.detach().cpu().clone()))
    return tr(x, kc, vc, ln, mk, bi)
t2s.t2s_transformer.decode_next_token = wdec

# prefill 后 cache
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

# embedding token 序（call0=prefill y）
emb_toks, kv_at_emb = [], []
orig_emb_fw = t2s.ar_audio_embedding.forward
def wemb(y):
    emb_toks.append(y.detach().cpu().clone())
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

# prefill 入口同轮捕获（prompt token 跨轮不稳定，必须同轮）
psd = {}
orig_psd = t2s.process_single_data
def wpsd(x, y, bert):
    xy, mk = orig_psd(x, y, bert)
    psd.update(x=x.detach().cpu().clone(), y=y.detach().cpu().clone(),
               bert=bert.detach().cpu().clone(),
               xy=xy.detach().cpu().clone(), mk=mk.detach().cpu().clone())
    return xy, mk
t2s.process_single_data = wpsd
# SoVITS 输入/音频捕获
svcap = {}
svm = list(tts.sovits_models.values())[0].vq_model.eval()
orig_svdec = svm.decode
def wsvdec(codes, text, ge, **kw):
    svcap.update(codes=codes.detach().cpu().clone(), text=text.detach().cpu().clone(),
                 ge=ge.detach().cpu().clone())
    o, attn = orig_svdec(codes, text, ge, **{**kw, "cuda_graph": False, "noise_scale": 0})
    svcap["audio"] = o.detach().cpu().clone()
    return o, attn
svm.decode = wsvdec
r = tts.infer(**KW)
N = len(dec_inputs)
print(f"dec_steps={N}, sample_calls={len(sample_calls)}, x_len={box['x_len']}, EOS={t2s.EOS}", flush=True)

def sv(name, a):
    np.ascontiguousarray(a).tofile(OUT / (name + ".bin"))

with torch.inference_mode():
    for i, t in enumerate(state["kc"]): sv(f"kc{i}", t.numpy())
    for i, t in enumerate(state["vc"]): sv(f"vc{i}", t.numpy())
    sv("ln0", state["ln"].numpy())
    sv("y", box["y"].numpy().astype(np.int64))
    sv("EOS", np.array([t2s.EOS], dtype=np.int64))
    sv("x_len", np.array([box["x_len"]], dtype=np.int64))
    W = t2s.ar_audio_embedding.word_embeddings.weight.detach().cpu().numpy()
    sv("emb", W)
    pe = t2s.ar_audio_position.pe.detach().cpu()
    alpha = float(t2s.ar_audio_position.alpha.detach().cpu().item())
    xs = float(t2s.ar_audio_position.x_scale)
    sv("xscale", np.array([xs], dtype=np.float32))
    pec = (alpha * pe).transpose(0, 1).detach().cpu().numpy()
    sv("pe_cache", np.ascontiguousarray(pec.reshape(4000, 512)))
    # 每 decode 步 s：token t_s = emb_toks[s+1]；sample call s+1 输出 t_{s+1}
    toks = [int(emb_toks[s + 1].flatten()[0]) for s in range(N)]
    # 最后一个 sample 输出（第 N+1 个 call 若存在，否则 -1）
    tout = [c[2] for c in sample_calls[1:]]
    assert len(tout) == N, f"{len(tout)} vs {N}"
    sv("tokens", np.array(toks, dtype=np.int64))
    sv("tokens_next", np.array(tout, dtype=np.int64))
    idxs = [k - box["x_len"] for k in kv_at_emb[1:N + 1]]
    sv("pe_idx", np.array(idxs, dtype=np.int64))
    for s, (x, mk) in enumerate(dec_inputs):
        sv(f"xy{s}", x.numpy())
        sv(f"mask{s}", mk.numpy().astype(np.uint8))
    for s, (lg, pv, _) in enumerate(sample_calls[1:]):
        sv(f"logits{s}", lg.numpy())
        sv(f"prev{s}", pv.numpy().astype(np.int64))
    # suppressed 标记：call0 恒 True；loop idx=s+1，s+1 < 10 则 True
    supp = [1 if (s + 1) < 10 else 0 for s in range(N)]
    sv("suppressed", np.array(supp, dtype=np.int64))
    sv("nsteps", np.array([N], dtype=np.int64))
    sv("sv_codes", svcap["codes"].numpy().astype(np.int64))
    sv("sv_text", svcap["text"].numpy().astype(np.int64))
    sv("sv_ge", svcap["ge"].numpy())
    sv("sv_audio", svcap["audio"].numpy())
    sv("xlen_ylen", np.array([psd["x"].shape[1], psd["y"].shape[1]], dtype=np.int64))
    sv("x_ids", psd["x"].numpy().astype(np.int64))
    sv("bert", psd["bert"].numpy())
    sv("xy_ref", psd["xy"].numpy())
    sv("prompt_mask", psd["mk"].numpy().astype(np.uint8))
    # 交叉验证：emb 公式重算 xy；sample 输出 == emb token 序
    ok = True
    for s in range(N):
        ye = W[toks[s]] * xs + pec[idxs[s], 0]
        d = np.abs(ye - dec_inputs[s][0].numpy()).max()
        if d > 1e-6:
            print(f"  step{s} xy-recompute MISMATCH {d}", flush=True)
            ok = False
    tok_match = all(t == sample_calls[s][2] for s, t in enumerate(toks))
    print(f"xy-recompute ok={ok}, tokens==sample_out_prefix={tok_match}", flush=True)
print("CLOSED-DUMP DONE", flush=True)
