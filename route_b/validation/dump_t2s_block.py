"""Dump 单个 T2SBlock.decode 步的 fp32 输入/权重/输出，供 C++ ggml 对照。"""
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

OUT = Path("/tmp/opencode/t2s_blk0")
OUT.mkdir(parents=True, exist_ok=True)

tts = TTS(device="xpu", dtype="float16")
tts.load_gpt_model()
t2s = list(tts.gpt_models.values())[0].t2s_model.to("cpu").float().eval()
blk = t2s.t2s_transformer.blocks[0]

torch.manual_seed(0)
B, H, K, HD, DIM, L = 1, 16, 64, 32, 512, 1
x = torch.randn(B, L, DIM)
kc = torch.randn(B, H, K, HD)
vc = torch.randn(B, H, K, HD)
kv_len = torch.tensor([10])
mask = torch.zeros(B, H, L, K, dtype=torch.bool)
mask[..., :11] = True  # True=attend（管线语义），当前位也 True
bi = torch.tensor([0])

with torch.inference_mode():
    # 注意：T2SBlock 签名是 (x, kc, vc, attn_mask, kv_len, bidx)，与 transformer 层顺序不同
    # CPU 的 SDPA 不支持该配置，这里用 eager 复刻（True=attend，XPU 管线语义）做精确参照
    import math
    from torch.nn import functional as F
    B, L, _ = x.shape
    NH, HD = 16, 32
    residual = x
    qkv = blk.qkv(x).view(B, L, 3, NH, HD)
    q, k, v = qkv.unbind(2)
    q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
    kc2 = kc.clone(); vc2 = vc.clone()
    kc2[bi, :, kv_len] = k.squeeze(2)
    vc2[bi, :, kv_len] = v.squeeze(2)
    s = (q @ kc2.transpose(-1, -2)) / math.sqrt(HD)
    s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    o = (p @ vc2).transpose(1, 2).reshape(B, L, 512)
    o = blk.out_proj(o)
    o = blk.norm1(residual + o)
    r = o
    o = blk.mlp(o)
    out = blk.norm2(r + o)

def sv(name, t):
    if isinstance(t, np.ndarray):
        a = np.ascontiguousarray(t)
    else:
        a = np.ascontiguousarray(t.detach().numpy())
    a.tofile(OUT / (name + ".bin"))
    (OUT / (name + ".shape")).write_text(",".join(map(str, a.shape)))

sv("x", x); sv("kc", kc); sv("vc", vc); sv("mask", mask.numpy().astype(np.uint8))
sv("kvlen", kv_len.numpy()); sv("out", out)
for n, p in [("qkv_w", blk.qkv.weight), ("qkv_b", blk.qkv.bias),
             ("out_w", blk.out_proj.weight), ("out_b", blk.out_proj.bias),
             ("n1_w", blk.norm1.weight), ("n1_b", blk.norm1.bias),
             ("mlp1_w", blk.mlp[0].weight), ("mlp1_b", blk.mlp[0].bias),
             ("mlp2_w", blk.mlp[2].weight), ("mlp2_b", blk.mlp[2].bias),
             ("n2_w", blk.norm2.weight), ("n2_b", blk.norm2.bias)]:
    a = p.detach()
    if n.endswith("_w"):
        a = a.T.contiguous()  # torch [out,in] -> ggml [in,out]
    sv(n, a)
print("dumped to", OUT, "out", tuple(out.shape), flush=True)
