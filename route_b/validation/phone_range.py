"""验非法参数：phone id 是否越界（embedding 大小 = len(Symbols.symbols)）。"""
import sys, os
sys.path.insert(0, ".")
sys.path.insert(0, "route_b/runner")
import math
import torch
from torch.nn import functional as _F
_F.scaled_dot_product_attention = lambda q, k, v, attn_mask=None, scale=None, **kw: (
    torch.softmax((q.float() @ k.float().transpose(-1, -2)) / (scale or math.sqrt(q.shape[-1])), dim=-1)
    @ v.float()).to(q.dtype)
import logging
logging.disable(logging.INFO)
from gsv_tts import TTS
from gsv_tts.TTS import get_phones_and_bert as _gpb
from gsv_tts.GPT_SoVITS.SoVITS.models import Symbols

print("len(Symbols.symbols) =", len(Symbols.symbols), flush=True)
emb_rows = None

tts = TTS(device="cpu", dtype="float32")
tts.load_gpt_model(); tts.load_sovits_model()
sv = list(tts.sovits_models.values())[0]
for name, m in sv.vq_model.named_modules():
    if hasattr(m, "text_embedding"):
        emb_rows = m.text_embedding.num_embeddings
        print(f"text_embedding rows = {emb_rows}  (module {name})", flush=True)

# 服务端收到的真实组合
cases = [
    ("ja", "target", "こんにちは。"),
]
ptext = "现在我的本体处在海的深层这里的端流很危险用于保护研究员的稳定结构正在被持续破坏我必须不断对它进行重构"
ptext_asr = "现在 我的本体处在海的深层这里的端流很危险用于保护研究员的稳定结构正在被持续破坏我必须不断对它进行重构"

for target_lang, kind, text in cases:
    for pl in ["ja", "zh"]:
        tts.init_language_module(target_lang, pl)
        for tag, t in [("prompt_text", ptext), ("prompt_text_asr", ptext_asr), ("target", text)]:
            try:
                ph, _, _, _ = _gpb(t, tts.tts_config, pl if tag != "target" else target_lang)
                mx = max(ph) if ph else -1
                mn = min(ph) if ph else -1
                flag = "OUT-OF-RANGE!" if emb_rows and mx >= emb_rows else "ok"
                print(f"[{pl}] {tag}: n={len(ph)} min={mn} max={mx} vs rows={emb_rows} -> {flag}", flush=True)
            except Exception as e:
                print(f"[{pl}] {tag}: ERROR {e}", flush=True)
