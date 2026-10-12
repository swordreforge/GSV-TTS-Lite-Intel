"""GPT 生成前后各提一次 y（新实例），看生成是否翻转 Hubert。"""
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
from gsv_tts.GPT_SoVITS.Featurizer import CNHubert
from gsv_tts.TTS import get_phones_and_bert as _gpb

tts = TTS(device="xpu", dtype="float32")
tts.load_gpt_model()
tts.load_sovits_model()
tts.init_language_module("ja")
sv = list(tts.sovits_models.values())[0]

hub1 = CNHubert(tts.cnhubert_path, tts.tts_config)
ya = tts._get_prompt(hub1, sv, "examples/AnAn.ogg").cpu().numpy().astype(np.int64).reshape(-1)
ya.tofile("/tmp/opencode/y_seq_a.bin")
print("pre-gen y done", flush=True)

# 一次真实 GPT 生成（XPU）
t2s = list(tts.gpt_models.values())[0].t2s_model.eval()
phones2, _, bert2, _ = _gpb("こんにちは。", tts.tts_config, "ja")
gpt_m, prompt, phones1, bert1 = tts._prepare_gpt_resources(
    str(tts.default_gpt_path), "examples/AnAn.ogg",
    "ちが……ちがう。レイア、貴様は間違っている。", "ja")
x = torch.LongTensor(phones1 + phones2).to("xpu").unsqueeze(0)
bert = torch.cat([bert1, bert2]).unsqueeze(0).to("xpu")
with torch.inference_mode():
    out = t2s.infer(x, prompt.to("xpu"), bert, top_k=15, top_p=1.0,
                    temperature=1.0, repetition_penalty=1.35)
o = out[0, 0].cpu().numpy() if out.dim() == 3 else out.cpu().numpy().reshape(-1)
print("gen len:", len(o), flush=True)

hub2 = CNHubert(tts.cnhubert_path, tts.tts_config)
yb = tts._get_prompt(hub2, sv, "examples/AnAn.ogg").cpu().numpy().astype(np.int64).reshape(-1)
yb.tofile("/tmp/opencode/y_seq_b.bin")
n = min(len(ya), len(yb))
print("pre-vs-post-gen diff:", int((ya[:n] != yb[:n]).sum()), "/", n, flush=True)
import collections
print("pre top3:", collections.Counter(ya.tolist()).most_common(3), flush=True)
print("post top3:", collections.Counter(yb.tolist()).most_common(3), flush=True)
print("SEQ DONE", flush=True)
