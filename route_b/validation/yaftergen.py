"""先跑一次真实 GPT 生成，再提 y（fresh hub），看家族。"""
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
import collections

tts = TTS(device="xpu", dtype="float32")
tts.load_gpt_model()
tts.load_sovits_model()
tts.init_language_module("ja")
sv = list(tts.sovits_models.values())[0]

# 先一次真实生成（与 freshy 同条件）
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

hub = CNHubert(tts.cnhubert_path, tts.tts_config)
y = tts._get_prompt(hub, sv, "examples/AnAn.ogg").cpu().numpy().astype(np.int64).reshape(-1)
c = collections.Counter(y.tolist())
print("post-gen y: uniq", len(c), "sil239", c.get(239, 0), "top3", c.most_common(3), flush=True)
