"""同进程：_prepare 路径 y vs 直接 _get_prompt y。"""
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
import collections

tts = TTS(device="xpu", dtype="float32")
tts.load_gpt_model()
tts.load_sovits_model()
tts.init_language_module("ja")
gpt_m, prompt_p, phones1, bert1 = tts._prepare_gpt_resources(
    str(tts.default_gpt_path), "examples/AnAn.ogg",
    "ちが……ちがう。レイア、貴様は間違っている。", "ja")
a = prompt_p.cpu().numpy().astype(np.int64).reshape(-1)
print("prepare-path uniq", len(set(a.tolist())), "top3", collections.Counter(a.tolist()).most_common(3), flush=True)
sv = list(tts.sovits_models.values())[0]
hub = CNHubert(tts.cnhubert_path, tts.tts_config)
b = tts._get_prompt(hub, sv, "examples/AnAn.ogg").cpu().numpy().astype(np.int64).reshape(-1)
print("direct-path uniq", len(set(b.tolist())), "top3", collections.Counter(b.tolist()).most_common(3), flush=True)
n = min(len(a), len(b))
print("prepare-vs-direct diff:", int((a[:n] != b[:n]).sum()), "/", n, flush=True)
