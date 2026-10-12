"""torch 长度普查：存每轮 token 数（XPU 自由跑）。"""
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
from gsv_tts.TTS import get_phones_and_bert as _gpb

N = int(os.environ.get("CENSUS_N", "10"))
tag = os.environ.get("CENSUS_TAG", "t")
import subprocess as _sp
tts = TTS(device="xpu", dtype="float32")
tts.load_gpt_model()
tts.load_sovits_model()
tts.init_language_module("ja")
_FRESH = os.environ.get("CENSUS_FRESH_Y", "0") == "1"
t2s = list(tts.gpt_models.values())[0].t2s_model.eval()
phones2, _, bert2, _ = _gpb("こんにちは。", tts.tts_config, "ja")
gpt_m, prompt, phones1, bert1 = tts._prepare_gpt_resources(
    str(tts.default_gpt_path), "examples/AnAn.ogg",
    "ちが……ちがう。レイア、貴様は間違っている。", "ja")
x = torch.LongTensor(phones1 + phones2).to("xpu").unsqueeze(0)
bert = torch.cat([bert1, bert2]).unsqueeze(0).to("xpu")
pr = prompt.to("xpu")
for i in range(N):
    if _FRESH:
        # 每轮重提 y（模拟 torch_pure 新鲜路径），存盘
        tts.prompt_audio_cache.pop("examples/AnAn.ogg", None)
        tts.cnhubert_model = None
        from gsv_tts.GPT_SoVITS.Featurizer import CNHubert as _H
        _hub = _H(tts.cnhubert_path, tts.tts_config)
        _sv = list(tts.sovits_models.values())[0]
        _yy = tts._get_prompt(_hub, _sv, "examples/AnAn.ogg")
        _yy.cpu().numpy().astype(np.int64).tofile(f"/tmp/opencode/freshy_{tag}{i}.bin")
    with torch.inference_mode():
        out = t2s.infer(x, pr, bert, top_k=15, top_p=1.0, temperature=1.0,
                        repetition_penalty=1.35)
    o = out[0, 0].cpu().numpy() if out.dim() == 3 else out.cpu().numpy().reshape(-1)
    o.astype(np.int64).tofile(f"/tmp/opencode/torch_toks_{tag}{i}.bin")
    print(f"{tag}{i}: len {len(o)} head {o[:5].tolist()}", flush=True)
print("CENSUS DONE", flush=True)
