"""XPU torch free-run，但 y 钉死为文件（测 freshness 假说）。"""
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

YFIX = os.environ.get("FIX_Y", "/tmp/opencode/y_xpu.bin")
N = int(os.environ.get("CENSUS_N", "3"))
tts = TTS(device="xpu", dtype="float32")
tts.load_gpt_model()
tts.load_sovits_model()
tts.init_language_module("ja")
gpt_m, prompt, phones1, bert1 = tts._prepare_gpt_resources(
    str(tts.default_gpt_path), "examples/AnAn.ogg",
    "ちが……ちがう。レイア、貴様は間違っている。", "ja")
# 钉死 y
yfix = torch.from_numpy(np.fromfile(YFIX, np.int64)).reshape(1, -1).to("xpu")
tts.prompt_audio_cache["examples/AnAn.ogg"]["prompt"] = yfix
t2s = list(tts.gpt_models.values())[0].t2s_model.eval()
phones2, _, bert2, _ = _gpb("こんにちは。", tts.tts_config, "ja")
x = torch.LongTensor(phones1 + phones2).to("xpu").unsqueeze(0)
bert = torch.cat([bert1, bert2]).unsqueeze(0).to("xpu")
for i in range(N):
    with torch.inference_mode():
        out = t2s.infer(x, yfix, bert, top_k=15, top_p=1.0, temperature=1.0,
                        repetition_penalty=1.35)
    o = out[0, 0].cpu().numpy() if out.dim() == 3 else out.cpu().numpy().reshape(-1)
    print(f"fixed-y-{i}: len {len(o)} head {o[:5].tolist()}", flush=True)
print("FIXED DONE", flush=True)
