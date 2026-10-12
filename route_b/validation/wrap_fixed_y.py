"""tts.infer 全包装 + 钉死 y：wrapper 是否关键。"""
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

tts = TTS(device="xpu", dtype="float32")
tts.load_gpt_model()
tts.load_sovits_model()
tts.init_language_module("ja")
# 预填缓存再钉死 y
tts._prepare_gpt_resources(str(tts.default_gpt_path), "examples/AnAn.ogg",
                           "ちが……ちがう。レイア、貴様は間違っている。", "ja")
yfix = torch.from_numpy(np.fromfile("/tmp/opencode/y_xpu.bin", np.int64)).reshape(1, -1).to("xpu")
tts.prompt_audio_cache["examples/AnAn.ogg"]["prompt"] = yfix
out = tts.infer(
    spk_audio_path="examples/laffey.mp3",
    prompt_audio_path="examples/AnAn.ogg",
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
    text="こんにちは。",
    top_k=15, top_p=1.0, temperature=1.0, repetition_penalty=1.35,
)
import soundfile as sf
a = np.asarray(out.audio_data, dtype=np.float32).reshape(-1)
sf.write("/tmp/opencode/wrap_fixedy.wav", a, out.samplerate)
print("wrap-fixed-y dur:", round(len(a) / out.samplerate, 1), flush=True)
