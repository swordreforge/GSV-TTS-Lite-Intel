"""上游原生 infer_vc 对照（单实例 XPU）。"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
from gsv_tts import TTS
import logging
logging.disable(logging.INFO)

tts = TTS(device="xpu", dtype="float32")
out = tts.infer_vc(
    spk_audio_path="examples/laffey.mp3",
    prompt_audio_path="32uwgy.wav",
    prompt_audio_text="现在我的本体处在海的深层这里的端流很危险用于保护研究员的稳定结构正在被持续破坏我必须不断对它进行重构",
    prompt_language="zh",
    noise_scale=0.5, speed=1.0,
)
import soundfile as sf
import numpy as np
a = np.asarray(out.audio_data, dtype=np.float32).reshape(-1)
sf.write("/tmp/opencode/vc_upstream.wav", a, out.samplerate)
print("upstream vc dur:", round(len(a) / out.samplerate, 1), flush=True)
