"""上游流程 + 存 y：给对照用。"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
from gsv_tts import TTS
import logging
logging.disable(logging.INFO)

TEXT = "现在我的本体处在海的深层这里的端流很危险用于保护研究员的稳定结构正在被持续破坏我必须不断对它进行重构"
tts = TTS(device="xpu", dtype="float32")
orig = TTS._get_prompt.__get__(tts, TTS)


def hooked(self, hub, sv, path):
    y = orig(hub, sv, path)
    y.cpu().numpy().astype("int64").tofile("/tmp/opencode/upstream_good_y.bin")
    print("upstream y saved", tuple(y.shape), flush=True)
    return y


TTS._get_prompt = hooked
out = tts.infer_vc(
    spk_audio_path="examples/laffey.mp3",
    prompt_audio_path="32uwgy.wav",
    prompt_audio_text=TEXT,
    prompt_language="zh",
    noise_scale=0.5, speed=1.0,
)
import soundfile as sf
import numpy as np
a = np.asarray(out.audio_data, dtype=np.float32).reshape(-1)
sf.write("/tmp/opencode/vc_up4.wav", a, out.samplerate)
print("dur:", round(len(a) / out.samplerate, 1), flush=True)
