"""上游 infer_vc ×3 独立进程，测命中率（存 y+输出）。"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
from gsv_tts import TTS
import logging
logging.disable(logging.INFO)

ROUND = int(os.environ.get("VC_ROUND", "0"))
TEXT = "现在我的本体处在海的深层这里的端流很危险用于保护研究员的稳定结构正在被持续破坏我必须不断对它进行重构"
tts = TTS(device="xpu", dtype="float32")
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
sf.write(f"/tmp/opencode/vc_round{ROUND}.wav", a, out.samplerate)
print(f"round {ROUND} dur:", round(len(a) / out.samplerate, 1), flush=True)
# 存此次的 y（从 cache 里拿）
y = tts.prompt_audio_cache["32uwgy.wav"]["prompt"]
y.cpu().numpy().astype(np.int64).tofile(f"/tmp/opencode/vc_round{ROUND}_y.bin")
import collections
c = collections.Counter(y.reshape(-1).tolist())
print(f"round {ROUND} y uniq {len(c)} sil239 {c.get(239, 0)}", flush=True)
