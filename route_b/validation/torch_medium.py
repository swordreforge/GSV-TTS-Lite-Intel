"""torch-XPU 中等文本：短之外的行为。"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
from gsv_tts import TTS
import logging
logging.disable(logging.INFO)

TEXT = ("その日、佐藤は朝から妙に落ち着かなかった。"
        "昨夜遅くまでかかって仕上げた企画書を、部長の田村に提出する日だったからだ。"
        "通勤電車の中で、彼は何度もスマートフォンの画面を開いては閉じ、また開いた。")
tts = TTS(device="xpu", dtype="float32")
out = tts.infer(
    spk_audio_path="examples/laffey.mp3",
    prompt_audio_path="examples/AnAn.ogg",
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
    text=TEXT,
    top_k=15, top_p=1.0, temperature=1.0, repetition_penalty=1.35,
)
import soundfile as sf
import numpy as np
a = np.asarray(out.audio_data, dtype=np.float32).reshape(-1)
sf.write("/tmp/opencode/torch_medium.wav", a, out.samplerate)
print("torch-medium dur:", round(len(a) / out.samplerate, 1), flush=True)
