"""Benchmark GSV-TTS-Lite on Intel XPU (warm latency, infer vs infer_batched)."""
import os, sys, time, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.disable(logging.INFO)

import torch
from gsv_tts import TTS

DEV = sys.argv[1] if len(sys.argv) > 1 else "xpu"
DT = sys.argv[2] if len(sys.argv) > 2 else "float16"

tts = TTS(device=DEV, dtype=DT, use_bert=False)
print("device:", DEV, "dtype:", DT, "config:", tts.tts_config.device, tts.tts_config.dtype)
tts.load_gpt_model()
tts.load_sovits_model()
tts.init_language_module("ja")

TXT = "へぇー、ここまでしてくれるんですね。"
KW = dict(
    spk_audio_path="examples/laffey.mp3",
    prompt_audio_path="examples/AnAn.ogg",
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
    text=TXT,
)

# warmup
a = tts.infer(**KW)
dur = len(a.audio_data) / a.samplerate
print(f"audio duration: {dur:.2f}s\n")

print("=== warm infer() ===")
for i in range(3):
    torch.xpu.synchronize() if DEV == "xpu" else None
    t = time.time()
    a = tts.infer(**KW)
    dt = time.time() - t
    print(f"  run {i+1}: {dt:.2f}s  RTF={dt/ (len(a.audio_data)/a.samplerate):.3f}")

print("\n=== warm infer_batched() ===")
BKW = dict(
    spk_audio_paths="examples/laffey.mp3",
    prompt_audio_paths="examples/AnAn.ogg",
    prompt_audio_texts="ちが……ちがう。レイア、貴様は間違っている。",
    texts=[TXT, TXT, TXT],
)
try:
    for i in range(3):
        torch.xpu.synchronize() if DEV == "xpu" else None
        t = time.time()
        clips = tts.infer_batched(**BKW)
        dt = time.time() - t
        dur = sum(len(c.audio_data) / c.samplerate for c in clips)
        print(f"  run {i+1}: {dt:.2f}s for {dur:.2f}s audio  RTF={dt/dur:.3f}")
except Exception as e:
    print("  infer_batched failed:", type(e).__name__, str(e)[:300])
