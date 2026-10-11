"""
GSV-TTS-Lite on Intel XPU (Arc GPU) — 非官方 Intel 生态运行入口。

官方指引只覆盖 CUDA / MPS / CPU，这里使用 Intel 的 XPU 后端 (torch.xpu)
在集成显卡上跑推理，并复用 oneDNN 的内存高效注意力内核。

用法:
    .venv-xpu/bin/python run_xpu.py
"""
import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
)

import torch  # noqa: E402

print("=" * 60)
print("torch      :", torch.__version__)
print("xpu avail  :", hasattr(torch, "xpu") and torch.xpu.is_available())
if hasattr(torch, "xpu") and torch.xpu.is_available():
    print("xpu device :", torch.xpu.get_device_name(0))
print("=" * 60)

from gsv_tts import TTS  # noqa: E402

# 显式指定 Intel XPU 后端与 float16 快路径
tts = TTS(device="xpu", dtype="float16", use_bert=False)

print("\n[1/4] 加载 GPT 模型 ...")
tts.load_gpt_model()

print("[2/4] 加载 SoVITS 模型 ...")
tts.load_sovits_model()

print("[3/4] 预热语言模块 ...")
tts.init_language_module("ja")

print("[4/4] 推理 ...")
t0 = time.time()
audio = tts.infer(
    spk_audio_path="examples/laffey.mp3",
    prompt_audio_path="examples/AnAn.ogg",
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
    text="へぇー、ここまでしてくれるんですね。",
)
dt = time.time() - t0

out_path = os.environ.get("GSV_OUT", "/tmp/opencode/gsv_xpu_out.wav")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

import soundfile as sf  # noqa: E402

sf.write(out_path, audio.audio_data, audio.samplerate)

dur = len(audio.audio_data) / audio.samplerate
print("\n" + "=" * 60)
print(f"生成音频      : {out_path}")
print(f"时长          : {dur:.2f} s")
print(f"端到端耗时    : {dt:.2f} s")
print(f"RTF           : {dt / dur:.3f}")
print("=" * 60)
