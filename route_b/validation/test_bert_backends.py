"""End-to-end: use_bert=True 中文推理，对比 RoBERTa torch-XPU / OV-GPU / OV-NPU。"""
import sys, os, time, logging
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# 仓库根（route_b/ 下脚本需回退两级）
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
logging.disable(logging.INFO)
import numpy as np
from gsv_tts import TTS

os.environ.pop("GSV_ROBERTA_BACKEND", None)

def run(tag):
    tts = TTS(device="xpu", dtype="float16", use_bert=True)
    tts.load_gpt_model()
    tts.load_sovits_model()
    tts.init_language_module("zh")
    kw = dict(
        spk_audio_path="examples/laffey.mp3",
        prompt_audio_path="examples/AnAn.ogg",
        prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
        prompt_language="ja",
        text="你好，世界，这是一个在英特尔显卡上运行的语音合成测试。",
        text_language="zh",
    )
    a = tts.infer(**kw)  # warm (含 BERT 编译/缓存)
    t = time.time()
    a = tts.infer(**kw)
    dt = time.time() - t
    dur = len(a.audio_data) / a.samplerate
    rms = float(np.sqrt((a.audio_data ** 2).mean()))
    print(f"[{tag}] audio={dur:.2f}s wall={dt:.2f}s RTF={dt/dur:.3f} rms={rms:.3f}", flush=True)

run("torch-XPU")
