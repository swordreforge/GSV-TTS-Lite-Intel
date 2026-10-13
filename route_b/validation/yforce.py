"""为指定音频暴力选 y：串行抽签 ×N + resyn 打分，留最优（达标才缓存）。"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
import math
import torch
from torch.nn import functional as _F
_F.scaled_dot_product_attention = lambda q, k, v, attn_mask=None, scale=None, **kw: (
    torch.softmax((q.float() @ k.float().transpose(-1, -2)) / (scale or math.sqrt(q.shape[-1])), dim=-1)
    @ v.float()).to(q.dtype)
import logging
logging.disable(logging.INFO)
import numpy as np
import subprocess
from gsv_tts import TTS
from gsv_tts.TTS import get_phones_and_bert as _gpb

AUDIO = sys.argv[1] if len(sys.argv) > 1 else "32uwgy.wav"
TEXT = sys.argv[2] if len(sys.argv) > 2 else None
LANG = sys.argv[3] if len(sys.argv) > 3 else "zh"
N = int(sys.argv[4]) if len(sys.argv) > 4 else 6

sys.path.insert(0, os.path.join(_ROOT, "route_b", "runner"))
from serve_api import _y_score, _resyn_score, _y_cache_key, _Y_CACHE_DIR

tts = TTS(device="cpu", dtype="float32")
tts.load_gpt_model(); tts.load_sovits_model(); tts.init_language_module(LANG)
_, ge = tts._prepare_sovits_resources(str(tts.default_sovits_path), "examples/laffey.mp3")
tts_x = TTS(device="xpu", dtype="float32"); tts_x.load_sovits_model()
svx = list(tts_x.sovits_models.values())[0].vq_model.eval()
if TEXT is None:
    from serve_api import _get_asr
    import soundfile as sf
    a0, sr0 = sf.read(AUDIO)
    if getattr(a0, "ndim", 1) > 1:
        a0 = a0.mean(axis=-1)
    n = int(len(a0) * 16000 / sr0)
    a16 = np.interp(np.linspace(0, len(a0), n), np.arange(len(a0)),
                    np.asarray(a0, dtype=np.float64)).astype(np.float32)
    segs, _ = _get_asr().transcribe(a16, beam_size=5, language=LANG)
    TEXT = "".join(s.text for s in segs)
    print("auto text:", TEXT[:60], flush=True)

import tempfile
results = []
for k in range(N):
    out = os.path.join(tempfile.gettempdir(), f"yforce_{os.getpid()}_{k}.bin")
    env = dict(os.environ)
    env["SYCL_CACHE_PERSISTENT"] = "0"
    env["OMP_NUM_THREADS"] = ["16", "1", "4", "8", "2", "12"][k % 6]
    r = subprocess.run([sys.executable, os.path.join(_ROOT, "route_b", "runner", "extract_y.py"),
                        AUDIO, out, "xpu"], env=env, capture_output=True, text=True, timeout=900)
    if r.returncode != 0 or not os.path.exists(out):
        print(f"try {k}: FAILED", flush=True)
        continue
    u, meta = _y_score(out)
    try:
        ratio = _resyn_score(out, TEXT, LANG, svx, ge, tts)
    except Exception as e:
        print(f"try {k}: resyn failed {e}", flush=True)
        continue
    print(f"try {k}: uniq={u:.3f} resyn={ratio:.3f}", flush=True)
    results.append((ratio, u, out))
    try:
        pass
    finally:
        pass
results.sort(key=lambda t: -t[0])
print("=== ranking ===", flush=True)
for ratio, u, out in results:
    print(f"resyn={ratio:.3f} uniq={u:.3f} {out}", flush=True)
if results and results[0][0] >= 0.15:
    ybin, _ = _y_cache_key(AUDIO, TEXT, LANG)
    import shutil
    os.makedirs(_Y_CACHE_DIR, exist_ok=True)
    shutil.copyfile(results[0][2], ybin)
    print("CACHED:", ybin, flush=True)
else:
    print("NO PASSING CANDIDATE", flush=True)
