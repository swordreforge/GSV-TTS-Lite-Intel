"""5 seed 清净率对照：同输入 runner 跑 5 seed，SoVITS 解码，转写数干净率。"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
import math, subprocess
import torch
from torch.nn import functional as _F
_F.scaled_dot_product_attention = lambda q, k, v, attn_mask=None, scale=None, **kw: (
    torch.softmax((q.float() @ k.float().transpose(-1, -2)) / (scale or math.sqrt(q.shape[-1])), dim=-1)
    @ v.float()).to(q.dtype)
import logging
logging.disable(logging.INFO)
import numpy as np
from gsv_tts import TTS
from gsv_tts.TTS import get_phones_and_bert

TEXT = "こんにちは。"
SEEDS = [1, 2, 3, 4, 5]
WD = "/tmp/opencode/rate5"
os.makedirs(WD, exist_ok=True)

tts = TTS(device="cpu", dtype="float32")
tts.load_gpt_model(); tts.load_sovits_model(); tts.init_language_module("ja")
sovits_m, ge = tts._prepare_sovits_resources(str(tts.default_sovits_path), "examples/laffey.mp3")
gpt_m, prompt, phones1, bert1 = tts._prepare_gpt_resources(
    str(tts.default_gpt_path), "examples/AnAn.ogg",
    "ちが……ちがう。レイア、貴様は間違っている。", "ja")
phones2, _, bert2, _ = get_phones_and_bert(TEXT, tts.tts_config, "ja")
x = torch.LongTensor(phones1 + phones2).unsqueeze(0)
bert = torch.cat([bert1, bert2]).unsqueeze(0)
np.ascontiguousarray(x.numpy().astype(np.int64).reshape(-1)).tofile(f"{WD}/x_ids.bin")
np.ascontiguousarray(prompt.numpy().astype(np.int64).reshape(-1)).tofile(f"{WD}/y.bin")
np.ascontiguousarray(bert.numpy()).tofile(f"{WD}/bert.bin")
np.array([x.shape[1], prompt.shape[1]], dtype=np.int64).tofile(f"{WD}/xlen_ylen.bin")
print("prep done", tuple(x.shape), flush=True)

env = dict(os.environ)
env["SYCL_CACHE_PERSISTENT"] = "0"
env["RUNNER_VALIDATE"] = "0"
env["RUNNER_TOP_K"] = "15"
env["RUNNER_IN_DIR"] = WD + "/"
BIN = os.path.expanduser("~/.cache/gsv/bin/t2s_runner")
for s in SEEDS:
    env["RUNNER_SEED"] = str(s)
    env["RUNNER_TOKENS_OUT"] = f"{WD}/toks_{s}.bin"
    r = subprocess.run([BIN], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    print(f"seed {s} done", flush=True)

tts_x = TTS(device="xpu", dtype="float32")
tts_x.load_sovits_model()
svx = list(tts_x.sovits_models.values())[0].vq_model.eval()
import soundfile as sf
for s in SEEDS:
    toks = np.fromfile(f"{WD}/toks_{s}.bin", np.int64)[1:]
    eos = np.where(toks == 1024)[0]
    toks = toks[:eos[0]] if len(eos) else toks
    with torch.inference_mode():
        audio, _ = svx.decode(torch.from_numpy(toks).unsqueeze(0).unsqueeze(0).to("xpu"),
                              torch.LongTensor(phones2).unsqueeze(0).to("xpu"),
                              ge.to("xpu"), cuda_graph=False, noise_scale=0)
    a = audio.detach().cpu().numpy().reshape(-1)
    sf.write(f"{WD}/seed_{s}.wav", a, 32000)
    print(f"seed {s}: {len(toks)} toks {len(a)/32000:.1f}s", flush=True)
print("RATE5 DONE", flush=True)
