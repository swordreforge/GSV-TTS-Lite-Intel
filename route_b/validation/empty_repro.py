"""复现空 seq 崩溃：32uwgy-y + 短句 phones → runner tokens 检查 → 空 decode 是否 segfault。"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
import subprocess
import faulthandler
faulthandler.enable()
import math
import torch
from torch.nn import functional as _F
_F.scaled_dot_product_attention = lambda q, k, v, attn_mask=None, scale=None, **kw: (
    torch.softmax((q.float() @ k.float().transpose(-1, -2)) / (scale or math.sqrt(q.shape[-1])), dim=-1)
    @ v.float()).to(q.dtype)
import logging
logging.disable(logging.INFO)
import numpy as np

RUNNER_BIN = os.path.join(os.path.expanduser("~"), ".cache", "gsv", "bin", "t2s_runner")
YBIN = "/home/swordreforge/.cache/gsv/prompt_y/00c056656a6a8261.ybin"
WD = "/tmp/opencode/emptyrepro"
os.makedirs(WD, exist_ok=True)


def _ensure_oneapi_env():
    if os.environ.get("ONEAPI_ROOT"):
        return
    sv = "/opt/intel/oneapi/setvars.sh"
    if not os.path.exists(sv):
        return
    out = subprocess.run(["bash", "-c", f"source {sv} >/dev/null 2>&1 && env -0"],
                         capture_output=True, check=True).stdout
    for entry in out.decode().split("\0"):
        if "=" in entry:
            k, v = entry.split("=", 1)
            os.environ[k] = v


_ensure_oneapi_env()
os.environ.setdefault("GSV_HUBERT_BACKEND", "openvino")
os.environ.setdefault("GSV_OV_DEVICE", "CPU")

from gsv_tts import TTS
from gsv_tts.TTS import get_phones_and_bert as _gpb
os.chdir(_ROOT)

tts = TTS(device="cpu", dtype="float32")
tts.load_gpt_model(); tts.load_sovits_model(); tts.init_language_module("zh")
sovits_m, ge = tts._prepare_sovits_resources(str(tts.default_sovits_path), "examples/laffey.mp3")
gpt_m, prompt, phones1, bert1 = tts._prepare_gpt_resources(
    str(tts.default_gpt_path), "examples/AnAn.ogg", "dummy", "zh")
prompt = torch.from_numpy(np.fromfile(YBIN, np.int64).reshape(1, -1))
print("y len:", tuple(prompt.shape), flush=True)
phones2, _, bert2, _ = _gpb("今天天气真不错，适合出去走走。", tts.tts_config, "zh")
all_ids = torch.LongTensor(phones1 + phones2).unsqueeze(0)
bert = torch.cat([bert1, bert2]).unsqueeze(0)


def sv(name, a):
    np.ascontiguousarray(a).tofile(os.path.join(WD, name + ".bin"))


sv("x_ids", all_ids.numpy().astype(np.int64).reshape(-1))
sv("y", prompt.numpy().astype(np.int64).reshape(-1))
sv("bert", bert.numpy())
sv("xlen_ylen", np.array([all_ids.shape[1], prompt.shape[1]], dtype=np.int64))

env = dict(os.environ)
env["RUNNER_IN_DIR"] = WD + "/"
env["RUNNER_VALIDATE"] = "0"
env["RUNNER_TOKENS_OUT"] = os.path.join(WD, "tokens.bin")
env["RUNNER_TOP_K"] = "15"
env["RUNNER_SEED"] = sys.argv[1] if len(sys.argv) > 1 else "12345"
print("seed:", env["RUNNER_SEED"], flush=True)
env["SYCL_CACHE_PERSISTENT"] = "0"
r = subprocess.run([RUNNER_BIN], env=env, capture_output=True, text=True, timeout=900)
print("runner rc:", r.returncode, flush=True)
toks = np.fromfile(os.path.join(WD, "tokens.bin"), np.int64)
print("tokens:", len(toks), toks[:10].tolist(), flush=True)
seq = toks[1:]
eos = np.where(seq == 1024)[0]
if len(eos):
    seq = seq[:eos[0]]
print("seq after cut:", len(seq), flush=True)

tts_x = TTS(device="xpu", dtype="float32")
tts_x.load_sovits_model()
svx = list(tts_x.sovits_models.values())[0].vq_model.eval()
codes = torch.from_numpy(seq.astype(np.int64)).unsqueeze(0).unsqueeze(0).to("xpu")
ph2 = torch.LongTensor(phones2).unsqueeze(0).to("xpu")
print("decoding, codes:", tuple(codes.shape), flush=True)
with torch.inference_mode():
    torch.xpu.synchronize()
    audio, _ = svx.decode(codes, ph2, ge.to("xpu"), cuda_graph=False, noise_scale=0.5)
    torch.xpu.synchronize()
print("DECODE-OK audio:", tuple(audio.shape), flush=True)
