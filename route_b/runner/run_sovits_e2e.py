"""SoVITS 端到端收尾：runner tokens + sv_text/sv_ge -> XPU decode(noise0) -> 对比 torch ref 音频。"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# 仓库根（route_b/ 下脚本需回退两级）
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
import torch, time
import numpy as np
from gsv_tts import TTS
import logging
logging.disable(logging.INFO)

CDIR = "/tmp/opencode/t2s_closed"
toks = np.fromfile(os.environ.get("RUNNER_TOKENS", "/tmp/opencode/runner_tokens.bin"), np.int64)
ref_codes = np.fromfile(f"{CDIR}/sv_codes.bin", np.int64).reshape(-1)
# torch infer 切片：sv_codes == runner[1:1+len]（t0 被切掉；实测对齐）
assert len(toks) >= len(ref_codes) + 1, (len(toks), len(ref_codes))
assert (toks[1:1 + len(ref_codes)] == ref_codes).all(), "token mismatch vs sv_codes"
toks = toks[1:1 + len(ref_codes)]
print(f"tokens aligned: {len(toks)} (runner raw {len(np.fromfile(os.environ.get('RUNNER_TOKENS', '/tmp/opencode/runner_tokens.bin'), np.int64))})", flush=True)
codes = torch.from_numpy(toks).unsqueeze(0).unsqueeze(0)  # [1,1,T]
text = torch.from_numpy(np.fromfile(f"{CDIR}/sv_text.bin", np.int64).reshape(1, -1))
ge = torch.from_numpy(np.fromfile(f"{CDIR}/sv_ge.bin", np.float32).reshape(1, 1024, 1))
ref = np.fromfile(f"{CDIR}/sv_audio.bin", np.float32)
print(f"codes={tuple(codes.shape)} text={tuple(text.shape)} ge={tuple(ge.shape)} ref_len={len(ref)}", flush=True)

tts = TTS(device="xpu", dtype="float32")
tts.load_sovits_model()
sv = list(tts.sovits_models.values())[0].vq_model.eval()
with torch.inference_mode():
    c, t, g = codes.to("xpu"), text.to("xpu"), ge.to("xpu")
    torch.xpu.synchronize(); t0 = time.time()
    for _ in range(3):
        o, _ = sv.decode(c, t, g, cuda_graph=False, noise_scale=0)
    torch.xpu.synchronize(); ms = (time.time() - t0) / 3 * 1000
got = o.detach().cpu().numpy().reshape(-1)
print(f"sovits-xpu: {tuple(o.shape)} {ms:.0f}ms", flush=True)
n = min(len(got), len(ref))
d = np.abs(got[:n] - ref[:n])
print(f"audio parity: maxdiff={d.max():.6f} mean={d.mean():.2e} len={n}", flush=True)
got.tofile("/tmp/opencode/runner_audio.bin")
print("SOVITS-E2E DONE", flush=True)
