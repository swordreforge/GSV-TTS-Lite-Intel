"""OV Hubert 确定性验证：每个音频 ×2 独立进程提取，diff 必须为 0。
用法：python route_b/validation/ov_determinism.py
输出：/tmp/opencode/ovdet/ 下 y bin + 明细日志。
串行跑（并行同负载调度一致，且省内存），device=cpu（不占 XPU）。
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)

import subprocess
import numpy as np
import collections

AUDIOS = [
    "examples/AnAn.ogg",
    "examples/laffey.mp3",
    "32uwgy.wav",
    "vc_zh.wav",
    "vc_auto.wav",
]

OUTDIR = "/tmp/opencode/ovdet"
os.makedirs(OUTDIR, exist_ok=True)


def run_once(audio, tag):
    out = os.path.join(OUTDIR, f"{tag}.bin")
    env = dict(os.environ)
    env["SYCL_CACHE_PERSISTENT"] = "0"
    env["GSV_HUBERT_BACKEND"] = "openvino"
    env["GSV_OV_DEVICE"] = "CPU"
    r = subprocess.run(
        [sys.executable, os.path.join(_ROOT, "route_b", "runner", "extract_y.py"),
         audio, out, "cpu"],
        env=env, capture_output=True, text=True, timeout=900)
    if r.returncode != 0 or not os.path.exists(out):
        print(f"[{tag}] FAILED rc={r.returncode} err={r.stderr.strip()[-200:]}", flush=True)
        return None
    y = np.fromfile(out, np.int64)
    c = collections.Counter(y.tolist())
    print(f"[{tag}] len={len(y)} uniq={len(c)} sil239={c.get(239, 0)} "
          f"stat={r.stdout.strip()[-100:]}", flush=True)
    return y


def main():
    os.chdir(_ROOT)
    results = []
    for audio in AUDIOS:
        base = os.path.splitext(os.path.basename(audio))[0]
        print(f"=== {audio} ===", flush=True)
        a = run_once(audio, base + "_r1")
        b = run_once(audio, base + "_r2")
        if a is None or b is None:
            results.append((audio, "ERROR", -1))
            continue
        d = int((a != b).sum()) if len(a) == len(b) else -2
        verdict = "PASS" if d == 0 else "FAIL"
        print(f"--- {base}: diff={d}/{len(a)} {verdict}", flush=True)
        results.append((audio, verdict, d))
    print("=== summary ===", flush=True)
    for audio, v, d in results:
        print(f"{v} diff={d} {audio}", flush=True)
    n_pass = sum(1 for _, v, _ in results if v == "PASS")
    print(f"TOTAL {n_pass}/{len(results)} PASS", flush=True)


if __name__ == "__main__":
    main()
