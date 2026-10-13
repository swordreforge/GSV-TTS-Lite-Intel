"""E2E 稳态 profile：phones/BERT → runner(GPT) → SoVITS(XPU)，分段计时找瓶颈。
稳态假设：y 缓存命中（AnAn.good.ybin），ge 按实测计时。
runner 单发进程（不碰线上 pool），SoVITS 与线上共 XPU（空闲时测）。
用法：python route_b/validation/prof_e2e.py
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
import math, re, subprocess, time
import faulthandler
faulthandler.enable()
import torch
from torch.nn import functional as _F
_F.scaled_dot_product_attention = lambda q, k, v, attn_mask=None, scale=None, **kw: (
    torch.softmax((q.float() @ k.float().transpose(-1, -2)) / (scale or math.sqrt(q.shape[-1])), dim=-1)
    @ v.float()).to(q.dtype)
import logging
logging.disable(logging.INFO)
import numpy as np

RUNNER_BIN = os.environ.get("RUNNER_BIN",
                            os.path.join(os.path.expanduser("~"), ".cache", "gsv", "bin", "t2s_runner"))


def _ensure_oneapi_env():
    if os.environ.get("ONEAPI_ROOT"):
        return
    sv = "/opt/intel/oneapi/setvars.sh"
    if not os.path.exists(sv):
        return
    try:
        out = subprocess.run(["bash", "-c", f"source {sv} >/dev/null 2>&1 && env -0"],
                             capture_output=True, check=True).stdout
        for entry in out.decode().split("\0"):
            if "=" in entry:
                k, v = entry.split("=", 1)
                os.environ[k] = v
    except Exception as e:
        print(f"oneapi env inject failed: {e}", flush=True)


_ensure_oneapi_env()
# 跟线上服务同一条路（systemd unit 也是这套 env）
os.environ.setdefault("GSV_HUBERT_BACKEND", "openvino")
os.environ.setdefault("GSV_OV_DEVICE", "CPU")
GOOD_Y = os.environ.get("PROMPT_Y_FILE",
                        os.path.join(os.path.expanduser("~"), ".cache", "gsv", "prompt_y", "AnAn.good.ybin"))
TOP_K = int(os.environ.get("RUNNER_TOP_K", "15"))
WORKDIR = "/tmp/opencode/prof_e2e"

CASES = [
    ("ja", "こんにちは。", "examples/laffey.mp3", "examples/AnAn.ogg",
     "ちが……ちがう。レイア、貴様は間違っている。"),
    ("zh", "现在我的本体处在海的深层，这里的端流很危险，我必须不断对它进行重构。", "examples/laffey.mp3",
     "examples/AnAn.ogg", "ignored-when-good-y"),
]


def sync_xpu():
    torch.xpu.synchronize()


def main():
    import soundfile as sf
    from gsv_tts import TTS
    from gsv_tts.TTS import get_phones_and_bert as _gpb

    os.makedirs(WORKDIR, exist_ok=True)
    tts = TTS(device="cpu", dtype="float32")
    tts.load_gpt_model(); tts.load_sovits_model()
    tts_x = TTS(device="xpu", dtype="float32"); tts_x.load_sovits_model()
    svx = list(tts_x.sovits_models.values())[0].vq_model.eval()

    for lang, text, spk, prompt_audio, prompt_text in CASES:
        wd = os.path.join(WORKDIR, lang)
        os.makedirs(wd, exist_ok=True)
        print(f"===== case {lang}: {text[:24]!r} =====", flush=True)
        t_all = time.time()
        tts.init_language_module(lang)

        t0 = time.time()
        print(f"[{lang}] stage: prepare_sovits_resources ...", flush=True)
        sovits_m, ge = tts._prepare_sovits_resources(str(tts.default_sovits_path), spk)
        ge_m = list(tts.sovits_models.values())[0]  # unused, keep ref
        t_ge = time.time() - t0

        t0 = time.time()
        print(f"[{lang}] stage: prepare_gpt_resources ...", flush=True)
        gpt_m, prompt, phones1, bert1 = tts._prepare_gpt_resources(
            str(tts.default_gpt_path), prompt_audio, prompt_text, lang)
        # 稳态：用缓存好 y 覆盖刚提取的 prompt（线上 cache-hit 等价）
        if os.path.exists(GOOD_Y) and os.path.basename(prompt_audio) == "AnAn.ogg":
            prompt = torch.from_numpy(np.fromfile(GOOD_Y, np.int64).reshape(1, -1))
        phones2, _, bert2, _ = _gpb(text, tts.tts_config, lang)
        all_ids = torch.LongTensor(phones1 + phones2).unsqueeze(0)
        bert = torch.cat([bert1, bert2]).unsqueeze(0)
        t_prep = time.time() - t0

        def sv(name, a):
            np.ascontiguousarray(a).tofile(os.path.join(wd, name + ".bin"))
        sv("x_ids", all_ids.numpy().astype(np.int64).reshape(-1))
        sv("y", prompt.numpy().astype(np.int64).reshape(-1))
        sv("bert", bert.numpy())
        sv("xlen_ylen", np.array([all_ids.shape[1], prompt.shape[1]], dtype=np.int64))

        env = dict(os.environ)
        env["RUNNER_IN_DIR"] = wd + "/"
        env["RUNNER_VALIDATE"] = "0"
        env["RUNNER_TOKENS_OUT"] = os.path.join(wd, "tokens.bin")
        env["RUNNER_TOP_K"] = str(TOP_K)
        env["SYCL_CACHE_PERSISTENT"] = "0"
        t0 = time.time()
        r = subprocess.run([RUNNER_BIN], env=env, capture_output=True, text=True, timeout=1800)
        t_runner = time.time() - t0
        if r.returncode != 0:
            print(f"RUNNER FAILED: {r.stderr[-500:]}", flush=True)
            continue
        pre = re.findall(r"prefill compute ([\d.]+)ms", r.stdout)
        step_pairs = re.findall(r"step (\d+) tok=\d+(?: expect=\d+ \w+)? time=([\d.]+)ms", r.stdout)
        with open(os.path.join(wd, "runner.log"), "w") as f:
            f.write(r.stdout)
        steps_all = [(int(s), float(t)) for s, t in step_pairs]
        steady = [t for s, t in steps_all if s >= 3]
        steps = [t for _, t in steps_all]
        toks = np.fromfile(os.path.join(wd, "tokens.bin"), np.int64)

        # SoVITS：复刻 driver/torch 切片规则——去 t0，首个 EOS(1024) 截断；phones 只喂目标段
        seq = toks[1:]
        eos = np.where(seq == 1024)[0]
        if len(eos):
            seq = seq[:eos[0]]
        codes = torch.from_numpy(seq.astype(np.int64)).unsqueeze(0).unsqueeze(0).to("xpu")
        ph = torch.LongTensor(phones2).unsqueeze(0).to("xpu")
        ge_x = ge.to("xpu") if torch.is_tensor(ge) else torch.as_tensor(np.asarray(ge)).to("xpu")
        with torch.inference_mode():
            sync_xpu(); t0 = time.time()
            audio, _ = svx.decode(codes, ph, ge_x, cuda_graph=False, noise_scale=0.5)
            sync_xpu(); t_sov = time.time() - t0
        a = audio.detach().cpu().numpy().reshape(-1)
        sf.write(os.path.join(wd, "out.wav"), a, 32000)
        dur = len(a) / 32000
        wall = time.time() - t_all
        n = len(steps)
        print(f"[prep phones/bert+y] {t_prep*1000:.0f}ms  (ge {t_ge*1000:.0f}ms)", flush=True)
        sm = f"{sum(steady)/len(steady):.1f}ms" if steady else "n/a"
        sx = f"{max(steady):.1f}ms" if steady else "n/a"
        print(f"[gpt runner] wall {t_runner:.2f}s steps={len(toks)} "
              f"prefill={pre[-1] if pre else '?'}ms "
              f"step_all_mean={sum(steps)/n:.1f}ms step_max={max(steps) if steps else 0:.1f}ms n={n} "
              f"steady_mean={sm} steady_max={sx} ns={len(steady)}", flush=True)
        print(f"[sovits xpu] {t_sov:.2f}s for {dur:.1f}s audio (rtf={t_sov/dur:.2f})", flush=True)
        print(f"[total] wall {wall:.2f}s audio {dur:.1f}s RTF={wall/dur:.2f}", flush=True)


if __name__ == "__main__":
    main()
