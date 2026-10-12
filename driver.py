"""端到端 driver：活文本 -> TTS特征准备 -> C++ runner -> tokens -> SoVITS -> wav。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subprocess
import torch
import numpy as np
from gsv_tts import TTS
import logging
logging.disable(logging.INFO)

import math
from torch.nn import functional as _F
def _eager_sdpa(q, k, v, attn_mask=None, scale=None, **kw):
    d = q.shape[-1]
    s = (q.float() @ k.float().transpose(-1, -2)) / (scale or math.sqrt(d))
    if attn_mask is not None:
        s = s.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else s + attn_mask.float()
    return (torch.softmax(s, dim=-1) @ v.float()).to(q.dtype)
_F.scaled_dot_product_attention = _eager_sdpa

RUNNER_BIN = "/tmp/opencode/t2s_runner"


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

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="こんにちは。")
    ap.add_argument("--spk", default="examples/laffey.mp3")
    ap.add_argument("--prompt-audio", default="examples/AnAn.ogg")
    ap.add_argument("--prompt-text", default="ちが……ちがう。レイア、貴様は間違っている。")
    ap.add_argument("--out", default="/tmp/opencode/driver_out.wav")
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--workdir", default="/tmp/opencode/driver_run")
    ap.add_argument("--check-torch", action="store_true", help="同 tensor 跑 torch GPT 对 token")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    tts = TTS(device="cpu", dtype="float32")
    tts.load_gpt_model(); tts.load_sovits_model(); tts.init_language_module("ja")

    sovits = list(tts.sovits_models.values())[0]
    vq_model = sovits.vq_model.eval()
    gpt = list(tts.gpt_models.values())[0]
    t2s_model = gpt.t2s_model.eval()

    from gsv_tts.TTS import get_phones_and_bert
    # 走 TTS.infer 相同的准备路径
    sovits_m, ge = tts._prepare_sovits_resources(str(tts.default_sovits_path), args.spk)
    gpt_m, prompt, phones1, bert1 = tts._prepare_gpt_resources(
        str(tts.default_gpt_path), args.prompt_audio, args.prompt_text, "ja")
    phones2, word2ph, bert2, norm_text = get_phones_and_bert(args.text, tts.tts_config, "ja")
    import torch as _t
    all_phoneme_ids = _t.LongTensor(phones1 + phones2).unsqueeze(0)
    bert = _t.cat([bert1, bert2]).unsqueeze(0)
    print(f"x={tuple(all_phoneme_ids.shape)} bert={tuple(bert.shape)} y={tuple(prompt.shape)} ge={tuple(ge.shape)} phones2={len(phones2)}", flush=True)

    def sv(name, a):
        np.ascontiguousarray(a).tofile(os.path.join(args.workdir, name + ".bin"))
    sv("x_ids", all_phoneme_ids.numpy().astype(np.int64).reshape(-1))
    sv("y", prompt.numpy().astype(np.int64).reshape(-1))
    sv("bert", bert.numpy())
    sv("xlen_ylen", np.array([all_phoneme_ids.shape[1], prompt.shape[1]], dtype=np.int64))

    env = dict(os.environ)
    env["RUNNER_IN_DIR"] = args.workdir + "/"
    env["RUNNER_VALIDATE"] = "0"
    env["RUNNER_TOKENS_OUT"] = os.path.join(args.workdir, "tokens.bin")
    env["RUNNER_TOP_K"] = str(args.top_k)
    r = subprocess.run([RUNNER_BIN], env=env, capture_output=True, text=True)
    print(r.stdout[-1500:] if len(r.stdout) > 1500 else r.stdout, flush=True)
    if r.returncode != 0:
        print("RUNNER FAILED", r.stderr[-2000:], flush=True)
        sys.exit(1)

    toks = np.fromfile(os.path.join(args.workdir, "tokens.bin"), np.int64)
    print(f"runner tokens: {len(toks)}", flush=True)

    if args.check_torch:
        with torch.inference_mode():
            ref = t2s_model.infer(all_phoneme_ids, prompt, bert, top_k=args.top_k,
                                  top_p=1.0, temperature=1.0, repetition_penalty=1.35)
        ref = ref[0, 0].cpu().numpy() if ref.dim() == 3 else ref.cpu().numpy().reshape(-1)
        # torch 切片规则：去 t0 + EOS 截断；对齐比较
        import itertools
        best = None
        for drop in (0, 1):
            for cut in (len(toks), len(ref)):
                n = min(len(toks) - drop, len(ref))
                if n <= 0:
                    continue
                eq = (toks[drop:drop + n] == ref[:n]).all()
                if eq:
                    best = (drop, n)
                    break
            if best:
                break
        print(f"torch-check: torch_len={len(ref)} runner_len={len(toks)} align={best}", flush=True)

    # torch 切片规则复刻：去 t0，首个 EOS 截断
    seq = toks[1:]
    eos = np.where(seq == 1024)[0]
    if len(eos):
        seq = seq[:eos[0]]
    print(f"feeding sovits: {len(seq)} tokens", flush=True)
    tts_x = TTS(device="xpu", dtype="float32")
    tts_x.load_sovits_model()
    svx = list(tts_x.sovits_models.values())[0].vq_model.eval()
    with torch.inference_mode():
        codes = torch.from_numpy(seq).unsqueeze(0).unsqueeze(0).to("xpu")
        ph2 = torch.LongTensor(phones2).unsqueeze(0).to("xpu")
        gex = ge.to("xpu")
        import time
        torch.xpu.synchronize(); t0 = time.time()
        audio, _ = svx.decode(codes, ph2, gex, cuda_graph=False, noise_scale=0)
        torch.xpu.synchronize(); ms = (time.time() - t0) * 1000
    a = audio.detach().cpu().numpy().reshape(-1)
    print(f"audio: {len(a)} samples {ms:.0f}ms", flush=True)
    import soundfile as sf
    sf.write(args.out, a, 32000)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
