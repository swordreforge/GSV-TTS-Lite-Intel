"""Intel 路径 API 服务：TTS特征准备(CPU) -> runner --serve 常驻(GPT) -> SoVITS(XPU) -> wav。

串行合成（全局锁）：runner serve 单 JOB 通道 + XPU 状态保护，个人用量够用。
多 chunk：复用 TTS.infer 的 cut_text，长文本分段后静音拼接。
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
import subprocess
import tempfile
import threading
import time
import logging
logging.disable(logging.INFO)

import math
import torch
import numpy as np
from torch.nn import functional as _F


def _eager_sdpa(q, k, v, attn_mask=None, scale=None, **kw):
    d = q.shape[-1]
    s = (q.float() @ k.float().transpose(-1, -2)) / (scale or math.sqrt(d))
    if attn_mask is not None:
        s = s.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else s + attn_mask.float()
    return (torch.softmax(s, dim=-1) @ v.float()).to(q.dtype)
_F.scaled_dot_product_attention = _eager_sdpa

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

RUNNER_BIN = os.environ.get(
    "RUNNER_BIN",
    os.path.join(os.path.expanduser("~"), ".cache", "gsv", "bin", "t2s_runner"),
)
EOS = 1024
SR = 32000


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


class RunnerServer:
    """runner --serve 常驻进程，一次 JOB 一收发（调用方加锁串行）。"""

    def __init__(self):
        _ensure_oneapi_env()
        self._env = dict(os.environ)
        self._env["SYCL_CACHE_PERSISTENT"] = "0"  # 持久缓存本环境段错误，见 INTEL_XPU.md
        self._env["RUNNER_STREAM"] = "1"  # 每步打印 TOK <id>，job() 会忽略，job_stream() 消费
        if os.environ.get("GPT_GGUF"):  # 微调 GPT 的 gguf，重启服务切换
            self._env["RUNNER_GGUF"] = os.environ["GPT_GGUF"]
        self._spawn()

    def _spawn(self):
        self.p = subprocess.Popen([RUNNER_BIN, "--serve"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1, env=self._env)
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("runner serve exited during init")
            if "SERVE-READY" in line:
                return

    def _ensure_alive(self):
        if self.p.poll() is not None:  # runner 死了就地重拉（约1s + 下一 JOB 重新 JIT）
            print("runner dead, respawning...", flush=True)
            self._spawn()

    def job(self, indir, out, top_k=15, seed=1234, timeout=1200):
        self._ensure_alive()
        self.p.stdin.write(f"JOB {indir} {out} {top_k} {seed}\n")
        self.p.stdin.flush()
        t0 = time.time()
        while True:
            if time.time() - t0 > timeout:
                raise TimeoutError("runner JOB timeout")
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("runner serve pipe closed")
            if line.startswith("JOB-DONE"):
                parts = dict(kv.split("=") for kv in line.strip().split()[1:])
                return int(parts["ntoks"]), float(parts["ms"])
            if line.startswith("JOB-ERR"):
                raise RuntimeError(f"runner: {line.strip()}")

    def job_stream(self, indir, out, top_k=15, seed=1234, on_token=None, timeout=1200):
        """JOB 并逐 token 回调（RUNNER_STREAM=1 时 runner 每步打印 TOK <id>）。"""
        self._ensure_alive()
        self.p.stdin.write(f"JOB {indir} {out} {top_k} {seed}\n")
        self.p.stdin.flush()
        t0 = time.time()
        while True:
            if time.time() - t0 > timeout:
                raise TimeoutError("runner JOB timeout")
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("runner serve pipe closed")
            if line.startswith("TOK ") and on_token is not None:
                try:
                    on_token(int(line.split()[1]))
                except ValueError:
                    pass
            elif line.startswith("JOB-DONE"):
                parts = dict(kv.split("=") for kv in line.strip().split()[1:])
                return int(parts["ntoks"]), float(parts["ms"])
            elif line.startswith("JOB-ERR"):
                raise RuntimeError(f"runner: {line.strip()}")

    def close(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


class TTSReq(BaseModel):
    text: str
    spk: str = "examples/laffey.mp3"
    prompt_audio: str = "examples/AnAn.ogg"
    prompt_text: str = "ちが……ちがう。レイア、貴様は間違っている。"
    language: str = "ja"
    top_k: int = 15  # greedy 短句易塌成静音，默认随机采样
    seed: int = 1234
    cut_minlen: int = 30  # 段越少总耗时越短；太大单段易撞 K=1024 上限被截断
    cut_mute: float = 0.4  # 段间静音秒数
    sovits_model: str | None = None  # 微调 .pth 即插即用；None=默认
    # GPT 微调切换=换 gguf，需重启服务（GPT_GGUF 环境变量指向新 gguf，多模型=多端口实例）


app = FastAPI(title="GSV-TTS-Lite Intel API")
_lock = threading.Lock()
_prog = {"state": "idle", "current": 0, "total": 0}  # 原子替换读，无锁可读
_rsrv: RunnerServer | None = None
_tts = None
_tts_x = None


def _get_tts():
    global _tts, _tts_x
    if _tts is None:
        from gsv_tts import TTS
        _tts = TTS(device="cpu", dtype="float32")
        _tts.load_gpt_model()
        _tts.load_sovits_model()
        _tts_x = TTS(device="xpu", dtype="float32")
        _tts_x.load_sovits_model()
    return _tts, _tts_x


@app.on_event("startup")
def _startup():
    global _rsrv
    _get_tts()
    _rsrv = RunnerServer()


@app.on_event("shutdown")
def _shutdown():
    if _rsrv is not None:
        _rsrv.close()


@app.get("/healthz")
def healthz():
    ok = _rsrv is not None and _rsrv.p.poll() is None
    return {"runner_alive": ok}


def _synth(req: TTSReq) -> str:
    from gsv_tts.TTS import get_phones_and_bert
    from gsv_tts.TextProcessor import cut_text
    import soundfile as sf

    tts, tts_x = _get_tts()
    tts.init_language_module(req.language)
    sv_path = req.sovits_model or str(tts.default_sovits_path)
    sovits_m, ge = tts_x._prepare_sovits_resources(sv_path, req.spk)
    svx = sovits_m.vq_model.eval()
    gpt_m, prompt, phones1, bert1 = tts._prepare_gpt_resources(
        str(tts.default_gpt_path), req.prompt_audio, req.prompt_text, req.language)

    cuts = cut_text(req.text, req.cut_minlen)
    if not cuts:
        raise ValueError("empty text after cut")
    global _prog
    _prog = {"state": "running", "current": 0, "total": len(cuts)}
    chunks = []
    with tempfile.TemporaryDirectory(prefix="api_job_") as wd:
        for i, tc in enumerate(cuts):
            _prog = {"state": "running", "current": i + 1, "total": len(cuts)}
            phones2, _, bert2, _ = get_phones_and_bert(tc, tts.tts_config, req.language)
            x = torch.LongTensor(phones1 + phones2).unsqueeze(0)
            bert = torch.cat([bert1, bert2]).unsqueeze(0)
            jd = os.path.join(wd, f"c{i}")
            os.makedirs(jd)
            np.ascontiguousarray(x.numpy().astype(np.int64).reshape(-1)).tofile(os.path.join(jd, "x_ids.bin"))
            np.ascontiguousarray(prompt.numpy().astype(np.int64).reshape(-1)).tofile(os.path.join(jd, "y.bin"))
            np.ascontiguousarray(bert.numpy()).tofile(os.path.join(jd, "bert.bin"))
            np.array([x.shape[1], prompt.shape[1]], dtype=np.int64).tofile(os.path.join(jd, "xlen_ylen.bin"))
            tokbin = os.path.join(jd, "tokens.bin")
            _rsrv.job(jd + "/", tokbin, top_k=req.top_k, seed=req.seed + i)
            toks = np.fromfile(tokbin, np.int64)
            # torch 切片规则复刻：去 t0，首个 EOS 截断（driver.py:130）
            seq = toks[1:]
            eos = np.where(seq == EOS)[0]
            if len(eos):
                seq = seq[:eos[0]]
            with torch.inference_mode():
                codes = torch.from_numpy(seq).unsqueeze(0).unsqueeze(0).to("xpu")
                ph2 = torch.LongTensor(phones2).unsqueeze(0).to("xpu")
                audio, _ = svx.decode(codes, ph2, ge.to("xpu"), cuda_graph=False, noise_scale=0)
            chunks.append(audio.detach().cpu().numpy().reshape(-1))
    mute = np.zeros(int(req.cut_mute * SR), dtype=np.float32)
    wav = chunks[0]
    for c in chunks[1:]:
        wav = np.concatenate([wav, mute, c])
    out = os.path.join(tempfile.gettempdir(), f"api_tts_{int(time.time()*1000)}.wav")
    sf.write(out, wav, SR)
    _prog = {"state": "idle", "current": 0, "total": 0}
    return out


@app.get("/progress")
def progress():
    return _prog


class StreamReq(TTSReq):
    stream_chunk: int = 25  # 每攒够多少 token 解码一窗（同 TTS.infer_stream）
    overlap_len: int = 5


def _synth_stream(req: StreamReq):
    """逐 token 流式：runner TOK 行驱动，每 25 token 一窗 SoVITS 解码。yield f32le PCM。"""
    from gsv_tts.TTS import get_phones_and_bert
    from gsv_tts.TextProcessor import cut_text

    tts, tts_x = _get_tts()
    tts.init_language_module(req.language)
    sv_path = req.sovits_model or str(tts.default_sovits_path)
    sovits_m, ge = tts_x._prepare_sovits_resources(sv_path, req.spk)
    svx = sovits_m.vq_model.eval()
    overlap_samples = req.overlap_len * svx.samples_per_frame
    gpt_m, prompt, phones1, bert1 = tts._prepare_gpt_resources(
        str(tts.default_gpt_path), req.prompt_audio, req.prompt_text, req.language)
    cuts = cut_text(req.text, req.cut_minlen)
    if not cuts:
        raise ValueError("empty text after cut")
    global _prog
    _prog = {"state": "streaming", "current": 0, "total": len(cuts)}
    gex = ge.to("xpu")
    with tempfile.TemporaryDirectory(prefix="api_stream_") as wd:
        for i, tc in enumerate(cuts):
            _prog = {"state": "streaming", "current": i + 1, "total": len(cuts)}
            phones2, _, bert2, _ = get_phones_and_bert(tc, tts.tts_config, req.language)
            x = torch.LongTensor(phones1 + phones2).unsqueeze(0)
            bert = torch.cat([bert1, bert2]).unsqueeze(0)
            jd = os.path.join(wd, f"c{i}")
            os.makedirs(jd)
            np.ascontiguousarray(x.numpy().astype(np.int64).reshape(-1)).tofile(os.path.join(jd, "x_ids.bin"))
            np.ascontiguousarray(prompt.numpy().astype(np.int64).reshape(-1)).tofile(os.path.join(jd, "y.bin"))
            np.ascontiguousarray(bert.numpy()).tofile(os.path.join(jd, "bert.bin"))
            np.array([x.shape[1], prompt.shape[1]], dtype=np.int64).tofile(os.path.join(jd, "xlen_ylen.bin"))
            ph2 = torch.LongTensor(phones2).unsqueeze(0).to("xpu")

            sampled = []  # 含 t0；可用序列 = sampled[1:]
            emitted = 0   # 已解码的可用 token 数
            vs = 0        # valid_start_idx（TTS.infer_stream 同规则）
            last_ov = None
            first_win = True
            import queue
            q = queue.Queue()

            def decode_window(codes_np, is_final):
                nonlocal vs, last_ov, first_win
                with torch.inference_mode():
                    codes = torch.from_numpy(codes_np).unsqueeze(0).unsqueeze(0).to("xpu")
                    audio, attn = svx.decode(codes, ph2, gex, noise_scale=0, cuda_graph=False,
                                             stream_mode=True, valid_start_idx=vs,
                                             overlap_len=req.overlap_len)
                    if last_ov is not None:
                        audio, _ = tts_x._sola_algorithm(last_ov, audio, overlap_samples)
                    last_ov = audio[:, :, -overlap_samples:].clone()
                    if not is_final:
                        audio = audio[:, :, :-overlap_samples]
                        attn = attn[:, :-req.overlap_len, :]
                        vs = attn.shape[1]
                    audio = audio[0, 0, :]
                    if first_win:
                        head = tts_x._find_head_threshold_offsets(audio)
                        audio = audio[head:]
                        first_win = False
                    if is_final:
                        sil = torch.zeros((int(req.cut_mute * 32000),), dtype=audio.dtype, device=audio.device)
                        audio = torch.concatenate([audio, sil])
                    return audio.float().cpu().numpy()

            def on_token(tid):
                nonlocal emitted
                sampled.append(tid)
                usable = len(sampled) - 1  # 去 t0
                # EOS 出现后停发新窗，留给 final flush
                if usable - emitted >= req.stream_chunk and EOS not in sampled[1:]:
                    seq = np.array(sampled[1:], dtype=np.int64)
                    q.put(decode_window(seq, False))
                    emitted = usable

            _rsrv.job_stream(jd + "/", os.path.join(jd, "tokens.bin"),
                             top_k=req.top_k, seed=req.seed + i, on_token=on_token)
            while not q.empty():
                yield q.get().astype(np.float32).tobytes()
            # final flush：首个 EOS 截断（同非流式规则）
            seq = np.array(sampled[1:], dtype=np.int64)
            eos = np.where(seq == EOS)[0]
            if len(eos):
                seq = seq[:eos[0]]
            if len(seq) > emitted:
                pcm = decode_window(seq, True)
                yield pcm.astype(np.float32).tobytes()
    _prog = {"state": "idle", "current": 0, "total": 0}


@app.post("/tts/stream")
def tts_stream(req: StreamReq):
    from fastapi.responses import StreamingResponse
    global _prog
    if not req.text or not req.text.strip():
        raise HTTPException(400, "empty text")
    _lock.acquire()  # 全程串行，生成器耗尽才释放
    def gen():
        try:
            yield from _synth_stream(req)
        except Exception as e:
            print(f"stream failed: {e}", flush=True)
            raise
        finally:
            global _prog
            _prog = {"state": "idle", "current": 0, "total": 0}
            _lock.release()
    return StreamingResponse(gen(), media_type="application/octet-stream")


@app.post("/tts")
def tts_endpoint(req: TTSReq):
    global _prog
    if not req.text or not req.text.strip():
        raise HTTPException(400, "empty text")
    with _lock:  # runner 单通道 + XPU 状态，全局串行
        try:
            path = _synth(req)
        except Exception as e:
            _prog = {"state": "idle", "current": 0, "total": 0}
            raise HTTPException(500, f"synth failed: {e}")
    return FileResponse(path, media_type="audio/wav", filename=os.path.basename(path))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "9880")))
