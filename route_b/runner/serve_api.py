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
import hashlib
import json
import contextlib
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


def _maybe_override_prompt(prompt):
    """PROMPT_Y_FILE 有值则用盘上验过的好 y（prompt 抽签波动大，好 y 落盘复用）。"""
    _pyf = os.environ.get("PROMPT_Y_FILE")
    if _pyf and os.path.exists(_pyf):
        prompt = torch.from_numpy(np.fromfile(_pyf, np.int64).reshape(1, -1))
        print(f"prompt y overridden from {_pyf}", flush=True)
    return prompt


def _resolve_prompt(prompt, audio_path, prompt_text, lang):
    """serving 用 y：显式文件优先，否则自动优选（落盘缓存）。"""
    _pyf = os.environ.get("PROMPT_Y_FILE")
    if _pyf and os.path.exists(_pyf):
        return _maybe_override_prompt(prompt)
    try:
        ybin = ensure_good_y(audio_path, prompt_text, lang)
        prompt = torch.from_numpy(np.fromfile(ybin, np.int64).reshape(1, -1))
        print(f"prompt y auto-selected: {ybin}", flush=True)
    except Exception as e:
        print(f"prompt y auto-select failed ({e}), using prepared", flush=True)
    return prompt


def strip_prompt_echo(sampled, prompt, maxk=12, mink=3):
    """去头部 prompt 回声：sampled 头部若为 prompt 尾的续写则剥掉（只剥一次）。
    sampled=去t0 EOS截断后；prompt=y 的 numpy。serving 质量修正，parity 工具不用。"""
    n = min(len(sampled), len(prompt), maxk)
    for k in range(n, mink - 1, -1):
        if list(sampled[:k]) == list(prompt[-k:]):
            return sampled[k:]
    return sampled


_Y_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "gsv", "prompt_y")
_Y_SELECT_N = int(os.environ.get("Y_SELECT_N", "1"))  # 抽签并不可靠（常全 B），默认单次+告警，人工 curated 为主


def _y_cache_key(audio_path, prompt_text, lang):
    st = os.stat(audio_path)
    h = hashlib.sha1(
        f"{os.path.abspath(audio_path)}|{st.st_mtime_ns}|{st.st_size}|{prompt_text}|{lang}".encode()
    ).hexdigest()[:16]
    return os.path.join(_Y_CACHE_DIR, h + ".ybin"), os.path.join(_Y_CACHE_DIR, h + ".json")


def _y_score(ybin):
    a = np.fromfile(ybin, np.int64)
    if len(a) == 0:
        return -1.0, {}
    import collections
    c = collections.Counter(a.tolist())
    return len(c) / len(a), {"len": len(a), "uniq": len(c),
                             "sil239_frac": round(c.get(239, 0) / len(a), 4)}


def ensure_good_y(audio_path, prompt_text="?", lang="?", n_try=None):
    """新音频 y 自动优选：extract_y × N（并行子进程），uniq 占比最高者落盘缓存。
    同进程重复提取无意义（0 漂移），必须跨进程抽签。返回 y 文件路径。"""
    n_try = n_try or _Y_SELECT_N
    ybin, meta = _y_cache_key(audio_path, prompt_text, lang)
    if os.path.exists(ybin):
        print(f"prompt y cache hit: {ybin}", flush=True)
        return ybin
    os.makedirs(_Y_CACHE_DIR, exist_ok=True)
    _ensure_oneapi_env()
    cands = []
    # 串行抽签（并行同负载调度一致，无意义）+ 轮换线程数逼出调度差异
    thread_cycle = ["16", "1", "4", "8", "2"]
    stats = []
    import shutil
    for k in range(n_try):
        out = os.path.join(tempfile.gettempdir(), f"ysel_{os.getpid()}_{k}.bin")
        env = dict(os.environ)
        env["SYCL_CACHE_PERSISTENT"] = "0"
        env["OMP_NUM_THREADS"] = thread_cycle[k % len(thread_cycle)]
        env["MKL_NUM_THREADS"] = thread_cycle[k % len(thread_cycle)]
        try:
            r = subprocess.run(
                [sys.executable, os.path.join(_HERE, "extract_y.py"), audio_path, out, "xpu"],
                env=env, capture_output=True, text=True, timeout=900)
            print(f"y-select cand {k} (threads={thread_cycle[k % len(thread_cycle)]}): "
                  f"{r.stdout.strip().splitlines()[-1] if r.stdout.strip() else 'no-output'}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"y-select cand {k}: TIMEOUT", flush=True)
            continue
        if r.returncode != 0 or not os.path.exists(out):
            print(f"y-select cand {k}: FAILED rc={r.returncode}", flush=True)
            continue
        score, meta_k = _y_score(out)
        stats.append((score, k, out, meta_k))
        cands.append(out)
    if not stats:
        raise RuntimeError("y-select: all extractions failed")
    stats.sort(key=lambda t: -t[0])
    best_score, best_k, best_out, best_meta = stats[0]
    shutil.copyfile(best_out, ybin)
    meta_doc = dict(best_meta)
    meta_doc.update({"audio": audio_path, "prompt_text": prompt_text, "lang": lang,
                     "score": round(best_score, 4), "n_try": n_try,
                     "all": [{"score": round(s, 4), **m} for s, _, _, m in stats]})
    with open(meta, "w") as f:
        json.dump(meta_doc, f, ensure_ascii=False, indent=1)
    for c in cands:
        try:
            os.remove(c)
        except OSError:
            pass
    print(f"prompt y selected: {ybin} score={best_score:.3f} {best_meta}", flush=True)
    if best_score < 0.35:
        print(f"WARNING: y looks collapsed (uniq/len={best_score:.3f}); "
              f"expect rambling — re-extract or curate manually, see INTEL_XPU.md", flush=True)
    return ybin


_POOL_SIZE = max(1, int(os.environ.get("RUNNER_POOL", "2")))
_pool = []
_pool_guard = threading.Lock()
_pool_next = 0
_prog = {"state": "idle", "current": 0, "total": 0}  # 原子替换读，无锁可读
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


@contextlib.contextmanager
def _lease_runner():
    """轮询租一个空闲 runner；全忙则阻塞等第一个。请求级并行，长文本不再独占全服。"""
    global _pool_next
    with _pool_guard:
        order = [_pool[(_pool_next + k) % len(_pool)] for k in range(len(_pool))]
        _pool_next = (_pool_next + 1) % len(_pool)
    for srv in order:
        if srv.lock.acquire(blocking=False):
            break
    else:
        srv = order[0]
        srv.lock.acquire()
    srv.busy = True
    try:
        yield srv
    finally:
        srv.busy = False
        srv.lock.release()


def _warmup():
    print(f"warmup: tiny synth x{len(_pool)} (JIT all runners)...", flush=True)
    for k in range(len(_pool)):
        p = _synth(TTSReq(text="あ。"))
        try:
            os.remove(p)
        except OSError:
            pass
        print(f"warmup {k + 1}/{len(_pool)} ok", flush=True)


from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(app):
    global _pool
    _get_tts()
    for i in range(_POOL_SIZE):
        _pool.append(RunnerServer(i))
    print(f"runner pool ready: {len(_pool)}", flush=True)
    try:
        _warmup()
    except Exception as e:
        print(f"warmup failed (non-fatal): {e}", flush=True)
    yield
    for s in _pool:
        s.close()


app = FastAPI(title="GSV-TTS-Lite Intel API", lifespan=_lifespan)
_rsrv: RunnerServer | None = None  # 兼容旧引用，实际走 _pool


@app.get("/healthz")
def healthz():
    info = [{"alive": s.p.poll() is None, "busy": s.busy} for s in _pool]
    return {"runners": info, "runner_alive": any(i["alive"] for i in info)}


def _synth(req: TTSReq) -> str:
    from gsv_tts.TTS import get_phones_and_bert
    from gsv_tts.TextProcessor import cut_text
    import soundfile as sf

    tts, tts_x = _get_tts()
    tts.init_language_module(req.language, req.prompt_language)
    sv_path = req.sovits_model or str(tts.default_sovits_path)
    sovits_m, ge = tts_x._prepare_sovits_resources(sv_path, req.spk)
    svx = sovits_m.vq_model.eval()
    gpt_m, prompt, phones1, bert1 = tts._prepare_gpt_resources(
        str(tts.default_gpt_path), req.prompt_audio, req.prompt_text, req.prompt_language)
    prompt = _resolve_prompt(prompt, req.prompt_audio, req.prompt_text, req.prompt_language)

    print(f"tts lang={req.language} prompt_lang={req.prompt_language} text={req.text[:20]!r}", flush=True)
    cuts = cut_text(req.text, req.cut_minlen)
    if not cuts:
        raise ValueError("empty text after cut")
    global _prog
    _prog = {"state": "running", "current": 0, "total": len(cuts)}
    chunks = []

    def _one_chunk(tc, i, wd, seed):
        phones2, _, bert2, norm_txt = get_phones_and_bert(tc, tts.tts_config, req.language)
        x = torch.LongTensor(phones1 + phones2).unsqueeze(0)
        bert = torch.cat([bert1, bert2]).unsqueeze(0)
        jd = os.path.join(wd, f"c{i}")
        os.makedirs(jd, exist_ok=True)
        np.ascontiguousarray(x.numpy().astype(np.int64).reshape(-1)).tofile(os.path.join(jd, "x_ids.bin"))
        np.ascontiguousarray(prompt.numpy().astype(np.int64).reshape(-1)).tofile(os.path.join(jd, "y.bin"))
        np.ascontiguousarray(bert.numpy()).tofile(os.path.join(jd, "bert.bin"))
        np.array([x.shape[1], prompt.shape[1]], dtype=np.int64).tofile(os.path.join(jd, "xlen_ylen.bin"))
        tokbin = os.path.join(jd, "tokens.bin")
        with _lease_runner() as _srv:
            _srv.job(jd + "/", tokbin, top_k=req.top_k, seed=seed, rep=req.repetition_penalty)
        toks = np.fromfile(tokbin, np.int64)
        # torch 切片规则复刻：去 t0，首个 EOS 截断（driver.py:130），再剥 prompt 回声
        seq = toks[1:]
        eos = np.where(seq == EOS)[0]
        if len(eos):
            seq = seq[:eos[0]]
        n0 = len(seq)
        seq = strip_prompt_echo(seq, prompt.numpy().reshape(-1))
        if len(seq) != n0:
            print(f"echo stripped: {n0} -> {len(seq)}", flush=True)
        with torch.inference_mode():
            codes = torch.from_numpy(seq).unsqueeze(0).unsqueeze(0).to("xpu")
            ph2 = torch.LongTensor(phones2).unsqueeze(0).to("xpu")
            audio, _ = svx.decode(codes, ph2, ge.to("xpu"), cuda_graph=False, noise_scale=req.noise_scale)
            audio = audio[0, 0, :]
            audio = audio[tts_x._find_head_threshold_offsets(audio):]
        return audio.detach().cpu().numpy().reshape(-1), norm_txt

    import random as _rd
    base_seed = req.seed if req.seed is not None else _rd.randint(0, 2**31 - 1)
    with tempfile.TemporaryDirectory(prefix="api_job_") as wd:
        for i, tc in enumerate(cuts):
            _prog = {"state": "running", "current": i + 1, "total": len(cuts)}
            attempts = 1 + (req.max_retries if req.verify else 0)
            best = None
            for att in range(attempts):
                seed = base_seed + att * 7919 + i
                audio, norm_txt = _one_chunk(tc, f"{i}a{att}", wd, seed)
                best = audio
                if not req.verify:
                    break
                if _chunk_passes(audio, norm_txt, req.language):
                    break
                print(f"chunk {i} attempt {att} failed gate, retrying", flush=True)
            chunks.append(best)
    mute = np.zeros(int(req.cut_mute * SR), dtype=np.float32)
    wav = chunks[0]
    for c in chunks[1:]:
        wav = np.concatenate([wav, mute, c])
    out = os.path.join(tempfile.gettempdir(), f"api_tts_{int(time.time()*1000)}.wav")
    sf.write(out, wav, SR)
    _prog = {"state": "idle", "current": 0, "total": 0}
    return out


_asr_model = None


def _get_asr():
    global _asr_model
    if _asr_model is None:
        from faster_whisper import WhisperModel
        _asr_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _asr_model


import re as _re
_PUNCT = _re.compile(r'[\s。、，「」『』…！？!?,.:;~・…、·\'"“”‘’（）()【】\-]+')


def _norm_txt(s):
    return _PUNCT.sub("", s)


def _chunk_passes(audio_np, norm_text, language):
    """ASR 门：chunk 转写头部须出现目标头（25 字内），否则判泄漏/复读。异常时放行不挡请求。"""
    try:
        m = _get_asr()
    except Exception as e:
        print(f"asr gate unavailable ({e}), passing", flush=True)
        return True
    try:
        clip = audio_np[:45 * SR] if len(audio_np) > 45 * SR else audio_np
        n = int(len(clip) * 16000 / SR)
        a16 = np.interp(np.linspace(0, len(clip), n), np.arange(len(clip)),
                        clip.astype(np.float64)).astype(np.float32)
        kw = {"beam_size": 5}
        if language in ("ja", "zh", "en"):
            kw["language"] = language
        segs, _ = m.transcribe(a16, **kw)
        trans = _norm_txt("".join(s.text for s in segs))
        tgt = _norm_txt(norm_text)[:12]
        if len(tgt) < 4:
            return True
        idx = trans.find(tgt)
        ok = 0 <= idx <= 25
        print(f"asr gate: tgt={tgt[:12]!r} idx={idx} -> {'PASS' if ok else 'FAIL'} "
              f"trans={trans[:40]!r}", flush=True)
        return ok
    except Exception as e:
        print(f"asr gate error ({e}), passing", flush=True)
        return True


@app.get("/progress")
def progress():
    return _prog


@app.post("/refresh_prompt")
def refresh_prompt(prompt_audio: str = "examples/AnAn.ogg",
                   prompt_text: str = "?", lang: str = "?"):
    """删缓存重提 y（抽签，返回统计；uniq 高者赢，服务重启或下次请求生效）。"""
    import glob as _glob
    deleted = 0
    for f in _glob.glob(os.path.join(_Y_CACHE_DIR, "*")):
        if ".good." in os.path.basename(f):
            continue  # curated 资产不动
        try:
            os.remove(f)
            deleted += 1
        except OSError:
            pass
    ybin = ensure_good_y(prompt_audio, prompt_text, lang)
    return {"ybin": ybin, "deleted": deleted}


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

    def __init__(self, idx=0):
        _ensure_oneapi_env()
        self.idx = idx
        self.lock = threading.Lock()
        self.busy = False
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
                raise RuntimeError(f"runner[{self.idx}] serve exited during init")
            if "SERVE-READY" in line:
                print(f"runner[{self.idx}] ready", flush=True)
                return

    def _ensure_alive(self):
        if self.p.poll() is not None:  # runner 死了就地重拉（约1s + 下一 JOB 重新 JIT）
            print("runner dead, respawning...", flush=True)
            self._spawn()

    def job(self, indir, out, top_k=15, seed=1234, rep=1.35, timeout=1200):
        self._ensure_alive()
        self.p.stdin.write(f"JOB {indir} {out} {top_k} {seed} {rep}\n")
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

    def job_stream(self, indir, out, top_k=15, seed=1234, rep=1.35, on_token=None, timeout=1200):
        """JOB 并逐 token 回调（RUNNER_STREAM=1 时 runner 每步打印 TOK <id>）。"""
        self._ensure_alive()
        self.p.stdin.write(f"JOB {indir} {out} {top_k} {seed} {rep}\n")
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
    prompt_language: str = "ja"  # 必须与 prompt 音频/文本的真实语言一致，否则 prompt 条件全错
    top_k: int = 15  # greedy 短句易塌成静音，默认随机采样
    noise_scale: float = 0.5  # 上游默认；0=确定性但发 flat/金属（parity 工具仍用 0）
    repetition_penalty: float = 1.35
    seed: int | None = None  # None=每次随机；固定 seed 则回声等问题可复现
    verify: bool = False  # ASR 质量门：chunk 转写对不上目标头就重抽（最多 max_retries 次）
    max_retries: int = 2
    cut_minlen: int = 30  # 段越少总耗时越短；太大单段易撞 K=1024 上限被截断
    cut_mute: float = 0.4  # 段间静音秒数
    sovits_model: str | None = None  # 微调 .pth 即插即用；None=默认
    # GPT 微调切换=换 gguf，需重启服务（GPT_GGUF 环境变量指向新 gguf，多模型=多端口实例）


class StreamReq(TTSReq):
    stream_chunk: int = 25  # 每攒够多少 token 解码一窗（同 TTS.infer_stream）
    overlap_len: int = 5


def _synth_stream(req: StreamReq):
    """逐 token 流式：runner TOK 行驱动，每 25 token 一窗 SoVITS 解码。yield f32le PCM。"""
    from gsv_tts.TTS import get_phones_and_bert
    from gsv_tts.TextProcessor import cut_text

    tts, tts_x = _get_tts()
    tts.init_language_module(req.language, req.prompt_language)
    sv_path = req.sovits_model or str(tts.default_sovits_path)
    sovits_m, ge = tts_x._prepare_sovits_resources(sv_path, req.spk)
    svx = sovits_m.vq_model.eval()
    overlap_samples = req.overlap_len * svx.samples_per_frame
    gpt_m, prompt, phones1, bert1 = tts._prepare_gpt_resources(
        str(tts.default_gpt_path), req.prompt_audio, req.prompt_text, req.prompt_language)
    prompt = _resolve_prompt(prompt, req.prompt_audio, req.prompt_text, req.prompt_language)
    print(f"tts lang={req.language} prompt_lang={req.prompt_language} text={req.text[:20]!r}", flush=True)
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

            import random as _rd
            base_seed = req.seed if req.seed is not None else _rd.randint(0, 2**31 - 1)
            py = prompt.numpy().reshape(-1)
            sampled = []  # 含 t0；可用序列 = sampled[1:]
            emitted = 0   # 已解码的可用 token 数（去回声后起算）
            skip = None   # 回声长度，首窗/flush 前一次性判定
            vs = 0        # valid_start_idx（TTS.infer_stream 同规则）
            last_ov = None
            first_win = True
            import queue
            q = queue.Queue()

            def decode_window(codes_np, is_final):
                nonlocal vs, last_ov, first_win
                with torch.inference_mode():
                    codes = torch.from_numpy(codes_np).unsqueeze(0).unsqueeze(0).to("xpu")
                    audio, attn = svx.decode(codes, ph2, gex, noise_scale=req.noise_scale, cuda_graph=False,
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
                nonlocal emitted, skip
                sampled.append(tid)
                if skip is None and len(sampled) - 1 >= 12:
                    n0 = len(sampled) - 1
                    skip = n0 - len(strip_prompt_echo(np.array(sampled[1:]), py))
                    if skip:
                        print(f"echo stripped: skip={skip}", flush=True)
                base = 1 + (skip or 0)
                usable = len(sampled) - base
                # EOS 出现后停发新窗，留给 final flush
                if usable - emitted >= req.stream_chunk and EOS not in sampled[base:]:
                    seq = np.array(sampled[base:], dtype=np.int64)
                    q.put(decode_window(seq, False))
                    emitted = usable

            with _lease_runner() as _srv:
                _srv.job_stream(jd + "/", os.path.join(jd, "tokens.bin"),
                                top_k=req.top_k, seed=base_seed + i, rep=req.repetition_penalty,
                                on_token=on_token)
            while not q.empty():
                yield q.get().astype(np.float32).tobytes()
            # final flush：首个 EOS 截断 + 回声剥离（同非流式规则）
            seq = np.array(sampled[1:], dtype=np.int64)
            eos = np.where(seq == EOS)[0]
            if len(eos):
                seq = seq[:eos[0]]
            if skip is None:
                skip = len(seq) - len(strip_prompt_echo(seq, py))
            seq = seq[skip:]
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
    # 无全局锁：并发由 runner pool 承接，生成器耗尽即收尾

    def gen():
        try:
            yield from _synth_stream(req)
        except Exception as e:
            print(f"stream failed: {e}", flush=True)
            raise
        finally:
            global _prog
            _prog = {"state": "idle", "current": 0, "total": 0}
    return StreamingResponse(gen(), media_type="application/octet-stream")


@app.post("/tts")
def tts_endpoint(req: TTSReq):
    global _prog
    if not req.text or not req.text.strip():
        raise HTTPException(400, "empty text")
    try:
        path = _synth(req)
    except Exception as e:
        _prog = {"state": "idle", "current": 0, "total": 0}
        raise HTTPException(500, f"synth failed: {e}")
    return FileResponse(path, media_type="audio/wav", filename=os.path.basename(path))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "9880")))
