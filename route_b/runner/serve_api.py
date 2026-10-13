"""Intel 路径 API 服务：TTS特征准备(CPU) -> runner --serve 常驻(GPT) -> SoVITS(XPU) -> wav。

串行合成（全局锁）：runner serve 单 JOB 通道 + XPU 状态保护，个人用量够用。
多 chunk：复用 TTS.infer 的 cut_text，长文本分段后静音拼接。
"""
import sys, os
import faulthandler
faulthandler.enable()  # native 崩溃时吐 Python 栈到日志（连环 heap/SEGV 取证中）
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

from fastapi import FastAPI, HTTPException, File, Form, UploadFile
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
    return prompt


def _resolve_prompt(prompt, audio_path, prompt_text, lang, resyn=None):
    """serving 用 y：显式文件优先（仅音频 basename 匹配时）否则自动优选（落盘缓存）。
    resyn 回调做内容验证。"""
    _pyf = os.environ.get("PROMPT_Y_FILE")
    _pya = os.environ.get("PROMPT_Y_AUDIO", "AnAn.ogg")
    if _pyf and os.path.exists(_pyf) and os.path.basename(audio_path) == os.path.basename(_pya):
        prompt = torch.from_numpy(np.fromfile(_pyf, np.int64).reshape(1, -1))
        print(f"prompt y overridden from {_pyf}", flush=True)
        return prompt
    try:
        ybin, _ = ensure_good_y(audio_path, prompt_text, lang, resyn=resyn)
        if ybin is None:
            print("prompt y auto-select rejected, using prepared", flush=True)
            return prompt
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


_TARGET_DBFS = float(os.environ.get("GSV_TARGET_DBFS", "-20.0"))  # 输出响度目标；设 0 关闭


def _normalize_loudness(audio):
    """RMS 归一化 + 峰值保护（不削波）。克隆 prompt y 偏糊时输出能量低，靠这层拉齐。"""
    if not _TARGET_DBFS:
        return audio
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return audio
    rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    if rms < 1e-7:
        return audio
    gain = (10.0 ** (_TARGET_DBFS / 20.0)) / rms
    peak = float(np.abs(a).max())
    if peak > 0 and peak * gain > 0.99:
        gain = 0.99 / peak
    return (a * gain).astype(np.float32)


_Y_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "gsv", "prompt_y")
_Y_SELECT_N = int(os.environ.get("Y_SELECT_N", "3"))  # resyn 早停，有好即收


def _y_cache_key(audio_path, prompt_text="", lang=""):
    # 内容哈希（y 只与音频有关；上传随机名/转写文本差异都不应影响命中）
    # 后端标记：torch/ov 提法不同值，串味必糊
    st = os.stat(audio_path)
    be = os.environ.get("GSV_HUBERT_BACKEND", "torch").lower()
    h = hashlib.sha1(f"{st.st_size}|{be}|".encode())
    with open(audio_path, "rb") as f:
        h.update(f.read(1 << 20))
        f.seek(max(0, st.st_size - (1 << 20)))
        h.update(f.read(1 << 20))
    key = h.hexdigest()[:16]
    return os.path.join(_Y_CACHE_DIR, key + ".ybin"), os.path.join(_Y_CACHE_DIR, key + ".json")


def _y_score(ybin):
    a = np.fromfile(ybin, np.int64)
    if len(a) == 0:
        return -1.0, {}
    import collections
    c = collections.Counter(a.tolist())
    return len(c) / len(a), {"len": len(a), "uniq": len(c),
                             "sil239_frac": round(c.get(239, 0) / len(a), 4)}


def _resyn_score(ybin, prompt_text, lang, svx, ge, tts):
    """y 质检：重合成→转写→与 prompt 文本模糊比。返回 ratio（0~1）。"""
    import soundfile as sf
    from gsv_tts.TTS import get_phones_and_bert
    y = np.fromfile(ybin, np.int64)
    phones, _, _, _ = get_phones_and_bert(prompt_text, tts.tts_config, lang)
    with torch.inference_mode():
        audio, _ = svx.decode(torch.from_numpy(y).unsqueeze(0).unsqueeze(0).to("xpu"),
                              torch.LongTensor(phones).unsqueeze(0).to("xpu"),
                              ge.to("xpu") if isinstance(ge, torch.Tensor) else ge,
                              cuda_graph=False, noise_scale=0)
    a = audio.detach().cpu().numpy().reshape(-1)
    n = int(len(a) * 16000 / SR)
    a16 = np.interp(np.linspace(0, len(a), n), np.arange(len(a)), a).astype(np.float32)
    m = _get_asr()
    kw = {"beam_size": 5}
    if lang in ("ja", "zh", "en"):
        kw["language"] = lang
    segs, _ = m.transcribe(a16, **kw)
    from difflib import SequenceMatcher
    t = _norm_txt("".join(s.text for s in segs))
    ref = _norm_txt(prompt_text)
    r = SequenceMatcher(None, t, ref).ratio() if ref else 0.0
    print(f"y resyn check: ratio={r:.3f} trans={t[:50]!r}", flush=True)
    return r


_Y_PASS = 0.25  # resyn-ratio 及格线（AnAn.good=0.35；0.05 级别必糊）


def ensure_good_y(audio_path, prompt_text="?", lang="?", n_try=None, resyn=None):
    """新音频 y 自动优选：extract（串行+线程轮换）→ 逐个 resyn 验证 → 过线即停。
    低于及格线不缓存（防投毒）。返回 (ybin_or_None, best_ratio)。
    resyn=None 时退化到 uniq 占比（老行为，阈值 0.35）。"""
    n_try = n_try or _Y_SELECT_N
    ybin, meta = _y_cache_key(audio_path, prompt_text, lang)
    if os.path.exists(ybin):
        print(f"prompt y cache hit: {ybin}", flush=True)
        return ybin, 1.0
    os.makedirs(_Y_CACHE_DIR, exist_ok=True)
    _ensure_oneapi_env()
    import shutil
    # 串行抽签（并行同负载调度一致，无意义）+ 轮换线程数逼出调度差异
    thread_cycle = ["16", "1", "4", "8", "2"]
    tried = []
    for k in range(n_try):
        out = os.path.join(tempfile.gettempdir(), f"ysel_{os.getpid()}_{k}.bin")
        env = dict(os.environ)
        env["SYCL_CACHE_PERSISTENT"] = "0"
        env["OMP_NUM_THREADS"] = thread_cycle[k % len(thread_cycle)]
        env["MKL_NUM_THREADS"] = thread_cycle[k % len(thread_cycle)]
        ok, score, meta_k, ratio = False, 0.0, {}, None
        try:
            r = subprocess.run(
                [sys.executable, os.path.join(_HERE, "extract_y.py"), audio_path, out, "xpu"],
                env=env, capture_output=True, text=True, timeout=900)
            print(f"y-select cand {k} (threads={thread_cycle[k % len(thread_cycle)]}): "
                  f"{r.stdout.strip().splitlines()[-1] if r.stdout.strip() else 'no-output'}", flush=True)
            if r.returncode != 0 or not os.path.exists(out):
                print(f"y-select cand {k}: FAILED rc={r.returncode}", flush=True)
                continue
            score, meta_k = _y_score(out)
            if resyn is not None:
                ratio = resyn(out)
                meta_k["resyn_ratio"] = round(ratio, 4)
            ok = True
        except subprocess.TimeoutExpired:
            print(f"y-select cand {k}: TIMEOUT", flush=True)
        except Exception as e:
            print(f"y-select cand {k}: ERROR ({e})", flush=True)
        finally:
            if ok:
                tried.append((ratio if ratio is not None else score, k, out, meta_k))
            elif os.path.exists(out):
                try:
                    os.remove(out)
                except OSError:
                    pass
        if ok and ratio is not None and ratio >= 0.3:
            print(f"y-select cand {k}: resyn {ratio:.3f} confident, stop", flush=True)
            break
        if ratio is not None and ratio >= 0.3:
            print(f"y-select cand {k}: resyn {ratio:.3f} confident, stop", flush=True)
            break
    if not tried:
        raise RuntimeError("y-select: all extractions failed")
    tried.sort(key=lambda t: -t[0])
    best, best_k, best_out, best_meta = tried[0]
    if resyn is not None and best < _Y_PASS:
        for _, _, c, _ in tried:
            try:
                os.remove(c)
            except OSError:
                pass
        print(f"y-select: best {best:.3f} < {_Y_PASS}, NOT cached (refusing to serve garbage). "
              f"retry later or supply cleaner source audio.", flush=True)
        return None, best
    shutil.copyfile(best_out, ybin)
    meta_doc = dict(best_meta)
    meta_doc.update({"audio": audio_path, "prompt_text": prompt_text, "lang": lang,
                     "score": round(best, 4), "n_try": n_try,
                     "all": [{"score": (round(s, 4) if isinstance(s, float) else s), **m}
                             for s, _, _, m in tried]})
    with open(meta, "w") as f:
        json.dump(meta_doc, f, ensure_ascii=False, indent=1)
    for _, _, c, _ in tried:
        if c != best_out:
            try:
                os.remove(c)
            except OSError:
                pass
    try:
        os.remove(best_out)
    except OSError:
        pass
    print(f"prompt y selected: {ybin} score={best:.3f} {best_meta}", flush=True)
    return ybin, best


_POOL_SIZE = max(1, int(os.environ.get("RUNNER_POOL", "2")))
_pool = []
_pool_guard = threading.Lock()
_pool_next = 0
_prog = {"state": "idle", "current": 0, "total": 0}  # 原子替换读，无锁可读
# XPU 串行锁：torch-xpu 解码 + SYCL runner 在并发下 native 崩（overlap 双杀实锤），整块串行。
# 单用户体感无差；并发请求排队不死。
_XPU_LOCK = threading.RLock()
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


def _prepare_sv(tts, tts_x, sv_path, spk):
    """svx = XPU 模型（解码用）；ge = 在 CPU 上计算再上卡。
    根因：XPU 版 _prepare_sovits_resources 算出的 ge 喂给 decode 会 native 段错误
    （隔离 A/B 实锤：CPU ge 3/3 干净、XPU ge 必崩，见 INTEL_XPU.md）。返回 (svx, ge_cpu)。"""
    if sv_path not in tts_x.sovits_models:
        tts_x.load_sovits_model(sv_path)
    svx = tts_x.sovits_models[sv_path].vq_model.eval()
    _, ge = tts._prepare_sovits_resources(sv_path, spk)  # 关键：CPU 算 ge
    return svx, ge


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
    svx, ge = _prepare_sv(tts, tts_x, sv_path, req.spk)
    gpt_m, prompt, phones1, bert1 = tts._prepare_gpt_resources(
        str(tts.default_gpt_path), req.prompt_audio, req.prompt_text, req.prompt_language)
    prompt = _resolve_prompt(
        prompt, req.prompt_audio, req.prompt_text, req.prompt_language,
        resyn=lambda yb: _resyn_score(yb, req.prompt_text, req.prompt_language, svx, ge, tts))

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
        # 取证+验毒：tokbin 落盘一份（/tmp 重启丢，仅供当轮排障）
        try:
            import shutil as _sh
            _sh.copyfile(tokbin, f"/tmp/opencode/last_tok_{i}a{seed}.bin")
        except Exception:
            pass
        # torch 切片规则复刻：去 t0，首个 EOS 截断（driver.py:130），再剥 prompt 回声
        seq = toks[1:]
        eos = np.where(seq == EOS)[0]
        if len(eos):
            seq = seq[:eos[0]]
        if len(seq) and (seq.min() < 0 or seq.max() > EOS):
            print(f"chunk {i}: POISON tokens (min={seq.min()} max={seq.max()} len={len(seq)}), "
                  f"seed={seed}, refusing", flush=True)
            return np.zeros(0, dtype=np.float32), norm_txt
        n0 = len(seq)
        seq = strip_prompt_echo(seq, prompt.numpy().reshape(-1))
        if len(seq) != n0:
            print(f"echo stripped: {n0} -> {len(seq)}", flush=True)
        if len(seq) == 0:
            # runner 直接 EOS：不进 SoVITS（空输入无意义），交由重试循环处理
            print(f"chunk {i}: empty generation (immediate EOS), seed={seed}", flush=True)
            return np.zeros(0, dtype=np.float32), norm_txt
        print(f"chunk {i}: codes={len(seq)} xlen={x.shape[1]} ylen={prompt.shape[1]}", flush=True)
        try:
            print(f"chunk {i}: xpu mem alloc={torch.xpu.memory_allocated()/2**20:.0f}MB "
                  f"reserved={torch.xpu.memory_reserved()/2**20:.0f}MB", flush=True)
        except Exception as e:
            print(f"chunk {i}: xpu mem query failed ({e})", flush=True)
        if len(seq) > 600:
            # runner 失控（无 EOS 狂奔）：500+ codes 的 SoVITS 解码是自杀行为，直接拒掉走重试
            print(f"chunk {i}: runaway generation ({len(seq)} codes), seed={seed}, refusing", flush=True)
            return np.zeros(0, dtype=np.float32), norm_txt
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
            # 空生成/门失败都重试；verify 只额外加 ASR 门，不决定是否重试
            attempts = 1 + req.max_retries
            best = None
            for att in range(attempts):
                seed = base_seed + att * 7919 + i
                audio, norm_txt = _one_chunk(tc, f"{i}a{att}", wd, seed)
                if len(audio) == 0:
                    print(f"chunk {i} attempt {att} empty generation, retrying", flush=True)
                    try:
                        torch.xpu.empty_cache()
                    except Exception:
                        pass
                    continue
                best = audio
                if not req.verify:
                    break
                if _chunk_passes(audio, norm_txt, req.language):
                    break
                print(f"chunk {i} attempt {att} failed gate, retrying", flush=True)
                try:
                    torch.xpu.empty_cache()  # 重试前放掉 XPU 缓存，防止重试风暴堆爆显存
                except Exception:
                    pass
            if best is None:
                raise HTTPException(500, f"chunk {i}: empty generation on all attempts "
                                         "(immediate EOS); try shorter text or another seed")
            chunks.append(best)
    mute = np.zeros(int(req.cut_mute * SR), dtype=np.float32)
    wav = chunks[0]
    for c in chunks[1:]:
        wav = np.concatenate([wav, mute, c])
    wav = _normalize_loudness(wav)
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


_VOICES_FILE = os.path.join(os.path.expanduser("~"), ".cache", "gsv", "voices.json")
_BUILTIN_VOICES = {
    "default": {"spk": "examples/laffey.mp3", "prompt_audio": "examples/AnAn.ogg",
                "prompt_text": "ちが……ちがう。レイア、貴様は間違っている。",
                "prompt_language": "ja", "sovits_model": None},
}


def _load_voices():
    try:
        with open(_VOICES_FILE) as f:
            d = json.load(f)
            return {**_BUILTIN_VOICES, **d}
    except (OSError, ValueError):
        return dict(_BUILTIN_VOICES)


def _save_voices(d):
    os.makedirs(os.path.dirname(_VOICES_FILE), exist_ok=True)
    user = {k: v for k, v in d.items() if k not in _BUILTIN_VOICES}
    with open(_VOICES_FILE, "w") as f:
        json.dump(user, f, ensure_ascii=False, indent=1)


@app.get("/voices")
def list_voices():
    return _load_voices()


class VoiceDef(BaseModel):
    name: str
    spk: str = "examples/laffey.mp3"
    prompt_audio: str = "examples/AnAn.ogg"
    prompt_text: str = "ちが……ちがう。レイア、貴様は間違っている。"
    prompt_language: str = "ja"
    sovits_model: str | None = None


@app.post("/voices")
def upsert_voice(v: VoiceDef):
    if not v.name or not v.name.strip():
        raise HTTPException(400, "empty name")
    for key in ("spk", "prompt_audio"):
        p = getattr(v, key)
        if not os.path.exists(p):
            raise HTTPException(400, f"file not found: {key}={p}")
    d = _load_voices()
    d[v.name] = {"spk": v.spk, "prompt_audio": v.prompt_audio, "prompt_text": v.prompt_text,
                 "prompt_language": v.prompt_language, "sovits_model": v.sovits_model}
    _save_voices(d)
    return {"ok": True, "name": v.name}


@app.delete("/voices/{name}")
def delete_voice(name: str):
    if name in _BUILTIN_VOICES:
        raise HTTPException(400, "cannot delete builtin")
    d = _load_voices()
    if name not in d:
        raise HTTPException(404, "no such voice")
    del d[name]
    _save_voices(d)
    return {"ok": True, "name": name}


def _apply_voice(req):
    """voice 预设填参：显式传的字段优先（model_fields_set），其余取预设。"""
    if not req.voice:
        return req
    d = _load_voices()
    if req.voice not in d:
        raise HTTPException(400, f"unknown voice: {req.voice}")
    p = d[req.voice]
    upd = {}
    for f in ("spk", "prompt_audio", "prompt_text", "prompt_language", "sovits_model"):
        if f not in req.model_fields_set:
            upd[f] = p[f]
    return req.model_copy(update=upd)


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
    ybin, ratio = ensure_good_y(prompt_audio, prompt_text, lang)
    return {"ybin": ybin, "ratio": ratio, "deleted": deleted}


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
    voice: str | None = None  # 音色预设名（/voices 管理）；显式字段优先于预设
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
    svx, ge = _prepare_sv(tts, tts_x, sv_path, req.spk)
    overlap_samples = req.overlap_len * svx.samples_per_frame
    gex = ge.to("xpu")
    gpt_m, prompt, phones1, bert1 = tts._prepare_gpt_resources(
        str(tts.default_gpt_path), req.prompt_audio, req.prompt_text, req.prompt_language)
    prompt = _resolve_prompt(
        prompt, req.prompt_audio, req.prompt_text, req.prompt_language,
        resyn=lambda yb: _resyn_score(yb, req.prompt_text, req.prompt_language, svx, gex, tts))
    print(f"tts lang={req.language} prompt_lang={req.prompt_language} text={req.text[:20]!r}", flush=True)
    cuts = cut_text(req.text, req.cut_minlen)
    if not cuts:
        raise ValueError("empty text after cut")
    global _prog
    _prog = {"state": "streaming", "current": 0, "total": len(cuts)}
    gated_head = False  # 首窗门：只验第 0 段首个音频窗（泄漏住头部）
    with tempfile.TemporaryDirectory(prefix="api_stream_") as wd:
        for i, tc in enumerate(cuts):
            _prog = {"state": "streaming", "current": i + 1, "total": len(cuts)}
            phones2, _, bert2, norm_txt = get_phones_and_bert(tc, tts.tts_config, req.language)
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
                b = q.get().astype(np.float32).tobytes()
                if i == 0 and req.verify and not gated_head:
                    gated_head = True
                    if not _chunk_passes(np.frombuffer(b, np.float32), norm_txt, req.language):
                        raise RuntimeError("stream head gate failed")
                yield b
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
                b = pcm.astype(np.float32).tobytes()
                if i == 0 and req.verify and not gated_head:
                    gated_head = True
                    if not _chunk_passes(np.frombuffer(b, np.float32), norm_txt, req.language):
                        raise RuntimeError("stream head gate failed")
                yield b
    _prog = {"state": "idle", "current": 0, "total": 0}


@app.post("/tts/stream")
def tts_stream(req: StreamReq):
    from fastapi.responses import StreamingResponse
    global _prog
    if not req.text or not req.text.strip():
        raise HTTPException(400, "empty text")
    req = _apply_voice(req)
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
    req = _apply_voice(req)
    try:
        path = _synth(req)
    except Exception as e:
        _prog = {"state": "idle", "current": 0, "total": 0}
        raise HTTPException(500, f"synth failed: {e}")
    return FileResponse(path, media_type="audio/wav", filename=os.path.basename(path))


@app.post("/vc")
async def vc_endpoint(file: UploadFile | None = File(None),
                      src_audio_path: str | None = Form(None),
                      spk: str = Form("examples/laffey.mp3"),
                      prompt_text: str = Form(""),
                      prompt_language: str = Form("auto"),
                      noise_scale: float = Form(0.5),
                      sovits_model: str | None = Form(None)):
    """变声：源音频内容 + 目标音色。file（上传）与 src_audio_path（服务端路径/URL）二选一；
    prompt_text 缺省时自动 ASR 转写源音频。"""
    import soundfile as sf
    suffix = ".wav"
    if file is not None:
        data = await file.read()
        if not data:
            raise HTTPException(400, "empty upload")
        suffix = os.path.splitext(file.filename or "")[1] or ".wav"
        src = os.path.join(tempfile.gettempdir(), f"vc_src_{int(time.time()*1000)}{suffix}")
        with open(src, "wb") as f:
            f.write(data)
    elif src_audio_path:
        if src_audio_path.startswith(("http://", "https://")):
            import urllib.request as _url
            src = os.path.join(tempfile.gettempdir(), f"vc_url_{int(time.time()*1000)}.wav")
            try:
                _url.urlretrieve(src_audio_path, src)
            except Exception as e:
                raise HTTPException(400, f"download failed: {e}")
        else:
            if not os.path.exists(src_audio_path):
                raise HTTPException(400, f"file not found: {src_audio_path}")
            src = src_audio_path
    else:
        raise HTTPException(400, "file or src_audio_path required")

    tts, tts_x = _get_tts()
    tts.init_language_module("ja", "zh", "en")
    sv_path = sovits_model or str(tts.default_sovits_path)
    svx, ge = _prepare_sv(tts, tts_x, sv_path, spk)
    sov_cpu = tts.sovits_models[sv_path]  # y 提取用 CPU sovits（extract_latent）
    if tts.cnhubert_model is None:
        from gsv_tts.GPT_SoVITS.Featurizer import CNHubert
        tts.cnhubert_model = CNHubert(tts.cnhubert_path, tts.tts_config)
    heard_lang = None
    if not prompt_text or not prompt_text.strip():
        try:
            m = _get_asr()
            a0, sr0 = sf.read(src)
            if getattr(a0, "ndim", 1) > 1:
                a0 = a0.mean(axis=-1)
            n = int(len(a0) * 16000 / sr0)
            a16 = np.interp(np.linspace(0, len(a0), n), np.arange(len(a0)),
                            np.asarray(a0, dtype=np.float64)).astype(np.float32)
            segs, info = m.transcribe(a16, beam_size=5)
            prompt_text = "".join(s.text for s in segs)
            heard_lang = info.language
            print(f"vc auto-transcribe [{heard_lang}]: {prompt_text[:60]!r}", flush=True)
        except Exception as e:
            raise HTTPException(500, f"auto-transcribe failed, pass prompt_text explicitly: {e}")
    lang = prompt_language if prompt_language != "auto" else (heard_lang or "ja")
    prompt = tts._get_prompt(tts.cnhubert_model, sov_cpu, src)
    try:
        ybin, _ = ensure_good_y(
            src, prompt_text, lang,
            resyn=lambda yb: _resyn_score(yb, prompt_text, lang, svx, ge, tts))
        if ybin is None:
            raise HTTPException(500, "prompt quality check failed (y collapsed); "
                                     "retry later or supply cleaner source audio")
        prompt = torch.from_numpy(np.fromfile(ybin, np.int64).reshape(1, -1))
        print(f"vc prompt y: {ybin}", flush=True)
    except HTTPException:
        raise
    except Exception as e:
        print(f"vc y auto-select failed ({e}), using extracted", flush=True)
    from gsv_tts.TTS import get_phones_and_bert
    phones, _, _, _ = get_phones_and_bert(prompt_text, tts.tts_config, lang)
    with torch.inference_mode():
        audio, _ = svx.decode(prompt.unsqueeze(0).to("xpu"),
                              torch.LongTensor(phones).unsqueeze(0).to("xpu"),
                              ge.to("xpu"), noise_scale=noise_scale, speed=1.0)
    a = audio[0, 0, :].float().cpu().numpy()
    mx = np.abs(a).max()
    if mx > 1:
        a = a / mx
    a = _normalize_loudness(a)
    a = np.concatenate([a, np.zeros(int(0.2 * SR), dtype=a.dtype)])
    out = os.path.join(tempfile.gettempdir(), f"vc_out_{int(time.time()*1000)}.wav")
    sf.write(out, a, SR)
    print(f"vc done: {len(a)/SR:.1f}s lang={lang}", flush=True)
    return FileResponse(out, media_type="audio/wav", filename=os.path.basename(out))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "9880")))
