"""Intel 路径 WebUI（薄前端）：调 localhost:9880 API，不跑 torch。
用法：.venv-xpu/bin/python route_b/runner/webui.py [--port 7860] [--api http://localhost:9880]
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
import argparse
import json
import tempfile

API = os.environ.get("API_BASE", "http://localhost:9880")
UPLOAD_DIR = os.path.join(os.path.expanduser("~"), ".cache", "gsv", "uploads")


def _save_upload(src_path, prefix="prompt_"):
    """浏览器上传 → 服务端持久路径（uploads 不在 /tmp，重启不丢）。"""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    import re, time
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(src_path or "up.wav"))[-60:]
    dst = os.path.join(UPLOAD_DIR, f"{prefix}{int(time.time()*1000)}_{base}")
    import shutil
    shutil.copyfile(src_path, dst)
    return dst


def _call(path, payload=None, method="GET"):
    import urllib.request
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(API + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return r.read()


def _voices():
    try:
        return list(json.loads(_call("/voices")).keys())
    except Exception:
        return ["default"]


def _to_pcm16_wav(raw: bytes) -> bytes:
    """float32 wav → PCM_16 wav（Firefox 解不了 float32，Chrome 行）。失败时原样返回。"""
    try:
        import io
        import soundfile as sf
        import numpy as np
        a, sr = sf.read(io.BytesIO(raw), dtype="float32")
        buf = io.BytesIO()
        sf.write(buf, np.asarray(a, dtype=np.float32), sr, subtype="PCM_16", format="WAV")
        return buf.getvalue()
    except Exception:
        return raw


_whisper_model = None


def _auto_transcribe(wav_path):
    """参考音频自动转写（small/cpu，慢但免手填）。返回 (text, lang)。模型常驻。"""
    global _whisper_model
    import soundfile as sf
    import numpy as np
    from faster_whisper import WhisperModel
    if _whisper_model is None:
        _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
    m = _whisper_model
    a, sr = sf.read(wav_path)
    a = np.asarray(a, dtype=np.float64)
    if getattr(a, "ndim", 1) > 1:
        a = a.mean(axis=-1)
    n = max(1, int(len(a) * 16000 / sr))
    a16 = np.interp(np.linspace(0, len(a), n), np.arange(len(a)), a).astype(np.float32)
    segs, info = m.transcribe(a16, beam_size=5)
    txt = "".join(s.text for s in segs).strip()
    lang = (getattr(info, "language", None) or "").lower()
    lang = {"zh": "zh", "ja": "ja", "en": "en"}.get(lang, "")
    return txt, lang


def synth(text, voice, language, prompt_language, top_k, verify,
          prompt_upload, prompt_text, cut_minlen, sovits_model):
    import gradio as gr
    if not text or not text.strip():
        raise gr.Error("empty text")
    payload = {"text": text, "language": language, "prompt_language": prompt_language,
               "top_k": int(top_k), "verify": bool(verify), "cut_minlen": int(cut_minlen)}
    if voice and voice != "default":
        payload["voice"] = voice
    if prompt_upload:  # 一次性克隆：上传优先于预设
        try:
            up = _save_upload(prompt_upload)
        except Exception as e:
            raise gr.Error(f"上传保存失败: {e}")
        # 上传即自动转写+检测语言：覆盖用户选错的参考语言（ja 前端啃中文会摆烂）
        auto_txt, auto_lang = "", ""
        try:
            auto_txt, auto_lang = _auto_transcribe(up)
        except Exception as e:
            print(f"auto-transcribe failed: {e}", flush=True)
        if not (prompt_text or "").strip():
            if not auto_txt:
                raise gr.Error("参考音频转写为空且未填参考文本，请手动填写")
            prompt_text = auto_txt
        if auto_lang:
            payload["prompt_language"] = auto_lang
        payload["prompt_audio"] = up
        payload["prompt_text"] = prompt_text.strip()
        payload["spk"] = up
    if (sovits_model or "").strip():
        payload["sovits_model"] = sovits_model.strip()
    print(f"webui->api /tts payload: {json.dumps(payload, ensure_ascii=False)[:600]}", flush=True)
    raw = _call("/tts", payload, method="POST")
    try:
        json.loads(raw)
        raise gr.Error(f"API error: {raw[:200]}")
    except (ValueError, UnicodeDecodeError):
        pass
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="webui_tts_")
    with os.fdopen(fd, "wb") as f:
        f.write(_to_pcm16_wav(raw))
    return path


def vc_convert(src_audio, spk_voice, prompt_text, prompt_language):
    import gradio as gr
    import requests
    if not src_audio:
        raise gr.Error("请上传源音频")
    spk = "examples/laffey.mp3"
    if spk_voice and spk_voice != "default":
        try:
            vs = json.loads(_call("/voices"))
            if spk_voice in vs:
                spk = vs[spk_voice]["spk"]
        except Exception:
            pass
    data = {"spk": spk, "prompt_text": prompt_text or "",
            "prompt_language": prompt_language}
    with open(src_audio, "rb") as f:
        files = {"file": (os.path.basename(src_audio), f.read())}
    try:
        r = requests.post(API + "/vc", data=data, files=files, timeout=1800)
    except Exception as e:
        raise gr.Error(f"API 不可达: {e}")
    if r.status_code != 200:
        raise gr.Error(f"API error {r.status_code}: {r.text[:200]}")
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="webui_vc_")
    with os.fdopen(fd, "wb") as f:
        f.write(_to_pcm16_wav(r.content))
    return path


def voice_add(name, spk, prompt_audio, prompt_text, prompt_language, sovits_model):
    import gradio as gr
    if not name or not name.strip():
        raise gr.Error("empty name")
    payload = {"name": name.strip(), "spk": spk, "prompt_audio": prompt_audio,
               "prompt_text": prompt_text, "prompt_language": prompt_language,
               "sovits_model": (sovits_model or "").strip() or None}
    try:
        r = _call("/voices", payload, method="POST")
        json.loads(r)
    except Exception as e:
        raise gr.Error(f"add failed: {e}")
    return refresh_voices()


def voice_del(name):
    import gradio as gr
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.Request(API + "/voices/" + name, method="DELETE")
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
    except Exception as e:
        raise gr.Error(f"delete failed: {e}")
    return refresh_voices()


def refresh_voices():
    import gradio as gr
    vs = _voices()
    table = [[k] for k in vs]
    dd = gr.Dropdown(choices=vs, value="default")
    return dd, table, gr.Dropdown(choices=vs, value="default")


def main():
    import gradio as gr
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--api", default="http://localhost:9880")
    args = ap.parse_args()
    global API
    API = args.api

    with gr.Blocks(title="GSV-TTS-Lite (Intel)") as app:
        gr.Markdown("## GSV-TTS-Lite · Intel 后端（C++ GPT + XPU SoVITS）")
        with gr.Tab("语音合成"):
            with gr.Row():
                with gr.Column():
                    text = gr.Textbox(label="文本", lines=5, value="こんにちは。",
                                      placeholder="输入要合成的文本…")
                    voice = gr.Dropdown(label="音色预设", choices=_voices(), value="default")
                    with gr.Row():
                        lang = gr.Radio(label="目标语言", choices=["ja", "zh", "en"], value="ja")
                        plang = gr.Radio(label="参考音频语言", choices=["ja", "zh", "en"], value="ja")
                    with gr.Row():
                        top_k = gr.Slider(label="top_k", minimum=1, maximum=50, value=15, step=1)
                        verify = gr.Checkbox(label="ASR 质量门（慢但稳）", value=False)
                    with gr.Row():
                        cut_minlen = gr.Slider(label="分段长度（弱模型长句糊就调小，如 10）",
                                               minimum=5, maximum=100, value=30, step=1)
                        sovits_box = gr.Textbox(label="SoVITS .pth（空=默认，微调音色填路径）", value="")
                    with gr.Accordion("一次性克隆（上传参考音频，不存预设）", open=False):
                        pup = gr.Audio(label="参考音频（上传，5秒以上干净干声）",
                                       type="filepath", sources=["upload"])
                        ptext = gr.Textbox(label="参考文本（空=自动转写；音频里说的内容）",
                                           lines=2, value="")
                    btn = gr.Button("合成", variant="primary")
                with gr.Column():
                    audio = gr.Audio(label="输出", type="filepath")
            btn.click(synth, inputs=[text, voice, lang, plang, top_k, verify,
                                     pup, ptext, cut_minlen, sovits_box], outputs=[audio])
        with gr.Tab("变声"):
            with gr.Row():
                with gr.Column():
                    src = gr.Audio(label="源音频（上传）", type="filepath", sources=["upload"])
                    spk_v = gr.Dropdown(label="目标音色（取其 spk）", choices=_voices(), value="default")
                    ptext = gr.Textbox(label="源转写词（空=自动 ASR）", lines=2, value="")
                    vclang = gr.Radio(label="源语言", choices=["auto", "ja", "zh", "en"], value="auto")
                    vbtn = gr.Button("变声", variant="primary")
                with gr.Column():
                    vaudio = gr.Audio(label="输出", type="filepath")
            vbtn.click(vc_convert, inputs=[src, spk_v, ptext, vclang], outputs=[vaudio])
        with gr.Tab("音色预设"):
            with gr.Row():
                with gr.Column():
                    vname = gr.Textbox(label="预设名", value="")
                    vspk = gr.Textbox(label="spk 音频路径", value="examples/laffey.mp3")
                    vpa = gr.Textbox(label="prompt 音频路径", value="examples/AnAn.ogg")
                    vpt = gr.Textbox(label="prompt 文本", lines=2,
                                     value="ちが……ちがう。レイア、貴様は間違っている。")
                    vpl = gr.Radio(label="prompt 语言", choices=["ja", "zh", "en"], value="ja")
                    vsov = gr.Textbox(label="SoVITS .pth（空=默认）", value="")
                    with gr.Row():
                        vadd = gr.Button("保存", variant="primary")
                        vdel = gr.Button("删除所选")
                with gr.Column():
                    vtable = gr.Dataframe(label="现有预设", headers=["name"], col_count=(1, "fixed"))
                    vsel = gr.Dropdown(label="选择", choices=_voices(), value="default")
                    vref = gr.Button("刷新列表")
            vadd.click(voice_add, inputs=[vname, vspk, vpa, vpt, vpl, vsov],
                       outputs=[vsel, vtable, voice])
            vdel.click(voice_del, inputs=[vsel], outputs=[vsel, vtable, voice])
            vref.click(refresh_voices, outputs=[vsel, vtable, voice])
    app.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
