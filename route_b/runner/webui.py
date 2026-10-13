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
import urllib.request

API = os.environ.get("API_BASE", "http://localhost:9880")


def _call(path, payload=None, method="GET"):
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


def synth(text, voice, language, top_k, verify):
    import gradio as gr
    if not text or not text.strip():
        raise gr.Error("empty text")
    payload = {"text": text, "language": language, "prompt_language": "ja",
               "top_k": int(top_k), "verify": bool(verify)}
    if voice and voice != "default":
        payload["voice"] = voice
    raw = _call("/tts", payload, method="POST")
    try:
        json.loads(raw)
        raise gr.Error(f"API error: {raw[:200]}")
    except (ValueError, UnicodeDecodeError):
        pass
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="webui_tts_")
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return path


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
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="文本", lines=5, value="こんにちは。",
                                  placeholder="输入要合成的文本…")
                voice = gr.Dropdown(label="音色预设", choices=_voices(), value="default")
                lang = gr.Radio(label="语言", choices=["ja", "zh", "en"], value="ja")
                with gr.Row():
                    top_k = gr.Slider(label="top_k", minimum=1, maximum=50, value=15, step=1)
                    verify = gr.Checkbox(label="ASR 质量门（慢但稳）", value=False)
                btn = gr.Button("合成", variant="primary")
            with gr.Column():
                audio = gr.Audio(label="输出", type="filepath")
        btn.click(synth, inputs=[text, voice, lang, top_k, verify], outputs=[audio])
    app.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
