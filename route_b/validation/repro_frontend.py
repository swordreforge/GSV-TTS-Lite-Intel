"""原样前端参数复现：prompt_text='' + prompt_language='ja' + 32uwgy，直调 webui.synth ×3。"""
import sys, traceback
sys.path.insert(0, "route_b/runner")
import webui
webui.API = "http://localhost:9880"

UP = "/home/swordreforge/GSV-TTS-Lite/32uwgy.wav"

for k in range(3):
    try:
        # 镜像 gradio 参数: [text, voice, language, prompt_language, top_k, verify, upload, prompt_text, cut_minlen, sovits]
        p = webui.synth("こんにちは。", "default", "ja", "ja", 15, False, UP, "", 30, "")
        print(f"RUN{k}: SYNTH-OK {p}", flush=True)
    except Exception as e:
        print(f"RUN{k}: SYNTH-FAIL {repr(e)[:160]}", flush=True)
