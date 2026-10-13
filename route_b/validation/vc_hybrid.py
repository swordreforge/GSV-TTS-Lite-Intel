"""杂交：上游单实例流程 + 缓存 y 文件。是 y 坏还是我们实现坏，一测便知。"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(os.path.dirname(_HERE)) == "route_b" else _HERE
if _ROOT != _HERE:
    sys.path.insert(0, _ROOT)
import torch
import numpy as np
from gsv_tts import TTS
import logging
logging.disable(logging.INFO)
from gsv_tts.TTS import get_phones_and_bert as _gpb

YFILE = "/tmp/opencode/hybrid_y.bin"
# 被测 y：先从“问题缓存”复制（若已删则跳过说明）
import shutil, glob
cands = sorted(glob.glob(os.path.expanduser("~/.cache/gsv/prompt_y/4930899a86727597.ybin")))
print("cached y present:", bool(cands), flush=True)

tts = TTS(device="xpu", dtype="float32")
tts.load_gpt_model()
tts.load_sovits_model()
tts.init_language_module("zh")
sovits, ge = tts._prepare_sovits_resources(str(tts.default_sovits_path), "examples/laffey.mp3")
vq = sovits.vq_model.eval()
if tts.cnhubert_model is None:
    from gsv_tts.GPT_SoVITS.Featurizer import CNHubert
    tts.cnhubert_model = CNHubert(tts.cnhubert_path, tts.tts_config)
TEXT = "现在我的本体处在海的深层这里的端流很危险用于保护研究员的稳定结构正在被持续破坏我必须不断对它进行重构"
prompt_fresh = tts._get_prompt(tts.cnhubert_model, sovits, "32uwgy.wav")
prompt_fresh.cpu().numpy().astype(np.int64).tofile("/tmp/opencode/hybrid_freshy.bin")
print("fresh y saved", tuple(prompt_fresh.shape), flush=True)
phones, _, _, _ = _gpb(TEXT, tts.tts_config, "zh")
ph = torch.LongTensor(phones).unsqueeze(0).to("xpu")
import soundfile as sf
with torch.inference_mode():
    for name, y in [("fresh", prompt_fresh)]:
        audio, _ = vq.decode(y.unsqueeze(0).to("xpu"), ph, ge.to("xpu"),
                             noise_scale=0.5, speed=1.0)
        a = audio[0, 0, :].float().cpu().numpy()
        sf.write(f"/tmp/opencode/hybrid_{name}.wav", a, 32000)
        print(name, round(len(a) / 32000, 1), flush=True)
print("HYBRID DONE", flush=True)
