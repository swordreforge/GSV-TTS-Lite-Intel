"""验证 GPT 单步 OV IR：真实 infer 抓输入，对 logits + 计时（torch XPU vs OV GPU）。"""
import sys, os, time, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.disable(logging.INFO)
from pathlib import Path
import numpy as np
import torch
import openvino as ov
from gsv_tts import TTS

GSV = Path.home() / ".cache" / "gsv"
IR = GSV / "ov_models" / "gpt_step_b1k1024.xml"

tts = TTS(device="xpu", dtype="float16")
tts.load_gpt_model()
tts.load_sovits_model()
tts.init_language_module("ja")
t2s = list(tts.gpt_models.values())[0].t2s_model

# 抓一个真实 decode 步的输入
cap = {}
tr = t2s.t2s_transformer.decode_next_token
def w(x, kc, vc, ln, mk, bi):
    if not cap:
        cap.update(x=x.detach().cpu().clone(), kc=[t.detach().cpu().clone() for t in kc],
                   vc=[t.detach().cpu().clone() for t in vc], ln=ln.detach().cpu().clone(),
                   mk=mk.detach().cpu().clone(), bi=bi.detach().cpu().clone())
    return tr(x, kc, vc, ln, mk, bi)
t2s.t2s_transformer.decode_next_token = w
KW = dict(spk_audio_path="examples/laffey.mp3", prompt_audio_path="examples/AnAn.ogg",
          prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
          text="へぇー、ここまでしてくれるんですね。")
tts.infer(**KW)
print("captured K =", cap["kc"][0].shape[2], flush=True)

D = torch.device("xpu")
with torch.inference_mode():
    x = cap["x"].to(D); kc = [t.to(D) for t in cap["kc"]]; vc = [t.to(D) for t in cap["vc"]]
    ln = cap["ln"].to(D); mk = cap["mk"].to(D); bi = cap["bi"].to(D)
    ref_h = t2s.t2s_transformer.decode_next_token(x, kc, vc, ln, mk, bi)
    ref_logits = t2s.ar_predict_layer(ref_h[:, -1]).float().cpu().numpy()
    torch.xpu.synchronize(); t = time.time(); N = 10
    for _ in range(N):
        t2s.t2s_transformer.decode_next_token(x, kc, vc, ln, mk, bi)
    torch.xpu.synchronize()
    print(f"torch XPU step: {(time.time()-t)/N*1000:.2f} ms", flush=True)

core = ov.Core()
t = time.time()
cm = core.compile_model(str(IR), "GPU")
print(f"OV GPU compile: {time.time()-t:.1f}s", flush=True)
KPad = 1024
f32 = lambda t: t.detach().cpu().float().numpy()
pad_kv = lambda t: np.pad(f32(t), ((0, 0), (0, 0), (0, KPad - t.shape[2]), (0, 0)))
mk = cap["mk"].detach().cpu().numpy().astype(bool)
mk_pad = np.pad(mk, ((0, 0), (0, 0), (0, 0), (0, KPad - mk.shape[3])))
feed = [f32(cap["x"]), *[pad_kv(t) for t in cap["kc"]], *[pad_kv(t) for t in cap["vc"]],
        cap["ln"].numpy(), mk_pad]
for _ in range(2):
    cm(feed)
t = time.time(); N = 10
for _ in range(N):
    out = cm(feed)
print(f"OV GPU step: {(time.time()-t)/N*1000:.2f} ms", flush=True)
ov_logits = out[0]
print(f"logits maxdiff={np.abs(ov_logits - ref_logits).max():.5f}", flush=True)
