"""Benchmark HuBERT: torch-XPU-fp16 vs OpenVINO-GPU-fp16 (correctness + latency)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np
import torch
import openvino as ov
import av

IR = Path.home() / ".cache" / "gsv" / "ov_models" / "hubert_fp16.xml"

def load_audio_16k(path):
    with av.open(path) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=stream.rate)
        frames = []
        for frame in container.decode(stream):
            for r in resampler.resample(frame):
                frames.append(r.to_ndarray())
        audio = np.concatenate(frames, axis=1)
    import torchaudio.functional as AF
    wav = torch.from_numpy(audio)
    wav16k = AF.resample(wav, stream.rate, 16000).mean(dim=0)
    silence = torch.zeros(int(16000 * 0.3))
    wav16k = torch.cat([wav16k, silence])
    return wav16k

wav = load_audio_16k("examples/AnAn.ogg")
print("wav len:", wav.shape, f"({wav.shape[0]/16000:.2f}s)")
x = wav.unsqueeze(0).numpy().astype(np.float32)

core = ov.Core()
print("compiling for GPU...")
t = time.time()
cm = core.compile_model(IR, "GPU")
print(f"GPU compile: {time.time()-t:.1f}s")
out = cm(x)[0]
print("OV GPU out:", out.shape, out.dtype)

# torch reference on XPU fp16
from transformers import HubertModel
SRC = Path.home() / ".cache" / "gsv" / "chinese-hubert-base"
ref = HubertModel.from_pretrained(SRC, local_files_only=True).eval().to("xpu", torch.float16)
with torch.inference_mode():
    for _ in range(2):  # warmup
        ref(torch.from_numpy(x).to("xpu", torch.float16))
    torch.xpu.synchronize()
    t = time.time()
    for _ in range(5):
        r = ref(torch.from_numpy(x).to("xpu", torch.float16))["last_hidden_state"].float().cpu().numpy()
    torch.xpu.synchronize()
    print(f"torch XPU fp16: {(time.time()-t)/5*1000:.0f} ms/run, out {r.shape}")

# OV GPU latency
for _ in range(2):
    cm(x)
t = time.time()
N = 5
for _ in range(N):
    o = cm(x)[0]
print(f"OV GPU fp16: {(time.time()-t)/N*1000:.0f} ms/run")

# correctness
a = o.astype(np.float64).ravel()
b = r.astype(np.float64).ravel()
diff = np.abs(a - b)
print(f"max abs diff: {diff.max():.5f}, mean: {diff.mean():.6f}")
print(f"cos sim: {float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b))):.6f}")
