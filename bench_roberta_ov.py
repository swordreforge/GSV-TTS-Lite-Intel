"""Benchmark RoBERTa-h3: torch-XPU vs OpenVINO-GPU (dynamic) vs OpenVINO-NPU (static)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np
import torch
import openvino as ov
from transformers import AutoModelForMaskedLM, AutoTokenizer

GSV = Path.home() / ".cache" / "gsv"
IR = GSV / "ov_models" / "roberta_h3_fp16.xml"
TEXT = "你好，世界，这是一个中文语音合成测试。"

tok = AutoTokenizer.from_pretrained(GSV / "chinese-roberta-wwm-ext-large", local_files_only=True)
enc = tok([TEXT], return_tensors="pt", padding=True, truncation=True, max_length=512)
print("seq len:", enc["input_ids"].shape)
ids_np = enc["input_ids"].numpy().astype(np.int64)
mask_np = enc["attention_mask"].numpy().astype(np.int64)

core = ov.Core()

print("--- OV GPU (dynamic) ---")
t = time.time()
cm_gpu = core.compile_model(IR, "GPU")
print(f"compile: {time.time()-t:.1f}s")
for _ in range(2):
    cm_gpu((ids_np, mask_np))
t = time.time(); N = 5
for _ in range(N):
    o_gpu = cm_gpu((ids_np, mask_np))[0]
print(f"OV GPU: {(time.time()-t)/N*1000:.0f} ms/run, out {o_gpu.shape}")

print("--- torch XPU fp16 ---")
m = AutoModelForMaskedLM.from_pretrained(GSV / "chinese-roberta-wwm-ext-large", local_files_only=True)
m.eval().to("xpu", torch.float16)
ids_x = enc["input_ids"].to("xpu")
mask_x = enc["attention_mask"].to("xpu")
with torch.inference_mode():
    for _ in range(2):
        m(input_ids=ids_x, attention_mask=mask_x, output_hidden_states=True)["hidden_states"][-3]
    torch.xpu.synchronize()
    t = time.time()
    for _ in range(N):
        r = m(input_ids=ids_x, attention_mask=mask_x, output_hidden_states=True)["hidden_states"][-3].float().cpu().numpy()
    torch.xpu.synchronize()
    print(f"torch XPU: {(time.time()-t)/N*1000:.0f} ms/run, out {r.shape}")

a = o_gpu.astype(np.float64).ravel(); b = r.astype(np.float64).ravel()
print(f"max abs diff: {np.abs(a-b).max():.5f}, cos sim: {float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))):.6f}")

print("--- OV NPU (static [1, S]) ---")
S = int(ids_np.shape[1])
model = core.read_model(IR)
model.reshape({i.get_any_name(): ov.PartialShape([1, S]) for i in model.inputs})
try:
    t = time.time()
    cm_npu = core.compile_model(model, "NPU")
    print(f"NPU compile OK ({time.time()-t:.1f}s)")
    t = time.time()
    for _ in range(N):
        o_npu = cm_npu((ids_np, mask_np))[0]
    print(f"OV NPU: {(time.time()-t)/N*1000:.0f} ms/run, out {o_npu.shape}")
    c = o_npu.astype(np.float64).ravel()
    print(f"NPU vs torch cos sim: {float(c@b/(np.linalg.norm(c)*np.linalg.norm(b))):.6f}")
except Exception as e:
    print("NPU FAIL:", str(e)[:400])
