"""Try NPU compile of HuBERT with STATIC shape."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np
import openvino as ov

IR = Path.home() / ".cache" / "gsv" / "ov_models" / "hubert_fp16.xml"
core = ov.Core()
model = core.read_model(IR)

STATIC_T = 48000  # 3 s @16kHz
model.reshape({model.inputs[0].get_any_name(): ov.PartialShape([1, STATIC_T])})
print("static shape:", model.inputs[0].get_partial_shape())

print("=== NPU compile, static [1, 48000] ===")
try:
    t = time.time()
    cm = core.compile_model(model, "NPU")
    print(f"NPU static compile OK ({time.time()-t:.1f}s)")
    x = np.random.randn(1, STATIC_T).astype(np.float32)
    t = time.time()
    o = cm(x)[0]
    print(f"NPU infer OK: {o.shape}, {(time.time()-t)*1000:.0f} ms")
except Exception as e:
    print("NPU static compile/infer FAIL:", str(e)[:500])
