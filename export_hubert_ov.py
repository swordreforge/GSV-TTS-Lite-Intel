"""Export chinese-hubert-base HubertModel to OpenVINO IR (FP16, dynamic time axis)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import torch
import torch.nn as nn
import openvino as ov
from transformers import HubertModel

SRC = Path.home() / ".cache" / "gsv" / "chinese-hubert-base"
OUT = Path.home() / ".cache" / "gsv" / "ov_models" / "hubert_fp16"
OUT.mkdir(parents=True, exist_ok=True)

print("loading HubertModel (CPU, fp32)...")
hf = HubertModel.from_pretrained(SRC, local_files_only=True)
hf = hf.float().eval()
print("export dtype:", next(hf.parameters()).dtype)

class HubertLastHidden(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, input_values):
        return self.m(input_values)["last_hidden_state"]

wrapped = HubertLastHidden(hf)
example = torch.randn(1, 48000)
print("converting...")
ov_model = ov.convert_model(wrapped, example_input=example)

# dynamic time axis: [1, -1]
inp = ov_model.inputs[0]
print("static shape:", inp.get_partial_shape())
ov_model.reshape({inp.get_any_name(): ov.PartialShape([1, -1])})
print("reshaped:", ov_model.inputs[0].get_partial_shape())

print("compress to fp16 + save...")
ov.save_model(ov_model, OUT.with_suffix(".xml"), compress_to_fp16=True)
print("saved:", str(OUT) + ".xml")
print("inputs:", [(i.get_any_name(), i.get_partial_shape(), i.get_element_type()) for i in ov_model.inputs])
print("outputs:", [(str(i), str(o.get_partial_shape())) for i, o in enumerate(ov_model.outputs)])
