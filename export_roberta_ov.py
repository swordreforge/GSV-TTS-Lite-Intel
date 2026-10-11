"""Export chinese-roberta-wwm-ext-large to OpenVINO IR (FP16, dynamic batch/seq).

只导出管线真正用的 hidden_states[-3] 这一个输出（1024 维），
而不是全部 25 层 hidden states。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import torch
import torch.nn as nn
import openvino as ov
from transformers import AutoModelForMaskedLM

SRC = Path.home() / ".cache" / "gsv" / "chinese-roberta-wwm-ext-large"
OUT = Path.home() / ".cache" / "gsv" / "ov_models" / "roberta_h3_fp16"

print("loading RoBERTa (CPU, fp32)...")
hf = AutoModelForMaskedLM.from_pretrained(SRC, local_files_only=True)
hf = hf.float().eval()
print("export dtype:", next(hf.parameters()).dtype)

class RobertaH3(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask,
                      output_hidden_states=True)["hidden_states"][-3]

wrapped = RobertaH3(hf)
ex_ids = torch.ones(1, 32, dtype=torch.long)
ex_mask = torch.ones(1, 32, dtype=torch.long)
print("converting...")
ov_model = ov.convert_model(wrapped, example_input=(ex_ids, ex_mask))
for i in ov_model.inputs:
    print("static:", i.get_any_name(), i.get_partial_shape(), i.get_element_type())

dyn = ov.PartialShape([-1, -1])
ov_model.reshape({i.get_any_name(): dyn for i in ov_model.inputs})
print("reshaped:", [str(i.get_partial_shape()) for i in ov_model.inputs])

OUT.parent.mkdir(parents=True, exist_ok=True)
ov.save_model(ov_model, OUT.with_suffix(".xml"), compress_to_fp16=True)
print("saved:", str(OUT) + ".xml")
