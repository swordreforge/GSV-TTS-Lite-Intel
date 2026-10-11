"""OpenVINO backend for chinese-roberta-wwm-ext-large (hidden_states[-3] only).

通过环境变量启用:
    GSV_ROBERTA_BACKEND=openvino
    GSV_OV_DEVICE=GPU   (默认; 动态 batch/seq，无需重编译)
    GSV_OV_DEVICE=NPU   (静态 shape，按桶缓存编译模型; 新长度首次编译约 6s)
    GSV_OV_DEVICE=CPU   (动态 shape)

NPU 说明: Meteor Lake NPU 只接受静态 shape，每遇到一个新 (batch, seq)
组合就要重新编译一次（约 6s）。这里把 seq 按 NPU_SEQ_BUCKET（默认 16）
向上取整为桶，并缓存每个桶的编译模型；常用长度只需编译一次。
"""
import logging
import math
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

from ...Config import Config
from transformers import AutoTokenizer


def ensure_roberta_ir(models_dir: Path) -> Path:
    ir = Path(models_dir) / "ov_models" / "roberta_h3_fp16.xml"
    if ir.exists():
        return ir

    logging.info("RoBERTa OpenVINO IR 不存在，正在从本地权重导出 ...")
    import torch.nn as nn
    import openvino as ov
    from transformers import AutoModelForMaskedLM

    src = Path(models_dir) / "chinese-roberta-wwm-ext-large"
    hf = AutoModelForMaskedLM.from_pretrained(src, local_files_only=True)
    hf = hf.float().eval()

    class RobertaH3(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)["hidden_states"][-3]

    ex_ids = torch.ones(1, 32, dtype=torch.long)
    ex_mask = torch.ones(1, 32, dtype=torch.long)
    ov_model = ov.convert_model(RobertaH3(hf), example_input=(ex_ids, ex_mask))
    dyn = ov.PartialShape([-1, -1])
    ov_model.reshape({i.get_any_name(): dyn for i in ov_model.inputs})
    ir.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, ir, compress_to_fp16=True)
    logging.info(f"RoBERTa IR 已导出: {ir}")
    return ir


class CNRobertaOV(nn.Module):
    def __init__(self, base_path, tts_config: Config, ov_device: str = "GPU"):
        super().__init__()
        import openvino as ov

        self.tts_config = tts_config
        self.ov_device = ov_device.upper()
        self.base_path = Path(base_path) if isinstance(base_path, str) else base_path
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.base_path), local_files_only=True)

        models_dir = self.base_path.parent
        ir = ensure_roberta_ir(models_dir)
        self.core = ov.Core()
        self.ir_path = str(ir)
        self.bucket_cache = {}
        self.bucket = max(1, int(os.environ.get("GSV_NPU_SEQ_BUCKET", "16")))

        if self.ov_device != "NPU":
            logging.info(f"Compiling RoBERTa IR for {self.ov_device}: {ir}")
            self.compiled = self.core.compile_model(self.ir_path, self.ov_device)
        else:
            self.compiled = None
            logging.info(f"RoBERTa NPU 模式: seq 按 {self.bucket} 取整分桶编译")

    def _get_npu_model(self, batch: int, seq: int):
        import openvino as ov

        bucket_seq = min(512, int(math.ceil(seq / self.bucket)) * self.bucket)
        key = (batch, bucket_seq)
        if key not in self.bucket_cache:
            logging.info(f"Compiling RoBERTa IR for NPU bucket batch={batch} seq={bucket_seq} ...")
            model = self.core.read_model(self.ir_path)
            model.reshape({
                "input_ids": ov.PartialShape([batch, bucket_seq]),
                "attention_mask": ov.PartialShape([batch, bucket_seq]),
            })
            self.bucket_cache[key] = self.core.compile_model(model, "NPU")
        return self.bucket_cache[key], bucket_seq

    def forward(self, word2ph_list: List):
        with torch.no_grad():
            texts = ["".join(word2ph["word"]) for word2ph in word2ph_list]

            inputs = self.tokenizer(
                texts,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=512,
            )
            ids = inputs["input_ids"].astype(np.int64)
            mask = inputs["attention_mask"].astype(np.int64)

            if self.ov_device == "NPU":
                compiled, bucket_seq = self._get_npu_model(ids.shape[0], ids.shape[1])
                if bucket_seq > ids.shape[1]:
                    pad = bucket_seq - ids.shape[1]
                    ids = np.pad(ids, ((0, 0), (0, pad)))
                    mask = np.pad(mask, ((0, 0), (0, pad)))
                hidden = compiled((ids, mask))[0]
                real_seq = inputs["attention_mask"].shape[1]
                hidden = hidden[:, :real_seq, :]
                attn = inputs["attention_mask"]
            else:
                hidden = self.compiled((ids, mask))[0]
                attn = mask

            hidden_states = torch.from_numpy(hidden)
            batch_phone_features = []
            for i in range(len(texts)):
                m = attn[i] == 1
                char_features = hidden_states[i][m]
                char_features = char_features[1:-1, :]

                repeats = torch.tensor(word2ph_list[i]["ph"])
                phone_feature = torch.repeat_interleave(char_features, repeats, dim=0)
                phone_feature = phone_feature.to(
                    device=self.tts_config.device, dtype=self.tts_config.dtype
                )
                batch_phone_features.append(phone_feature)

            return batch_phone_features
