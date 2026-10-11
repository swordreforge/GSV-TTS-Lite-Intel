from transformers import logging as tf_logging

tf_logging.set_verbosity_error()

import logging

logging.getLogger("numba").setLevel(logging.WARNING)

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)

import os
import torch.nn as nn
from ...Config import Config


class CNHubert(nn.Module):
    def __init__(self, base_path, tts_config: Config):
        super().__init__()
        backend = os.environ.get("GSV_HUBERT_BACKEND", "torch").lower()
        if backend == "openvino":
            from .hubert_ov import load_hubert_ov
            from pathlib import Path
            ov_device = os.environ.get("GSV_OV_DEVICE", "GPU").upper()
            # IR 与权重同目录: <models_dir>/ov_models/hubert_fp16.xml
            models_dir = Path(base_path).parent
            self.model = load_hubert_ov(models_dir, ov_device)
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_path, local_files_only=True)
        else:
            # 明确指定 dtype，确保 MPS 设备使用 float32
            self.model = HubertModel.from_pretrained(base_path, local_files_only=True, torch_dtype=tts_config.dtype)
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_path, local_files_only=True)
        self.eval()
        self.to(tts_config.device, tts_config.dtype)

    def forward(self, x):
        input_values = self.feature_extractor(x, return_tensors="pt", sampling_rate=16000).input_values.to(x.device)
        feats = self.model(input_values)["last_hidden_state"]
        return feats