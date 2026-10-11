"""OpenVINO backend for chinese-hubert-base (drop-in for transformers HubertModel).

通过环境变量启用:
    GSV_HUBERT_BACKEND=openvino  GSV_OV_DEVICE=GPU|CPU  (默认 GPU)

说明:
- IR 为 FP16、时间轴动态 ([1, -1])，GPU/CPU 可直接跑变长音频。
- NPU 只接受静态 shape（实测动态 shape 编译失败），不适合本管线中
  变长 prompt 音频的场景，因此集成只支持 GPU/CPU；NPU 仅作 benchmark
  验证（见 try_hubert_npu.py）。
"""
import logging
import os
from pathlib import Path

import numpy as np
import torch


class HubertOVModel:
    """模拟 HubertModel 的最小接口: wav Tensor -> {"last_hidden_state": Tensor}"""

    def __init__(self, ir_path: Path, device: str = "GPU"):
        import openvino as ov

        self.ir_path = Path(ir_path)
        self.ov_device = device
        self.core = ov.Core()
        logging.info(f"Compiling HuBERT IR for {device}: {self.ir_path}")
        self.compiled = self.core.compile_model(str(self.ir_path), device)
        # transformers 侧 HubetModel 在 XPU 上跑 fp16；OV 侧以 fp32 输入，
        # 输出转回调用方 tensor 的 device/dtype，保持数值行为一致。
        self.device = None

    def __call__(self, wav: torch.Tensor):
        x = wav.detach().to("cpu", torch.float32).numpy()
        out = self.compiled(x)[0]
        return {"last_hidden_state": torch.from_numpy(out).to(device=wav.device, dtype=wav.dtype)}

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self


def ensure_hubert_ir(models_dir: Path) -> Path:
    """IR 不存在时用本地权重自动导出一次（约 1~2 分钟，仅首次）。"""
    ir = Path(models_dir) / "ov_models" / "hubert_fp16.xml"
    if ir.exists():
        return ir

    logging.info("HuBERT OpenVINO IR 不存在，正在从本地权重导出 ...")
    import torch.nn as nn
    import openvino as ov
    from transformers import HubertModel

    src = Path(models_dir) / "chinese-hubert-base"
    hf = HubertModel.from_pretrained(src, local_files_only=True)
    hf = hf.float().eval()

    class HubertLastHidden(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_values):
            return self.m(input_values)["last_hidden_state"]

    ov_model = ov.convert_model(HubertLastHidden(hf), example_input=torch.randn(1, 48000))
    ov_model.reshape({ov_model.inputs[0].get_any_name(): ov.PartialShape([1, -1])})
    ir.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, ir, compress_to_fp16=True)
    logging.info(f"HuBERT IR 已导出: {ir}")
    return ir


def load_hubert_ov(models_dir: Path, device: str = "GPU") -> HubertOVModel:
    return HubertOVModel(ensure_hubert_ir(models_dir), device)
