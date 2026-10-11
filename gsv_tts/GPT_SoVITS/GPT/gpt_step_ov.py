"""GPT 单步解码的 OpenVINO 运行封装（函数式 KV-cache，采样仍在 Python）。

用法:
    GSV_GPT_BACKEND=openvino  GSV_OV_DEVICE=GPU

- IR: <models_dir>/ov_models/gpt_step_b1k1024.xml（需先运行 export_gpt_step_ov.py）
- 自带 K=1024 的 KV-cache（与模型同 dtype/device）；prefill 复用 torch
  process_prompt 写入同一份 cache，decode 步走 OV。
- 仅支持 batch=1 的 infer() 路径；infer_batched 仍走 torch。
"""
import logging
from pathlib import Path

import numpy as np
import torch


class GPTStepOV:
    K = 1024

    def __init__(self, models_dir, t2s_model, ov_device: str = "GPU"):
        import openvino as ov

        self.t2s = t2s_model
        self.ov_device = ov_device
        ir = Path(models_dir) / "ov_models" / "gpt_step_b1k1024.xml"
        if not ir.exists():
            raise FileNotFoundError(f"GPT IR 不存在，请先运行 export_gpt_step_ov.py: {ir}")
        self.core = ov.Core()
        logging.info(f"Compiling GPT-step IR for {ov_device}: {ir}")
        self.compiled = self.core.compile_model(str(ir), ov_device)
        self.kc = self.vc = self.ln = None

    def reset(self, dtype, device):
        t2s = self.t2s
        L, B, H = t2s.num_layers, 1, t2s.num_head
        hd = t2s.model_dim // H
        self.kc = [torch.zeros(B, H, self.K, hd, dtype=dtype, device=device) for _ in range(L)]
        self.vc = [torch.zeros(B, H, self.K, hd, dtype=dtype, device=device) for _ in range(L)]
        self.ln = torch.zeros((B,), dtype=torch.int64, device=device)

    def step(self, xy_pos, mask):
        """一次 decode：输入均为同 device tensor；返回 logits（同 device/dtype），
        并原地推进自带的 kv cache/len。"""
        f32 = lambda t: t.detach().to("cpu", torch.float32).numpy()
        feed = [f32(xy_pos), *[f32(t) for t in self.kc], *[f32(t) for t in self.vc],
                self.ln.detach().cpu().numpy(),
                mask.detach().cpu().numpy()]
        out = self.compiled(feed)
        n = len(self.kc)
        dev, dt = xy_pos.device, xy_pos.dtype
        for i in range(n):
            self.kc[i].copy_(torch.from_numpy(out[1 + i]).to(dev, dt))
            self.vc[i].copy_(torch.from_numpy(out[1 + n + i]).to(dev, dt))
        self.ln.copy_(torch.from_numpy(out[1 + 2 * n]).to(dev))
        return torch.from_numpy(out[0]).to(dev, dt)
