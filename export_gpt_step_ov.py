"""导出 GPT 单步解码为 OpenVINO IR（函数式 KV-cache，不碰采样）。

边界: xy_pos[B,1,512] + k/v cache + kv_len + mask + batch_idx
   -> logits[B,1025] + new_k + new_v + new_len
逐行复刻 T2SBlock.decode_next_token，仅把 in-place 写改成函数式。
默认桶: B=1, max_kv=1024（与运行时最大桶一致）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
import openvino as ov

GSV = Path.home() / ".cache" / "gsv"
OUT = GSV / "ov_models" / "gpt_step_b1k1024"
B, K, H, HD, DIM = 1, 1024, 16, 32, 512


class GPTStep(nn.Module):
    def __init__(self, t2s):
        super().__init__()
        self.blocks = t2s.t2s_transformer.blocks
        self.predict = t2s.ar_predict_layer
        self.n = t2s.t2s_transformer.num_blocks
        self.nh = t2s.num_head
        self.hd = DIM // t2s.num_head

    def forward(self, x, k_all, v_all, kv_len, mask):
        # 函数式 cache 写：OV 前端不支持 index_put_，用算术等价改写
        # （B=1 直接 bake，不再需要 bidx 输入）
        p = kv_len[0]
        sel = (torch.arange(K, dtype=torch.int64) == p).reshape(1, 1, K, 1)
        new_ks, new_vs = [], []
        for i in range(self.n):
            blk = self.blocks[i]
            kc = k_all[i].clone()
            vc = v_all[i].clone()
            Bl, L, _ = x.shape
            residual = x
            qkv = blk.qkv(x).view(Bl, L, 3, self.nh, self.hd)
            q, k, v = qkv.unbind(2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            wv = sel.to(kc.dtype)
            kc = kc * (1 - wv) + k.squeeze(2).unsqueeze(2) * wv
            vc = vc * (1 - wv) + v.squeeze(2).unsqueeze(2) * wv
            x = F.scaled_dot_product_attention(q, kc, vc, attn_mask=mask)
            x = x.transpose(1, 2).reshape(Bl, L, DIM)
            x = blk.out_proj(x)
            x = residual + x
            x = blk.norm1(x)
            residual = x
            x = blk.mlp(x)
            x = residual + x
            x = blk.norm2(x)
            new_ks.append(kc)
            new_vs.append(vc)
        new_len = kv_len + 1
        logits = self.predict(x[:, -1])
        return logits, new_ks, new_vs, new_len


def main():
    from gsv_tts import TTS
    import logging
    logging.disable(logging.INFO)
    print("loading GPT...", flush=True)
    tts = TTS(device="xpu", dtype="float16")
    tts.load_gpt_model()
    t2s = list(tts.gpt_models.values())[0].t2s_model.to("cpu").float().eval()
    step = GPTStep(t2s).float().eval()

    D = torch.device("cpu")
    x = torch.randn(B, 1, DIM, dtype=torch.float32)
    k_all = [torch.randn(B, H, K, HD, dtype=torch.float32) for _ in range(t2s.num_layers)]
    v_all = [torch.randn(B, H, K, HD, dtype=torch.float32) for _ in range(t2s.num_layers)]
    # k_all[i] 形状与 bucket.k_cache[i] 一致: [B,H,K,HD]
    kv_len = torch.tensor([10], dtype=torch.int64)
    mask = torch.zeros(B, H, 1, K, dtype=torch.bool)
    mask[..., 10:] = True

    print("converting (24 blocks, K=1024)...", flush=True)
    ov_model = ov.convert_model(step, example_input=(x, k_all, v_all, kv_len, mask))
    print("inputs:", len(ov_model.inputs), "outputs:", len(ov_model.outputs))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, OUT.with_suffix(".xml"), compress_to_fp16=True)
    print("saved:", str(OUT) + ".xml", flush=True)


if __name__ == "__main__":
    main()
