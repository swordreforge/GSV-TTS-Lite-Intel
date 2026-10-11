"""管线真实形状下 XPU math-SDPA 是否正确：decode 步 (hd=32) 与 prompt 步。"""
import torch
import torch.nn.functional as F

D = torch.device("xpu")
torch.manual_seed(0)

def check(tag, qb, kb, vb, mask):
    o = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask)
    B, H, Lq, Hd = qb.shape
    s = (qb.float().transpose(1, 2).reshape(B * H, Lq, Hd)
         @ kb.float().transpose(1, 2).reshape(B * H, -1, Hd).transpose(1, 2))
    if mask is not None:
        s = s.masked_fill(mask.expand(B, H, Lq, -1).reshape(B * H, Lq, -1), float("-inf"))
    ref = (torch.softmax(s, dim=-1) @ vb.float().transpose(1, 2).reshape(B * H, -1, Hd))
    ref = ref.view(B, H, Lq, Hd)
    torch.xpu.synchronize()
    print(f"{tag}: maxdiff={(o.float() - ref).abs().max():.5f}", flush=True)

# decode 步: q(1,16,1,32), kv(1,16,200,32), mask(1,16,1,200)
qb = torch.randn(1, 16, 1, 32, device=D, dtype=torch.float16)
kb = torch.randn(1, 16, 200, 32, device=D, dtype=torch.float16)
vb = torch.randn(1, 16, 200, 32, device=D, dtype=torch.float16)
mk = torch.zeros(1, 16, 1, 200, device=D, dtype=torch.bool)
check("decode hd=32 L=200 mask", qb, kb, vb, mk)
check("decode hd=32 L=200 nomask", qb, kb, vb, None)

# prompt 步: (1,16,128,32) 全长
qb = torch.randn(1, 16, 128, 32, device=D, dtype=torch.float16)
kb = torch.randn(1, 16, 128, 32, device=D, dtype=torch.float16)
vb = torch.randn(1, 16, 128, 32, device=D, dtype=torch.float16)
mk = torch.zeros(1, 16, 128, 128, device=D, dtype=torch.bool)
check("prompt hd=32 L=128 mask", qb, kb, vb, mk)

# 之前出错的形状复现
qb = torch.randn(1, 8, 128, 64, device=D, dtype=torch.float16)
kb = torch.randn(1, 8, 128, 64, device=D, dtype=torch.float16)
vb = torch.randn(1, 8, 128, 64, device=D, dtype=torch.float16)
check("hd=64 L=128 nomask", qb, kb, vb, None)
