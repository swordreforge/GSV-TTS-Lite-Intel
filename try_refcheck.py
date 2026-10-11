"""确认参照系：torch XPU math-SDPA vs 纯 eager softmax。"""
import torch

D = torch.device("xpu")
torch.manual_seed(0)
BH, N, HD = 8, 128, 64
q = torch.randn(BH, N, HD, device=D, dtype=torch.float16)
k = torch.randn(BH, N, HD, device=D, dtype=torch.float16)
v = torch.randn(BH, N, HD, device=D, dtype=torch.float16)

o_sdpa = torch.nn.functional.scaled_dot_product_attention(
    q.view(1, BH, N, HD), k.view(1, BH, N, HD), v.view(1, BH, N, HD)).view(BH, N, HD)
s = (q.float() @ k.float().transpose(1, 2))
o_eager = (torch.softmax(s, dim=-1) @ v.float()).to(torch.float16)
torch.xpu.synchronize()
print(f"SDPA-vs-EAGER maxdiff={(o_sdpa.float() - o_eager.float()).abs().max():.5f}", flush=True)
print(f"EAGER mean={o_eager.float().mean():.4f} SDPA mean={o_sdpa.float().mean():.4f}", flush=True)
