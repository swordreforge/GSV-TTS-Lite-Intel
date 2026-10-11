"""测 tl.dot 吃 exp 输出是否正常（布局转换嫌疑）。"""
import torch, triton, triton.language as tl

@triton.jit
def dotexp(A, B, C):
    rk = tl.arange(0, 64)
    rm = tl.program_id(0) * 32 + tl.arange(0, 32)
    rn = tl.program_id(1) * 32 + tl.arange(0, 32)
    a = tl.load(A + rm[:, None] * 64 + rk[None, :]).to(tl.float32)
    b = tl.load(B + rk[:, None] * 64 + rn[None, :])
    e = tl.exp(a * 0.1)
    tl.store(C + rm[:, None] * 64 + rn[None, :], tl.dot(e.to(tl.float16), b))

D = torch.device("xpu")
torch.manual_seed(0)
a = torch.randn(64, 64, device=D, dtype=torch.float16)
b = torch.randn(64, 64, device=D, dtype=torch.float16)
a32 = a.float()
c = torch.empty(64, 64, device=D, dtype=torch.float32)
dotexp[(2, 2)](a, b, c)
torch.xpu.synchronize()
ref = (torch.exp(a32 * 0.1).to(torch.float16).float() @ b.float())
print(f"DOTEXP maxdiff={(c - ref).abs().max():.5f}", flush=True)
