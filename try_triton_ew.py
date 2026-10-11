"""二分 FA: exp / max / sum / div 在 XPU 上是否正常。"""
import torch, triton, triton.language as tl

@triton.jit
def ew(X, Y, N: tl.constexpr):
    offs = tl.program_id(0) * 128 + tl.arange(0, 128)
    m = tl.load(X + offs[:, None] * 128 + tl.arange(0, 64)[None, :])
    e = tl.exp(m - tl.max(m, axis=1)[:, None])
    s = tl.sum(e, axis=1)
    o = e / s[:, None]
    tl.store(Y + offs[:, None] * 128 + tl.arange(0, 64)[None, :], o)

D = torch.device("xpu")
torch.manual_seed(1)
x = torch.randn(4, 128, 64, device=D, dtype=torch.float32) * 3
y = torch.empty_like(x)
ew[(4,)](x, y, 4)
torch.xpu.synchronize()
ref = torch.softmax(x.float(), dim=-1)
print(f"EW maxdiff={(y - ref).abs().max():.5f}", flush=True)
