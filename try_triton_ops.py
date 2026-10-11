"""逐个测 exp / max / sum / div。"""
import torch, triton, triton.language as tl

@triton.jit
def k_exp(X, Y):
    offs = tl.arange(0, 128)
    tl.store(Y + offs, tl.exp(tl.load(X + offs).to(tl.float32)))

@triton.jit
def k_max(X, Y):
    offs = tl.program_id(0) * 128 + tl.arange(0, 128)
    m = tl.load(X + offs[:, None] * 64 + tl.arange(0, 64)[None, :])
    tl.store(Y + offs, tl.max(m, axis=1))

@triton.jit
def k_sum(X, Y):
    offs = tl.program_id(0) * 128 + tl.arange(0, 128)
    m = tl.load(X + offs[:, None] * 64 + tl.arange(0, 64)[None, :])
    tl.store(Y + offs, tl.sum(m, axis=1))

@triton.jit
def k_div(X, Y):
    offs = tl.program_id(0) * 128 + tl.arange(0, 128)
    m = tl.load(X + offs[:, None] * 64 + tl.arange(0, 64)[None, :])
    tl.store(Y + offs[:, None] * 64 + tl.arange(0, 64)[None, :], m / 2.0)

D = torch.device("xpu")
x1 = (torch.randn(128, device=D) * 3).to(torch.float32)
y1 = torch.empty_like(x1)
k_exp[(1,)](x1, y1)
torch.xpu.synchronize()
print(f"EXP maxdiff={(y1 - torch.exp(x1)).abs().max():.6f}", flush=True)

x2 = torch.randn(4, 128, 64, device=D, dtype=torch.float32)
y2 = torch.empty(4, 128, device=D)
k_max[(4,)](x2, y2)
torch.xpu.synchronize()
print(f"MAX maxdiff={(y2 - x2.amax(-1)).abs().max():.6f}", flush=True)

k_sum[(4,)](x2, y2)
torch.xpu.synchronize()
print(f"SUM maxdiff={(y2 - x2.sum(-1)).abs().max():.6f}", flush=True)

k_div[(4,)](x2, y2)
torch.xpu.synchronize()
print("DIV skipped-storage-check", flush=True)
