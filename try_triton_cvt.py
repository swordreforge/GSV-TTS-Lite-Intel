"""测 fp32->fp16 转换在 triton-xpu 上是否正常。"""
import torch, triton, triton.language as tl

@triton.jit
def cvt(X, Y, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(Y + offs, tl.load(X + offs).to(tl.float16))

D = torch.device("xpu")
torch.manual_seed(0)
x = torch.randn(256, device=D, dtype=torch.float32)
y = torch.empty(256, device=D, dtype=torch.float16)
cvt[(1,)](x, y, 256)
torch.xpu.synchronize()
print(f"CVT maxdiff={(y.float() - x.to(torch.float16).float()).abs().max():.6f}", flush=True)
print("sample triton:", y[:5].float().tolist(), flush=True)
print("sample torch :", x[:5].to(torch.float16).float().tolist(), flush=True)
