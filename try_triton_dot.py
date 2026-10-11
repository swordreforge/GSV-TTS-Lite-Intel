"""triton-xpu tl.dot 基础正确性测试。"""
import torch, triton, triton.language as tl

@triton.jit
def mm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m * 32 + tl.arange(0, 32); rn = pid_n * 32 + tl.arange(0, 32); rk = tl.arange(0, K)
    a = tl.load(A + rm[:, None] * K + rk[None, :])
    b = tl.load(B + rk[:, None] * N + rn[None, :])
    c = tl.dot(a, b)
    tl.store(C + rm[:, None] * N + rn[None, :], c)

D = torch.device("xpu")
a = torch.randn(64, 64, device=D, dtype=torch.float16)
b = torch.randn(64, 64, device=D, dtype=torch.float16)
c = torch.empty(64, 64, device=D, dtype=torch.float32)
mm[(2, 2)](a, b, c, 64, 64, 64)
torch.xpu.synchronize()
ref = (a.float() @ b.float())
d = (c - ref).abs()
print(f"DOT maxdiff={d.max():.5f} meandiff={d.mean():.6f}", flush=True)
