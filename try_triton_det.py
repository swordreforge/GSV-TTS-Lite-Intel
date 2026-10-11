"""triton-xpu tl.dot 确定性测试：同一 kernel 连跑两次比对。"""
import torch, triton, triton.language as tl

@triton.jit
def mm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m * 32 + tl.arange(0, 32); rn = pid_n * 32 + tl.arange(0, 32); rk = tl.arange(0, K)
    a = tl.load(A + rm[:, None] * K + rk[None, :])
    b = tl.load(B + rk[:, None] * N + rn[None, :])
    tl.store(C + rm[:, None] * N + rn[None, :], tl.dot(a, b))

D = torch.device("xpu")
torch.manual_seed(0)
a = torch.randn(64, 64, device=D, dtype=torch.float16)
b = torch.randn(64, 64, device=D, dtype=torch.float16)
c1 = torch.empty(64, 64, device=D, dtype=torch.float32)
c2 = torch.empty(64, 64, device=D, dtype=torch.float32)
mm[(2, 2)](a, b, c1, 64, 64, 64)
mm[(2, 2)](a, b, c2, 64, 64, 64)
torch.xpu.synchronize()
print(f"RUN1vsRUN2 maxdiff={(c1 - c2).abs().max():.6f}", flush=True)
ref = (a.float() @ b.float())
print(f"RUN1vsREF maxdiff={(c1 - ref).abs().max():.6f}", flush=True)
print(f"RUN2vsREF maxdiff={(c2 - ref).abs().max():.6f}", flush=True)
