"""映射 triton-xpu tl.dot 在不同 K 下的正确性。"""
import torch, triton, triton.language as tl

@triton.jit
def mmk(A, B, C, K: tl.constexpr):
    rm = tl.program_id(0) * 32 + tl.arange(0, 32)
    rn = tl.program_id(1) * 32 + tl.arange(0, 32)
    rk = tl.arange(0, K)
    tl.store(C + rm[:, None] * 64 + rn[None, :],
             tl.dot(tl.load(A + rm[:, None] * K + rk[None, :]),
                    tl.load(B + rk[:, None] * 64 + rn[None, :])))

D = torch.device("xpu")
for K in (16, 32, 64, 128):
    torch.manual_seed(0)
    a = torch.randn(64, K, device=D, dtype=torch.float16)
    b = torch.randn(K, 64, device=D, dtype=torch.float16)
    c = torch.empty(64, 64, device=D, dtype=torch.float32)
    mmk[(2, 2)](a, b, c, K)
    torch.xpu.synchronize()
    print(f"K={K}: maxdiff={(c - (a.float() @ b.float())).abs().max():.5f}", flush=True)
