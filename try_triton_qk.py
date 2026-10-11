"""二分 FA kernel: 只算 S=QK^T，看 dot 在循环里是否正常。"""
import torch, triton, triton.language as tl

@triton.jit
def qk_only(Q, K, S,
            stride_qb, stride_qm, stride_qk,
            stride_kb, stride_km, stride_kk,
            stride_sb, stride_sm, stride_sn,
            N_CTX: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid_b = tl.program_id(1)
    pid_m = tl.program_id(0)
    offs_m = pid_m * 32 + tl.arange(0, 32)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, 128)
    q = tl.load(Q + pid_b * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    k = tl.load(K + pid_b * stride_kb + offs_n[None, :] * stride_km + offs_d[:, None] * stride_kk)
    s = tl.dot(q, k)
    tl.store(S + pid_b * stride_sb + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn, s)

D = torch.device("xpu")
torch.manual_seed(0)
BH, N, HD = 16, 128, 64
q = torch.randn(BH, N, HD, device=D, dtype=torch.float16)
k = torch.randn(BH, N, HD, device=D, dtype=torch.float16)
s = torch.empty(BH, N, N, device=D, dtype=torch.float32)
qk_only[(4, BH)](q, k, s,
                 q.stride(0), q.stride(1), q.stride(2),
                 k.stride(0), k.stride(1), k.stride(2),
                 s.stride(0), s.stride(1), s.stride(2),
                 N_CTX=N, HEAD_DIM=HD)
torch.xpu.synchronize()
ref = (q.float() @ k.float().transpose(1, 2))
d = (s - ref).abs()
print(f"QK-ONLY maxdiff={d.max():.5f} meandiff={d.mean():.6f}", flush=True)
