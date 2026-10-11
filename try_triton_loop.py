"""二分 FA: 循环内 BLOCK_N=32 的 dot 累加是否正常；另测 K=32 的独立 dot。"""
import torch, triton, triton.language as tl

@triton.jit
def loop_dot(Q, K, S,
             stride_qb, stride_qm, stride_qk,
             stride_kb, stride_km, stride_kk,
             stride_sb, stride_sm, stride_sn,
             N_CTX: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_b = tl.program_id(1)
    pid_m = tl.program_id(0)
    offs_m = pid_m * 32 + tl.arange(0, 32)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, 128)
    q = tl.load(Q + pid_b * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(K + pid_b * stride_kb + (start_n + tl.arange(0, BLOCK_N))[None, :] * stride_km + offs_d[:, None] * stride_kk)
        s = tl.dot(q, k)
        tl.store(S + pid_b * stride_sb + offs_m[:, None] * stride_sm + (start_n + tl.arange(0, BLOCK_N))[None, :] * stride_sn, s)

@triton.jit
def mm32(A, B, C):
    rm = tl.program_id(0) * 32 + tl.arange(0, 32)
    rn = tl.program_id(1) * 32 + tl.arange(0, 32)
    rk = tl.arange(0, 32)
    c = tl.dot(tl.load(A + rm[:, None] * 32 + rk[None, :]),
               tl.load(B + rk[:, None] * 32 + rn[None, :]))
    tl.store(C + rm[:, None] * 32 + rn[None, :], c)

D = torch.device("xpu")
torch.manual_seed(0)
BH, N, HD = 4, 128, 64
q = torch.randn(BH, N, HD, device=D, dtype=torch.float16)
k = torch.randn(BH, N, HD, device=D, dtype=torch.float16)
s = torch.empty(BH, N, N, device=D, dtype=torch.float32)
loop_dot[(4, BH)](q, k, s,
                  q.stride(0), q.stride(1), q.stride(2),
                  k.stride(0), k.stride(1), k.stride(2),
                  s.stride(0), s.stride(1), s.stride(2),
                  N_CTX=N, HEAD_DIM=HD, BLOCK_N=32)
torch.xpu.synchronize()
ref = (q.float() @ k.float().transpose(1, 2))
print(f"LOOP-DOT maxdiff={(s - ref).abs().max():.5f}", flush=True)

a = torch.randn(64, 32, device=D, dtype=torch.float16)
b = torch.randn(32, 64, device=D, dtype=torch.float16)
c = torch.empty(64, 64, device=D, dtype=torch.float32)
mm32[(2, 2)](a, b, c)
torch.xpu.synchronize()
print(f"K32-DOT maxdiff={(c - a.float() @ b.float()).abs().max():.5f}", flush=True)
