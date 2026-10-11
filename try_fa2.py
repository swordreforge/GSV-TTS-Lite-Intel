"""FA v2: 方形分块 64x64 + 循环探针，确认 kernel 真在跑。"""
import torch, triton, triton.language as tl

@triton.jit
def ew_ok(X, Y, SB0: tl.constexpr, SB1: tl.constexpr, SB2: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 128 + tl.arange(0, 128)
    m = tl.load(X + offs[:, None] * SB1 + tl.arange(0, 64)[None, :] * SB2)
    e = tl.exp(m - tl.max(m, axis=1)[:, None])
    o = e / tl.sum(e, axis=1)[:, None]
    tl.store(Y + offs[:, None] * SB1 + tl.arange(0, 64)[None, :] * SB2, o)

@triton.jit
def fa2(Q, K, V, Out, Probe,
        N_CTX: tl.constexpr, HEAD_DIM: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_b = tl.program_id(1)
    pid_m = tl.program_id(0)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BN)
    m_prev = tl.full([BM], float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, HEAD_DIM], dtype=tl.float32)
    q = tl.load(Q + pid_b * N_CTX * HEAD_DIM + offs_m[:, None] * HEAD_DIM + offs_d[None, :])
    for start_n in range(0, N_CTX, BN):
        k = tl.load(K + pid_b * N_CTX * HEAD_DIM + (start_n + offs_n)[None, :] * HEAD_DIM + offs_d[:, None])
        s = tl.dot(q, k)
        tl.store(Probe + pid_b * 100 + start_n, tl.max(s))
        m_new = tl.maximum(m_prev, tl.max(s, axis=1))
        alpha = tl.exp(m_prev - m_new)
        p = tl.exp(s - m_new[:, None])
        v = tl.load(V + pid_b * N_CTX * HEAD_DIM + (start_n + offs_n)[:, None] * HEAD_DIM + offs_d[None, :])
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v.to(tl.float16)).to(tl.float32)
        l_prev = l_prev * alpha + tl.sum(p, axis=1)
        m_prev = m_new
    acc = acc / l_prev[:, None]
    tl.store(Out + pid_b * N_CTX * HEAD_DIM + offs_m[:, None] * HEAD_DIM + offs_d[None, :], acc)

D = torch.device("xpu")
torch.manual_seed(1)
x = torch.randn(4, 128, 64, device=D, dtype=torch.float32) * 3
y = torch.empty_like(x)
ew_ok[(4,)](x, y, x.stride(0), x.stride(1), x.stride(2))
torch.xpu.synchronize()
print(f"EW-OK maxdiff={(y - torch.softmax(x, -1)).abs().max():.6f}", flush=True)

torch.manual_seed(0)
BH, N, HD = 8, 128, 64
q = torch.randn(BH, N, HD, device=D, dtype=torch.float16)
k = torch.randn(BH, N, HD, device=D, dtype=torch.float16)
v = torch.randn(BH, N, HD, device=D, dtype=torch.float16)
out = torch.full((BH, N, HD), float('nan'), device=D, dtype=torch.float32)
probe = torch.zeros(BH, 100, device=D, dtype=torch.float32)
fa2[(2, BH)](q, k, v, out, probe, N_CTX=N, HEAD_DIM=HD, BM=64, BN=64)
torch.xpu.synchronize()
out2 = torch.empty(BH, N, HD, device=D, dtype=torch.float32)
probe2 = torch.zeros(BH, 100, device=D, dtype=torch.float32)
fa2[(2, BH)](q, k, v, out2, probe2, N_CTX=N, HEAD_DIM=HD, BM=64, BN=64)
torch.xpu.synchronize()
import math
n_nan = int(torch.isnan(out).sum())
print("NAN-COUNT in out:", n_nan, "of", out.numel(), flush=True)
print("INPROC out1vsout2:", round(float((out - out2).abs().max()), 5), flush=True)
print("INPROC probe1vsprobe2:", round(float((probe - probe2).abs().max()), 5), flush=True)
print("PROBE(row0):", [round(float(probe[0, i]), 2) for i in (0, 64)], flush=True)
o_r = torch.nn.functional.scaled_dot_product_attention(
    q.view(1, BH, N, HD), k.view(1, BH, N, HD), v.view(1, BH, N, HD)).view(BH, N, HD)
torch.xpu.synchronize()
print(f"FA2 maxdiff={(out.float() - o_r.float()).abs().max():.5f}", flush=True)
