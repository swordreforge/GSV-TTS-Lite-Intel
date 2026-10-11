"""mask=None vs 有 mask：XPU math-SDPA 哪条路错；triton FA vs eager 定案。"""
import torch, triton, triton.language as tl

@triton.jit
def fa2(Q, K, V, Out, N_CTX: tl.constexpr, HEAD_DIM: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr):
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
        s = tl.dot(q, k).to(tl.float32) * 0.125
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
torch.manual_seed(0)
BH, N, HD = 8, 128, 64
q = torch.randn(BH, N, HD, device=D, dtype=torch.float16)
k = torch.randn(BH, N, HD, device=D, dtype=torch.float16)
v = torch.randn(BH, N, HD, device=D, dtype=torch.float16)
qb, kb, vb = q.view(1, BH, N, HD), k.view(1, BH, N, HD), v.view(1, BH, N, HD)

s = (q.float() @ k.float().transpose(1, 2))
o_eager = (torch.softmax(s, dim=-1) @ v.float())

o_nomask = torch.nn.functional.scaled_dot_product_attention(qb, kb, vb)
falsemask = torch.zeros(1, BH, N, N, device=D, dtype=torch.bool)
o_falsemask = torch.nn.functional.scaled_dot_product_attention(qb, kb, vb, attn_mask=falsemask)
s_masked = s.masked_fill(falsemask[0, 0], float("-inf"))
o_eager_masked = (torch.softmax(s_masked, dim=-1) @ v.float())
torch.xpu.synchronize()

print(f"A: SDPA-nomask vs eager        : {(o_nomask.float() - o_eager).abs().max():.5f}", flush=True)
print(f"B: SDPA-falsemask vs eager    : {(o_falsemask.float() - o_eager_masked).abs().max():.5f}", flush=True)

out = torch.empty(BH, N, HD, device=D, dtype=torch.float32)
fa2[(2, BH)](q, k, v, out, N_CTX=N, HEAD_DIM=HD, BM=64, BN=64)
torch.xpu.synchronize()
print(f"C: triton-FA vs eager         : {(out - o_eager).abs().max():.5f}", flush=True)
print(f"D: triton-FA vs SDPA(math)     : {(out - o_nomask.float()).abs().max():.5f}", flush=True)
