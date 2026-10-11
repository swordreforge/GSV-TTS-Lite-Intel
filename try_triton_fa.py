"""最小 Triton flash-attention 前向能否在 Intel Arc (XPU) 上跑通。"""
import sys, time
import torch
import triton
import triton.language as tl

print("triton:", triton.__version__)

@triton.jit
def _fa_fwd(Q, K, V, Out, L,
            stride_qb, stride_qm, stride_qk,
            stride_kb, stride_km, stride_kk,
            stride_vb, stride_vm, stride_vk,
            stride_ob, stride_om, stride_ok,
            stride_lb,
            N_CTX: tl.constexpr, HEAD_DIM: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_b = tl.program_id(1)
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, 64)
    m_prev = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q = tl.load(Q + pid_b * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(K + pid_b * stride_kb + (start_n + offs_n)[None, :] * stride_km + offs_d[:, None] * stride_kk)
        s = tl.dot(q, k)
        m_new = tl.maximum(m_prev, tl.max(s, axis=1))
        alpha = tl.exp(m_prev - m_new)
        p = tl.exp(s - m_new[:, None])
        v = tl.load(V + pid_b * stride_vb + (start_n + offs_n)[:, None] * stride_vm + offs_d[None, :] * stride_vk)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v.to(tl.float16)).to(tl.float32)
        l_prev = l_prev * alpha + tl.sum(p, axis=1)
        m_prev = m_new
    acc = acc / l_prev[:, None]
    tl.store(Out + pid_b * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok, acc.to(tl.float16))
    tl.store(L + pid_b * stride_lb + offs_m, m_prev + tl.log(l_prev))

def fa_triton(q, k, v):
    B_H, N, D = q.shape[0], q.shape[1], q.shape[2]
    out = torch.empty_like(q)
    L = torch.empty((B_H, N), device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(N, 32), B_H)
    _fa_fwd[grid](q, k, v, out, L,
                  q.stride(0), q.stride(1), q.stride(2),
                  k.stride(0), k.stride(1), k.stride(2),
                  v.stride(0), v.stride(1), v.stride(2),
                  out.stride(0), out.stride(1), out.stride(2),
                  L.stride(0),
                  N_CTX=N, HEAD_DIM=D, BLOCK_M=32, BLOCK_N=64)
    return out

D = torch.device("xpu")
torch.manual_seed(0)
BH, N, HD = 16, 128, 64  # 16 heads folded into batch
q = torch.randn(BH, N, HD, device=D, dtype=torch.float16)
k = torch.randn(BH, N, HD, device=D, dtype=torch.float16)
v = torch.randn(BH, N, HD, device=D, dtype=torch.float16)

try:
    o_t = fa_triton(q, k, v)
    torch.xpu.synchronize()
    o_r = torch.nn.functional.scaled_dot_product_attention(
        q.view(1, 16, N, HD), k.view(1, 16, N, HD), v.view(1, 16, N, HD)).view(BH, N, HD)
    torch.xpu.synchronize()
    d = (o_t.float() - o_r.float()).abs()
    print(f"TRITON-FA works on XPU! max diff vs math: {d.max():.5f}")
    for _ in range(3): fa_triton(q, k, v)
    torch.xpu.synchronize(); t = time.time()
    for _ in range(20): fa_triton(q, k, v)
    torch.xpu.synchronize(); print(f"triton FA: {(time.time()-t)/20*1000:.2f} ms")
    t = time.time()
    for _ in range(20):
        torch.nn.functional.scaled_dot_product_attention(
            q.view(1, 16, N, HD), k.view(1, 16, N, HD), v.view(1, 16, N, HD))
    torch.xpu.synchronize(); print(f"torch math: {(time.time()-t)/20*1000:.2f} ms")
except Exception as e:
    print("TRITON-FA FAIL:", type(e).__name__, str(e)[:500])
