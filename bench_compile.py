"""torch.compile 低果子验证: SoVITS flow_dec + GPT decode_next_token (XPU, inductor)."""
import sys, os, time, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.disable(logging.INFO)
import torch
from gsv_tts import TTS

print("torch:", torch.__version__)
tts = TTS(device="xpu", dtype="float16")
tts.load_gpt_model(); tts.load_sovits_model(); tts.init_language_module("ja")

KW = dict(spk_audio_path="examples/laffey.mp3", prompt_audio_path="examples/AnAn.ogg",
          prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
          text="へぇー、ここまでしてくれるんですね。")
tts.infer(**KW)  # warm, 建好 runtime buckets

D = torch.device("xpu")

# ---------- 1) SoVITS flow_dec ----------
vq = list(tts.sovits_models.values())[0].vq_model
z_p = torch.randn(1, 192, 208, device=D, dtype=torch.float16)
y_mask = torch.ones(1, 1, 208, device=D, dtype=torch.float16)
ge = torch.randn(1, 1024, 1, device=D, dtype=torch.float16)
with torch.inference_mode():
    ref = vq.flow_dec(z_p, y_mask, ge)
    torch.xpu.synchronize()
    t = time.time(); N = 10
    for _ in range(N):
        vq.flow_dec(z_p, y_mask, ge)
    torch.xpu.synchronize()
    print(f"SoVITS flow_dec eager: {(time.time()-t)/N*1000:.1f} ms/run out={tuple(ref.shape)}")

try:
    c_flow = torch.compile(vq.flow_dec, mode="max-autotune")
    with torch.inference_mode():
        c_flow(z_p, y_mask, ge)  # compile
        torch.xpu.synchronize()
        t = time.time()
        for _ in range(N):
            out_c = c_flow(z_p, y_mask, ge)
        torch.xpu.synchronize()
        print(f"SoVITS flow_dec compiled: {(time.time()-t)/N*1000:.1f} ms/run")
    d = (out_c.float() - ref.float()).abs()
    print(f"  max abs diff: {d.max():.5f}")
except Exception as e:
    print("  flow_dec compile FAIL:", type(e).__name__, str(e)[:300])

# ---------- 2) GPT decode_next_token (整 transformer, 取一个 runtime bucket) ----------
t2s = list(tts.gpt_models.values())[0].t2s_model
print("heads:", t2s.num_head, "layers:", t2s.num_layers, "dim:", t2s.embedding_dim)
bkt = None
for bs, blist in t2s.cuda_graph_buckets.items():
    bkt = blist[-1]; break
print("bucket: batch=", bkt.batch_size, "max_kv=", bkt.max_kv_cache)
with torch.inference_mode():
    ref2 = t2s.t2s_transformer.decode_next_token(
        bkt.graph_xy_pos, bkt.k_cache, bkt.v_cache,
        bkt.kv_cache_len, bkt.decode_attn_mask, bkt.batch_indices)
    torch.xpu.synchronize()
    t = time.time(); N = 20
    for _ in range(N):
        t2s.t2s_transformer.decode_next_token(
            bkt.graph_xy_pos, bkt.k_cache, bkt.v_cache,
            bkt.kv_cache_len, bkt.decode_attn_mask, bkt.batch_indices)
    torch.xpu.synchronize()
    print(f"GPT decode_next_token eager: {(time.time()-t)/N*1000:.2f} ms/step")

try:
    c_dec = torch.compile(t2s.t2s_transformer.decode_next_token, mode="max-autotune")
    with torch.inference_mode():
        c_dec(bkt.graph_xy_pos, bkt.k_cache, bkt.v_cache,
              bkt.kv_cache_len, bkt.decode_attn_mask, bkt.batch_indices)
        torch.xpu.synchronize()
        t = time.time()
        for _ in range(N):
            out_c2 = c_dec(bkt.graph_xy_pos, bkt.k_cache, bkt.v_cache,
                           bkt.kv_cache_len, bkt.decode_attn_mask, bkt.batch_indices)
        torch.xpu.synchronize()
        print(f"GPT decode_next_token compiled: {(time.time()-t)/N*1000:.2f} ms/step")
    d = (out_c2.float() - ref2.float()).abs()
    print(f"  max abs diff: {d.max():.5f}")
except Exception as e:
    print("  decode compile FAIL:", type(e).__name__, str(e)[:300])
