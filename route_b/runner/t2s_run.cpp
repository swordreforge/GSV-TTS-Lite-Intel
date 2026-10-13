// 路线 B runner：gguf 加载 + pe现场生成 + entry + prefill + decode闭环 + sampler（SYCL）。
// 两种模式：单次（默认，兼容 validation）/ 常驻服务（--serve：stdin 接 JOB，权重+SYCL 编译常驻）。
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-sycl.h"
#include "gguf.h"
#include "sampler.cpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <chrono>

static const int H = 16, HD = 32, DIM = 512, NL = 24, VOCAB = 1025;
static const char * GGUF_DEFAULT = "/home/swordreforge/.cache/gsv/gguf/t2s-s1v3-f16.gguf";
static std::string GGUF_PATH() { const char * e = getenv("RUNNER_GGUF"); return e ? e : GGUF_DEFAULT; }
static const char * CDIR = "/tmp/opencode/t2s_closed/";
static std::string IDIR() { const char * e = getenv("RUNNER_IN_DIR"); return e ? e : CDIR; }
static bool VALIDATE() { const char * e = getenv("RUNNER_VALIDATE"); return !e || atoi(e); }
static const char * EDIR = "/tmp/opencode/t2s_emb/";
static const int K = 1024;  // 覆盖 645 步 greedy（torch bucket 到 786）
static const char * ARCH = "gpt-sovits-t2s";
static const int EOS_ID = 1024;
static const int SUPP[3] = {280, 486, 1024};

static std::vector<uint8_t> g_gguf_data;
static std::map<std::string, std::pair<size_t, size_t>> g_map;

static void gguf_load() {
    std::string gp = GGUF_PATH();
    FILE * f = fopen(gp.c_str(), "rb");
    if (!f) { printf("no gguf\n"); exit(1); }
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    g_gguf_data.resize(sz);
    if (fread(g_gguf_data.data(), 1, sz, f) != sz) { printf("read short\n"); exit(1); }
    fclose(f);
    FILE * f2 = fopen(gp.c_str(), "rb");
    struct gguf_init_params p = { true, nullptr };
    struct gguf_context * ctx = gguf_init_from_file_ptr(f2, p);
    if (!ctx) { printf("gguf parse fail\n"); exit(1); }
    size_t data_off = gguf_get_data_offset(ctx);
    int64_t n = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n; i++) {
        const char * nm = gguf_get_tensor_name(ctx, i);
        size_t off = gguf_get_tensor_offset(ctx, i);
        const int64_t * ne = gguf_get_tensor_ne(ctx, i);
        size_t nel = 1;
        for (int d = 0; d < 4; d++) nel *= (size_t)ne[d];
        g_map[nm] = { data_off + off, nel };
    }
    printf("gguf tensors: %lld\n", (long long)n);
    fclose(f2);
}

static const uint8_t * gguf_bytes(const std::string & name, size_t expect_n) {
    auto it = g_map.find(name);
    if (it == g_map.end()) { printf("missing tensor %s\n", name.c_str()); exit(1); }
    if (it->second.second != expect_n) { printf("ne mismatch %s: %zu vs %zu\n", name.c_str(), it->second.second, expect_n); exit(1); }
    return g_gguf_data.data() + it->second.first;
}

static std::vector<float> load_f32(const std::string & dir, const std::string & name, size_t n) {
    FILE * f = fopen((dir + name + ".bin").c_str(), "rb");
    if (!f) { printf("missing %s%s\n", dir.c_str(), name.c_str()); exit(1); }
    std::vector<float> v(n);
    if (fread(v.data(), sizeof(float), n, f) != n) { printf("short %s\n", name.c_str()); exit(1); }
    fclose(f);
    return v;
}
static std::vector<int64_t> load_i64(const std::string & dir, const std::string & name, size_t n) {
    FILE * f = fopen((dir + name + ".bin").c_str(), "rb");
    if (!f) { printf("missing %s%s\n", dir.c_str(), name.c_str()); exit(1); }
    std::vector<int64_t> v(n);
    if (fread(v.data(), sizeof(int64_t), n, f) != n) { printf("short %s\n", name.c_str()); exit(1); }
    fclose(f);
    return v;
}
static std::vector<uint8_t> load_u8(const std::string & dir, const std::string & name, size_t n) {
    FILE * f = fopen((dir + name + ".bin").c_str(), "rb");
    if (!f) { printf("missing %s%s\n", dir.c_str(), name.c_str()); exit(1); }
    std::vector<uint8_t> v(n);
    if (fread(v.data(), 1, n, f) != n) { printf("short %s\n", name.c_str()); exit(1); }
    fclose(f);
    return v;
}

// ---- 常驻状态（init_once 建好，job 间复用） ----
struct Lyr { ggml_tensor *qw, *qb, *ow, *ob, *n1w, *n1b, *m1w, *m1b, *m2w, *m2b, *n2w, *n2b; };
struct Weights { ggml_tensor *text_emb, *audio_emb, *bert_w, *bert_b, *predict_w; std::vector<Lyr> ly; };
static std::string G_A;
static Weights G_WS, G_WC;
static ggml_tensor *G_text_emb, *G_audio_emb, *G_bert_w, *G_bert_b, *G_predict_w;
static float G_alpha_t, G_alpha_a;
static std::vector<float> G_pe;
static ggml_backend_t G_backend, G_backend_cpu;
static ggml_backend_buffer_t G_buf, G_bufc;
static ggml_gallocr_t G_galloc, G_galloc_cpu;
static std::vector<std::pair<ggml_tensor *, std::vector<float>>> G_keep16, G_keep16c;
static std::vector<std::pair<ggml_tensor *, std::vector<float>>> G_keep32, G_keep32c;
static ggml_context * G_ctxw;

static Weights build_weights(ggml_context * ctx,
                             std::vector<std::pair<ggml_tensor *, std::vector<float>>> & k16,
                             std::vector<std::pair<ggml_tensor *, std::vector<float>>> & k32) {
    auto T16 = [&](const std::string & gname, int64_t n0, int64_t n1, bool untranspose = false) {
        // fp32 化：gguf 内 fp16，加载即转 fp32（fp16 权重已证伪：随机生成退化）
        size_t n = (size_t)n0 * n1;
        auto * h16 = (const ggml_fp16_t *)gguf_bytes(gname, n);
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1);
        std::vector<float> raw(n);
        for (size_t j = 0; j < n; j++) raw[j] = ggml_fp16_to_fp32(h16[j]);
        if (untranspose) {
            std::vector<float> tu(n);
            for (int64_t i = 0; i < n0; i++)
                for (int64_t o = 0; o < n1; o++) tu[o * n0 + i] = raw[i * n1 + o];
            raw.swap(tu);
        }
        k16.emplace_back(t, std::move(raw));
        return t;
    };
    auto T32 = [&](const std::string & gname, int64_t n0) {
        size_t n = (size_t)n0;
        const uint8_t * src = gguf_bytes(gname, n);
        auto * h = (const ggml_fp16_t *)src;
        std::vector<float> v(n);
        for (size_t i = 0; i < n; i++) v[i] = ggml_fp16_to_fp32(h[i]);
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n0);
        k32.emplace_back(t, std::move(v));
        return t;
    };
    Weights W;
    W.text_emb = T16(G_A + ".ar_text_emb.weight", 512, 732);
    W.audio_emb = T16(G_A + ".ar_audio_emb.weight", 512, VOCAB);
    W.bert_w = T16(G_A + ".bert_proj.weight", 1024, 512, true);
    W.bert_b = T32(G_A + ".bert_proj.bias", 512);
    W.predict_w = T16(G_A + ".predict.weight", 512, VOCAB, true);
    W.ly.resize(NL);
    char gb[128];
    for (int i = 0; i < NL; i++) {
        snprintf(gb, 128, "%s.blk.%d.qkv.weight", G_A.c_str(), i); W.ly[i].qw = T16(gb, DIM, 3 * DIM, true);
        snprintf(gb, 128, "%s.blk.%d.qkv.bias", G_A.c_str(), i); W.ly[i].qb = T32(gb, 3 * DIM);
        snprintf(gb, 128, "%s.blk.%d.out.weight", G_A.c_str(), i); W.ly[i].ow = T16(gb, DIM, DIM, true);
        snprintf(gb, 128, "%s.blk.%d.out.bias", G_A.c_str(), i); W.ly[i].ob = T32(gb, DIM);
        snprintf(gb, 128, "%s.blk.%d.n1.weight", G_A.c_str(), i); W.ly[i].n1w = T32(gb, DIM);
        snprintf(gb, 128, "%s.blk.%d.n1.bias", G_A.c_str(), i); W.ly[i].n1b = T32(gb, DIM);
        snprintf(gb, 128, "%s.blk.%d.mlp1.weight", G_A.c_str(), i); W.ly[i].m1w = T16(gb, DIM, 4 * DIM, true);
        snprintf(gb, 128, "%s.blk.%d.mlp1.bias", G_A.c_str(), i); W.ly[i].m1b = T32(gb, 4 * DIM);
        snprintf(gb, 128, "%s.blk.%d.mlp2.weight", G_A.c_str(), i); W.ly[i].m2w = T16(gb, 4 * DIM, DIM, true);
        snprintf(gb, 128, "%s.blk.%d.mlp2.bias", G_A.c_str(), i); W.ly[i].m2b = T32(gb, DIM);
        snprintf(gb, 128, "%s.blk.%d.n2.weight", G_A.c_str(), i); W.ly[i].n2w = T32(gb, DIM);
        snprintf(gb, 128, "%s.blk.%d.n2.bias", G_A.c_str(), i); W.ly[i].n2b = T32(gb, DIM);
    }
    return W;
}

static void init_once() {
    gguf_load();
    G_A = ARCH;

    struct ggml_init_params params = { 2048ULL * 1024 * 1024, nullptr, true };  // SYCL+CPU 双份权重
    G_ctxw = ggml_init(params);

    G_WS = build_weights(G_ctxw, G_keep16, G_keep32);
    G_text_emb = G_WS.text_emb; G_audio_emb = G_WS.audio_emb;
    G_bert_w = G_WS.bert_w; G_bert_b = G_WS.bert_b; G_predict_w = G_WS.predict_w;
    { auto at = gguf_bytes(G_A + ".ar_text_pos_alpha", 1);
      auto aa = gguf_bytes(G_A + ".ar_audio_pos_alpha", 1);
      G_alpha_t = ggml_fp16_to_fp32(((ggml_fp16_t *)at)[0]);
      G_alpha_a = ggml_fp16_to_fp32(((ggml_fp16_t *)aa)[0]); }
    printf("alphas(gguf): text=%.7f audio=%.7f\n", G_alpha_t, G_alpha_a);

    G_backend = ggml_backend_sycl_init(0);
    G_backend_cpu = ggml_backend_cpu_init();
    G_buf = ggml_backend_alloc_ctx_tensors(G_ctxw, G_backend);
    for (auto & kv : G_keep16) ggml_backend_tensor_set(kv.first, kv.second.data(), 0, kv.second.size() * sizeof(float));
    for (auto & kv : G_keep32) ggml_backend_tensor_set(kv.first, kv.second.data(), 0, kv.second.size() * sizeof(float));
    // ---- CPU 镜像：prefill 跑 CPU（0.2s），SYCL 只跑 decode ----
    G_WC = build_weights(G_ctxw, G_keep16c, G_keep32c);
    G_bufc = ggml_backend_alloc_ctx_tensors(G_ctxw, G_backend_cpu);
    for (auto & kv : G_keep16c) ggml_backend_tensor_set(kv.first, kv.second.data(), 0, kv.second.size() * sizeof(float));
    for (auto & kv : G_keep32c) ggml_backend_tensor_set(kv.first, kv.second.data(), 0, kv.second.size() * sizeof(float));
    G_galloc_cpu = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    G_galloc = ggml_gallocr_new(ggml_backend_sycl_buffer_type(0));

    // ---- pe 现场生成（L 无关，job 间复用） ----
    G_pe.assign(4000 * 512, 0);
    for (int p = 0; p < 4000; p++)
        for (int d = 0; d < 256; d++) {
            float div = expf((float)(d * 2) * (float)(-(log(10000.0) / 512.0)));
            G_pe[p * 512 + d * 2] = sinf(p * div);
            G_pe[p * 512 + d * 2 + 1] = cosf(p * div);
        }
}

// 单 job：输入→entry→prefill→decode→落盘。ctxj 按 job 建删，权重/backends 常驻。
// 环境变量：RUNNER_IN_DIR / RUNNER_VALIDATE / RUNNER_TOKENS_OUT / RUNNER_TOP_K / RUNNER_SEED
// 返回 0=ok；ntoks 输出 sampled token 数。
static int run_job(int * ntoks_out) {
    auto tick = []() { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };

    // ---- 输入（host） ----
    auto xyl = load_i64(IDIR(), "xlen_ylen", 2);
    int X = (int)xyl[0], Y = (int)xyl[1], L = X + Y;
    auto vx = load_i64(IDIR(), "x_ids", X);
    auto vy = load_i64(IDIR(), "y", Y);
    FILE * fb = fopen((IDIR() + "bert.bin").c_str(), "rb");
    if (!fb) { printf("missing %sbert.bin\n", IDIR().c_str()); return 1; }
    std::vector<float> vbert((size_t)X * 1024);
    if (fread(vbert.data(), 4, vbert.size(), fb) != vbert.size()) { printf("bert short\n"); fclose(fb); return 1; }
    fclose(fb);

    struct ggml_init_params jparams = { 512ULL * 1024 * 1024, nullptr, true };
    ggml_context * ctxj = ggml_init(jparams);
    ggml_backend_buffer_t bufj = nullptr, bufjc = nullptr;

    // SYCL job tensors（先建→SYCL buffer；顺序决定归属，alloc 跳过已有 buffer）
    std::vector<ggml_tensor *> kc(NL), vc(NL);
    for (int i = 0; i < NL; i++) {
        kc[i] = ggml_new_tensor_3d(ctxj, GGML_TYPE_F32, HD, K, H);
        vc[i] = ggml_new_tensor_3d(ctxj, GGML_TYPE_F32, HD, K, H);
    }
    ggml_tensor * x1 = ggml_new_tensor_2d(ctxj, GGML_TYPE_F32, DIM, 1);    // decode 输入
    ggml_tensor * bias3 = ggml_new_tensor_3d(ctxj, GGML_TYPE_F32, K, 1, H);
    bufj = ggml_backend_alloc_ctx_tensors(ctxj, G_backend);
    // CPU job tensors（后建→CPU buffer）
    std::vector<ggml_tensor *> kc_c(NL), vc_c(NL);
    for (int i = 0; i < NL; i++) {
        kc_c[i] = ggml_new_tensor_3d(ctxj, GGML_TYPE_F32, HD, K, H);
        vc_c[i] = ggml_new_tensor_3d(ctxj, GGML_TYPE_F32, HD, K, H);
    }
    ggml_tensor * xpf_c = ggml_new_tensor_2d(ctxj, GGML_TYPE_F32, DIM, L);
    ggml_tensor * pbias_c = ggml_new_tensor_3d(ctxj, GGML_TYPE_F32, L, L, H);
    bufjc = ggml_backend_alloc_ctx_tensors(ctxj, G_backend_cpu);
    const float scale = 1.0f / sqrtf((float)HD);
    { std::vector<float> z((size_t)HD * K * H, 0);
      for (int i = 0; i < NL; i++) {
          ggml_backend_tensor_set(kc[i], z.data(), 0, z.size() * sizeof(float));
          ggml_backend_tensor_set(vc[i], z.data(), 0, z.size() * sizeof(float));
          ggml_backend_tensor_set(kc_c[i], z.data(), 0, z.size() * sizeof(float));
          ggml_backend_tensor_set(vc_c[i], z.data(), 0, z.size() * sizeof(float));
      } }

    if (VALIDATE()) {
        auto vpec = load_f32(EDIR, "pe_cache", (size_t)4000 * 512);
        float m = 0;
        for (int i = 0; i < 4000 * 512; i++) { float dd = fabsf(G_pe[i] * G_alpha_a - vpec[i]); if (dd > m) m = dd; }
        printf("pe regen vs dump: maxdiff=%.7f\n", m);
    }

    // ---- entry（host） ----
    std::vector<float> emb_t(512 * 732), emb_a(512 * VOCAB), bwh(1024 * 512), bbh(512);
    { std::vector<float> r0(512 * 732), r1(512 * VOCAB), r2(1024 * 512);
      ggml_backend_tensor_get(G_text_emb, r0.data(), 0, r0.size() * sizeof(float));
      ggml_backend_tensor_get(G_audio_emb, r1.data(), 0, r1.size() * sizeof(float));
      ggml_backend_tensor_get(G_bert_w, r2.data(), 0, r2.size() * sizeof(float));
      ggml_backend_tensor_get(G_bert_b, bbh.data(), 0, bbh.size() * sizeof(float));
      auto * h0 = r0.data(), * h1 = r1.data(), * h2 = r2.data();
      for (int v = 0; v < 732; v++)
          for (int d = 0; d < 512; d++) emb_t[(size_t)v * 512 + d] = h0[(size_t)d * 732 + v];
      for (int v = 0; v < VOCAB; v++)
          for (int d = 0; d < 512; d++) emb_a[(size_t)v * 512 + d] = h1[(size_t)d * VOCAB + v];
      for (size_t i = 0; i < bwh.size(); i++) bwh[i] = h2[i]; }
    std::vector<float> xy((size_t)L * 512);
    for (int xi = 0; xi < X; xi++) {
        int tid = (int)vx[xi];
        for (int d = 0; d < 512; d++) {
            float acc = bbh[d];
            for (int i = 0; i < 1024; i++) acc += vbert[(size_t)xi * 1024 + i] * bwh[(size_t)i * 512 + d];
            xy[(size_t)xi * 512 + d] = emb_t[(size_t)tid * 512 + d] + acc + G_alpha_t * G_pe[(size_t)xi * 512 + d];
        }
    }
    for (int yi = 0; yi < Y; yi++) {
        int tid = (int)vy[yi];
        for (int d = 0; d < 512; d++)
            xy[(size_t)(X + yi) * 512 + d] = emb_a[(size_t)tid * 512 + d] + G_alpha_a * G_pe[(size_t)yi * 512 + d];
    }
    if (VALIDATE()) {
        auto vxy = load_f32(IDIR(), "xy_ref", (size_t)L * 512);
        float m = 0;
        for (size_t i = 0; i < xy.size(); i++) { float dd = fabsf(xy[i] - vxy[i]); if (dd > m) m = dd; }
        printf("entry xy maxdiff=%.7f\n", m);
    }
    // prompt bias [L,L,16]（全头同 pattern；公式已验证 bad=0，prod 直接生成）
    std::vector<uint8_t> vpm((size_t)H * L * L);
    for (int h = 0; h < H; h++)
        for (int r = 0; r < L; r++)
            for (int c = 0; c < L; c++)
                vpm[(size_t)h * L * L + r * L + c] =
                    (r < X) ? (c < X ? 1 : 0) : ((c < X || c - X <= r - X) ? 1 : 0);
    if (VALIDATE()) {
        auto vpmf = load_u8(IDIR(), "prompt_mask", (size_t)H * L * L);
        int bad = 0;
        for (size_t i = 0; i < vpm.size(); i++) if (vpm[i] != vpmf[i]) bad++;
        printf("prompt mask file-vs-formula bad=%d\n", bad);
    }
    // scores dst[m=key_t,n=query_t,h]；torch mask[query,key] -> bias[m,n] = mask[n,m]
    std::vector<float> pb((size_t)L * L * H);
    for (int h = 0; h < H; h++)
        for (int m = 0; m < L; m++)
            for (int n = 0; n < L; n++)
                // pbias 张量布局 [L,L,H] F 语义：flat = m + L*n + L*L*h；vpm (h,q,k)：取 q=n,k=m
                pb[((size_t)m + (size_t)L * n) + (size_t)L * L * h] = vpm[((size_t)h * L + n) * L + m] ? 0.0f : -1e30f;
    ggml_backend_tensor_set(xpf_c, xy.data(), 0, xy.size() * sizeof(float));
    ggml_backend_tensor_set(pbias_c, pb.data(), 0, pb.size() * sizeof(float));

    // ================= PREFILL =================
    ggml_tensor * cur = xpf_c;
    std::vector<ggml_tensor *> pcpys;
    for (int i = 0; i < NL; i++) {
        ggml_tensor * qkv = ggml_add(ctxj, ggml_mul_mat(ctxj, G_WC.ly[i].qw, cur), G_WC.ly[i].qb);
        const size_t qts = ggml_type_size(qkv->type);
        // q/k/v [32,L,16]：element (d,l,h) at l*1536*qts + h*32*qts + d*qts
        ggml_tensor * q = ggml_view_3d(ctxj, qkv, HD, L, H, (size_t)1536 * qts, (size_t)HD * qts, 0);
        ggml_tensor * k = ggml_view_3d(ctxj, qkv, HD, L, H, (size_t)1536 * qts, (size_t)HD * qts, (size_t)DIM * qts);
        ggml_tensor * v = ggml_view_3d(ctxj, qkv, HD, L, H, (size_t)1536 * qts, (size_t)HD * qts, (size_t)2 * DIM * qts);
        // 写 cache [0,L)：整块 cpy（strides 兼容：src nb=[qts,1536qts,32qts]，dst [2,64,32768]）
        ggml_tensor * dstk = ggml_view_3d(ctxj, kc_c[i], HD, L, H, (size_t)HD * sizeof(float),
                                          (size_t)HD * K * sizeof(float), 0);
        ggml_tensor * dstv = ggml_view_3d(ctxj, vc_c[i], HD, L, H, (size_t)HD * sizeof(float),
                                          (size_t)HD * K * sizeof(float), 0);
        // k/v 切分是跨步视图（nb1=1536），先 cont 再 cpy（跨步源 cpy 不可信）
        pcpys.push_back(ggml_cpy(ctxj, ggml_cont(ctxj, k), dstk));
        pcpys.push_back(ggml_cpy(ctxj, ggml_cont(ctxj, v), dstv));
        // SYCL 跨步 batch mul 慢 35x：先 cont 再乘（copy 开销可忽略）
        ggml_tensor * sc = ggml_scale(ctxj, ggml_mul_mat(ctxj, ggml_cont(ctxj, k), ggml_cont(ctxj, q)), scale);
        ggml_tensor * p = ggml_soft_max(ctxj, ggml_add(ctxj, sc, pbias_c));
        // O[d,l,h] = sum_t V[d,t,h] P[t,l,h]：Vt=[L,32,16]
        ggml_tensor * vT = ggml_cont(ctxj, ggml_transpose(ctxj, v));
        ggml_tensor * ob = ggml_mul_mat(ctxj, vT, p);
        ggml_tensor * o = ggml_reshape_2d(ctxj, ggml_cont(ctxj, ggml_permute(ctxj, ob, 0, 2, 1, 3)), DIM, L);
        ggml_tensor * proj = ggml_add(ctxj, ggml_mul_mat(ctxj, G_WC.ly[i].ow, o), G_WC.ly[i].ob);
        ggml_tensor * r1 = ggml_add(ctxj, cur, proj);
        ggml_tensor * n1 = ggml_add(ctxj, ggml_mul(ctxj, ggml_norm(ctxj, r1, 1e-5f), G_WC.ly[i].n1w), G_WC.ly[i].n1b);
        ggml_tensor * h1 = ggml_add(ctxj, ggml_mul_mat(ctxj, G_WC.ly[i].m1w, n1), G_WC.ly[i].m1b);
        ggml_tensor * h3 = ggml_add(ctxj, ggml_mul_mat(ctxj, G_WC.ly[i].m2w, ggml_relu(ctxj, h1)), G_WC.ly[i].m2b);
        cur = ggml_add(ctxj, ggml_mul(ctxj, ggml_norm(ctxj, ggml_add(ctxj, n1, h3), 1e-5f), G_WC.ly[i].n2w), G_WC.ly[i].n2b);
    }
    // predict last 列
    ggml_tensor * lastcol = ggml_view_2d(ctxj, cur, DIM, 1, DIM * sizeof(float), (size_t)(L - 1) * DIM * sizeof(float));
    ggml_tensor * plogits = ggml_mul_mat(ctxj, G_WC.predict_w, lastcol);
    ggml_cgraph * gfp = ggml_new_graph_custom(ctxj, 32768, false);
    for (auto c : pcpys) ggml_build_forward_expand(gfp, c);
    ggml_build_forward_expand(gfp, plogits);
    printf("prefill graph n_nodes=%d\n", ggml_graph_n_nodes(gfp));
    ggml_gallocr_alloc_graph(G_galloc_cpu, gfp);
    auto t0 = tick();
    ggml_backend_graph_compute(G_backend_cpu, gfp);
    ggml_backend_synchronize(G_backend);
    printf("prefill compute %.0fms\n", ms(t0, tick()));
    std::vector<float> gotp(VOCAB);
    ggml_backend_tensor_get(plogits, gotp.data(), 0, gotp.size() * sizeof(float));
    // cache 回 SYCL（decode 用）
    for (int i = 0; i < NL; i++) {
        std::vector<float> kcb((size_t)HD * K * H);
        std::vector<float> vcb((size_t)HD * K * H);
        ggml_backend_tensor_get(kc_c[i], kcb.data(), 0, kcb.size() * sizeof(float));
        ggml_backend_tensor_get(vc_c[i], vcb.data(), 0, vcb.size() * sizeof(float));
        ggml_backend_tensor_set(kc[i], kcb.data(), 0, kcb.size() * sizeof(float));
        ggml_backend_tensor_set(vc[i], vcb.data(), 0, vcb.size() * sizeof(float));
    }
    if (VALIDATE()) {
        auto vpr = load_f32(IDIR(), "torch_pflogits", VOCAB);
        float m = 0;
        for (int j = 0; j < VOCAB; j++) { float dd = fabsf(gotp[j] - vpr[j]); if (dd > m) m = dd; }
        printf("prefill logits maxdiff=%.6f\n", m);
    }

    // prefill cache 校验（vs closed dump post-prefill）
    if (VALIDATE()) {
        auto k0 = load_f32(CDIR, "kc0", (size_t)HD * 512 * H);  // 文件 512 宽，只对前半
        std::vector<float> raw((size_t)HD * K * H);
        ggml_backend_tensor_get(kc[0], raw.data(), 0, raw.size() * sizeof(float));
        auto * h = raw.data();
        float m = 0;
        for (int hh = 0; hh < H; hh++)
            for (int t = 0; t < 512; t++)
                for (int d = 0; d < HD; d++) {
                    float a = h[d + HD * t + HD * K * hh];
                    float b = k0[(size_t)hh * K * HD + t * HD + d];
                    float dd = fabsf(a - b);
                    if (dd > m) m = dd;
                }
        printf("prefill kc0 maxdiff=%.6f\n", m);
    }

    // first sample（V=1024，恒 suppress）
    std::vector<int> prev;
    for (auto t : vy) prev.push_back((int)t);
    SampleOpt so;
    so.top_k = getenv("RUNNER_TOP_K") ? atoi(getenv("RUNNER_TOP_K")) : 1;
    so.top_p = 1.0f; so.temp = 1.0f;
    so.rep = getenv("RUNNER_REP") ? (float)atof(getenv("RUNNER_REP")) : 1.35f;
    so.nrn = getenv("RUNNER_NOREPEAT") ? atoi(getenv("RUNNER_NOREPEAT")) : 0;
    so.nrn_plen = (int)vy.size();
    printf("sampler: top_k=%d rep=%.2f norepeat=%d\n", so.top_k, so.rep, so.nrn);
    // teacher-forcing：RUNNER_FORCE_FILE 指定 token 序列（含t0），逐 yo步喂入不采样
    std::vector<int> forced;
    if (getenv("RUNNER_FORCE_FILE")) {
        FILE * ff = fopen(getenv("RUNNER_FORCE_FILE"), "rb");
        if (ff) {
            fseek(ff, 0, SEEK_END); size_t n = ftell(ff) / 8; fseek(ff, 0, SEEK_SET);
            std::vector<int64_t> tmp(n);
            if (fread(tmp.data(), 8, n, ff) == n)
                for (auto v : tmp) forced.push_back((int)v);
            fclose(ff);
        }
        printf("teacher-forcing: %d toks\n", (int)forced.size());
    }
    std::mt19937 rng(getenv("RUNNER_SEED") ? atoi(getenv("RUNNER_SEED")) : 1234);

    std::vector<float> lg0(gotp.begin(), gotp.begin() + 1024);
    for (int j : SUPP) if (j < 1024) lg0[j] = -INFINITY;
    auto pr0 = logits_to_probs(lg0.data(), 1024, prev.data(), (int)prev.size(), so);
    if (getenv("DUMP_PROBS_DIR")) {
        static FILE * pf0 = nullptr;
        if (!pf0) { char pp[600]; snprintf(pp, 600, "%s/probs0.bin", getenv("DUMP_PROBS_DIR")); pf0 = fopen(pp, "wb"); }
        if (pf0) { fwrite(pr0.data(), 4, 1024, pf0); fclose(pf0); pf0 = (FILE *)1; }
    }
    int t_cur = multinomial_exp(pr0, rng);
    if (!forced.empty()) t_cur = forced[0];
    bool stream = getenv("RUNNER_STREAM") && atoi(getenv("RUNNER_STREAM"));
    if (stream) printf("TOK %d\n", t_cur);
    int NSTEPS_ALL = 0;
    std::vector<int64_t> tref0, tnext;
    if (VALIDATE()) {
        NSTEPS_ALL = (int)load_i64(IDIR(), "nsteps", 1)[0];
        tref0 = load_i64(IDIR(), "tokens", NSTEPS_ALL);
        printf("t0=%d ref=%lld %s\n", t_cur, (long long)tref0[0], t_cur == tref0[0] ? "OK" : "MISMATCH");
    } else {
        printf("t0=%d (prod mode)\n", t_cur);
    }

    // ================= DECODE LOOP =================
    if (VALIDATE()) tnext = load_i64(IDIR(), "tokens_next", NSTEPS_ALL);
    int ln = L, nsteps = 0, NSTEPS = VALIDATE() ? NSTEPS_ALL : (K - ln - 1);
    if (ln + NSTEPS > K) NSTEPS = K - ln - 1;
    if (getenv("RUNNER_MAX_STEPS")) { int m = atoi(getenv("RUNNER_MAX_STEPS")); if (m < NSTEPS) NSTEPS = m; }
    printf("NSTEPS=%d\n", NSTEPS);
    bool allok = VALIDATE() ? (t_cur == tref0[0]) : true;
    prev.push_back(t_cur);
    for (int s = 0; s < NSTEPS; s++) {
        // entry：xy = emb[t_cur]*1 + alpha_a*pe[ln - X]
        int pi = ln - X;
        { std::vector<float> xy1(DIM);
          for (int d = 0; d < DIM; d++) xy1[d] = emb_a[(size_t)t_cur * 512 + d] + G_alpha_a * G_pe[(size_t)pi * 512 + d];
          ggml_backend_tensor_set(x1, xy1.data(), 0, xy1.size() * sizeof(float));
          std::vector<float> b3((size_t)K * H);
          for (int h = 0; h < H; h++)
              for (int t = 0; t < K; t++) b3[h * K + t] = (t <= ln) ? 0.0f : -1e30f;
          ggml_backend_tensor_set(bias3, b3.data(), 0, b3.size() * sizeof(float)); }
        // 步级 ctx：decode 图按步建删，避免 645 步 tensor 元数据堆满 job ctx
        struct ggml_init_params sparams = { 64ULL * 1024 * 1024, nullptr, true };
        ggml_context * ctxs = ggml_init(sparams);
        ggml_tensor * c2 = x1;
        std::vector<ggml_tensor *> cpys;
        cpys.reserve(2 * NL * H);
        const size_t cts = sizeof(float);
        for (int i = 0; i < NL; i++) {
            ggml_tensor * qkv = ggml_add(ctxs, ggml_mul_mat(ctxs, G_WS.ly[i].qw, c2), G_WS.ly[i].qb);
            const size_t qts = ggml_type_size(qkv->type);
            ggml_tensor * qb2 = ggml_reshape_3d(ctxs, ggml_view_1d(ctxs, qkv, DIM, 0), HD, 1, H);
            ggml_tensor * k = ggml_reshape_3d(ctxs, ggml_view_1d(ctxs, qkv, DIM, DIM * qts), HD, H, 1);
            ggml_tensor * vv = ggml_reshape_3d(ctxs, ggml_view_1d(ctxs, qkv, DIM, 2 * DIM * qts), HD, H, 1);
            for (int hw = 0; hw < H; hw++) {
                ggml_tensor * khw = ggml_view_2d(ctxs, k, HD, 1, HD * qts, (size_t)hw * HD * qts);
                ggml_tensor * dstk = ggml_view_2d(ctxs, kc[i], HD, 1, HD * cts,
                                                 ((size_t)hw * HD * K + (size_t)ln * HD) * cts);
                cpys.push_back(ggml_cpy(ctxs, khw, dstk));
                ggml_tensor * vhw = ggml_view_2d(ctxs, vv, HD, 1, HD * qts, (size_t)hw * HD * qts);
                ggml_tensor * dstv = ggml_view_2d(ctxs, vc[i], HD, 1, HD * cts,
                                                 ((size_t)hw * HD * K + (size_t)ln * HD) * cts);
                cpys.push_back(ggml_cpy(ctxs, vhw, dstv));
            }
            ggml_tensor * scb = ggml_scale(ctxs, ggml_mul_mat(ctxs, kc[i], qb2), scale);
            ggml_tensor * pbb = ggml_soft_max(ctxs, ggml_add(ctxs, scb, bias3));
            ggml_tensor * vtb = ggml_cont(ctxs, ggml_transpose(ctxs, vc[i]));
            ggml_tensor * ob = ggml_mul_mat(ctxs, vtb, pbb);
            ggml_tensor * o = ggml_reshape_2d(ctxs, ob, DIM, 1);
            ggml_tensor * proj = ggml_add(ctxs, ggml_mul_mat(ctxs, G_WS.ly[i].ow, o), G_WS.ly[i].ob);
            ggml_tensor * r1 = ggml_add(ctxs, c2, proj);
            ggml_tensor * n1 = ggml_add(ctxs, ggml_mul(ctxs, ggml_norm(ctxs, r1, 1e-5f), G_WS.ly[i].n1w), G_WS.ly[i].n1b);
            ggml_tensor * h1 = ggml_add(ctxs, ggml_mul_mat(ctxs, G_WS.ly[i].m1w, n1), G_WS.ly[i].m1b);
            ggml_tensor * h3 = ggml_add(ctxs, ggml_mul_mat(ctxs, G_WS.ly[i].m2w, ggml_relu(ctxs, h1)), G_WS.ly[i].m2b);
            c2 = ggml_add(ctxs, ggml_mul(ctxs, ggml_norm(ctxs, ggml_add(ctxs, n1, h3), 1e-5f), G_WS.ly[i].n2w), G_WS.ly[i].n2b);
        }
        ggml_tensor * logits = ggml_mul_mat(ctxs, G_predict_w, c2);
        ggml_cgraph * gf = ggml_new_graph_custom(ctxs, 16384, false);
        for (auto c : cpys) ggml_build_forward_expand(gf, c);
        ggml_build_forward_expand(gf, logits);
        ggml_gallocr_alloc_graph(G_galloc, gf);
        auto ts0 = tick();
        ggml_backend_graph_compute(G_backend, gf);
        ggml_backend_synchronize(G_backend);
        double dms = ms(ts0, tick());
        std::vector<float> got(VOCAB);
        ggml_backend_tensor_get(logits, got.data(), 0, got.size() * sizeof(float));
        ggml_free(ctxs);
        bool sup = (s + 1) < 10;
        if (sup) for (int j : SUPP) got[j] = -INFINITY;
        auto pr = logits_to_probs(got.data(), VOCAB, prev.data(), (int)prev.size(), so);
        bool is_forced = (int)forced.size() > s + 1;
        if (getenv("DUMP_PROBS_DIR")) {
            static FILE * pf = nullptr;
            if (!pf) { char pp[600]; snprintf(pp, 600, "%s/probs.bin", getenv("DUMP_PROBS_DIR")); pf = fopen(pp, "wb"); }
            if (pf) fwrite(pr.data(), 4, VOCAB, pf);
        }
        int tout = multinomial_exp(pr, rng);
        if (is_forced) tout = forced[s + 1];
        if (stream) printf("TOK %d\n", tout);
        if (VALIDATE()) {
            int expect = (int)tnext[s];
            if (s < 3 || s % 50 == 0 || tout != expect)
                printf("  step %d tok=%d expect=%d %s time=%.1fms\n", s, tout, expect, tout == expect ? "OK" : "MISMATCH", dms);
            if (tout != expect) { allok = false; break; }
        } else if (s < 3 || s % 50 == 0) {
            printf("  step %d tok=%d time=%.1fms\n", s, tout, dms);
        }
        prev.push_back(tout);
        t_cur = tout;
        ln++;
        nsteps++;
        if ((s + 1) % 5 == 0 && tout == EOS_ID) break;
    }
    printf(allok ? "RUNNER ALL OK steps=%d\n" : "RUNNER DIVERGED\n", nsteps);
    int ntoks = 0;
    {
        const char * op = getenv("RUNNER_TOKENS_OUT");
        if (op) {
            FILE * f = fopen(op, "wb");
            if (f) {
                // prev = y + sampled：只存 sampled 部分
                size_t ylen = vy.size();
                std::vector<int64_t> toks;
                for (size_t i = ylen; i < prev.size(); i++) toks.push_back(prev[i]);
                fwrite(toks.data(), 8, toks.size(), f); fclose(f);
                ntoks = (int)toks.size();
                printf("tokens saved: %d to %s\n", ntoks, op);
            } else {
                printf("tokens save failed: %s\n", op);
            }
        } else {
            ntoks = (int)prev.size() - (int)vy.size();
        }
    }
    if (ntoks_out) *ntoks_out = ntoks;
    if (bufj) ggml_backend_buffer_free(bufj);
    if (bufjc) ggml_backend_buffer_free(bufjc);
    ggml_free(ctxj);
    return allok ? 0 : 1;
}

int main(int argc, char ** argv) {
    setbuf(stdout, nullptr);
    bool serve = argc > 1 && strcmp(argv[1], "--serve") == 0;
    init_once();
    if (!serve) {
        int nt = 0;
        return run_job(&nt);
    }
    // 常驻服务：stdin 行协议 JOB <in_dir> <tokens_out> <top_k> <seed>；QUIT 退出
    setenv("RUNNER_VALIDATE", "0", 1);
    printf("SERVE-READY\n");
    char line[1024];
    while (fgets(line, sizeof(line), stdin)) {
        if (strncmp(line, "QUIT", 4) == 0) break;
        char indir[512] = {0}, out[512] = {0};
        int top_k = 1, seed = 1234;
        float rep = 1.35f;
        if (sscanf(line, "JOB %511s %511s %d %d %f", indir, out, &top_k, &seed, &rep) < 4) {
            printf("JOB-ERR bad-line\n");
            continue;
        }
        setenv("RUNNER_IN_DIR", indir, 1);
        setenv("RUNNER_TOKENS_OUT", out, 1);
        char tk[16], sd[16];
        snprintf(tk, 16, "%d", top_k); setenv("RUNNER_TOP_K", tk, 1);
        snprintf(sd, 16, "%d", seed); setenv("RUNNER_SEED", sd, 1);
        char rp[16];
        snprintf(rp, 16, "%.3f", rep); setenv("RUNNER_REP", rp, 1);
        auto tick = []() { return std::chrono::steady_clock::now(); };
        auto t0 = tick();
        int nt = 0;
        int rc = run_job(&nt);
        double dms = std::chrono::duration<double, std::milli>(tick() - t0).count();
        printf("JOB-DONE ntoks=%d rc=%d ms=%.0f\n", nt, rc, dms);
    }
    return 0;
}
