// 路线 B 验证：24 层 T2S 单步 decode 全量复刻 vs torch logits。
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

static bool g_synth = false;
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <utility>

static const int H = 16, HD = 32, DIM = 512, NL = 24, VOCAB = 1025;
static const char * DIR = "/tmp/opencode/t2s_full/";
static int K = 512, KV_POS = 0;

static std::vector<std::pair<ggml_tensor *, std::vector<float>>> g_data;

static std::vector<float> load_bin(const std::string & name, size_t n) {
    FILE * f = fopen((std::string(DIR) + name + ".bin").c_str(), "rb");
    if (!f) { printf("missing %s\n", name.c_str()); exit(1); }
    std::vector<float> v(n);
    if (fread(v.data(), sizeof(float), n, f) != n) { printf("short %s\n", name.c_str()); exit(1); }
    fclose(f);
    return v;
}

static ggml_tensor * T(ggml_context * ctx, const std::string & name,
                       std::initializer_list<int64_t> dims, size_t override_n = 0) {
    std::vector<int64_t> ne(dims);
    while (ne.size() < 4) ne.push_back(1);
    size_t n = 1;
    for (auto d : ne) n *= d;
    if (override_n) n = override_n;
    auto v = load_bin(name, n);
    ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
    g_data.emplace_back(t, std::move(v));
    return t;
}

int main() {
    { FILE * f = fopen((std::string(DIR) + "K.txt").c_str(), "r");
      if (f) { fscanf(f, "%d", &K); fclose(f); } }
    { FILE * f = fopen((std::string(DIR) + "ln.bin").c_str(), "rb");
      int64_t lv = 0;
      if (f) { size_t r = fread(&lv, sizeof(int64_t), 1, f); (void)r; fclose(f); }
      KV_POS = (int)lv; }
    printf("K=%d kv_pos=%d\n", K, KV_POS);

    struct ggml_init_params params = { 1024 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * x = T(ctx, "x", {DIM, 1});
    std::vector<ggml_tensor *> kc(NL), vc(NL);
    for (int i = 0; i < NL; i++) {
        char n1[32], n2[32];
        snprintf(n1, 32, "kc%d", i); snprintf(n2, 32, "vc%d", i);
        kc[i] = T(ctx, n1, {HD, K, H});
        vc[i] = T(ctx, n2, {HD, K, H});
    }
    // mask [1,H,1,K] uint8 True=attend
    FILE * fm = fopen((std::string(DIR) + "mask.bin").c_str(), "rb");
    std::vector<uint8_t> vmask(H * K);
    if (fread(vmask.data(), 1, H * K, fm) != (size_t)(H * K)) { printf("mask short\n"); return 1; }
    fclose(fm);

    struct Lyr { ggml_tensor *qw, *qb, *ow, *ob, *n1w, *n1b, *m1w, *m1b, *m2w, *m2b, *n2w, *n2b; };
    std::vector<Lyr> ly(NL);
    for (int i = 0; i < NL; i++) {
        char n[32];
        snprintf(n, 32, "b%d.qkv_w", i); ly[i].qw = T(ctx, n, {DIM, 3 * DIM});
        snprintf(n, 32, "b%d.qkv_b", i); ly[i].qb = T(ctx, n, {3 * DIM});
        snprintf(n, 32, "b%d.out_w", i); ly[i].ow = T(ctx, n, {DIM, DIM});
        snprintf(n, 32, "b%d.out_b", i); ly[i].ob = T(ctx, n, {DIM});
        snprintf(n, 32, "b%d.n1_w", i); ly[i].n1w = T(ctx, n, {DIM});
        snprintf(n, 32, "b%d.n1_b", i); ly[i].n1b = T(ctx, n, {DIM});
        snprintf(n, 32, "b%d.mlp1_w", i); ly[i].m1w = T(ctx, n, {DIM, 4 * DIM});
        snprintf(n, 32, "b%d.mlp1_b", i); ly[i].m1b = T(ctx, n, {4 * DIM});
        snprintf(n, 32, "b%d.mlp2_w", i); ly[i].m2w = T(ctx, n, {4 * DIM, DIM});
        snprintf(n, 32, "b%d.mlp2_b", i); ly[i].m2b = T(ctx, n, {DIM});
        snprintf(n, 32, "b%d.n2_w", i); ly[i].n2w = T(ctx, n, {DIM});
        snprintf(n, 32, "b%d.n2_b", i); ly[i].n2b = T(ctx, n, {DIM});
    }
    ggml_tensor * pw = T(ctx, "predict_w", {DIM, VOCAB});

    const float scale = 1.0f / sqrtf((float)HD);
    ggml_tensor * cur = x;
    std::vector<ggml_tensor *> layer_nodes;
    ggml_tensor * attn_probe = nullptr;
    ggml_tensor * head0_probe = nullptr;
    ggml_tensor * s_probe = nullptr;
    ggml_tensor * qh_probe = nullptr;
    ggml_tensor * p_probe = nullptr;
    for (int i = 0; i < NL; i++) {
        ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, ly[i].qw, cur), ly[i].qb);
        ggml_tensor * q = ggml_reshape_3d(ctx, ggml_view_1d(ctx, qkv, DIM, 0), HD, H, 1);
        ggml_tensor * k = ggml_reshape_3d(ctx, ggml_view_1d(ctx, qkv, DIM, DIM * sizeof(float)), HD, H, 1);
        ggml_tensor * v = ggml_reshape_3d(ctx, ggml_view_1d(ctx, qkv, DIM, 2 * DIM * sizeof(float)), HD, H, 1);
        // cache 布局 [D,T,H]：写 t=KV_POS 需按 head 逐个 cpy
        for (int h = 0; h < H; h++) {
            ggml_tensor * khw = ggml_view_2d(ctx, k, HD, 1, HD * sizeof(float), (size_t)h * HD * sizeof(float));
            ggml_tensor * dstk = ggml_view_2d(ctx, kc[i], HD, 1, HD * sizeof(float),
                                             (size_t)h * HD * K * sizeof(float) + (size_t)KV_POS * HD * sizeof(float));
            ggml_tensor * c1 = ggml_cpy(ctx, khw, dstk);
            ggml_tensor * vhw = ggml_view_2d(ctx, v, HD, 1, HD * sizeof(float), (size_t)h * HD * sizeof(float));
            ggml_tensor * dstv = ggml_view_2d(ctx, vc[i], HD, 1, HD * sizeof(float),
                                             (size_t)h * HD * K * sizeof(float) + (size_t)KV_POS * HD * sizeof(float));
            ggml_tensor * c2 = ggml_cpy(ctx, vhw, dstv);
            (void)c1; (void)c2;
        }
        std::vector<ggml_tensor *> heads;
        for (int h = 0; h < H; h++) {
            ggml_tensor * qh = ggml_view_2d(ctx, q, HD, 1, HD * sizeof(float), (size_t)h * HD * sizeof(float));
            ggml_tensor * kh = ggml_view_2d(ctx, kc[i], HD, K, HD * sizeof(float), (size_t)h * HD * K * sizeof(float));
            ggml_tensor * s = ggml_scale(ctx, ggml_mul_mat(ctx, kh, qh), scale);
            std::vector<float> bias(K);
            for (int t = 0; t < K; t++) bias[t] = vmask[h * K + t] ? 0.0f : -1e30f;
            ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
            g_data.emplace_back(b, std::move(bias));
            ggml_tensor * p = ggml_soft_max(ctx, ggml_add(ctx, s, b));
            ggml_tensor * vh = ggml_view_2d(ctx, vc[i], HD, K, HD * sizeof(float), (size_t)h * HD * K * sizeof(float));
            ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, vh));
            heads.push_back(ggml_mul_mat(ctx, vt, p));
        if (i == 0 && h == 0) head0_probe = heads.back();
        if (i == 0 && h == 0) { s_probe = s; p_probe = p; qh_probe = qh; }
        }
        ggml_tensor * o = heads[0];
        for (int h = 1; h < H; h++) o = ggml_concat(ctx, o, heads[h], 0);
        if (i == 0) attn_probe = o;
        ggml_tensor * proj = ggml_add(ctx, ggml_mul_mat(ctx, ly[i].ow, o), ly[i].ob);
        ggml_tensor * r1 = ggml_add(ctx, cur, proj);
        ggml_tensor * n1 = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, r1, 1e-5f), ly[i].n1w), ly[i].n1b);
        ggml_tensor * h1 = ggml_add(ctx, ggml_mul_mat(ctx, ly[i].m1w, n1), ly[i].m1b);
        ggml_tensor * h3 = ggml_add(ctx, ggml_mul_mat(ctx, ly[i].m2w, ggml_relu(ctx, h1)), ly[i].m2b);
        cur = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, ggml_add(ctx, n1, h3), 1e-5f), ly[i].n2w), ly[i].n2b);
        layer_nodes.push_back(cur);
    }
    ggml_tensor * logits = ggml_mul_mat(ctx, pw, cur);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    for (auto & kv : g_data) ggml_backend_tensor_set(kv.first, kv.second.data(), 0, kv.second.size() * sizeof(float));

    // bq load 验证
    {
        std::vector<float> gl(3 * DIM);
        ggml_backend_tensor_get(ly[0].qb, gl.data(), 0, gl.size() * sizeof(float));
        auto vr = load_bin("b0.qkv_b", 3 * DIM);
        float m = 0;
        for (int j = 0; j < 3 * DIM; j++) { float d = fabsf(gl[j] - vr[j]); if (d > m) m = d; }
        printf("  Bload maxdiff=%.6f\n", m);
    }
    // x load 验证
    {
        std::vector<float> gl(DIM);
        // cur for layer0 IS x; find it via layer input: use qw's counterpart is overkill; read x node:
        ggml_backend_tensor_get(x, gl.data(), 0, gl.size() * sizeof(float));
        auto vr = load_bin("x", DIM);
        float m = 0;
        for (int j = 0; j < DIM; j++) { float d = fabsf(gl[j] - vr[j]); if (d > m) m = d; }
        printf("  Xload maxdiff=%.6f\n", m);
    }
    // 权重 load 验证
    {
        std::vector<float> gl(DIM * 3 * DIM);
        ggml_backend_tensor_get(ly[0].qw, gl.data(), 0, gl.size() * sizeof(float));
        auto vr = load_bin("b0.qkv_w", DIM * 3 * DIM);
        float m = 0;
        for (size_t j = 0; j < gl.size(); j++) { float d = fabsf(gl[j] - vr[j]); if (d > m) m = d; }
        printf("  Wload maxdiff=%.6f\n", m);
    }
    // qh 验证
    {
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(gf, qh_probe);
        ggml_backend_graph_compute(backend, gf);
        auto vr = load_bin("torch_q0", HD);
        std::vector<float> gl(HD);
        ggml_backend_tensor_get(qh_probe, gl.data(), 0, gl.size() * sizeof(float));
        float m = 0;
        for (int j = 0; j < HD; j++) { float d = fabsf(gl[j] - vr[j]); if (d > m) m = d; }
        printf("  qh0 maxdiff=%.6f\n", m);
    }
    // scores/probs 验证
    {
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(gf, s_probe);
        ggml_backend_graph_compute(backend, gf);
        auto vr = load_bin("torch_s0b", K);
        std::vector<float> gl(K);
        ggml_backend_tensor_get(s_probe, gl.data(), 0, gl.size() * sizeof(float));
        float m = 0;
        for (int j = 0; j < K; j++) { float d = fabsf(gl[j] - vr[j]); if (d > m) m = d; }
        printf("  scores0 maxdiff=%.6f\n", m);
        {
            float mpre = 0, m264 = 0, mpost = 0;
            for (int j = 0; j < K; j++) {
                float d = fabsf(gl[j] - vr[j]);
                if (j < 264) { if (d > mpre) mpre = d; }
                else if (j == 264) m264 = d;
                else { if (d > mpost) mpost = d; }
            }
            printf("  scores0 err pre[0,264)=%.4f at264=%.4f post=%.4f\n", mpre, m264, mpost);
        }
    }
    {
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(gf, p_probe);
        ggml_backend_graph_compute(backend, gf);
        auto vr = load_bin("torch_p0b", K);
        std::vector<float> gl(K);
        ggml_backend_tensor_get(p_probe, gl.data(), 0, gl.size() * sizeof(float));
        float m = 0;
        for (int j = 0; j < K; j++) { float d = fabsf(gl[j] - vr[j]); if (d > m) m = d; }
        printf("  probs0 maxdiff=%.6f\n", m);
    }
    // head0 attention-out 验证
    {
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(gf, head0_probe);
        ggml_backend_graph_compute(backend, gf);
        auto vr = load_bin("torch_head0b", HD);
        std::vector<float> gl(HD);
        ggml_backend_tensor_get(head0_probe, gl.data(), 0, gl.size() * sizeof(float));
        float m = 0;
        for (int j = 0; j < HD; j++) { float d = fabsf(gl[j] - vr[j]); if (d > m) m = d; }
        printf("  head0 maxdiff=%.6f\n", m);
    }
    // layer0 attention-out 验证
    {
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(gf, attn_probe);
        ggml_backend_graph_compute(backend, gf);
        auto vr = load_bin("npy_attn0", DIM);
        std::vector<float> gl(DIM);
        ggml_backend_tensor_get(attn_probe, gl.data(), 0, gl.size() * sizeof(float));
        float m = 0;
        for (int j = 0; j < DIM; j++) { float d = fabsf(gl[j] - vr[j]); if (d > m) m = d; }
        printf("  attn0 maxdiff=%.6f\n", m);
        {
            double _s = 0, _a = 0, _mx = 0;
            for (int j = 0; j < DIM; j++) { _s += gl[j]; _a += fabsf(gl[j]); if (fabsf(gl[j]) > _mx) _mx = fabsf(gl[j]); }
            printf("  attn0 C++ stats: mean=%.4f absmean=%.4f maxabs=%.4f\n", _s / DIM, _a / DIM, _mx);
        }
    }
    // 逐层验证：每个前缀单独成图，保证输出节点内存不被复用
    for (int i = 0; i < NL; i++) {
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(gf, layer_nodes[i]);
        ggml_backend_graph_compute(backend, gf);
        char n[32];
        snprintf(n, 32, "ly%d", i);
        auto vr = load_bin(n, DIM);
        std::vector<float> gl(DIM);
        ggml_backend_tensor_get(layer_nodes[i], gl.data(), 0, gl.size() * sizeof(float));
        float m = 0;
        for (int j = 0; j < DIM; j++) { float d = fabsf(gl[j] - vr[j]); if (d > m) m = d; }
        double _s = 0, _a = 0;
        for (int j = 0; j < DIM; j++) { _s += gl[j]; _a += fabsf(gl[j]); }
        printf("  layer %2d maxdiff=%.6f mean=%.4f absmean=%.4f\n", i, m, _s / DIM, _a / DIM);
        if (i == 0) {
            // 找误差最大的 8 个位置，看是否聚在某个头里
            std::vector<std::pair<float,int>> ds;
            for (int j = 0; j < DIM; j++) ds.emplace_back(fabsf(gl[j] - load_bin("ly0", DIM)[j]), j);
            std::sort(ds.begin(), ds.end(), std::greater<std::pair<float,int>>());
            printf("  top-err idx:");
            for (int j = 0; j < 8; j++) printf(" %d(%.2f,h%d)", ds[j].second, ds[j].first, ds[j].second / 32);
            printf("\n");
        }
        if (m > 1e-3) break;
    }
    // 全图 logits
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, logits);
    ggml_backend_graph_compute(backend, gf);

    auto vref = load_bin("ref_logits", VOCAB);
    std::vector<float> got(VOCAB);
    ggml_backend_tensor_get(logits, got.data(), 0, got.size() * sizeof(float));
    float md = 0;
    for (int i = 0; i < VOCAB; i++) { float d = fabsf(got[i] - vref[i]); if (d > md) md = d; }
    printf("GGML-FULLSTEP maxdiff=%.6f\n", md);
    ggml_free(ctx);
    return 0;
}
