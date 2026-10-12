// 路线 B 验证：5 步 teacher 回放（decode+predict），验证 KV 跨步管理。
// 修复版：cpy 节点显式 expand 进图（否则 cpy 是死节点，cache 从不更新）。
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-sycl.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

static const int H = 16, HD = 32, DIM = 512, NL = 24, VOCAB = 1025;
static const char * DIR = "/tmp/opencode/t2s_loop/";
static const char * WDIR = "/tmp/opencode/t2s_full/";
static const int K = 512, NSTEPS = 5;

static std::vector<std::pair<ggml_tensor *, std::vector<float>>> g_data;

static std::vector<float> load_bin_dir(const std::string & dir, const std::string & name, size_t n) {
    FILE * f = fopen((dir + name + ".bin").c_str(), "rb");
    if (!f) { printf("missing %s%s\n", dir.c_str(), name.c_str()); exit(1); }
    std::vector<float> v(n);
    if (fread(v.data(), sizeof(float), n, f) != n) { printf("short %s\n", name.c_str()); exit(1); }
    fclose(f);
    return v;
}

static ggml_tensor * T(ggml_context * ctx, const std::string & dir, const std::string & name,
                       int64_t n0, int64_t n1 = 1, int64_t n2 = 1, int64_t n3 = 1) {
    size_t n = (size_t)n0 * n1 * n2 * n3;
    auto v = load_bin_dir(dir, name, n);
    ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n0, n1, n2, n3);
    g_data.emplace_back(t, std::move(v));
    return t;
}

int main() {
    setbuf(stdout, nullptr);
    struct ggml_init_params params = { 1024ULL * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);

    std::vector<ggml_tensor *> kc(NL), vc(NL);
    for (int i = 0; i < NL; i++) {
        char n1[32], n2[32];
        snprintf(n1, 32, "kc%d", i); snprintf(n2, 32, "vc%d", i);
        kc[i] = T(ctx, DIR, n1, HD, K, H);
        vc[i] = T(ctx, DIR, n2, HD, K, H);
    }
    int ln0 = 0;
    { FILE * f = fopen((std::string(DIR) + "ln0.bin").c_str(), "rb");
      int64_t lv = 0;
      if (!f || fread(&lv, 8, 1, f) != 1) { printf("ln0 bad\n"); return 1; }
      fclose(f); ln0 = (int)lv; }
    printf("ln0=%d\n", ln0);

    struct Lyr { ggml_tensor *qw, *qb, *ow, *ob, *n1w, *n1b, *m1w, *m1b, *m2w, *m2b, *n2w, *n2b; };
    std::vector<Lyr> ly(NL);
    for (int i = 0; i < NL; i++) {
        char n[32];
        snprintf(n, 32, "b%d.qkv_w", i); ly[i].qw = T(ctx, WDIR, n, DIM, 3 * DIM);
        snprintf(n, 32, "b%d.qkv_b", i); ly[i].qb = T(ctx, WDIR, n, 3 * DIM);
        snprintf(n, 32, "b%d.out_w", i); ly[i].ow = T(ctx, WDIR, n, DIM, DIM);
        snprintf(n, 32, "b%d.out_b", i); ly[i].ob = T(ctx, WDIR, n, DIM);
        snprintf(n, 32, "b%d.n1_w", i); ly[i].n1w = T(ctx, WDIR, n, DIM);
        snprintf(n, 32, "b%d.n1_b", i); ly[i].n1b = T(ctx, WDIR, n, DIM);
        snprintf(n, 32, "b%d.mlp1_w", i); ly[i].m1w = T(ctx, WDIR, n, DIM, 4 * DIM);
        snprintf(n, 32, "b%d.mlp1_b", i); ly[i].m1b = T(ctx, WDIR, n, 4 * DIM);
        snprintf(n, 32, "b%d.mlp2_w", i); ly[i].m2w = T(ctx, WDIR, n, 4 * DIM, DIM);
        snprintf(n, 32, "b%d.mlp2_b", i); ly[i].m2b = T(ctx, WDIR, n, DIM);
        snprintf(n, 32, "b%d.n2_w", i); ly[i].n2w = T(ctx, WDIR, n, DIM);
        snprintf(n, 32, "b%d.n2_b", i); ly[i].n2b = T(ctx, WDIR, n, DIM);
    }
    ggml_tensor * pw = T(ctx, WDIR, "predict_w", DIM, VOCAB);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DIM, 1);
    std::vector<ggml_tensor *> biases;
    for (int i = 0; i < NL * H; i++)
        biases.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1));

    const float scale = 1.0f / sqrtf((float)HD);

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    for (auto & kv : g_data) ggml_backend_tensor_set(kv.first, kv.second.data(), 0, kv.second.size() * sizeof(float));
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_sycl_buffer_type(0));

    int ln = ln0;
    for (int s = 0; s < NSTEPS; s++) {
        char n[64];
        snprintf(n, 64, "step%d_x", s);
        auto vx = load_bin_dir(DIR, n, (size_t)DIM);
        snprintf(n, 64, "step%d_mask", s);
        FILE * fm = fopen((std::string(DIR) + n + ".bin").c_str(), "rb");
        std::vector<uint8_t> vmask(H * K);
        if (!fm || fread(vmask.data(), 1, H * K, fm) != (size_t)(H * K)) { printf("mask short\n"); return 1; }
        fclose(fm);
        snprintf(n, 64, "step%d_logits", s);
        auto vref = load_bin_dir(DIR, n, VOCAB);

        ggml_backend_tensor_set(x, vx.data(), 0, vx.size() * sizeof(float));
        for (int i = 0; i < NL * H; i++) {
            int h = i % H;
            std::vector<float> bias(K);
            for (int t = 0; t < K; t++) bias[t] = vmask[h * K + t] ? 0.0f : -1e30f;
            ggml_backend_tensor_set(biases[i], bias.data(), 0, bias.size() * sizeof(float));
        }

        ggml_tensor * cur = x;
        std::vector<ggml_tensor *> cpys;
        cpys.reserve(2 * NL * H);
        for (int i = 0; i < NL; i++) {
            ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, ly[i].qw, cur), ly[i].qb);
            ggml_tensor * q = ggml_reshape_3d(ctx, ggml_view_1d(ctx, qkv, DIM, 0), HD, H, 1);
            ggml_tensor * k = ggml_reshape_3d(ctx, ggml_view_1d(ctx, qkv, DIM, DIM * sizeof(float)), HD, H, 1);
            ggml_tensor * v = ggml_reshape_3d(ctx, ggml_view_1d(ctx, qkv, DIM, 2 * DIM * sizeof(float)), HD, H, 1);
            for (int hw = 0; hw < H; hw++) {
                ggml_tensor * khw = ggml_view_2d(ctx, k, HD, 1, HD * sizeof(float), (size_t)hw * HD * sizeof(float));
                ggml_tensor * dstk = ggml_view_2d(ctx, kc[i], HD, 1, HD * sizeof(float),
                                                 ((size_t)hw * HD * K + (size_t)ln * HD) * sizeof(float));
                cpys.push_back(ggml_cpy(ctx, khw, dstk));
                ggml_tensor * vhw = ggml_view_2d(ctx, v, HD, 1, HD * sizeof(float), (size_t)hw * HD * sizeof(float));
                ggml_tensor * dstv = ggml_view_2d(ctx, vc[i], HD, 1, HD * sizeof(float),
                                                 ((size_t)hw * HD * K + (size_t)ln * HD) * sizeof(float));
                cpys.push_back(ggml_cpy(ctx, vhw, dstv));
            }
            std::vector<ggml_tensor *> heads;
            for (int h = 0; h < H; h++) {
                ggml_tensor * qh = ggml_view_2d(ctx, q, HD, 1, HD * sizeof(float), (size_t)h * HD * sizeof(float));
                ggml_tensor * kh = ggml_view_2d(ctx, kc[i], HD, K, HD * sizeof(float), (size_t)h * HD * K * sizeof(float));
                ggml_tensor * sc = ggml_scale(ctx, ggml_mul_mat(ctx, kh, qh), scale);
                ggml_tensor * p = ggml_soft_max(ctx, ggml_add(ctx, sc, biases[i * H + h]));
                ggml_tensor * vh = ggml_view_2d(ctx, vc[i], HD, K, HD * sizeof(float), (size_t)h * HD * K * sizeof(float));
                ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, vh));
                heads.push_back(ggml_mul_mat(ctx, vt, p));
            }
            ggml_tensor * o = heads[0];
            for (int h = 1; h < H; h++) o = ggml_concat(ctx, o, heads[h], 0);
            ggml_tensor * proj = ggml_add(ctx, ggml_mul_mat(ctx, ly[i].ow, o), ly[i].ob);
            ggml_tensor * r1 = ggml_add(ctx, cur, proj);
            ggml_tensor * n1 = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, r1, 1e-5f), ly[i].n1w), ly[i].n1b);
            ggml_tensor * h1 = ggml_add(ctx, ggml_mul_mat(ctx, ly[i].m1w, n1), ly[i].m1b);
            ggml_tensor * h3 = ggml_add(ctx, ggml_mul_mat(ctx, ly[i].m2w, ggml_relu(ctx, h1)), ly[i].m2b);
            cur = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, ggml_add(ctx, n1, h3), 1e-5f), ly[i].n2w), ly[i].n2b);
        }
        ggml_tensor * logits = ggml_mul_mat(ctx, pw, cur);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);
        for (auto c : cpys) ggml_build_forward_expand(gf, c);
        ggml_build_forward_expand(gf, logits);
        if (s == 0) {
            int nn = ggml_graph_n_nodes(gf), ncpy = 0, idx_cpy = -1, idx_mul = -1;
            for (int i = 0; i < nn; i++) {
                auto * nd = ggml_graph_node(gf, i);
                if (nd->op == GGML_OP_CPY) { ncpy++; if (idx_cpy < 0) idx_cpy = i; }
                if (nd->op == GGML_OP_MUL_MAT && idx_mul < 0) idx_mul = i;
            }
            printf("graph n_nodes=%d ncpy=%d first_cpy=%d first_mul=%d\n", nn, ncpy, idx_cpy, idx_mul);
        }
        ggml_gallocr_alloc_graph(galloc, gf);
        auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::vector<float> got(VOCAB);
        ggml_backend_tensor_get(logits, got.data(), 0, got.size() * sizeof(float));
        float m = 0;
        for (int j = 0; j < VOCAB; j++) { float d = fabsf(got[j] - vref[j]); if (d > m) m = d; }
        printf("  loop step %d maxdiff=%.6f time=%.1fms\n", s, m, ms);
        ln++;
    }
    ggml_free(ctx);
    return 0;
}
