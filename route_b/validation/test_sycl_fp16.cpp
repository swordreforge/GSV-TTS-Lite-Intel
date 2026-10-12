// 路线 B：fp16 精度验证。权重+cache 存 F16，5 步 teacher-forcing vs fp32 torch ref。
// 复用 test_loop_fix 结构，T() 换成 F16 版。
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-sycl.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

static const int H = 16, HD = 32, DIM = 512, NL = 24, VOCAB = 1025;
static const char * DIR = "/tmp/opencode/t2s_loop/";
static const char * WDIR = "/tmp/opencode/t2s_full/";
static const int K = 512, NSTEPS = 5;

static std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> g_raw;  // 保活 F16 数据

static std::vector<float> load_bin_dir(const std::string & dir, const std::string & name, size_t n) {
    FILE * f = fopen((dir + name + ".bin").c_str(), "rb");
    if (!f) { printf("missing %s%s\n", dir.c_str(), name.c_str()); exit(1); }
    std::vector<float> v(n);
    if (fread(v.data(), sizeof(float), n, f) != n) { printf("short %s\n", name.c_str()); exit(1); }
    fclose(f);
    return v;
}

static std::vector<std::pair<ggml_tensor *, std::vector<float>>> g_f32;  // F32 bias 保活
static ggml_tensor * T32(ggml_context * ctx, const std::string & dir, const std::string & name,
                          int64_t n0, int64_t n1 = 1, int64_t n2 = 1, int64_t n3 = 1) {
    size_t n = (size_t)n0 * n1 * n2 * n3;
    auto v = load_bin_dir(dir, name, n);
    ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n0, n1, n2, n3);
    g_f32.emplace_back(t, std::move(v));
    return t;
}

// F16 tensor：fp32 文件 -> fp16 存储
static ggml_tensor * T16(ggml_context * ctx, const std::string & dir, const std::string & name,
                         int64_t n0, int64_t n1 = 1, int64_t n2 = 1, int64_t n3 = 1) {
    size_t n = (size_t)n0 * n1 * n2 * n3;
    auto v = load_bin_dir(dir, name, n);
    std::vector<uint8_t> raw(n * sizeof(ggml_fp16_t));
    auto * h = (ggml_fp16_t *)raw.data();
    for (size_t i = 0; i < n; i++) h[i] = ggml_fp32_to_fp16(v[i]);
    ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n0, n1, n2, n3);
    g_raw.emplace_back(t, std::move(raw));
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
        kc[i] = T16(ctx, DIR, n1, HD, K, H);
        vc[i] = T16(ctx, DIR, n2, HD, K, H);
    }
    int ln0 = 0;
    { FILE * f = fopen((std::string(DIR) + "ln0.bin").c_str(), "rb");
      int64_t lv = 0;
      if (!f || fread(&lv, 8, 1, f) != 1) { printf("ln0 bad\n"); return 1; }
      fclose(f); ln0 = (int)lv; }

    struct Lyr { ggml_tensor *qw, *qb, *ow, *ob, *n1w, *n1b, *m1w, *m1b, *m2w, *m2b, *n2w, *n2b; };
    std::vector<Lyr> ly(NL);
    for (int i = 0; i < NL; i++) {
        char n[32];
        snprintf(n, 32, "b%d.qkv_w", i); ly[i].qw = T16(ctx, WDIR, n, DIM, 3 * DIM);
        snprintf(n, 32, "b%d.qkv_b", i); ly[i].qb = T32(ctx, WDIR, n, 3 * DIM);
        snprintf(n, 32, "b%d.out_w", i); ly[i].ow = T16(ctx, WDIR, n, DIM, DIM);
        snprintf(n, 32, "b%d.out_b", i); ly[i].ob = T32(ctx, WDIR, n, DIM);
        snprintf(n, 32, "b%d.n1_w", i); ly[i].n1w = T32(ctx, WDIR, n, DIM);
        snprintf(n, 32, "b%d.n1_b", i); ly[i].n1b = T32(ctx, WDIR, n, DIM);
        snprintf(n, 32, "b%d.mlp1_w", i); ly[i].m1w = T16(ctx, WDIR, n, DIM, 4 * DIM);
        snprintf(n, 32, "b%d.mlp1_b", i); ly[i].m1b = T32(ctx, WDIR, n, 4 * DIM);
        snprintf(n, 32, "b%d.mlp2_w", i); ly[i].m2w = T16(ctx, WDIR, n, 4 * DIM, DIM);
        snprintf(n, 32, "b%d.mlp2_b", i); ly[i].m2b = T32(ctx, WDIR, n, DIM);
        snprintf(n, 32, "b%d.n2_w", i); ly[i].n2w = T32(ctx, WDIR, n, DIM);
        snprintf(n, 32, "b%d.n2_b", i); ly[i].n2b = T32(ctx, WDIR, n, DIM);
    }
    ggml_tensor * pw = T16(ctx, WDIR, "predict_w", DIM, VOCAB);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DIM, 1);
    std::vector<ggml_tensor *> biases;
    for (int i = 0; i < NL * H; i++)
        biases.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1));

    const float scale = 1.0f / sqrtf((float)HD);

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    for (auto & kv : g_raw)
        ggml_backend_tensor_set(kv.first, kv.second.data(), 0, kv.second.size());
    for (auto & kv : g_f32)
        ggml_backend_tensor_set(kv.first, kv.second.data(), 0, kv.second.size() * sizeof(float));
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
        const size_t cts = ggml_type_size(kc[0]->type);  // cache F16 -> 2
        for (int i = 0; i < NL; i++) {
            ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, ly[i].qw, cur), ly[i].qb);
            const size_t qts = ggml_type_size(qkv->type);
            ggml_tensor * q = ggml_reshape_3d(ctx, ggml_view_1d(ctx, qkv, DIM, 0), HD, H, 1);
            ggml_tensor * k = ggml_reshape_3d(ctx, ggml_view_1d(ctx, qkv, DIM, DIM * qts), HD, H, 1);
            ggml_tensor * v = ggml_reshape_3d(ctx, ggml_view_1d(ctx, qkv, DIM, 2 * DIM * qts), HD, H, 1);
            for (int hw = 0; hw < H; hw++) {
                ggml_tensor * khw = ggml_view_2d(ctx, k, HD, 1, HD * qts, (size_t)hw * HD * qts);
                ggml_tensor * dstk = ggml_view_2d(ctx, kc[i], HD, 1, HD * cts,
                                                 ((size_t)hw * HD * K + (size_t)ln * HD) * cts);
                cpys.push_back(ggml_cpy(ctx, khw, dstk));
                ggml_tensor * vhw = ggml_view_2d(ctx, v, HD, 1, HD * qts, (size_t)hw * HD * qts);
                ggml_tensor * dstv = ggml_view_2d(ctx, vc[i], HD, 1, HD * cts,
                                                 ((size_t)hw * HD * K + (size_t)ln * HD) * cts);
                cpys.push_back(ggml_cpy(ctx, vhw, dstv));
            }
            std::vector<ggml_tensor *> heads;
            for (int h = 0; h < H; h++) {
                ggml_tensor * qh = ggml_view_2d(ctx, q, HD, 1, HD * qts, (size_t)h * HD * qts);
                ggml_tensor * kh = ggml_view_2d(ctx, kc[i], HD, K, HD * cts, (size_t)h * HD * K * cts);
                ggml_tensor * sc = ggml_scale(ctx, ggml_mul_mat(ctx, kh, qh), scale);
                ggml_tensor * p = ggml_soft_max(ctx, ggml_add(ctx, sc, biases[i * H + h]));
                ggml_tensor * vh = ggml_view_2d(ctx, vc[i], HD, K, HD * cts, (size_t)h * HD * K * cts);
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
        ggml_gallocr_alloc_graph(galloc, gf);
        auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::vector<float> got(VOCAB);
        ggml_backend_tensor_get(logits, got.data(), 0, got.size() * sizeof(float));
        float m = 0;
        int arg = 0, argr = 0;
        for (int j = 0; j < VOCAB; j++) {
            float d = fabsf(got[j] - vref[j]); if (d > m) m = d;
            if (got[j] > got[arg]) arg = j;
            if (vref[j] > vref[argr]) argr = j;
        }
        printf("  fp16 step %d maxdiff=%.6f argmax=%d ref_argmax=%d %s time=%.1fms\n", s, m, arg, argr, arg == argr ? "OK" : "FLIP", ms);
        ln++;
    }
    ggml_free(ctx);
    return 0;
}
