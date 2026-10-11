// 路线 B 验证：单个 T2SBlock.decode 步的 ggml 复刻 vs torch 参照。
// 布局：全部 fp32；权重已是 ggml [in,out]；mask True=attend（管线语义）。
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

static const int B = 1, H = 16, K = 64, HD = 32, DIM = 512, L = 1;
static const int KV_POS = 10;
static const char * DIR = "/tmp/opencode/t2s_blk0/";

static std::vector<float> load_bin(const std::string & name, size_t n) {
    std::string p = std::string(DIR) + name + ".bin";
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) { printf("missing %s\n", p.c_str()); exit(1); }
    std::vector<float> v(n);
    size_t r = fread(v.data(), sizeof(float), n, f);
    fclose(f);
    if (r != n) { printf("short read %s: %zu != %zu\n", p.c_str(), r, n); exit(1); }
    return v;
}

static std::vector<uint8_t> load_mask() {
    FILE * f = fopen((std::string(DIR) + "mask.bin").c_str(), "rb");
    std::vector<uint8_t> v(H * K);
    size_t r = fread(v.data(), 1, H * K, f);
    fclose(f);
    if (r != (size_t)(H * K)) { printf("mask short\n"); exit(1); }
    return v;
}

static std::vector<std::pair<ggml_tensor *, const std::vector<float> *>> g_pending;

static ggml_tensor * new_from_vec(ggml_context * ctx, const std::vector<float> & v,
                                  int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1) {
    ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    g_pending.emplace_back(t, &v);
    return t;
}

int main() {
    auto vx = load_bin("x", DIM * L);
    auto vkc = load_bin("kc", HD * H * K);
    auto vvc = load_bin("vc", HD * H * K);
    auto vmask = load_mask();
    auto vout = load_bin("out", DIM * L);

    struct ggml_init_params params = { 256 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * x = new_from_vec(ctx, vx, DIM, L);
    ggml_tensor * kc = new_from_vec(ctx, vkc, HD, H, K);
    ggml_tensor * vc = new_from_vec(ctx, vvc, HD, H, K);
    auto Wq = new_from_vec(ctx, load_bin("qkv_w", DIM * 3 * DIM), DIM, 3 * DIM);
    auto bq = new_from_vec(ctx, load_bin("qkv_b", 3 * DIM), 3 * DIM);
    auto Wo = new_from_vec(ctx, load_bin("out_w", DIM * DIM), DIM, DIM);
    auto bo = new_from_vec(ctx, load_bin("out_b", DIM), DIM);
    auto n1w = new_from_vec(ctx, load_bin("n1_w", DIM), DIM);
    auto n1b = new_from_vec(ctx, load_bin("n1_b", DIM), DIM);
    auto W1 = new_from_vec(ctx, load_bin("mlp1_w", DIM * 4 * DIM), DIM, 4 * DIM);
    auto b1 = new_from_vec(ctx, load_bin("mlp1_b", 4 * DIM), 4 * DIM);
    auto W2 = new_from_vec(ctx, load_bin("mlp2_w", 4 * DIM * DIM), 4 * DIM, DIM);
    auto b2 = new_from_vec(ctx, load_bin("mlp2_b", DIM), DIM);
    auto n2w = new_from_vec(ctx, load_bin("n2_w", DIM), DIM);
    auto n2b = new_from_vec(ctx, load_bin("n2_b", DIM), DIM);

    ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, Wq, x), bq);
    ggml_tensor * q = ggml_view_1d(ctx, qkv, DIM, 0);
    ggml_tensor * k = ggml_view_1d(ctx, qkv, DIM, DIM * sizeof(float));
    ggml_tensor * v = ggml_view_1d(ctx, qkv, DIM, 2 * DIM * sizeof(float));
    ggml_tensor * q4 = ggml_reshape_3d(ctx, q, HD, H, L);
    ggml_tensor * k4 = ggml_reshape_3d(ctx, k, HD, H, L);
    ggml_tensor * v4 = ggml_reshape_3d(ctx, v, HD, H, L);

    // 写 cache（pos=10）
    ggml_tensor * kc_slot = ggml_view_3d(ctx, kc, HD, H, 1, HD * sizeof(float),
                                         HD * H * sizeof(float), (size_t)KV_POS * HD * H * sizeof(float));
    ggml_tensor * vc_slot = ggml_view_3d(ctx, vc, HD, H, 1, HD * sizeof(float),
                                         HD * H * sizeof(float), (size_t)KV_POS * HD * H * sizeof(float));
    ggml_tensor * w1 = ggml_cpy(ctx, k4, kc_slot);
    ggml_tensor * w2 = ggml_cpy(ctx, v4, vc_slot);

    const float scale = 1.0f / sqrtf((float)HD);
    std::vector<ggml_tensor *> heads;
    for (int h = 0; h < H; h++) {
        ggml_tensor * qh = ggml_view_2d(ctx, q4, HD, L, HD * sizeof(float), (size_t)h * HD * sizeof(float));
        // NOTE: view + cpy 依赖排序由图保证（w1/w2 先于 scores 使用 kc/vc）
        ggml_tensor * kh = ggml_view_2d(ctx, w1, HD, K, HD * H * sizeof(float), (size_t)h * HD * sizeof(float));
        ggml_tensor * s = ggml_scale(ctx, ggml_mul_mat(ctx, kh, qh), scale);
        // bias: True=attend -> 0, 否则 -1e30
        std::vector<float> bias(K * L);
        for (int t = 0; t < K; t++) bias[t] = vmask[h * K + t] ? 0.0f : -1e30f;
        ggml_tensor * b = new_from_vec(ctx, bias, K, L);
        ggml_tensor * p = ggml_soft_max(ctx, ggml_add(ctx, s, b));
        ggml_tensor * vh = ggml_view_2d(ctx, w2, HD, K, HD * H * sizeof(float), (size_t)h * HD * sizeof(float));
        ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, vh));
        heads.push_back(ggml_mul_mat(ctx, vt, p));
    }
    // concat heads -> [512,1]
    ggml_tensor * o = heads[0];
    for (int h = 1; h < H; h++) o = ggml_concat(ctx, o, heads[h], 0);
    ggml_tensor * proj = ggml_add(ctx, ggml_mul_mat(ctx, Wo, o), bo);
    ggml_tensor * r1 = ggml_add(ctx, x, proj);
    ggml_tensor * n1 = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, r1, 1e-5f), n1w), n1b);
    ggml_tensor * h1 = ggml_add(ctx, ggml_mul_mat(ctx, W1, n1), b1);
    ggml_tensor * h2 = ggml_relu(ctx, h1);
    ggml_tensor * h3 = ggml_add(ctx, ggml_mul_mat(ctx, W2, h2), b2);
    ggml_tensor * r2 = ggml_add(ctx, n1, h3);
    ggml_tensor * res = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, r2, 1e-5f), n2w), n2b);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, res);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    for (auto & kv : g_pending) {
        ggml_backend_tensor_set(kv.first, kv.second->data(), 0,
                                kv.second->size() * sizeof(float));
    }
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> got(DIM * L);
    ggml_backend_tensor_get(res, got.data(), 0, got.size() * sizeof(float));
    float maxdiff = 0;
    for (int i = 0; i < DIM * L; i++) {
        float d = fabsf(got[i] - vout[i]);
        if (d > maxdiff) maxdiff = d;
    }
    printf("GGML-BLOCK maxdiff=%.6f\n", maxdiff);
    ggml_free(ctx);
    return 0;
}
