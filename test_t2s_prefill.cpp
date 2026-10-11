// 路线 B 验证：prefill（process_prompt）全量复刻 vs torch。
// 与 decode 的区别：cache 写 [0:L) 全段；attention 用当步 k/v；mask [B,H,L,L]。
// 布局规则：torch (B,H,T,D) -> ggml [D,T,H]；权重原字节声明 [in,out]。
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <cstring>
#include <cstring>

static const int H = 16, HD = 32, DIM = 512, NL = 24;
static const char * WDIR = "/tmp/opencode/t2s_full/";
static int K = 512, L = 0;

static std::string g_dir() {
    const char * e = getenv("PREF_DIR");
    return e ? e : "/tmp/opencode/t2s_prefill/";
}

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
                       std::initializer_list<int> dims) {
    std::vector<int64_t> ne;
    for (int d : dims) ne.push_back(d);
    while (ne.size() < 4) ne.push_back(1);
    size_t n = 1;
    for (auto d : ne) n *= d;
    auto v = load_bin_dir(dir, name, n);
    ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
    g_data.emplace_back(t, std::move(v));
    return t;
}

int main() {
    { FILE * f = fopen((g_dir() + "mask.bin").c_str(), "rb");
      if (!f) { printf("no mask\n"); return 1; }
      fseek(f, 0, SEEK_END);
      L = (int)sqrt((double)(ftell(f) / H));
      fclose(f); }
    printf("prefill L=%d K=%d\n", L, K);
    fflush(stdout);

    struct ggml_init_params params = { 2048ULL * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * x = T(ctx, g_dir(), "x", {DIM, L});
    std::vector<ggml_tensor *> kc(NL), vc(NL);
    for (int i = 0; i < NL; i++) {
        char n1[32], n2[32];
        snprintf(n1, 32, "kc%d", i); snprintf(n2, 32, "vc%d", i);
        kc[i] = T(ctx, g_dir(), n1, {HD, K, H});
        vc[i] = T(ctx, g_dir(), n2, {HD, K, H});
    }
    FILE * fm = fopen((g_dir() + "mask.bin").c_str(), "rb");
    std::vector<uint8_t> vmask(H * L * L);
    if (fread(vmask.data(), 1, H * L * L, fm) != (size_t)(H * L * L)) { printf("mask short\n"); return 1; }
    fclose(fm);

    struct Lyr { ggml_tensor *qw, *qb, *ow, *ob, *n1w, *n1b, *m1w, *m1b, *m2w, *m2b, *n2w, *n2b; };
    std::vector<Lyr> ly(NL);
    for (int i = 0; i < NL; i++) {
        char n[32];
        snprintf(n, 32, "b%d.qkv_w", i); ly[i].qw = T(ctx, WDIR, n, {DIM, 3 * DIM});
        snprintf(n, 32, "b%d.qkv_b", i); ly[i].qb = T(ctx, WDIR, n, {3 * DIM});
        snprintf(n, 32, "b%d.out_w", i); ly[i].ow = T(ctx, WDIR, n, {DIM, DIM});
        snprintf(n, 32, "b%d.out_b", i); ly[i].ob = T(ctx, WDIR, n, {DIM});
        snprintf(n, 32, "b%d.n1_w", i); ly[i].n1w = T(ctx, WDIR, n, {DIM});
        snprintf(n, 32, "b%d.n1_b", i); ly[i].n1b = T(ctx, WDIR, n, {DIM});
        snprintf(n, 32, "b%d.mlp1_w", i); ly[i].m1w = T(ctx, WDIR, n, {DIM, 4 * DIM});
        snprintf(n, 32, "b%d.mlp1_b", i); ly[i].m1b = T(ctx, WDIR, n, {4 * DIM});
        snprintf(n, 32, "b%d.mlp2_w", i); ly[i].m2w = T(ctx, WDIR, n, {4 * DIM, DIM});
        snprintf(n, 32, "b%d.mlp2_b", i); ly[i].m2b = T(ctx, WDIR, n, {DIM});
        snprintf(n, 32, "b%d.n2_w", i); ly[i].n2w = T(ctx, WDIR, n, {DIM});
        snprintf(n, 32, "b%d.n2_b", i); ly[i].n2b = T(ctx, WDIR, n, {DIM});
    }

    const float scale = 1.0f / sqrtf((float)HD);
    ggml_tensor * cur = x;
    ggml_tensor * q0_node = nullptr;
    ggml_tensor * st_h1 = nullptr, * st_h3 = nullptr, * st_n2 = nullptr;
    ggml_tensor * c_s0 = nullptr, * c_p0 = nullptr, * c_o0 = nullptr;
    ggml_tensor * st_o = nullptr, * st_proj = nullptr, * st_n1 = nullptr;
    std::vector<ggml_tensor *> layer_nodes;
    std::vector<ggml_tensor *> heads0;
    for (int i = 0; i < NL; i++) {
        ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, ly[i].qw, cur), ly[i].qb);
        ggml_tensor * q = ggml_view_3d(ctx, qkv, HD, H, L, HD * sizeof(float), 3 * DIM * sizeof(float), 0);
        if (i == 0) q0_node = q;
        ggml_tensor * k = ggml_view_3d(ctx, qkv, HD, H, L, HD * sizeof(float), 3 * DIM * sizeof(float), DIM * sizeof(float));
        ggml_tensor * v = ggml_view_3d(ctx, qkv, HD, H, L, HD * sizeof(float), 3 * DIM * sizeof(float), 2 * DIM * sizeof(float));
        for (int h = 0; h < H; h++) {
            ggml_tensor * khw = ggml_view_2d(ctx, k, HD, L, 3 * DIM * sizeof(float), (size_t)h * HD * sizeof(float));
            ggml_tensor * dstk = ggml_view_2d(ctx, kc[i], HD, L, HD * sizeof(float),
                                             (size_t)h * HD * K * sizeof(float));
            ggml_cpy(ctx, khw, dstk);
            ggml_tensor * vhw = ggml_view_2d(ctx, v, HD, L, 3 * DIM * sizeof(float), (size_t)h * HD * sizeof(float));
            ggml_tensor * dstv = ggml_view_2d(ctx, vc[i], HD, L, HD * sizeof(float),
                                             (size_t)h * HD * K * sizeof(float));
            ggml_cpy(ctx, vhw, dstv);
        }
        std::vector<ggml_tensor *> heads;
        for (int h = 0; h < H; h++) {
            ggml_tensor * qh = ggml_view_2d(ctx, q, HD, L, 3 * DIM * sizeof(float), (size_t)h * HD * sizeof(float));
            ggml_tensor * kh = ggml_view_2d(ctx, k, HD, L, 3 * DIM * sizeof(float), (size_t)h * HD * sizeof(float));
            ggml_tensor * s = ggml_scale(ctx, ggml_mul_mat(ctx, kh, qh), scale);
            // s 与 torch 同 flat 顺序（已验证 exact），bias 照 torch 语义直接加
            std::vector<float> bias(L * L);
            for (int a = 0; a < L; a++)
                for (int b2 = 0; b2 < L; b2++)
                    bias[a * L + b2] = vmask[(h * L + a) * L + b2] ? 0.0f : -1e30f;
            ggml_tensor * bb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, L);
            g_data.emplace_back(bb, std::move(bias));
            ggml_tensor * p = ggml_soft_max(ctx, ggml_add(ctx, s, bb));
            if (i == 0 && h == 0) { c_s0 = s; c_p0 = p; }
            ggml_tensor * vh = ggml_view_2d(ctx, v, HD, L, 3 * DIM * sizeof(float), (size_t)h * HD * sizeof(float));
            ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, vh));
            heads.push_back(ggml_mul_mat(ctx, vt, p));
        }
        ggml_tensor * o = heads[0];
        for (int h = 1; h < H; h++) o = ggml_concat(ctx, o, heads[h], 0);
        if (i == 0) heads0 = heads;
        // c_o0 取 heads0[0]，后面统一探
        if (i == 0) st_o = o;
        ggml_tensor * proj = ggml_add(ctx, ggml_mul_mat(ctx, ly[i].ow, o), ly[i].ob);
        if (i == 0) st_proj = proj;
        ggml_tensor * r1 = ggml_add(ctx, cur, proj);
        ggml_tensor * n1 = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, r1, 1e-5f), ly[i].n1w), ly[i].n1b);
        if (i == 0) st_n1 = n1;
        ggml_tensor * h1 = ggml_add(ctx, ggml_mul_mat(ctx, ly[i].m1w, n1), ly[i].m1b);
        if (i == 0) st_h1 = h1;
        ggml_tensor * h3 = ggml_add(ctx, ggml_mul_mat(ctx, ly[i].m2w, ggml_relu(ctx, h1)), ly[i].m2b);
        if (i == 0) { st_h3 = h3; }
        cur = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, ggml_add(ctx, n1, h3), 1e-5f), ly[i].n2w), ly[i].n2b);
        if (i == 0) st_n2 = cur;
        layer_nodes.push_back(cur);
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    for (auto & kv : g_data) ggml_backend_tensor_set(kv.first, kv.second.data(), 0, kv.second.size() * sizeof(float));

    // head0 s/p/o 探针（owned 节点，对新鲜参照 f_s0/f_p0/f_o0）
    {
        struct { const char * n; ggml_tensor * t; int nn; } ps[] = {
            {"f_s0", c_s0, L * L}, {"f_p0", c_p0, L * L}, {"f_o0", heads0[0], HD * L}};
        for (auto & pr : ps) {
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, false);
            ggml_build_forward_expand(gf, pr.t);
            ggml_backend_graph_compute(backend, gf);
            auto vr = load_bin_dir(g_dir(), pr.n, pr.nn);
            std::vector<float> gl(pr.nn);
            ggml_backend_tensor_get(pr.t, gl.data(), 0, gl.size() * sizeof(float));
            float m = 0;
            for (int j = 0; j < pr.nn; j++) { float d = fabsf(gl[j] - vr[j]); if (d > m) m = d; }
            printf("  fresh %s maxdiff=%.6f\n", pr.n, m);
            if (!strcmp(pr.n, "f_s0")) {
                // vr 是 torch(query,key) flat=a*263+b；试 C++(i,j) 的两种解释
                auto vr2 = load_bin_dir(g_dir(), "f_s0", (size_t)L * L);
                std::vector<float> g2(L * L);
                ggml_backend_tensor_get(pr.t, g2.data(), 0, g2.size() * sizeof(float));
                float md = 0;
                for (int a = 0; a < L; a++) for (int b = 0; b < L; b++) {
                    // 假设 C++[i,j] = torch(query=i,key=j)
                    float d = fabsf(g2[a * L + b] - vr2[a * L + b]);
                    if (d > md) md = d;
                }
                printf("  s-as-(query,key) maxdiff=%.6f\n", md);
            }
            if (!strcmp(pr.n, "f_p0")) {
                printf("  p0 got[0:5]: %.4f %.4f %.4f %.4f %.4f\n", gl[0], gl[1], gl[2], gl[3], gl[4]);
            {
                double r0 = 0, c0 = 0;
                for (int i = 0; i < 263; i++) { r0 += vr[i]; c0 += gl[i]; }
                printf("  p0 query0 rowsum: ref=%.4f cxx=%.4f\n", r0, c0);
            }
                printf("  p0 ref[0:5]: %.4f %.4f %.4f %.4f %.4f\n", vr[0], vr[1], vr[2], vr[3], vr[4]);
            }
        }
    }
    // mlp 段探针
    {
        struct { const char * n; ggml_tensor * t; int nn; const char * f; } ps[] = {
            {"g_h1", st_h1, 4 * DIM * L, "g_h1"}, {"g_h3", st_h3, DIM * L, "g_h3"}, {"g_n2", st_n2, DIM * L, "g_n2"}};
        for (auto & pr : ps) {
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, false);
            ggml_build_forward_expand(gf, pr.t);
            ggml_backend_graph_compute(backend, gf);
            auto vr = load_bin_dir(g_dir(), pr.f, pr.nn);
            std::vector<float> gl(pr.nn);
            ggml_backend_tensor_get(pr.t, gl.data(), 0, gl.size() * sizeof(float));
            float m = 0;
            for (int j = 0; j < pr.nn; j++) { float d = fabsf(gl[j] - vr[j]); if (d > m) m = d; }
            printf("  stage %s maxdiff=%.6f\n", pr.n, m);
            if (!strcmp(pr.n, "g_h1")) {
                FILE * f = fopen("/tmp/opencode/t2s_prefill/cxx_h1.bin", "wb");
                fwrite(gl.data(), 4, gl.size(), f); fclose(f);
            }
        }
    }
    // per-head attention 输出探针
    for (int h = 0; h < H; h++) {
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, false);
        ggml_build_forward_expand(gf, heads0[h]);
        ggml_backend_graph_compute(backend, gf);
        char n[32];
        snprintf(n, 32, "t_head%d", h);
        auto vr = load_bin_dir(g_dir(), n, (size_t)HD * L);
        std::vector<float> gl(HD * L);
        ggml_backend_tensor_get(heads0[h], gl.data(), 0, gl.size() * sizeof(float));
        float m = 0;
        for (int j = 0; j < HD * L; j++) { float d = fabsf(gl[j] - vr[j]); if (d > m) m = d; }
        printf("  head %2d attn-out maxdiff=%.6f\n", h, m);
    }
    // layer0 分段探针（只用 owned 节点）
    {
        struct { const char * n; ggml_tensor * t; } ps[] = {
            {"t_attn", st_o}, {"t_proj", st_proj}, {"t_n1", st_n1}};
        for (auto & pr : ps) {
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, false);
            ggml_build_forward_expand(gf, pr.t);
            ggml_backend_graph_compute(backend, gf);
            auto vr = load_bin_dir(g_dir(), pr.n, (size_t)DIM * L);
            std::vector<float> gl(DIM * L);
            ggml_backend_tensor_get(pr.t, gl.data(), 0, gl.size() * sizeof(float));
            float m = 0;
            for (int j = 0; j < DIM * L; j++) { float d = fabsf(gl[j] - vr[j]); if (d > m) m = d; }
            printf("  stage %s maxdiff=%.6f\n", pr.n, m);
            if (!strcmp(pr.n, "g_h1")) {
                FILE * f = fopen("/tmp/opencode/t2s_prefill/cxx_h1.bin", "wb");
                fwrite(gl.data(), 4, gl.size(), f); fclose(f);
            }
        }
    }
    const char * stages[] = {"t_attn", "t_proj", "t_n1"};
    ggml_tensor * stage_nodes[] = {nullptr, nullptr, nullptr};
    for (int i = 0; i < NL; i++) {
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, false);
        ggml_build_forward_expand(gf, layer_nodes[i]);
        ggml_backend_graph_compute(backend, gf);
        char n[32];
        if (i == 0) { auto vr0 = load_bin_dir(g_dir(), "fresh_ly0", (size_t)DIM * L);
            std::vector<float> gl0(DIM * L);
            ggml_backend_tensor_get(layer_nodes[i], gl0.data(), 0, gl0.size() * sizeof(float));
            float m0 = 0;
            for (int j = 0; j < DIM * L; j++) { float dd = fabsf(gl0[j] - vr0[j]); if (dd > m0) m0 = dd; }
            printf("  prefill layer %2d vs fresh maxdiff=%.6f\n", i, m0);
            if (m0 > 1e-3) break;
            continue; }
        snprintf(n, 32, "ly%d", i);
        auto vr = load_bin_dir(g_dir(), n, (size_t)DIM * L);
        std::vector<float> gl(DIM * L);
        ggml_backend_tensor_get(layer_nodes[i], gl.data(), 0, gl.size() * sizeof(float));
        float m = 0;
        for (int j = 0; j < DIM * L; j++) { float d = fabsf(gl[j] - vr[j]); if (d > m) m = d; }
        printf("  prefill layer %2d vs fresh maxdiff=%.6f\n", i, m);
    }
    ggml_free(ctx);
    return 0;
}
