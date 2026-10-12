// 路线 B 验证：closed-loop greedy token 循环。
// 每步：emb+pos(entry) -> 24层decode -> predict -> suppress -> sampler.cpp -> token，比对 torch 全程记录。
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-sycl.h"
#include <chrono>
#include "sampler.cpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

static const int H = 16, HD = 32, DIM = 512, NL = 24, VOCAB = 1025;
static const char * DIR = "/tmp/opencode/t2s_closed/";
static const char * WDIR = "/tmp/opencode/t2s_full/";
static const char * EDIR = "/tmp/opencode/t2s_emb/";
static const int K = 512;

static std::vector<std::pair<ggml_tensor *, std::vector<float>>> g_data;

static std::vector<float> load_bin_dir(const std::string & dir, const std::string & name, size_t n) {
    FILE * f = fopen((dir + name + ".bin").c_str(), "rb");
    if (!f) { printf("missing %s%s\n", dir.c_str(), name.c_str()); exit(1); }
    std::vector<float> v(n);
    if (fread(v.data(), sizeof(float), n, f) != n) { printf("short %s\n", name.c_str()); exit(1); }
    fclose(f);
    return v;
}
static std::vector<int64_t> load_i64_dir(const std::string & dir, const std::string & name, size_t n) {
    FILE * f = fopen((dir + name + ".bin").c_str(), "rb");
    if (!f) { printf("missing %s%s\n", dir.c_str(), name.c_str()); exit(1); }
    std::vector<int64_t> v(n);
    if (fread(v.data(), sizeof(int64_t), n, f) != n) { printf("short %s\n", name.c_str()); exit(1); }
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
    int NSTEPS = (int)load_i64_dir(DIR, "nsteps", 1)[0];
    int S = NSTEPS;
    int EOS = (int)load_i64_dir(DIR, "EOS", 1)[0];
    printf("nsteps=%d validating first %d, EOS=%d\n", NSTEPS, S, EOS);

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
    { auto v = load_i64_dir(DIR, "ln0", 1); ln0 = (int)v[0]; }
    if (ln0 + S > K) S = K - ln0 - 1;
    printf("capped S=%d by cache K=%d\n", S, K);
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

    // emb + pe（torch 行序 -> ggml [D,N]）
    auto vemb = load_bin_dir(EDIR, "emb", (size_t)VOCAB * DIM);
    ggml_tensor * emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DIM, VOCAB);
    auto vpe = load_bin_dir(EDIR, "pe_cache", (size_t)4000 * DIM);
    ggml_tensor * pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DIM, 4000);
    float xscale = load_bin_dir(EDIR, "xscale", 1)[0];

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DIM, 1);
    ggml_tensor * bias3 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, H);

    const float scale = 1.0f / sqrtf((float)HD);

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    for (auto & kv : g_data) ggml_backend_tensor_set(kv.first, kv.second.data(), 0, kv.second.size() * sizeof(float));
    ggml_backend_tensor_set(emb, vemb.data(), 0, vemb.size() * sizeof(float));
    ggml_backend_tensor_set(pe, vpe.data(), 0, vpe.size() * sizeof(float));
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_sycl_buffer_type(0));

    auto tokens = load_i64_dir(DIR, "tokens", NSTEPS);
    auto tnext = load_i64_dir(DIR, "tokens_next", NSTEPS);
    auto peidx = load_i64_dir(DIR, "pe_idx", NSTEPS);
    auto supp = load_i64_dir(DIR, "suppressed", NSTEPS);
    // prev 历史：y + 已采样
    FILE * fy = fopen((std::string(DIR) + "y.bin").c_str(), "rb");
    fseek(fy, 0, SEEK_END); size_t ylen = ftell(fy) / 8; fseek(fy, 0, SEEK_SET);
    std::vector<int64_t> y64(ylen);
    if (fread(y64.data(), 8, ylen, fy) != ylen) { printf("y short\n"); return 1; }
    fclose(fy);
    std::vector<int> prev;
    for (auto v : y64) prev.push_back((int)v);

    SampleOpt so;
    so.top_k = 1; so.top_p = 1.0f; so.temp = 1.0f; so.rep = 1.35f;
    std::mt19937 rng(1234);

    int ln = ln0;
    bool allok = true;
    for (int s = 0; s < S; s++) {
        int tok = (int)tokens[s];
        int pi = (int)peidx[s];
        // entry: xy = emb[tok]*scale + pe[pi]
        auto vxy = load_bin_dir(DIR, std::string("xy") + std::to_string(s), DIM);
        {
            int32_t t32 = tok, p32 = pi;
            ggml_tensor * et = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            ggml_tensor * pt = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            // NOTE：et/pt 建在 alloc 之后，无存储——改用 host 直算 entry（entry 已由 test_emb 验证）
            (void)et; (void)pt;
        }
        // entry 沿用已验证的 host 直算（test_emb 已覆盖 ggml 路径）
        std::vector<float> xy(DIM);
        for (int j = 0; j < DIM; j++) xy[j] = vemb[(size_t)tok * DIM + j] * xscale + vpe[(size_t)pi * DIM + j];
        float me = 0;
        for (int j = 0; j < DIM; j++) { float d = fabsf(xy[j] - vxy[j]); if (d > me) me = d; }

        ggml_backend_tensor_set(x, xy.data(), 0, xy.size() * sizeof(float));
        // mask: True for t <= ln（全头同值）
        {
            std::vector<float> bias((size_t)K * H);
            for (int h = 0; h < H; h++)
                for (int t = 0; t < K; t++) bias[h * K + t] = (t <= ln) ? 0.0f : -1e30f;
            ggml_backend_tensor_set(bias3, bias.data(), 0, bias.size() * sizeof(float));
        }

        ggml_tensor * cur = x;
        std::vector<ggml_tensor *> cpys;
        cpys.reserve(2 * NL * H);
        for (int i = 0; i < NL; i++) {
            ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, ly[i].qw, cur), ly[i].qb);
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
            ggml_tensor * qb = ggml_reshape_3d(ctx, ggml_view_1d(ctx, qkv, DIM, 0), HD, 1, H);
            ggml_tensor * scb = ggml_scale(ctx, ggml_mul_mat(ctx, kc[i], qb), scale);
            ggml_tensor * pb = ggml_soft_max(ctx, ggml_add(ctx, scb, bias3));
            ggml_tensor * vtb = ggml_cont(ctx, ggml_transpose(ctx, vc[i]));
            ggml_tensor * ob = ggml_mul_mat(ctx, vtb, pb);
            ggml_tensor * o = ggml_reshape_2d(ctx, ob, DIM, 1);
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
        auto vref = load_bin_dir(DIR, std::string("logits") + std::to_string(s), VOCAB);
        // vref 是 torch sample 入口 logits = raw + rep-penalty(in-place) + suppress(-inf)；
        // C++ got 是 raw：先同序施加 rep，再比（suppressed 位跳过）
        std::vector<float> gotp = got;
        {
            std::vector<char> done(VOCAB, 0);
            for (size_t j = 0; j < prev.size(); j++) {
                int t = prev[j];
                if (t < 0 || t >= VOCAB || done[t]) continue;
                done[t] = 1;
                float sc = gotp[t];
                gotp[t] = sc < 0 ? sc * 1.35f : sc / 1.35f;
            }
        }
        float ml = 0;
        for (int j = 0; j < VOCAB; j++) {
            if (supp[s] && (j == 280 || j == 486 || j == EOS)) continue;
            float d = fabsf(gotp[j] - vref[j]); if (d > ml) ml = d;
        }

        // suppress + sample
        if (supp[s]) {
            got[280] = -INFINITY; got[486] = -INFINITY; got[EOS] = -INFINITY;
        }
        auto probs = logits_to_probs(got.data(), VOCAB, prev.data(), (int)prev.size(), so);
        int tout = multinomial_exp(probs, rng);
        int expect = (int)tnext[s];
        // prev 交叉验证
        auto pvref = load_i64_dir(DIR, std::string("prev") + std::to_string(s), prev.size());
        bool pvok = true;
        for (size_t j = 0; j < prev.size(); j++) if (prev[j] != (int)pvref[j]) { pvok = false; break; }
        if (s < 5 || s % 50 == 0 || tout != expect)
            printf("  closed step %d tok_in=%d entry=%.2e logits=%.2e sample=%d expect=%d %s prev=%s time=%.1fms\n",
               s, tok, me, ml, tout, expect, tout == expect ? "OK" : "MISMATCH", pvok ? "ok" : "PREV-DIFF", ms);
        if (tout != expect) allok = false;
        prev.push_back(tout);
        ln++;
        if (!pvok) allok = false;
        if (tout != expect) break;  // 闭环发散则停
    }
    printf(allok ? "CLOSED-LOOP ALL OK\n" : "CLOSED-LOOP DIVERGED\n");
    ggml_free(ctx);
    return allok ? 0 : 1;
}
