// 路线 B 验证：embedding + position 入口（ggml_get_rows + scale + add），vs torch xy_ref。
// 输入：/tmp/opencode/t2s_emb/{emb,pe_cache,tokens,pe_idx,xy*,xscale}
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

static const char * DIR = "/tmp/opencode/t2s_emb/";
static const int DIM = 512, PE_LEN = 4000, NSTEPS = 5;

static std::vector<float> load_bin(const std::string & name, size_t n) {
    FILE * f = fopen((std::string(DIR) + name + ".bin").c_str(), "rb");
    if (!f) { printf("missing %s\n", name.c_str()); exit(1); }
    std::vector<float> v(n);
    if (fread(v.data(), sizeof(float), n, f) != n) { printf("short %s\n", name.c_str()); exit(1); }
    fclose(f);
    return v;
}

int main() {
    setbuf(stdout, nullptr);
    int V = 0;
    { FILE * f = fopen((std::string(DIR) + "emb.bin").c_str(), "rb");
      if (!f) { printf("no emb\n"); return 1; }
      fseek(f, 0, SEEK_END); V = (int)(ftell(f) / (DIM * sizeof(float))); fclose(f); }
    printf("V=%d\n", V);

    struct ggml_init_params params = { 512ULL * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);

    auto vemb = load_bin("emb", (size_t)V * DIM);
    // torch [V,D] bytes -> ggml [D,V]
    ggml_tensor * emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DIM, V);
    auto vpe = load_bin("pe_cache", (size_t)PE_LEN * DIM);
    // torch [4000,D] rows -> ggml [D,4000]
    ggml_tensor * pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DIM, PE_LEN);
    float xscale = load_bin("xscale", 1)[0];

    ggml_tensor * tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * peidx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    // out = rows(emb,tok)*scale + rows(pe,peidx)
    ggml_tensor * e = ggml_get_rows(ctx, emb, tok);          // [D,1]
    ggml_tensor * es = ggml_scale(ctx, e, xscale);
    ggml_tensor * p = ggml_get_rows(ctx, pe, peidx);         // [D,1]
    ggml_tensor * out = ggml_add(ctx, es, p);               // [D,1]

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(emb, vemb.data(), 0, vemb.size() * sizeof(float));
    ggml_backend_tensor_set(pe, vpe.data(), 0, vpe.size() * sizeof(float));
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    ggml_gallocr_alloc_graph(galloc, gf);

    FILE * ft = fopen((std::string(DIR) + "tokens.bin").c_str(), "rb");
    FILE * fi = fopen((std::string(DIR) + "pe_idx.bin").c_str(), "rb");
    if (!ft || !fi) { printf("no tokens/pe_idx\n"); return 1; }
    for (int s = 0; s < NSTEPS; s++) {
        int64_t t = 0, pi = 0;
        if (fread(&t, 8, 1, ft) != 1 || fread(&pi, 8, 1, fi) != 1) { printf("short tok/idx\n"); return 1; }
        int32_t t32 = (int32_t)t, pi32 = (int32_t)pi;
        ggml_backend_tensor_set(tok, &t32, 0, sizeof(t32));
        ggml_backend_tensor_set(peidx, &pi32, 0, sizeof(pi32));
        ggml_backend_graph_compute(backend, gf);
        std::vector<float> got(DIM);
        ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));
        char rn[32]; snprintf(rn, 32, "xy%d", s);
        auto vref = load_bin(rn, DIM);
        float m = 0;
        for (int j = 0; j < DIM; j++) { float d = fabsf(got[j] - vref[j]); if (d > m) m = d; }
        printf("  emb step %d tok=%lld pe=%lld maxdiff=%.7f\n", s, (long long)t, (long long)pi, m);
    }
    fclose(ft); fclose(fi);
    ggml_free(ctx);
    return 0;
}
