// 隔离验证 ggml_mul_mat(Wq, x) + b
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <cstdio>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

static std::vector<float> load_bin(const std::string & p, size_t n) {
    FILE * f = fopen(p.c_str(), "rb");
    std::vector<float> v(n);
    fread(v.data(), sizeof(float), n, f);
    fclose(f);
    return v;
}

int main() {
    const char * D = "/tmp/opencode/t2s_full/";
    auto vx = load_bin(std::string(D) + "x.bin", 512);
    auto vW = load_bin(std::string(D) + "b0.qkv_w.bin", 512 * 1536);
    auto vb = load_bin(std::string(D) + "b0.qkv_b.bin", 1536);
    auto vref = load_bin(std::string(D) + "synth_ref.bin", 1536);
    { static bool once = true; if (once) { once = false;
        unsigned s = 0;
        for (size_t i = 0; i < vx.size(); i++) { vx[i] = ((s = s * 1103515245 + 12345) >> 16) / 65536.0f - 0.5f; }
        for (size_t i = 0; i < vW.size(); i++) { vW[i] = ((s = s * 1103515245 + 12345) >> 16) / 65536.0f - 0.5f; }
        // numpy ref for synthetic: recompute below via file
    } }

    struct ggml_init_params params = { 64 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 1);
    ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 1536);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1536);
    ggml_tensor * y = ggml_mul_mat(ctx, W, x);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, y);
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(x, vx.data(), 0, vx.size() * 4);
    ggml_backend_tensor_set(W, vW.data(), 0, vW.size() * 4);
    ggml_backend_tensor_set(b, vb.data(), 0, vb.size() * 4);
    ggml_backend_graph_compute(be, gf);
    std::vector<float> got(1536);
    ggml_backend_tensor_get(y, got.data(), 0, got.size() * 4);
    float m = 0;
    for (int i = 0; i < 1536; i++) { float d = fabsf(got[i] - vref[i]); if (d > m) m = d; }
    printf("MULISO maxdiff=%.6f\n", m);
    {
        std::vector<float> a(got.begin(), got.end()), b(vref.begin(), vref.end());
        std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
        float ms = 0;
        for (int i = 0; i < 1536; i++) { float d = fabsf(a[i] - b[i]); if (d > ms) ms = d; }
        printf("MULISO sorted-maxdiff=%.6f\n", ms);
    printf("got[0:6]: %.4f %.4f %.4f %.4f %.4f %.4f\n", got[0], got[1], got[2], got[3], got[4], got[5]);
    printf("ref[0:6]: %.4f %.4f %.4f %.4f %.4f %.4f\n", vref[0], vref[1], vref[2], vref[3], vref[4], vref[5]);
    }
    return 0;
}
