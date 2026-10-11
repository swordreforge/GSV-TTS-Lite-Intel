// 同形状随机数据对照：区分形状 bug 还是数值 bug
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <cstdio>
#include <vector>
#include <cmath>
#include <random>
#include <cmath>

int main() {
    std::mt19937 rng(0);
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> vx(512), vW(512 * 1536);
    for (auto & v : vx) v = nd(rng);
    for (auto & v : vW) v = nd(rng);
    std::vector<float> ref(1536, 0);
    for (int i = 0; i < 1536; i++) {
        double s = 0;
        for (int j = 0; j < 512; j++) s += (double)vW[j * 1536 + i] * vx[j];
        ref[i] = (float)s;
    }
    struct ggml_init_params params = { 64 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 1);
    ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 1536);
    ggml_tensor * y = ggml_mul_mat(ctx, W, x);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, y);
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(x, vx.data(), 0, vx.size() * 4);
    ggml_backend_tensor_set(W, vW.data(), 0, vW.size() * 4);
    ggml_backend_graph_compute(be, gf);
    std::vector<float> got(1536);
    ggml_backend_tensor_get(y, got.data(), 0, got.size() * 4);
    float m = 0;
    for (int i = 0; i < 1536; i++) { float d = fabsf(got[i] - ref[i]); if (d > m) m = d; }
    printf("RAND512 maxdiff=%.6f\n", m);
    printf("got[0:4]: %.4f %.4f %.4f %.4f\n", got[0], got[1], got[2], got[3]);
    printf("ref[0:4]: %.4f %.4f %.4f %.4f\n", ref[0], ref[1], ref[2], ref[3]);
    printf("ref std: %.4f got std: %.4f\n",
        sqrtf((ref[0]*ref[0]+ref[1]*ref[1])/2), sqrtf((got[0]*got[0]+got[1]*got[1])/2));
    return 0;
}
