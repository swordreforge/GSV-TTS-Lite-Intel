// 用 arange 模式一锤定音：ggml 到底怎么解释 buffer 布局
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <cstdio>
#include <vector>
#include <cmath>

int main() {
    // buffer 按 C-order [512,1536] 填 arange：buf[f] = f
    std::vector<float> vW(512 * 1536), vx(512);
    for (size_t i = 0; i < vW.size(); i++) vW[i] = (float)i;
    for (int i = 0; i < 512; i++) vx[i] = 1.0f;  // x 全 1：结果 = 行和/列和，直接暴露求和轴
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
    // x=1：若按 C-order[512,1536] 行求和：out[i] = sum_{j} (j*1536+i) = 1536*(511*512/2)... 大数
    // 若按转置解释：out[i] = sum_j (i*512+j) = 512*i + 511*512/2
    double expectC = 0, expectT = 0;
    for (int j = 0; j < 512; j++) expectC += j * 1536 + 0;   // C-order 行和 i=0
    for (int j = 0; j < 512; j++) expectT += 0 * 512 + j;    // 转置解释 i=0
    printf("got[0]=%.1f  C-order-row-sum=%.1f  transposed=%.1f\n", got[0], expectC, expectT);
    printf("got[1]=%.1f\n", got[1]);
    return 0;
}
