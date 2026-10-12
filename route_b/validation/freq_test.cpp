// 采样器频率测试：固定 probs 下 20 万次抽样，频率≈probs？
#include "sampler.cpp"
#include <cstdio>
#include <random>
int main() {
    const int V = 8;
    float p[V] = {0.40f, 0.25f, 0.15f, 0.10f, 0.05f, 0.03f, 0.015f, 0.005f};
    std::vector<float> probs(p, p + V);
    std::mt19937 rng(1234);
    const int N = 200000;
    std::vector<int> cnt(V, 0);
    for (int i = 0; i < N; i++) cnt[multinomial_exp(probs, rng)]++;
    double chi2 = 0;
    for (int i = 0; i < V; i++) {
        double e = N * p[i];
        double d = cnt[i] - e;
        chi2 += d * d / e;
        printf("tok %d: got %.4f expect %.4f\n", i, cnt[i] / (double)N, p[i]);
    }
    printf("chi2=%.1f (df=7, crit95=14.1)\n", chi2);
    return 0;
}
