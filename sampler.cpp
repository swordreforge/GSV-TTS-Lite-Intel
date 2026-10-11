// 路线 B：采样器 C++ 实现（对标 utils.sample / logits_to_probs）。
// 确定性部分（penalty/top-k/top-p/softmax）要求与 torch 逐元素一致；
// multinomial 用指数分布 trick（统计等价，token 级不对齐）。
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

struct SampleOpt {
    int top_k = 15;
    float top_p = 1.0f;
    float temp = 1.0f;
    float rep = 1.35f;
};

// 返回 probs（已含全部 mask）；idx 用 probs 做 exponential multinomial
static std::vector<float> logits_to_probs(const float * logits, int V,
                                          const int * prev, int nprev,
                                          const SampleOpt & o) {
    std::vector<float> lg(logits, logits + V);
    for (int i = 0; i < V; i++) lg[i] /= std::max(o.temp, 1e-5f);
    if (nprev > 0 && o.rep != 1.0f) {
        // 与 torch 对齐：先 gather 原值、变换、scatter（重复 token 只算一次，last-wins）
        std::vector<char> done(V, 0);
        for (int j = 0; j < nprev; j++) {
            int t = prev[j];
            if (done[t]) continue;
            done[t] = 1;
            float s = lg[t];
            lg[t] = s < 0 ? s * o.rep : s / o.rep;
        }
    }
    // 顺序与 torch logits_to_probs 一致：rep -> top-p -> top-k -> softmax
    if (o.top_p < 1.0f) {
        std::vector<int> idx(V);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
        // softmax over sorted（fp32 顺序累加，与 torch 逐位对齐）
        float mx = lg[idx[0]];
        std::vector<float> se(V);
        float sum = 0;
        for (int i = 0; i < V; i++) { se[i] = expf(lg[idx[i]] - mx); sum += se[i]; }
        float cum = 0;
        std::vector<char> rem(V, 0);
        for (int i = 0; i < V; i++) {
            cum += se[i] / sum;
            if (cum > o.top_p) rem[i] = 1;
        }
        rem[0] = 0;
        if (getenv("SAMP_DBG")) {
            printf("DBG topp idx[0:5]=%d %d %d %d %d\n", idx[0], idx[1], idx[2], idx[3], idx[4]);
            printf("DBG topp se[0:5]=%.4f %.4f %.4f %.4f %.4f sum=%.4f\n", se[0], se[1], se[2], se[3], se[4], (float)sum);
        }
        if (getenv("SAMP_DUMP")) {
            FILE * f1 = fopen("/tmp/opencode/t2s_sample2/cxx_sorted.bin", "wb");
            for (int i = 0; i < V; i++) { float v = lg[idx[i]]; fwrite(&v, 4, 1, f1); }
            fclose(f1);
            FILE * f2 = fopen("/tmp/opencode/t2s_sample2/cxx_cum.bin", "wb");
            double cc = 0;
            for (int i = 0; i < V; i++) { float v = se[i] / (float)sum; cc += v; float o = (float)cc; fwrite(&o, 4, 1, f2); }
            fclose(f2);
        }
        if (getenv("SAMP_DUMP")) {
            FILE * f = fopen("/tmp/opencode/t2s_sample2/cxx_rem.bin", "wb");
            unsigned char * b = (unsigned char *)malloc(V);
            for (int i = 0; i < V; i++) b[i] = rem[i] ? 1 : 0;
            fwrite(b, 1, V, f); fclose(f); free(b);
        }
        const float NEG = -1e30f;
        for (int i = 0; i < V; i++)
            if (rem[i]) lg[idx[i]] = NEG;
    }
    if (o.top_k > 0) {
        int k = std::min(o.top_k, V);
        std::vector<float> cp = lg;
        std::nth_element(cp.begin(), cp.begin() + k - 1, cp.end(), std::greater<float>());
        float pivot = cp[k - 1];
        const float NEG = -1e30f;
        for (int i = 0; i < V; i++)
            if (lg[i] < pivot) lg[i] = NEG;
    }
    float mx = *std::max_element(lg.begin(), lg.end());
    std::vector<float> probs(V);
    float sum = 0;
    for (int i = 0; i < V; i++) { probs[i] = expf(lg[i] - mx); sum += probs[i]; }
    for (int i = 0; i < V; i++) probs[i] /= sum;
    if (getenv("SAMP_DUMP")) {
        FILE * f0 = fopen("/tmp/opencode/t2s_sample2/cxx_topk_masked.bin", "wb");
        fwrite(lg.data(), 4, V, f0); fclose(f0);
        FILE * f = fopen("/tmp/opencode/t2s_sample2/cxx_probs.bin", "wb");
        fwrite(probs.data(), 4, V, f); fclose(f);
        FILE * g = fopen("/tmp/opencode/t2s_sample2/cxx_exp.bin", "wb");
        for (int i = 0; i < V; i++) { float e = expf(lg[i] - mx); fwrite(&e, 4, 1, g); }
        fclose(g);
    }
    if (getenv("SAMP_DBG")) printf("DBG mx=%.4f sum=%.4f n=%d\n", mx, sum, V);
    if (getenv("SAMP_DBG")) {
        float t = expf(4.7979274f - 10.088423f);
        printf("DBG direct expf(-5.29)=%.8f lg894=%.4f\n", t, lg[894]);
    }
    return probs;
}

static int multinomial_exp(const std::vector<float> & probs, std::mt19937 & rng) {
    std::exponential_distribution<float> ed(1.0f);
    int best = 0;
    float bestv = -1e30f;
    for (int i = 0; i < (int)probs.size(); i++) {
        float q = ed(rng);
        float v = probs[i] / q;
        if (v > bestv) { bestv = v; best = i; }
    }
    return best;
}

#ifdef SAMPLER_TEST
#include <string>
static std::vector<float> load_bin(const std::string & p, size_t n) {
    FILE * f = fopen(p.c_str(), "rb");
    std::vector<float> v(n);
    fread(v.data(), sizeof(float), n, f);
    fclose(f);
    return v;
}
static std::vector<int> load_i32(const std::string & p, size_t n) {
    FILE * f = fopen(p.c_str(), "rb");
    std::vector<int> v(n);
    fread(v.data(), sizeof(int), n, f);
    fclose(f);
    return v;
}
int main() {
    const char * D = "/tmp/opencode/t2s_sample3/";
    auto lg = load_bin(std::string(D) + "logits.bin", 1025);
    auto pv = load_i32(std::string(D) + "prev.bin", 300);
    const char * which = getenv("SAMP_WHICH") ? getenv("SAMP_WHICH") : "probs";
    auto pr = load_bin(std::string(D) + std::string(which) + ".bin", 1025);
    SampleOpt o;
    std::string w = getenv("SAMP_WHICH") ? getenv("SAMP_WHICH") : "topk";
    if (w == "plain") { o.top_k = 0; o.top_p = 1.0f; o.rep = 1.0f; }
    else if (w == "rep-only") { o.top_k = 0; o.top_p = 1.0f; }
    else if (w == "topk") { o.top_p = 1.0f; }
    else if (w == "topp") { o.top_p = 0.9f; }
    auto probs = logits_to_probs(lg.data(), 1025, pv.data(), 300, o);
    float m = 0;
    for (int i = 0; i < 1025; i++) { float d = fabsf(probs[i] - pr[i]); if (d > m) m = d; }
    int bi = 0;
    for (int i = 0; i < 1025; i++) { float d = fabsf(probs[i] - pr[i]); if (d > fabsf(probs[bi] - pr[bi])) bi = i; }
    printf("SAMPLER probs maxdiff=%.7f at %d (mine=%.6f ref=%.6f logit=%.4f)\n", m, bi, probs[bi], pr[bi], lg[bi]);
    return 0;
}
#endif
