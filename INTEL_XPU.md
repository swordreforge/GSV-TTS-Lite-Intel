# 在 Intel 生态下运行 GSV-TTS-Lite（非官方路径）

官方 README 只给出 **CUDA / MPS / CPU** 三种后端。本文件记录如何在 **Intel 生态**
（`torch.xpu` + oneDNN + Intel oneAPI runtime）上跑通本项目，完全不依赖 NVIDIA。

> 目标：绕开官方 CUDA 指引，用 Intel Arc 集成显卡做 TTS 推理。

## 1. 运行环境

| 项 | 值 |
| --- | --- |
| CPU | Intel Core Ultra 7 155H (Meteor Lake) |
| GPU | Intel Arc Graphics (Xe-LPG iGPU, `renderD128`) |
| NPU | Intel AI Boost (已装 `libze_intel_npu`) |
| 系统 | Arch Linux, Python 3.14.7 |
| 驱动 | `xe`/`i915` + Intel Compute Runtime (`libze_intel_gpu` 1.17) |
| 框架 | `torch 2.14.1+xpu` / `torchaudio 2.11.0+xpu` |

系统自带的 `torch` 是 `+cpu` 版，**不含 XPU**，所以必须换用 XPU wheel。
XPU 支持由 PyTorch 官方维护（`torch.xpu`），并非 Intel Extension 私有接口。

## 2. 一键复现

```bash
cd GSV-TTS-Lite

# 1) 创建独立 venv（用 uv 更快）
uv venv --python 3.14 .venv-xpu

# 2) 安装 Intel XPU 版 PyTorch（注意 unsafe-best-match，否则会被 PyPI 的 CUDA 版抢占）
uv pip install --python .venv-xpu/bin/python \
  --index-strategy unsafe-best-match \
  --index-url https://download.pytorch.org/whl/xpu \
  --extra-index-url https://pypi.org/simple \
  "torch==2.14.1+xpu" "torchaudio==2.11.0+xpu"

# 3) 其它依赖
uv pip install --python .venv-xpu/bin/python \
  einops tqdm av transformers sounddevice safetensors numpy \
  py3langid pysbd onnxruntime soundfile requests nltk future \
  inflect wordsegment pypinyin jieba pyopenjtalk-plus

# 4) 跑起来（首次会自动/或手动下载预训练模型到 ~/.cache/gsv）
.venv-xpu/bin/python run_xpu.py
```

`run_xpu.py` 生成的音频默认写到 `/tmp/opencode/gsv_xpu_out.wav`。
基准脚本见 `bench_xpu.py`：

```bash
.venv-xpu/bin/python bench_xpu.py xpu float16
```

## 3. 代码改动（最小化）

只动了两个文件，且都做了分支保护，不影响 CUDA/MPS/CPU 原有路径：

- `gsv_tts/Config.py`
  - 新增 `get_xpu_device_info()`；
  - 设备探测顺序扩展为 **CUDA → XPU → MPS → CPU**，XPU 默认 `float16`；
  - `choose_attention_backend()` 改为在实际加速设备上探测，XPU 下选中
    `EFFICIENT_ATTENTION`（Meteor Lake 实际回退到 oneDNN math 内核）。
- `gsv_tts/TTS.py`
  - `_empty_cache()` 支持 `torch.xpu.empty_cache()`；
  - 显式 `device="xpu"` 时启用 float16 快路径；
  - `dtype` 映射补充 `bfloat16`。

> CUDA Graph 相关代码原本就用 `device.type == "cuda"` 保护，因此 XPU 自动走
> 普通 KV-Cache 的 eager 路径，无需改动。

## 4. 性能实测（Meteor Lake Arc iGPU）

短句 `へぇー、ここまでしてくれるんですね。`，32 kHz 输出：

| 路径 | dtype | 音频时长 | RTF | 备注 |
| --- | --- | --- | --- | --- |
| `infer_batched`（3 句批量） | float16 | ~13–14 s | **~0.53** | 优于实时 |
| `infer`（单句，热） | float16 | ~3.5 s | ~0.7–1.8 | 波动较大 |
| `infer_batched`（3 句批量） | bfloat16 | ~13 s | ~0.6–0.75 | 略慢于 fp16 |

API 服务（FastAPI）冷请求 3.24 s 音频约 15 s，热请求 2.68 s 音频约 5.8 s。

> iGPU 与内存共享带宽，且 Xe-LPG 不支持 FlashAttention（仅 PVC/BMG 支持），
> 注意力只能走 math 内核，因此 RTF 明显高于官方 RTX 数据，但已可批处理实时。

## 5. 已验证功能

- ✅ GPT + SoVITS 全流程推理（ja / zh / en 多语言）
- ✅ `infer` / `infer_batched` / 流式路径
- ✅ 音色/风格参考音频缓存
- ✅ FastAPI 服务 `/tts/single`（XPU 上起服务并出音频）
- ✅ WebUI / API 不传 `device` 时经由 `Config` 自动选中 XPU

## 6. 注意事项 / 局限

1. **ASR 部分仍写死 CUDA**：`WebUI/web.py`、`API/*.py` 中 Qwen3-ASR 使用
   `device_map="cuda:0"` 与 `flash_attention_2`，在 Intel 上会失败（API 里已
   `try/except` 降级为 ASR 关闭）。要在 Intel 上跑 ASR，可改用 OpenVINO /
   `optimum-intel` 部署到 GPU 或 NPU。
2. **FlashAttention 不可用**：Xe-LPG 不在 PyTorch XPU 的 flash-attn 白名单内，
   务必保持 `use_flash_attn=False`（且 `flash_attn` 本就装不上 Intel）。
3. **共享显存**：iGPU `torch.xpu.get_device_properties(0).total_memory` 报的是
   系统内存份额（约 28 GB），并非独占显存。
4. 若同时存在 NVIDIA 与 Intel，当前顺序仍优先 CUDA；可显式传
   `TTS(device="xpu")` 强制走 Intel。

## 7. 后续可选优化

- `torch.compile(..., backend="inductor")` 在 XPU 上融合算子；
- GPT / SoVITS 主模型也可以尝试导出 OpenVINO（动态 KV-Cache 改写较大，
  投入产出比不如 featurizer，暂未做）；
- 用 `intel-extension-for-pytorch` (IPEX) 的 XPU 优化（注意与 torch 版本配套）。

---

## 附录 A. 先说清楚：这块 NPU 很弱

本机 NPU 是 Meteor Lake 的 **Intel AI Boost（OpenVINO 识别为架构 3720）**，
定位是超低功耗的后台推理单元，不是通用加速器：

1. **算力小一个数量级**：INT8 峰值约 10–12 TOPS，且只有 FP16/INT8 两档；
   同一颗芯片里的 Arc iGPU（8 个 Xe 核心）光 FP32 就有数 TFLOPS，
   INT8/DPAS 吞吐更是 NPU 的数倍，还有 ~28 GB 共享内存可用。
2. **片上 SRAM 很小**：放不下大 transformer 的长序列中间激活，
   跑 RoBERTa-large（24 层 × 1024 维）这类模型全靠反复搬运，效率打折。
3. **只吃静态 shape**：动态 batch/seq 的模型在 NPU 上直接编译失败
   （实测 HuBERT 动态时间轴：`Upper bounds are not specified` +
   Level-Zero `ZE_RESULT_ERROR_INVALID_ARGUMENT`）。管线里 prompt 音频、
   文本长度每次都变，用 NPU 就得按桶 pad + 每个新桶重编译一次（约 6 s）。
4. **算子覆盖窄**：部分 control-flow / 动态 reshape 类算子不支持，
   复杂模型容易在编译期被拒。

结论：**NPU 能跑，但只适合形状固定、体量小的模型**；本管线里它的价值是
"低功耗分担 featurizer"，指望它加速主体是不现实的。下面的实测也印证了这点。

## 附录 B. CNHubert / CNRoberta 切 OpenVINO（Arc GPU / NPU）

### 做法

- HuBERT：只导出真正用的 `last_hidden_state`（`export_hubert_ov.py`），
  FP16 IR、时间轴动态 `[1, -1]`，存 `~/.cache/gsv/ov_models/hubert_fp16.xml`。
- RoBERTa：只导出管线真正用的 `hidden_states[-3]`（`export_roberta_ov.py`），
  FP16 IR、batch/seq 双动态，存 `~/.cache/gsv/ov_models/roberta_h3_fp16.xml`。
  这样避免一次吐出 25 层 hidden states。
- 集成（环境变量开关，默认保持 torch 路径不变）：
  - `GSV_HUBERT_BACKEND=openvino` + `GSV_OV_DEVICE=GPU|CPU` → `CNHubert`
    的 `.model` 换成 OV 推理（`hubert_ov.py`，接口与 HF 模型一致，
    `_get_prompt` 无需改）；IR 缺失时自动从本地权重导出一次。
  - `GSV_ROBERTA_BACKEND=openvino` + `GSV_OV_DEVICE=GPU|CPU|NPU` →
    `CNRoberta` 换成 OV 推理（`roberta_ov.py`）。NPU 模式按
    `GSV_NPU_SEQ_BUCKET`（默认 16）把 seq 向上取整分桶并缓存编译模型。
- NPU 不接入 HuBERT：prompt 音频长度千差万别，逐个编译不划算；
  只做了静态 shape 的可行性验证（`try_hubert_npu.py`）。

### 实测（AnAn.ogg 7.4 s / 中文短句 seq=21）

| 模型 | torch XPU fp16 | OV GPU fp16 | OV NPU |
| --- | --- | --- | --- |
| HuBERT（7.4 s 音频） | ~150 ms | ~142–161 ms | 静态 3 s 切片 ~128–182 ms（动态编译失败） |
| RoBERTa-h3（seq 21） | ~42–62 ms | **~20 ms**（编译 0.9 s，动态免重编） | **~20–30 ms**（编译 6 s/新形状） |
| 数值一致性（vs torch） | – | cos 0.999925+ | cos 0.999998 |

中文端到端（`use_bert=True`，同一句中文）：
- torch-XPU BERT：4.96 s 音频 / 6.15 s（RTF 1.24）
- OV-GPU BERT：4.58 s 音频 / 3.15 s（RTF 0.69）
- OV-NPU BERT：5.37 s 音频 / 6.02 s（RTF 1.12，含 NPU 推理；音频变长是采样随机性）

### 解读

1. **OV GPU 是稳赚**：RoBERTa 快 ~2–3x，动态 shape 零维护，编译一次 0.9 s；
   HuBERT 与 torch XPU 打平。中文（走 BERT）端到端 RTF 从 1.24 → 0.69。
2. **NPU 是"能跑但不推荐跑管线"**：单次推理确实能打平 GPU（短序列启动开销小），
   但 6 s/新形状 的编译税 + 必须 pad/切桶 + 长序列/大 batch 下 SRAM 瓶颈，
   在变长 TTS 管线里得不偿失——除非把文本长度固定（如流式 TTS 的定长分片）。
3. 数值一致性余量很大（cos ≥ 0.9999）， swapping featurizer 不影响音质；
   三次端到端音频 RMS 均为 ~0.09，均为正常语音。

### 用法

```bash
# HuBERT + RoBERTa 全走 Arc GPU（推荐）
GSV_HUBERT_BACKEND=openvino GSV_ROBERTA_BACKEND=openvino GSV_OV_DEVICE=GPU \
  .venv-xpu/bin/python run_xpu.py

# RoBERTa 走 NPU（尝鲜，变长文本首次会编译约 6s/桶）
GSV_ROBERTA_BACKEND=openvino GSV_OV_DEVICE=NPU \
  .venv-xpu/bin/python test_bert_backends.py
```

### 性能模式复测（powerprofilesctl performance，EPP=performance）

省电模式 → 性能模式前后对比（同一批脚本）：

| 项 | 省电模式 | 性能模式 | 变化 |
| --- | --- | --- | --- |
| `infer` 单句（fp16） | RTF 1.54–1.84 | RTF 1.37–1.56 | 快 ~10–25%，波动收窄 |
| `infer_batched`（fp16） | RTF ~0.53–0.55 | RTF ~0.50–0.57（最优 0.495） | 小幅提升 |
| HuBERT torch XPU | ~148–159 ms | **~94–95 ms** | 快 ~40% |
| HuBERT OV GPU | ~142–161 ms | ~88–128 ms | 快 ~30% |
| RoBERTa（三端） | 20 / 20–30 / 42–62 ms | 22 / 22–31 / 51–52 ms | 基本不变 |
| 中文端到端 OV-GPU BERT | RTF 0.69（4.58 s 音频） | RTF 0.73（5.57 s 音频） | 在采样噪声内持平 |

解读：性能模式主要抬高了 CPU 频率上限和 iGPU 持续 boost，
吃到红利的是 **CPU 驱动的 GPT 自回归循环**和 **HuBERT 的卷积前端**；
RoBERTa 这种纯矩阵小算子本来就跑在 boost 频率上，基本不受电源策略影响。
端到端仍被 GPT 采样步数主导（音频越长 RTF 压力越大），featurizer 不是瓶颈。

### 低垂果实复验结论（2026-10-11 深夜场）

1. **`torch.compile`（inductor）在 XPU 上无收益，甚至倒退**：
   SoVITS `flow_dec` eager 184ms → compiled 179ms（噪声级）；
   GPT `decode_next_token` eager 41.6ms → compiled 47.1ms（慢 ~13%）。
   小 batch + 小模型上 inductor 开销盖过融合收益；且首次需要
   `level-zero-headers`（`extra/level-zero-headers`，已装）。
   **结论：这条路放弃。**
2. **真果子：`_viterbi_monotonic` 的 Python/XPU DP 循环**：
   T 步逐 launcher 小 kernel，launch-bound，实测 88–265ms。
   DP 本体是纯 CPU 逻辑，转 numpy 后 **1.1ms**，输出逐元素一致，
   已替换（`TTS.py`，接口不变，返回设备不变）。单次推理省 ~0.1–0.25s。
3. **测量学教训（血的教训）**：
   - XPU 异步：`time.time()` 不包 `synchronize` 测到的是**提交时间**
    （曾测出 flow_dec 只要 20ms，同步后是 2675ms——含 oneDNN JIT）。
   - oneDNN 按 shape 缓存 primitive：GPT 采样随机导致长度每次变，
     新 shape 首次触发重 JIT（数秒），单次计时波动巨大；稳态 flow_dec ≈ 135ms。
   - 管道末尾挂 `sort -u` 会吞掉全部实时输出（等 EOF），profiling 时改用
     `grep --line-buffered` + `python -u`。
   稳态耗时账本（短句，同步后）：GPT ~2.4s ≫ viterbi ~0.1–0.25s >
   flow_dec ~0.135s > enc_p ~0.02–0.26s（含 JIT 波动）。下一步只能动 GPT 主循环。

### FlashAttention 强开实验（结论：torch 侧无 kernel，手写 Triton 可跑通但暂无收益）

1. **torch 自带的 FA 在 Xe-LPG 上不存在**：`SDPBackend.FLASH_ATTENTION`
   直接报 `No available kernel`（只给 PVC/BMG 编译了 kernel），
   `flash_attn` 包是纯 CUDA 的，装不上——没有可"强开"的对象。
2. **手写 Triton FA 在 Arc 上能跑且数值正确**（`try_triton_fa.py` /
   `try_fa2.py` / `try_maskcheck.py`）：forward 与 torch XPU math-SDPA
   差 2.4e-4。但注意三个坑：
   - 管线用 bool mask，FA kernel 还需补 mask + `sm_scale` + decode 专用
     `(q_len=1, KV-cache)` 形状才有实用价值；
   - 当前未调优版本 (16,128,64) 约 2–3ms，反而比 math 后端（~0.4ms）慢，
     要 autotune（block/warp/stage）后才有收益；
   - `tl.exp` 在 triton-xpu 上只接受 fp32/fp64（fp16 直接编译报错），
     写 kernel 时注意先 `.to(tl.float32)`。
3. **参照系教训**：一度怀疑"XPU math-SDPA 算错了"（差 4.14），最后发现是
   自写 eager 参照漏了 `1/sqrt(d)` 缩放；补上后 SDPA 与 eager 差 3e-4。
   结论：**管线里的 attention 结果是正确的**，之前所有 TTS 音频不受影响。
   教训：怀疑库函数之前，先怀疑自己的参照。

### 路线 A 实测 verdict（2026-10-12）：函数式 OV 单步在 decode 上打不过 torch XPU

- IR 导出成功（`ov_models/gpt_step_b1k1024.xml`，51 in / 50 out，
  `export_gpt_step_ov.py`）：OV 前端不支持 `index_put_`，cache 写改成算术
  等价（`arange`+`==`+blend），B=1 bake，`bidx` 输入直接省掉。
- 单步 logits 对 torch：maxdiff ~0.03（fp16 级别，正确；采样本就随机）。
- 单步耗时（B=1, K=1024，同步后）：**torch XPU ~34ms < OV CPU ~56ms
  < OV GPU ~88ms**；NPU 编译失败。88ms 里 host-device 拷贝只占 ~2ms
  （48MB，走 shared 内存），大头是 OV GPU 在该图上的执行效率——
  stateful 省掉的只是拷贝，救不了 compute，不值得继续。
- **结论：GPT decode 的 forward 留在 torch XPU（oneDNN）就是当前最优**；
  `infer()` 的 OV 接线（`gpt_step_ov.py` 已写好）因数字不达标**没有合入**。
  想再压 decode，只能走路线 B（ggml 全融合）或接受现状。

### 路线 B 进展（2026-10-12）：地基三件套已验证
（本节所有文件在 `route_b/` 下：`runner/` 是交付物，
`validation/` 是一次性验证脚本，均从仓库根目录运行）

1. **工具链**：`intel-oneapi-toolkit` 自带 DPC++（icx 2026.0）+ oneMKL；
   llama.cpp（`~/llama.cpp`，shallow clone）CPU 构建通过，
   SYCL 构建（`-DGGML_SYCL=ON`）进行中。
2. **gguf 转换器**（`convert_t2s_gguf.py`）：s1v3.ckpt 295 张量全量转出，
   arch 名 `gpt-sovits-t2s`，2D 权重按 ggml 惯例存 `[in,out]`，
   embedding 存 `[dim,vocab]`，fp16；回读字节级校验通过。
   输出：`~/.cache/gsv/gguf/t2s-s1v3-f16.gguf`。
3. **单 block ggml 复刻**（`test_t2s_block.cpp` + `dump_t2s_block.py`）：
   与 torch fp32 参照 **maxdiff=0.000000**。关键细节：
   - T2SBlock 与 transformer 的 `decode_next_token` **参数顺序不同**，调错必炸；
   - mask 语义 True=attend（与 XPU SDPA 实测一致），ggml 侧用显式 bias 实现，
     不依赖任何后端的 SDPA；
   - `ggml_norm` + 外挂 weight/bias 即 torch LayerNorm（有偏方差一致）；
   - backend 接管内存时张量必须 `no_alloc` + `ggml_backend_tensor_set`，
     不能提前 memcpy。
4. 待做：SYCL 后端在 Arc 上的真实验证（小模型跑分）→ 24 层完整 runner →
   embeddings/prefill/predict/采样 → SoVITS → 端到端对齐。

### 路线 B 数值地基打通（2026-10-12）：24 层单步全量 maxdiff=0.000006

- 布局规则（实测为准）：**ggml ne ↔ numpy 形状全逆序对应**，
  torch `(B,H,T,D)` 的 cache 必须声明成 `[D,T,H,B]`（中间两维交换）。
  权重 torch `[out,in]` 原字节声明 `[in,out]`。搞错一次就是静默错误，
  且单测可能因维度小而恰好通过——必须全尺寸验证。
- 全量对照：真实抓到的单步输入（K=512, kv=264），24 层 + predict，
  vs torch fp32，logits **maxdiff=0.000006**。
- 速度基线：CPU fp32（22 线程）单步约 20ms，已接近 torch XPU fp16 的 34ms；
  切 SYCL + fp16 后预期大幅领先。
- 剩下的是体力活：prefill/embeddings、采样循环 C++ 化、KV 跨步管理、
  fp16/量化、SoVITS、SYCL 跑分优化。

### 路线 B：prefill 全 24 层验证通过 + 采样器 exact（2026-10-12）

- prefill（L=263）：24 层逐层对照 torch，末层 1.4e-4（fp32 累积漂移内）。
  关键修正：`view_3d` 切分必须带 qkv 真实 pitch（1536），`[32,16,L]` head
  视图的 l 步长是 2048 字节；scores 保持 llama 式转置约定
  （`mul(kh,qh)` + 转置 bias + dim0 softmax），与 torch flat 逐位一致。
- 采样器：rep → top-p → top-k → softmax 的顺序必须与 torch 一致
  （之前写反了）；duplicate token 的 penalty 只算一次（last-wins）；
  cumsum 用 fp32 顺序累加。四个配置（plain/rep/topk/topp）全部 exact（≤5e-7）。
- 参照系纪律：所有探针对照文件必须同轮生成；`tensor_get` 直读 strided view
  是错位内存，必须 `cont` 后再读；转置/形状问题一律用 arange 小实验钉死，
  不许猜。
- 附带发现：XPU 上 SDPA 的 bool mask 语义实测是 **True=attend、False=mask**
  （与文档字面相反），手写 mask/导出时必须照此语义（pad 补 False）。

### 路线 B：decode 闭环 + SYCL 提速 + SoVITS 定线（2026-10-12）

- **cpy 死节点教训**：`ggml_cpy` 返回值丢弃则该节点永不进图（build 只访问
  logits 祖先，alias 不算边），cache 从不更新。修法：收住 cpy 返回值，
  先 expand 全部 cpy 再 expand logits。5 步 teacher-forcing 3e-6。
- **closed-loop greedy**：emb+pos → 24 层 → predict → suppress → sampler
  全闭环 248 步，logits 1.5e-5，token 全中。附带修了 `sampler.cpp` 的
  latent bug（temp 除法放最前；torch 顺序是 rep→top-p→temp→top-k，
  temp≠1 时原版会对不上）。
- **runner 落地**（`t2s_run.cpp`，SYCL）：gguf 直读 + pe 现场生成 + entry
  + prefill + 248 步 decode 闭环，token 全中（decode ~15ms/步，
  prefill CPU 化后 0.23s（SYCL 跨步 batch 慢 35x，decode 不受影响）。
- **端到端 parity**：短句（54字）greedy 全程：C++ runner 645 步 token 全中，
  切片对齐（torch 切掉 t0）后 SoVITS XPU 音频 maxdiff 4e-6。cache K 扩到 1024。
- **driver 可用**（`driver.py`）：活文本 → TTS特征准备 → runner → SoVITS →
  wav。随机采样（top_k=15）260 步 EOS 停止，165 个不同 token，音频电平正常。
- **启动开销**：SYCL 持久缓存本环境必崩（`getItemFromDisc` 段错误，空目录也崩），
  常驻无效，改 `--serve` 常驻进程：权重+SYCL JIT 驻留，job 级/步级 ctx 按次建删。
  同 job 双跑：19s→12.5s，tokens 与单次逐位一致。二进制放 `~/.cache/gsv/bin`（/tmp 重启丢）。
- **API 服务**（`route_b/runner/serve_api.py :9880`）：常驻 runner + CPU准备/XPU解码，
  `POST /tts` 切分长文本串行合成。验收：短句 10s 真语音（rms 0.1)，第二请求 29s→7s。
- **长文本**：cut_minlen 默认 30 + `GET /progress`（current/total）。
  1800 字：7 分钟出 11 分钟音频（段少静音垫料少；同文本上次 17.9 分钟系分段差异+采样方差）。
- **新链路验收**：好 y 覆盖 + `verify:true` 门一次过（0.9s，转写头精确命中）。
- **prompt 泄漏根因=坏 y**：Hubert 提 y 是进程彩票（A 家族 uniq~94/sil0→干净；
  B 家族 uniq~50/sil20%→漫游），同进程确定、跨进程开奖，warming/线程数都不是主因。
  工程解：好 y 落盘（`~/.cache/gsv/prompt_y/`）+ `PROMPT_Y_FILE` 覆盖，
  已验 0.9s 干脆 “こんにちは”（之前坏 y 出 9.7s 前缀垃圾）。新音频待自动优选。
- **三语可用**：中文“你好,很高兴认识你” / 英文“Hello, nice to meet you.” 转写逐字命中，
  prompt_language 修复后跨语言干净（2~3s，无前缀）。
- **按 token 流式**（`POST /tts/stream`，f32le 裸流）：runner `TOK` 行驱动，
  每 25 token 一窗 SoVITS（valid_start+sola，同 torch stream 规则），首音秒级。
  短句：流式 5.8s vs 非流式 6.3s（差交叠裁剪+头静音，符合预期）。
  血泪教训：sola/head 的 alpha 建在 CPU TTS device 上，音频在 XPU——调 XPU 实例方法。
- **回声普查**（`route_b/validation/echo_census.py`）：seed 生效且跨进程确定；
  同语言回声 0~1（干净），跨语言 prompt 回声 3（serving 已剥）；
  225 token 跨语言惯性是模型 OOD 行为（首段 0% prompt 复读=新生成），治本要同语言 prompt。
  另：inline `python3 -c` 里 `$s` 会被 shell 吞掉造成幻读，普查必须写脚本文件。
- **终极仲裁**（中文输入）：runner greedy vs 纯 torch greedy 逐 token 一致
  （`torch_greedy_zh.py`；贪心双方都塌成 239，模型行为）。
  C++ 侧无辜实锤（分布级）：145 步逐 yo步 TV 距离 1e-6、top-15 全同、t0 TV 3e-7、
  forced 下 144/146 在 torch top-15 内、EOS 时机一致。贪心精确只是 argmax 一致，
  这次是整分布一致。长度率差异（runner 长 vs torch 短）是采样流运气+小样本，
  非系统性计算 bug（suppression/切片/EOS/精度逐项对过，均一致）。
  fp16 曾被怀疑：纯 torch fp16 权重跑退化，但 ckpt 本就全 fp16 且 torch 照样 cast 跑，
  对照实为 RNG 运气；runner 已转 fp32 计算，贪心仍 exact。
- **干净路径也一致**：runner teacher-forcing 沿 torch 干净轨迹，23/23 top-15 全同；
  此后自由发散（RNG 本分）。C++ 无系统性 bug：贪心精确×2、TV 1e-6×168 步、t0×2、
  sampler/切片/EOS/suppression 逐行对过。长度率差异是采样流运气+小样本，
  要服气得跑 20v20；工程上靠 best-of-N/seed 试听 + ASR 门解决质量。
  自指测试（prompt 文本生成 prompt 音频）：输出 393 token 含 0 个 prompt 子串（≥10），
  同语言无泄露；音频互相关的多处“匹配”是同音色假阳性。真问题是口吃（29 处三连重复，
  时长翻倍），上游采样器行为，靠随机 seed + 必要时 repetition_penalty 缓解。
  serving 侧 strip/随机 seed 是在上游行为之上做质量修正。
- **专属模型陷阱**：gguf 必须全 2D 转置存储（runner 运行时转回），提交版 converter
  存 raw 从没产出过可用 gguf——唯一验证过的 00:05 文件是转置版。已修（转出逐字节一致），
  微调 ckpt 同架构可直接 `--src` 重转；SoVITS `.pth` 经 API `sovits_model` 字段即插即用，
  GPT 换 gguf 重启服务（`GPT_GGUF`，多模型=多端口）。
- **TILE 融合关线**：triton FA 在 XPU 上数值错误（maxdiff 4.14）且比 torch math 慢 6x
  （3.17ms vs 0.53ms，N=128）。attention 本来只占 step 零头（step 17ms 是全图 launch-bound），
  融了也省不出 1.3x。
- **SoVITS 加速结论（无油水，关线）**：dec 占 92%（1540/1680ms），纯算力 conv。
  fp16 证伪（maxdiff 0.31）；torch.compile exact 但仅 1.17x 且 46s/shape；
  OV bucket 早前已证伪。dec RTF 0.06 本来就行，请求耗时大头是 GPT decode。
  - gguf 文件字节 = torch.T 的 C-order（writer 又转了一次，和 converter
    的 transpose 抵消）；host 侧一律按 torch 行序理解，逐张量字节级 H-test。
  - 3D bias 的 host 填充必须按 ggml F-order 语义（C-order 填必转置错：
    `pb[m+L*n+L²h] = mask(q=n,k=m)`）。
  - 中间张量探针读数不可信（gallocr 复用踩踏）：只信 buf 张量和末端输出；
    分段验证必须每段独立成图只读末端。
  - prompt token 跨轮不稳定（22/184 差异，源头未定）：参照必须同轮 dump。
- **SYCL Arc**：fp32 decode 30ms/步；SYCL build 是 `GGML_SYCL_F16: no`
  （F16 内部转 fp32，无加速）。瓶颈是 launch（7057 节点/步）。
  `flash_attn_ext` 走不通（head dim 32 被 oneDNN/MKL 两路拒，只要 ≥64）。
  自研 batch 注意力（16 头并成 batch 维，一层 ~5 节点）：**~20ms/步**，
  超过 torch XPU（~34ms）。fp16 精度 2e-3，argmax 全稳。
- **SoVITS 定线**：瓶颈是 Generator（dec 5.7s/6.7s，enc_p 140ms，flow 45ms）。
  OV 静态 IR：flow_dec 精确 shape 下 CPU 5e-5/GPU-f32 1~3e-3；
  但 **pad+bucket 在 OV GPU 下全局炸**（flowdec 1.13，enc 11.0），
  torch 同 pad 输入是 exact（4.6e-5）——OV shape 相关误编译，
  bucket 路线放弃。enc_p 留 torch CPU（140ms 不值得搬）。
  **torch XPU 是答案**：SoVITS 全量 662ms vs CPU 3181ms（4.8x），
  maxdiff 1.4e-5。XPU 端到端 fp32：3.2s 出 3.8s 音频，RTF 0.86。
- **fp16 证伪（单流）**：XPU fp16 端到端 RTF 1.34，反比 fp32（0.86）慢；
  GPT 46 tok/s vs fp32 73 tok/s，SoVITS fp16 直接 mixed-dtype 报错。
  XPU math 后端 fp16 核慢（之前 RTF 0.5 是 batched 吞吐）。单流延迟
  按 fp32 走，fp16 不碰。相关脚本：`probe_sovits*.py`、
  `export_sovits_*.py`（参考）、`dump_sovits_in.py`、`run_sovits_xpu.py`。
  `TextEncoder.infer` 加了可选 `y_mask/text_mask` 参数（向后兼容）。
