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
