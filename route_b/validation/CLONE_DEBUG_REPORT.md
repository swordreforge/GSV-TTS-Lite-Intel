# 语音克隆崩溃复盘报告

> 时间跨度：2026-10-13 深夜 ~ 2026-10-14
> 范围：Web 端语音克隆（上传参考音频 one-shot clone）100% 报错 / 服务反复奔溃
> 结论：**根因 = XPU 上计算 speaker embedding `ge` 导致 SoVITS decode native SEGV；已修复并验收 0/9 → 9/9**
> 服务状态：由用户手动停止（systemd 显示 20:40 Stopped），非崩溃

---

## 1. 症状（原始现场）

- Web 端做语音克隆：**无论文本长短、怎么采样，几乎 100% 报错**（用户原话）。
- 默认 TTS（AnAn 内置参考）一直正常，只有克隆（上传参考音频）必挂。
- 崩溃表现随视角不同而不同：
  - 浏览器：波形框显示「错误」。
  - API 直连：`RemoteDisconnected` / `HTTP 500` / `Connection refused`。
  - 服务端：native `Segmentation fault` / `Aborted`（journal 里 7+ 次 `status=11/SEGV`、`status=6/ABRT`）。
- 伴随：`corrupted double-linked list`（glibc heap 报错），以及 Level Zero `UR_RESULT_ERROR_UNKNOWN`。

---

## 2. 排查过程中**被证伪**的假设（附证据，避免重复劳动）

| # | 假设 | 排除方法 | 结论 |
|---|---|---|---|
| 1 | 前端发了畸形参数 | playwright 抓 `/gradio_api/upload` + `/queue/join` 全文 | join payload 字段/文件/长度全对，**前端无罪** |
| 2 | phone id 越界（embedding OOB） | `phone_range.py`：算 ja/zh 各组合 phone 值域 | max=323 < 732，**不越界** |
| 3 | 毒 token / 空 seq / 失控 codes | 存盘 `last_tok_*.bin` + poison/runaway/empty 三重 guard | 崩溃现场 token 值域干净（0~1021+EOS），**非输入越界** |
| 4 | 显存压力 / OOM | 解码前读 `torch.xpu.memory_allocated` | 仅 770MB，**无压力** |
| 5 | 并发（多点几次） | overlap 双请求压测 | 并发确实崩，但**单人 serial 也崩** → 非唯一因 |
| 6 | 多线程 worker 问题 | 崩盘 tokens + 3 worker 线程并发解 | 全 clean，**排除** |
| 7 | 长进程老化/累积泄漏 | 单进程 50 连解 | 全 clean，**排除** |
| 8 | 常驻 runner 跨 job 状态泄漏 | 5 连打触发"1 成 4 败" | 有相关性（空生成），但**不是 SEGV 根因** |
| 9 | luming 外来模型 / hybrid | 单独 :9881 实例 + hybrid 测试 | 是**另一独立问题**（见 §6），与本次崩溃无关 |

> 关键误导点：所有隔离测试我都习惯用 **CPU tts** 算 `ge`，而服务端一直用 **XPU tts** 算 `ge`——这个差异让隔离侧"从不崩"、服务端"必崩"，长期掩盖了真因。

---

## 3. 根因（实锤）

**`tts_x._prepare_sovits_resources()`（XPU）算出的 speaker embedding `ge`，喂进 SoVITS `decode()` 会触发 native SEGV。**

### 决定性 A/B 实验（`route_b/validation/ge_ab.py`）

唯一变量：`ge` 从 CPU tts 取 vs 从 XPU tts 取。

```
===== GE=cpu =====
[cpu] ge device=cpu
[cpu] DEC0 OK (1, 1, 12800)
[cpu] DEC1 OK (1, 1, 19200)
[cpu] DEC2 OK (1, 1, 25600)
[cpu] CLEAN
===== GE=xpu =====
[xpu] ge device=xpu:0
Fatal Python error: Segmentation fault
```

- CPU ge → 3/3 干净
- XPU ge → 必崩

### 为什么这解释了一切

- 默认 TTS（AnAn + laffey spk）**偶尔**能跑，是因为默认链路里 ge 的路径/时序恰好没踩中（或踩中概率低）。
- 我写的每一个隔离复现脚本（`exact_repro`、`coexist`、`thrtable`、`aging`、`ge32`）都用 CPU tts 算 ge → **必然全部通过**，造成"无法复现"的假象。
- 服务端 `_synth` / `_synth_stream` / `/vc` 一直用 `tts_x._prepare_sovits_resources` → **必然崩**。

### 崩溃点（faulthandler 固定落点）

```
File "route_b/runner/serve_api.py", line 387 in _one_chunk
    codes = torch.from_numpy(seq).unsqueeze(0).unsqueeze(0).to("xpu")
    ...
    audio, _ = svx.decode(codes, ph2, ge.to("xpu"), cuda_graph=False, noise_scale=...)
```

即 `ge.to("xpu")` / `decode` 处；C 栈落在 `libtorch_cpu.so` 的 `to_copy` / `native::to`。

---

## 4. 修复（4 层，均已提交并验收）

### 4.1 根治 SEGV：ge 一律在 CPU 计算（`819a44e`）

新增 `_prepare_sv(tts, tts_x, sv_path, spk)`：
- `svx` 仍用 XPU 模型（解码要快）
- `ge` 改用 **CPU tts** 计算（`tts._prepare_sovits_resources`），再 `.to("xpu")`

三处调用点统一替换：`_synth` / `_synth_stream` / `/vc`（VC 顺带修正了原先 CPU ge 被丢弃的问题）。

### 4.2 参考语言错配（`bb1e9a3`）

现象：参考音频是中文却选了 `参考音频语言=ja` → ja 前端（JPCommonLabel）啃中文报 `No phoneme` → 条件全错 → 生成 9 码即 EOS「摆烂」。

- 实测：`plang=ja`（错配）3/6 成功；`plang=zh`（正确）6/6。
- 修法：webui 上传时**始终自动转写并检测语言**，覆盖用户手选。

### 4.3 空生成重试（`337d58b`）

`attempts = 1 + max_retries`（默认 3 次），不再受 `verify` 开关限制；空生成无条件换 seed 重试。

### 4.4 输出响度归一化（`cc6267e`）

现象：克隆输出比默认小 ~25 dB（克隆 peak 0.037 vs 默认 0.651）。

- 原因：该 prompt y 偏糊 → 模型输出能量低；上游 `TTS.py` 只在 `>1` 时向下压，**从不向上归一化**。
- 修法：`/tts`、`/vc` 输出统一 RMS→-20 dBFS + 峰值保护（不削波）；`GSV_TARGET_DBFS=0` 可关。
- 修复后：克隆与默认均稳定 -20.0 dBFS。

---

## 5. 验收

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 原样前端参数 9 连（上传 32uwgy + ja 目标） | **0/9**（SEGV / RemoteDisconnected / 500） | **9/9 成功** |
| 服务存活 | 反复死（NRestarts 递增） | 全程不倒 |
| 输出内容 | 空/崩 | ASR = `こんにちは`，rms 0.099 |
| 克隆音量 | -40.6 dBFS | -20.0 dBFS（与默认齐平） |

验证脚本：`route_b/validation/repro_frontend.py`（原样前端参数复现）、`five_shot.py`、`ge_ab.py`、`isolate_cachedy.py`。

---

## 6. 本次顺带处理的其它问题

- **luming v2-pro-plus 模型实测**（`b310c3b`）：ckpt→gguf 295 tensors 正常；短句全对，30 字整句早停；`cut_minlen=10` 细切后结构全对但生僻词含糊（训得浅，372 steps 的上限）；**hybrid（底座 GPT + luming SoVITS）会崩主服务 → 外来 .pth 隔离到独立实例，永不进主实例**。
- **OV Hubert 确定性**（`21524dd`）：5 音频 ×2 独立进程全 0 漂移（torch 版跨进程抽签 0~65 乱飘）→ 服务已切 OV + 后端标记缓存。
- **E2E profile**（`ca0f6b1`）：GPT decode 稳态 19ms/step、SoVITS 热态 0.33 RTF，无廉价大头可砍。
- **WebUI 增强**（`a4485fc`、`7ac399e`、`cc6267e` 等）：一次性克隆上传、`cut_minlen`、`sovits_model`、跨 tab 联动、PCM_16（修 Firefox 无法播 float32）、whisper 常驻。
- **服务持久化**（`b7c26af`）：systemd user unit `gsv-api` / `gsv-webui` + linger。

---

## 7. 剩余已知问题 / 未做

1. **32uwgy 的 y（329）质量一般**：resyn 0.56 偏低，生僻词含糊、偶发空生成。建议换干净参考音频重新抽签选 y。
2. **跨语言（ja 目标 + zh 参考）偶发摆烂**：同语言目标稳定得多。
3. **流式端点（/tts/stream）未做响度归一化**：实时无法回溯，需客户端自行处理。
4. **XPU ge 为何会崩**：根因已定位到"设备"这一层并绕过，但底层机理（SyCL/oneDNN 在 ref encoder 上的行为）未深挖；属已知规避。
5. **luming 长句**：需 `cut_minlen=10` 细切。

---

## 8. 经验教训

1. **隔离复现必须对齐"非显而易见的环境变量"**：这次是 `TTS(device=...)` 的选择（CPU vs XPU）。我所有隔离脚本都隐式用了 CPU tts，等于改掉了自变量，导致长期"无法复现"。做 A/B 时，先列出所有隐式选择。
2. **native 崩溃先用 `faulthandler.enable()` + 固定落点行号**：比反复看业务日志有效得多。
3. **别在证据不足时下结论**：中途"毒 token / 并发 / 前端畸形"几个假设都被后续证据推翻；每次推翻都要回退假设，而不是叠加解释。
4. **输入侧证伪要穷尽**：值域、长度、空、失控、哈希——全过之后才应该怀疑"环境/设备状态"。
5. **一个 bug 常伴生多个问题**：崩溃修好后，才露出"空生成""音量小"这些被崩溃掩盖的次级问题。

---

## 附：本次相关提交（时间倒序）

```
cc6267e output loudness normalization (RMS -20dBFS, peak-protected); GSV_TARGET_DBFS=0 disables
03b0768 docs: clone fix verified 0/9 -> 9/9
337d58b empty-generation: always retry (3 attempts), not gated by verify
91e4239 docs: prompt-language mismatch collapse + autodetect fix
bb1e9a3 webui: always auto-detect ref language on upload (fixes ja-on-zh collapse)
1b26689 docs: ge-on-XPU segfault root cause + isolation A/B tools
819a44e ROOT CAUSE FIX: ge computed on CPU (XPU ge segfaults SoVITS decode)
0a2217e xpu lock scaffold + phone-range validation
1dbf8ad webui: log full /tts payload per request
a503ca9 pre-decode xpu memory log
7a7da5a webui: resident whisper model (no reload per click)
a73fdc7 tokbin persist + poison-token guard
09a448d runaway codes guard (>600 refuse+retry) + pre-decode length log
c29200d webui: auto-transcribe empty prompt text (parity with VC tab)
702ff07 empty-generation guard (loud 500), crash verdict docs, playwright+bisect tools
123495e serve_api: faulthandler for native crash forensics
7ac399e webui: PCM_16 convert for Firefox playback
3726d94 retry path xpu.empty_cache (luming verify-storm fix)
a4485fc webui: one-shot clone upload, cut_minlen, sovits_model, cross-tab voice refresh
b310c3b luming v2pp verdict: short-ok, long needs chunking, hybrid crashes main - keep isolated
```

新增验证工具：`ge_ab.py`、`isolate_cachedy.py`、`repro_frontend.py`、`five_shot.py`、`phone_range.py`、`pw_clone_cap.py`、`empty_repro.py`、`exact_repro.py`。
