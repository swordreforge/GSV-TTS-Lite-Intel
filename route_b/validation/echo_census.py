"""回声普查（真文件）：各 seed 头部与 prompt 尾的最长续写匹配。"""
import numpy as np
import os

y = np.fromfile("/tmp/opencode/ja_stoch/y.bin", np.int64)
files = {
    "ja seed1234 full": "/tmp/opencode/ja_stoch/tokens.bin",
    "ja seed7 cap": "/tmp/opencode/echo2_s7.bin",
    "ja seed42 cap": "/tmp/opencode/echo2_s42.bin",
    "ja seed99 cap": "/tmp/opencode/echo2_s99.bin",
    "ja seed2024 cap": "/tmp/opencode/echo2_s2024.bin",
    "ja seed1234 cap": "/tmp/opencode/echo2_s1234.bin",
}
yzh = np.fromfile("/tmp/opencode/zh_an/y.bin", np.int64)
files["zh seed1234 full"] = "/tmp/opencode/zh_an/tokens.bin"

for name, p in files.items():
    if not os.path.exists(p):
        print(name, "MISSING");
        continue
    t = np.fromfile(p, np.int64)
    c = t[1:]
    ref = yzh if name.startswith("zh") else y
    best = 0
    for L in range(1, min(40, len(c), len(ref))):
        if list(c[:L]) == list(ref[-L:]):
            best = L
    print(f"{name}: sampled {len(t)} echo_len {best} head {c[:6].tolist()}")
