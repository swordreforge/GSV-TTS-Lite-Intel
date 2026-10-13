"""5 连打原样 payload，测崩率（区分确定性 vs 间歇）。"""
import json, urllib.request, time, sys

base = "http://localhost:9880"
UP = "/home/swordreforge/.cache/gsv/uploads/prompt_1791908424222_32uwgy.wav"
PT = open("/tmp/opencode/32trans.txt").read().strip()

payload = {"text": "こんにちは。", "language": "ja", "prompt_language": "ja",
           "top_k": 15, "verify": False, "cut_minlen": 30,
           "prompt_audio": UP, "prompt_text": PT, "spk": UP}

for k in range(5):
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(base + "/tts", data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = r.read()
        print(f"TRY{k}: OK bytes={len(d)} {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print(f"TRY{k}: FAIL {repr(e)[:100]}", flush=True)
        time.sleep(2)
    # 等健康恢复再打下一发
    for _ in range(30):
        try:
            urllib.request.urlopen(base + "/healthz", timeout=3).read()
            break
        except Exception:
            time.sleep(5)
print("DONE", flush=True)
