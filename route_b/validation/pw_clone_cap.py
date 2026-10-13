"""抓 webui 克隆请求的浏览器发包：upload + queue/join payload。只取证，不求结果。"""
import sys, json
from playwright.sync_api import sync_playwright

URL = "http://localhost:7860/"
PROMPT_WAV = "/home/swordreforge/GSV-TTS-Lite/32uwgy.wav"
PROMPT_TEXT = open("/tmp/opencode/32trans.txt").read().strip() if __import__("os").path.exists(
    "/tmp/opencode/32trans.txt") else "测试"

captured = []


def on_request(req):
    try:
        body = req.post_data
    except Exception:
        body = None
    if req.method == "POST" and ("/upload" in req.url or "/join" in req.url or "/push" in req.url):
        entry = {"url": req.url.split("localhost:7860")[-1], "nbytes": len(body) if body else 0}
        if body and len(body) < 3000:
            entry["body"] = body[:3000]
        elif body:
            entry["body_head"] = body[:300]
        captured.append(entry)
        print("CAP:", entry["url"], entry["nbytes"], flush=True)


with sync_playwright() as p:
    br = p.chromium.launch(executable_path="/usr/bin/chromium",
                           args=["--no-sandbox", "--disable-dev-shm-usage"])
    pg = br.new_page()
    pg.on("request", on_request)
    pg.goto(URL, wait_until="networkidle", timeout=60000)
    pg.get_by_text("一次性克隆", exact=False).first.click(timeout=10000)
    # 参考音频上传
    file_input = pg.locator("input[type=file]").nth(1)  # 第0个是变声页的（隐藏tab也有input）
    n_inputs = pg.locator("input[type=file]").count()
    print("file inputs:", n_inputs, flush=True)
    # 展开后属于 TTS tab 的 input：逐个试 set_input_files，取成功的
    done = False
    for i in range(n_inputs):
        try:
            pg.locator("input[type=file]").nth(i).set_input_files(PROMPT_WAV, timeout=5000)
            print(f"upload ok on input[{i}]", flush=True)
            done = True
            break
        except Exception as e:
            print(f"input[{i}] skip: {str(e)[:80]}", flush=True)
    pg.get_by_label("参考文本", exact=False).fill(PROMPT_TEXT)
    pg.get_by_label("文本", exact=True).fill("你好。")
    pg.get_by_role("button", name="合成", exact=True).click()
    pg.wait_for_timeout(90000)
    br.close()

open("/tmp/opencode/pw_cap.json", "w").write(json.dumps(captured, ensure_ascii=False, indent=1)[:6000])
print("SAVED", len(captured), flush=True)
