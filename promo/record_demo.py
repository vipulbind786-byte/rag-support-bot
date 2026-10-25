from playwright.sync_api import sync_playwright
import os

OUT = "/mnt/v/ai-sandbox/workspace/workspace/projects/Freelance/projects/rag-support-bot/promo"
URL = "https://vrxdr-rag-support-bot.static.hf.space/index.html"
probes = [
    "what is the grand total of invoice 0142",
    "which invoice has the higher grand total",
    "how many days to return a product",
    "best pizza in pune",
]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        record_video_dir=OUT + "/video",
        record_video_size={"width": 1920, "height": 1080},
    )
    page = ctx.new_page()
    page.goto(URL, wait_until="networkidle")
    page.wait_for_timeout(1200)
    inp = page.query_selector("input[type=text], input[type=search], textarea")
    btn = page.query_selector("button")
    for q in probes:
        inp.fill("")
        inp.fill(q)
        page.wait_for_timeout(700)
        btn.click()
        page.wait_for_timeout(2600)
    stress = page.query_selector("button:has-text('stress test')")
    if stress:
        stress.click()
        page.wait_for_timeout(7000)
    page.wait_for_timeout(1000)
    video = page.video
    ctx.close()
    browser.close()
    path = video.path()
    print("video:", path)
    print("size:", os.path.getsize(path))
