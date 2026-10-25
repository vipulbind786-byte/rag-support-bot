import subprocess, glob, easyocr, re

P = "/mnt/v/ai-sandbox/workspace/workspace/projects/Freelance/projects/rag-support-bot/promo"
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

# Locate time windows for each key moment in demo.mp4
marks = {}
for f in sorted(glob.glob(P + "/demo_check/d_*.png")):
    t = int(re.search(r"d_(\d+)", f).group(1)) - 1
    txt = " ".join(x[1].lower() for x in reader.readtext(f))
    if "5,60,244" in txt: marks.setdefault("total", []).append(t)
    if "14 days" in txt: marks.setdefault("policy", []).append(t)
    if "pizza" in txt: marks.setdefault("pizza", []).append(t)
    if "32" in txt and ("pass" in txt or "ok" in txt or "✓" in txt or "done" in txt): marks.setdefault("stress_rows", []).append(t)
print(marks)
