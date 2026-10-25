import easyocr, glob, re, json

toks = ["5,60,244", "3,93,160", "14 days", "pizza", "abstain", "pass",
        "proof 1", "proof 2", "proof 3", "proof 4", "proof 5",
        "live demo", "no cloud", "zero cost", "your documents", "your answers",
        "rag support bot", "32", "bm25", "rrf", "cited answer",
        "static.hf.space", "never leaves the premises"]

r = easyocr.Reader(['en'], gpu=False, verbose=False)
timeline = {t: [] for t in toks}
for f in sorted(glob.glob('procheck2/q_*.png')):
    idx = int(re.search(r'(\d+)\.png', f).group(1))
    t = (idx - 1) / 4
    txt = " ".join(x[1].lower() for x in r.readtext(f, detail=1) if x[2] > 0.3)
    squash = txt.replace(" ", "").replace("-", "")
    for tk in toks:
        if tk in txt or tk.replace(" ", "") in squash:
            timeline[tk].append(round(t, 2))

with open("pro_ocr_timeline.json", "w") as fh:
    json.dump(timeline, fh, indent=1)
for tk in toks:
    v = timeline[tk]
    print(f"{tk:24} {'t=%s..%s (%d frames)' % (v[0], v[-1], len(v)) if v else 'NOT FOUND'}")
print("DONE")
