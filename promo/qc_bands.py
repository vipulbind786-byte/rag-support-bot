import glob, math
from PIL import Image

BG = (11, 17, 32)

def bands(img):
    px = img.load()
    w, h = img.size
    rows = []
    for y in range(h):
        cnt = 0
        for x in range(0, w, 4):
            r, g, b = px[x, y][:3]
            if abs(r - BG[0]) + abs(g - BG[1]) + abs(b - BG[2]) > 30:
                cnt += 1
        rows.append(cnt)
    bands = []
    y = 0
    while y < h:
        if rows[y] > 2:
            y0 = y
            while y < h and rows[y] > 2:
                y += 1
            top = max(rows[y0:y]) * 4 / w
            bands.append((y0, y, top))
        else:
            y += 1
    return bands

frames = sorted(glob.glob("frames/f_*.png"))
prev = None
for f in frames:
    n = int(f.split("_")[1].split(".")[0])
    t = (n - 1) * 0.5
    img = Image.open(f).convert("RGB")
    bs = bands(img)
    print(f"t={t:4.1f}s  bands={len(bs)}  " + " | ".join(f"y{a}-{b} lum{c:.2f}" for a, b, c in bs))
