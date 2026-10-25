from PIL import Image
import glob, easyocr

BG = (11, 17, 32)
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

def bands(img):
    px = img.load(); w, h = img.size
    rows = []
    for y in range(h):
        cnt = 0
        for x in range(0, w, 4):
            r, g, b = px[x, y][:3]
            if abs(r - BG[0]) + abs(g - BG[1]) + abs(b - BG[2]) > 30:
                cnt += 1
        rows.append(cnt)
    out = []
    y = 0
    while y < h:
        if rows[y] > 2:
            y0 = y
            while y < h and rows[y] > 2: y += 1
            out.append((y0, y, max(rows[y0:y]) * 4 / w))
        else: y += 1
    return out

frames = sorted(glob.glob("hybrid_frames/f_*.png"))
prev = None
for f in frames:
    n = int(f.split("_")[1].split(".")[0]); t = (n - 1) * 0.5
    img = Image.open(f).convert("RGB")
    bs = bands(img)
    if bs != prev:
        print(f"t={t:4.1f}s bands={len(bs)} lum_max={max(c for _, _, c in bs):.2f}")
        prev = bs
