import subprocess

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

# centers verified by OCR of the settled stills
shots = [
    ("settled_5.png",  2.4, 1.32, 760, 430, "PROOF 1", "Grand total computed in code - 0 LLM calls"),
    ("settled_8.png",  2.4, 1.30, 800, 430, "PROOF 2", "Two invoices in one PDF - scoped correctly"),
    ("settled_11.png", 2.4, 1.28, 960, 420, "PROOF 3", "Policy v2.0 supersedes the older 30-day rule"),
    ("settled_13.png", 2.4, 1.30, 760, 430, "PROOF 4", "Off-topic query - abstains, escalates to human"),
    ("settled_19.png", 2.6, 1.22, 960, 630, "PROOF 5", "32 adversarial probes - 32 PASS"),
]

for i, (src, dur, zend, cx, cy, badge, cap) in enumerate(shots, 1):
    d = int(dur * 30)
    vf = (
        f"zoompan=z='1+({zend}-1)*on/{d}':"
        f"x='{cx}-(1920/zoom/2)':y='{cy}-(1080/zoom/2)':d={d}:s=1920x1080:fps=30,"
        f"drawtext=fontfile={MONO}:text='{badge}':fontcolor=f87171:fontsize=32:x=46:y=46:"
        f"alpha='if(gt(t\\,0.2)\\,min(1\\,(t-0.2)/0.3)\\,0)',"
        f"drawtext=fontfile={FONT}:text='{cap}':fontcolor=ffffff:fontsize=42:"
        f"x=(w-text_w)/2:y=985:alpha='if(gt(t\\,0.5)\\,min(1\\,(t-0.5)/0.4)\\,0)',"
        f"drawtext=fontfile={MONO}:text='vrxdr-rag-support-bot.static.hf.space':"
        f"fontcolor=38bdf8:fontsize=26:x=(w-text_w)/2:y=1042:alpha=0.5"
    )
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-vf", vf,
                    "-c:v", "h264", "-pix_fmt", "yuv420p", f"shot{i}.mp4"], check=True)
    print("built shot", i)
