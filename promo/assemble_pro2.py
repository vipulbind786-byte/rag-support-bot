import subprocess

P = "/mnt/v/ai-sandbox/workspace/workspace/projects/Freelance/projects/rag-support-bot/promo"
parts = ["intro_full.mp4", "shot1.mp4", "shot2.mp4", "shot3.mp4", "shot4.mp4", "shot5.mp4", "endcard.mp4"]

for p in parts:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p, "-vf", "fps=30",
                    "-c:v", "h264", "-pix_fmt", "yuv420p", "-r", "30", "n_" + p], check=True)

np = ["n_" + p for p in parts]
durs = []
for p in np:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", p], capture_output=True, text=True)
    durs.append(float(r.stdout.strip()))
print("durs:", [round(d, 3) for d in durs])

xf = 0.5
filters = []
prev = "0:v"
cum = durs[0]
for i in range(1, len(np)):
    nxt = f"x{i}"
    filters.append(f"[{prev}][{i}:v]xfade=transition=dissolve:duration={xf}:offset={cum-xf:.3f}[{nxt}]")
    prev = nxt
    cum += durs[i] - xf

cmd = ["ffmpeg", "-y", "-v", "error"]
for p in np:
    cmd += ["-i", p]
cmd += ["-filter_complex", ";".join(filters), "-map", f"[{prev}]",
        "-c:v", "h264", "-pix_fmt", "yuv420p", "launch-teaser-pro.mp4"]
subprocess.run(cmd, check=True)
print("built launch-teaser-pro.mp4 expected", round(cum, 2))
