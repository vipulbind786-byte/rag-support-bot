# Launch Teaser — Build Script (v2, vision-verified)

Path: `promo/VIDEO_BUILD_SCRIPT.md`
Repo: `/mnt/v/ai-sandbox/workspace/workspace/projects/Freelance/projects/rag-support-bot`
Goal: build `promo/launch-teaser.mp4` (or a richer cut) in a fresh agent session, with
visual QC. Hand this whole file to the new chat as context.

## Facts to use (all verified 25 Oct 2026 — do NOT invent others)
- Product: Offline RAG support bot. Hybrid retrieval (dense vectors + BM25 + TF-IDF,
  RRF fusion), deterministic numeric answers in code, local 0.5B LLM (qwen2.5 via
  Ollama, temp 0) for phrasing, ChromaDB, cross-encoder reranker (bge-reranker-base).
- Metrics: 14/14 golden eval, 32/32 brutal stress suite. Deterministic answers for
  totals/GST/comparisons/policy versions/entity attributes. Abstains when corpus
  cannot answer. Injection memos neutralized at ingest.
- Demo link (LIVE, verified HTTP 200): https://vrxdr-rag-support-bot.static.hf.space/index.html
  (runs entirely in the browser, no server).
- Repo: https://github.com/vipulbind786-byte/rag-support-bot
- Sample probe that shows the killer feature: "how many days to return a product"
  -> "14 days (policy v2.0, March 2026 — supersedes the older 30-day rule)".
  Off-topic probe: "best pizza in pune" -> abstains, escalates.

## KNOWN BUG in current v1 (fix in v2)
The 0-3s phase shows "Your documents." and "Your answers." SIMULTANEOUSLY (both
enabled, overlapping at 1.2-3.0s). v2 must show them SEQUENTIALLY with fade in/out
and ZERO temporal overlap.

## Storyboard (24s total) — exact timings
Colors: background #0b1120, primary text #ffffff, accent #38bdf8, success #34d399,
alert #f87171, muted #9ca3af. Fonts: DejaVu Sans Bold (headlines), DejaVu Sans Mono
(details). Every text card fades in 0.4s and fades out 0.4s. No two cards overlap.

| t (s)   | On screen |
|---------|-----------|
| 0.0-2.5 | "Your documents." |
| 2.5-4.5 | "Your answers." |
| 4.5-7.0 | "No cloud. No API keys. Zero cost." |
| 7.0-11.5| title: "Offline RAG Support Bot"; pipeline line: "chunks -> BM25 + TF-IDF -> RRF -> rerank -> cited answer" |
| 11.5-16 | Q: "What is the return window?" then (t 12.5+) A: "14 days - policy v2.0, cited - not guessed" |
| 16-18.5 | Q: "What is your blueberry policy?" then (t 17+) A: "Not in the corpus. Escalating to a human." |
| 18.5-21.5| headline: "32/32 brutal stress suite"; line: "multi-invoice scoping / injection defense / Hinglish" |
| 21.5-24 | "Live demo (runs in your browser)" + "vrxdr-rag-support-bot.static.hf.space" + "Data never leaves the premises." |

## Option A — pure ffmpeg build (no recording, $0, reproducible)
Run in the repo root. Uses per-card fade via the drawtext `alpha` expression, so
overlap is structurally impossible.

```bash
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
MONO=/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf
OUT=promo/launch-teaser.mp4
BG=0x0b1120

ffmpeg -y -v error -f lavfi -i "color=c=$BG:s=1920x1080:d=24" -t 24 -vf "\
drawtext=fontfile=$FONT:text='Your documents.':fontcolor=white:fontsize=96:x=(w-text_w)/2:y=(h-text_h)/2:alpha='clip((t-0.3)/0.4,0,1)*clip((2.5-t)/0.4,0,1)',\
drawtext=fontfile=$FONT:text='Your answers.':fontcolor=38bdf8:fontsize=96:x=(w-text_w)/2:y=(h-text_h)/2:alpha='clip((t-2.7)/0.4,0,1)*clip((4.5-t)/0.4,0,1)',\
drawtext=fontfile=$FONT:text='No cloud. No API keys. Zero cost.':fontcolor=white:fontsize=64:x=(w-text_w)/2:y=(h-text_h)/2:alpha='clip((t-4.7)/0.4,0,1)*clip((7.0-t)/0.4,0,1)',\
drawtext=fontfile=$FONT:text='Offline RAG Support Bot':fontcolor=38bdf8:fontsize=80:x=(w-text_w)/2:y=130:alpha='clip((t-7.2)/0.4,0,1)*clip((11.5-t)/0.4,0,1)',\
drawtext=fontfile=$MONO:text='chunks - BM25 + TF-IDF - RRF - rerank - cited answer':fontcolor=9ca3af:fontsize=44:x=(w-text_w)/2:y=660:alpha='clip((t-7.8)/0.4,0,1)*clip((11.5-t)/0.4,0,1)',\
drawtext=fontfile=$MONO:text='What is the return window?':fontcolor=white:fontsize=50:x=(w-text_w)/2:y=460:alpha='clip((t-11.7)/0.4,0,1)*clip((16.0-t)/0.4,0,1)',\
drawtext=fontfile=$FONT:text='14 days - policy v2.0, cited - not guessed':fontcolor=34d399:fontsize=46:x=(w-text_w)/2:y=580:alpha='clip((t-12.5)/0.4,0,1)*clip((16.0-t)/0.4,0,1)',\
drawtext=fontfile=$MONO:text='What is your blueberry policy?':fontcolor=white:fontsize=50:x=(w-text_w)/2:y=740:alpha='clip((t-16.2)/0.4,0,1)*clip((18.5-t)/0.4,0,1)',\
drawtext=fontfile=$FONT:text='Not in the corpus. Escalating to a human.':fontcolor=f87171:fontsize=46:x=(w-text_w)/2:y=860:alpha='clip((t-17.0)/0.4,0,1)*clip((18.5-t)/0.4,0,1)',\
drawtext=fontfile=$FONT:text='32/32 brutal stress suite':fontcolor=38bdf8:fontsize=82:x=(w-text_w)/2:y=460:alpha='clip((t-18.7)/0.4,0,1)*clip((21.5-t)/0.4,0,1)',\
drawtext=fontfile=$MONO:text='multi-invoice scoping - injection defense - Hinglish':fontcolor=9ca3af:fontsize=42:x=(w-text_w)/2:y=600:alpha='clip((t-19.0)/0.4,0,1)*clip((21.5-t)/0.4,0,1)',\
drawtext=fontfile=$FONT:text='Live demo (runs in your browser)':fontcolor=white:fontsize=58:x=(w-text_w)/2:y=420:alpha='clip((t-21.7)/0.4,0,1)*clip((24.0-t)/0.4,0,1)',\
drawtext=fontfile=$MONO:text='vrxdr-rag-support-bot.static.hf.space':fontcolor=38bdf8:fontsize=50:x=(w-text_w)/2:y=560:alpha='clip((t-22.0)/0.4,0,1)*clip((24.0-t)/0.4,0,1)',\
drawtext=fontfile=$MONO:text='Data never leaves the premises.':fontcolor=34d399:fontsize=42:x=(w-text_w)/2:y=700:alpha='clip((t-22.0)/0.4,0,1)*clip((24.0-t)/0.4,0,1)'" \
-c:v h264 -pix_fmt yuv420p "$OUT"
```

NOTE: ffmpeg drawtext cannot render Unicode arrows directly; use hyphens "->"-style
plain words or ffmpeg's `expansion` — keep ASCII, OCR QC will flag mojibake.

## Option B — hybrid cut (recommended for the real launch)
Best-performing structure: 8s text intro (Option A phases 0-7) + 14s SCREEN
RECORDING of the live Space demo + 2s end card. Clients trust actual UI more than
kinetic text.

Recording steps (Windows host, OBS or ShareX, 1080p, 30fps):
1. Open https://vrxdr-rag-support-bot.static.hf.space/index.html
2. Type into the demo box, one probe per shot, pause ~2s before/after each:
   a) "what is the grand total of invoice 0142"  -> shows ₹5,60,244.00 computed, cited
   b) "which invoice has the higher grand total"  -> comparison sentence
   c) "how many days to return a product"         -> "14 days ... supersedes 30-day rule"
   d) "best pizza in pune"                        -> abstains: "escalate to a human"
3. Click the built-in "Run stress test" button -> 32/32 rows scroll (film 5s of it).
4. Cut the best 14s, place between intro and end card.
5. Optional: overlay the demo URL lower-third during recording.

Recording on WSL instead of Windows: `ffmpeg -f x11grab -framerate 30 -i :0.0 -s 1920x1080 rec.mp4`
only if an X server is up; otherwise record from Windows side and copy the file into
the repo folder.

## QC checklist (the new agent MUST run all of these before pushing)
1. `ffprobe promo/launch-teaser.mp4` -> duration 22-25s, 1920x1080, h264, yuv420p.
2. Extract frames every 0.5s:
   `ffmpeg -i promo/launch-teaser.mp4 -vf fps=2 frames/f_%03d.png`
3. Vision model INSPECTS frames directly (it has vision): confirm
   - no frame shows two text cards at once except title+pipeline line (7-11.5s)
     and Q+A pairs (11.5-16s, 16-18.5s, 21.5-24s),
   - text never touches another text,
   - fades visible at phase boundaries.
4. Fallback OCR QC (if no vision): `python3 - <<EOF` with easyocr `reader.readtext()`
   over frames (easyocr 1.7.2 is installed; method is readtext, not read) and diff
   extracted strings against the storyboard table above.
5. Numbers must be EXACTLY: 32/32, 14 days, v2.0, the static.hf.space URL. Never 30 days,
   never 10/10, never huggingface.co/spaces link as the primary CTA.
6. Upload: `git add promo/launch-teaser.mp4 && git commit -m "..." && git push origin main`.
7. Verify on GitHub page that the new file size/duration matches ffprobe output.

## Suggested prompt for the new chat
"Build the launch teaser for the rag-support-bot repo using
promo/VIDEO_BUILD_SCRIPT.md as the exact spec. Run Option A, QC every frame with your
vision (check overlap per the storyboard), commit+push, and report ffprobe numbers."
