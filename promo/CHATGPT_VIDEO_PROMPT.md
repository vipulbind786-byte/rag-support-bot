# ChatGPT prompt for a production-grade launch video (no screen recording)

Paste everything below the line into ChatGPT. Output will be a scene-by-scene
video spec you can hand to an editor / AI video tool (Runway, Sora, CapCut,
After Effects, or ffmpeg+motion scripts).

---

You are a senior motion-design director for tech product launch films. Design a
25-second launch teaser for an offline AI support bot. No screen recording, no
captured footage — everything must be built as motion graphics: kinetic
typography, animated mock UI, abstract data visuals.

PRODUCT (use only these facts, invent nothing):
- Name: "Offline RAG Support Bot"
- One-line pitch: "Your documents. Your answers. No cloud, no API keys, zero cost."
- What it does: answers support questions ONLY from a company's own documents
  (GST invoices, warranty PDFs, support policies). If the corpus cannot answer,
  it refuses and escalates to a human instead of guessing.
- Tech highlights (show as animated visual metaphors, not as text walls):
  hybrid retrieval (BM25 + TF-IDF + dense vectors fused with RRF), cross-encoder
  reranking, deterministic numeric answers computed in code (invoice totals, GST
  sums), a local 0.5B model for phrasing, ChromaDB vector store — all on the
  client's own machine.
- Proof points (exact numbers, do not alter): 32/32 adversarial stress-test probes
  pass; multi-invoice scoping (two invoices in one PDF answered separately);
  policy version resolution ("14 days, policy v2.0 — supersedes the old 30-day
  rule"); hidden prompt-injection memos inside PDFs neutralized before the model
  ever sees them; Hinglish queries handled ("mouse ka price kitna hai").
- Live demo (browser-only, no server): vrxdr-rag-support-bot.static.hf.space
- Audience: Indian SMBs, hospitals, legal & finance firms — decision-makers who
  care about data residency more than price.

CURRENT VIDEO PROBLEMS TO FIX (the new spec must solve all of these):
1. Opening cards "Your documents." / "Your answers." overlap and visually collapse
   into each other during the first 5 seconds — replace with clean sequential
   fade-in/fade-out cards, zero overlap.
2. The whole film is static text on a solid dark background — it feels like a
   slideshow, not a launch film. It needs depth, motion, and rhythm.
3. Silent video — needs a sound design plan (percussive ambient bed, subtle UI
   clicks on answer reveals, a rising motif at the 32/32 reveal, a final
   resolution chord).
4. No visual proof — viewers never see the product behaving. Build animated mock
   UI: a chat panel where a query types itself, a green answer appears with a
   citation chip (source + page + score), and an off-topic query gets a red
   "Not in corpus — escalating to human" response. That refusal moment is the
   hero of the film.

STYLE TARGET: premium B2B tech aesthetic — think Linear/Stripe launch films.
Dark navy base (#0b1120), electric blue accent (#38bdf8), emerald success (#34d399),
coral alert (#f87171). Crisp mono for code/numbers, bold sans for headlines.
Subtle grain, soft glow, slow parallax on background grid lines, confident
2.5–4 second beats, no cheesy transitions, no stock footage, no AI-face clips.

STRUCTURE (25s, six scenes):
- Scene 1 (0–4s): Hook. "Your documents." then "Your answers." — sequential cards
  with fade, each floating slightly, letter-by-letter reveal.
- Scene 2 (4–8s): Problem. Quick cuts of messy PDFs (invoice numbers scrambling,
  contradictory policy pages, a hidden memo labelled "ignore previous
  instructions") with an X stamp — the chaos a naive chatbot chokes on.
- Scene 3 (8–14s): Mechanism. Animated pipeline diagram: document chunks stream
  through BM25/TF-IDF/vectors lanes, merge via RRF, rerank, answer with a citation
  chip. Show one real interaction: "what is the return window?" → "14 days ·
  policy v2.0 [support_policy.txt, p.1]" chip slides in.
- Scene 4 (14–18s): Trust moment. "What is your blueberry policy?" → answer box
  shows "Not in the corpus. Escalating to a human." with a calm red glow. Caption:
  "It abstains. It never guesses."
- Scene 5 (18–22s): Proof. A results table scrolls with 32 rows ticking to green
  one by one, ending on a bold "32/32" stamp. Micro-labels flash: multi-invoice
  scoping · injection defense · Hinglish.
- Scene 6 (22–25s): Close. "Offline. On-premise. Zero cost." + the demo URL +
  "Data never leaves the premises." — logo lockup, soft fade out.

DELIVERABLE FORMAT:
For each scene give me: (a) exact frame description, (b) motion/animation spec
with timing to 0.1s, (c) on-screen text verbatim, (d) sound-design note,
(e) transition to next scene. Then give a final 60-second assembly note: total
duration, pacing curve, and which single scene to cut if we need a 15-second
version for Reels/TikTok.
