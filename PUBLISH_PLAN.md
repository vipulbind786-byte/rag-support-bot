# Launch publish plan — Offline RAG Support Bot

## Assets (all ready, verified)
- Demo (live, no login, runs in browser): https://vrxdr-rag-support-bot.static.hf.space/index.html
- Repo: https://github.com/vipulbind786-byte/rag-support-bot
- Launch film: promo/launch-teaser-v5.mp4 (25s, 1920x1080, sound)
- Proof metrics: 32/32 stress suite, 14/14 golden eval, deterministic numeric answers, offline/zero-cost

## X/Twitter post (fits 280 chars with link)
Most "chat with your docs" bots guess. Mine refuses:

- answers only from your own documents, with citations
- computes invoice totals/GST in code, never from the model
- abstains + escalates when the corpus can't answer
- 32/32 adversarial stress probes pass

Runs fully offline on your own machine. Live demo (browser only):
https://vrxdr-rag-support-bot.static.hf.space/index.html

Attach: launch-teaser-v5.mp4

## LinkedIn post
Your support bot should know when it doesn't know.

I built an offline RAG support bot for document-heavy businesses (GST invoices, warranty PDFs, policy files). Instead of the usual "LLM + vector DB" demo, it does three things competitors skip:

1. Evidence before language — hybrid retrieval (dense vectors + BM25 + TF-IDF fused with RRF), then a cross-encoder reranker. Every answer carries a citation: source file, page, score.
2. Deterministic math — invoice totals, GST sums, comparisons and policy-version conflicts are computed in plain code. The local 0.5B model only phrases the answer; it never invents numbers.
3. It abstains — "What is your blueberry return policy?" gets "Not in the corpus. Escalating to a human." That refusal is the product.

Verified with a 32-probe adversarial stress suite: multi-invoice scoping, OCR-corrupted amounts, contradictory policy versions (14-day v2.0 supersedes the old 30-day rule), indirect prompt-injection memos neutralized at ingest, Hinglish queries. 32/32 pass.

Everything runs on the client's own machine: ChromaDB, local model, no cloud, no API keys. For hospitals, legal and finance, data residency is the actual product — not price.

Live demo (runs entirely in your browser): https://vrxdr-rag-support-bot.static.hf.space/index.html
Source: https://github.com/vipulbind786-byte/rag-support-bot

Hashtags: #RAG #LocalAI #DataResidency #BuildInPublic #AIEngineering

## Order of publishing
1. LinkedIn first (client audience; video upload natively).
2. X with the teaser + demo link.
3. HF Space discussion post announcing the launch.
4. One Dev.to article: "Why your RAG bot should refuse to answer" — link the repo.
5. WhatsApp/Telegram: share demo link in relevant freelancer/small-business groups.

## Free/low-cost next levers after launch
- Pin the repo (GitHub profile visibility, zero cost).
- Add "Built with" badges + STRESS_REPORT.md link in README (already there).
- Product Hunt listing (free, needs account) — schedule for the week after first social posts.
- DM 10 local SMEs/ca with the demo link and the line: "your GST invoices answered offline, no cloud." That's the actual sales pipeline.
