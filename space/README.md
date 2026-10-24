---
title: Offline RAG Support Bot Demo
emoji: 🤖
colorHex: "#38bdf8"
sdk: static
---

# Offline RAG Support Bot — live demo

A support bot that answers **only from your own documents**, with citations, and **abstains instead of hallucinating** when the corpus can't answer.

This demo runs the entire retrieval pipeline **in your browser** (zero server, zero cost, permanent):

1. **Dirty-text repair** — scanned/OCR artifacts fixed deterministically: `46,00000` → `46,000.00`, garbled digit runs regrouped, repeated footers stripped, buried instruction memos neutralized at ingest before they can ever be retrieved
2. **Structured fact extraction** — invoices parsed per invoice (two invoices in one PDF stay distinct), catalog rows rejoined from flattened table cells, SLA matrix headers rebuilt, policy versions resolved by version number, warranty attributes scoped per entity
3. **Deterministic answers** — totals, GST, comparisons, stock/price lookups, version precedence and entity attributes are computed in plain JavaScript: **zero LLM calls**. The same functions run in the production backend (Python), so demo and backend give identical numbers
4. **Hybrid retrieval fallback** — BM25 + TF-IDF fused with RRF, then rarity-weighted reranker-lite, gated by a coverage + relevance floor so off-topic and live-status questions abstain instead of inventing answers
5. **Citations** — every retrieval-backed answer shows the exact source sentence + document

Try it: ask "what is the grand total of invoice 0142" (₹5,60,244.00), then "which invoice has the higher grand total", then "what is your policy on return of blueberries" — the first two are computed in code, the third correctly abstains.

Paste your own document text and the index rebuilds instantly.

**Built-in brutal stress test** (button on the page): 32 adversarial probes covering multi-invoice scoping, OCR-corrupted amounts, contradictory policy versions, entity traps, indirect prompt injection, off-topic abstention, live-status abstention and Hinglish queries — **32/32 pass, 23 answered deterministically**.

**Production version** (same pipeline, on the client's own machine): real embeddings (nomic-embed-text / fastembed bge-small), a true cross-encoder reranker (`bge-reranker-base`), a local 0.5B phrasing model, ChromaDB — all offline, no API keys, no cloud. Source code: [github.com/vipulbind786-byte/rag-support-bot](https://github.com/vipulbind786-byte/rag-support-bot). Backend eval: **14/14 golden probes** and **32/32 brutal stress probes** (`eval/stress_test.py`), findings in `STRESS_REPORT.md`.
