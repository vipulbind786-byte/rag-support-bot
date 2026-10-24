---
title: Offline RAG Support Bot Demo
emoji: 🤖
colorHex: "#38bdf8"
sdk: static
---

# Offline RAG Support Bot — live demo

A support bot that answers **only from your own documents**, with citations, and **abstains instead of hallucinating** when the corpus can't answer.

This demo runs the entire retrieval pipeline **in your browser** (zero server, zero cost, permanent):

1. **Chunking** — line-aware chunks (invoice/ledger rows are never split mid-line; Add/Less/Grand Total blocks stay atomic) with a 20-line overlap tail
2. **Hybrid retrieval** — BM25 (exact terms/numbers) + TF-IDF (distinctive terms), fused with **RRF** (reciprocal rank voting)
3. **Relevance gate** — if fewer than 50% of the query's content terms appear in the corpus, the bot returns *no* answer and escalates to a human
4. **Reranker-lite** — candidates re-ranked by rarity-weighted query-term coverage (rare, discriminative terms win)
5. **Citations** — every answer shows the exact source sentence + filename

Try it: ask "What is the refund window?" then ask "What is the blueberry sourcing policy?" — the first gets a cited answer, the second correctly abstains.

Paste your own document text and the index rebuilds instantly.

**Production version** (same pipeline, on the client's own machine): real embeddings (nomic-embed-text / fastembed bge-small), a true cross-encoder reranker (`bge-reranker-base`), a local 0.5B phrasing model, ChromaDB — all offline, no API keys, no cloud. Source code: [github.com/vipulbind786-byte/rag-support-bot](https://github.com/vipulbind786-byte/rag-support-bot). Eval hit@5: **14/14** on the bundled sample corpus (10 policy + 4 invoice golden questions) with 3/3 off-topic probes abstaining.
