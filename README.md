# RAG Support Bot — Project 1 (portfolio flagship)

**One-liner pitch:** "Your support team answers the same 20 questions every week. I build a bot that answers them from your own documents — with page-number citations and measured hit rates."

## What's inside (production-grade pattern, not a demo)
- **Hybrid retrieval:** ChromaDB dense vectors (real Ollama `nomic-embed-text` embeddings) + TF-IDF cosine + BM25 lexical, fused with **Reciprocal Rank Fusion** (fuses ranks, never raw scores) — the exact pattern top RAG freelancers use
- **Two-stage retrieval:** RRF finds candidates fast, then a **local cross-encoder reranker (BAAI bge-reranker-base, CPU, $0)** orders them by true query-chunk relevance and prunes dead candidates before the LLM sees them. Degrades gracefully to RRF-only if the model is missing.
- **Citations:** every answer carries source + page + relevance score chips (RRF score + rerank score)
- **SSE streaming** (`/query/stream`): citations arrive BEFORE the first answer token
- **Eval harness** (`eval/eval.py`: 14 golden Q&A pairs across policy + invoice, prints hit@5 — currently **14/14 = 1.00**)
- **Anti-hallucination double gate:** corpus-coverage gate + reranker relevance floor (measured: true answers ≥0.065, off-topic probes ≤0.02, floor set at 0.05) → bot abstains and escalates instead of inventing
- **Security:** optional API-key auth (`RAG_API_KEY` env), per-IP rate limit (30/min), 25MB upload cap
- **LLM-agnostic:** local Ollama (`qwen2.5:0.5b`) by default, OPENAI_API_KEY if set, else zero-cost extractive fallback (the $0-stack pattern)

## Run
```
cd backend && pip install -r requirements.txt
python3 -m uvicorn app:app --port 8000
# open http://localhost:8000
python3 eval/eval.py   # prints hit@5
```

## API
- GET /health
- GET /documents
- POST /documents/upload (PDF/TXT/MD)
- DELETE /documents
- POST /query {question}
- POST /query/stream {question} (SSE)

## Demo script for Loom video (2 min)
1. Upload the sample support policy PDF
2. Ask "What is the refund window?" → answer with citation chips (page + score)
3. Ask a nonsense question ("What is your blueberry policy?") → bot says "not in documents, escalate" (anti-hallucination)
4. Show eval hit@5 = 1.00 number

## Promo
`promo/launch-teaser-v5.mp4` — 25s, 1920x1080 six-scene launch film (h264 + sound bed). Build spec: `promo/VIDEO_BUILD_SCRIPT.md`.

## Stress testing (real-world dirty docs, not toy corpora)
- `eval/stress_test.py` — 32 adversarial probes across multi-invoice PDFs, flattened tables, contradictory policy versions, entity traps, prompt injection, off-topic abstention, live-state questions. Result: **32/32**.
- Corpus: `acme_vendor_bundle.pdf` (18 pages: two invoices, SLA matrix, catalog, versioned policies, two-column FAQ, garbled OCR amounts, a buried injection memo) + `dirty_warranty_invoice.pdf` (soft-hyphen number corruption, repeated footers, decoy GSTINs).
- Full findings: `STRESS_REPORT.md`.
- Why it matters: every probe maps to a documented industry failure mode (OWASP prompt injection, Air Canada stale-policy liability, table-extraction accuracy). This is the part competitors get wrong.

## Remaining upgrades (genuinely not built yet)
1. OCR/vision layer for image-only scanned PDFs (text-layer PDFs only today)
2. PDF.js in-browser citation preview (Kotaemon-style)
3. Deploy: Render/Oracle free tier + Caddy

## Done: HF Space demo mirrors the deterministic router
`space/pipeline.js` is a JS port of the backend pipeline (ingest repair, per-invoice fact extraction, deterministic numeric/table/policy/entity answers) with the same 32-probe stress suite built into the page — **32/32 in-browser, identical numbers to the backend**.

## Notes
- If ChromaDB import fails on your env, the index auto-falls back to TF-IDF+BM25 hybrid — retrieval still works. (Verified 25 Oct 2026: ChromaDB works on Python 3.14 — 20 chunks stored with real 768-dim nomic-embed-text vectors; the fallback is only for exotic envs.)
- LLM phrasing verified end-to-end via Ollama on 11434 (`qwen2.5:0.5b`, temperature pinned to 0 for determinism). llama-server on 8080 closed connections during testing — use Ollama or set OPENAI_API_KEY.
