# RAG Support Bot — Project 1 (portfolio flagship)

**One-liner pitch:** "Your support team answers the same 20 questions every week. I build a bot that answers them from your own documents — with page-number citations and measured hit rates."

## What's inside (production-grade pattern, not a demo)
- **Hybrid retrieval:** ChromaDB dense vectors + TF-IDF cosine + BM25 lexical, fused with **Reciprocal Rank Fusion** (fuses ranks, never raw scores) — the exact pattern top RAG freelancers use
- **Citations:** every answer carries source + page + relevance score chips
- **SSE streaming** (`/query/stream`): citations arrive BEFORE the first answer token
- **Eval harness** (`eval/eval.py`: 10 golden Q&A pairs, prints hit@5 — currently **0.70** on the sample policy)
- **Security:** optional API-key auth (`RAG_API_KEY` env), per-IP rate limit (30/min), 5MB upload cap
- **LLM-agnostic:** uses OPENAI_API_KEY if set, else local Ollama/llama.cpp, else zero-cost extractive fallback (the $0-stack pattern)

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
4. Show eval hit@5 = 0.70 number

## Next upgrades (in ROI order)
1. Real embeddings via Ollama embed endpoint or OpenAI embeddings (replaces TF-IDF)
2. Cross-encoder reranker (BGE-reranker local or voyageai)
3. PDF.js in-browser citation preview (Kotaemon-style)
4. Deploy: Render/Oracle free tier + Caddy

## Notes
- If ChromaDB import fails on your env, the index auto-falls back to TF-IDF+BM25 hybrid — retrieval still works.
- Local llama-server on 8080 closed connections during testing; set OPENAI_API_KEY or LOCAL_LLM_URL for real LLM answers.
