"""
RAG Support Bot — hybrid retrieval (vector + BM25 + RRF) with citations.
FastAPI backend. Run: uvicorn app:app --port 8000
"""
import os
import json
import time
import hmac
import hashlib
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from rag import RAGIndex
from llm import answer_from_context, answer_stream, deterministic_answer

API_KEY = os.environ.get("RAG_API_KEY", "")  # set to require auth; empty = open demo
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 25_000_000))  # 25 MB cap
RATE_WINDOW = 60
RATE_LIMIT = 30  # queries/min per IP

# Routes that read or mutate the indexed corpus. The UI and /health stay public
# so the demo page can load and monitoring can probe it; everything corpus-facing
# is gated when RAG_API_KEY is set. Fail-closed: a protected route without a
# valid Bearer token is rejected before the handler runs.
PROTECTED_PREFIXES = ("/query", "/documents")


def is_authorized(request) -> bool:
    """Constant-time Bearer check. Never logs or echoes the key."""
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return False
    return hmac.compare_digest(header[len("Bearer "):], API_KEY)

app = FastAPI(title="RAG Support Bot", version="1.0.0")
index = RAGIndex(os.path.join(os.path.dirname(__file__), "data"))
_buckets: dict[str, list[float]] = {}


def check_rate(ip: str) -> bool:
    now = time.time()
    b = [t for t in _buckets.get(ip, []) if now - t < RATE_WINDOW]
    if len(b) >= RATE_LIMIT:
        _buckets[ip] = b
        return False
    b.append(now)
    _buckets[ip] = b
    return True


@app.middleware("http")
async def auth_and_rate(request, call_next):
    path = request.url.path
    protected = path.startswith(PROTECTED_PREFIXES)
    if protected and API_KEY and not is_authorized(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if protected and not check_rate(request.client.host):
        return JSONResponse({"error": "rate limit exceeded"}, status_code=429)
    return await call_next(request)


@app.get("/health")
def health():
    return {"status": "ok", "documents": index.doc_count()}


@app.get("/documents")
def list_documents():
    return {"documents": index.list_documents()}


@app.post("/documents/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file too large (max {MAX_UPLOAD_BYTES} bytes)")
    doc_id = index.add_document(file.filename, data)
    return {"document_id": doc_id, "filename": file.filename, "chunks": index.chunk_count()}


@app.delete("/documents")
def clear_documents():
    index.reset()
    return {"cleared": True}


@app.post("/query")
async def query(payload: dict):
    q = (payload.get("question") or "").strip()
    if not q:
        raise HTTPException(400, "question is required")
    hits = index.search(q, top_k=int(payload.get("top_k", 6)))
    facts = index.structured_facts()
    det = deterministic_answer(q, facts)
    answer = det or answer_from_context(q, hits, facts)
    return {"answer": answer, "citations": hits,
            "answered_by": "deterministic" if det else "llm"}


@app.post("/query/stream")
async def query_stream(payload: dict):
    """SSE streaming: citations arrive BEFORE the first LLM token."""
    q = (payload.get("question") or "").strip()
    if not q:
        raise HTTPException(400, "question is required")
    hits = index.search(q, top_k=int(payload.get("top_k", 6)))
    facts = index.structured_facts()

    def gen():
        yield f"data: {json.dumps({'type': 'citations', 'citations': hits})}\n\n"
        for token in answer_stream(q, hits, facts):
            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "frontend"), html=True), name="ui")
