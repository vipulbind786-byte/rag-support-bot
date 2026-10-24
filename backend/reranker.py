"""
Cross-encoder reranker layer: second-stage precision scoring.

First stage (vector + BM25 + TF-IDF fused by RRF) finds candidates fast but
orders them by popularity of rank, not true relevance. A cross-encoder reads
(query, chunk) TOGETHER and outputs a calibrated relevance score — this is the
component Kotaemon/RAGDoll-type production systems use and the one our demo
was missing.

Model: BAAI/bge-reranker-base (Apache-2.0, ~420MB, CPU-friendly, local, $0).
Loaded lazily; if sentence-transformers or the model is missing the pipeline
gracefully falls back to RRF-only ordering (never crashes).
"""
import os
import re

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "/mnt/v/ai-sandbox/workspace/models/bge-reranker-base",
)
RERANKER_TIMEOUT_S = int(os.getenv("RERANKER_TIMEOUT_S", 20))

_model = None
_failed = False

SENT_RE = re.compile(r"[^.;!?]*[.;!?]")


def _sentences(text: str) -> list[str]:
    """Sentence-level units: a 120-word chunk dilutes the cross-encoder score;
    scoring query-vs-sentence and taking the max is the standard fix and also
    matches how humans cite (one decisive line)."""
    parts = [s.strip() for s in SENT_RE.findall(text) if len(s.strip()) >= 20]
    if not parts:
        return [text[:300]]
    return [s[:300] for s in parts]


def _get_model():
    global _model, _failed
    if _failed:
        return None
    if _model is None:
        try:
            from sentence_transformers import CrossEncoder
            _model = CrossEncoder(RERANKER_MODEL, device="cpu")
            print(f"[reranker] loaded {RERANKER_MODEL} (CPU)")
        except Exception as e:
            print(f"[warn] reranker unavailable ({e}); RRF-only ordering")
            _failed = True
            return None
    return _model


def rerank(query: str, texts: list[str]) -> list[float] | None:
    """Return one relevance score (0..1) per chunk — the MAX of the whole-chunk
    score and each of its sentence scores — or None if unavailable.

    Whole-chunk scoring handles table/invoice text (no sentence punctuation);
    sentence scoring handles prose where the answer is one line buried in a
    120-word chunk. Both are batched in a single model.predict call.
    """
    scored = rerank_with_snippets(query, texts)
    if scored is None:
        return None
    return [s for s, _ in scored]


def rerank_with_snippets(query: str, texts: list[str]):
    """Return (score, best_sentence) per chunk. The best sentence is the
    answer-bearing line the reranker found — we anchor the citation snippet
    on it so the proof is always visible to the user."""
    model = _get_model()
    if model is None or not texts:
        return None
    try:
        pairs, owner = [], []
        for ci, text in enumerate(texts):
            pairs.append((query, text[:900]))
            owner.append(ci)
            for sent in _sentences(text)[:10]:
                pairs.append((query, sent))
                owner.append(ci)
        scores = model.predict(pairs, batch_size=64)
        best = [(0.0, "")] * len(texts)
        for ci, pair, s in zip(owner, pairs, scores):
            if s > best[ci][0]:
                best[ci] = (float(s), pair[1])
        return best
    except Exception as e:
        print(f"[warn] rerank inference failed: {e}")
        _failed = True
        return None
