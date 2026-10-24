"""
Hybrid RAG index: ChromaDB dense vectors + BM25 lexical + RRF fusion.
Why hybrid: vector-only misses exact identifiers (order numbers, refund windows);
BM25-only misses semantics. RRF fuses RANKS (never raw scores) — the standard
production pattern from top freelancer builds.
"""
import os
import re
import json
import hashlib
from pypdf import PdfReader

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

# Second-stage cross-encoder reranker (see reranker.py). Optional: the pipeline
# degrades to RRF-only if the model/dependency is missing.
RERANK_CANDIDATES = 10          # candidates first stage sends to the reranker
# bge-reranker scores long 120-word chunks lower than short ones (512-token
# dilution), so scoring is done per-sentence and the chunk keeps its MAX.
# Measured on the golden set: true positives score >= 0.065, off-topic probes
# (blueberry policy, Mars CEO, quantum computing) score <= 0.02 — a 0.05 floor
# cleanly separates "corpus can answer" from "abstain and escalate".
RERANK_DEAD_FLOOR = 0.05

STOPWORDS = set("""a an the is are was were what which who how when where why do does did
can could should would will for with and or but not of in on at to by from about as it its
my your our their this that these those i you we they me he she him her be been being have
has had i am no yes if then there here""".split())


def _stem(t: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(t) > len(suf) + 2 and t.endswith(suf):
            t = t[: -len(suf)]
            break
    if len(t) > 3 and t.endswith("e"):
        t = t[:-1]
    return t


def content_terms(text: str) -> set[str]:
    return {_stem(t) for t in TOKEN_RE.findall(text.lower())
            if t not in STOPWORDS and len(t) >= 2}

# Embedding backend switch (same hybrid pipeline, two engines):
#   ollama    -> local dev/demo: nomic-embed-text 768-dim via local Ollama
#   fastembed -> deployments (HF Spaces): ONNX bge-small-en-v1.5, ~50MB, CPU,
#               no server dependency, faster cold start
# Falls back to TF-IDF if neither is reachable.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "ollama")  # ollama | fastembed

_fastembed_model = None


def fastembed_embed(texts: list[str]) -> list[list[float]] | None:
    global _fastembed_model
    try:
        if _fastembed_model is None:
            from fastembed import TextEmbedding
            _fastembed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return [list(map(float, v)) for v in _fastembed_model.embed(texts)]
    except Exception as e:
        print(f"[warn] fastembed unavailable: {e}")
        return None


def ollama_embed(texts: list[str]) -> list[list[float]] | None:
    """Batch-embed texts via Ollama /api/embed. Returns None on failure."""
    try:
        import urllib.request
        req = urllib.request.Request(
            OLLAMA_URL + "/api/embed",
            json.dumps({"model": EMBED_MODEL, "input": texts}).encode(),
        )
        req.add_header("Content-Type", "application/json")
        resp = urllib.request.urlopen(req, timeout=60)
        return json.load(resp)["embeddings"]
    except Exception as e:
        print(f"[warn] embed endpoint unavailable: {e}")
        return None


def embed(texts: list[str]) -> list[list[float]] | None:
    if EMBED_BACKEND == "fastembed":
        return fastembed_embed(texts)
    return ollama_embed(texts)


def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(__import__("io").BytesIO(data))
        pages = []
        for i, page in enumerate(reader.pages):
            # pypdf >= 1.x uses extract_text(); some forks expose get_text()
            text = ""
            for meth in ("extract_text", "get_text"):
                try:
                    text = getattr(page, meth)() or ""
                    break
                except Exception:
                    continue
            if not text and hasattr(page, "get_contents"):
                try:
                    text = (page.get_contents() or "").decode("utf-8", errors="replace")
                except Exception:
                    text = ""
            pages.append(f"[page {i + 1}] " + text)
        return "\n".join(pages)
    return data.decode("utf-8", errors="replace")


TOTAL_ROW_RE = re.compile(
    r"\b(Add\s*:|Less\s*:|Grand\s+Total|Sub\s*-?\s*Total|Total\s+Amount)\b", re.I
)


def chunk_text(text: str, chunk_size=120, overlap=20) -> list[str]:
    """Line-aware sliding-window chunking.

    Why: word-window chunking (the old approach) cut invoice/ledger rows in
    half, so a chunk could end with "Grand Total 11.00 Units" and the amount
    lived in the next chunk — a small model then quotes "11.00 Units" as the
    total. Boundaries now never split a line, and a contiguous run of
    Add/Less/Grand Total lines is kept as one indivisible unit.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return [text] if text.strip() else []

    # 1. group lines into units: totals blocks are atomic, others fill ~chunk_size
    units: list[str] = []
    i = 0
    while i < len(lines):
        if TOTAL_ROW_RE.search(lines[i]):
            block = []
            while i < len(lines) and TOTAL_ROW_RE.search(lines[i]):
                block.append(lines[i])
                i += 1
            units.append("\n".join(block))
            continue
        cur, w = [], 0
        while i < len(lines) and w < chunk_size:
            cur.append(lines[i])
            w += len(lines[i].split())
            i += 1
        units.append("\n".join(cur))

    # 2. carry an overlap tail from the previous unit (whole lines only)
    chunks = []
    for k, unit in enumerate(units):
        if k > 0 and overlap:
            tail_lines = units[k - 1].split("\n")[-1].split()
            tail = " ".join(tail_lines[-overlap:])
            chunks.append(f"{tail}\n{unit}" if tail else unit)
        else:
            chunks.append(unit)
    return chunks or [text]


def normalize_invoice_totals(text: str) -> str:
    """Deterministic numeric normalization for invoice-like documents.

    Appends one grounded line per Grand Total line so the answer layer can cite
    the amount without parsing "11.00 Units". The amount is the LAST rupee-format
    number on that line — nothing is invented or computed.
    """
    if not re.search(r"GSTIN|Invoice No", text, re.I):
        return text

    def fix_line(m):
        line = m.group(0)
        amounts = re.findall(r"\d[\d,]*\.\d{2}", line)
        if not amounts:
            return line
        return f"{line}\n[normalized] Grand Total amount: ` {amounts[-1]}"

    return re.sub(r".*Grand\s+Total.*", fix_line, text)


class RAGIndex:
    def __init__(self, persist_dir: str):
        os.makedirs(persist_dir, exist_ok=True)
        try:
            import chromadb
            self.chroma = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.chroma.get_or_create_collection("docs")
            self.vector_ok = True
        except Exception as e:
            print(f"[warn] vector store unavailable, BM25-only mode: {e}")
            self.vector_ok = False
        self.documents = []   # {id, filename}
        self.chunks = []      # {text, source, page, token_list}
        self._bm25 = None
        self._docs_json = os.path.join(persist_dir, "documents.json")
        self._load()

    def _seed_vectors(self, start: int) -> None:
        """Store real Ollama embeddings in Chroma (no Chroma default model needed)."""
        if not self.vector_ok:
            return
        texts = [self.chunks[i]["text"] for i in range(start, len(self.chunks))]
        if not texts:
            return
        embs = ollama_embed(texts)
        if embs is None:
            self.vector_ok = False
            print("[warn] falling back to TF-IDF dense stand-in")
            return
        try:
            self.collection.add(
                ids=[f"chunk_{i}" for i in range(start, len(self.chunks))],
                embeddings=embs,
                metadatas=[{"source": self.chunks[i]["source"]} for i in range(start, len(self.chunks))],
            )
        except Exception as e:
            print(f"[warn] vector add failed: {e}")
            self.vector_ok = False

    def _load(self):
        if os.path.exists(self._docs_json):
            saved = json.load(open(self._docs_json))
            self.documents = saved["documents"]
            self.chunks = saved["chunks"]
            if self.vector_ok:
                try:
                    # rebuild vectors with the configured embedding model
                    # (drop any legacy default-model embeddings for consistency)
                    try:
                        self.chroma.delete_collection("docs")
                    except Exception:
                        pass
                    self.collection = self.chroma.get_or_create_collection("docs")
                    self._seed_vectors(0)
                except Exception as e:
                    print(f"[warn] vector re-seed failed: {e}")
                    self.vector_ok = False
            self._rebuild_bm25()

    def _save(self):
        with open(self._docs_json, "w") as f:
            json.dump({"documents": self.documents, "chunks": self.chunks}, f)

    def _rebuild_bm25(self):
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([c["token_list"] for c in self.chunks]) if self.chunks else None
        except Exception:
            self._bm25 = None

    def _page_of(self, text: str):
        m = re.match(r"\[page (\d+)\]", text)
        return int(m.group(1)) if m else None

    def add_document(self, filename: str, data: bytes) -> str:
        doc_id = hashlib.sha256(data).hexdigest()[:12]
        if any(d["id"] == doc_id for d in self.documents):
            return doc_id  # already indexed
        text = extract_text(filename, data)
        text = normalize_invoice_totals(text)
        new_chunks = chunk_text(text)
        start = len(self.chunks)
        for chunk in new_chunks:
            self.chunks.append({
                "text": chunk,
                "source": filename,
                "page": self._page_of(chunk),
                "token_list": TOKEN_RE.findall(chunk.lower()),
            })
        if self.vector_ok:
            self._seed_vectors(start)
        self.documents.append({"id": doc_id, "filename": filename})
        self._rebuild_bm25()
        self._save()
        return doc_id

    def reset(self):
        if self.vector_ok:
            try:
                self.chroma.delete_collection("docs")
                self.collection = self.chroma.get_or_create_collection("docs")
            except Exception:
                pass
        self.documents, self.chunks, self._bm25 = [], [], None
        self._save()

    def doc_count(self):
        return len(self.documents)

    def chunk_count(self):
        return len(self.chunks)

    def list_documents(self):
        return self.documents

    def _vector_rank(self, query: str, k: int) -> list[int]:
        if not self.vector_ok or not self.chunks:
            return []
        try:
            emb = ollama_embed([query])
            if emb is None:
                return []
            res = self.collection.query(query_embeddings=[emb[0]],
                                       n_results=min(max(1, k * 2), self.chunk_count()))
            ids = res["ids"][0] if res and res["ids"] else []
            return [int(cid.split("_")[1]) for cid in ids if cid.startswith("chunk_")]
        except Exception as e:
            print(f"[warn] vector query failed: {e}")
            return []

    def _tfidf_rank(self, query: str, k: int) -> list[int]:
        """Dependency-free cosine similarity over TF-IDF vectors (dense-ish signal)."""
        import numpy as np
        if not self.chunks:
            return []
        docs = [c["token_list"] for c in self.chunks]
        vocab = {}
        for d in docs:
            for t in d:
                vocab.setdefault(t, 0)
                vocab[t] += 1
        n = len(docs)
        def vec(tokens):
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            v = np.zeros(len(vocab))
            for t, f in tf.items():
                if t in vocab:
                    v[list(vocab).index(t)] = f * np.log(1 + n / (1 + vocab[t]))
            return v
        qv = vec(TOKEN_RE.findall(query.lower()))
        if not qv.any():
            return []
        scores = []
        for i, d in enumerate(docs):
            dv = vec(d)
            denom = (np.linalg.norm(qv) * np.linalg.norm(dv)) or 1e-9
            scores.append((float(np.dot(qv, dv) / denom), i))
        scores.sort(reverse=True)
        return [i for _, i in scores[:k]]

    def _bm25_rank(self, query: str, k: int) -> list[int]:
        if not self._bm25:
            return []
        tokens = TOKEN_RE.findall(query.lower())
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [i for i in order[:k]]

    @staticmethod
    def rrf_fuse(rank_lists: list[list[int]], k_out: int, rr_k: int = 60):
        scores: dict[int, float] = {}
        for lst in rank_lists:
            for rank, item in enumerate(lst):
                scores[item] = scores.get(item, 0.0) + 1.0 / (rr_k + rank + 1)
        return sorted(scores, key=lambda x: -scores[x])[:k_out], scores

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        fused, scores = self.rrf_fuse([
            self._vector_rank(query, top_k),
            self._tfidf_rank(query, top_k),
            self._bm25_rank(query, top_k),
        ], RERANK_CANDIDATES)
        # Relevance gate (anti-hallucination): if the query's content terms
        # are less than half present anywhere in the indexed corpus, the corpus
        # has no basis to answer — return zero hits so the answer layer says
        # "not found, escalate" instead of hallucinating. A chunk counts as
        # relevant only if it shares at least one content term with the query.
        chunk_terms = [content_terms(c["text"]) for c in self.chunks]
        vocab = set()
        for terms in chunk_terms:
            vocab |= terms
        q_terms = content_terms(query)
        if q_terms and len(q_terms & vocab) / len(q_terms) < 0.5:
            return []
        relevant = [i for i in fused if q_terms & chunk_terms[i]]
        if not relevant:
            return []

        # --- second stage: cross-encoder reranking -------------------------
        # RRF tells us WHICH chunks are popular across three retrievers; the
        # cross-encoder tells us HOW relevant each one actually is to THIS query.
        # Order candidates by reranker score, and drop anything below the floor.
        rerank_scores = None
        try:
            from reranker import rerank
            rerank_scores = rerank(query, [self.chunks[i]["text"] for i in relevant])
        except Exception as e:
            print(f"[warn] reranker import failed: {e}")
        if rerank_scores is not None:
            # If NOTHING scores above the dead floor, the corpus can't answer:
            # abstain rather than let the 0.5b model phrase noise into an answer.
            if max(rerank_scores) < RERANK_DEAD_FLOOR:
                return []
            paired = [(i, s) for i, s in zip(relevant, rerank_scores)
                      if s >= RERANK_DEAD_FLOOR]
            paired.sort(key=lambda p: -p[1])
            ordered = [i for i, _ in paired]
        else:
            ordered = relevant

        hits = []
        for idx in ordered[:top_k]:
            c = self.chunks[idx]
            # snippet: with line-aware chunking a chunk IS a coherent block
            # (rows/totals never split), so show it in full — the old 600-char
            # cut could hide the answer line from the citation while the chunk
            # still ranked high. Cap stays only as a guard for huge blocks.
            text = c["text"][:1200]
            hit = {
                "text": text,
                "source": c["source"],
                "page": c["page"],
                "score": round(scores[idx], 3),
            }
            if rerank_scores is not None:
                hit["rerank_score"] = round(rerank_scores[relevant.index(idx)], 4)
            hits.append(hit)
        return hits
