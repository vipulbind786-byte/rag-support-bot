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

# Real embeddings via Ollama (nomic-embed-text, free, local).
# Falls back to TF-IDF if the embed endpoint is unreachable.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


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


def chunk_text(text: str, chunk_size=120, overlap=20) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return chunks or [text]


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
        ], top_k)
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
        hits = []
        for idx in fused:
            if idx not in relevant:
                continue
            c = self.chunks[idx]
            # snippet: full chunk text (capped), but guarantee the sentence
            # containing the best query-term match is included — this is what
            # makes citations verifiable and evals honest.
            text = c["text"]
            if len(text) > 600:
                terms = [t for t in TOKEN_RE.findall(query.lower()) if t in text.lower()]
                if terms:
                    pos = text.lower().find(terms[0])
                    start = max(0, pos - 250)
                    text = text[start:start + 600]
                else:
                    text = text[:600]
            hits.append({
                "text": text,
                "source": c["source"],
                "page": c["page"],
                "score": round(scores[idx], 3),
            })
        return hits
