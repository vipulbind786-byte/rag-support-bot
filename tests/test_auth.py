"""Auth-boundary regression test. Run: python3 tests/test_auth.py

Covers the middleware in backend/app.py: which routes are protected when
RAG_API_KEY is set, and that they stay open when it is not.

Deliberately stubs RAGIndex so the tests never read/write backend/data, never
re-seed Chroma, and never call an LLM provider. No network, no real corpus.
The API key is a test-local value; it is never printed.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

TEST_KEY = "test-secret-not-real"


class StubIndex:
    """Minimal stand-in for rag.RAGIndex: enough surface for the routes."""

    def __init__(self, *args, **kwargs):
        self.docs = []

    def doc_count(self):
        return len(self.docs)

    def list_documents(self):
        return [{"id": d} for d in self.docs]

    def add_document(self, filename, data):
        self.docs.append(filename)
        return "stub-id"

    def chunk_count(self):
        return 1

    def reset(self):
        self.docs.clear()

    def search(self, query, top_k=6):
        return []

    def structured_facts(self):
        return {}


import rag
rag.RAGIndex = StubIndex

import app as app_module
from fastapi.testclient import TestClient


def reset_state():
    app_module._buckets.clear()


def check(label, condition):
    if not condition:
        raise AssertionError(f"FAIL: {label}")
    print(f"ok: {label}")


def main():
    # --- Key configured: every corpus-facing route must require the key ---
    app_module.API_KEY = TEST_KEY
    client = TestClient(app_module.app)
    reset_state()
    auth = {"Authorization": f"Bearer {TEST_KEY}"}

    protected = [
        ("GET", "/documents"),
        ("POST", "/documents/upload"),
        ("DELETE", "/documents"),
        ("POST", "/query"),
        ("POST", "/query/stream"),
    ]

    for method, path in protected:
        reset_state()
        r = client.request(method, path)
        check(f"{method} {path} without key -> 401", r.status_code == 401)

        reset_state()
        r = client.request(method, path, headers={"Authorization": "Bearer wrong-key"})
        check(f"{method} {path} with invalid key -> 401", r.status_code == 401)

        reset_state()
        r = client.request(method, path, headers={"Authorization": "Basic " + TEST_KEY})
        check(f"{method} {path} with non-Bearer scheme -> 401", r.status_code == 401)

    # 401 responses must not leak the key or any part of it.
    reset_state()
    body = client.get("/documents").text
    check("401 body does not contain the key", TEST_KEY not in body)
    check("401 body is the generic unauthorized error", "unauthorized" in body)

    # Prefix-sharing route (/documents/upload) is covered by the /documents prefix.
    reset_state()
    r = client.post("/documents/upload", files={"file": ("x.txt", b"hello")})
    check("upload without key -> 401", r.status_code == 401)

    # Valid key: routes behave exactly as before the fix.
    reset_state()
    r = client.get("/documents", headers=auth)
    check("GET /documents with valid key -> 200", r.status_code == 200)
    check("document list shape preserved", "documents" in r.json())

    reset_state()
    r = client.post("/documents/upload", headers=auth, files={"file": ("policy.txt", b"Refund window: 30 days")})
    check("upload with valid key -> 200", r.status_code == 200)
    j = r.json()
    check("upload response preserved (document_id/filename/chunks)",
          {"document_id", "filename", "chunks"} <= set(j))

    reset_state()
    r = client.delete("/documents", headers=auth)
    check("delete with valid key -> 200", r.status_code == 200)
    check("delete response preserved (cleared)", r.json().get("cleared") is True)

    reset_state()
    r = client.post("/query", headers=auth, json={"question": "what is the refund window?"})
    check("query with valid key -> 200", r.status_code == 200)
    check("query response keys preserved (answer/citations/answered_by)",
          {"answer", "citations", "answered_by"} <= set(r.json()))

    reset_state()
    r = client.post("/query/stream", headers=auth, json={"question": "refund window"})
    check("query/stream with valid key -> 200", r.status_code == 200)
    check("SSE payload preserved (citations event then done)",
          "citations" in r.text and "done" in r.text)

    # Empty question still rejected as before.
    reset_state()
    r = client.post("/query", headers=auth, json={"question": ""})
    check("empty question -> 400 (unchanged)", r.status_code == 400)

    # Rate limiting still applies to protected routes.
    reset_state()
    blocked = False
    for _ in range(app_module.RATE_LIMIT + 1):
        r = client.post("/query", headers=auth, json={"question": "x"})
        if r.status_code == 429:
            blocked = True
            break
    check("protected route still rate limited -> 429", blocked)

    # /health and the UI stay public so monitoring and the demo page work.
    reset_state()
    r = client.get("/health", headers=auth)
    check("GET /health stays public with key set -> 200", r.status_code == 200)
    check("health reports document count", "documents" in r.json())

    # --- Key not configured: documented open-demo behavior is preserved ---
    app_module.API_KEY = ""
    client2 = TestClient(app_module.app)

    reset_state()
    r = client2.get("/documents")
    check("no key configured: GET /documents open -> 200", r.status_code == 200)

    reset_state()
    r = client2.post("/documents/upload", files={"file": ("sop.txt", b"hello")})
    check("no key configured: upload open -> 200", r.status_code == 200)

    reset_state()
    r = client2.delete("/documents")
    check("no key configured: delete open -> 200", r.status_code == 200)

    reset_state()
    r = client2.post("/query", json={"question": "hello"})
    check("no key configured: query open -> 200", r.status_code == 200)

    # Constant-time comparison helper: a key sharing a prefix but differing must fail.
    app_module.API_KEY = TEST_KEY
    class FakeReq:
        def __init__(self, header):
            self.headers = {"Authorization": header}
    check("prefix-sharing wrong key rejected",
          not app_module.is_authorized(FakeReq("Bearer " + TEST_KEY[:-1])))
    check("exact key accepted", app_module.is_authorized(FakeReq("Bearer " + TEST_KEY)))
    check("missing header rejected", not app_module.is_authorized(FakeReq("")))

    print("\nALL AUTH TESTS PASSED")


if __name__ == "__main__":
    main()
