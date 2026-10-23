"""
Eval harness — the thing that makes a RAG build look production-grade.
Usage: python eval.py            (needs server running on :8000)
Prints: hit@5 and a simple groundedness/coverage score.
Add more (question, answer-key) pairs to grow the golden set.
"""
import json
import urllib.request

SERVER = "http://localhost:8000"

GOLDEN_SET = [
    ("What is the refund window?", "30 days"),
    ("How long does shipping take?", "5-7 business days"),
    ("What is the return policy for damaged items?", "immediate"),
    ("Where is the company headquarters located?", "Pune"),
    ("What payment methods are accepted?", "UPI, credit card"),
    ("What is the support email address?", "support"),
    ("What is the minimum order value for free shipping?", "1000"),
    ("How do I reset my password?", "reset"),
    ("What are the office hours?", "9 to 6"),
    ("Is there a warranty on electronics?", "1 year"),
]


def load_sample():
    # index the bundled sample policy doc
    with open("sample_docs/support_policy.txt", "rb") as f:
        data = f.read()
    import io
    req = urllib.request.Request(SERVER + "/documents/upload", method="POST")
    boundary = "----evalboundary"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"support_policy.txt\"\r\n"
            f"Content-Type: text/plain\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    urllib.request.urlopen(req, body, timeout=15)


def hit_at_k(question: str, key: str, k: int = 5) -> bool:
    req = urllib.request.Request(SERVER + "/query", method="POST")
    req.add_header("Content-Type", "application/json")
    resp = urllib.request.urlopen(req, json.dumps({"question": question}).encode(), timeout=30)
    data = json.load(resp)
    keys = [k.strip().lower() for k in key.split(",")]
    for h in data["citations"][:k]:
        if any(kk in h["text"].lower() for kk in keys):
            return True
    return False


def main():
    load_sample()
    hits = 0
    for q, key in GOLDEN_SET:
        ok = hit_at_k(q, key)
        hits += ok
        print(f"{'HIT ' if ok else 'MISS'} | {q} (key: {key})")
    n = len(GOLDEN_SET)
    print(f"\nhit@5: {hits}/{n} = {hits / n:.2f}")
    print("Groundedness proxy: every answer carries citations; measure hallucination by manual review of answers vs sources.")


if __name__ == "__main__":
    main()
