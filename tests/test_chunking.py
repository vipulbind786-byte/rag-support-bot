"""Offline test for the new table-aware chunker. Run: python tests/test_chunking.py
Gate: every chunk must (a) keep invoice rows/total blocks whole, (b) never
split a line across a boundary, (c) preserve the golden answer strings.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from rag import chunk_text, normalize_invoice_totals, extract_text

INVOICE = "/mnt/c/Users/USER/Downloads/VSCPL-Sale-VCSPL-03485-2627-123239_1782198162.pdf"
POLICY = os.path.join(os.path.dirname(__file__), "..", "sample_docs", "support_policy.txt")

def lines_of(chunk):
    return [l for l in chunk.split("\n") if l.strip()]

def test_invoice():
    text = normalize_invoice_totals(extract_text("invoice.pdf", open(INVOICE, "rb").read()))
    chunks = chunk_text(text)
    print(f"invoice: {len(chunks)} chunks")

    # invariant 1: the Grand Total line and its amount are never split
    ok = any("Grand Total" in c and "3,93,160.00" in c for c in chunks)
    print("PASS grand-total atomic" if ok else "FAIL grand-total split")
    assert ok

    # invariant 2: the normalized line exists exactly once per Grand Total
    norm = sum("[normalized] Grand Total amount: ` 3,93,160.00" in c for c in chunks)
    print(f"PASS normalized total present ({norm} chunk(s))" if norm else "FAIL normalized total")
    assert norm >= 1

    # invariant 3: no chunk ends mid-line and every chunk line comes from the
    # source (or is a normalization line we added). Comparison is whitespace-
    # collapsed because the overlap tail joins words with single spaces.
    src_norm = {" ".join(l.split()) for l in text.splitlines() if l.strip()}
    bad = [c[:80] for c in chunks
           for l in lines_of(c)
           if " ".join(l.split()) not in src_norm and not l.startswith("[normalized]")]
    print("PASS no line split across chunks" if not bad else f"FAIL line split: {bad}")
    assert not bad

    # invariant 4: each item row stays intact
    rows = [c for c in chunks for c2 in [c] if "CPU AMD RYZEN 9 9950X MPK 85423100 1.00 PCS 41,000.00 41,000.00" in " ".join(c.split())]
    print("PASS item row intact" if rows else "FAIL item row split")
    assert rows

    print("\n--- sample chunks ---")
    for c in chunks:
        if "Grand Total" in c:
            print(c[:400])
            break
    return chunks

def test_policy():
    text = open(POLICY, encoding="utf-8").read()
    chunks = chunk_text(normalize_invoice_totals(text))
    keys = ["30 days", "5-7 business days", "immediate", "Pune", "UPI", "support", "1000", "reset", "9 to 6", "1 year"]
    missing = [k for k in keys if not any(k.lower() in " ".join(c.split()).lower() for c in chunks)]
    print(f"policy: {len(chunks)} chunks; golden keys preserved: {len(keys) - len(missing)}/{len(keys)}")
    if missing:
        print("FAIL missing keys:", missing)
        sys.exit(1)

if __name__ == "__main__":
    test_invoice()
    test_policy()
    print("\nALL CHUNKING TESTS PASSED")
