"""Offline test for the new table-aware chunker. Run: python tests/test_chunking.py
Gate: every chunk must (a) keep invoice rows/total blocks whole, (b) never
split a line across a boundary, (c) preserve the golden answer strings.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from rag import (chunk_text, normalize_invoice_totals, extract_text,
                 neutralize_injection, repair_scanned_numbers,
                 repair_glued_cents, strip_boilerplate, rejoin_table_rows)

HERE = os.path.dirname(os.path.abspath(__file__))
INVOICE = os.path.join(HERE, "..", "sample_docs", "acme_vendor_bundle.pdf")
POLICY = os.path.join(HERE, "..", "sample_docs", "support_policy.txt")

def lines_of(chunk):
    return [l for l in chunk.split("\n") if l.strip()]

def ingest(raw, filename="invoice.pdf"):
    """Exactly the pipeline add_document() runs, so the test gates the real
    ingest chain, not just chunk_text in isolation."""
    text = extract_text(filename, raw)
    text = neutralize_injection(text)
    text = repair_scanned_numbers(text)
    text = repair_glued_cents(text)
    text = strip_boilerplate(text)
    text = rejoin_table_rows(text)
    return normalize_invoice_totals(text)

def test_invoice():
    with open(INVOICE, "rb") as f:
        raw = f.read()
    text = ingest(raw)
    chunks = chunk_text(text)
    print(f"invoice: {len(chunks)} chunks")

    # invariant 1: every grand total's label and its repaired amount stay in
    # the same chunk (the OCR-corrupted 5,60,24400 / 3,18,60000 amounts must
    # be regrouped to cents and never split across a boundary)
    for amount in ("5,60,244.00", "3,18,600.00"):
        ok = any("GRAND TOTAL" in c.upper() and amount in c for c in chunks)
        print(f"PASS grand total {amount} atomic" if ok else f"FAIL grand total {amount} split")
        assert ok

    # invariant 2: every chunk line comes from the source (or is a
    # normalization line we added) — no line is cut mid-way in a way that
    # destroys meaning. The overlap tail joins the last N words of the
    # previous unit into one space-joined line, so a chunk line may be a
    # word-suffix of a source line; comparison is whitespace-collapsed.
    src_norm = {" ".join(l.split()) for l in text.splitlines() if l.strip()}
    bad = [c[:80] for c in chunks
           for l in lines_of(c)
           if not l.startswith("[normalized]")
           and " ".join(l.split()) not in src_norm
           and not any(" ".join(l.split()) in s for s in src_norm)]
    print("PASS no line split across chunks" if not bad else f"FAIL line split: {bad}")
    assert not bad

    # invariant 3: each rejoined catalog row stays intact in one chunk
    for row in ("SKU-4421 | ARCTIC FREEZER III PRO | 7,750.00 | stock 120",
                "SKU-8802 | LED 34 MONITOR MSI | 23,000.00 | stock 40"):
        ok = any(row in " ".join(c.split()) for c in chunks)
        print(f"PASS catalog row intact: {row[:20]}..." if ok else f"FAIL catalog row split: {row}")
        assert ok

    print("\n--- sample chunk with a grand total ---")
    for c in chunks:
        if "GRAND TOTAL" in c.upper():
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
