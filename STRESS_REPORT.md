# RAG Support Bot — Brutal Real-World Stress Test Report

Date: 2026-10-24 · Corpus: `acme_vendor_bundle.pdf` (18 pages) + `dirty_warranty_invoice.pdf` (4 pages) + `support_policy.txt`
Server: local `:8000`, model `qwen2.5:0.5b` (fully offline, zero paid API)

## Headline result
- **32/32 brutal probes PASS** (`eval/stress_test.py`)
- **14/14 golden eval PASS** (`eval/eval.py`)
- **Chunking + fact unit tests PASS** (`tests/`)
- **~21/32 answers computed in pure Python — LLM never called** for numeric/table/entity/version questions. Tiny model is only a phrasing layer.

The stress corpus was NOT clean. It deliberately contains: two interleaved columns, a multi-page catalog whose header only appears on page 1, TWO invoices with overlapping products in ONE PDF, contradictory policy versions (v1 30 days / 60% fee vs v2 14 days / 40% fee), a nested SLA matrix, a two-column FAQ, scanned/garbled amounts (`46,00000` = lost decimal), and a buried prompt-injection memo.

## What was broken, then fixed (each fix tested before moving on)

| # | Real failure mode | Symptom found | Fix | Status |
|---|-------------------|---------------|-----|--------|
| 1 | Multi-invoice PDF collapse | Invoice 0142 silently overwritten by 0377; wrong totals | Segment facts per `Invoice No`; scope numeric answer by invoice number in query | PASS |
| 2 | Scanned-number corruption | `46,00000`, `7,75000` (pypdf drops decimal dot) | `repair_scanned_numbers` + `repair_glued_cents` | PASS |
| 3 | pypdf table flattening | "stock of GPU 4070" → model invents wrong number | `rejoin_table_rows` + `parse_catalog`; answer in Python | PASS |
| 4 | Multi-level SLA header loss | P1/24x7 cells detached from labels | `parse_sla_matrix` re-attaches labels | PASS |
| 5 | Contradictory policy versions | Bot quoted superseded 30-day rule | `parse_policy_versions` picks highest version | PASS |
| 6 | Entity-attribute confusion | Mouse question got monitor's warranty | `parse_warranty_terms`; entity-scoped answer | PASS |
| 7 | Indirect prompt injection | Poisoned memo ranked 0.99 for unrelated queries; leaked token | Strip injection line **at ingest** (`neutralize_injection`) + output guard | PASS (0 leaks) |
| 8 | Live-state hallucination | "is my order delivered right now" answered from static docs | Live-state abstention guard | PASS |
| 9 | Off-topic leak | Would answer "best pizza in pune" | Coverage gate + rerank floor → abstain | PASS |
| 10 | Hinglish filler words | "invoice number kya hai" abstained wrongly | Hinglish stopwords | PASS |
| 11 | Router ordering bug | "which invoice has higher grand total" hijacked by grand-total branch | Single `deterministic_answer()` entry, comparison before generic numeric | PASS |

## Competitor-class problems vs scope creep

Checked against documented industry failure modes (LangChain/LlamaIndex issues, OWASP, Air Canada precedent). The classification:

IN-SCOPE (genuine, competitors' core problems — fixed):
- Table flattening / garbled numbers: **28% of enterprise factual questions need table data**; this is the #1 RAG accuracy lever. Not scope creep.
- Contradictory/stale policy versions: real legal precedent (Air Canada held liable for chatbot quoting an old policy). Not scope creep.
- Indirect prompt injection: OWASP LLM Top-10. Not scope creep.
- Abstention on unanswerable questions: the whole "know your limits" literature. Not scope creep.

OUT-OF-SCOPE for a tiny offline bot (would be scope creep — NOT built):
- Vision/OCR on image-only scans. We assume text-layer PDFs. Adding a VLM/OCR tier breaks the "offline, zero-cost" contract and is a different product.
- VLM table parsing (LlamaParse/Docling-class). We handle born-digital tables deterministically; full vision parsing is unnecessary here.
- Multi-language beyond Hinglish, per-page confidence scoring, fine-tuned classifiers. Nice-to-have, not required for the demo.

## Reproduce
```
python3 eval/stress_test.py     # 32/32
python3 eval/eval.py            # 14/14
python3 tests/test_facts.py     # ALL PASSED
python3 tests/test_chunking.py  # ALL PASSED
```

## Honest limits
- Corpus is text-layer PDFs. A pure scanned (image-only) PDF would return nothing — that's an OCR gap, not handled by design.
- The catalog parser expects the `SKU | name | price | stock` shape; unusual table layouts may not rejoin.
- 0.5B model still does procedural phrasing; retrieval is correct but wording can be imperfect. All verifiable facts are computed in code, so correctness does not depend on the model.
