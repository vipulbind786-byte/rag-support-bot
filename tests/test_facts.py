"""Deterministic invoice-fact regression test. Run: python3 tests/test_facts.py
No server, no LLM. Confirms numeric answers never depend on the model.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from rag import parse_invoice_facts, repair_scanned_numbers, strip_boilerplate, normalize_invoice_totals
from llm import compute_numeric_answer, _fmt_rupee

TEXT = """
ITEM LIST
1. ARCTIC FREEZER III PRO 360 BLACK     qty 50   unit 7,750.00   total 3,87,500.00   (HSN 84145099)
2. LED 34" MONITOR MSI 342CQRF   qty 2   unit 23,000.00   total
46,000.00  warranty 2 yrs from delivery
3. MOUSE LOGITECH G502 X      qty 3   1,500.00 each   total 4,500.00
4. AULA KEYBOARD MECHANICAL F99PROGRN  qty 1   7,600.00
Sub Total : 4,36,500.00
Add  : Freight & Forwarding Charge 41,260.00
Less : Early buyer discount Rs 84,600
IGST @ 18% on goods: 59,973.59
CGST 9% each: 29,986.79  x2
GRAND TOTAL: 3,93,160.00  (Rupees Three Lakh Ninety Three Thousand One Hundred Sixty Only)
GSTIN of seller: 27AABCV9876K3PQ
Invoice No: BE/2026/04-781
"""

facts = parse_invoice_facts(TEXT)

def check(cond, label):
    print(("PASS " if cond else "FAIL ") + label)
    return cond

ok = 0
ok += check(len(facts["line_items"]) == 4, "4 line items parsed")
totals = {i["name"]: i["total"] for i in facts["line_items"]}
ok += check(totals.get("ARCTIC FREEZER III PRO 360 BLACK") == 387500.0, "freezer total 387500")
ok += check(totals.get('LED 34" MONITOR MSI 342CQRF') == 46000.0, "monitor total 46000")
ok += check(totals.get("MOUSE LOGITECH G502 X") == 4500.0, "mouse total 4500")
ok += check(facts["named"]["grand_total"] == 393160.0, "grand total 393160")
ok += check(facts["named"]["igst"] == 59973.59, "igst 59973.59")

ans = compute_numeric_answer("which product has the highest amount", facts)
ok += check(ans and "ARCTIC" in ans and "3,87,500.00" in ans, "highest = ARCTIC 3,87,500.00")
ok += check(compute_numeric_answer("total gst payable", facts) == "Total GST payable: ₹1,19,947.17 (IGST ₹59,973.59 + CGST ₹29,986.79 × 2).", "total gst 1,19,947.17")
ok += check("4,500.00" in compute_numeric_answer("mouse ka total amount", facts), "mouse via LLM-free path")
ok += check(compute_numeric_answer("how many line items", facts) == "4 line items in the invoice.", "line count")
ok += check(compute_numeric_answer("what is the warranty on electronics", facts) is None, "procedural query returns None (goes to LLM)")
ok += check(_fmt_rupee(46000.0) == "46,000.00", "rupee grouping 46000")
ok += check(_fmt_rupee(100000.0) == "1,0,000.00", "rupee grouping lakh")

TOTAL = 13
print("ALL FACT TESTS PASSED" if ok == TOTAL else f"{TOTAL-ok} failures")
