"""
LLM layer: provider-agnostic. Uses OPENAI_API_KEY if set, else falls back to a
local Ollama/llama.cpp endpoint, else a deterministic extractive answer so the
demo still works with zero API cost (the $0-stack pattern from RAGDoll).
"""
import os
import re
import json
import urllib.request

LOCAL_LLM = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1/chat/completions")
LOCAL_MODEL = os.environ.get("LOCAL_LLM_MODEL", "qwen2.5:0.5b")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM = (
    "You are a support assistant. Answer ONLY from the provided context. "
    "If the context does not contain the answer, say so and suggest escalation. "
    "Cite sources as [source, page]. "
    "Keep monetary amounts exactly as written in the context, including the "
    "currency symbol and Indian digit grouping (e.g. ₹3,93,160.00) — never "
    "convert or reformat them."
)


def _build_prompt(question: str, hits: list[dict]) -> str:
    q_tokens = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) > 2}
    blocks = []
    for h in hits:
        lines = [l.strip() for l in h["text"].splitlines() if l.strip()]
        # put query-matching lines first so the small model sees the answer
        # before filler (e.g. product rows ahead of the totals block)
        matched = [l for l in lines if q_tokens & set(re.findall(r"[a-z0-9]+", l.lower()))]
        rest = [l for l in lines if l not in matched]
        blocks.append((len(matched), f"[{h['source']}, page {h['page'] or '-'}] " + "\n".join(matched + rest)))
    blocks.sort(key=lambda b: -b[0])  # most query-relevant block first
    ctx = "\n\n".join(b for _, b in blocks) or "(no documents indexed)"
    return f"Context:\n{ctx}\n\nQuestion: {question}"


def _extractive_answer(question: str, hits: list[dict]) -> str:
    """Zero-cost fallback: return the query-relevant LINES of the best hits."""
    if not hits:
        return "I could not find an answer in the indexed documents. Please escalate to a human agent."
    q_tokens = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) > 2}
    parts = []
    for h in hits[:3]:
        cite = f"{h['source']}" + (f", p.{h['page']}" if h['page'] else "")
        lines = [l.strip() for l in h["text"].splitlines() if l.strip()]
        matched = [l for l in lines if q_tokens & set(re.findall(r"[a-z0-9]+", l.lower()))]
        snippet = " ".join(matched or lines)[:200]
        parts.append(f"- {snippet} [{cite}]")
    return "Based on your documents:\n" + "\n".join(parts)


def _llm_call(prompt: str) -> str:
    if OPENAI_KEY:
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps({
                "model": OPENAI_MODEL,
                "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt}],
                "max_tokens": 500,
            }).encode(),
            headers={"Authorization": f"Bearer {OPENAI_KEY}",
                    "Content-Type": "application/json"},
        )
    else:
        req = urllib.request.Request(
            LOCAL_LLM,
            data=json.dumps({
                "model": LOCAL_MODEL,
                "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt}],
                "max_tokens": 500,
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[warn] LLM call failed ({e}); using extractive fallback")
        return ""


def answer_from_context(question: str, hits: list[dict], stream: bool = False):
    # Hard anti-hallucination guard: no relevant context -> never call the LLM.
    # This is the demo test clients ask for ("ask about blueberry policy").
    if not hits:
        return ("I could not find an answer in the indexed documents. "
                "Please escalate to a human agent.")
    answer = _llm_call(_build_prompt(question, hits))
    # degenerate LLM output or failure -> zero-cost extractive fallback
    if not answer or len(answer.strip()) < 20 or answer.strip() in ("[]", "null", "None"):
        answer = _extractive_answer(question, hits)
    return answer


def answer_stream(question: str, hits: list[dict]):
    """Generator: yields answer text in 20-char chunks for SSE."""
    answer = answer_from_context(question, hits)
    for i in range(0, len(answer), 20):
        yield answer[i:i + 20]
