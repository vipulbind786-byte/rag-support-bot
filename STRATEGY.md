# STRATEGY.md — Offline RAG Support Bot: repo context, marketing, sales

Last updated: 26 Oct 2026. This is the single source of truth for *why* this repo exists and *how* it turns into money. Code is in `backend/`, proof is in `STRESS_REPORT.md`, distribution plan is here + `PUBLISH_PLAN.md` + `LINKEDIN.md`.

## 1. What this repo actually is (grounded, no exaggeration)

**Product:** an offline RAG support bot for document-heavy businesses (GST invoices, warranty PDFs, policy files). It answers from the client's own corpus — and refuses when the answer isn't there.

**Verified capabilities (all tested, do not claim more):**
- Hybrid retrieval: ChromaDB dense vectors (Ollama `nomic-embed-text`) + TF-IDF + BM25, fused with RRF. Live, measured (20 chunks, 768-dim).
- Two-stage reranking: BAAI bge-reranker-base (CPU, $0). Relevance floor measured: true answers ≥ 0.065, off-topic probes ≤ 0.02, abstain threshold 0.05.
- Deterministic math: invoice totals, GST sums, policy-version conflicts computed in plain code. The 0.5B local model only phrases; never invents numbers. Temperature pinned to 0.
- Anti-hallucination double gate: corpus-coverage gate + reranker floor → abstain + human escalation.
- Eval: 14/14 golden eval (hit@5 = 1.00). Stress suite: 32/32 adversarial probes (multi-invoice scoping, OCR-corrupted amounts, contradictory policy versions, prompt-injection memos neutralized at ingest, Hinglish queries).
- Security: optional API-key auth, per-IP rate limit (30/min), 25MB upload cap.
- Runs fully local: no cloud, no API keys, no data leaving the machine.

**Honest gaps (state them, don't hide them):**
- Corpus is synthetic (Acme vendor bundle). No real client documents yet — this is the biggest sales gap.
- 0.5B model phrasing is simple; fine for support answers, not conversational polish.
- HF Space demo is static frontend; the real engine runs locally. Demo shows the pattern, not full local power.

## 2. Positioning (the one sentence that sells)

> "Most 'chat with your docs' bots guess. Mine refuses — every answer carries a page-number citation, and when the corpus can't answer, it escalates to a human. It runs on your own machine: your GST invoices never touch the cloud."

Why this works for Indian SMEs, CAs, clinics, legal: **data residency is the product, not a feature.** Price is secondary. GST docs are legally sensitive; "no cloud" removes the objection that kills the deal.

## 3. Target market (ranked by fit, not size)

1. **CA firms / bookkeepers** — highest invoice volume, understand GST, local trust matters. Pragramaj/UP first, then metros.
2. **Trading/logistics SMEs** — daily invoices, warranty claims, supplier disputes.
3. **Clinics/hospitals (small)** — policy + patient docs, zero tolerance for data leaving.
4. **Legal firms** — precedent/policy conflict detection maps directly to the stress-suite "14-day v2.0 supersedes 30-day rule" feature.

Not the market (yet): enterprise, big tech, US clients. Solo founder, local trust, low-cost — stay where the buyer can meet you.

## 4. Marketing funnel (current stage)

Stage: **proof built, audience zero.** Everything below is sequential.

1. **Launch content (done)** — GitHub repo live, HF Space demo live, 25s teaser (`promo/launch-teaser-v5.mp4`), LinkedIn launch post published 25 Oct.
2. **Network build (active)** — 8 LinkedIn connection requests/day (see `LINKEDIN.md`). Without reach, content is noise.
3. **Real-client pilot (next priority)** — take ONE messy real document set (a friend's GST invoice PDF, a clinic's actual warranty terms), run the stress suite on it. Converts "32/32 on our test corpus" → "worked on a client's actual docs." This is the single highest-ROI unverified claim in the sales story.
4. **Public proof loop** — every pilot becomes a short case note in the repo + a LinkedIn post. Repo doubles as sales collateral.
5. **Dev.to article** — "Why your RAG bot should refuse to answer." Dev.to traffic + backlink to repo. Write after 1st pilot so it has a real example.
6. **Product Hunt** — only after 2-3 posts and ~100 followers; PH launch needs an audience to land.

## 5. Sales model (how money actually comes in)

- **Freelance build (primary, ₹):** custom bot for a firm's document set: ₹15k-40k per deployment depending on corpus size and integration. Local install = trust story.
- **Monthly retainer (₹):** document updates, new policy versions, re-indexing: ₹2-5k/month per client. Recurring > one-off.
- **Demo-to-pitch flow:** demo link → 5-minute call on their docs → they upload ONE PDF and ask their own question → the refusal moment sells the product.
- **Open-source ≠ free labour:** repo is public for credibility. Paid = setup + custom corpus + support. Say this explicitly in DMs.

## 6. Weekly rhythm (non-negotiable)

| Day | Action |
|---|---|
| Mon | 8 connection requests (5 cat 1+2, 3 cat 3+4) |
| Tue | Reply all comments; post 1 short insight (stress-suite finding, screenshot) |
| Wed | 8 requests; DM 2 new CA/SME contacts with demo link |
| Thu | Reply; improve repo/README from feedback |
| Fri | 8 requests; write/update pilot case note |
| Sat | 8 requests; review: followers, replies, DM replies, pipeline |
| Sun | Off (founder rest, focus block 1-4 pm weekdays only) |

## 7. Metric gates (fail-closed governance, same as engineering)

Do NOT escalate spend/effort until gates pass:
- Gate A: 100 followers + ≥3 meaningful post replies → start Dev.to article.
- Gate B: ≥2 real DM conversations from content → attempt pilot #1.
- Gate C: 1 pilot with real docs passing stress suite → Product Hunt + paid pitch.
- Gate D: 2 pilots → pricing page in README.
Until Gate C, the honest status is "portfolio flagship with synthetic corpus" — never claim client validation.

## 8. Asset map

| Asset | Location | Status |
|---|---|---|
| Engine | `backend/` | verified, 32/32, 14/14 |
| Proof report | `STRESS_REPORT.md` | live in repo |
| Demo | https://vrxdr-rag-support-bot.static.hf.space/index.html | live |
| Repo | https://github.com/vipulbind786-byte/rag-support-bot | live |
| Teaser | `promo/launch-teaser-v5.mp4` | final, needs LinkedIn native upload |
| Publish copy | `PUBLISH_PLAN.md` | X copy ready, X auth pending |
| Networking playbook | `LINKEDIN.md` | active |
| This strategy | `STRATEGY.md` | source of truth |
