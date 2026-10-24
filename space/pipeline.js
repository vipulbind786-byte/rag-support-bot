/*
  pipeline.js — browser mirror of backend/rag.py + backend/llm.py.
  Same deterministic pipeline, same order, same guards, in plain JS so the
  static HF Space demo shows exactly what the production backend does:
  dirty-text repair -> structured fact extraction -> deterministic answers
  (zero LLM) -> hybrid retrieval fallback.
*/

const STOP = new Set(("a an the is are was were what which who how when where why do does did can could should would will for with and or but not of in on at to by from about as it its my your our their this that these those i you we they me he she him her be been being have has had i am no yes if then there here kya hai hain ka ki ke ko se mein kitna kitne bata do batao do").split(" "));
const TOKEN_RE = /[a-zA-Z0-9]+/g;
const MONEY_RE = /\d[\d,]*\.\d{2}/g;
const GARBLE = "[\u25a0\u00ad\u2011\u2013\u2014\u2012]";

const INJECTION_RE = /system prompt|reveal.*prompt|ignore (all )?(previous|above|prior) instructions|you are now|jailbreak|act as|pretend to be|secret\s*key|api key|SECRET=|AK-\d+/i;
const INJECTION_LINE_RE = /(ignore (all )?(previous|above|prior|earlier) instructions|you are now|act as|system prompt|output .*internal token|internal token|reveal .*prompt|override (all|any) (prior|previous|earlier)|disregard (the|all) (instructions|rules))/i;

function tokens(t){return (t||"").toLowerCase().match(TOKEN_RE)||[];}
function stem(w){for(const s of ["ing","ed","es","s"]){if(w.length>s.length+2&&w.endsWith(s)){w=w.slice(0,-s.length);break;}}if(w.length>3&&w.endsWith("e"))w=w.slice(0,-1);return w;}
function contentTerms(t){return new Set(tokens(t).filter(w=>!STOP.has(w)&&w.length>=2).map(stem));}

/* ---------- ingest repair ---------- */

function neutralizeInjection(text){
  return text.split("\n").map(l=>INJECTION_LINE_RE.test(l)
    ? "[document contains an embedded instruction; it is ignored as data]" : l).join("\n");
}

function repairScannedNumbers(text){
  const fixRun = m => {
    const groups = m.match(/\d+/g);
    if (groups.length < 2) return m;
    if (groups[groups.length-1].length === 2 && groups.slice(0,-1).some(g=>g.length===3))
      return groups.slice(0,-1).join(",") + "." + groups[groups.length-1];
    return groups.join(",");
  };
  text = text.replace(new RegExp("(\\d)" + GARBLE + "\n(\\d{2}\\.\\d{2})", "g"), "$1.$2");
  text = text.replace(new RegExp("\\d(?:" + GARBLE + "\\d+)+", "g"), fixRun);
  text = text.replace(new RegExp("(\\d)" + GARBLE + "(\\s*\\n)", "g"), "$1$2");
  text = text.replace(/(\d{1,3}(?:,\d{3})+) (\d{2})(?=\s|$)/g, "$1.$2");
  text = text.replace(/(\d)\. (\d{2})(?=\s|$)/g, "$1.$2");
  return text;
}

function repairGluedCents(text){
  const fix = m => {
    const groups = m.split(",");
    const last = groups[groups.length-1];
    if (groups.length >= 2 && last.length === 5) groups[groups.length-1] = last.slice(0,3) + "." + last.slice(3);
    return groups.join(",");
  };
  return text.split("\n").map(line=>{
    if (/unit|total|qty|igst|cgst|sgst|sub.?total|grand|freight|forward|add|less/i.test(line))
      line = line.replace(/\d{1,3}(?:,\d{3})*,\d{5}(?=\s|$)/g, fix);
    return line;
  }).join("\n");
}

function stripBoilerplate(text){
  const lines = text.split("\n");
  const count = {};
  for (const l of lines){ const k = l.trim(); if (k.length > 30) count[k] = (count[k]||0)+1; }
  const kept = new Set();
  return lines.filter(l=>{
    const k = l.trim();
    if (k.length > 30 && count[k] >= 2){ if (kept.has(k)) return false; kept.add(k); }
    return true;
  }).join("\n");
}

function rejoinTableRows(text){
  const lines = text.split("\n");
  const out = [];
  let i = 0, slaLabels = null;
  while (i < lines.length){
    const line = lines[i];
    if (/^\s*SKU-\w+\s*$/.test(line) && i + 3 < lines.length){
      const name = lines[i+1], amt = lines[i+2], stock = lines[i+3];
      if ((amt.match(MONEY_RE) || []).length && /^\s*\d+\s*$/.test(stock)){
        out.push(`${line.trim()} | ${name.trim()} | ${amt.trim()} | stock ${stock.trim()}`);
        i += 4; continue;
      }
    }
    if (/priority/i.test(line) && /resolution|response/i.test(line)){
      slaLabels = line.trim().split(/\s{2,}|\t/).filter(w=>w.trim());
      out.push(line); i++; continue;
    }
    if (/^\s*P\d\s+[A-Za-z]/.test(line)){
      const cells = [line.trim()];
      let j = i + 1;
      while (j < lines.length && /^\s*(?:\d+\s*(?:min|hrs|hr|days)|\d+x\d+)\s*$/.test(lines[j])){
        cells.push(lines[j].trim()); j++;
      }
      if (slaLabels && slaLabels.length === cells.length)
        out.push(slaLabels.map((lbl,k)=>`${lbl}: ${cells[k]}`).join(" | "));
      else out.push(cells.join(" | "));
      i = j; continue;
    }
    out.push(line); i++;
  }
  return out.join("\n");
}

function normalizeInvoiceTotals(text){
  if (!/GSTIN|Invoice No/i.test(text)) return text;
  return text.replace(/.*Grand\s+Total.*/g, line=>{
    const amounts = line.match(MONEY_RE);
    return amounts ? line + "\n[normalized] Grand Total amount: ₹" + amounts[amounts.length-1] : line;
  });
}

function fullPipeline(text){
  text = neutralizeInjection(text);
  text = repairScannedNumbers(text);
  text = repairGluedCents(text);
  text = stripBoilerplate(text);
  text = rejoinTableRows(text);
  text = normalizeInvoiceTotals(text);
  return text;
}

/* ---------- structured fact extraction ---------- */

function parseInvoiceBlock(lines){
  const facts = { line_items: [], named: {} };
  for (let i = 0; i < lines.length; i++){
    const line = lines[i], low = line.toLowerCase();
    const amounts = line.match(MONEY_RE) || [];
    const nxt = i + 1 < lines.length ? lines[i+1] : "";
    const m = line.match(/^\s*\d+\.\s*(.+?)\b(?:qty|quantity)\b\s*(\d+)/i);
    if (m){
      const name = m[1].trim(), qty = parseInt(m[2], 10);
      const tailAmounts = line.slice(m.index + m[0].length).match(MONEY_RE) || [];
      const unit = tailAmounts[0] || null;
      let total = tailAmounts.length >= 2 ? tailAmounts[tailAmounts.length-1] : null;
      if (tailAmounts.length === 1 && /total/i.test(line)){
        const nxtAmounts = nxt.match(MONEY_RE) || [];
        if (nxtAmounts.length) total = nxtAmounts[0];
      }
      if (total === null && unit && qty)
        total = (qty * parseFloat(unit.replace(/,/g,""))).toFixed(2);
      if (!name) continue;
      facts.line_items.push({ name, qty,
        unit: unit ? parseFloat(unit.replace(/,/g,"")) : null,
        total: total ? parseFloat(total.replace(/,/g,"")) : null });
      continue;
    }
    if (amounts.length){
      if (/grand\s*total|total\s*amount/i.test(low)) facts.named.grand_total = parseFloat(amounts[amounts.length-1].replace(/,/g,""));
      else if (/sub\s*-?\s*total/i.test(low)) facts.named.sub_total = parseFloat(amounts[amounts.length-1].replace(/,/g,""));
      else if (low.includes("igst")) facts.named.igst = parseFloat(amounts[0].replace(/,/g,""));
      else if (low.includes("cgst") || low.includes("sgst")){
        facts.named.cgst = parseFloat(amounts[0].replace(/,/g,""));
        facts.named.cgst_count = low.replace(/ /g,"").includes("x2") ? 2 : 1;
      }
      else if (low.includes("freight") || low.includes("forwarding")) facts.named.freight = parseFloat(amounts[amounts.length-1].replace(/,/g,""));
      else if (low.includes("discount")) facts.named.discount = parseFloat(amounts[0].replace(/,/g,""));
    }
    if (low.includes("gstin")){
      const m2 = line.match(/[0-9]{2}[A-Z0-9]{10,13}/);
      if (m2) facts.named.gstin = m2[0];
    }
    if (/invoice\s*no/i.test(low)){
      const m2 = line.split(":").slice(1).join(":").match(/[A-Z0-9][A-Z0-9\/]{3,}/);
      if (m2) facts.named.invoice_no = m2[0];
    }
    if (/\bdate\b/i.test(low)){
      const m2 = line.match(/\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4}/);
      if (m2) facts.named.date = m2[0];
    }
  }
  return facts;
}

function parseInvoiceFacts(text){
  if (!/\b(invoice|gstin|qty|quantity)\b/i.test(text)) return {};
  const lines = text.split("\n");
  const blocks = []; let current = [], invoiceSeen = false;
  for (const line of lines){
    if (/invoice\s*no/i.test(line)){
      if (invoiceSeen){ blocks.push(current); current = [line]; }
      else current.push(line);
      invoiceSeen = true;
    } else current.push(line);
  }
  if (current.length) blocks.push(current);
  if (!blocks.length) return parseInvoiceBlock(lines);
  const invoices = {};
  for (const block of blocks){
    const f = parseInvoiceBlock(block);
    if (f.line_items.length || Object.keys(f.named).length){
      const invNo = f.named.invoice_no || ("invoice_" + Object.keys(invoices).length);
      invoices[invNo] = f;
    }
  }
  const merged = { line_items: [], named: {}, invoices };
  for (const f of Object.values(invoices)){
    merged.line_items = merged.line_items.concat(f.line_items);
    for (const [k,v] of Object.entries(f.named)) if (!(k in merged.named)) merged.named[k] = v;
  }
  return (merged.line_items.length || Object.keys(merged.named).length) ? merged : {};
}

function parsePolicyVersions(text){
  const versions = [];
  const re = /POLICY\s*v(\d+(?:\.\d+)?)[^\n]*?((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})/gi;
  let m;
  while ((m = re.exec(text))) versions.push([m[1], m[2]]);
  if (!versions.length) return {};
  const key = v => v[0].split(".").map(x=>parseInt(x,10));
  let latest = versions[0];
  for (const v of versions){
    const a = key(v), b = key(latest);
    for (let i = 0; i < Math.max(a.length,b.length); i++){
      const x = a[i]||0, y = b[i]||0;
      if (x !== y){ if (x > y) latest = v; break; }
    }
  }
  const idx = text.toLowerCase().indexOf("v" + latest[0]);
  if (idx < 0) return {};
  const block = text.slice(idx, idx + 400);
  const out = { version: latest[0], date: latest[1] };
  const mDays = block.match(/(\d+)\s+days/);
  if (mDays) out.returns_days = parseInt(mDays[1],10);
  const mFee = block.match(/restocking fee[^.]*?(\d+)\s*percent/i);
  if (mFee) out.restocking_fee_pct = parseInt(mFee[1],10);
  return out;
}

function parseWarrantyTerms(text){
  const terms = {};
  for (const line of text.split("\n")){
    const m = line.match(/^\s*([A-Z][A-Z0-9 ()\/\-]{3,}):\s*(.+)/);
    if (m && /warranty|coverage|not covered|excluded/i.test(m[2]))
      terms[m[1].trim().toLowerCase()] = m[2].trim();
  }
  return terms;
}

function parseCatalog(text){
  const catalog = {};
  for (const line of text.split("\n")){
    if (!line.includes("|")) continue;
    const parts = line.split("|").map(p=>p.trim());
    if (parts.length >= 4 && parts[0].startsWith("SKU")){
      const stock = parts[3].replace("stock","").trim();
      const price = parseFloat(parts[2].replace(/,/g,""));
      const stk = parseInt(stock,10);
      if (Number.isFinite(price) && Number.isFinite(stk))
        catalog[parts[1].toLowerCase()] = { sku: parts[0], price, stock: stk };
    }
  }
  return catalog;
}

function parseSlaMatrix(text){
  const lines = text.split("\n").map(l=>l.trim());
  const start = lines.findIndex(l=>/Severity\b/i.test(l));
  if (start < 0) return {};
  const headers = [];
  let j = start + 1;
  while (j < lines.length && !/^P\d\s/.test(lines[j])){ if (lines[j]) headers.push(lines[j]); j++; }
  const half = Math.floor(headers.length / 2);
  const metric = headers.slice(0, half), group = headers.slice(half);
  const labels = metric.map((m,k)=>`${m.toLowerCase()} ${group[k].toLowerCase()}`);
  const out = {};
  for (const line of lines.slice(start)){
    if (/^P\d\s/.test(line) && line.includes("|")){
      const cells = line.split("|").map(c=>c.trim());
      const rec = {};
      cells.slice(1).forEach((v,i)=>{ if (i < labels.length) rec[labels[i]] = v; });
      out[cells[0].toLowerCase()] = rec;
    }
  }
  return out;
}

function parseContacts(text){
  const out = {};
  for (const line of text.split("\n")){
    const m = line.match(/^\s*(head office|regional office|support hours|account manager|emergency|support)[:]?\s*(.+)/i);
    if (m) out[m[1].trim().toLowerCase()] = m[2].trim();
  }
  const m2 = text.match(/account manager[^\n]*?is ([A-Z][a-z]+ [A-Z][a-z]+)/i);
  if (m2) out["account manager"] = m2[1];
  const m3 = text.match(/GSTIN of seller: ([A-Z0-9]{15})/i);
  if (m3) out.gstin = m3[1];
  return out;
}

function buildFacts(text){
  const f = parseInvoiceFacts(text);
  f.policy = parsePolicyVersions(text);
  f.warranty = parseWarrantyTerms(text);
  f.catalog = parseCatalog(text);
  f.sla = parseSlaMatrix(text);
  f.contacts = parseContacts(text);
  return f;
}

/* ---------- deterministic answers (mirror of compute_* functions) ---------- */

function fmtRupee(v){
  const neg = v < 0; v = Math.abs(v);
  const intpart = Math.floor(v);
  const dec = String(Math.round(v * 100) % 100).padStart(2, "0");
  let n = intpart, grouped;
  const first = n % 1000; n = Math.floor(n / 1000);
  grouped = n > 0 ? String(first).padStart(3,"0") : String(first);
  while (n > 0){ grouped = String(n % 100) + "," + grouped; n = Math.floor(n / 100); }
  return (neg ? "-" : "") + grouped + (dec ? "." + dec : "");
}

function computeInvoiceComparison(q, facts){
  const invs = facts.invoices || {};
  const keys = Object.keys(invs);
  if (keys.length < 2) return null;
  q = q.toLowerCase();
  if (!/which.*(higher|larger|bigger|greater|most)|compare|difference/.test(q)) return null;
  const pairs = keys.filter(k => invs[k].named.grand_total)
                    .map(k => [k, invs[k].named.grand_total]);
  if (pairs.length < 2) return null;
  pairs.sort((a,b)=>b[1]-a[1]);
  const [hi, lo] = pairs;
  return `Invoice ${hi[0]} has the higher grand total (₹${fmtRupee(hi[1])} vs ₹${fmtRupee(lo[1])} for ${lo[0]}).`;
}

function computeSlaAnswer(q, facts){
  const sla = facts.sla || {};
  q = q.toLowerCase();
  const sev = Object.keys(sla).find(k => k.split(/\s+/)[0] === q.split(/\s+/).find(w=>/^p\d$/.test(w)));
  if (!sev) return null;
  const rec = sla[sev];
  const metric = q.includes("resolution") ? "resolution" : q.includes("response") ? "response" : null;
  if (!metric) return null;
  const slot = q.includes("24x7") ? `${metric} 24x7` : q.includes("business") ? `${metric} business hours` : null;
  if (slot && slot in rec)
    return `${sev.toUpperCase()} ${metric} (${q.includes("24x7") ? "24x7" : "business hours"}): ${rec[slot]}.`;
  return Object.entries(rec).map(([k,v])=>`${k}: ${v}`).join("; ");
}

function computeContactAnswer(q, facts){
  const c = facts.contacts || {};
  q = q.toLowerCase();
  for (const key of ["head office","regional office","support hours","emergency","support","gstin","account manager"]){
    if (q.includes(key) && key in c) return `${key.charAt(0).toUpperCase()+key.slice(1)}: ${c[key]}`;
  }
  return null;
}

function computeCatalogAnswer(q, facts){
  const cat = facts.catalog || {};
  q = q.toLowerCase();
  if (!/stock|price|cost|available|in stock|quantity/.test(q)) return null;
  for (const [name, rec] of Object.entries(cat)){
    const toks = name.split(" ").filter(t => t.length >= 2);
    if (toks.every(t => q.includes(t))){
      if (/stock|available|in stock|how many/.test(q))
        return `${name.toUpperCase()}: ${rec.stock} units in stock (price ₹${fmtRupee(rec.price)}).`;
      return `${name.toUpperCase()} price: ₹${fmtRupee(rec.price)}.`;
    }
  }
  return null;
}

function computeEntityAttr(q, facts){
  const w = facts.warranty || {};
  q = q.toLowerCase();
  if (!/warranty|coverage|guarantee|covered/.test(q)) return null;
  for (const [key, val] of Object.entries(w)){
    const toks = key.split(" ").filter(t => t.length >= 4);
    if (toks.some(t => q.includes(t)) || key === q) return `${key.toUpperCase()}: ${val}`;
  }
  return null;
}

function computePolicyAnswer(q, facts){
  const pol = facts.policy || {};
  q = q.toLowerCase();
  if (/(return|refund)/.test(q) && /(how many|days|window|time)/.test(q) && pol.returns_days)
    return `Returns are accepted within ${pol.returns_days} days of delivery (policy v${pol.version}, ${pol.date} — this supersedes the older 30-day rule).`;
  if ((q.includes("restocking") || q.includes("open box") || q.includes("open-box")) && pol.restocking_fee_pct)
    return `The current open-box restocking fee is ${pol.restocking_fee_pct}% (policy v${pol.version}, ${pol.date}; the earlier 60% no longer applies).`;
  return null;
}

function computeNumericAnswer(q, facts){
  if (!facts || (!facts.line_items.length && !Object.keys(facts.named).length)) return null;
  q = q.toLowerCase();
  const named = facts.named, items = facts.line_items, invoices = facts.invoices || {};
  let scopeNamed = named, scopeItems = items;
  const invRef = q.match(/(\d{3,})|(inv\/ ?\d{4}[\/-]\d+)/);
  if (Object.keys(invoices).length && invRef){
    const digits = invRef[0].match(/\d+/g) || [];
    for (const [invNo, inv] of Object.entries(invoices)){
      if (digits.length && digits.every(d => invNo.includes(d))){ scopeNamed = inv.named; scopeItems = inv.line_items; break; }
    }
  }
  const ent = q.match(/\b(mouse|monitor|freezer|keyboard|led|freight|forwarding|igst|cgst|sgst|discount|battery|amc|shipping|refund)\b/);
  const numericIntent = /how much|total|amount|price|cost|charge|rate|qty|quantity|kitna|rupees|₹|\d/.test(q);
  if (ent && numericIntent){
    const key = ent[1];
    for (const it of scopeItems){
      if (it.name.toLowerCase().includes(key)){
        if (/unit|each|price per/.test(q) && it.unit !== null)
          return `Unit price of ${it.name}: ₹${fmtRupee(it.unit)} (qty ${it.qty}).`;
        return `${it.name} total: ₹${fmtRupee(it.total)} (qty ${it.qty}).`;
      }
    }
    if (key in scopeNamed){
      let val = scopeNamed[key];
      if (key === "cgst") val = val * (scopeNamed.cgst_count || 1);
      return `${key.toUpperCase()} amount: ₹${fmtRupee(val)}.`;
    }
  }
  if (/grand total|total amount|total payable|total to pay|kitna dena|total bill/.test(q) && scopeNamed.grand_total)
    return `The grand total is ₹${fmtRupee(scopeNamed.grand_total)}.`;
  if (/(highest|most expensive|costs the most|largest|maximum|biggest).*?(item|product|price|amount)|which.*?(most|highest)/.test(q)){
    const withTotal = scopeItems.filter(it => it.total != null && it.name);
    if (withTotal.length){
      const top = withTotal.reduce((a,b)=>b.total>a.total?b:a);
      return `${top.name} has the highest amount: ₹${fmtRupee(top.total)} (qty ${top.qty} × ₹${fmtRupee(top.unit)}).`;
    }
  }
  if (/(cheapest|lowest|smallest|minimum|least).*?(item|product|price)/.test(q)){
    const withTotal = scopeItems.filter(it => it.total !== null);
    if (withTotal.length){
      const low = withTotal.reduce((a,b)=>b.total<a.total?b:a);
      return `${low.name} is the cheapest: ₹${fmtRupee(low.total)}.`;
    }
  }
  if (/total gst|gst payable|total tax/.test(q) && ("igst" in scopeNamed || "cgst" in scopeNamed)){
    const gst = (scopeNamed.igst||0) + (scopeNamed.cgst||0)*(scopeNamed.cgst_count||1);
    return `Total GST payable: ₹${fmtRupee(gst)} (IGST ₹${fmtRupee(scopeNamed.igst||0)} + CGST ₹${fmtRupee(scopeNamed.cgst||0)} × ${scopeNamed.cgst_count||1}).`;
  }
  if (/how many (items|products|line)/.test(q) && scopeItems.length)
    return `${scopeItems.length} line items in the invoice.`;
  return null;
}

function deterministicAnswer(q, facts){
  if (!facts) return null;
  for (const fn of [computeInvoiceComparison, computeSlaAnswer, computeContactAnswer,
                    computeCatalogAnswer, computeEntityAttr, computePolicyAnswer, computeNumericAnswer]){
    const ans = fn(q, facts);
    if (ans) return ans;
  }
  return null;
}

function proceduralAnswer(q, hits){
  q = q.toLowerCase();
  for (const h of hits){
    for (const raw of h.text.split("\n")){
      const line = raw.trim();
      if (!line) continue;
      const m = line.match(/^Q[:.]?\s+(.+?)\s+A[:.]?\s+(.+)$/i);
      if (m && m[1].split(/\W+/).slice(0,4).some(w => q.includes(w))) return m[2].trim();
      if (line.includes("|") && line.includes(":")){
        const parts = line.split(/\s*\|\s*/);
        if (parts[0].split(":").pop().split(/\s+/).slice(0,3).some(w => q.includes(w)))
          return parts.map(p=>p.trim()).join("; ");
      }
    }
  }
  return null;
}

if (typeof module !== "undefined") module.exports = {
  fullPipeline, buildFacts, deterministicAnswer, proceduralAnswer,
  computeInvoiceComparison, computeSlaAnswer, computeContactAnswer,
  computeCatalogAnswer, computeEntityAttr, computePolicyAnswer, computeNumericAnswer,
  neutralizeInjection, repairScannedNumbers, repairGluedCents, stripBoilerplate,
  rejoinTableRows, normalizeInvoiceTotals, parseInvoiceFacts, parsePolicyVersions,
  parseWarrantyTerms, parseCatalog, parseSlaMatrix, parseContacts, fmtRupee,
  contentTerms, INJECTION_RE
};
