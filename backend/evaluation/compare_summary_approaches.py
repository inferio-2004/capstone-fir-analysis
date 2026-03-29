#!/usr/bin/env python3
"""
Benchmark: OLD (600-char truncation) vs NEW (Stage-1 enriched) fir_summary
==========================================================================
The fir_summary is used for ONE purpose in Stage 2:
  → build_fact_query(fir_summary, ipc_sections)
  → produces a search query → searches Indian Kanoon → retrieves precedent cases

This benchmark:
  1. Builds search queries from both summaries (Groq API call)
  2. Searches Indian Kanoon with both queries (Kanoon API call)
  3. Compares which retrieved cases are more relevant to the FIR
"""
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from groq_prompts import build_fact_query
from indian_kanoon import search_kanoon, _clean_html, _is_actual_case, _extract_date_from_title

REPO = Path(__file__).resolve().parents[2]

# ── Load data ────────────────────────────────────────────────────────
with open(REPO / "src_dataset_files" / "fir_sample.json", encoding="utf-8") as f:
    fir_data = json.load(f)
with open(REPO / "output" / "fir_analysis_result_chains.json", encoding="utf-8") as f:
    analysis = json.load(f)


# ── Build both summaries ─────────────────────────────────────────────
old_summary = fir_data.get("incident_description", "")[:600]

def build_enriched_summary(analysis: dict, fir_data: dict) -> str:
    intent = analysis.get("analysis", {}).get("intent_identification", {})
    reasoning = analysis.get("analysis", {}).get("legal_reasoning", {})
    primary = intent.get("primary_intent", "Unknown")
    secondary = intent.get("secondary_intents", [])
    legal_basis = reasoning.get("legal_basis", "")
    parts = [f"Primary Intent: {primary}."]
    if secondary:
        parts.append(f"Related Intents: {', '.join(secondary)}.")
    if legal_basis:
        parts.append(legal_basis[:400])
    raw = fir_data.get("incident_description", "")
    if raw:
        joined = " ".join(parts)
        remaining = 600 - len(joined)
        if remaining > 50:
            parts.append(f"Incident excerpt: {raw[:remaining]}")
    return " ".join(parts)

new_summary = build_enriched_summary(analysis, fir_data)

# Extract IPC sections from analysis
mapped_sections = []
for s in analysis.get("applicable_statutes", []):
    prim = s.get("primary", {})
    if prim.get("law") == "IPC":
        mapped_sections.append(f"IPC Section {prim['section']}")
ipc_sections = [re.search(r"(\d+\w*)", s).group(1) for s in mapped_sections if re.search(r"(\d+\w*)", s)]

# Ground-truth keywords for relevance scoring (this is a robbery/snatching case)
RELEVANCE_KEYWORDS = [
    "robbery", "robbed", "snatching", "snatch",
    "theft", "stolen", "steal",
    "assault", "assaulted", "attacked",
    "hurt", "injury", "injured",
    "knife", "weapon", "armed",
    "dacoity", "dacoit",
    "loot", "looted", "looting",
    "394", "390", "356", "392", "397",
    "gold", "jewel", "ornament", "chain",
    "mobile", "phone",
    "conviction", "convicted", "guilty", "sentenced",
]

sep = "=" * 72


def extract_cases_from_result(result: dict, label: str, limit: int = 10) -> list[dict]:
    """Extract actual court cases from Kanoon search result."""
    cases = []
    if result.get("error"):
        return cases
    for doc in result.get("docs", []):
        if len(cases) >= limit:
            break
        title = _clean_html(doc.get("title", "Unknown"))
        docsource = doc.get("docsource", "Unknown")
        if not _is_actual_case(title, docsource):
            continue
        snippet = _clean_html(doc.get("headline", ""))[:500]
        cases.append({
            "tid": doc.get("tid"),
            "title": title,
            "court": docsource,
            "snippet": snippet,
            "label": label,
        })
    return cases


def relevance_score(cases: list[dict]) -> dict:
    """Score case relevance: how many case titles+snippets match crime keywords."""
    total_hits = 0
    per_case = []
    for c in cases:
        text = f"{c['title']} {c['snippet']}".lower()
        hits = [kw for kw in RELEVANCE_KEYWORDS if kw in text]
        total_hits += len(hits)
        per_case.append({"title": c["title"][:80], "hits": len(hits), "keywords": hits})
    avg = total_hits / len(cases) if cases else 0
    return {"total_hits": total_hits, "avg_per_case": avg, "per_case": per_case}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 1: Generate search queries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(sep)
print("  STEP 1: Generate Indian Kanoon search queries from both summaries")
print(sep)

print(f"\n  OLD summary ({len(old_summary)} chars):")
print(f"    {old_summary[:100]}...")
print(f"\n  NEW summary ({len(new_summary)} chars):")
print(f"    {new_summary[:100]}...")

print(f"\n  IPC sections: {ipc_sections}")

print("\n  Calling Groq → build_fact_query(OLD)...")
t0 = time.time()
query_old = build_fact_query(old_summary, ipc_sections)
t_old = time.time() - t0

print("  Calling Groq → build_fact_query(NEW)...")
t0 = time.time()
query_new = build_fact_query(new_summary, ipc_sections)
t_new = time.time() - t0

print(f"\n  OLD query ({t_old:.2f}s): \"{query_old}\"")
print(f"  NEW query ({t_new:.2f}s): \"{query_new}\"")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 2: Search Indian Kanoon with both queries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{sep}")
print("  STEP 2: Search Indian Kanoon with both queries")
print(sep)

print(f"\n  Searching Kanoon with OLD query: \"{query_old}\"")
result_old = search_kanoon(query_old)
cases_old = extract_cases_from_result(result_old, "OLD")
print(f"  → Retrieved {len(cases_old)} actual court cases")

print(f"\n  Searching Kanoon with NEW query: \"{query_new}\"")
result_new = search_kanoon(query_new)
cases_new = extract_cases_from_result(result_new, "NEW")
print(f"  → Retrieved {len(cases_new)} actual court cases")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 3: Display retrieved cases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{sep}")
print("  STEP 3: Retrieved cases comparison")
print(sep)

print(f"\n  ── Cases from OLD query ──")
for i, c in enumerate(cases_old, 1):
    print(f"    {i}. {c['title'][:90]}")
    print(f"       Court: {c['court']}")
    if c["snippet"]:
        print(f"       Snippet: {c['snippet'][:120]}...")

print(f"\n  ── Cases from NEW query ──")
for i, c in enumerate(cases_new, 1):
    print(f"    {i}. {c['title'][:90]}")
    print(f"       Court: {c['court']}")
    if c["snippet"]:
        print(f"       Snippet: {c['snippet'][:120]}...")

# Check overlap
tids_old = {c["tid"] for c in cases_old}
tids_new = {c["tid"] for c in cases_new}
overlap = tids_old & tids_new
only_old = tids_old - tids_new
only_new = tids_new - tids_old

print(f"\n  Overlap: {len(overlap)} cases appear in BOTH result sets")
print(f"  Only in OLD: {len(only_old)} unique cases")
print(f"  Only in NEW: {len(only_new)} unique cases")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 4: Relevance scoring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{sep}")
print("  STEP 4: Relevance scoring (keyword matching against FIR crime type)")
print(f"  Ground truth: This is a robbery/snatching case with knife, assault, gold jewelry theft")
print(f"  Keywords checked: {RELEVANCE_KEYWORDS[:10]}... ({len(RELEVANCE_KEYWORDS)} total)")
print(sep)

rel_old = relevance_score(cases_old)
rel_new = relevance_score(cases_new)

print(f"\n  ── OLD query relevance ──")
print(f"    Total keyword hits: {rel_old['total_hits']}")
print(f"    Avg hits per case:  {rel_old['avg_per_case']:.1f}")
for pc in rel_old["per_case"]:
    print(f"    [{pc['hits']:2d} hits] {pc['title']}")
    if pc["keywords"]:
        print(f"             Keywords: {', '.join(pc['keywords'][:8])}")

print(f"\n  ── NEW query relevance ──")
print(f"    Total keyword hits: {rel_new['total_hits']}")
print(f"    Avg hits per case:  {rel_new['avg_per_case']:.1f}")
for pc in rel_new["per_case"]:
    print(f"    [{pc['hits']:2d} hits] {pc['title']}")
    if pc["keywords"]:
        print(f"             Keywords: {', '.join(pc['keywords'][:8])}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FINAL VERDICT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{sep}")
print("  FINAL COMPARISON")
print(sep)

print(f"""
  Metric                        OLD (truncation)    NEW (enriched)     Winner
  ───────────────────────────────────────────────────────────────────────────
  Cases retrieved                {len(cases_old):<19d} {len(cases_new):<18d} {'NEW' if len(cases_new) > len(cases_old) else 'OLD' if len(cases_old) > len(cases_new) else 'TIE'}
  Total keyword hits             {rel_old['total_hits']:<19d} {rel_new['total_hits']:<18d} {'NEW' if rel_new['total_hits'] > rel_old['total_hits'] else 'OLD' if rel_old['total_hits'] > rel_new['total_hits'] else 'TIE'}
  Avg relevance per case         {rel_old['avg_per_case']:<19.1f} {rel_new['avg_per_case']:<18.1f} {'NEW' if rel_new['avg_per_case'] > rel_old['avg_per_case'] else 'OLD' if rel_old['avg_per_case'] > rel_new['avg_per_case'] else 'TIE'}
  Unique cases (not in other)    {len(only_old):<19d} {len(only_new):<18d} {'NEW' if len(only_new) > len(only_old) else 'OLD' if len(only_old) > len(only_new) else 'TIE'}
""")

scores = {"OLD": 0, "NEW": 0, "TIE": 0}
for old_v, new_v in [
    (len(cases_old), len(cases_new)),
    (rel_old["total_hits"], rel_new["total_hits"]),
    (rel_old["avg_per_case"], rel_new["avg_per_case"]),
]:
    if old_v > new_v: scores["OLD"] += 1
    elif new_v > old_v: scores["NEW"] += 1
    else: scores["TIE"] += 1

winner = "NEW (Stage-1 enriched)" if scores["NEW"] > scores["OLD"] else \
         "OLD (600-char truncation)" if scores["OLD"] > scores["NEW"] else "TIE"
print(f"  RESULT:  OLD wins {scores['OLD']}  |  NEW wins {scores['NEW']}  |  TIE {scores['TIE']}")
print(f"  WINNER:  {winner}")
print()

print("  NOTE: The fir_summary ONLY affects Strategy 1 (LLM-generated query).")
print("  Strategy 2 ('Section XXX IPC accused conviction sentence') and Strategy 3")
print("  are section-based and identical regardless of summary approach.")
print("  So the summary's impact is limited to the FIRST Kanoon search call.")
