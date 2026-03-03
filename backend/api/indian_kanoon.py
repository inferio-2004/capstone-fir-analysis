#!/usr/bin/env python3
"""
Indian Kanoon API — Case Law Search & Pipeline Orchestration (Stage 2)
======================================================================
Searches Indian Kanoon for real case law matching IPC sections from Stage 1,
fetches judgment text, and delegates to groq_prompts for summarisation,
verdict prediction, and section-influence ranking.

API: https://api.indiankanoon.org  (500 free calls/day)
"""

import os
import re

import requests
from dotenv import load_dotenv

from kanoon_cache import cache_key, load_cache, save_cache
from groq_prompts import (
    summarize_case,
    predict_verdict,
    rank_section_influence,
    build_fact_query,
)

load_dotenv()

# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------
KANOON_BASE_URL = "https://api.indiankanoon.org"
KANOON_API_KEY = os.environ.get("KANOON_API_KEY", "")

MAX_PER_SECTION = 2       # cases per IPC section
MAX_TOTAL_CASES = 5       # total cases to fetch & summarise


# ---------------------------------------------------------------------------
#  Low-level helpers
# ---------------------------------------------------------------------------
def _headers() -> dict:
    return {"Authorization": f"Token {KANOON_API_KEY}", "Accept": "application/json"}


def _clean_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def _extract_date_from_title(title: str) -> str:
    m = re.search(r"on\s+(\d{1,2}\s+\w+,?\s+\d{4})", title)
    if m:
        return m.group(1)
    m = re.search(r"\((\d{4})\)", title)
    return m.group(1) if m else ""


def _extract_ipc_sections(mapped_sections: list[str]) -> list[str]:
    sections = []
    for s in mapped_sections:
        m = re.match(r"(?:IPC|Indian Penal Code)\s+(?:Section\s+)?(\d+[A-Za-z]*)", s.strip(), re.IGNORECASE)
        if m:
            sections.append(m.group(1))
    return list(dict.fromkeys(sections))


def _is_actual_case(title: str, docsource: str) -> bool:
    """Return True only if the document looks like a court judgment."""
    title_lower = title.lower()
    source_lower = docsource.lower()

    if re.match(r"^Section\s+\d+", title, re.IGNORECASE):
        return False

    skip_keywords = [
        "penal code", "ranbir penal code", "criminal procedure code",
        "- act", "- rules", "- regulation", "- ordinance", "- bill",
        "constitution of india", "bare act",
    ]
    for kw in skip_keywords:
        if kw in title_lower or kw in source_lower:
            return False

    act_sources = ["union of india - section", "state of ", "central government"]
    for src in act_sources:
        if source_lower.startswith(src) and "vs" not in title_lower:
            return False

    if docsource.endswith("- Act") or docsource.endswith("- Code"):
        return False

    return True


# ---------------------------------------------------------------------------
#  Kanoon API calls
# ---------------------------------------------------------------------------
def search_kanoon(query: str, page: int = 0) -> dict:
    ck = cache_key(f"search:{query}:p{page}")
    cached = load_cache(ck)
    if cached:
        return cached
    try:
        resp = requests.post(f"{KANOON_BASE_URL}/search/", headers=_headers(),
                             data={"formInput": query, "pagenum": page}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        save_cache(ck, data)
        return data
    except requests.exceptions.HTTPError:
        return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}", "docs": []}
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "docs": []}


def get_doc_detail(tid: int) -> dict:
    """Fetch full judgment text from Indian Kanoon by TID."""
    ck = cache_key(f"doc:{tid}")
    cached = load_cache(ck)
    if cached:
        return cached
    try:
        resp = requests.post(f"{KANOON_BASE_URL}/doc/{tid}/", headers=_headers(), timeout=20)
        resp.raise_for_status()
        data = resp.json()
        save_cache(ck, data)
        return data
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
#  Main pipeline: search → fetch docs → summarise → predict verdict
# ---------------------------------------------------------------------------
def search_and_analyze(
    mapped_sections: list[str],
    fir_summary: str = "",
    max_per_section: int = MAX_PER_SECTION,
    max_total: int = MAX_TOTAL_CASES,
) -> dict:
    """
    Full Stage 2 pipeline:
    1. Search Indian Kanoon using fact-enriched queries with IPC sections
    2. Fetch judgment text for top cases
    3. Summarise each case with Groq LLM (via groq_prompts)
    4. Predict verdict/punishment based on all precedents
    5. Rank section influence on verdict
    """
    print("\n" + "=" * 80)
    print("INDIAN KANOON API - EXACT INPUTS")
    print("=" * 80)
    print(f"\n[INPUT 1] mapped_sections (from Stage 1):")
    for i, sec in enumerate(mapped_sections, 1):
        print(f"  {i}. {sec}")
    print(f"\n[INPUT 2] fir_summary (first 600 chars):")
    print(f"  {fir_summary}")
    print("\n" + "=" * 80)

    if not KANOON_API_KEY:
        return {"status": "error", "sections_searched": [], "cases": [],
                "verdict_prediction": None, "api_calls_used": 0,
                "error": "KANOON_API_KEY is not set in .env"}

    ipc_sections = _extract_ipc_sections(mapped_sections)
    print(f"\n[STEP 1] Extracted IPC sections: {ipc_sections}")
    if not ipc_sections and not fir_summary:
        return {"status": "no_results", "sections_searched": [], "cases": [],
                "verdict_prediction": None, "api_calls_used": 0, "error": None}

    all_cases: list[dict] = []
    seen_tids: set = set()
    api_calls = 0

    def _add_cases_from_result(result, section_label, limit):
        added = 0
        if result.get("error"):
            print(f"[Kanoon] Search error: {result['error']}")
            return added
        for doc in result.get("docs", []):
            if added >= limit or len(all_cases) >= max_total:
                break
            tid = doc.get("tid")
            if tid in seen_tids:
                continue
            title = _clean_html(doc.get("title", "Unknown Case"))
            docsource = doc.get("docsource", "Unknown Court")
            if not _is_actual_case(title, docsource):
                continue
            seen_tids.add(tid)
            all_cases.append({
                "title": title, "tid": tid, "court": docsource,
                "snippet": _clean_html(doc.get("headline", ""))[:500],
                "section": section_label,
                "date": _extract_date_from_title(title),
                "url": f"https://indiankanoon.org/doc/{tid}/" if tid else "",
                "summary": "",
            })
            added += 1
        return added

    # Strategy 1: LLM-crafted fact+section query
    fact_query = build_fact_query(fir_summary, ipc_sections) if fir_summary else ""
    if fact_query:
        print(f"\n[KANOON SEARCH QUERY 1 - LLM-generated]\n  Query: \"{fact_query}\"")
        result = search_kanoon(fact_query)
        api_calls += 1
        _add_cases_from_result(result, "Fact-match", min(3, max_total))

    # Strategy 2: Section + "conviction accused"
    for i, sec in enumerate(ipc_sections, 1):
        if len(all_cases) >= max_total:
            break
        query = f"Section {sec} IPC accused conviction sentence"
        print(f"\n[KANOON SEARCH QUERY {i + 1}]\n  Query: \"{query}\"")
        result = search_kanoon(query)
        api_calls += 1
        _add_cases_from_result(result, f"IPC {sec}", max_per_section)

    # Strategy 3: Broader fallback
    if len(all_cases) < 2:
        for sec in ipc_sections:
            if len(all_cases) >= max_total:
                break
            query = f"Section {sec} IPC guilty punishment"
            result = search_kanoon(query)
            api_calls += 1
            _add_cases_from_result(result, f"IPC {sec}", max_per_section)

    if not all_cases:
        return {"status": "no_results", "sections_searched": [f"IPC {s}" for s in ipc_sections],
                "cases": [], "verdict_prediction": None, "api_calls_used": api_calls, "error": None}

    # Fetch judgment text + summarise
    print(f"[Kanoon] Fetching & summarizing {len(all_cases)} cases...")
    for case in all_cases:
        tid = case["tid"]
        if not tid:
            continue
        doc_data = get_doc_detail(tid)
        api_calls += 1
        if doc_data.get("error"):
            continue
        doc_text = _clean_html(doc_data.get("doc", "")) or case["snippet"]
        case["summary"] = summarize_case(case["title"], doc_text, case["section"])

    # Predict verdict
    print("[Kanoon] Predicting verdict based on precedents...")
    cases_with_summaries = [c for c in all_cases if c.get("summary")]
    verdict = predict_verdict(
        fir_summary=fir_summary,
        ipc_sections=[f"IPC {s}" for s in ipc_sections],
        case_summaries=cases_with_summaries or all_cases,
    )

    # Rank section influence
    print("[Kanoon] Ranking section influence on verdict...")
    section_influence = rank_section_influence(
        mapped_sections=mapped_sections,
        fir_summary=fir_summary,
        verdict=verdict,
        case_summaries=cases_with_summaries or all_cases,
    )

    return {
        "status": "success" if cases_with_summaries else "partial",
        "sections_searched": [f"IPC {s}" for s in ipc_sections],
        "cases": all_cases,
        "verdict_prediction": verdict,
        "section_influence": section_influence,
        "api_calls_used": api_calls,
        "error": None,
    }


# ---------------------------------------------------------------------------
#  Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Indian Kanoon + Groq Verdict Prediction — Test")
    print("=" * 60)

    test_sections = ["IPC 356", "IPC 390", "IPC 394"]
    test_fir = (
        "On 2025-06-20 at around 09:15 PM, the victim was assaulted near "
        "the market by an unknown person who snatched her handbag containing "
        "cash and documents."
    )

    result = search_and_analyze(test_sections, test_fir)

    print(f"\nStatus: {result['status']}")
    print(f"Sections: {result['sections_searched']}")
    print(f"API calls: {result['api_calls_used']}")
    print(f"Cases: {len(result['cases'])}")

    for i, c in enumerate(result["cases"], 1):
        print(f"\n  {i}. {c['title']}")
        print(f"     Court: {c['court']} | {c['section']}")
        if c["summary"]:
            print(f"     Summary: {c['summary'][:250]}...")

    vp = result.get("verdict_prediction", {})
    if vp:
        print(f"\n{'=' * 60}")
        print(f"  Verdict:    {vp.get('predicted_verdict', 'N/A')}")
        print(f"  Punishment: {vp.get('predicted_punishment', 'N/A')}")

