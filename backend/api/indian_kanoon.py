#!/usr/bin/env python3
"""
Indian Kanoon API + Groq LLM — Case Law Search, Summarization & Verdict Prediction
====================================================================================
Stage 2: Searches Indian Kanoon for real case law matching IPC sections from
Stage 1, fetches judgment text, summarizes each case with Groq LLM, and
predicts likely verdict/punishment for the accused.

API: https://api.indiankanoon.org  (500 free calls/day)
LLM: Groq (llama-3.1-8b-instant)
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import Optional

import requests
from groq import Groq
from dotenv import load_dotenv
from model_config import get_preferred_groq_model

load_dotenv()

# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------
KANOON_BASE_URL = "https://api.indiankanoon.org"
KANOON_API_KEY = os.environ.get("KANOON_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "output" / "kanoon_cache"

MAX_PER_SECTION = 2       # cases per IPC section
MAX_TOTAL_CASES = 5       # total cases to fetch & summarize
GROQ_MODEL = get_preferred_groq_model("llama-3.1-8b-instant")


# ---------------------------------------------------------------------------
#  Low-level helpers
# ---------------------------------------------------------------------------
def _headers() -> dict:
    return {"Authorization": f"Token {KANOON_API_KEY}", "Accept": "application/json"}


def _cache_key(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()


def _load_cache(key: str) -> Optional[dict]:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save_cache(key: str, data):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / f"{key}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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


# ---------------------------------------------------------------------------
#  Kanoon API
# ---------------------------------------------------------------------------
def search_kanoon(query: str, page: int = 0) -> dict:
    ck = _cache_key(f"search:{query}:p{page}")
    cached = _load_cache(ck)
    if cached:
        return cached
    try:
        resp = requests.post(f"{KANOON_BASE_URL}/search/", headers=_headers(),
                             data={"formInput": query, "pagenum": page}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        _save_cache(ck, data)
        return data
    except requests.exceptions.HTTPError:
        return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}", "docs": []}
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "docs": []}


def get_doc_detail(tid: int) -> dict:
    """Fetch full judgment text from Indian Kanoon by TID."""
    ck = _cache_key(f"doc:{tid}")
    cached = _load_cache(ck)
    if cached:
        return cached
    try:
        resp = requests.post(f"{KANOON_BASE_URL}/doc/{tid}/", headers=_headers(), timeout=20)
        resp.raise_for_status()
        data = resp.json()
        _save_cache(ck, data)
        return data
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
#  Groq LLM
# ---------------------------------------------------------------------------
_groq_client = None


def _get_groq():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def summarize_case(case_title: str, case_text: str, ipc_section: str) -> str:
    """Summarize a court judgment in 3-5 sentences using Groq LLM."""
    truncated = case_text[:3500]
    prompt = f"""Summarize this Indian court judgment concisely in 3-5 sentences.
Focus on: what the case was about, the charges (especially {ipc_section}),
the court's decision, and the punishment/sentence given.

Case: {case_title}
Judgment text:
{truncated}

Summary:"""

    try:
        resp = _get_groq().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a legal analyst who summarizes Indian court judgments concisely. Focus on charges, decisions, and sentences."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Kanoon] Groq summarization error: {e}")
        return ""


def predict_verdict(fir_summary: str, ipc_sections: list[str], case_summaries: list[dict]) -> dict:
    """Predict likely verdict & punishment based on FIR + retrieved precedents."""
    cases_ctx = ""
    for i, cs in enumerate(case_summaries, 1):
        cases_ctx += (
            f"\n[Case {i}] {cs['title']} (Section {cs['section']}, Court: {cs['court']})\n"
            f"  Summary: {cs['summary']}\n"
        )

    sections_str = ", ".join(ipc_sections)

    prompt = f"""You are a senior Indian legal analyst. Based on the FIR details and real court
precedents below, predict the most likely verdict and punishment for the accused.

FIR DETAILS:
{fir_summary}

APPLICABLE IPC SECTIONS: {sections_str}

REAL COURT PRECEDENTS FROM INDIAN KANOON:
{cases_ctx}

Provide your prediction in this exact JSON format (no markdown, no code fences):
{{
  "predicted_verdict": "Likely Guilty / Likely Acquittal / Mixed",
  "predicted_punishment": "Specific punishment prediction based on precedents",
  "punishment_range": "Minimum to maximum sentence range under the applicable sections",
  "bail_likelihood": "High / Medium / Low — with brief reasoning",
  "confidence": 0.75,
  "reasoning": "2-3 sentence explanation citing the precedent cases"
}}"""

    try:
        resp = _get_groq().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a senior Indian criminal law expert. Predict verdicts based on real precedents. Return ONLY valid JSON, no markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "predicted_verdict": "Analysis completed",
            "predicted_punishment": raw if raw else "Could not parse LLM response",
            "punishment_range": "N/A",
            "bail_likelihood": "N/A",
            "confidence": 0,
            "reasoning": "LLM output was not valid JSON — raw response shown above.",
        }
    except Exception as e:
        print(f"[Kanoon] Verdict prediction error: {e}")
        return {
            "predicted_verdict": "Error",
            "predicted_punishment": str(e),
            "punishment_range": "N/A",
            "bail_likelihood": "N/A",
            "confidence": 0,
            "reasoning": f"LLM call failed: {e}",
        }


def rank_section_influence(
    mapped_sections: list[str],
    fir_summary: str,
    verdict: dict,
    case_summaries: list[dict],
) -> list[dict]:
    """Rank each applicable section by how much it is likely to influence the verdict.

    Returns a list of dicts sorted by influence_score (highest first):
      { section, law, influence_score (0-100), influence_level, reasoning }
    """
    if not mapped_sections:
        return []

    sections_str = ", ".join(mapped_sections)
    cases_ctx = ""
    for i, cs in enumerate(case_summaries[:5], 1):
        cases_ctx += f"\n[Case {i}] {cs.get('title','')} (Section {cs.get('section','')})\n  Summary: {cs.get('summary','')}\n"

    verdict_str = json.dumps(verdict, indent=2) if verdict else "No verdict prediction available"

    prompt = f"""You are a senior Indian criminal law analyst. Given the FIR details, the 
applicable sections, real court precedents, and the predicted verdict below, 
rank EACH applicable section by how strongly it influences the verdict outcome.

FIR DETAILS:
{fir_summary}

APPLICABLE SECTIONS: {sections_str}

PRECEDENT CASES:
{cases_ctx}

PREDICTED VERDICT:
{verdict_str}

For EACH section listed in APPLICABLE SECTIONS, provide:
- influence_score: 0-100 (100 = most influential on the final verdict)
- influence_level: "Primary" / "Supporting" / "Minor"
- reasoning: 1-2 sentences explaining why this section carries that level of influence

Return ONLY valid JSON array (no markdown, no code fences):
[
  {{
    "section": "IPC 394",
    "influence_score": 90,
    "influence_level": "Primary",
    "reasoning": "This section carries the heaviest punishment and is the main charge."
  }}
]"""

    try:
        resp = _get_groq().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a senior Indian criminal law expert. Return ONLY valid JSON array, no markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        rankings = json.loads(raw)

        # Normalise and sort
        for r in rankings:
            r["influence_score"] = max(0, min(100, int(r.get("influence_score", 0))))
        rankings.sort(key=lambda r: r["influence_score"], reverse=True)
        return rankings
    except Exception as e:
        print(f"[Kanoon] Section influence ranking error: {e}")
        # Fallback: equal weight for every section
        n = len(mapped_sections)
        return [
            {
                "section": s,
                "influence_score": round(100 / n),
                "influence_level": "Unknown",
                "reasoning": "Could not compute influence — LLM call failed.",
            }
            for s in mapped_sections
        ]


def _build_fact_query(fir_summary: str, ipc_sections: list[str]) -> str:
    """Use Groq LLM to distill FIR facts into an optimal Indian Kanoon search query."""
    if not fir_summary or not GROQ_API_KEY:
        return ""
    sections_str = ", ".join([f"IPC {s}" for s in ipc_sections]) if ipc_sections else "unknown"
    prompt = f"""You are an expert at searching Indian Kanoon (indiankanoon.org) for relevant criminal case judgments.

Given this FIR, create a search query that would find SIMILAR CRIMINAL CASES (actual court judgments, NOT statutes or Acts).

FIR: {fir_summary[:400]}
Applicable Sections: {sections_str}

Rules:
- Include the specific IPC section numbers (e.g. "Section 323 IPC")
- Include key crime-related terms (e.g. "conviction", "accused", "sentence")
- Include 2-3 factual keywords from the FIR (e.g. "property dispute", "assault")
- Keep it under 20 words
- Do NOT include generic terms like "penal code" or "law"

Output ONLY the search query, nothing else."""

    try:
        resp = _get_groq().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Output only a search query. No explanations."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=60,
        )
        q = resp.choices[0].message.content.strip().strip('"').strip("'")
        return q
    except Exception as e:
        print(f"[Kanoon] Query generation error: {e}")
        return ""


def _is_actual_case(title: str, docsource: str) -> bool:
    """Return True only if the document looks like an actual court judgment (not a statute/act/code)."""
    title_lower = title.lower()
    source_lower = docsource.lower()

    # Skip bare statute definition pages
    if re.match(r"^Section\s+\d+", title, re.IGNORECASE):
        return False

    # Skip Acts, Codes, Rules, Regulations
    skip_keywords = [
        "penal code", "ranbir penal code", "criminal procedure code",
        "- act", "- rules", "- regulation", "- ordinance", "- bill",
        "constitution of india", "bare act",
    ]
    for kw in skip_keywords:
        if kw in title_lower or kw in source_lower:
            return False

    # Skip sources that are Act repositories
    act_sources = [
        "union of india - section", "state of ", "central government",
    ]
    for src in act_sources:
        if source_lower.startswith(src) and "vs" not in title_lower:
            return False

    # Prefer results that have "vs" (actual cases are "X vs Y")
    # But don't require it — some cases use "v." or "versus"
    # Just reject obvious non-cases
    if docsource.endswith("- Act") or docsource.endswith("- Code"):
        return False

    return True


# ---------------------------------------------------------------------------
#  Main pipeline: search → fetch docs → summarize → predict verdict
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
    3. Summarize each case with Groq LLM
    4. Predict verdict/punishment based on all precedents
    """
    print("\n" + "="*80)
    print("INDIAN KANOON API - EXACT INPUTS")
    print("="*80)
    print(f"\n[INPUT 1] mapped_sections (from Stage 1):")
    for i, sec in enumerate(mapped_sections, 1):
        print(f"  {i}. {sec}")
    print(f"\n[INPUT 2] fir_summary (first 600 chars):")
    print(f"  {fir_summary}")
    print("\n" + "="*80)
    
    if not KANOON_API_KEY:
        return {"status": "error", "sections_searched": [], "cases": [],
                "verdict_prediction": None, "api_calls_used": 0,
                "error": "KANOON_API_KEY is not set in .env"}

    ipc_sections = _extract_ipc_sections(mapped_sections)
    print(f"\n[STEP 1] Extracted IPC sections: {ipc_sections}")
    if not ipc_sections and not fir_summary:
        return {"status": "no_results", "sections_searched": [], "cases": [],
                "verdict_prediction": None, "api_calls_used": 0, "error": None}

    all_cases = []
    seen_tids = set()
    api_calls = 0

    def _add_cases_from_result(result, section_label, limit):
        """Extract valid cases from a Kanoon search result — strict filtering."""
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
                "title": title,
                "tid": tid,
                "court": docsource,
                "snippet": _clean_html(doc.get("headline", ""))[:500],
                "section": section_label,
                "date": _extract_date_from_title(title),
                "url": f"https://indiankanoon.org/doc/{tid}/" if tid else "",
                "summary": "",
            })
            added += 1
        return added

    # ---- Strategy 1: LLM-crafted fact+section query (best relevance) ----
    fact_query = _build_fact_query(fir_summary, ipc_sections) if fir_summary else ""
    if fact_query:
        print(f"\n[KANOON SEARCH QUERY 1 - LLM-generated]")
        print(f"  Query: \"{fact_query}\"")
        result = search_kanoon(fact_query)
        api_calls += 1
        _add_cases_from_result(result, "Fact-match", min(3, max_total))

    # ---- Strategy 2: Section + "conviction accused" (targeted per section) ----
    for i, sec in enumerate(ipc_sections, 1):
        if len(all_cases) >= max_total:
            break
        query = f"Section {sec} IPC accused conviction sentence"
        print(f"\n[KANOON SEARCH QUERY {i+1} - Fallback for IPC {sec}]")
        print(f"  Query: \"{query}\"")
        result = search_kanoon(query)
        api_calls += 1
        _add_cases_from_result(result, f"IPC {sec}", max_per_section)

    # ---- Strategy 3: Broader fallback if we still have too few ----
    if len(all_cases) < 2:
        print(f"\n[KANOON SEARCH - Strategy 3: Broader fallback]")
        for sec in ipc_sections:
            if len(all_cases) >= max_total:
                break
            query = f"Section {sec} IPC guilty punishment"
            print(f"  Query: \"{query}\"")
            result = search_kanoon(query)
            api_calls += 1
            _add_cases_from_result(result, f"IPC {sec}", max_per_section)

    if not all_cases:
        return {"status": "no_results", "sections_searched": [f"IPC {s}" for s in ipc_sections],
                "cases": [], "verdict_prediction": None, "api_calls_used": api_calls, "error": None}

    # ---- Step 2: Fetch judgment text + summarize ----
    print(f"[Kanoon] Fetching & summarizing {len(all_cases)} cases...")

    for case in all_cases:
        tid = case["tid"]
        if not tid:
            continue
        doc_data = get_doc_detail(tid)
        api_calls += 1
        if doc_data.get("error"):
            print(f"[Kanoon] Doc fetch error for {tid}: {doc_data['error']}")
            continue

        doc_text = _clean_html(doc_data.get("doc", "")) or case["snippet"]
        case["summary"] = summarize_case(case["title"], doc_text, case["section"])

    # ---- Step 3: Predict verdict ----
    print("[Kanoon] Predicting verdict based on precedents...")
    cases_with_summaries = [c for c in all_cases if c.get("summary")]

    verdict = predict_verdict(
        fir_summary=fir_summary,
        ipc_sections=[f"IPC {s}" for s in ipc_sections],
        case_summaries=cases_with_summaries or all_cases,
    )

    # ---- Step 4: Rank section influence on verdict ----
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
        print(f"     URL: {c['url']}")
        if c["summary"]:
            print(f"     Summary: {c['summary'][:250]}...")

    vp = result.get("verdict_prediction", {})
    if vp:
        print(f"\n{'=' * 60}")
        print("VERDICT PREDICTION:")
        print(f"  Verdict:    {vp.get('predicted_verdict', 'N/A')}")
        print(f"  Punishment: {vp.get('predicted_punishment', 'N/A')}")
        print(f"  Range:      {vp.get('punishment_range', 'N/A')}")
        print(f"  Bail:       {vp.get('bail_likelihood', 'N/A')}")
        print(f"  Confidence: {vp.get('confidence', 'N/A')}")
        print(f"  Reasoning:  {vp.get('reasoning', 'N/A')}")
