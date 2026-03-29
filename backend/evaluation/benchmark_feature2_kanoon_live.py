#!/usr/bin/env python3
"""
benchmark_feature2_kanoon_live.py

Live benchmark comparing LLM vs Extractive summarization on real Indian Kanoon cases.

Usage:
  python backend/evaluation/benchmark_feature2_kanoon_live.py
"""

import os
import re
import html
from collections import Counter

import numpy as np
import requests
from dotenv import load_dotenv

# Add parent directories to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

from groq_prompts import summarize_case_with_llm

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KANOON_BASE_URL = "https://api.indiankanoon.org"
KANOON_API_KEY = os.environ.get("KANOON_API_KEY", "")

SEARCH_QUERIES = [
    "robbery house trespass knife theft",
    "murder circumstantial evidence conviction",
    "assault grievous hurt IPC 325",
]

# Legal keywords for TF-IDF boost
LEGAL_KEYWORDS = {
    'murder', 'culpable', 'homicide', 'robbery', 'theft', 'assault', 'hurt',
    'grievous', 'knife', 'weapon', 'trespass', 'conviction', 'acquittal',
    'sentence', 'punishment', 'imprisonment', 'fine', 'bail', 'custody',
    'witness', 'evidence', 'circumstantial', 'confession', 'dying', 'declaration',
    'investigation', 'accused', 'complaint', 'fir', 'charge', 'ipc', 'section',
    'offence', 'guilty', 'proven', 'beyond', 'reasonable', 'doubt', 'appeal',
    'dismissed', 'allowed', 'upheld', 'reversed', 'remanded', 'trial', 'court',
}

# Header/garbage patterns
HEADER_TRIGGER_WORDS = [
    'JUDGMENT', 'ORDER', 'HELD:', 'FACTS:', 'J U D G M E N T',
    'judgment', 'order', 'held:', 'facts:'
]

GARBAGE_PATTERNS = [
    r'Author:', r'Bench:', r'Reserved on:', r'Equivalent citations:',
    r'VERSUS', r'PETITIONER', r'RESPONDENT', r'CITATION',
]


def _headers() -> dict:
    return {"Authorization": f"Token {KANOON_API_KEY}", "Accept": "application/json"}


def _clean_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


# =============================================================================
# STAGE 1: Strip document header noise
# =============================================================================
def _strip_header(text: str) -> str:
    """Find first occurrence of trigger words and discard everything before."""
    text = _clean_html(text)
    
    # Find the earliest occurrence of any trigger word
    earliest_pos = len(text)
    for trigger in HEADER_TRIGGER_WORDS:
        pos = text.find(trigger)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
    
    if earliest_pos < len(text):
        return text[earliest_pos:]
    
    # Fallback: discard first 20% of sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    skip_count = max(1, len(sentences) // 5)
    return " ".join(sentences[skip_count:])


# =============================================================================
# STAGE 2: Filter garbage sentences
# =============================================================================
def _is_garbage_sentence(sentence: str) -> bool:
    """Check if sentence should be discarded."""
    # More than 40% uppercase characters (headers/citations)
    chars = [c for c in sentence if c.isalpha()]
    if chars:
        upper_ratio = sum(1 for c in chars if c.isupper()) / len(chars)
        if upper_ratio > 0.40:
            return True
    
    # Contains HTML artifacts
    if "&amp;" in sentence or "&#" in sentence:
        return True
    
    # Contains 3+ consecutive ALL-CAPS words
    words = sentence.split()
    caps_count = 0
    for word in words:
        if word.isalpha() and word.isupper() and len(word) > 1:
            caps_count += 1
            if caps_count >= 3:
                return True
        else:
            caps_count = 0
    
    # Fewer than 8 words
    if len(words) < 8:
        return True
    
    # Matches garbage patterns
    for pattern in GARBAGE_PATTERNS:
        if re.search(pattern, sentence, re.IGNORECASE):
            return True
    
    return False


def _filter_sentences(sentences: list[str]) -> list[str]:
    """Filter out garbage sentences."""
    return [s for s in sentences if not _is_garbage_sentence(s)]


# =============================================================================
# STAGE 3: TF-IDF + legal keyword boost
# =============================================================================
def _compute_tfidf_scores(sentences: list[str]) -> np.ndarray:
    """Compute TF-IDF scores with legal keyword boost."""
    if not sentences:
        return np.array([])
    
    n = len(sentences)
    sentence_tokens = [_tokenize_words(s) for s in sentences]
    
    if not any(sentence_tokens):
        return np.zeros(n)
    
    vocab = sorted({w for tokens in sentence_tokens for w in tokens})
    if not vocab:
        return np.zeros(n)
    
    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    
    # TF matrix
    tf = np.zeros((n, V), dtype=np.float64)
    for i, tokens in enumerate(sentence_tokens):
        if not tokens:
            continue
        den = len(tokens)
        for w, cnt in Counter(tokens).items():
            tf[i, w2i[w]] = cnt / den
    
    # IDF vector
    df = np.zeros(V, dtype=np.float64)
    for j, term in enumerate(vocab):
        df[j] = sum(1 for tokens in sentence_tokens if term in tokens)
    idf = np.log(n / np.maximum(df, 1.0))
    
    # Base TF-IDF scores
    scores = (tf * idf).sum(axis=1)
    
    # Legal keyword boost
    for i, tokens in enumerate(sentence_tokens):
        legal_count = sum(1 for t in tokens if t in LEGAL_KEYWORDS)
        if legal_count > 0:
            scores[i] *= (1 + 0.1 * legal_count)  # 10% boost per legal keyword
    
    return scores


# =============================================================================
# STAGE 4: Post-process output
# =============================================================================
def _post_process(sentences: list[str]) -> list[str]:
    """Strip HTML entities and whitespace."""
    cleaned = []
    for s in sentences:
        # Strip HTML entities
        s = html.unescape(s)
        # Strip leading/trailing whitespace
        s = s.strip()
        if s:
            cleaned.append(s)
    return cleaned


# =============================================================================
# MAIN EXTRACTIVE SUMMARIZER
# =============================================================================
def summarize_kanoon_text(text: str) -> str:
    """
    4-stage extractive summarizer for Indian Kanoon judgment text.
    
    Stage 1: Strip document header noise
    Stage 2: Filter garbage sentences  
    Stage 3: Apply TF-IDF + legal keyword boost
    Stage 4: Post-process output
    """
    if not text or not str(text).strip():
        return ""
    
    # Stage 1: Strip header
    text = _strip_header(text)
    
    # Split into sentences
    raw_sentences = re.split(r"(?<=[.!?])\s+", str(text).strip())
    sentences = [p.strip() for p in raw_sentences if p.strip()]
    
    if not sentences:
        return ""
    
    # Stage 2: Filter garbage
    sentences = _filter_sentences(sentences)
    
    if not sentences:
        return ""
    
    # Stage 3: TF-IDF + legal boost scoring
    scores = _compute_tfidf_scores(sentences)
    
    if len(scores) == 0 or scores.sum() == 0:
        # Fallback: take first few sentences
        k = min(3, len(sentences))
        selected = sentences[:k]
    else:
        n = len(sentences)
        k = min(max(3, n // 5), n, 6)  # Cap at 6 sentences
        top_idx = np.argsort(-scores)[:k]
        selected_indices = sorted(int(i) for i in top_idx)
        selected = [sentences[i] for i in selected_indices]
    
    # Stage 4: Post-process
    selected = _post_process(selected)
    
    return " ".join(selected)


# =============================================================================
# Kanoon API Functions
# =============================================================================
def search_kanoon(query: str, page: int = 0) -> dict:
    """Search Indian Kanoon API."""
    if not KANOON_API_KEY:
        return {"error": "KANOON_API_KEY not set", "docs": []}
    
    try:
        resp = requests.post(
            f"{KANOON_BASE_URL}/search/",
            headers=_headers(),
            data={"formInput": query, "pagenum": page},
            timeout=15
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP {e.response.status_code}: {str(e)}", "docs": []}
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "docs": []}


def get_doc_detail(tid: int) -> dict:
    """Fetch full judgment text from Indian Kanoon by TID."""
    if not KANOON_API_KEY:
        return {"error": "KANOON_API_KEY not set"}
    
    try:
        resp = requests.post(
            f"{KANOON_BASE_URL}/doc/{tid}/",
            headers=_headers(),
            timeout=20
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


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


def _extract_date_from_title(title: str) -> str:
    m = re.search(r"on\s+(\d{1,2}\s+\w+,?\s+\d{4})", title)
    if m:
        return m.group(1)
    m = re.search(r"\((\d{4})\)", title)
    return m.group(1) if m else ""


def fetch_cases_for_query(query: str, max_cases: int = 1) -> list[dict]:
    """Fetch real cases from Kanoon for a given query."""
    print(f"\n[SEARCH] Query: \"{query}\"")
    
    result = search_kanoon(query)
    if result.get("error"):
        print(f"  [ERROR] {result['error']}")
        return []
    
    cases = []
    for doc in result.get("docs", []):
        if len(cases) >= max_cases:
            break
            
        tid = doc.get("tid")
        if not tid:
            continue
            
        title = _clean_html(doc.get("title", "Unknown Case"))
        docsource = doc.get("docsource", "Unknown Court")
        
        if not _is_actual_case(title, docsource):
            continue
        
        # Fetch full document
        doc_data = get_doc_detail(tid)
        if doc_data.get("error"):
            continue
            
        doc_text = _clean_html(doc_data.get("doc", ""))
        if not doc_text or len(doc_text) < 200:
            continue
        
        cases.append({
            "title": title,
            "tid": tid,
            "court": docsource,
            "date": _extract_date_from_title(title),
            "url": f"https://indiankanoon.org/doc/{tid}/",
            "text": doc_text,
            "snippet": _clean_html(doc.get("headline", ""))[:500],
        })
    
    print(f"  [FOUND] {len(cases)} valid case(s)")
    return cases


# =============================================================================
# Benchmark Runner
# =============================================================================
def run_benchmark():
    """Run the benchmark comparing LLM vs Extractive summarization."""
    print("=" * 90)
    print("FEATURE 2 KANOON LIVE BENCHMARK")
    print("Comparing: LLM Summarization vs Extractive Summarization")
    print("=" * 90)
    
    if not KANOON_API_KEY:
        print("\n[ERROR] KANOON_API_KEY is not set in .env file")
        return
    
    all_results = []
    
    for query in SEARCH_QUERIES:
        cases = fetch_cases_for_query(query, max_cases=1)
        
        for case in cases:
            print(f"\n{'=' * 90}")
            print(f"CASE: {case['title']}")
            print(f"Court: {case['court']} | Date: {case['date']}")
            print(f"URL: {case['url']}")
            print(f"Text length: {len(case['text'])} chars")
            print("=" * 90)
            
            # Show raw text preview (first 300 chars)
            raw_preview = case['text'][:300].replace('\n', ' ')
            print(f"\n[RAW KANOON TEXT - First 300 chars]\n{raw_preview}...")
            
            # Get LLM summary
            print("\n[PROCESSING] LLM Summary...")
            llm_summary = summarize_case_with_llm(case['text'][:3500])
            
            # Get Extractive summary
            print("[PROCESSING] Extractive Summary...")
            extractive_summary = summarize_kanoon_text(case['text'])
            
            print("\n" + "-" * 90)
            print("LLM SUMMARY:")
            print("-" * 90)
            print(llm_summary if llm_summary else "[No summary generated]")
            
            print("\n" + "-" * 90)
            print("EXTRACTIVE SUMMARY:")
            print("-" * 90)
            print(extractive_summary if extractive_summary else "[No summary generated]")
            
            all_results.append({
                "query": query,
                "case": case,
                "llm_summary": llm_summary,
                "extractive_summary": extractive_summary,
            })
    
    # Final summary
    print("\n" + "=" * 90)
    print("BENCHMARK COMPLETE")
    print("=" * 90)
    print(f"\nTotal cases processed: {len(all_results)}")
    
    llm_success = sum(1 for r in all_results if r['llm_summary'])
    extractive_success = sum(1 for r in all_results if r['extractive_summary'])
    
    print(f"LLM summaries generated: {llm_success}/{len(all_results)}")
    print(f"Extractive summaries generated: {extractive_success}/{len(all_results)}")


if __name__ == "__main__":
    run_benchmark()
