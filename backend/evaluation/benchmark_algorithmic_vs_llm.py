#!/usr/bin/env python3
"""
Benchmark: Algorithmic vs LLM for Case Summarisation (2d) & Section Influence Ranking (2f)
===========================================================================================

Runs both the LLM-based and algorithmic approaches on real Kanoon cases and
the 5 benchmark test cases, then compares them with quantitative metrics.

Metrics for Summarisation (2d):
  - Key Fact Recall       : % of gold facts (verdict, punishment, section, parties)
                            captured by the summary
  - Information Density   : facts extracted per character (efficiency)
  - Time                  : wall-clock seconds

Metrics for Section Influence Ranking (2f):
  - Rank Correlation      : Spearman's ρ between LLM and algorithmic rankings
  - Top-1 Agreement       : Does the #1 most-influential section match?
  - Level Agreement       : % of sections assigned the same influence_level
  - Score MAE             : Mean absolute error of influence_score (0-100)
  - Time                  : wall-clock seconds

Usage:
    cd backend
    python evaluation/benchmark_algorithmic_vs_llm.py 2>&1
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean

# ---------------------------------------------------------------------------
# Path & imports
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
API_DIR = REPO_ROOT / "backend" / "api"
sys.path.insert(0, str(API_DIR))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / "backend" / ".env")
load_dotenv(REPO_ROOT / ".env")

from kanoon_cache import cache_key, load_cache
from groq_prompts import summarize_case, rank_section_influence, predict_verdict
from indian_kanoon import _clean_html, search_kanoon, get_doc_detail, _extract_ipc_sections


# ═══════════════════════════════════════════════════════════════════════════
#  ALGORITHMIC IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
#  Algo 2d: Extractive Case Summarisation
# ---------------------------------------------------------------------------
_VERDICT_SIGNALS = {
    "convicted": 3, "acquitted": 3, "guilty": 3, "not guilty": 3,
    "sentenced": 3, "punishment": 2, "imprisonment": 2,
    "rigorous imprisonment": 2, "fine": 1, "bail": 2,
    "dismissed": 2, "allowed": 2, "upheld": 2, "set aside": 2,
    "death penalty": 3, "life imprisonment": 3, "years": 1,
    "convicted under": 3, "sentence of": 2,
}
_CHARGE_SIGNALS = {
    "section": 2, "ipc": 2, "charged": 2, "offence": 1,
    "accused": 1, "complainant": 1, "prosecution": 1,
    "appellant": 1, "respondent": 1,
}


def extractive_summarize(case_title: str, case_text: str, ipc_section: str,
                         max_sentences: int = 5) -> str:
    """Score sentences by legal relevance, using head+tail of long judgments."""
    # For long judgments, take head (intro/charges) + tail (verdict/sentence)
    # Verdicts are almost always in the final paragraphs of Indian judgments.
    HEAD_CHARS = 2000
    TAIL_CHARS = 2500
    if len(case_text) > HEAD_CHARS + TAIL_CHARS:
        text_window = case_text[:HEAD_CHARS] + " ... " + case_text[-TAIL_CHARS:]
    else:
        text_window = case_text

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text_window)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) <= max_sentences:
        return " ".join(sentences)[:600]

    scored = []
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        score = 0.0

        # Verdict/outcome signals (heavily weighted)
        for kw, w in _VERDICT_SIGNALS.items():
            if kw in sent_lower:
                score += w
        # Charge/section signals
        for kw, w in _CHARGE_SIGNALS.items():
            if kw in sent_lower:
                score += w
        # Boost if it mentions the specific IPC section
        sec_num = re.search(r"\d+[A-Za-z]*", ipc_section)
        if sec_num and sec_num.group() in sent:
            score += 4
        # Positional bias: first 2 sentences (case intro) + last 10 (verdict zone)
        if i < 2:
            score += 2
        if i >= len(sentences) - 10:
            score += 2.5
        # Extra boost for sentences that look like dispositive clauses
        if any(p in sent_lower for p in [
            "appeal is", "conviction is", "sentence is", "order is",
            "we hold", "we direct", "accordingly", "in the result",
            "in view of", "for the foregoing", "hereby",
        ]):
            score += 3

        scored.append((score, i, sent))

    # Take top-scored sentences, preserve original order
    scored.sort(key=lambda x: x[0], reverse=True)
    top = sorted(scored[:max_sentences], key=lambda x: x[1])
    return " ".join(s[2] for s in top)[:700]


# ---------------------------------------------------------------------------
#  Algo 2f: Algorithmic Section Influence Ranking (punishment severity)
# ---------------------------------------------------------------------------
def _normalize_pdf_text(text: str) -> str:
    """Collapse newlines and multiple spaces for easier regex matching."""
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text


def _extract_max_punishment(section_text: str) -> tuple[int, str]:
    """Parse the maximum punishment from statute text.
    
    Strategy: search for punishment indicators directly, using
    context-aware filtering to exclude false positives from
    descriptive text (e.g., "offence punishable with death").
    Handles PDF-extracted text with broken words.
    """
    # Normalize: collapse newlines/extra spaces but preserve word boundaries
    text = _normalize_pdf_text(section_text)
    text_lower = text.lower()

    # --- Life imprisonment (actual punishment, not "offence punishable with...") ---
    # Look for "imprisonment for life" and determine if it's an actual
    # punishment or a descriptor of another offence.
    if "imprisonment for life" in text_lower:
        # Check if it's near "offence...with" (descriptor) or "punished with" (actual)
        # False positive pattern: "offence [puni shable/punishable] with [death or] imprisonment for life"
        is_descriptor = bool(re.search(
            r"offence.{0,30}with.{0,40}imprisonment\s+for\s+life",
            text_lower, re.DOTALL
        ))
        is_actual = bool(re.search(
            r"shall\s+be\s+punish.{0,20}with.{0,20}imprisonment\s+for\s+life",
            text_lower, re.DOTALL
        ))
        if is_actual and not is_descriptor:
            return 95, "Life imprisonment"
        if is_actual and is_descriptor:
            pass  # ambiguous, fall through to years check

    # --- Extract max years (flexible for broken words like "se ven", "te n") ---
    word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12, "fourteen": 14, "twenty": 20}
    # Also handle common PDF-broken number words
    broken_map = {"se ven": 7, "thr ee": 3, "te n": 10, "fo ur": 4,
                  "fi ve": 5, "eig ht": 8, "ni ne": 9, "tw o": 2,
                  "elev en": 11, "twel ve": 12, "twen ty": 20}

    max_years = 0

    # Check for broken number words first
    for broken, val in broken_map.items():
        if broken in text_lower and "extend" in text_lower:
            max_years = max(max_years, val)

    # Pattern: "extend to X years" with normal words
    for m in re.finditer(r"extend\s+to\s+(\w+)\s+year", text_lower):
        val = m.group(1)
        try:
            max_years = max(max_years, int(val))
        except ValueError:
            max_years = max(max_years, word_map.get(val, 0))

    # Also try with an extra word between (broken "to" etc.)
    for m in re.finditer(r"extend\s+\w+\s+(\w+)\s+year", text_lower):
        val = m.group(1)
        try:
            max_years = max(max_years, int(val))
        except ValueError:
            max_years = max(max_years, word_map.get(val, 0))

    # Also look for "not less than X years"
    for m in re.finditer(r"not\s+(?:be\s+)?less\s+than\s+(\w+)\s+year", text_lower):
        val = m.group(1)
        try:
            max_years = max(max_years, int(val))
        except ValueError:
            max_years = max(max_years, word_map.get(val, 0))

    if max_years > 0:
        if max_years >= 10:
            return 80 + min(max_years, 20), f"Up to {max_years} years RI"
        elif max_years >= 5:
            return 55 + max_years, f"Up to {max_years} years imprisonment"
        else:
            return 25 + max_years * 5, f"Up to {max_years} years imprisonment"

    if "fine" in text_lower and "imprisonment" not in text_lower:
        return 10, "Fine only"
    if re.search(r"(?:punish|shall).*imprisonment", text_lower, re.DOTALL):
        return 30, "Imprisonment (duration not parsed)"

    return 20, "Punishment not parsed"


def rank_section_influence_algorithmic(
    mapped_sections: list[str],
    statute_chunks: dict,
) -> list[dict]:
    """Rank sections by max punishment severity from statute text.
    
    Constructive-liability sections (34, 109, 120B etc.) are detected and
    de-ranked since they have no standalone punishment.
    """
    # Sections that amplify others but have no independent punishment
    CONSTRUCTIVE_SECTIONS = {"34", "109", "120B", "120b", "149", "114"}

    rankings = []
    for section_str in mapped_sections:
        m = re.match(r"(?:IPC|BNS)\s+(?:Section\s+)?(.+)", section_str.strip(), re.IGNORECASE)
        if m:
            law = "IPC" if section_str.strip().upper().startswith("IPC") else "BNS"
            section_id = m.group(1).strip()
        else:
            parts = section_str.split()
            law = parts[0] if parts else "IPC"
            section_id = parts[-1] if len(parts) >= 2 else ""

        chunk_key = f"{law.lower()}_{section_id}"
        section_text = ""
        full_text = ""
        if chunk_key in statute_chunks:
            section_text = statute_chunks[chunk_key].get("section_text", "")
            full_text = statute_chunks[chunk_key].get("full_text", "")
        else:
            for c in statute_chunks.values():
                if c.get("section_id") == section_id and c.get("law") == law:
                    section_text = c.get("section_text", "")
                    full_text = c.get("full_text", "")
                    break

        # Constructive liability → always Minor
        if section_id in CONSTRUCTIVE_SECTIONS:
            score = 15
            punishment_desc = "Constructive liability (no standalone punishment)"
        else:
            # Try both section_text and full_text; take the higher score
            score1, desc1 = _extract_max_punishment(section_text)
            score2, desc2 = _extract_max_punishment(full_text) if full_text != section_text else (score1, desc1)
            score, punishment_desc = (score2, desc2) if score2 > score1 else (score1, desc1)

        level = "Primary" if score >= 75 else "Supporting" if score >= 40 else "Minor"
        rankings.append({
            "section": section_str,
            "influence_score": score,
            "influence_level": level,
            "reasoning": f"Max punishment: {punishment_desc}. "
                         f"{'Gravest charge — drives sentencing.' if score >= 75 else 'Supporting charge — adds to overall severity.' if score >= 40 else 'Minor charge — limited sentencing impact.'}",
        })
    rankings.sort(key=lambda r: r["influence_score"], reverse=True)
    return rankings


# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
#  Summary evaluation: Key Fact Recall
# ---------------------------------------------------------------------------
# Legal fact keywords to look for in summaries
_FACT_CATEGORIES = {
    "verdict": ["convicted", "acquitted", "guilty", "not guilty", "dismissed",
                "allowed", "upheld", "set aside", "discharged"],
    "punishment": ["imprisonment", "rigorous imprisonment", "fine", "death",
                   "sentence", "years", "months", "life imprisonment",
                   "punished", "sentenced"],
    "charges": ["section", "ipc", "under section", "charged", "offence",
                "bns", "penal code"],
    "parties": ["accused", "appellant", "respondent", "complainant",
                "prosecution", "petitioner", "state"],
}


def compute_fact_recall(summary: str, categories: dict = None) -> dict:
    """Check how many legal fact categories are captured in the summary."""
    if categories is None:
        categories = _FACT_CATEGORIES
    summary_lower = summary.lower()
    results = {}
    hits = 0
    total = len(categories)
    for cat, keywords in categories.items():
        found = any(kw in summary_lower for kw in keywords)
        results[cat] = found
        if found:
            hits += 1
    results["recall_score"] = hits / total if total > 0 else 0
    return results


def compute_info_density(summary: str) -> float:
    """Facts per character — higher = more efficient."""
    if not summary:
        return 0.0
    recall = compute_fact_recall(summary)
    return recall["recall_score"] / len(summary) * 1000  # per 1000 chars


# ---------------------------------------------------------------------------
#  Ranking evaluation: correlation & agreement
# ---------------------------------------------------------------------------
def _spearman_rho(ranks_a: list[int], ranks_b: list[int]) -> float:
    """Compute Spearman rank correlation without scipy."""
    n = len(ranks_a)
    if n < 2:
        return 1.0
    d_sq = sum((a - b) ** 2 for a, b in zip(ranks_a, ranks_b))
    return 1 - (6 * d_sq) / (n * (n ** 2 - 1))


def compare_rankings(llm_ranks: list[dict], algo_ranks: list[dict]) -> dict:
    """Compare LLM and algorithmic section rankings."""
    # Build section→rank maps (rank by position in sorted list, 1-indexed)
    llm_map = {r["section"]: i for i, r in enumerate(llm_ranks, 1)}
    algo_map = {r["section"]: i for i, r in enumerate(algo_ranks, 1)}

    common_sections = [s for s in llm_map if s in algo_map]
    if not common_sections:
        return {"spearman_rho": 0, "top1_agree": False, "level_agreement": 0,
                "score_mae": 100, "common_sections": 0}

    # Rank correlation
    llm_order = [llm_map[s] for s in common_sections]
    algo_order = [algo_map[s] for s in common_sections]
    rho = _spearman_rho(llm_order, algo_order)

    # Top-1 agreement
    llm_top1 = llm_ranks[0]["section"] if llm_ranks else ""
    algo_top1 = algo_ranks[0]["section"] if algo_ranks else ""
    top1_agree = llm_top1 == algo_top1

    # Level agreement
    llm_levels = {r["section"]: r.get("influence_level", "") for r in llm_ranks}
    algo_levels = {r["section"]: r.get("influence_level", "") for r in algo_ranks}
    level_matches = sum(1 for s in common_sections if llm_levels.get(s) == algo_levels.get(s))
    level_agreement = level_matches / len(common_sections)

    # Score MAE
    llm_scores = {r["section"]: r.get("influence_score", 0) for r in llm_ranks}
    algo_scores = {r["section"]: r.get("influence_score", 0) for r in algo_ranks}
    mae = mean(abs(llm_scores.get(s, 0) - algo_scores.get(s, 0)) for s in common_sections)

    return {
        "spearman_rho": round(rho, 4),
        "top1_agree": top1_agree,
        "level_agreement": round(level_agreement, 4),
        "score_mae": round(mae, 2),
        "common_sections": len(common_sections),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_statute_chunks() -> dict:
    """Load statute_chunks_complete.jsonl into a dict keyed by chunk_id."""
    path = REPO_ROOT / "output" / "statute_chunks_complete.jsonl"
    chunks = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks[obj["chunk_id"]] = obj
    return chunks


def load_cached_kanoon_docs() -> list[dict]:
    """Scan kanoon_cache for doc-type entries (full judgment text)."""
    cache_dir = REPO_ROOT / "output" / "kanoon_cache"
    docs = []
    for p in sorted(cache_dir.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "doc" in data and "title" in data:
                title = _clean_html(data.get("title", "Unknown"))
                doc_text = _clean_html(data.get("doc", ""))
                if len(doc_text) > 200:
                    docs.append({"title": title, "doc_text": doc_text, "file": p.name})
        except Exception:
            pass
    return docs


def load_benchmark_cases() -> list[dict]:
    """Load the 5 benchmark test cases."""
    path = REPO_ROOT / "output" / "benchmark_test_cases.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
#  BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_summarization_benchmark(cached_docs: list[dict], max_cases: int = 5):
    """Compare LLM vs extractive summarisation on real cached Kanoon cases."""
    print("\n" + "=" * 70)
    print("  BENCHMARK 2d: Case Summarisation — LLM vs Extractive")
    print("=" * 70)

    results = []
    docs_to_test = cached_docs[:max_cases]
    print(f"  Testing on {len(docs_to_test)} cached Kanoon judgments\n")

    for i, doc in enumerate(docs_to_test, 1):
        title = doc["title"][:80]
        doc_text = doc["doc_text"]
        ipc_section = "IPC"  # generic since we don't know the exact section from cache
        # Try to extract a section mention from the text
        sec_match = re.search(r"Section\s+(\d+[A-Za-z]*)\s+(?:of\s+)?(?:the\s+)?(?:Indian\s+)?(?:Penal\s+Code|IPC)", doc_text[:2000], re.IGNORECASE)
        if sec_match:
            ipc_section = f"IPC {sec_match.group(1)}"

        print(f"  Case {i}: {title}")
        print(f"    Section: {ipc_section} | Text length: {len(doc_text)} chars")

        # --- LLM summary ---
        t0 = time.perf_counter()
        llm_summary = summarize_case(title, doc_text, ipc_section)
        llm_time = time.perf_counter() - t0

        # --- Extractive summary ---
        t0 = time.perf_counter()
        algo_summary = extractive_summarize(title, doc_text, ipc_section)
        algo_time = time.perf_counter() - t0

        # --- Evaluate both ---
        llm_recall = compute_fact_recall(llm_summary)
        algo_recall = compute_fact_recall(algo_summary)
        llm_density = compute_info_density(llm_summary)
        algo_density = compute_info_density(algo_summary)

        result = {
            "case": title,
            "ipc_section": ipc_section,
            "llm": {
                "summary": llm_summary,
                "length": len(llm_summary),
                "time_s": round(llm_time, 3),
                "fact_recall": llm_recall,
                "info_density": round(llm_density, 4),
            },
            "algo": {
                "summary": algo_summary,
                "length": len(algo_summary),
                "time_s": round(algo_time, 3),
                "fact_recall": algo_recall,
                "info_density": round(algo_density, 4),
            },
        }
        results.append(result)

        print(f"    LLM:  recall={llm_recall['recall_score']:.0%}  density={llm_density:.4f}  time={llm_time:.3f}s  len={len(llm_summary)}")
        print(f"    ALGO: recall={algo_recall['recall_score']:.0%}  density={algo_density:.4f}  time={algo_time:.3f}s  len={len(algo_summary)}")
        print()

    # --- Aggregate ---
    n = len(results)
    if n == 0:
        return {"error": "No cases tested"}

    agg = {
        "num_cases": n,
        "llm_avg_recall": round(mean(r["llm"]["fact_recall"]["recall_score"] for r in results), 4),
        "algo_avg_recall": round(mean(r["algo"]["fact_recall"]["recall_score"] for r in results), 4),
        "llm_avg_density": round(mean(r["llm"]["info_density"] for r in results), 4),
        "algo_avg_density": round(mean(r["algo"]["info_density"] for r in results), 4),
        "llm_avg_time_s": round(mean(r["llm"]["time_s"] for r in results), 3),
        "algo_avg_time_s": round(mean(r["algo"]["time_s"] for r in results), 3),
        "llm_avg_length": round(mean(r["llm"]["length"] for r in results)),
        "algo_avg_length": round(mean(r["algo"]["length"] for r in results)),
    }
    agg["speedup_x"] = round(agg["llm_avg_time_s"] / max(agg["algo_avg_time_s"], 0.0001), 1)
    agg["recall_diff"] = round(agg["algo_avg_recall"] - agg["llm_avg_recall"], 4)

    print("─" * 70)
    print("  SUMMARISATION AGGREGATE")
    print("─" * 70)
    print(f"  {'':30s}  {'LLM':>10s}  {'ALGO':>10s}  {'Diff':>10s}")
    print(f"  {'Avg Fact Recall':30s}  {agg['llm_avg_recall']:>9.0%}  {agg['algo_avg_recall']:>9.0%}  {agg['recall_diff']:>+10.0%}")
    print(f"  {'Avg Info Density':30s}  {agg['llm_avg_density']:>10.4f}  {agg['algo_avg_density']:>10.4f}")
    print(f"  {'Avg Time (s)':30s}  {agg['llm_avg_time_s']:>10.3f}  {agg['algo_avg_time_s']:>10.3f}  {agg['speedup_x']}x faster")
    print(f"  {'Avg Length (chars)':30s}  {agg['llm_avg_length']:>10d}  {agg['algo_avg_length']:>10d}")
    print()

    return {"aggregate": agg, "per_case": results}


def run_ranking_benchmark(benchmark_cases: list[dict], statute_chunks: dict):
    """Compare LLM vs algorithmic section influence ranking on benchmark test cases."""
    print("\n" + "=" * 70)
    print("  BENCHMARK 2f: Section Influence Ranking — LLM vs Algorithmic")
    print("=" * 70)

    results = []
    print(f"  Testing on {len(benchmark_cases)} benchmark FIR cases\n")

    for case in benchmark_cases:
        case_id = case["case_id"]
        # Extract sections from gold_reasoning_text
        gold_text = case.get("gold_reasoning_text", "")
        section_matches = re.findall(r"IPC\s+(?:Section\s+)?(\d+[A-Za-z]*)", gold_text)
        mapped_sections = [f"IPC {s}" for s in dict.fromkeys(section_matches)]

        if not mapped_sections:
            print(f"  {case_id}: No sections found in gold data, skipping")
            continue

        fir_summary = case.get("incident", "")[:600]
        print(f"  {case_id}: sections={mapped_sections}")

        # We need a verdict + case_summaries for the LLM call
        # Build minimal case summaries from statute context
        mock_cases = []
        for sec in mapped_sections:
            mock_cases.append({
                "title": f"Precedent for {sec}",
                "section": sec,
                "court": "High Court",
                "summary": "",
                "snippet": case.get("statute_context", "")[:300],
            })

        # First get a verdict (needed by LLM rank_section_influence)
        ipc_sections = [s.replace("IPC ", "") for s in mapped_sections]
        mock_verdict = {
            "predicted_verdict": "Likely Guilty",
            "predicted_punishment": "Imprisonment",
            "confidence": 0.8,
        }

        # --- LLM ranking ---
        t0 = time.perf_counter()
        llm_ranks = rank_section_influence(
            mapped_sections=mapped_sections,
            fir_summary=fir_summary,
            verdict=mock_verdict,
            case_summaries=mock_cases,
        )
        llm_time = time.perf_counter() - t0

        # --- Algorithmic ranking ---
        t0 = time.perf_counter()
        algo_ranks = rank_section_influence_algorithmic(mapped_sections, statute_chunks)
        algo_time = time.perf_counter() - t0

        # --- Compare ---
        comparison = compare_rankings(llm_ranks, algo_ranks)

        result = {
            "case_id": case_id,
            "sections": mapped_sections,
            "llm_ranking": llm_ranks,
            "algo_ranking": algo_ranks,
            "llm_time_s": round(llm_time, 3),
            "algo_time_s": round(algo_time, 3),
            "comparison": comparison,
        }
        results.append(result)

        print(f"    LLM  top: {llm_ranks[0]['section'] if llm_ranks else 'N/A'} (score={llm_ranks[0]['influence_score'] if llm_ranks else 0})  time={llm_time:.3f}s")
        print(f"    ALGO top: {algo_ranks[0]['section'] if algo_ranks else 'N/A'} (score={algo_ranks[0]['influence_score'] if algo_ranks else 0})  time={algo_time:.3f}s")
        print(f"    Spearman ρ={comparison['spearman_rho']:.3f}  Top1={comparison['top1_agree']}  Level={comparison['level_agreement']:.0%}  MAE={comparison['score_mae']:.1f}")
        print()

    # --- Aggregate ---
    n = len(results)
    if n == 0:
        return {"error": "No cases tested"}

    agg = {
        "num_cases": n,
        "avg_spearman_rho": round(mean(r["comparison"]["spearman_rho"] for r in results), 4),
        "top1_agreement_rate": round(mean(1 if r["comparison"]["top1_agree"] else 0 for r in results), 4),
        "avg_level_agreement": round(mean(r["comparison"]["level_agreement"] for r in results), 4),
        "avg_score_mae": round(mean(r["comparison"]["score_mae"] for r in results), 2),
        "llm_avg_time_s": round(mean(r["llm_time_s"] for r in results), 3),
        "algo_avg_time_s": round(mean(r["algo_time_s"] for r in results), 3),
    }
    agg["speedup_x"] = round(agg["llm_avg_time_s"] / max(agg["algo_avg_time_s"], 0.0001), 1)

    print("─" * 70)
    print("  RANKING AGGREGATE")
    print("─" * 70)
    print(f"  Avg Spearman ρ:           {agg['avg_spearman_rho']:>8.4f}   (1.0 = perfect rank agreement)")
    print(f"  Top-1 Agreement:          {agg['top1_agreement_rate']:>8.0%}   (same #1 most-influential section)")
    print(f"  Avg Level Agreement:      {agg['avg_level_agreement']:>8.0%}   (Primary/Supporting/Minor match)")
    print(f"  Avg Score MAE:            {agg['avg_score_mae']:>8.1f}   (0 = identical scores, out of 100)")
    print(f"  LLM avg time:             {agg['llm_avg_time_s']:>8.3f}s")
    print(f"  ALGO avg time:            {agg['algo_avg_time_s']:>8.3f}s")
    print(f"  Speedup:                  {agg['speedup_x']}x")
    print()

    return {"aggregate": agg, "per_case": results}


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  ALGORITHMIC vs LLM — Quality & Speed Benchmark")
    print("=" * 70)

    # Load data
    print("\n[*] Loading statute chunks...", end=" ", flush=True)
    statute_chunks = load_statute_chunks()
    print(f"{len(statute_chunks)} chunks")

    print("[*] Loading cached Kanoon docs...", end=" ", flush=True)
    cached_docs = load_cached_kanoon_docs()
    print(f"{len(cached_docs)} judgment docs")

    print("[*] Loading benchmark test cases...", end=" ", flush=True)
    benchmark_cases = load_benchmark_cases()
    print(f"{len(benchmark_cases)} cases")

    # ------------------------------------------------------------------
    # Benchmark 2d: Summarisation
    # ------------------------------------------------------------------
    summ_results = run_summarization_benchmark(cached_docs, max_cases=5)

    # ------------------------------------------------------------------
    # Benchmark 2f: Section Influence Ranking
    # ------------------------------------------------------------------
    rank_results = run_ranking_benchmark(benchmark_cases, statute_chunks)

    # ------------------------------------------------------------------
    # Combined report
    # ------------------------------------------------------------------
    report = {
        "benchmark": "Algorithmic vs LLM — Quality & Speed",
        "summarisation_2d": {
            "aggregate": summ_results.get("aggregate", {}),
            "per_case": [
                {
                    "case": r["case"],
                    "llm_recall": r["llm"]["fact_recall"]["recall_score"],
                    "algo_recall": r["algo"]["fact_recall"]["recall_score"],
                    "llm_density": r["llm"]["info_density"],
                    "algo_density": r["algo"]["info_density"],
                    "llm_time_s": r["llm"]["time_s"],
                    "algo_time_s": r["algo"]["time_s"],
                    "llm_summary_preview": r["llm"]["summary"][:200],
                    "algo_summary_preview": r["algo"]["summary"][:200],
                }
                for r in summ_results.get("per_case", [])
            ],
        },
        "ranking_2f": {
            "aggregate": rank_results.get("aggregate", {}),
            "per_case": [
                {
                    "case_id": r["case_id"],
                    "sections": r["sections"],
                    "llm_ranking": [
                        {"section": rk["section"], "score": rk["influence_score"],
                         "level": rk["influence_level"]}
                        for rk in r["llm_ranking"]
                    ],
                    "algo_ranking": [
                        {"section": rk["section"], "score": rk["influence_score"],
                         "level": rk["influence_level"]}
                        for rk in r["algo_ranking"]
                    ],
                    "spearman_rho": r["comparison"]["spearman_rho"],
                    "top1_agree": r["comparison"]["top1_agree"],
                    "level_agreement": r["comparison"]["level_agreement"],
                    "score_mae": r["comparison"]["score_mae"],
                    "llm_time_s": r["llm_time_s"],
                    "algo_time_s": r["algo_time_s"],
                }
                for r in rank_results.get("per_case", [])
            ],
        },
        "verdict": {
            "summarisation_viable": summ_results.get("aggregate", {}).get("recall_diff", -1) >= -0.15,
            "ranking_viable": rank_results.get("aggregate", {}).get("avg_spearman_rho", 0) >= 0.6,
        },
    }

    # Save
    out_path = REPO_ROOT / "output" / "algorithmic_vs_llm_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("  FINAL VERDICT")
    print("=" * 70)
    sa = summ_results.get("aggregate", {})
    ra = rank_results.get("aggregate", {})
    print(f"  Summarisation: recall diff = {sa.get('recall_diff', 0):+.0%} | speedup = {sa.get('speedup_x', 0)}x")
    print(f"  Ranking:       Spearman ρ  = {ra.get('avg_spearman_rho', 0):.3f} | top-1 agree = {ra.get('top1_agreement_rate', 0):.0%} | speedup = {ra.get('speedup_x', 0)}x")
    print()
    v = report["verdict"]
    print(f"  2d Summarisation viable?  {'YES' if v['summarisation_viable'] else 'NO'}  (recall within 15% of LLM)")
    print(f"  2f Ranking viable?        {'YES' if v['ranking_viable'] else 'NO'}  (Spearman ρ ≥ 0.6)")
    print()
    print(f"  Report saved → {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
