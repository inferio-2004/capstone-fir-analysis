"""
Groq LLM prompt functions for Stage 2 — case summarisation, verdict
prediction, section-influence ranking, and fact-to-query translation.

All functions use the Groq SDK directly (not LangChain) and are
stateless — they receive the data they need and return structured dicts.
"""

import json
import os
import re

from groq import Groq
from dotenv import load_dotenv
from model_config import groq_chat_with_fallback

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

_groq_client = None


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ---------------------------------------------------------------------------
#  Case summarisation
# ---------------------------------------------------------------------------
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
        return groq_chat_with_fallback(
            _get_groq(),
            role="summarisation",
            messages=[
                {"role": "system", "content": "You are a legal analyst who summarizes Indian court judgments concisely. Focus on charges, decisions, and sentences."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
    except Exception as e:
        print(f"[Kanoon] Groq summarization error (all models failed): {e}")
        return ""


def summarize_case_with_llm(case_text: str) -> str:
    """Summarize a court judgment using Groq LLM instead of extractive methods."""
    # Note: Preprocessing (Facts/Held/Judgment extraction) is now handled 
    # at the pipeline level in indian_kanoon.py before calling this.
    truncated = case_text[:2500]
    prompt = f"""Summarize this judgment in MAX 120 words.
Focus ONLY on:
1. Primary charges.
2. Court's final decision.
3. Sentence/Punishment.

TEXT:
{truncated}

SUMMARY:"""

    try:
        # Use direct Llama-3.1-8B for speed and conciseness, as verified in benchmark
        resp = _get_groq().chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Concise legal summarizer. 120 words max. No preamble."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=250
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Kanoon] Groq summarization error: {e}")
        return ""


# ---------------------------------------------------------------------------
#  Verdict prediction
# ---------------------------------------------------------------------------
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
        raw = groq_chat_with_fallback(
            _get_groq(),
            role="summarisation",
            messages=[
                {"role": "system", "content": "You are a senior Indian criminal law expert. Predict verdicts based on real precedents. Return ONLY valid JSON, no markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
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


# ---------------------------------------------------------------------------
#  Section influence ranking
# ---------------------------------------------------------------------------
def rank_section_influence(
    mapped_sections: list[str],
    fir_summary: str,
    verdict: dict,
    case_summaries: list[dict],
) -> list[dict]:
    """
    Deterministic section influence scoring (Sum-to-100 Algorithm).
    Replaced LLM-based ranking with a weighted feature-driven approach.
    """
    if not mapped_sections:
        return []

    # Weights from AGENT_README_BENCHMARK.md Section 4.2
    WEIGHTS = {
        "intent_match": 5.0,
        "punishment": 1.0,  # multiplied by severity score
        "precedent": 3.0,
        "constructive": -2.0,  # penalty for constructive/procedural
    }

    # Internal metadata for scoring (heuristic derivation for Stage 1 sections)
    raw_scores = []
    for section_name in mapped_sections:
        # 1. Feature: Intent Match (Heuristic: does the section number appear in the summary?)
        intent_match = any(word.lower() in fir_summary.lower() for word in section_name.split())

        # 2. Feature: Punishment Severity (Heuristic derivation from common IPC sections)
        severity = 2  # Default moderate
        lower_name = section_name.lower()
        if any(s in lower_name for s in ["302", "307", "376", "395", "396"]):
            severity = 5  # Severe (Life/Death)
        elif any(s in lower_name for s in ["304", "326", "364", "392", "409"]):
            severity = 4  # High (10+ yrs)
        elif any(s in lower_name for s in ["325", "354", "380", "420", "467"]):
            severity = 3  # Moderate (5-10 yrs)

        # 3. Feature: Section Type (Heuristic)
        stype = "primary"
        if any(s in lower_name for s in ["34", "149", "120b"]):
            stype = "constructive"
        elif any(s in lower_name for s in ["procedural", "100", "101"]):
            stype = "procedural"

        # 4. Feature: Precedent Frequency
        precedent_freq = sum(1 for cs in case_summaries if cs.get("section", "") in section_name)

        # Weighted Raw Score calculation
        raw_score = 0
        if intent_match:
            raw_score += WEIGHTS["intent_match"]
        raw_score += WEIGHTS["punishment"] * severity
        if precedent_freq > 0:
            raw_score += WEIGHTS["precedent"]
        if stype in ("constructive", "procedural"):
            raw_score += WEIGHTS["constructive"]

        raw_score = max(raw_score, 0)
        raw_scores.append(raw_score)

    # Independent Scoring (0-100 per section)
    # We map a reasonable maximum raw score to 100 for better distribution.
    # Intent(5) + Punishment(5) = 10 as baseline for 100%.
    MAX_RAW = 10.0
    influence_scores = [min(100, int((s / MAX_RAW) * 100)) for s in raw_scores]

    # Output construction
    results = []
    for i, section_name in enumerate(mapped_sections):
        score = influence_scores[i]

        # Level assignment
        if score >= 60:
            level = "Primary"
        elif score >= 20:
            level = "Supporting"
        else:
            level = "Minor"

        # Reasoning building from extracted features
        reasoning_parts = []
        if any(word.lower() in fir_summary.lower() for word in section_name.split()):
            reasoning_parts.append("Direct intent match.")

        lower_name = section_name.lower()
        if any(s in lower_name for s in ["302", "307", "376", "395", "396"]):
            reasoning_parts.append("Carries severe punishment (life/10+ yrs).")
        elif any(s in lower_name for s in ["304", "326", "364", "392", "409"]):
            reasoning_parts.append("Carries severe punishment (life/10+ yrs).")
        elif any(s in lower_name for s in ["325", "354", "380", "420", "467", "504"]):
            reasoning_parts.append("Carries moderate punishment.")

        if any(s in lower_name for s in ["34", "149", "120b"]):
            reasoning_parts.append("Constructive liability — reduced weight.")
        elif any(s in lower_name for s in ["procedural", "100", "101"]):
            reasoning_parts.append("Procedural section — reduced weight.")

        if sum(1 for cs in case_summaries if cs.get("section", "") in section_name) > 0:
            reasoning_parts.append("Found in similar precedents.")

        reasoning = " ".join(reasoning_parts) if reasoning_parts else "Identified as applicable statute."

        results.append({
            "section": section_name,
            "influence_score": score,
            "influence_level": level,
            "reasoning": reasoning,
        })

    return sorted(results, key=lambda x: x["influence_score"], reverse=True)


# ---------------------------------------------------------------------------
#  Fact → Kanoon search query
# ---------------------------------------------------------------------------
def build_fact_query(fir_summary: str, ipc_sections: list[str]) -> str:
    """
    Deterministic query builder — no LLM needed.
    Anchors on section numbers + crime keywords extracted from FIR.
    """
    if not ipc_sections:
        return "criminal conviction judgment"

    # Use only the top 2 most severe sections (already sorted by Stage 1)
    primary_sections = ipc_sections[:2]
    section_str = " ".join([f"Section {s} IPC" for s in primary_sections])

    CRIME_KEYWORDS = {
        "extortion": ["extort", "forced transfer", "compelled"],
        "murder": ["killed", "death", "murdered"],
        "assault": ["rod", "weapon", "struck", "injured"],
        "robbery": ["snatched", "robbery", "dacoity"],
        "threat": ["threatened", "intimidat"],
        "rape": ["rape", "sexual assault"],
        "fraud": ["cheating", "deceived", "misrepresent"],
        "kidnapping": ["abduct", "kidnap"],
        "theft": ["stole", "theft", "stolen"],
    }

    fir_lower = fir_summary.lower()
    matched_keywords = [
        kw for kw, triggers in CRIME_KEYWORDS.items()
        if any(t in fir_lower for t in triggers)
    ]

    keyword_str = " ".join(matched_keywords[:2])
    suffix = "conviction judgment"
    query = f"{section_str} {keyword_str} {suffix}".strip()
    return query
