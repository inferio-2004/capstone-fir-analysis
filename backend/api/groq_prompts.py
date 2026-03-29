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

    truncated = case_text[:3500]

    prompt = f"""Summarize this Indian court judgment concisely in 3-5 sentences.

Focus on: what the case was about, the charges, the court's decision, and the punishment/sentence given.



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

    """Rank each applicable section by how much it influences the verdict."""

    if not mapped_sections:

        return []



    sections_str = ", ".join(mapped_sections)

    cases_ctx = ""

    for i, cs in enumerate(case_summaries[:5], 1):

        cases_ctx += (

            f"\n[Case {i}] {cs.get('title','')} (Section {cs.get('section','')})\n"

            f"  Summary: {cs.get('summary','')}\n"

        )



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

        raw = groq_chat_with_fallback(

            _get_groq(),

            role="summarisation",

            messages=[

                {"role": "system", "content": "You are a senior Indian criminal law expert. Return ONLY valid JSON array, no markdown."},

                {"role": "user", "content": prompt},

            ],

            temperature=0.3,

            max_tokens=800,

        )

        raw = re.sub(r"^```(?:json)?\s*", "", raw)

        raw = re.sub(r"\s*```$", "", raw)

        rankings = json.loads(raw)



        for r in rankings:

            r["influence_score"] = max(0, min(100, int(r.get("influence_score", 0))))

        rankings.sort(key=lambda r: r["influence_score"], reverse=True)

        return rankings

    except Exception as e:

        print(f"[Kanoon] Section influence ranking error: {e}")

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





# ---------------------------------------------------------------------------

#  Fact → Kanoon search query

# ---------------------------------------------------------------------------

def build_fact_query(fir_summary: str, ipc_sections: list[str]) -> str:

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

        result = groq_chat_with_fallback(

            _get_groq(),

            role="summarisation",

            messages=[

                {"role": "system", "content": "Output only a search query. No explanations."},

                {"role": "user", "content": prompt},

            ],

            temperature=0.2,

            max_tokens=60,

        )

        return result.strip('"').strip("'")

    except Exception as e:

        print(f"[Kanoon] Query generation error: {e}")

        return ""

