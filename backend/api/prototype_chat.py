#!/usr/bin/env python3
"""
LexIR Prototype — 3-Stage Legal Intelligence Chat
Stage 1: FIR Analysis & Section Mapping  (uses existing analysis result)
Stage 2: Case Similarity Search          (auto, no LLM)
Stage 3: Precedent Q&A                   (interactive, LLM-powered)

Usage:
    python prototype_chat.py
    python prototype_chat.py --fir path/to/fir.json
"""

import json
import sys
import textwrap
from pathlib import Path

# -- Resolve paths --
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend" / "api"))

from legalqa_index import LegalQAIndex
from case_similarity import display_similar_cases
from precedent_qa import PrecedentQA, display_qa_answer

# ======================================================================
#  DISPLAY HELPERS
# ======================================================================
W = 70  # display width


def banner():
    b = "=" * W
    print(f"\n  {b}")
    print(f"  {'LexIR — Legal Intelligence & Retrieval':^{W}}")
    print(f"  {'Automated FIR Analysis with Legal Code Mapping':^{W}}")
    print(f"  {b}")
    print(f"  {'Prototype Demo — 3-Stage Chat':^{W}}")
    print(f"  {b}\n")


def display_stage1(fir: dict, analysis: dict):
    """Display Stage 1: FIR summary + section mapping."""
    b = "=" * W
    d = "-" * W
    print(f"\n  {b}")
    print(f"  STAGE 1: FIR ANALYSIS & LEGAL SECTION MAPPING")
    print(f"  {b}")

    # FIR summary
    print(f"\n    FIR ID:       {fir.get('fir_id', 'N/A')}")
    print(f"    Date:         {fir.get('date', 'N/A')}")
    print(f"    Complainant:  {fir.get('complainant_name', 'N/A')}")
    accused = fir.get('accused_names', [])
    print(f"    Accused:      {', '.join(accused) if accused else 'N/A'}")
    print(f"    Victim:       {fir.get('victim_name', 'N/A')}")
    print(f"    Location:     {fir.get('location', 'N/A')}")

    # Incident (wrapped)
    incident = fir.get('incident_description', 'N/A')
    wrapped = textwrap.wrap(incident, width=W - 18)
    print(f"    Incident:     {wrapped[0]}")
    for line in wrapped[1:]:
        print(f"                  {line}")

    # Analysis
    intent = analysis.get("analysis", {}).get("intent_identification", {})
    reasoning = analysis.get("analysis", {}).get("legal_reasoning", {})

    print(f"\n  {d}")
    print(f"    Primary Intent: {intent.get('primary_intent', 'N/A')} "
          f"(confidence: {intent.get('confidence', 'N/A')})")
    secondary = intent.get("secondary_intents", [])
    if secondary:
        print(f"    Secondary:      {', '.join(secondary)}")
    print(f"    Severity:       {reasoning.get('severity_assessment', 'N/A').upper()}")

    # Applicable statutes
    print(f"\n  {d}")
    print(f"    Applicable Sections:")
    print(f"  {d}")

    statutes = analysis.get("applicable_statutes", [])
    for i, s in enumerate(statutes, 1):
        primary = s.get("primary", {})
        law = primary.get("law", "?")
        sec = primary.get("section", "?")
        title = primary.get("title", "N/A")
        reasoning_text = primary.get("reasoning", "N/A")

        print(f"\n    {i}. {law} {sec} — {title}")

        # Corresponding sections
        corr = s.get("corresponding_sections", [])
        for c in corr:
            print(f"       → {c['law']} {c['section']}")

        reason_lines = textwrap.wrap(reasoning_text, width=W - 12)
        print(f"       Reasoning: {reason_lines[0]}")
        for rl in reason_lines[1:]:
            print(f"                  {rl}")

    # Legal basis
    legal_basis = reasoning.get("legal_basis", "")
    if legal_basis:
        print(f"\n  {d}")
        basis_lines = textwrap.wrap(legal_basis, width=W - 6)
        print(f"    Legal Basis:")
        for bl in basis_lines:
            print(f"      {bl}")

    print(f"\n    Chunks Retrieved: {analysis.get('retrieved_data', {}).get('total_chunks_retrieved', '?')}")
    print(f"    After Filtering:  {analysis.get('retrieved_data', {}).get('chunks_after_filtering', '?')}")
    print(f"    Confidence:       {analysis.get('confidence', '?')}")
    print(f"\n  {b}")


# ======================================================================
#  MAIN
# ======================================================================
def main():
    banner()

    # -- Load FIR --
    fir_path = REPO_ROOT / "src_dataset_files" / "fir_sample.json"
    if len(sys.argv) > 2 and sys.argv[1] == "--fir":
        fir_path = Path(sys.argv[2])

    print(f"  [INPUT] Loading FIR from: {fir_path.name}")
    with open(fir_path, "r", encoding="utf-8") as f:
        fir = json.load(f)

    # -- Load existing Stage 1 analysis --
    analysis_path = REPO_ROOT / "output" / "fir_analysis_result_chains.json"
    print(f"  [INPUT] Loading Stage 1 analysis from: {analysis_path.name}")
    with open(analysis_path, "r", encoding="utf-8") as f:
        analysis = json.load(f)

    # -- Display Stage 1 --
    display_stage1(fir, analysis)

    # -- Build context for later stages --
    fir_summary = fir.get("incident_description", "")[:200]
    mapped_sections = []
    for s in analysis.get("applicable_statutes", []):
        p = s.get("primary", {})
        mapped_sections.append(f"{p.get('law', '')} {p.get('section', '')}")
        for c in s.get("corresponding_sections", []):
            mapped_sections.append(f"{c['law']} {c['section']}")

    # -- Initialize LegalQA index --
    print("\n  [INIT] Building Legal QA index (first run embeds 10K questions)...")
    qa_index = LegalQAIndex()

    # -- Stage 2: Case Similarity (automatic, no LLM) --
    print("\n  [STAGE 2] Searching for similar past cases...")
    search_query = fir_summary + " " + " ".join(mapped_sections)
    sim_result = qa_index.search_similar_cases(search_query)
    print(display_similar_cases(sim_result))

    # -- Stage 3: Precedent Q&A (interactive) --
    print(f"\n  {'=' * W}")
    print(f"  STAGE 3: PRECEDENT Q&A (Interactive)")
    print(f"  {'=' * W}")
    print(f"  Ask questions about the identified sections or related case law.")
    print(f"  Type /quit to exit  |  /cases to re-show similar cases")
    print(f"  {'=' * W}\n")

    # Initialize LLM for Q&A
    qa_engine = PrecedentQA()

    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("\n  Goodbye!")
            break
        if user_input.lower() == "/cases":
            print(display_similar_cases(sim_result))
            continue

        # Retrieve matching precedents
        retrieval = qa_index.answer_question(
            question=user_input,
            fir_context=fir_summary,
            mapped_sections=mapped_sections,
            top_k=5,
        )

        is_no_match = retrieval["status"] == "no_match"

        if is_no_match:
            # No LLM call needed
            from precedent_qa import PrecedentQA as _PQA
            answer = _PQA._format_no_match(None, mapped_sections)
        else:
            # LLM synthesis
            print("  [Searching precedents & synthesizing answer...]")
            answer = qa_engine.synthesize(
                user_question=user_input,
                retrieval_result=retrieval,
                fir_summary=fir_summary,
                mapped_sections=mapped_sections,
            )

        print(display_qa_answer(answer, is_no_match=is_no_match))

    # -- Export session --
    export_path = REPO_ROOT / "output" / "prototype_demo_output.json"
    export_data = {
        "fir": fir,
        "stage1_analysis": analysis,
        "stage2_similar_cases": sim_result,
        "mapped_sections": mapped_sections,
    }
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Session exported to: {export_path.name}")


if __name__ == "__main__":
    main()
