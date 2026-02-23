#!/usr/bin/env python3
"""
Case Similarity Search (Stage 2)
Given FIR analysis results, find similar past cases from IndicLegalQA.
"""

from __future__ import annotations
import textwrap


def display_similar_cases(similarity_result: dict, width: int = 70) -> str:
    """Format the similarity search result for terminal display."""
    
    status = similarity_result["status"]
    cases = similarity_result["cases"]
    lines = []

    border = "=" * width
    lines.append(f"\n{'':>2}{border}")
    lines.append(f"{'':>2}  STAGE 2: SIMILAR PAST CASES")
    lines.append(f"{'':>2}{border}")

    if status == "no_match":
        lines.append("")
        lines.append(f"{'':>4}⚠  No closely matching cases found in the precedent")
        lines.append(f"{'':>4}   database for this specific type of offence.")
        lines.append("")
        lines.append(f"{'':>4}This may be because:")
        lines.append(f"{'':>4}• The offence type is uncommon in SC/HC records")
        lines.append(f"{'':>4}• The dataset focuses on reported Supreme Court judgments")
        lines.append("")
        lines.append(f"{'':>4}Your Section Mapping (Stage 1) remains valid.")
        lines.append(f"{'':>4}You can still ask legal questions in Stage 3.")
        lines.append(f"{'':>2}{border}")
        return "\n".join(lines)

    if status == "partial_match":
        lines.append("")
        lines.append(f"{'':>4}⚠  No direct precedents found, but these cases share")
        lines.append(f"{'':>4}   some legal elements:")
        lines.append("")
    else:
        lines.append("")
        lines.append(f"{'':>4}Based on your FIR, here are related court cases:")
        lines.append("")

    for i, case in enumerate(cases, 1):
        sim_label = f"{case['similarity_pct']}% relevance"
        lines.append(f"{'':>4}{i}. {case['case_name']}")
        lines.append(f"{'':>7}Date: {case['date']}  |  {sim_label}  |  {case['qa_count']} Q&A pairs")
        
        # Issue (wrapped)
        issue_lines = textwrap.wrap(case["issue"], width=width - 14)
        lines.append(f"{'':>7}Issue: {issue_lines[0]}")
        for il in issue_lines[1:]:
            lines.append(f"{'':>14}{il}")
        
        # Key outcome (wrapped)
        key_lines = textwrap.wrap(case["key"], width=width - 14)
        lines.append(f"{'':>7}Key:   {key_lines[0]}")
        for kl in key_lines[1:]:
            lines.append(f"{'':>14}{kl}")
        lines.append("")

    if status == "partial_match":
        lines.append(f"{'':>4}Note: Low similarity — use as general reference only.")
        lines.append("")

    lines.append(f"{'':>2}{border}")
    return "\n".join(lines)
