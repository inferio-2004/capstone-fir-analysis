"""
Response formatters — transform raw pipeline output into frontend-ready JSON.
"""


def format_stage1(fir: dict, analysis: dict, mapped_sections: list) -> dict:
    """
    Structure Stage 1 results into a clean JSON payload for the frontend.
    """
    intent = analysis.get("analysis", {}).get("intent_identification", {})
    reasoning = analysis.get("analysis", {}).get("legal_reasoning", {})

    statutes = []
    for s in analysis.get("applicable_statutes", []):
        primary = s.get("primary", {})
        corresponding = [
            {"law": c["law"], "section": c["section"], "extract": c.get("extract", "")}
            for c in s.get("corresponding_sections", [])
        ]
        statutes.append({
            "primary": {
                "law": primary.get("law", ""),
                "section": primary.get("section", ""),
                "title": primary.get("title", ""),
                "reasoning": primary.get("reasoning", ""),
                "extract": primary.get("extract", ""),
            },
            "corresponding_sections": corresponding,
        })

    return {
        "fir_summary": {
            "fir_id": fir.get("fir_id", "N/A"),
            "date": fir.get("date", "N/A"),
            "complainant": fir.get("complainant_name", "N/A"),
            "accused": fir.get("accused_names", []),
            "victim": fir.get("victim_name", "N/A"),
            "location": fir.get("location", "N/A"),
            "incident": fir.get("incident_description", "N/A"),
        },
        "intent": {
            "primary": intent.get("primary_intent", "N/A"),
            "confidence": intent.get("confidence", 0),
            "secondary": intent.get("secondary_intents", []),
        },
        "severity": reasoning.get("severity_assessment", "unknown"),
        "legal_basis": reasoning.get("legal_basis", ""),
        "statutes": statutes,
        "mapped_sections": mapped_sections,
        "chunks_retrieved": analysis.get("retrieved_data", {}).get("total_chunks_retrieved", 0),
        "chunks_after_filtering": analysis.get("retrieved_data", {}).get("chunks_after_filtering", 0),
        "confidence": analysis.get("confidence", 0),
    }


def extract_mapped_sections(analysis: dict) -> list[str]:
    """Pull mapped IPC/BNS section strings from a RAG analysis result (deduplicated)."""
    seen = set()
    sections = []
    for s in analysis.get("applicable_statutes", []):
        p = s.get("primary", {})
        key = f"{p.get('law', '')} {p.get('section', '')}"
        if key not in seen:
            seen.add(key)
            sections.append(key)
        for c in s.get("corresponding_sections", []):
            ckey = f"{c['law']} {c['section']}"
            if ckey not in seen:
                seen.add(ckey)
                sections.append(ckey)
    return sections


def extract_primary_sections(analysis: dict) -> list[str]:
    """Pull only the primary statute sections from a RAG analysis result (deduplicated)."""
    seen = set()
    sections = []
    for s in analysis.get("applicable_statutes", []):
        p = s.get("primary", {})
        key = f"{p.get('law', '')} {p.get('section', '')}"
        if key not in seen:
            seen.add(key)
            sections.append(key)
    return sections
