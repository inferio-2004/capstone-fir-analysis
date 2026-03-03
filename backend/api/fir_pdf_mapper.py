from __future__ import annotations

from datetime import datetime


def _safe(value, default: str = "") -> str:
    return default if value is None else str(value)


def _parse_date(raw: str | None) -> tuple[str, str, str]:
    """Return (day_name, date_str, time_str) from an ISO-ish date string."""
    if not raw:
        return ("", "", "")
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return (dt.strftime("%A"), dt.strftime("%d/%m/%Y"), dt.strftime("%I:%M %p"))
    except Exception:
        return ("", raw, "")


def build_fir_pdf_payload(fir: dict, analysis: dict | None = None) -> dict:
    """
    Map FIR input data + optional LLM analysis output to the fields of
    FORM – IF1 (First Information Report under Section 154 Cr.P.C).

    Parameters
    ----------
    fir : dict
        Raw FIR input (complainant_name, accused_names, incident_description, …)
    analysis : dict | None
        Output of the RAG chain pipeline. Expected shape:
        {
          "analysis": {
            "intent_identification": {"primary_intent": "...", ...},
            "legal_reasoning": {
              "applicable_statutes": [{"section": "IPC 394", "law": "...", "reasoning": "..."}],
              "legal_basis": "...",
              "severity_assessment": "...",
              "confidence": 0.93,
            }
          },
          "applicable_statutes": [{"primary": {"law": "IPC", "section": "394", ...}, ...}],
          ...
        }
    """
    accused = fir.get("accused_names") or []
    day_name, date_str, time_str = _parse_date(fir.get("date"))

    # ── 2. Acts & Sections  (from LLM analysis) ──────────────────────────
    act_section_rows: list[tuple[str, str]] = []
    if analysis:
        statutes = []
        # Try enriched statutes first
        for s in analysis.get("applicable_statutes") or []:
            p = s.get("primary", s)
            statutes.append((p.get("law", "IPC"), p.get("section", "")))
        # Fallback to analysis.legal_reasoning.applicable_statutes
        if not statutes:
            lr = (analysis.get("analysis") or {}).get("legal_reasoning") or {}
            for s in lr.get("applicable_statutes") or []:
                raw_sec = s.get("section", "")
                if raw_sec.upper().startswith("IPC"):
                    act_section_rows.append(("Indian Penal Code, 1860", raw_sec.split()[-1]))
                elif raw_sec.upper().startswith("BNS"):
                    act_section_rows.append(("Bharatiya Nyaya Sanhita, 2023", raw_sec.split()[-1]))
                else:
                    act_section_rows.append(("IPC", raw_sec))
            statutes = act_section_rows
        if not act_section_rows:
            for law, sec in statutes:
                act_name = {
                    "IPC": "Indian Penal Code, 1860",
                    "BNS": "Bharatiya Nyaya Sanhita, 2023",
                }.get(law.upper(), law)
                act_section_rows.append((act_name, sec))

    # Pad to 3 rows (form has 3 Act/Section lines + 1 "Other")
    while len(act_section_rows) < 3:
        act_section_rows.append(("", ""))
    other_acts = "; ".join(
        f"{a} § {s}" for a, s in act_section_rows[3:]
    ) if len(act_section_rows) > 3 else ""

    # ── 9/10. Properties ─────────────────────────────────────────────────
    incident = _safe(fir.get("incident_description", ""))
    victim_impact = _safe(fir.get("victim_impact", ""))

    # ── 12. FIR Contents ──────────────────────────────────────────────────
    fir_contents_parts = [incident]
    if victim_impact:
        fir_contents_parts.append(f"Victim Impact: {victim_impact}")
    evidence = _safe(fir.get("evidence", ""))
    if evidence:
        fir_contents_parts.append(f"Evidence: {evidence}")
    fir_contents = "\n\n".join(fir_contents_parts)

    # ── Severity / legal basis from analysis ──────────────────────────────
    severity = ""
    legal_basis = ""
    if analysis:
        lr = (analysis.get("analysis") or {}).get("legal_reasoning") or {}
        severity = lr.get("severity_assessment", "")
        legal_basis = lr.get("legal_basis", "")

    fields = {
        # 1. Header
        "1_district": _safe(fir.get("district", "")),
        "1_police_station": _safe(fir.get("police_station", "")),
        "1_year": date_str[:4] if len(date_str) >= 4 else "",
        "1_fir_number": _safe(fir.get("fir_id", "")),
        "1_date": date_str,

        # 2. Acts & Sections  (3 rows + other)
        "2i_act": act_section_rows[0][0],
        "2i_sections": act_section_rows[0][1],
        "2ii_act": act_section_rows[1][0],
        "2ii_sections": act_section_rows[1][1],
        "2iii_act": act_section_rows[2][0],
        "2iii_sections": act_section_rows[2][1],
        "2iv_other_acts_sections": other_acts,

        # 3. Occurrence of offence
        "3a_day": day_name,
        "3a_date": date_str,
        "3a_time": time_str,
        "3b_info_received_date": "",
        "3b_info_received_time": "",
        "3c_gd_entry_no": "",
        "3c_gd_time": "",

        # 4. Type of information
        "4_type_of_information": "Written",

        # 5. Place of occurrence
        "5a_direction_distance": _safe(fir.get("location", "")),
        "5a_beat_no": "",
        "5b_address": _safe(fir.get("location", "")),
        "5c_outside_ps": "",
        "5c_district": "",

        # 6. Complainant information
        "6a_name": _safe(fir.get("complainant_name", "")),
        "6b_father_husband_name": _safe(fir.get("father_husband_name", "")),
        "6c_dob": _safe(fir.get("complainant_dob", "")),
        "6d_nationality": _safe(fir.get("nationality", "Indian")),
        "6e_passport_no": "",
        "6e_passport_date_of_issue": "",
        "6e_passport_place_of_issue": "",
        "6f_occupation": _safe(fir.get("occupation", "")),
        "6g_address": _safe(fir.get("complainant_address", "")),

        # 7. Accused details
        "7_accused_details": "\n".join(
            f"{i+1}. {a}" for i, a in enumerate(accused)
        ) if accused else "Unknown",

        # 8. Delay reason
        "8_delay_reason": _safe(fir.get("delay_reason", "")),

        # 9. Properties stolen/involved
        "9_properties_stolen": _safe(fir.get("properties_stolen", "")),

        # 10. Total value of properties
        "10_total_value": _safe(fir.get("property_value", "")),

        # 11. Inquest report
        "11_inquest_report": "",

        # 12. FIR Contents
        "12_fir_contents": fir_contents,

        # 13. Action taken  (auto-fill boilerplate with sections)
        "13_action_taken": (
            f"Since the above report reveals commission of offence(s) u/s "
            f"{', '.join(s for _, s in act_section_rows[:3] if s)} "
            f"as mentioned at Item No. 2, registered the case and took up "
            f"the investigation."
        ) if any(s for _, s in act_section_rows) else "",
        "13_rank": "",

        # 14. Signature (left blank — physical signature)
        "14_signature": "",

        # 15. Despatch to court
        "15_despatch_date": "",
        "15_despatch_time": "",

        # Officer-in-charge
        "officer_name": "",
        "officer_rank": "",
        "officer_no": "",
    }

    # Extra analysis metadata (not form fields, but useful for display)
    analysis_summary = {}
    if analysis:
        ii = (analysis.get("analysis") or {}).get("intent_identification") or {}
        analysis_summary = {
            "primary_intent": ii.get("primary_intent", ""),
            "severity": severity,
            "legal_basis": legal_basis,
            "statutes_applied": [
                f"{a} § {s}" for a, s in act_section_rows if s
            ],
        }

    return {
        "fields": fields,
        "analysis_summary": analysis_summary,
        "metadata": {
            "form": "FORM IF-1 (First Information Report under Section 154 Cr.P.C)",
            "field_count": len([v for v in fields.values() if v]),
            "accused_count": len(accused),
        },
    }
