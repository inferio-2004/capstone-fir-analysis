#!/usr/bin/env python3
"""
FIR PDF Generator — FORM IF-1 (First Information Report under Section 154 Cr.P.C)
==================================================================================
Generates a filled PDF matching the standard Indian FIR form layout.
Uses fpdf2 (pure-Python, no external binaries needed).
"""
from __future__ import annotations

import io
from pathlib import Path

from fpdf import FPDF


class FIRPDF(FPDF):
    """Custom FPDF subclass with helpers for the IF-1 form layout."""

    # Page margins
    LEFT = 15
    RIGHT = 15

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(self.LEFT, 15, self.RIGHT)
        self.usable_w = 210 - self.LEFT - self.RIGHT  # A4 width minus margins

    # ── helper: section label + value on same line ────────────────────────
    def field_line(self, label: str, value: str, label_w: int = 55):
        self.set_font("Helvetica", "B", 9)
        self.cell(label_w, 6, label, ln=0)
        self.set_font("Helvetica", "", 9)
        # If the value is too long for one line, wrap it
        remaining_w = self.usable_w - label_w
        if self.get_string_width(value) > remaining_w:
            self.multi_cell(remaining_w, 5, value)
        else:
            self.cell(0, 6, value, ln=1)

    def field_line_inline(self, *pairs, h: int = 6):
        """Render alternating label/value pairs on one line.
        pairs = (label, value, width, label, value, width, ...)
        Each group is (label_text, value_text, total_cell_width).
        """
        for i in range(0, len(pairs), 3):
            label, value, w = pairs[i], str(pairs[i+1]), pairs[i+2]
            self.set_font("Helvetica", "B", 9)
            lw = self.get_string_width(label) + 2
            self.cell(lw, h, label, ln=0)
            self.set_font("Helvetica", "", 9)
            # Truncate value if it would overflow the allocated cell width
            val_w = w - lw
            if val_w > 0 and self.get_string_width(value) > val_w - 2:
                while value and self.get_string_width(value + "...") > val_w - 2:
                    value = value[:-1]
                value = value + "..."
            self.cell(max(val_w, 5), h, value, ln=0)
        self.ln(h)

    def section_header(self, number: str, text: str):
        self.ln(1)
        self.set_font("Helvetica", "B", 10)
        self.cell(8, 7, f"{number}.", ln=0)
        self.cell(0, 7, text, ln=1)

    def wrapped_text(self, text: str, indent: int = 8):
        self.set_font("Helvetica", "", 9)
        x = self.get_x()
        self.set_x(x + indent)
        self.multi_cell(self.usable_w - indent, 5, text)
        self.ln(2)

    def dotted_line(self, w: int | None = None):
        y = self.get_y()
        x1 = self.get_x()
        x2 = x1 + (w or self.usable_w)
        self.set_draw_color(180, 180, 180)
        self.dashed_line(x1, y, x2, y, dash_length=1, space_length=1)
        self.set_draw_color(0, 0, 0)
        self.ln(2)


def generate_fir_pdf(fields: dict) -> bytes:
    """
    Generate a filled FIR PDF as bytes.

    Parameters
    ----------
    fields : dict
        The ``fields`` dict from ``build_fir_pdf_payload()``.

    Returns
    -------
    bytes
        PDF file content ready to be served or saved.
    """
    pdf = FIRPDF()
    pdf.add_page()

    # ═══════════════════════════  TITLE  ═══════════════════════════════════
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 7, "FORM - IF1 - (Integrated Form)", ln=1, align="C")
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 6, "FIRST INFORMATION REPORT", ln=1, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, "(Under Section 154 Cr.P.C)", ln=1, align="C")
    pdf.ln(4)

    f = fields  # shorthand

    # ═══════════════════════════  1. Header  ══════════════════════════════
    pdf.section_header("1", "")
    pdf.field_line_inline(
        "Dist: ", f.get("1_district", ""), 35,
        "P.S.: ", f.get("1_police_station", ""), 40,
        "Year: ", f.get("1_year", ""), 20,
        "F.I.R. No.: ", f.get("1_fir_number", ""), 40,
        "Date: ", f.get("1_date", ""), 35,
    )
    pdf.ln(3)

    # ═══════════════════════════  2. Acts & Sections  ════════════════════
    pdf.section_header("2", "Acts & Sections")
    lh = 6  # line height for act rows
    for i, label in enumerate(["(i)", "(ii)", "(iii)"], 1):
        act_key = f"2{'i'*i}_act"
        sec_key = f"2{'i'*i}_sections"
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(12, lh, f"  {label}", ln=0)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(12, lh, "*Act: ", ln=0)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(70, lh, f.get(act_key, ""), ln=0)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(20, lh, "*Sections:", ln=0)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, lh, f.get(sec_key, ""), ln=1)

    other = f.get("2iv_other_acts_sections", "")
    if other:
        pdf.cell(12, lh, "  (iv)", ln=0)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(35, lh, "* Other Acts & Sections: ", ln=0)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, lh, other, ln=1)
    pdf.ln(3)

    # ═══════════════════════════  3. Occurrence  ═════════════════════════
    pdf.section_header("3", "")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(8, 6, "", ln=0)
    pdf.field_line_inline(
        "(a) * Occurrence of Offence: * Day: ", f.get("3a_day", ""), 80,
        "* Date: ", f.get("3a_date", ""), 40,
        "* Time: ", f.get("3a_time", ""), 35,
    )
    pdf.cell(8, 6, "", ln=0)
    pdf.field_line_inline(
        "(b) Information received at P.S.  Date: ", f.get("3b_info_received_date", ""), 90,
        "Time: ", f.get("3b_info_received_time", ""), 40,
    )
    pdf.cell(8, 6, "", ln=0)
    pdf.field_line_inline(
        "(c) General Diary Reference: Entry No(s): ", f.get("3c_gd_entry_no", ""), 95,
        "Time: ", f.get("3c_gd_time", ""), 40,
    )
    pdf.ln(3)

    # ═══════════════════════════  4. Type of information  ═════════════════
    pdf.section_header("4", "Type of information:")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(8, 6, "", ln=0)
    pdf.cell(0, 6, f"* {f.get('4_type_of_information', 'Written')} / Oral", ln=1)
    pdf.ln(3)

    # ═══════════════════════════  5. Place of occurrence  ═════════════════
    pdf.section_header("5", "Place of occurrence:")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(8, 6, "", ln=0)
    pdf.field_line_inline(
        "(a) Direction and Distance from P.S.: ", f.get("5a_direction_distance", ""), 100,
        "Beat No.: ", f.get("5a_beat_no", ""), 40,
    )
    pdf.field_line("     (b) * Address:", f.get("5b_address", ""), label_w=35)
    if f.get("5c_outside_ps"):
        pdf.field_line("     (c) Outside P.S.:", f.get("5c_outside_ps", ""), label_w=40)
    pdf.ln(3)

    # ═══════════════════════════  6. Complainant  ═════════════════════════
    pdf.section_header("6", "Complainant / Information:")
    pdf.field_line("     (a) Name:", f.get("6a_name", ""), label_w=35)
    pdf.field_line("     (b) Father's / Husband's Name:", f.get("6b_father_husband_name", ""), label_w=60)
    pdf.field_line_inline(
        "     (c) Date / Year of Birth: ", f.get("6c_dob", ""), 70,
        "(d) Nationality: ", f.get("6d_nationality", "Indian"), 50,
    )
    if f.get("6e_passport_no"):
        pdf.field_line_inline(
            "     (e) Passport No: ", f.get("6e_passport_no", ""), 55,
            "Date of Issue: ", f.get("6e_passport_date_of_issue", ""), 45,
            "Place of Issue: ", f.get("6e_passport_place_of_issue", ""), 45,
        )
    pdf.field_line("     (f) Occupation:", f.get("6f_occupation", ""), label_w=35)
    pdf.field_line("     (g) Address:", f.get("6g_address", ""), label_w=35)
    pdf.ln(3)

    # ═══════════════════════════  7. Accused  ═════════════════════════════
    pdf.section_header("7", "Details of known / suspected / unknown / accused with full particulars:")
    pdf.wrapped_text(f.get("7_accused_details", "Unknown"))
    pdf.ln(2)

    # ═══════════════════════════  8. Delay reason  ════════════════════════
    pdf.section_header("8", "Reasons for delay in reporting by the complainant / Informant:")
    pdf.wrapped_text(f.get("8_delay_reason", "N/A"))
    pdf.ln(2)

    # ═══════════════════════════  9. Properties  ══════════════════════════
    pdf.section_header("9", "Particulars of properties stolen / involved:")
    pdf.wrapped_text(f.get("9_properties_stolen", "As per FIR contents."))
    pdf.ln(2)

    # ═══════════════════════════  10. Total value  ════════════════════════
    pdf.section_header("10", "")
    pdf.field_line("     * Total value of properties stolen / involved:", f.get("10_total_value", ""), label_w=90)
    pdf.ln(2)

    # ═══════════════════════════  11. Inquest  ════════════════════════════
    pdf.section_header("11", "")
    pdf.field_line("     * Inquest Report / U.D. Case No., if any:", f.get("11_inquest_report", ""), label_w=85)
    pdf.ln(2)

    # ═══════════════════════════  12. FIR Contents  ═══════════════════════
    pdf.section_header("12", "F.I.R. Contents (Attach separate sheets, if required):")
    pdf.wrapped_text(f.get("12_fir_contents", ""))
    pdf.ln(3)

    # ═══════════════════════════  13. Action taken  ═══════════════════════
    pdf.section_header("13", "Action taken:")
    action = f.get("13_action_taken", "")
    if action:
        pdf.wrapped_text(action)
    else:
        pdf.wrapped_text(
            "Since the above report reveals commission of offence(s) u/s as "
            "mentioned at Item No. 2, registered the case and took up the "
            "investigation / direction / ........................ Rank ............ "
            "to take up the investigation transferred to P.S. ........................ "
            "on point of jurisdiction."
        )
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 5,
        "F.I.R. read over to the complainant / Informant, admitted to be correctly "
        "recorded and copy given to the Complainant / Informant free of cost."
    )
    pdf.ln(5)

    # ═══════════════════════  Officer & Signatures  ═══════════════════════
    # Ensure enough space — start a new page if less than 60mm remain
    if pdf.get_y() > 240:
        pdf.add_page()

    # Two columns: left = complainant signature, right = officer block
    y_before = pdf.get_y()
    half_w = pdf.usable_w / 2

    # Left column — complainant signature
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(half_w, 6, "14.  Signature / Thumb-impression", ln=1)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(half_w, 6, "       of the complainant / informant", ln=1)
    y_after_left = pdf.get_y() + 15  # leave space for actual signature

    # Right column — officer block (drawn at same starting Y)
    pdf.set_xy(pdf.LEFT + half_w, y_before)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(half_w, 6, "Signature of the Officer-in-charge,", ln=0)
    pdf.set_xy(pdf.LEFT + half_w, y_before + 6)
    pdf.cell(half_w, 6, "Police Station", ln=0)
    pdf.set_xy(pdf.LEFT + half_w, y_before + 12)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(15, 6, "* Name:", ln=0)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(half_w - 15, 6, f.get("officer_name", ""), ln=0)
    pdf.set_xy(pdf.LEFT + half_w, y_before + 18)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(15, 6, "* Rank:", ln=0)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(30, 6, f.get("officer_rank", ""), ln=0)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(10, 6, "No.:", ln=0)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, f.get("officer_no", ""), ln=0)
    y_after_right = y_before + 28

    # Move cursor below whichever column is taller
    pdf.set_y(max(y_after_left, y_after_right))
    pdf.ln(5)

    # ═══════════════════════════  15.  Despatch  ══════════════════════════
    pdf.section_header("15", "Date & time of despatch to the court:")
    pdf.field_line_inline(
        "     Date: ", f.get("15_despatch_date", ""), 55,
        "Time: ", f.get("15_despatch_time", ""), 45,
    )
    pdf.ln(3)

    # ── output ────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
