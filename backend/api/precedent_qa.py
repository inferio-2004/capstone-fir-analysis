#!/usr/bin/env python3
"""
Precedent Q&A (Stage 3)
Interactive question-answering grounded in IndicLegalQA case precedents.
Uses Groq LLM to synthesize answers from retrieved QA pairs.
"""

from __future__ import annotations
import os
import textwrap
from groq import Groq
from dotenv import load_dotenv
from model_config import groq_chat_with_fallback

load_dotenv()

SYSTEM_PROMPT = """You are an Indian legal expert assistant. Answer the user's question 
using ONLY the provided case precedents retrieved from Indian court judgments. 

Rules:
- Cite the case name and date for every claim you make
- If the precedents don't contain relevant info, say so honestly
- Be concise (3-6 sentences)
- Do NOT invent cases or citations not in the provided precedents
- Relate your answer back to the user's FIR context when relevant"""


class PrecedentQA:
    """Synthesize answers from retrieved legal QA precedents using Groq LLM."""

    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        print(f"[PrecedentQA] Groq LLM ready (role=qa, fallback enabled)")

    def synthesize(self, user_question: str, retrieval_result: dict,
                   fir_summary: str = "", mapped_sections: list = None) -> str:
        """
        Build prompt from retrieved precedents + send to LLM.
        Returns the synthesized answer string.
        """
        status = retrieval_result.get("status", "no_match")
        precedents = retrieval_result.get("precedents", [])

        # If nothing found, return fallback
        if status == "no_match" or not precedents:
            return self._format_no_match(mapped_sections or [])

        # Build context block
        context_parts = []
        if fir_summary:
            context_parts.append(f"FIR Context: {fir_summary}")
        if mapped_sections:
            context_parts.append(f"Mapped Sections: {', '.join(mapped_sections)}")

        precedent_block = ""
        for i, p in enumerate(precedents[:5], 1):
            precedent_block += (
                f"\n[Precedent {i}] Case: {p['case_name']} ({p['date']})\n"
                f"  Q: {p['question']}\n"
                f"  A: {p['answer']}\n"
            )

        user_msg = (
            f"{'  '.join(context_parts)}\n\n"
            f"Retrieved Precedents:{precedent_block}\n\n"
            f"User Question: {user_question}"
        )

        # Call Groq LLM with fallback
        try:
            return groq_chat_with_fallback(
                self.client,
                role="qa",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=500,
            )
        except Exception as e:
            return f"[LLM Error] {e}"

    def _format_no_match(self, sections: list) -> str:
        lines = [
            "No relevant precedents found in the case database for this question.",
            "",
        ]
        if sections:
            lines.append("However, based on the mapped sections from Stage 1:")
            for s in sections:
                lines.append(f"  • {s}")
            lines.append("")
            lines.append("For specific case precedents, consult SCC Online or Indian Kanoon.")
        return "\n".join(lines)


def display_qa_answer(answer: str, is_no_match: bool = False, width: int = 70) -> str:
    """Format the QA answer for terminal display."""
    border = "=" * width
    lines = []
    lines.append(f"\n{'':>2}{border}")
    lines.append(f"{'':>2}  PRECEDENT Q&A")
    lines.append(f"{'':>2}{'-' * width}")
    lines.append("")

    if is_no_match:
        lines.append(f"{'':>4}⚠  {answer.splitlines()[0]}")
        for line in answer.splitlines()[1:]:
            lines.append(f"{'':>4}{line}")
    else:
        for para in answer.split("\n"):
            wrapped = textwrap.wrap(para, width=width - 6)
            for wl in wrapped:
                lines.append(f"{'':>4}{wl}")
            if not wrapped:
                lines.append("")

    lines.append("")
    lines.append(f"{'':>2}{border}")
    return "\n".join(lines)
