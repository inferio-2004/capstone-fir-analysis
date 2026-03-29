#!/usr/bin/env python3
"""
Benchmark: Feature 2 — Extractive Summarization (Step 2d)
==========================================================
Tests the TF-IDF extractive summarization algorithm on sample legal texts.

Run: python benchmark_feature_2_summarization.py
"""

import re
import math
from collections import Counter
import numpy as np


LEGAL_BOOST_WORDS = {"convicted", "acquitted", "sentenced", "punished",
                     "ipc", "bns", "section", "guilty", "imprisonment",
                     "fine", "held", "court"}


def _tokenize_words(text: str) -> list[str]:
    """Simple word tokenizer."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def summarize_text(text: str) -> str:
    """
    Extractive summary via per-sentence TF-IDF (matrix over sentences × terms).
    Top-K sentences (K = max(3, n//5), capped at n), returned in original order.
    
    Algorithm:
    1. Split input text into sentences
    2. Build TF-IDF matrix over sentences
    3. Score each sentence by sum of TF-IDF scores
    4. Pick top-K where K = max(3, len(sentences) // 5)
    5. Return in ORIGINAL order (not score order)
    """
    if not text or not str(text).strip():
        return ""

    raw = str(text).strip()
    # Split on sentence boundaries
    parts = re.split(r"(?<=[.!?])\s+", raw)
    sentences = [p.strip() for p in parts if p.strip()]
    if not sentences:
        return ""

    n = len(sentences)
    sentence_tokens = [_tokenize_words(s) for s in sentences]
    if not any(sentence_tokens):
        return sentences[0]

    # Build vocabulary
    vocab = sorted({w for tokens in sentence_tokens for w in tokens})
    if not vocab:
        return sentences[0]

    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    # TF matrix: term frequency per sentence
    tf = np.zeros((n, V), dtype=np.float64)
    for i, tokens in enumerate(sentence_tokens):
        if not tokens:
            continue
        den = len(tokens)
        for w, cnt in Counter(tokens).items():
            tf[i, w2i[w]] = cnt / den

    # IDF: inverse document frequency
    df = np.zeros(V, dtype=np.float64)
    for j, term in enumerate(vocab):
        df[j] = sum(1 for tokens in sentence_tokens if term in tokens)
    idf = np.log(n / np.maximum(df, 1.0))

    # Score = sum of TF-IDF for all words in sentence
    scores = (tf * idf).sum(axis=1)

    # Apply legal keyword boost (1.3x multiplier for sentences with legal keywords)
    for i, tokens in enumerate(sentence_tokens):
        if tokens and any(w in LEGAL_BOOST_WORDS for w in tokens):
            scores[i] *= 1.3

    # Select top-K sentences
    k = min(max(3, n // 5), n)  # At least 3, at most 20% of sentences
    top_idx = np.argsort(-scores)[:k]

    # Return in ORIGINAL order (re-sort by position)
    selected_indices = sorted(int(i) for i in top_idx)
    return " ".join(sentences[i] for i in selected_indices)


def run_benchmarks():
    """Run summarization on sample legal texts and show results."""

    test_cases = [
        {
            "name": "Short Judgment (Theft Case)",
            "text": """
                The accused was apprehended near the railway station on January 15, 2023. 
                The complainant alleged that her purse containing cash and jewelry was stolen. 
                The police recovered the stolen items from the possession of the accused. 
                The accused claimed that he found the items lying on the ground. 
                However, the court found this explanation implausible given the circumstances. 
                Three eyewitnesses testified that they saw the accused snatching the purse. 
                The medical examination confirmed injuries on the complainant's arm. 
                The defense argued that the accused had no prior criminal record. 
                The prosecution presented CCTV footage showing the theft. 
                After considering all evidence, the court convicted the accused under Section 379 IPC.
            """,
        },
        {
            "name": "Medium Judgment (Assault Case)",
            "text": """
                This case relates to an incident that occurred on March 3, 2022, at approximately 8:30 PM. 
                The victim was walking home from work when he was attacked by three persons. 
                The accused were identified as neighbors who had previous disputes with the victim. 
                The victim suffered multiple injuries including a fractured arm and head trauma. 
                He was immediately taken to the hospital where he received treatment for two weeks. 
                The investigating officer recorded statements from the victim and witnesses. 
                The medical report confirmed that the injuries were caused by blunt weapons. 
                Two eyewitnesses positively identified all three accused during the identification parade. 
                The accused denied all charges and claimed they were elsewhere at the time of incident. 
                Their alibi was investigated and found to be false by the police. 
                The motive was established as a property dispute between the families. 
                The prosecution examined twelve witnesses during the trial. 
                The defense counsel argued that there were inconsistencies in witness statements. 
                However, the court found the prosecution evidence reliable and consistent. 
                The accused were convicted under Sections 323 and 325 IPC for causing hurt.
            """,
        },
        {
            "name": "Long Judgment (Murder Case)",
            "text": """
                The present case involves the brutal murder of the deceased on the night of June 12, 2021. 
                The deceased was a local businessman who operated a grocery store in the neighborhood. 
                The accused was his business partner with whom he had financial disputes. 
                On the fateful night, the deceased was last seen leaving his shop around 9 PM. 
                His body was discovered the next morning in an abandoned warehouse. 
                The post-mortem examination revealed death caused by multiple stab wounds. 
                The murder weapon, a kitchen knife, was recovered from the crime scene. 
                Forensic analysis confirmed the presence of the accused's fingerprints on the weapon handle. 
                DNA samples from under the victim's nails matched the accused's profile. 
                The prosecution relied on circumstantial evidence to establish guilt. 
                The motive was firmly established through bank records showing embezzlement. 
                The accused had transferred large sums from their joint business account. 
                When confronted, the accused threatened to kill the deceased. 
                This threat was witnessed by two employees of the shop. 
                Phone location data placed the accused at the crime scene during the time of murder. 
                The accused's phone records showed suspicious calls to the victim that evening. 
                The defense argued that the evidence was planted by the police. 
                They claimed the accused had an alibi of being at a family function. 
                However, this alibi was contradicted by CCTV footage from the venue. 
                The trial court examined fifteen witnesses over a period of six months. 
                After careful consideration of all evidence, the court found the accused guilty. 
                The conviction was recorded under Section 302 IPC for murder.
            """,
        },
    ]

    print("=" * 80)
    print("FEATURE 2 BENCHMARK: Extractive Summarization (TF-IDF)")
    print("=" * 80)
    print()

    for tc in test_cases:
        name = tc["name"]
        text = tc["text"]

        # Count sentences in original
        raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        total_sentences = len(raw_sentences)

        # Generate summary
        summary = summarize_text(text)

        # Count sentences in summary
        summary_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]
        summary_count = len(summary_sentences)

        print(f"\n{'─' * 80}")
        print(f"Test Case: {name}")
        print(f"{'─' * 80}")
        print(f"Original: {total_sentences} sentences")
        print(f"Summary:  {summary_count} sentences (compression ratio: {summary_count/total_sentences:.1%})")
        print(f"\n📄 ORIGINAL TEXT (first 200 chars):\n{text[:200].strip()}...")
        print(f"\n📝 SUMMARY:\n{summary}")
        print()

    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print("\nKey Metrics:")
    print("- Algorithm: TF-IDF based extractive summarization")
    print("- No external LLM calls (pure Python + NumPy)")
    print("- Sentences selected: max(3, n//5) = at least 3, at most 20% of sentences")
    print("- Output order: Original sentence order (not by score)")


if __name__ == "__main__":
    run_benchmarks()
