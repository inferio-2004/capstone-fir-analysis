#!/usr/bin/env python3
"""
Legal QA Local Index
Embeds the IndicLegalQA dataset questions and provides fast in-memory
similarity search. Caches embeddings to disk for reuse.

This is the lightweight prototype — in production, this would be a
Pinecone namespace ("legal-qa") instead of local numpy.
"""

import json
import re
import os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Paths
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "src_dataset_files" / "IndicLegalQA Dataset_10K.json"
CACHE_DIR = REPO_ROOT / "output" / "legalqa_cache"

# Similarity thresholds
STRONG_MATCH = 0.45
PARTIAL_MATCH = 0.30

# Question-type classification patterns
QUESTION_TYPE_PATTERNS = {
    "issue": re.compile(
        r"main issue|primary issue|central issue|primary legal issue|"
        r"what was the case about|crux of the case",
        re.IGNORECASE,
    ),
    "charges": re.compile(
        r"charges? against|convicted|sections? (of|under)|offence|accused of",
        re.IGNORECASE,
    ),
    "outcome": re.compile(
        r"final decision|final outcome|outcome of|supreme court.*(decide|rule|held)|"
        r"what did the court decide|what was the result|high court.*decision",
        re.IGNORECASE,
    ),
    "punishment": re.compile(
        r"punishment|sentence|imprison|penalty|death sentence|bail|acquit",
        re.IGNORECASE,
    ),
    "evidence": re.compile(
        r"evidence|witness|testimony|forensic|identification",
        re.IGNORECASE,
    ),
}


def classify_question_type(question: str) -> str:
    """Classify a question into a type based on patterns."""
    for qtype, pattern in QUESTION_TYPE_PATTERNS.items():
        if pattern.search(question):
            return qtype
    return "other"


class LegalQAIndex:
    """In-memory vector index for the IndicLegalQA dataset."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 256):
        print("[LegalQA] Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

        # Load dataset
        print(f"[LegalQA] Loading dataset from {DATASET_PATH.name}...")
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        print(f"[LegalQA] {len(self.records)} QA pairs loaded")

        # Pre-classify question types
        for rec in self.records:
            rec["question_type"] = classify_question_type(rec["question"])

        # Build or load cached embeddings
        self.embeddings = self._load_or_build_embeddings()
        print(f"[LegalQA] Embeddings ready  shape={self.embeddings.shape}")

    # ------------------------------------------------------------------
    def _load_or_build_embeddings(self) -> np.ndarray:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        emb_path = CACHE_DIR / "question_embeddings.npy"
        meta_path = CACHE_DIR / "record_count.txt"

        # Check cache validity
        if emb_path.exists() and meta_path.exists():
            cached_count = int(meta_path.read_text().strip())
            if cached_count == len(self.records):
                print("[LegalQA] Loading cached embeddings...")
                return np.load(str(emb_path))

        # Build embeddings
        print(f"[LegalQA] Embedding {len(self.records)} questions (one-time, ~2-3 min)...")
        questions = [r["question"] for r in self.records]
        embeddings = self.model.encode(
            questions,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # Cache to disk
        np.save(str(emb_path), embeddings)
        meta_path.write_text(str(len(self.records)))
        print("[LegalQA] Embeddings cached to disk")

        return embeddings

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 15) -> list:
        """
        Semantic search over QA questions.
        Returns list of {record, score} dicts sorted by similarity.
        """
        q_emb = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        # Cosine similarity (embeddings are already L2-normalised → dot product)
        scores = (self.embeddings @ q_emb.T).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "record": self.records[idx],
                "score": float(scores[idx]),
            })
        return results

    # ------------------------------------------------------------------
    def search_similar_cases(self, fir_text: str, top_k_pairs: int = 20,
                             max_cases: int = 3) -> dict:
        """
        Stage 2: Case Similarity Search.
        Embed FIR text → find similar QA pairs → group by case → return top cases.
        Returns {status, cases: [{case_name, date, similarity, issue, key, qa_pairs}]}
        """
        results = self.search(fir_text, top_k=top_k_pairs)

        # Group by case_name
        case_groups = {}
        for r in results:
            cname = r["record"]["case_name"]
            if cname not in case_groups:
                case_groups[cname] = {
                    "case_name": cname,
                    "date": r["record"].get("judgment_date") or r["record"].get("judgement_date") or "N/A",
                    "scores": [],
                    "qa_pairs": [],
                }
            case_groups[cname]["scores"].append(r["score"])
            case_groups[cname]["qa_pairs"].append(r["record"])

        # Rank cases by average similarity
        for cg in case_groups.values():
            cg["avg_score"] = float(np.mean(cg["scores"]))

        ranked = sorted(case_groups.values(), key=lambda x: x["avg_score"], reverse=True)

        # Determine match quality
        top_score = ranked[0]["avg_score"] if ranked else 0

        if top_score >= STRONG_MATCH:
            status = "strong_match"
        elif top_score >= PARTIAL_MATCH:
            status = "partial_match"
        else:
            status = "no_match"

        # Extract issue/key for top cases
        cases_out = []
        for case in ranked[:max_cases]:
            # Also fetch ALL QA pairs for this case from the full dataset
            all_case_qa = [r for r in self.records if r["case_name"] == case["case_name"]]

            issue = None
            key = None
            for qa in all_case_qa:
                if qa["question_type"] == "issue" and not issue:
                    issue = qa["answer"][:250]
                if qa["question_type"] == "outcome" and not key:
                    key = qa["answer"][:250]

            # Fallbacks
            if not issue:
                issue = all_case_qa[0]["answer"][:250] if all_case_qa else "N/A"
            if not key:
                key = all_case_qa[-1]["answer"][:250] if all_case_qa else "N/A"

            cases_out.append({
                "case_name": case["case_name"],
                "date": case["date"],
                "similarity_pct": round(case["avg_score"] * 100),
                "issue": issue,
                "key": key,
                "qa_count": len(all_case_qa),
            })

        return {"status": status, "cases": cases_out}

    # ------------------------------------------------------------------
    def answer_question(self, question: str, fir_context: str = "",
                        mapped_sections: list = None, top_k: int = 5) -> dict:
        """
        Stage 3: Precedent Q&A retrieval.
        Returns retrieved QA pairs ready to be sent to LLM.
        """
        # Combine question with FIR context for better retrieval
        enriched_query = question
        if mapped_sections:
            enriched_query += " " + " ".join(mapped_sections)

        results = self.search(enriched_query, top_k=top_k)

        top_score = results[0]["score"] if results else 0

        if top_score < PARTIAL_MATCH:
            return {
                "status": "no_match",
                "precedents": [],
                "fallback_sections": mapped_sections or [],
            }

        precedents = []
        for r in results:
            if r["score"] >= PARTIAL_MATCH:
                precedents.append({
                    "case_name": r["record"]["case_name"],
                    "date": r["record"].get("judgment_date") or r["record"].get("judgement_date") or "N/A",
                    "question": r["record"]["question"],
                    "answer": r["record"]["answer"][:300],
                    "score": round(r["score"], 4),
                })

        return {
            "status": "strong_match" if top_score >= STRONG_MATCH else "partial_match",
            "precedents": precedents,
        }
