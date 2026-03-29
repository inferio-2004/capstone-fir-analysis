#!/usr/bin/env python3
"""
Benchmark: Negative Rule Filtering — Semantic vs Literal Matching
===============================================================

Compares the current literal keyword-based negative rule filter with a semantic similarity approach
(using SentenceTransformer) for filtering out negative rules from statute chunks or case facts.

Outputs a report comparing precision, recall, and speed.
"""

import json
import os
import time
from pathlib import Path
from statistics import mean

from sentence_transformers import SentenceTransformer, util

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Load negative rules and test data
# ---------------------------------------------------------------------------
NEGATIVE_RULES_PATH = REPO_ROOT / "output" / "negative_rules_comprehensive.json"
STATUTE_CHUNKS_PATH = REPO_ROOT / "output" / "statute_chunks_complete.jsonl"


# --- Load negative rules: extract all failure_examples as negative rule phrases ---

with open(NEGATIVE_RULES_PATH, "r", encoding="utf-8") as f:
    negative_rule_objs = json.load(f)

# Build per-offence embeddings index
offence_embeddings = {}
for obj in negative_rule_objs:
    offence = obj["offence"]
    phrases = obj.get("failure_examples", [])
    if phrases:
        offence_embeddings[offence] = {
            "phrases": phrases,
            "embeddings": None  # to be filled after model is loaded
        }

# Use a sample of statute chunks for testing

# --- Use user-provided input for statute chunks ---
user_input = user_input = [
    # --- NEGATIVE (should be filtered = True) ---
    {"text": "The accused took the property with the owner's full permission.", "expected": True, "offence": "Theft"},
    {"text": "The injury was caused accidentally when the complainant slipped.", "expected": True, "offence": "Hurt"},
    {"text": "The payment demanded was part of a legitimate debt recovery process.", "expected": True, "offence": "Extortion"},
    {"text": "Both parties entered the agreement with complete knowledge and consent.", "expected": True, "offence": "Cheating"},
    {"text": "The group of five had no shared intention to rob anyone.", "expected": True, "offence": "Dacoity"},
    {"text": "The physical contact occurred during a sports match with mutual consent.", "expected": True, "offence": "Hurt"},
    {"text": "The accused used the entrusted funds exactly as authorized.", "expected": True, "offence": "Criminal Breach of Trust"},
    {"text": "The statement was a general remark not directed at any specific person.", "expected": True, "offence": "Criminal Intimidation"},
    {"text": "The accused believed he had a legal right to reclaim the property.", "expected": True, "offence": "Theft"},
    {"text": "The restraint was carried out under a valid court order.", "expected": True, "offence": "Wrongful Confinement"},

    # --- VALID (should NOT be filtered = False) ---
    {"text": "The accused forcibly entered the house at night and took gold jewelry without the owner's knowledge.", "expected": False, "offence": "Theft"},
    {"text": "The accused threatened to kill the complainant unless he handed over cash.", "expected": False, "offence": "Extortion"},
    {"text": "The victim sustained a fractured arm after being struck with an iron rod.", "expected": False, "offence": "Grievous Hurt"},
    {"text": "The accused deliberately misrepresented the property documents to induce the complainant to transfer money.", "expected": False, "offence": "Cheating"},
    {"text": "Five persons surrounded the complainant and forcibly snatched his bag.", "expected": False, "offence": "Dacoity"},
    {"text": "The accused confined the complainant in a locked room against his will for three days.", "expected": False, "offence": "Wrongful Confinement"},
    {"text": "The accused sent repeated messages threatening to harm the complainant's family.", "expected": False, "offence": "Criminal Intimidation"},
    {"text": "The accused diverted funds entrusted to him for construction into his personal account.", "expected": False, "offence": "Criminal Breach of Trust"},
    {"text": "The accused intentionally struck the complainant on the head causing unconsciousness.", "expected": False, "offence": "Hurt"},
    {"text": "The accused lured the minor away from school premises without parental consent.", "expected": False, "offence": "Kidnapping"},
]
chunks = [{"chunk_id": i, "section_text": t} for i, t in enumerate(user_input)]
chunk_texts = [c["section_text"] for c in chunks]

# ---------------------------------------------------------------------------
# Literal matching baseline
# ---------------------------------------------------------------------------
def literal_negative_filter(text, rules):
    text_lower = text.lower()
    for rule in rules:
        if rule.lower() in text_lower:
            return True
    return False

# ---------------------------------------------------------------------------
# Semantic matching (SentenceTransformer)
# ---------------------------------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")
# Fill in embeddings for each offence
for offence, data in offence_embeddings.items():
    data["embeddings"] = model.encode(data["phrases"], convert_to_tensor=True)

def semantic_negative_filter_scoped(text, offence, threshold=0.52):
    if offence not in offence_embeddings or not offence_embeddings[offence]["phrases"]:
        return False, 0.0
    rule_embeds = offence_embeddings[offence]["embeddings"]
    text_emb = model.encode([text], convert_to_tensor=True)
    cos_sims = util.cos_sim(text_emb, rule_embeds)[0]
    max_sim = float(cos_sims.max())
    return max_sim >= threshold, max_sim

# ---------------------------------------------------------------------------
# Benchmark both methods
# ---------------------------------------------------------------------------
results = []
literal_preds = []
semantic_preds = []
expected_labels = []
for i, chunk in enumerate(user_input):
    text = chunk["text"]
    expected = chunk["expected"]
    offence = chunk["offence"]
    # Literal (still global)
    t0 = time.perf_counter()
    literal = literal_negative_filter(text, [p for d in offence_embeddings.values() for p in d["phrases"]])
    t1 = time.perf_counter()
    # Semantic (offence-scoped)
    semantic, sim = semantic_negative_filter_scoped(text, offence)
    t2 = time.perf_counter()
    results.append({
        "chunk_id": i,
        "literal": literal,
        "semantic": semantic,
        "semantic_score": sim,
        "offence": offence,
        "literal_time": t1 - t0,
        "semantic_time": t2 - t1,
        "expected": expected,
        "text": text[:200],
    })
    literal_preds.append(literal)
    semantic_preds.append(semantic)
    expected_labels.append(expected)

# ---------------------------------------------------------------------------
# Aggregate and compare
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    tp = sum((p and t) for p, t in zip(y_pred, y_true))
    tn = sum((not p and not t) for p, t in zip(y_pred, y_true))
    fp = sum((p and not t) for p, t in zip(y_pred, y_true))
    fn = sum((not p and t) for p, t in zip(y_pred, y_true))
    accuracy = (tp + tn) / len(y_true) if y_true else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return accuracy, precision, recall, f1, tp, tn, fp, fn

num = len(results)
lit_count = sum(r["literal"] for r in results)
sem_count = sum(r["semantic"] for r in results)
agree = sum(r["literal"] == r["semantic"] for r in results)

lit_acc, lit_prec, lit_rec, lit_f1, lit_tp, lit_tn, lit_fp, lit_fn = compute_metrics(expected_labels, literal_preds)
sem_acc, sem_prec, sem_rec, sem_f1, sem_tp, sem_tn, sem_fp, sem_fn = compute_metrics(expected_labels, semantic_preds)

print(f"Tested {num} statute chunks.")
print(f"Literal filtered: {lit_count}")
print(f"Semantic filtered: {sem_count}")
print(f"Agreement: {agree/num:.1%}")
print(f"Avg literal time: {mean(r['literal_time'] for r in results)*1000:.2f} ms")
print(f"Avg semantic time: {mean(r['semantic_time'] for r in results)*1000:.2f} ms")
print("")
print("Literal filter metrics:")
print(f"  Accuracy:  {lit_acc:.2%}")
print(f"  Precision: {lit_prec:.2%}")
print(f"  Recall:    {lit_rec:.2%}")
print(f"  F1 score:  {lit_f1:.2%}")
print(f"  TP: {lit_tp}, TN: {lit_tn}, FP: {lit_fp}, FN: {lit_fn}")
print("")
print("Semantic filter metrics:")
print(f"  Accuracy:  {sem_acc:.2%}")
print(f"  Precision: {sem_prec:.2%}")
print(f"  Recall:    {sem_rec:.2%}")
print(f"  F1 score:  {sem_f1:.2%}")
print(f"  TP: {sem_tp}, TN: {sem_tn}, FP: {sem_fp}, FN: {sem_fn}")

# Save detailed results
with open(REPO_ROOT / "output" / "negative_rule_semantic_benchmark.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Detailed results saved to output/negative_rule_semantic_benchmark.json")
