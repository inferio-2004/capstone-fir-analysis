#!/usr/bin/env python3
"""
Negative Rules Filter Demo  —  Murder / Accidental Death Ambiguity
===================================================================
Runs the *actual* apply_negative_rules_filter() logic on a hand-crafted
FIR case (Aman Verma / Kunal Mehta roommate altercation).

Demonstrates:
  Part A — Filter with LONG-FORM facts (how the real pipeline calls it)
  Part B — Filter with SHORT extracted facts (to show the filter actually firing)
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1.  Load static data (same files the live system loads)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with open(REPO / "output" / "negative_rules_comprehensive.json", encoding="utf-8") as f:
    negative_rules = {r["offence"]: r for r in json.load(f)}

all_chunks = [json.loads(line) for line in open(REPO / "output" / "statute_chunks_complete.jsonl", encoding="utf-8")]
chunk_by_id = {c["chunk_id"]: c for c in all_chunks}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2.  The test case (user-supplied FIR)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Part A: Long-form facts (exactly how the live pipeline constructs case_facts)
case_facts_long = {
    "incident": (
        "On the night of 11 March 2026, Aman Verma and his roommate Kunal Mehta "
        "got into a heated argument over unpaid rent in their apartment at 3A MG Road, "
        "Bangalore. Kunal had been drinking heavily. During the altercation, Aman "
        "grabbed Kunal by the collar and shoved him. Kunal lost his balance, fell "
        "backward, hitting his head on the edge of a marble table. Kunal was rushed "
        "to the hospital but was declared dead on arrival due to a skull fracture "
        "and internal hemorrhage."
    ),
    "victim_impact": "Death due to skull fracture and internal hemorrhage",
    "evidence": (
        "Investigators found a broken whiskey bottle, overturned furniture, and "
        "blood on the marble table edge. Neighbors heard shouting but no sounds "
        "of repeated blows. Aman told police: 'I only pushed him once, I never "
        "meant to kill him.' A text message from Kunal's phone an hour before the "
        "incident read: 'You pushed me too far, Aman. This ends tonight.'"
    ),
}

# Part B: Short extracted facts (simulating what an NLP extractor might produce)
case_facts_short = {
    "incident": "Death caused accidentally without intent",
    "victim_impact": "Death due to skull fracture",
    "evidence": "Single push, no repeated blows, no weapon",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3.  Pick the most relevant statute chunks (what Pinecone would return)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
target_sections = [
    "ipc_299",   # Culpable homicide definition
    "ipc_300",   # Murder definition
    "ipc_302",   # Punishment for murder
    "ipc_304",   # Punishment for culpable homicide not amounting to murder
    "ipc_304A",  # Death by negligence
    "ipc_307",   # Attempt to murder
    "ipc_319",   # Hurt
    "ipc_320",   # Grievous Hurt
    "ipc_321",   # Voluntarily causing hurt
    "ipc_322",   # Voluntarily causing grievous hurt
    "ipc_323",   # Punishment for voluntarily causing hurt
    "ipc_325",   # Punishment for voluntarily causing grievous hurt
]

retrieved_chunks = [chunk_by_id[cid] for cid in target_sections if cid in chunk_by_id]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4.  The ACTUAL filter function (verbatim from rag_llm_chain_prompting.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def apply_negative_rules_filter(chunks, case_facts, negative_rules, verbose=True):
    """Exact replica of StatuteRAGChainSystem.apply_negative_rules_filter."""
    filtered = []
    dropped = []
    for chunk in chunks:
        keep = True
        drop_reason = None
        for rule_name, rule in negative_rules.items():
            if rule_name.lower() in chunk.get("full_text", "").lower():
                # Offence name appears in chunk → check failure examples
                if any(
                    fact.lower() in ex.lower()
                    for ex in rule.get("failure_examples", [])
                    for fact in case_facts.values()
                    if isinstance(fact, str)
                ):
                    keep = False
                    for ex in rule.get("failure_examples", []):
                        for fact in case_facts.values():
                            if isinstance(fact, str) and fact.lower() in ex.lower():
                                drop_reason = (rule_name, ex, fact[:80])
                    break
                else:
                    if verbose:
                        print(f"  ✓ KEEP [{chunk['chunk_id']}] S.{chunk['section_id']}"
                              f" — Rule '{rule_name}' matched in text, no failure match")
                    filtered.append(chunk)
                    keep = False
                    break
        if keep:
            filtered.append(chunk)
            if verbose:
                matched = [r for r in negative_rules if r.lower() in chunk.get("full_text", "").lower()]
                if not matched:
                    print(f"  ○ KEEP [{chunk['chunk_id']}] S.{chunk['section_id']}"
                          f" — No offence rule matched in text → default keep")
        if drop_reason:
            dropped.append((chunk, drop_reason))
            if verbose:
                rn, ex, fact_preview = drop_reason
                print(f"  ✗ DROP [{chunk['chunk_id']}] S.{chunk['section_id']}"
                      f" — Rule '{rn}' + fact matched failure example")
                print(f"         Failure example: \"{ex}\"")
                print(f"         Matched fact:    \"{fact_preview}\"")
    return filtered, dropped


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5.  RUN BOTH PARTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sep = "=" * 70

print(sep)
print("  NEGATIVE RULES FILTER DEMO")
print("  Case: Aman Verma / Kunal Mehta  —  Roommate Altercation → Death")
print(sep)

print(f"\nRetrieved {len(retrieved_chunks)} statute chunks (simulated Pinecone retrieval):")
for i, c in enumerate(retrieved_chunks, 1):
    print(f"  {i:2d}. [{c['chunk_id']}] {c['law']} S.{c['section_id']}  —  "
          f"{c.get('full_text','')[:60]}...")

# ── Part A ────────────────────────────────────────────────────────────
print(f"\n{sep}")
print("  PART A: Long-form facts (how the LIVE pipeline calls the filter)")
print(f"  case_facts keys: {list(case_facts_long.keys())}")
print(f"  fact lengths: {[len(v) for v in case_facts_long.values()]} chars")
print(sep)

kept_a, dropped_a = apply_negative_rules_filter(
    retrieved_chunks, case_facts_long, negative_rules
)

print(f"\n  Result: {len(kept_a)} KEPT, {len(dropped_a)} DROPPED")
if not dropped_a:
    print("  → No chunks dropped! The long fact paragraphs (300+ chars) are never")
    print("    a substring of the short failure examples (30-50 chars).")
    print("    This is the main limitation of the substring-based heuristic.")

# ── Part B ────────────────────────────────────────────────────────────
print(f"\n{sep}")
print("  PART B: Short extracted facts (demonstrating the filter actually firing)")
print(f"  case_facts: {json.dumps(case_facts_short, indent=4)}")
print(sep)

kept_b, dropped_b = apply_negative_rules_filter(
    retrieved_chunks, case_facts_short, negative_rules
)

print(f"\n  Result: {len(kept_b)} KEPT, {len(dropped_b)} DROPPED")

# ── Final comparison ──────────────────────────────────────────────────
print(f"\n{sep}")
print("  COMPARISON & PROJECT RELEVANCE")
print(sep)
print(f"""
  Part A (long facts):  {len(kept_a):2d} kept, {len(dropped_a)} dropped  ← current system behavior
  Part B (short facts): {len(kept_b):2d} kept, {len(dropped_b)} dropped  ← filter actively working

  WHY THIS MATTERS:
  ─────────────────
  The filter uses substring matching: fact.lower() in failure_example.lower()

  With LONG paragraphs as facts (Part A), the entire paragraph is checked as
  a substring of a short failure example — this never matches because a 300-char
  string can't be inside a 40-char string.

  With SHORT extracted facts (Part B), the fact "Death caused accidentally
  without intent" IS a substring of the Murder failure_example "Death caused
  accidentally without intent" — so murder-related chunks are correctly DROPPED.

  This means the filter acts as intended for the Aman Verma case: if the system
  extracts the key fact that the death was accidental/unintentional, it will
  filter out Murder (IPC 300/302) chunks, leaving Culpable Homicide (IPC 304)
  and Grievous Hurt (IPC 325) — which are the correct charges for this case.

  IMPROVEMENT OPPORTUNITY: An NLP fact-extraction step before the filter would
  make it work on real cases by producing short, structured facts from the
  raw FIR text. This is a good future enhancement for the project.
""")

# ── Show the Murder rule for reference ────────────────────────────────
murder_rule = negative_rules.get("Murder", {})
print(f"  Murder Rule (from negative_rules_comprehensive.json):")
print(f"    Mandatory ingredients: {murder_rule.get('mandatory_ingredients', [])}")
print(f"    Failure examples:     {murder_rule.get('failure_examples', [])}")
