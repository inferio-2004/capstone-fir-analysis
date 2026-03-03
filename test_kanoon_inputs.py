#!/usr/bin/env python3
"""
Test script to show exact Indian Kanoon API inputs
"""
import json
import sys
from pathlib import Path

# Add backend/api to path
sys.path.insert(0, str(Path(__file__).parent / "backend" / "api"))

from indian_kanoon import search_and_analyze

# Load sample FIR
fir_path = Path(__file__).parent / "src_dataset_files" / "fir_sample.json"
with open(fir_path, "r", encoding="utf-8") as f:
    sample_fir = json.load(f)

# Load the precomputed Stage 1 analysis (matching the sample FIR)
analysis_path = Path(__file__).parent / "output" / "fir_analysis_result.json"
with open(analysis_path, "r", encoding="utf-8") as f:
    analysis = json.load(f)

# Build mapped_sections from the analysis
mapped_sections = []
for statute in analysis["applicable_statutes"]:
    primary = statute["primary"]
    mapped_sections.append(f"{primary['law']} {primary['section']}")
    for corr in statute["corresponding_sections"]:
        mapped_sections.append(f"{corr['law']} {corr['section']}")

# Get FIR summary
fir_summary = sample_fir["incident_description"][:600]

print("\n" + "="*80)
print("RUNNING INDIAN KANOON SEARCH WITH SAMPLE FIR")
print("="*80)

# Run the search (this will print all the inputs with our new logging)
result = search_and_analyze(
    mapped_sections=mapped_sections,
    fir_summary=fir_summary,
)

print("\n" + "="*80)
print("STAGE 2 OUTPUT SUMMARY")
print("="*80)
print(f"Status: {result['status']}")
print(f"Total cases found: {len(result.get('cases', []))}")
print(f"API calls used: {result.get('api_calls_used', 0)}")

if result.get('verdict_prediction'):
    print("\nVERDICT PREDICTION:")
    vp = result['verdict_prediction']
    print(f"  Verdict: {vp.get('predicted_verdict')}")
    print(f"  Punishment: {vp.get('predicted_punishment')}")
    print(f"  Punishment Range: {vp.get('punishment_range')}")
    print(f"  Bail Likelihood: {vp.get('bail_likelihood')}")
    print(f"  Confidence: {vp.get('confidence')}")

print("\n" + "="*80)
