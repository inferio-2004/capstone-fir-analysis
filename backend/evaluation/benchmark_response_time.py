#!/usr/bin/env python3
"""
LexIR Pipeline Response-Time Benchmark
=======================================
Measures wall-clock time for each stage (and sub-step) of the LexIR
analysis pipeline across N runs, then writes a JSON report to output/.

Stages measured
---------------
  Stage 1 — RAG Analysis (rag_llm_chain_prompting.py)
    1a  Initial vector retrieval (Pinecone)
    1b  Negative-rules filter
    1c  Chain 1: Intent identification (LLM)
    1d  Intent-driven second retrieval
    1e  Chain 2: Legal reasoning (LLM)
    1f  Statute enrichment (IPC↔BNS mapping + extracts)

  Stage 2 — Indian Kanoon Case Search (indian_kanoon.py)
    2a  Build fact query (LLM)
    2b  Kanoon API search (all strategies)
    2c  Fetch judgment text for each case
    2d  Summarize each case (LLM)
    2e  Predict verdict (LLM)
    2f  Rank section influence (LLM)

  Stage 3 — Precedent Q&A (precedent_qa.py)
    3a  Synthesize answer (LLM)

Usage:
    cd backend
    python evaluation/benchmark_response_time.py          # 3 runs (default)
    python evaluation/benchmark_response_time.py --runs 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, stdev

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
API_DIR = REPO_ROOT / "backend" / "api"
sys.path.insert(0, str(API_DIR))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / "backend" / ".env")
load_dotenv(REPO_ROOT / ".env")

from rag_llm_chain_prompting import StatuteRAGChainSystem
from indian_kanoon import search_and_analyze
from precedent_qa import PrecedentQA
from formatters import extract_mapped_sections
from intent_queries import intent_to_retrieval_queries


# ---------------------------------------------------------------------------
# Load sample FIR
# ---------------------------------------------------------------------------
def load_sample_fir() -> dict:
    p = REPO_ROOT / "src_dataset_files" / "fir_sample.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Timed Stage 1 — RAG Analysis (with sub-step instrumentation)
# ---------------------------------------------------------------------------
def timed_stage1(rag: StatuteRAGChainSystem, fir_data: dict) -> dict:
    """Run Stage 1 with per-sub-step timing. Returns timing dict + analysis result."""
    times = {}
    case_facts = {
        "incident": fir_data.get("incident_description", ""),
        "victim_impact": fir_data.get("victim_impact", ""),
        "evidence": fir_data.get("evidence", ""),
    }
    query_text = f"{case_facts['incident']} {case_facts['victim_impact']}"

    # 1a — Initial vector retrieval
    t0 = time.perf_counter()
    retrieved_chunks = rag.retrieve_relevant_statutes(query_text, top_k=20)
    times["1a_vector_retrieval"] = time.perf_counter() - t0

    # 1b — Negative-rules filter
    t0 = time.perf_counter()
    filtered_chunks = rag.apply_negative_rules_filter(retrieved_chunks, case_facts)
    times["1b_negative_rules_filter"] = time.perf_counter() - t0

    # 1c — Chain 1: Intent identification
    intent_input = {
        "complainant": fir_data.get("complainant_name", "Unknown"),
        "accused": ", ".join(fir_data.get("accused_names", [])),
        "incident": fir_data.get("incident_description", "Unknown"),
        "victim_impact": fir_data.get("victim_impact", "Unknown"),
        "evidence": fir_data.get("evidence", "Unknown"),
    }
    t0 = time.perf_counter()
    intent_str = rag._invoke_with_fallback("intent", rag.intent_prompt, intent_input)
    times["1c_intent_identification_llm"] = time.perf_counter() - t0

    try:
        intent_result = json.loads(intent_str)
    except Exception:
        intent_result = {"primary_intent": "Unknown", "confidence": 0, "secondary_intents": []}

    # 1d — Intent-driven second retrieval
    t0 = time.perf_counter()
    for iq in intent_to_retrieval_queries(
        intent_result.get("primary_intent", ""),
        intent_result.get("secondary_intents", []),
    ):
        extra = rag.retrieve_relevant_statutes(iq, top_k=8)
        filtered_chunks = rag._merge_chunks(filtered_chunks, extra)
    filtered_chunks.sort(key=lambda c: c.get("similarity_score", 0), reverse=True)
    times["1d_intent_driven_retrieval"] = time.perf_counter() - t0

    # 1e — Chain 2: Legal reasoning
    statute_ctx = "\n".join(
        f"[{c['law']} {c['section_id']}] {c['section_text']}" for c in filtered_chunks[:20]
    )
    reasoning_input = {
        "incident": fir_data.get("incident_description", "Unknown"),
        "victim_impact": fir_data.get("victim_impact", "None reported"),
        "evidence": fir_data.get("evidence", "None reported"),
        "accused": ", ".join(fir_data.get("accused_names", [])),
        "victim": fir_data.get("victim_name", "Unknown"),
        "primary_intent": intent_result.get("primary_intent", "Unknown"),
        "statute_context": statute_ctx,
    }
    t0 = time.perf_counter()
    reasoning_str = rag._invoke_with_fallback("reasoning", rag.reasoning_prompt, reasoning_input)
    times["1e_legal_reasoning_llm"] = time.perf_counter() - t0

    try:
        reasoning_result = json.loads(reasoning_str)
    except Exception:
        reasoning_result = {
            "applicable_statutes": [], "legal_basis": "Error parsing",
            "severity_assessment": "unknown", "confidence": 0,
        }

    # 1f — Statute enrichment
    t0 = time.perf_counter()
    enriched = rag._enrich_statutes(reasoning_result.get("applicable_statutes", []))
    times["1f_statute_enrichment"] = time.perf_counter() - t0

    # Assemble full Stage 1 result (same structure as analyze_fir_with_chains)
    analysis = {
        "status": "success",
        "chain_prompting_stages": 2,
        "fir_summary": {
            "complainant": fir_data.get("complainant_name"),
            "accused": fir_data.get("accused_names"),
            "incident": fir_data.get("incident_description", "")[:150],
        },
        "analysis": {
            "intent_identification": intent_result,
            "legal_reasoning": reasoning_result,
        },
        "retrieved_data": {
            "total_chunks_retrieved": len(retrieved_chunks),
            "chunks_after_filtering": len(filtered_chunks),
        },
        "applicable_statutes": enriched,
        "severity": reasoning_result.get("severity_assessment", "unknown"),
        "confidence": reasoning_result.get("confidence", 0),
    }

    times["stage1_total"] = sum(times.values())
    return {"times": times, "analysis": analysis}


# ---------------------------------------------------------------------------
# Timed Stage 2 — Indian Kanoon (sub-step instrumented wrapper)
# ---------------------------------------------------------------------------
def timed_stage2(mapped_sections: list[str], fir_summary: str) -> dict:
    """Run Stage 2 with per-sub-step timing."""
    import indian_kanoon as ik
    from groq_prompts import build_fact_query, summarize_case_with_llm, predict_verdict, rank_section_influence

    times = {}
    ipc_sections = ik._extract_ipc_sections(mapped_sections)

    # 2a — Build fact query (LLM)
    t0 = time.perf_counter()
    fact_query = build_fact_query(fir_summary, ipc_sections) if fir_summary else ""
    times["2a_build_fact_query_llm"] = time.perf_counter() - t0

    # 2b — Kanoon API searches (all strategies)
    all_cases = []
    seen_tids = set()
    api_calls = 0

    def _add_cases(result, section_label, limit):
        added = 0
        if result.get("error"):
            return added
        for doc in result.get("docs", []):
            if added >= limit or len(all_cases) >= ik.MAX_TOTAL_CASES:
                break
            tid = doc.get("tid")
            if tid in seen_tids:
                continue
            title = ik._clean_html(doc.get("title", "Unknown Case"))
            docsource = doc.get("docsource", "Unknown Court")
            if not ik._is_actual_case(title, docsource):
                continue
            seen_tids.add(tid)
            all_cases.append({
                "title": title, "tid": tid, "court": docsource,
                "snippet": ik._clean_html(doc.get("headline", ""))[:500],
                "section": section_label,
                "date": ik._extract_date_from_title(title),
                "url": f"https://indiankanoon.org/doc/{tid}/" if tid else "",
                "summary": "",
            })
            added += 1
        return added

    t0 = time.perf_counter()
    if fact_query:
        result = ik.search_kanoon(fact_query)
        api_calls += 1
        _add_cases(result, "Fact-match", min(3, ik.MAX_TOTAL_CASES))
    for sec in ipc_sections:
        if len(all_cases) >= ik.MAX_TOTAL_CASES:
            break
        result = ik.search_kanoon(f"Section {sec} IPC accused conviction sentence")
        api_calls += 1
        _add_cases(result, f"IPC {sec}", ik.MAX_PER_SECTION)
    if len(all_cases) < 2:
        for sec in ipc_sections:
            if len(all_cases) >= ik.MAX_TOTAL_CASES:
                break
            result = ik.search_kanoon(f"Section {sec} IPC guilty punishment")
            api_calls += 1
            _add_cases(result, f"IPC {sec}", ik.MAX_PER_SECTION)
    times["2b_kanoon_api_search"] = time.perf_counter() - t0

    # 2c — Fetch judgment text
    t0 = time.perf_counter()
    doc_texts = {}
    for case in all_cases:
        tid = case["tid"]
        if not tid:
            continue
        doc_data = ik.get_doc_detail(tid)
        api_calls += 1
        if not doc_data.get("error"):
            doc_texts[tid] = ik._clean_html(doc_data.get("doc", "")) or case["snippet"]
    times["2c_fetch_judgments"] = time.perf_counter() - t0

    # 2d — Summarize each case and keep only relevant matches (LLM)
    t0 = time.perf_counter()
    relevant_cases = []
    for case in all_cases:
        tid = case["tid"]
        if tid and tid in doc_texts:
            summary_result = summarize_case_with_llm(
                doc_texts[tid],
                case_title=case["title"],
                fir_summary=fir_summary,
                ipc_sections=[f"IPC {s}" for s in ipc_sections],
                return_metadata=True,
            )
            if summary_result.get("relevant") and summary_result.get("summary"):
                case["summary"] = summary_result["summary"]
                relevant_cases.append(case)
    times["2d_summarize_cases_llm"] = time.perf_counter() - t0

    # 2e — Predict verdict (LLM)
    cases_with_summaries = [c for c in relevant_cases if c.get("summary")]
    t0 = time.perf_counter()
    verdict = predict_verdict(
        fir_summary=fir_summary,
        ipc_sections=[f"IPC {s}" for s in ipc_sections],
        case_summaries=cases_with_summaries or relevant_cases,
    )
    times["2e_predict_verdict_llm"] = time.perf_counter() - t0

    # 2f — Rank section influence (LLM)
    t0 = time.perf_counter()
    section_influence = rank_section_influence(
        mapped_sections=mapped_sections,
        fir_summary=fir_summary,
        verdict=verdict,
        case_summaries=cases_with_summaries or relevant_cases,
    )
    times["2f_rank_influence_llm"] = time.perf_counter() - t0

    times["stage2_total"] = sum(times.values())

    stage2_result = {
        "status": "success" if cases_with_summaries else "partial",
        "cases": relevant_cases,
        "verdict_prediction": verdict,
        "section_influence": section_influence,
        "api_calls_used": api_calls,
    }
    return {"times": times, "result": stage2_result}


# ---------------------------------------------------------------------------
# Timed Stage 3 — Precedent Q&A
# ---------------------------------------------------------------------------
def timed_stage3(qa_engine: PrecedentQA, fir_summary: str,
                 mapped_sections: list[str], stage2_cases: list[dict]) -> dict:
    """Run Stage 3 with a standard legal question and measure response time."""
    test_question = "What is the likely punishment for the accused under the applicable sections?"

    # Build mock precedents from Stage 2 cases so synthesize actually calls the LLM
    mock_precedents = []
    for c in stage2_cases[:3]:
        mock_precedents.append({
            "case_name": c.get("title", "Unknown"),
            "date": c.get("date", "N/A"),
            "question": "What was the court's decision?",
            "answer": c.get("summary", c.get("snippet", "No summary available")),
        })
    if not mock_precedents:
        mock_precedents = [{"case_name": "Test", "date": "2024",
                            "question": "Punishment?", "answer": "Imprisonment."}]

    t0 = time.perf_counter()
    answer = qa_engine.synthesize(
        user_question=test_question,
        retrieval_result={"status": "found", "precedents": mock_precedents},
        fir_summary=fir_summary,
        mapped_sections=mapped_sections,
    )
    elapsed = time.perf_counter() - t0

    return {
        "times": {"3a_qa_synthesize_llm": elapsed, "stage3_total": elapsed},
        "question": test_question,
        "answer_length": len(answer),
    }


# ---------------------------------------------------------------------------
# Aggregate stats helper
# ---------------------------------------------------------------------------
def aggregate_runs(all_runs: list[dict]) -> dict:
    """Compute mean, stdev, min, max for every timing key across runs."""
    keys = sorted({k for run in all_runs for k in run})
    stats = {}
    for k in keys:
        values = [run[k] for run in all_runs if k in run]
        if not values:
            continue
        stats[k] = {
            "mean_s": round(mean(values), 3),
            "stdev_s": round(stdev(values), 3) if len(values) > 1 else 0.0,
            "min_s": round(min(values), 3),
            "max_s": round(max(values), 3),
            "runs": len(values),
        }
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LexIR Pipeline Response-Time Benchmark")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs (default: 3)")
    args = parser.parse_args()
    num_runs = args.runs

    print("=" * 70)
    print("LexIR Pipeline Response-Time Benchmark")
    print(f"Runs: {num_runs}")
    print("=" * 70)

    # Load FIR
    fir_data = load_sample_fir()
    print(f"[*] FIR loaded: {fir_data.get('fir_id', 'N/A')}")

    # Initialise components (one-time cost, not benchmarked)
    print("[*] Initialising RAG system (Pinecone + LLM)... ", end="", flush=True)
    t_init = time.perf_counter()
    rag = StatuteRAGChainSystem()
    qa = PrecedentQA()
    init_time = time.perf_counter() - t_init
    print(f"done ({init_time:.1f}s)")

    all_stage1_times = []
    all_stage2_times = []
    all_stage3_times = []
    all_total_times = []
    run_details = []

    for run_idx in range(1, num_runs + 1):
        print(f"\n{'─' * 70}")
        print(f"  Run {run_idx}/{num_runs}")
        print(f"{'─' * 70}")

        run_start = time.perf_counter()

        # --- Stage 1 ---
        print("  [Stage 1] RAG analysis...", flush=True)
        s1 = timed_stage1(rag, fir_data)
        s1_times = s1["times"]
        analysis = s1["analysis"]
        mapped_sections = extract_mapped_sections(analysis)
        fir_summary = fir_data.get("incident_description", "")[:600]
        print(f"    → {s1_times['stage1_total']:.2f}s  ({len(mapped_sections)} sections mapped)")

        # --- Stage 2 ---
        print("  [Stage 2] Indian Kanoon...", flush=True)
        s2 = timed_stage2(mapped_sections, fir_summary)
        s2_times = s2["times"]
        n_cases = len(s2["result"]["cases"])
        print(f"    → {s2_times['stage2_total']:.2f}s  ({n_cases} cases, {s2['result']['api_calls_used']} API calls)")

        # --- Stage 3 ---
        print("  [Stage 3] Q&A...", flush=True)
        s3 = timed_stage3(qa, fir_summary, mapped_sections, s2["result"]["cases"])
        s3_times = s3["times"]
        print(f"    → {s3_times['stage3_total']:.2f}s")

        run_total = time.perf_counter() - run_start

        # Collect
        all_stage1_times.append(s1_times)
        all_stage2_times.append(s2_times)
        all_stage3_times.append(s3_times)
        all_total_times.append(run_total)

        per_run = {**s1_times, **s2_times, **s3_times, "pipeline_total": run_total}

        # Computed aggregates: LLM-only vs retrieval-only
        per_run["llm_intent_classification"] = s1_times["1c_intent_identification_llm"]
        per_run["llm_legal_reasoning"] = s1_times["1e_legal_reasoning_llm"]
        per_run["llm_all_total"] = (
            s1_times["1c_intent_identification_llm"]
            + s1_times["1e_legal_reasoning_llm"]
            + s2_times["2a_build_fact_query_llm"]
            + s2_times["2d_summarize_cases_llm"]
            + s2_times["2e_predict_verdict_llm"]
            + s2_times["2f_rank_influence_llm"]
            + s3_times["3a_qa_synthesize_llm"]
        )
        per_run["kanoon_retrieval_total"] = (
            s2_times["2b_kanoon_api_search"]
            + s2_times["2c_fetch_judgments"]
        )
        per_run["vector_retrieval_total"] = (
            s1_times["1a_vector_retrieval"]
            + s1_times["1d_intent_driven_retrieval"]
        )
        per_run["non_llm_total"] = run_total - per_run["llm_all_total"]

        run_details.append(per_run)

        print(f"  ▸ Total: {run_total:.2f}s  |  S1={s1_times['stage1_total']:.2f}s  S2={s2_times['stage2_total']:.2f}s  S3={s3_times['stage3_total']:.2f}s")

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    combined_runs = []
    for rd in run_details:
        combined_runs.append(rd)

    stats = aggregate_runs(combined_runs)

    report = {
        "benchmark": "LexIR Pipeline Response-Time",
        "fir_id": fir_data.get("fir_id", "N/A"),
        "num_runs": num_runs,
        "initialization_time_s": round(init_time, 3),
        "summary": {
            "stage1_avg_s": stats.get("stage1_total", {}).get("mean_s", 0),
            "stage2_avg_s": stats.get("stage2_total", {}).get("mean_s", 0),
            "stage3_avg_s": stats.get("stage3_total", {}).get("mean_s", 0),
            "pipeline_avg_s": stats.get("pipeline_total", {}).get("mean_s", 0),
        },
        "llm_vs_retrieval_breakdown": {
            "llm_intent_classification": stats.get("llm_intent_classification", {}),
            "llm_legal_reasoning": stats.get("llm_legal_reasoning", {}),
            "llm_all_total": stats.get("llm_all_total", {}),
            "kanoon_retrieval_total": stats.get("kanoon_retrieval_total", {}),
            "vector_retrieval_total": stats.get("vector_retrieval_total", {}),
            "non_llm_total": stats.get("non_llm_total", {}),
        },
        "stage1_substeps": {k: v for k, v in stats.items() if k.startswith("1")},
        "stage2_substeps": {k: v for k, v in stats.items() if k.startswith("2")},
        "stage3_substeps": {k: v for k, v in stats.items() if k.startswith("3")},
        "totals": {k: v for k, v in stats.items() if "total" in k or "pipeline" in k},
        "per_run_detail": [
            {k: round(v, 3) for k, v in rd.items()} for rd in run_details
        ],
    }

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  RESPONSE-TIME BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Runs: {num_runs}  |  FIR: {fir_data.get('fir_id')}")
    print(f"  Init (one-time): {init_time:.1f}s")
    print()

    def _row(label, key):
        s = stats.get(key, {})
        if not s:
            return
        print(f"  {label:<36s}  {s['mean_s']:>7.3f}s  ±{s['stdev_s']:>6.3f}s  (min={s['min_s']:.3f}, max={s['max_s']:.3f})")

    print("  Stage 1 — RAG Analysis")
    _row("  Vector retrieval (Pinecone)", "1a_vector_retrieval")
    _row("  Negative-rules filter", "1b_negative_rules_filter")
    _row("  Intent identification (LLM)", "1c_intent_identification_llm")
    _row("  Intent-driven retrieval", "1d_intent_driven_retrieval")
    _row("  Legal reasoning (LLM)", "1e_legal_reasoning_llm")
    _row("  Statute enrichment", "1f_statute_enrichment")
    _row("  ── Stage 1 TOTAL", "stage1_total")
    print()

    print("  Stage 2 — Indian Kanoon")
    _row("  Build fact query (LLM)", "2a_build_fact_query_llm")
    _row("  Kanoon API search", "2b_kanoon_api_search")
    _row("  Fetch judgments", "2c_fetch_judgments")
    _row("  Summarize cases (LLM)", "2d_summarize_cases_llm")
    _row("  Predict verdict (LLM)", "2e_predict_verdict_llm")
    _row("  Rank influence (LLM)", "2f_rank_influence_llm")
    _row("  ── Stage 2 TOTAL", "stage2_total")
    print()

    print("  Stage 3 — Q&A")
    _row("  Q&A synthesize (LLM)", "3a_qa_synthesize_llm")
    _row("  ── Stage 3 TOTAL", "stage3_total")
    print()

    _row("  ═══ PIPELINE TOTAL", "pipeline_total")
    print()

    print("  LLM vs Retrieval Breakdown")
    _row("  Intent classification (LLM)", "llm_intent_classification")
    _row("  Legal reasoning (LLM)", "llm_legal_reasoning")
    _row("  All LLM calls combined", "llm_all_total")
    _row("  Kanoon retrieval (API+fetch)", "kanoon_retrieval_total")
    _row("  Vector retrieval (Pinecone)", "vector_retrieval_total")
    _row("  Non-LLM total", "non_llm_total")
    print()

    # Percentage breakdown
    llm_avg = stats.get("llm_all_total", {}).get("mean_s", 0)
    pipe_avg = stats.get("pipeline_total", {}).get("mean_s", 1)
    kanoon_avg = stats.get("kanoon_retrieval_total", {}).get("mean_s", 0)
    vec_avg = stats.get("vector_retrieval_total", {}).get("mean_s", 0)
    non_llm_avg = stats.get("non_llm_total", {}).get("mean_s", 0)
    print("  Time Distribution (% of pipeline)")
    print(f"    LLM calls:          {llm_avg/pipe_avg*100:5.1f}%  ({llm_avg:.1f}s)")
    print(f"    Kanoon retrieval:   {kanoon_avg/pipe_avg*100:5.1f}%  ({kanoon_avg:.1f}s)")
    print(f"    Vector retrieval:   {vec_avg/pipe_avg*100:5.1f}%  ({vec_avg:.1f}s)")
    print(f"    Other (filter/enrich): {(non_llm_avg-kanoon_avg-vec_avg)/pipe_avg*100:5.1f}%  ({non_llm_avg-kanoon_avg-vec_avg:.1f}s)")
    print()

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    out_path = REPO_ROOT / "output" / "response_time_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved → {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
