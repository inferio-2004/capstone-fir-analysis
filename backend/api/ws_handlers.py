"""
WebSocket message handlers for the LexIR pipeline.

Each handler receives:
    msg          — the parsed client JSON message
    session_id   — unique connection identifier
    send         — coroutine to send a JSON dict to the client
    send_status  — coroutine to send a status message (stage, text)
    deps         — ServerDeps dataclass with shared singletons
"""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Awaitable, Any, Optional

from formatters import format_stage1, extract_mapped_sections

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
#  Shared dependencies (injected by the server at startup)
# ---------------------------------------------------------------------------
@dataclass
class ServerDeps:
    """Singletons shared across all WS handlers."""
    qa_engine: Any = None                        # PrecedentQA
    kanoon_searcher: Optional[Callable] = None   # indian_kanoon.search_and_analyze
    ensure_rag_system: Optional[Callable] = None # lazy-loader for RAG system
    sessions: dict = field(default_factory=dict)
    audit: Optional[Callable] = None             # _audit()

# Type aliases for send helpers
SendFn = Callable[[dict], Awaitable[None]]
StatusFn = Callable[[int, str], Awaitable[None]]


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _load_fir(msg: dict, send_status, is_default_ok: bool = True) -> dict | None:
    """Return FIR data from the message, or load the sample FIR as fallback."""
    fir_data = msg.get("fir")
    if fir_data is None and is_default_ok:
        fir_path = REPO_ROOT / "src_dataset_files" / "fir_sample.json"
        with open(fir_path, "r", encoding="utf-8") as f:
            fir_data = json.load(f)
    return fir_data


async def _run_stage2(mapped_sections, fir_summary, deps: ServerDeps):
    """Run Indian Kanoon search + verdict prediction (Stage 2) in a thread."""
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            None,
            lambda: deps.kanoon_searcher(
                mapped_sections=mapped_sections,
                fir_summary=fir_summary,
            ),
        )
    except Exception as e:
        return {"status": "error", "cases": [], "verdict_prediction": None, "error": str(e)}


async def _run_rag_analysis(fir_data, deps: ServerDeps):
    """Run the full RAG chain analysis in a thread."""
    system = deps.ensure_rag_system()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, system.analyze_fir_with_chains, fir_data)


# ---------------------------------------------------------------------------
#  Handler: start_analysis (stages 1 & 2)
# ---------------------------------------------------------------------------
async def handle_start_analysis(
    msg: dict, session_id: str,
    send: SendFn, send_status: StatusFn,
    deps: ServerDeps,
):
    ctx = deps.sessions[session_id]

    # 1. Load FIR
    fir_data = msg.get("fir")
    is_sample = fir_data is None
    if is_sample:
        fir_data = _load_fir(msg, send_status)
        await send_status(1, "No FIR provided — using sample FIR")

    ctx["fir"] = fir_data
    await send({"type": "fir_loaded", "stage": 1, "fir": fir_data})
    deps.audit("fir_loaded", session_id, fir_data.get("fir_id", ""))

    # 2. Stage 1
    if is_sample:
        await send_status(1, "Loading pre-computed FIR analysis...")
        analysis_path = REPO_ROOT / "output" / "fir_analysis_result_chains.json"
        if analysis_path.exists():
            with open(analysis_path, "r", encoding="utf-8") as f:
                analysis = json.load(f)
        else:
            alt_path = REPO_ROOT / "output" / "fir_analysis_result.json"
            if alt_path.exists():
                with open(alt_path, "r", encoding="utf-8") as f:
                    analysis = json.load(f)
            else:
                await send({"type": "error", "message": "No pre-computed analysis found."})
                return
    else:
        await send_status(1, "Running live RAG analysis on your FIR (Pinecone + LLM)... ~30-60 s")
        try:
            analysis = await _run_rag_analysis(fir_data, deps)
        except Exception as e:
            await send({"type": "error", "message": f"RAG analysis failed: {e}"})
            return

    ctx["analysis"] = analysis
    mapped_sections = extract_mapped_sections(analysis)
    ctx["mapped_sections"] = mapped_sections
    fir_summary = fir_data.get("incident_description", "")[:600]
    ctx["fir_summary"] = fir_summary

    stage1_data = format_stage1(fir_data, analysis, mapped_sections)
    stage1_data["_raw_analysis"] = analysis
    await send({"type": "stage1_result", "stage": 1, "data": stage1_data})
    deps.audit("stage1_complete", session_id)

    # 3. Stage 2
    await send_status(2, "Searching Indian Kanoon for real case law & predicting verdict...")
    stage2_result = await _run_stage2(mapped_sections, fir_summary, deps)
    ctx["sim_result"] = stage2_result
    await send({"type": "stage2_result", "stage": 2, "data": stage2_result})
    deps.audit("stage2_complete", session_id)

    await send_status(3, "Ready for questions. Send 'ask_question' messages.")


# ---------------------------------------------------------------------------
#  Handler: run_full_analysis (live RAG chain — slow path)
# ---------------------------------------------------------------------------
async def handle_full_analysis(
    msg: dict, session_id: str,
    send: SendFn, send_status: StatusFn,
    deps: ServerDeps,
):
    ctx = deps.sessions[session_id]

    fir_data = msg.get("fir")
    if not fir_data:
        fir_data = _load_fir(msg, send_status)
        await send_status(1, "No FIR provided — using sample FIR")

    ctx["fir"] = fir_data
    await send({"type": "fir_loaded", "stage": 1, "fir": fir_data})

    await send_status(1, "Running full RAG chain analysis (Pinecone + LLM)... This may take 30-60 seconds.")
    try:
        analysis = await _run_rag_analysis(fir_data, deps)
    except Exception as e:
        await send({"type": "error", "message": f"RAG analysis failed: {e}"})
        return

    ctx["analysis"] = analysis
    mapped_sections = extract_mapped_sections(analysis)
    ctx["mapped_sections"] = mapped_sections
    fir_summary = fir_data.get("incident_description", "")[:600]
    ctx["fir_summary"] = fir_summary

    stage1_data = format_stage1(fir_data, analysis, mapped_sections)
    stage1_data["_raw_analysis"] = analysis
    await send({"type": "stage1_result", "stage": 1, "data": stage1_data})
    deps.audit("stage1_live_complete", session_id)

    # Stage 2
    await send_status(2, "Searching Indian Kanoon for real case law & predicting verdict...")
    stage2_result = await _run_stage2(mapped_sections, fir_summary, deps)
    ctx["sim_result"] = stage2_result
    await send({"type": "stage2_result", "stage": 2, "data": stage2_result})
    deps.audit("stage2_complete", session_id)

    await send_status(3, "Ready for questions. Send 'ask_question' messages.")


# ---------------------------------------------------------------------------
#  Handler: ask_question (Stage 3 Q&A)
# ---------------------------------------------------------------------------
async def handle_ask_question(
    msg: dict, session_id: str,
    send: SendFn, send_status: StatusFn,
    deps: ServerDeps,
):
    ctx = deps.sessions[session_id]
    question = msg.get("question", "").strip()

    if not question:
        await send({"type": "error", "message": "No question provided."})
        return

    if not ctx.get("fir"):
        await send({"type": "error", "message": "No analysis started. Send 'start_analysis' first."})
        return

    deps.audit("question", session_id, question[:100])
    await send_status(3, "Searching precedents & synthesizing answer...")

    fir_summary = ctx["fir_summary"]
    mapped_sections = ctx["mapped_sections"]

    loop = asyncio.get_event_loop()

    answer_text = await loop.run_in_executor(
        None,
        lambda: deps.qa_engine.synthesize(
            user_question=question,
            retrieval_result={"status": "direct", "precedents": []},
            fir_summary=fir_summary,
            mapped_sections=mapped_sections,
        ),
    )

    await send({
        "type": "qa_answer",
        "stage": 3,
        "data": {
            "question": question,
            "answer": answer_text,
            "is_no_match": False,
            "precedents_used": 0,
            "retrieval_status": "direct",
        },
    })
    deps.audit("qa_answer", session_id, f"q={question[:50]}")
