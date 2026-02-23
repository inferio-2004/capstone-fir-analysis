#!/usr/bin/env python3
"""
LexIR Backend — FastAPI + WebSocket Server
===========================================
Wraps the 3-stage Legal Intelligence pipeline as a WebSocket API
for real-time communication with a React frontend.

WebSocket Protocol (JSON messages):
  Client → Server:
    { "type": "start_analysis", "fir": {...} }          — Run stages 1 & 2
    { "type": "start_analysis" }                         — Run with default FIR
    { "type": "ask_question", "question": "..." }        — Stage 3 Q&A
    { "type": "show_cases" }                             — Re-send stage 2 results
    { "type": "run_full_analysis", "fir": {...} }        — Run RAG chain from scratch (slow)

  Server → Client:
    { "type": "status",         "stage": 0, "message": "..." }
    { "type": "fir_loaded",     "stage": 1, "fir": {...} }
    { "type": "stage1_result",  "stage": 1, "data": {...} }
    { "type": "stage2_result",  "stage": 2, "data": {...} }
    { "type": "qa_answer",      "stage": 3, "data": {...} }
    { "type": "error",          "message": "..." }

REST Endpoints:
    GET  /health              — Health check
    GET  /api/fir/sample      — Get sample FIR JSON
    POST /api/fir/upload      — Upload FIR image/PDF for OCR
    POST /api/fir/json        — Submit FIR as JSON

Usage:
    cd backend
    uvicorn server:app --reload --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import os
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
#  Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
API_DIR = REPO_ROOT / "backend" / "api"
sys.path.insert(0, str(API_DIR))

# ---------------------------------------------------------------------------
#  FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LexIR — Legal Intelligence & Retrieval API",
    version="1.0.0",
    description="WebSocket-powered legal FIR analysis pipeline",
)

# CORS — allow React dev server (localhost:3000 / 5173) and any origin in dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
#  Global singletons (loaded once at startup)
# ---------------------------------------------------------------------------
qa_engine = None         # PrecedentQA   — Groq-backed LLM
rag_system = None        # StatuteRAGChainSystem — Pinecone + LLM chains

kanoon_searcher = None   # IndianKanoon API wrapper

# Per-session state (keyed by connection id)
sessions: dict = {}


# ---------------------------------------------------------------------------
#  Startup / Shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    """Load heavy models once when the server starts."""
    global qa_engine

    print("[SERVER] Initializing PrecedentQA (Groq LLM)...")
    from precedent_qa import PrecedentQA
    qa_engine = PrecedentQA()

    # Indian Kanoon API (lightweight — no heavy model)
    global kanoon_searcher
    from indian_kanoon import search_and_analyze
    kanoon_searcher = search_and_analyze
    print("[SERVER] ✓ Indian Kanoon API ready")

    print("[SERVER] ✓ All models loaded — server ready")


def _ensure_rag_system():
    """Lazy-load the StatuteRAGChainSystem (only when full analysis requested)."""
    global rag_system
    if rag_system is None:
        print("[SERVER] Loading StatuteRAGChainSystem (Pinecone + LLM)...")
        from rag_llm_chain_prompting import StatuteRAGChainSystem
        rag_system = StatuteRAGChainSystem()
        rag_system.create_chains()
        print("[SERVER] ✓ RAG system loaded")
    return rag_system


# ---------------------------------------------------------------------------
#  Pydantic models for REST endpoints
# ---------------------------------------------------------------------------
class FIRInput(BaseModel):
    fir_id: str = "UNKNOWN"
    date: Optional[str] = None
    complainant_name: Optional[str] = None
    accused_names: list = []
    victim_name: Optional[str] = None
    incident_description: str = ""
    victim_impact: str = ""
    evidence: str = ""
    location: str = ""
    police_station: str = ""


# ---------------------------------------------------------------------------
#  REST Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": qa_engine is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/fir/sample")
async def get_sample_fir():
    """Return the sample FIR JSON bundled with the repo."""
    fir_path = REPO_ROOT / "src_dataset_files" / "fir_sample.json"
    if not fir_path.exists():
        raise HTTPException(status_code=404, detail="Sample FIR not found")
    with open(fir_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/fir/json")
async def submit_fir_json(fir: FIRInput):
    """Accept a FIR as JSON and store it for the next WebSocket session."""
    return {"status": "ok", "fir": fir.dict()}


@app.post("/api/fir/upload")
async def upload_fir_image(file: UploadFile = File(...)):
    """
    Accept an image/PDF of a FIR, run OCR, return structured JSON.
    Requires Tesseract (pytesseract) to be installed on the system.
    """
    from ocr_to_fir import process_path_to_fir

    # Save to temp file
    tmp_dir = REPO_ROOT / "output" / "uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{uuid.uuid4().hex}_{file.filename}"

    content = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)

    try:
        fir_json = process_path_to_fir(tmp_path)
        return {"status": "ok", "fir": fir_json}
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"OCR failed: {e}")
    finally:
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()


# ---------------------------------------------------------------------------
#  Audit logging
# ---------------------------------------------------------------------------
AUDIT_LOG = REPO_ROOT / "logs" / "audit_log.jsonl"


def _audit(action: str, session_id: str, detail: str = ""):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "session_id": session_id,
        "detail": detail,
    }
    try:
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # non-critical


# ---------------------------------------------------------------------------
#  WebSocket Handler
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = uuid.uuid4().hex[:12]
    sessions[session_id] = {
        "fir": None,
        "analysis": None,
        "sim_result": None,
        "kanoon_result": None,
        "mapped_sections": [],
        "fir_summary": "",
    }
    _audit("ws_connect", session_id)

    async def send(msg: dict):
        """Send JSON message to the client."""
        await ws.send_json(msg)

    async def send_status(stage: int, message: str):
        await send({"type": "status", "stage": stage, "message": message})

    try:
        await send_status(0, "Connected to LexIR server. Send 'start_analysis' to begin.")

        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await send({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "")

            # ----------------------------------------------------------
            #  START ANALYSIS — stages 1 & 2
            # ----------------------------------------------------------
            if msg_type == "start_analysis":
                await _handle_start_analysis(msg, session_id, send, send_status)

            # ----------------------------------------------------------
            #  RUN FULL ANALYSIS — from-scratch RAG chain (Stage 1 live)
            # ----------------------------------------------------------
            elif msg_type == "run_full_analysis":
                await _handle_full_analysis(msg, session_id, send, send_status)

            # ----------------------------------------------------------
            #  ASK QUESTION — Stage 3 Q&A
            # ----------------------------------------------------------
            elif msg_type == "ask_question":
                await _handle_ask_question(msg, session_id, send, send_status)

            # ----------------------------------------------------------
            #  SEARCH KANOON \u2014 re-run Indian Kanoon search
            # ----------------------------------------------------------
            elif msg_type == "search_kanoon":
                ctx = sessions[session_id]
                if ctx.get("sim_result"):
                    await send({"type": "stage2_result", "stage": 2, "data": ctx["sim_result"]})
                else:
                    await send({"type": "error", "message": "No analysis run yet."})

            # ----------------------------------------------------------
            #  SHOW CASES — re-send stage 2
            # ----------------------------------------------------------
            elif msg_type == "show_cases":
                ctx = sessions[session_id]
                if ctx["sim_result"]:
                    await send({
                        "type": "stage2_result",
                        "stage": 2,
                        "data": ctx["sim_result"],
                    })
                else:
                    await send({"type": "error", "message": "No analysis run yet. Send 'start_analysis' first."})

            # ----------------------------------------------------------
            #  UNKNOWN
            # ----------------------------------------------------------
            else:
                await send({"type": "error", "message": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        _audit("ws_disconnect", session_id)
    except Exception as e:
        _audit("ws_error", session_id, str(e))
        try:
            await send({"type": "error", "message": f"Server error: {e}"})
        except Exception:
            pass
    finally:
        sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
#  Handler: start_analysis (live RAG chain for custom FIR, pre-computed for sample)
# ---------------------------------------------------------------------------
async def _handle_start_analysis(msg, session_id, send, send_status):
    ctx = sessions[session_id]

    # 1. Load FIR
    fir_data = msg.get("fir")
    is_sample = fir_data is None
    if is_sample:
        # Load sample FIR
        fir_path = REPO_ROOT / "src_dataset_files" / "fir_sample.json"
        with open(fir_path, "r", encoding="utf-8") as f:
            fir_data = json.load(f)
        await send_status(1, "No FIR provided — using sample FIR")

    ctx["fir"] = fir_data
    await send({"type": "fir_loaded", "stage": 1, "fir": fir_data})
    _audit("fir_loaded", session_id, fir_data.get("fir_id", ""))

    # 2. Stage 1 — Use pre-computed for sample FIR, live RAG for custom FIR
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
            system = _ensure_rag_system()
            loop = asyncio.get_event_loop()
            analysis = await loop.run_in_executor(
                None, system.analyze_fir_with_chains, fir_data
            )
        except Exception as e:
            await send({"type": "error", "message": f"RAG analysis failed: {e}"})
            return

    ctx["analysis"] = analysis

    # Build mapped sections
    mapped_sections = []
    for s in analysis.get("applicable_statutes", []):
        p = s.get("primary", {})
        mapped_sections.append(f"{p.get('law', '')} {p.get('section', '')}")
        for c in s.get("corresponding_sections", []):
            mapped_sections.append(f"{c['law']} {c['section']}")
    ctx["mapped_sections"] = mapped_sections

    fir_summary = fir_data.get("incident_description", "")[:600]
    ctx["fir_summary"] = fir_summary

    # Format stage 1 result for the frontend
    stage1_data = _format_stage1(fir_data, analysis, mapped_sections)
    await send({"type": "stage1_result", "stage": 1, "data": stage1_data})
    _audit("stage1_complete", session_id)

    # 3. Stage 2 \u2014 Indian Kanoon case law search + verdict prediction
    await send_status(2, "Searching Indian Kanoon for real case law & predicting verdict...")

    loop = asyncio.get_event_loop()
    try:
        stage2_result = await loop.run_in_executor(
            None,
            lambda: kanoon_searcher(
                mapped_sections=mapped_sections,
                fir_summary=fir_summary,
            ),
        )
    except Exception as e:
        stage2_result = {"status": "error", "cases": [], "verdict_prediction": None, "error": str(e)}

    ctx["sim_result"] = stage2_result
    await send({"type": "stage2_result", "stage": 2, "data": stage2_result})
    _audit("stage2_complete", session_id)

    await send_status(3, "Ready for questions. Send 'ask_question' messages.")


# ---------------------------------------------------------------------------
#  Handler: run_full_analysis (live RAG chain — slow)
# ---------------------------------------------------------------------------
async def _handle_full_analysis(msg, session_id, send, send_status):
    ctx = sessions[session_id]

    fir_data = msg.get("fir")
    if not fir_data:
        fir_path = REPO_ROOT / "src_dataset_files" / "fir_sample.json"
        with open(fir_path, "r", encoding="utf-8") as f:
            fir_data = json.load(f)
        await send_status(1, "No FIR provided — using sample FIR")

    ctx["fir"] = fir_data
    await send({"type": "fir_loaded", "stage": 1, "fir": fir_data})

    await send_status(1, "Running full RAG chain analysis (Pinecone + LLM)... This may take 30-60 seconds.")

    try:
        system = _ensure_rag_system()
        loop = asyncio.get_event_loop()
        analysis = await loop.run_in_executor(
            None, system.analyze_fir_with_chains, fir_data
        )
    except Exception as e:
        await send({"type": "error", "message": f"RAG analysis failed: {e}"})
        return

    ctx["analysis"] = analysis

    # Build mapped sections
    mapped_sections = []
    for s in analysis.get("applicable_statutes", []):
        p = s.get("primary", {})
        mapped_sections.append(f"{p.get('law', '')} {p.get('section', '')}")
        for c in s.get("corresponding_sections", []):
            mapped_sections.append(f"{c['law']} {c['section']}")
    ctx["mapped_sections"] = mapped_sections

    fir_summary = fir_data.get("incident_description", "")[:600]
    ctx["fir_summary"] = fir_summary

    stage1_data = _format_stage1(fir_data, analysis, mapped_sections)
    await send({"type": "stage1_result", "stage": 1, "data": stage1_data})
    _audit("stage1_live_complete", session_id)

    # Stage 2 \u2014 Indian Kanoon case law + verdict prediction
    await send_status(2, "Searching Indian Kanoon for real case law & predicting verdict...")
    loop = asyncio.get_event_loop()
    try:
        stage2_result = await loop.run_in_executor(
            None,
            lambda: kanoon_searcher(
                mapped_sections=mapped_sections,
                fir_summary=fir_summary,
            ),
        )
    except Exception as e:
        stage2_result = {"status": "error", "cases": [], "verdict_prediction": None, "error": str(e)}

    ctx["sim_result"] = stage2_result
    await send({"type": "stage2_result", "stage": 2, "data": stage2_result})
    _audit("stage2_complete", session_id)

    await send_status(3, "Ready for questions. Send 'ask_question' messages.")


# ---------------------------------------------------------------------------
#  Handler: ask_question (Stage 3)
# ---------------------------------------------------------------------------
async def _handle_ask_question(msg, session_id, send, send_status):
    ctx = sessions[session_id]
    question = msg.get("question", "").strip()

    if not question:
        await send({"type": "error", "message": "No question provided."})
        return

    if not ctx.get("fir"):
        await send({"type": "error", "message": "No analysis started. Send 'start_analysis' first."})
        return

    _audit("question", session_id, question[:100])
    await send_status(3, "Searching precedents & synthesizing answer...")

    fir_summary = ctx["fir_summary"]
    mapped_sections = ctx["mapped_sections"]

    loop = asyncio.get_event_loop()

    # Synthesize answer using Groq LLM directly (no embedding retrieval)
    answer_text = await loop.run_in_executor(
        None,
        lambda: qa_engine.synthesize(
            user_question=question,
            retrieval_result={"status": "direct", "precedents": []},
            fir_summary=fir_summary,
            mapped_sections=mapped_sections,
        ),
    )
    is_no_match = False
    retrieval = {"status": "direct", "precedents": []}

    await send({
        "type": "qa_answer",
        "stage": 3,
        "data": {
            "question": question,
            "answer": answer_text,
            "is_no_match": is_no_match,
            "precedents_used": 0,
            "retrieval_status": "direct",
        },
    })
    _audit("qa_answer", session_id, f"q={question[:50]} match={retrieval['status']}")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _format_stage1(fir: dict, analysis: dict, mapped_sections: list) -> dict:
    """
    Structure Stage 1 results into a clean JSON payload for the frontend.
    """
    intent = analysis.get("analysis", {}).get("intent_identification", {})
    reasoning = analysis.get("analysis", {}).get("legal_reasoning", {})

    statutes = []
    for s in analysis.get("applicable_statutes", []):
        primary = s.get("primary", {})
        corresponding = [
            {"law": c["law"], "section": c["section"], "extract": c.get("extract", "")}
            for c in s.get("corresponding_sections", [])
        ]
        statutes.append({
            "primary": {
                "law": primary.get("law", ""),
                "section": primary.get("section", ""),
                "title": primary.get("title", ""),
                "reasoning": primary.get("reasoning", ""),
                "extract": primary.get("extract", ""),
            },
            "corresponding_sections": corresponding,
        })

    return {
        "fir_summary": {
            "fir_id": fir.get("fir_id", "N/A"),
            "date": fir.get("date", "N/A"),
            "complainant": fir.get("complainant_name", "N/A"),
            "accused": fir.get("accused_names", []),
            "victim": fir.get("victim_name", "N/A"),
            "location": fir.get("location", "N/A"),
            "incident": fir.get("incident_description", "N/A"),
        },
        "intent": {
            "primary": intent.get("primary_intent", "N/A"),
            "confidence": intent.get("confidence", 0),
            "secondary": intent.get("secondary_intents", []),
        },
        "severity": reasoning.get("severity_assessment", "unknown"),
        "legal_basis": reasoning.get("legal_basis", ""),
        "statutes": statutes,
        "mapped_sections": mapped_sections,
        "chunks_retrieved": analysis.get("retrieved_data", {}).get("total_chunks_retrieved", 0),
        "chunks_after_filtering": analysis.get("retrieved_data", {}).get("chunks_after_filtering", 0),
        "confidence": analysis.get("confidence", 0),
    }


# ---------------------------------------------------------------------------
#  Run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
