# LexIR — AI-Powered FIR Analysis & Legal Precedent System

An end-to-end legal intelligence application that analyses First Information Reports (FIRs), identifies applicable IPC & BNS sections using RAG, retrieves real court precedents from Indian Kanoon, and predicts likely verdicts — all through a React + FastAPI WebSocket interface.

---

## Features

| Stage | What it does |
|-------|-------------|
| **Stage 1 — FIR Analysis** | RAG chain (Pinecone + Groq LLM) classifies the FIR into applicable IPC sections and maps them to corresponding BNS sections |
| **Stage 2 — Precedent Search & Verdict Prediction** | Searches Indian Kanoon API for real case law, summarises each judgment with Groq LLM, and predicts the likely verdict/punishment |
| **Stage 3 — Legal Q&A** | Ask follow-up questions about the FIR — answered by Groq LLM using the analysis context |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, Lucide icons |
| Backend | FastAPI, WebSocket, Python 3.10+ |
| LLM | Groq (`llama-3.1-8b-instant`, `openai/gpt-oss-120b`) |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector DB | Pinecone (serverless) |
| Case Law | Indian Kanoon API |
| Utilities | LangChain, python-dotenv, requests |

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10 or higher |
| Node.js | 18 or higher (with npm) |
| Git | any recent version |
| API Keys | Groq, Pinecone, Indian Kanoon (see below) |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/inferio-2004/capstone-fir-analysis.git
cd capstone-fir-analysis
```

### 2. Create the `.env` file

Create a file named `.env` **in the project root** (same level as this README):

```env
# Groq LLM  —  https://console.groq.com/keys  (free tier)
GROQ_API_KEY=your_groq_api_key_here

# Pinecone Vector DB  —  https://www.pinecone.io/  (free tier)
PINECONE_API_KEY=your_pinecone_api_key_here

# Indian Kanoon  —  https://api.indiankanoon.org/  (500 free calls/day)
KANOON_API_KEY=your_kanoon_api_key_here
```

> **How to get the keys:**
>
> | Service | Steps |
> |---------|-------|
> | **Groq** | Sign up at https://console.groq.com → API Keys → Create API Key |
> | **Pinecone** | Sign up at https://www.pinecone.io → Create a project → API Keys |
> | **Indian Kanoon** | Register at https://api.indiankanoon.org → get your token |

### 3. Set up the backend

```bash
# Create & activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install Python dependencies
pip install -r backend/requirements.txt
```

### 4. Set up the frontend

```bash
cd frontend
npm install
cd ..
```

> `npm install` reads `frontend/package.json` and installs all dependencies automatically.  
> If you prefer to install them manually:
> ```bash
> cd frontend
> npm install react react-dom react-scripts lucide-react web-vitals @testing-library/react @testing-library/jest-dom @testing-library/dom @testing-library/user-event
> ```

### 5. Start the servers

Open **two terminals** from the project root:

**Terminal 1 — Backend (port 8000):**
```bash
cd backend
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend (port 3000):**
```bash
cd frontend
npm start
```

### 6. Open the app

Navigate to **http://localhost:3000** in your browser.  
The app connects to the backend via WebSocket at `ws://localhost:8000/ws`.

---

## Model Benchmarking (Groq)

Run the automated summarization benchmark (extractive + abstractive metrics) and auto-select the best Groq model:

```bash
python backend/evaluation/benchmark_groq_summarization.py
```

This writes `output/model_benchmark_latest.json` with:
- ranked comparison of 3 available Groq models,
- time + estimated cost comparison,
- extractive metrics: ROUGE, BLEU, METEOR,
- abstractive metrics: faithfulness + hallucination,
- `best_model` used automatically by Stage 2/Stage 3 modules unless `GROQ_MODEL` is set.

---

## Usage

1. **Submit a FIR** — Paste FIR text into the form or click "Use Sample FIR" to load a pre-defined example.
2. **Stage 1** — The system identifies applicable IPC sections and their BNS equivalents.
3. **Stage 2** — Real court cases matching those sections are fetched from Indian Kanoon, summarised, and a verdict prediction is shown.
4. **Ask Questions** — Type any follow-up question in the chat input (e.g., "What is the maximum punishment under IPC 323?").

---

## Project Structure

```
capstone-fir-analysis/
├── .env                                # API keys (not committed)
├── README.md
│
├── backend/
│   ├── requirements.txt                # Python dependencies
│   ├── server.py                       # FastAPI + WebSocket server
│   └── api/
│       ├── indian_kanoon.py            # Indian Kanoon search, summarise, verdict
│       ├── rag_llm_chain_prompting.py  # Stage 1 RAG chain (Pinecone + Groq)
│       ├── precedent_qa.py             # Groq LLM wrapper for Q&A
│       ├── ocr_to_fir.py              # OCR pipeline (optional)
│       ├── case_similarity.py          # Case-similarity utilities
│       ├── legalqa_index.py            # Embedding-based QA (legacy)
│       └── retrieval_query_constructor.py
│
├── frontend/
│   ├── package.json                    # Node dependencies
│   └── src/
│       ├── App.js / App.css            # Root component & styles
│       ├── hooks/
│       │   └── useLexIR.js             # WebSocket hook
│       └── components/
│           ├── ChatArea.js             # Main chat + stage cards
│           ├── ChatInput.js            # Message input bar
│           ├── FIRForm.js              # FIR submission form
│           ├── Sidebar.js              # Connection status sidebar
│           ├── Stage1Card.js           # IPC/BNS analysis display
│           └── Stage2Card.js           # Kanoon cases + verdict display
│
├── preprocessing/                      # One-time data prep scripts
│   ├── deploy_to_pinecone.py           # Upload statute vectors to Pinecone
│   ├── rag_llm_integration.py          # RAG integration testing
│   └── test_vector_db.py              # Verify Pinecone connection
│
├── output/                             # Generated data & caches
│   ├── ipc_bns_mappings_complete.json
│   ├── fir_analysis_result_chains.json
│   ├── kanoon_cache/                   # Cached Kanoon API responses
│   └── ...
│
├── src_dataset_files/                  # Source datasets
│   ├── fir_sample.json
│   ├── ipc_bns_mapping_starter.json
│   └── IndicLegalQA Dataset_10K.json
│
└── logs/
    └── audit_log.jsonl
```

---

## Backend Dependencies (`backend/requirements.txt`)

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework + WebSocket support |
| `uvicorn[standard]` | ASGI server |
| `websockets` | WebSocket protocol |
| `python-multipart` | File upload handling |
| `groq` | Groq LLM client |
| `langchain-groq` | LangChain ↔ Groq bridge |
| `langchain-core` | Prompt templates, output parsers |
| `sentence-transformers` | MiniLM-L6 embedding model |
| `pinecone-client` | Pinecone vector DB SDK |
| `numpy` | Array operations |
| `requests` | HTTP client (Indian Kanoon API) |
| `python-dotenv` | `.env` file loader |
| `pydantic` | Data validation |
| `PyPDF2` | PDF parsing (preprocessing) |

### Optional (OCR support)

```bash
pip install pytesseract Pillow pdf2image
```
Also requires [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed on the system.

---

## Frontend Dependencies (`frontend/package.json`)

| Package | Purpose |
|---------|---------|
| `react` / `react-dom` | UI framework (v19) |
| `react-scripts` | Create React App tooling |
| `lucide-react` | Icon library |
| `web-vitals` | Performance monitoring |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key for LLM calls |
| `PINECONE_API_KEY` | Yes | Pinecone API key for vector search |
| `KANOON_API_KEY` | Yes | Indian Kanoon API token |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `GROQ_API_KEY not found` | Ensure `.env` exists in the project root with the correct key |
| `Pinecone connection failed` | Check `PINECONE_API_KEY` and that the `statute-embeddings` index exists |
| `KANOON_API_KEY is not set` | Add the Indian Kanoon token to `.env` |
| WebSocket won't connect | Make sure the backend is running on port 8000; check CORS |
| `ModuleNotFoundError` | Activate the virtual environment and run `pip install -r backend/requirements.txt` |
| Frontend build errors | Run `npm install` inside `frontend/` |
| Kanoon returns no results | You may have hit the 500-call daily limit; wait until the next day |

---

## API Endpoints (Backend)

| Endpoint | Type | Description |
|----------|------|-------------|
| `GET /` | HTTP | Health check (`{ "status": "ok", ... }`) |
| `/ws` | WebSocket | Main communication channel for all stages |
| `POST /api/fir/pdf-payload` | HTTP | Convert FIR JSON into key-value payload for PDF form-fill APIs |

### WebSocket Message Types (client → server)

| `type` | Payload | Action |
|--------|---------|--------|
| `start_analysis` | `{ fir }` | Run Stage 1 (RAG) + Stage 2 (Kanoon + verdict) |
| `full_analysis` | `{ fir }` | Same as above (alias) |
| `ask_question` | `{ question }` | Stage 3 follow-up Q&A |

---

## License

This project is part of a capstone initiative. Contact the team for licensing details.

---

**Last Updated:** February 23, 2026
