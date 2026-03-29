# LexIR — AI-Powered FIR Analysis & Legal Precedent System

An end-to-end legal intelligence application that analyses First Information Reports (FIRs), identifies applicable IPC & BNS sections using RAG, retrieves real court precedents from Indian Kanoon, and predicts likely verdicts — all through a React + FastAPI WebSocket interface.

---

## Features

| Stage | What it does |
|-------|-------------|
| **Google Authentication** | Secure login using Google Identity Services with persistent sessions |
| **User Isolation** | Every user has their own private chat history and analysis records |
| **Stage 1 — FIR Analysis** | RAG chain (Pinecone + Groq LLM) classifies the FIR into applicable IPC sections and maps them to corresponding BNS sections |
| **Stage 2 — Precedent Search & Verdict Prediction** | Searches Indian Kanoon API for real case law, summarises each judgment with Groq LLM, and predicts the likely verdict/punishment |
| **Stage 3 — Legal Q&A** | Ask follow-up questions about the FIR — answered by Groq LLM using the analysis context |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, Lucide icons, @react-oauth/google |
| Backend | FastAPI, WebSocket, Python 3.10+ |
| Database | MongoDB (User storage & Chat history) |
| LLM | Groq (`llama-3.1-8b-instant`) |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector DB | Pinecone (serverless) |
| Case Law | Indian Kanoon API |
| Utilities | LangChain, python-dotenv, requests, motor |

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10 or higher |
| Node.js | 18 or higher (with npm) |
| MongoDB | Running on `localhost:27017` (default) |
| Git | any recent version |
| API Keys | Google Client ID, Groq, Pinecone, Indian Kanoon (see below) |

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
# Google Auth
REACT_APP_GOOGLE_CLIENT_ID=your_google_client_id_here.apps.googleusercontent.com

# Groq LLM  —  https://console.groq.com/keys  (free tier)
GROQ_API_KEY=your_groq_api_key_here

# Pinecone Vector DB  —  https://www.pinecone.io/  (free tier)
PINECONE_API_KEY=your_pinecone_api_key_here

# Indian Kanoon  —  https://api.indiankanoon.org/  (500 free calls/day)
KANOON_API_KEY=your_kanoon_api_key_here
```

> **Note on Google Client ID:** Ensure your local URL (e.g., `http://localhost:3000`) is added to the **Authorized JavaScript origins** in your Google Cloud Console.

### 3. Set up the backend

```bash
# Create & activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install Python dependencies
pip install -r backend/requirements.txt
pip install google-auth motor pyjwt
```

### 4. Set up the frontend

```bash
cd frontend
npm install
cd ..
```

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

---

## Usage

1. **Login** — Sign in with your Google account. Your name and profile picture will appear in the sidebar.
2. **Submit a FIR** — Paste FIR text or click "Use Sample FIR".
3. **Stage 1** — The system identifies applicable IPC sections and their BNS equivalents.
4. **Stage 2** — Real court cases matching those sections are fetched from Indian Kanoon.
5. **Ask Questions** — Type any follow-up question in the chat input.
6. **Persistence** — Your analyses are saved to your account. You can rename or delete them from the Sidebar history.

---

## Project Structure

```
capstone-fir-analysis/
├── .env                                # API keys (Centralized)
├── README.md
│
├── backend/
│   ├── server.py                       # FastAPI server with Auth & Mongo integration
│   └── api/
│       ├── ws_handlers.py              # WebSocket message logic (Multi-user aware)
│       └── ...
│
├── frontend/
│   ├── package.json                    # Uses dotenv-cli to load root .env
│   └── src/
│       ├── App.js                      # Root component with Auth routing
│       ├── hooks/
│       │   └── useLexIR.js             # WebSocket hook with user scoping
│       └── components/
│           ├── LoginPage.js            # Google Auth screen
│           ├── Sidebar.js              # User info & History list
│           └── ...
```

---

## License

This project is part of a capstone initiative. Contact the team for licensing details.

---

**Last Updated:** March 29, 2026
