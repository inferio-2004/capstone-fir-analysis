# FIR Analysis System - AI-Powered Legal Document Processing

An intelligent system that analyzes First Information Reports (FIRs) using NLP and legal text embeddings to automatically classify applicable IPC and BNS sections.

## Features

✅ **OCR-based FIR Digitization** - Extract text from scanned/handwritten FIR documents  
✅ **NLP-Powered Analysis** - Automatic extraction of legal concepts from FIR text  
✅ **Statute Classification** - Automated mapping to applicable IPC and BNS sections  
✅ **Explicit Mappings** - View overlapping IPC ↔ BNS provisions  

---

## Prerequisites

- **Python 3.8+**
- **Tesseract OCR** (for handwritten/scanned FIR processing)
- **Git**
- **API Keys** (Groq, Pinecone) - see setup below

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/inferio-2004/capstone-fir-analysis.git
cd capstone-fir-analysis
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR

**Windows:**
- Download installer: https://github.com/UB-Mannheim/tesseract/wiki
- Run the installer (default: `C:\Program Files\Tesseract-OCR`)
- Add to `.env`: `TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe`

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu):**
```bash
sudo apt-get install tesseract-ocr
```

---

## API Setup

### Get Groq API Key

1. Go to https://console.groq.com/keys
2. Sign up (free account)
3. Click "Create API Key"
4. Copy the key

### Get Pinecone API Key

1. Go to https://www.pinecone.io/
2. Sign up (free tier available)
3. Create a project
4. Navigate to API Keys section
5. Copy your API key and index name

---

## Configuration

### 1. Create `.env` File

Create a `.env` file in the project root:

```
# Groq LLM API
GROQ_API_KEY=your_groq_api_key_here

# Pinecone Vector DB
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=statute-embeddings
PINECONE_ENVIRONMENT=us-east-1

# Optional: Tesseract Path (Windows only, if not in default location)
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

**⚠️ IMPORTANT:** Never commit `.env` file - it's in `.gitignore` by default

### 2. Load Environment Variables

The code uses `python-dotenv` to automatically load `.env` - no additional setup needed.

---

## Running the System

### Test Vector Database Connection

```bash
python preprocessing/test_vector_db.py
```

Expected output:
```
✓ Pinecone connected
✓ Embedding model loaded
✓ Embeddings are searchable
```

### Analyze a FIR

```bash
python backend/api/rag_llm_chain_prompting.py
```

### Extract Text from Scanned FIR

```bash
python backend/api/ocr_to_fir.py /path/to/fir_image.jpg output.json
```

---

## Project Structure

```
capstone-fir-analysis/
├── backend/
│   ├── api/
│   │   ├── ocr_to_fir.py              # OCR pipeline
│   │   ├── rag_llm_chain_prompting.py # Main analysis engine
│   │   └── retrieval_query_constructor.py
│   └── preprocessing/
│       ├── extract_bns_and_comparison.py
│       └── extract_final.py
├── preprocessing/
│   ├── deploy_to_pinecone.py
│   ├── test_vector_db.py
│   └── rag_llm_integration.py
├── src/
│   ├── fir_sample.json               # Sample FIR input
│   ├── bns.pdf                       # BNS statute document
│   └── Section2_RAG_Instructions_for_AI.txt
├── output/                            # Generated outputs
│   ├── bns_ipc_mappings_final.json
│   ├── fir_analysis_result.json
│   └── statute_vectors.jsonl
├── logs/                              # Audit logs
└── .env                               # Local configuration (not committed)
```

---

## Troubleshooting

### Issue: "Groq API key not found"
**Solution:** Ensure `.env` file exists in project root with correct `GROQ_API_KEY`

### Issue: "Tesseract not found"
**Solution:** Install Tesseract OCR (see Installation section)

### Issue: "Pinecone connection failed"
**Solution:** 
- Check internet connection
- Verify `PINECONE_API_KEY` in `.env`
- Ensure index `statute-embeddings` exists in Pinecone console

### Issue: "pytesseract not installed"
**Solution:** 
```bash
pip install pytesseract pillow pdf2image
```

---

## Technology Stack

**Core Technologies:**
- **Neural Embeddings:** SentenceTransformers (all-MiniLM-L6-v2)
- **Vector Database:** Pinecone
- **Chain Prompting:** LangChain + Groq LLMs
- **OCR Pipeline:** Tesseract, pytesseract, PIL

**Backend:**
- Python 3.8+
- LangChain, sentence-transformers
- Groq API, Pinecone SDK

---

## Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Commit changes: `git commit -m "Add feature"`
3. Push to branch: `git push origin feature/your-feature`
4. Create Pull Request

---

## License

This project is part of a capstone initiative. Contact the team for licensing details.

---

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review API documentation:
   - Groq: https://console.groq.com/docs
   - Pinecone: https://docs.pinecone.io/

---

**Last Updated:** January 28, 2026 
