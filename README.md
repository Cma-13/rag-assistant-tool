# Document Query Tool

A document intelligence and question-answering system built with Retrieval-Augmented Generation (RAG). Upload a PDF document and ask natural-language questions about it, answers are generated using only the content of the document, with a local, self-hosted language model (no external API calls, no data leaving your machine).

This project was built to explore document intelligence, information retrieval, and RAG-based AI agent architectures end-to-end.

---

## What it does

- **Upload** a PDF document through a chat interface
- **Ask questions** about it in plain English and get grounded, accurate answers
- **List** previously uploaded documents
- **Summarize** an entire document on request
- **Chat naturally** greetings and small talk are handled without unnecessary document lookups
- Refuses to answer (rather than guess) when a question isn't covered by the uploaded document

---

## How it works

The system follows a standard RAG pipeline, with an agent layer on top that decides how to handle each message:

```
PDF Upload
   │
   ▼
OCR (Tesseract + PyMuPDF)  ──►  Raw text extracted from the document
   │
   ▼
Chunking (LangChain text splitter)  ──►  Text split into overlapping segments
   │
   ▼
Embedding (multilingual-e5-base)  ──►  Each chunk converted into a vector
   │
   ▼
Storage (PostgreSQL + pgvector)  ──►  Chunks and vectors stored for retrieval
   │
   ▼
─────────────── on each question ───────────────
   │
   ▼
Agent (intent routing)  ──►  Decides: SEARCH / LIST / SUMMARY / CHAT
   │
   ▼
Retrieval (vector similarity search, scoped to the active document)
   │
   ▼
Relevance check  ──►  Confirms retrieved content actually answers the question
   │
   ▼
Generation (local LLM via Ollama)  ──►  Final answer, grounded in retrieved context
```

### The Agent layer

Rather than always running the same fixed retrieve-then-generate flow, incoming messages are classified into one of four actions before anything else happens:

| Action | Trigger | Behavior |
|---|---|---|
| `SEARCH` | A specific question about the document | Retrieves relevant chunks, verifies relevance, generates a grounded answer |
| `LIST` | "What documents do you have?" | Returns the list of uploaded documents (rule-based detection) |
| `SUMMARY` | "Tell me about this document" / "Summarize this" | Generates a broad overview from chunks spread across the whole document |
| `CHAT` | Greetings, small talk | Responds conversationally without touching the document store |

To reduce hallucination, `SEARCH` results go through a relevance check before an answer is generated: if the retrieved content doesn't actually address the question (either by retrieval-distance confidence or an LLM-based check for borderline cases), the system responds with "I don't know" instead of guessing.

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend API | Python, FastAPI |
| PDF parsing | PyMuPDF, Tesseract OCR (pytesseract) |
| Text chunking | LangChain text splitters |
| Embeddings | `intfloat/multilingual-e5-base` (Sentence Transformers) |
| Vector storage | PostgreSQL + pgvector |
| Language model | Ollama, running `llama3.2:3b` locally |
| Frontend | Next.js (TypeScript, App Router, Tailwind CSS) |

---

## Project structure

```
multilingual-rag-assistant/
├── backend/
│   ├── main.py          # FastAPI app and endpoints (/upload, /query)
│   ├── pipeline.py       # Core RAG pipeline: OCR, chunking, embedding, retrieval,
│   │                      generation, and the agent's decision logic
│   ├── requirements.txt
│   └── .env              # Database credentials 
└── frontend/
    └── src/app/page.tsx  # Chat interface
```

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL 15+ with the [pgvector](https://github.com/pgvector/pgvector) extension installed
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and on your system PATH
- [Ollama](https://ollama.com) installed, with the `llama3.2:3b` model pulled:
  ```bash
  ollama pull llama3.2:3b
  ```

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file in `backend/`:
```
DB_PASSWORD=your_postgres_password
```

Set up the database:
```sql
CREATE DATABASE document_query_db;
\c document_query_db
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    source_file TEXT NOT NULL,
    page_number INTEGER,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(768),
    created_at TIMESTAMP DEFAULT NOW()
);
```

Run the API:
```bash
uvicorn main:app --reload
```
The API will be available at `http://127.0.0.1:8000`, with interactive docs at `/docs`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000`.

---

## Usage

1. Open the app and upload a PDF using the upload button
2. Once processing finishes, ask questions about the document in the chat box
3. Try things like:
   - A specific factual question ("When was X founded?")
   - "Tell me about this document" (triggers a summary)
   - "What documents do you have?" (lists uploads)
   - A casual greeting ("Hi, how are you?")

