# Document Query Tool

A document intelligence and question-answering system built with Retrieval-Augmented Generation (RAG). Upload PDF documents and ask natural-language questions across all of them. Answers are generated using only the content of your documents, and every answer shows which document it came from. Table and key-detail requests are returned as clean tables.

The language model can run locally (Ollama) or on Groq, with an automatic fallback to the local model when Groq is unavailable.

This project was built to explore document intelligence, information retrieval, RAG, and AI agent architectures (tool calling and MCP) end-to-end.

---

## What it does

- **Upload** PDF documents (text-based, table-heavy or scanned) through a chat interface
- **Ask questions** across all your uploaded documents and get grounded answers with the source document shown
- **Extract tables and key details** ("show the full invoice as a table", "items with quantity above 100"), view them as tables, copy them or download as CSV
- **List** your uploaded documents
- **Summarize** one document, or get a short **overview of all documents**
- **Chat naturally**: greetings and small talk are handled without unnecessary document lookups
- **Follow-up questions** such as "explain that more simply" are understood from the last two turns
- Refuses to answer (rather than guess) when a question isn't covered by the documents
- **Accounts**: sign up and log in; each user only sees their own documents
- **Chat history** is kept in the browser for each user, with a Clear chat button
- **MCP server**: the knowledge base can be used from MCP-compatible apps such as Claude Desktop

---

## How it works

The system follows a RAG pipeline, with an agent layer on top that decides how to handle each message:

```
PDF Upload  (duplicate check by SHA-256)
   │
   ▼
Extraction  ──►  Text layer (PyMuPDF) + tables (pdfplumber);
                 Tesseract OCR only for pages with no text layer
   │
   ▼
Chunking (LangChain, heading-aware)  ──►  Overlapping segments, split at section headings
   │
   ▼
Embedding (multilingual-e5-base)  ──►  Each chunk converted into a vector
   │
   ▼
Storage (PostgreSQL + pgvector)  ──►  Chunks and vectors stored per user
   │
   ▼
─────────────── on each question ───────────────
   │
   ▼
Agent (intent routing)  ──►  LIST / SUMMARY / OVERVIEW / TABLE / CHAT / SEARCH
   │
   ▼
SEARCH: agent loop  ──►  The model calls a search tool (up to 4 times)
   │                      Hybrid retrieval: vector search + keyword search, merged
   ▼
Answer (Groq, or Ollama as fallback)  ──►  Grounded in the retrieved passages, with sources
```

### The Agent layer

Every message is classified into one of six actions before anything else happens. Simple rules handle the clear cases, and the language model only decides between chat and search.

| Action | Trigger | Behavior |
|---|---|---|
| `SEARCH` | A specific question about the documents | Agent loop with hybrid retrieval, then a grounded answer with sources |
| `TABLE` | "Show ... as a table", "items where ...", "list the rows" | Extracts a table or key-value pairs in a fixed JSON structure, exactly as written |
| `SUMMARY` | "Summarize the story / this document" | Reads the whole document in order and writes a short summary |
| `OVERVIEW` | "Overview of all documents" | One or two sentences about each uploaded document |
| `LIST` | "What documents do you have?" | Returns the list of uploaded documents |
| `CHAT` | Greetings, small talk, "what can you do?" | Responds conversationally without touching the documents |

**Agentic search.** In `SEARCH` the model is given a `search_documents` tool. The first search is always forced, so it can never answer from general knowledge. It sees which documents contain relevant passages and can search again inside each one, so a question touching several documents doesn't lose one. If the passages don't state the answer, the model says it couldn't find it.

**Hybrid retrieval.** Meaning-based (vector) search and keyword (full-text) search are merged by rank, so both paraphrased questions and exact words, numbers and IDs are found.

**Model fallback.** Groq (`openai/gpt-oss-120b`) is used when `LLM_PROVIDER=groq`. If Groq hits a rate limit or errors, search requests fall back to the local Ollama model. Table requests don't fall back (the small model isn't reliable for tables) and show a short "usage limit reached" message.

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend API | Python, FastAPI |
| PDF parsing | PyMuPDF, pdfplumber, Tesseract OCR (pytesseract) |
| Text chunking | LangChain text splitters |
| Embeddings | `intfloat/multilingual-e5-base` (Sentence Transformers) |
| Storage and search | PostgreSQL + pgvector, PostgreSQL full-text search |
| Language models | Groq `openai/gpt-oss-120b` and Ollama `llama3.2:3b` |
| Authentication | JWT (python-jose), bcrypt (passlib) |
| Integration | MCP server (MCP Python SDK) |
| Frontend | Next.js (TypeScript, App Router, Tailwind CSS) |

---

## Project structure

```
rag-assistant-tool/
├── backend/
│   ├── main.py          # FastAPI app and endpoints
│   ├── pipeline.py      # Core pipeline: extraction, chunking, embedding, retrieval,
│   │                    # agent routing, agent search loop, tables, summaries
│   ├── llm.py           # Groq / Ollama switch with automatic fallback
│   ├── auth.py          # Password hashing and JWT helpers
│   ├── mcp_server.py    # MCP server for the knowledge base
│   ├── requirements.txt
│   └── .env             # Database password, model settings, API keys
└── frontend/
    ├── app/page.tsx                   # Chat interface, tables, source chips, chat history
    └── app/components/AuthModal.tsx   # Login and signup window
```

### API endpoints

| Endpoint | Purpose |
|---|---|
| `POST /signup`, `POST /login` | Create an account / get a token |
| `POST /upload` | Upload a PDF (requires login) |
| `POST /query` | Ask a question |
| `GET /documents` | List the signed-in user's documents |

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL 15+ with the [pgvector](https://github.com/pgvector/pgvector) extension installed
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and on your system PATH
- [Ollama](https://ollama.com) installed, with the `llama3.2:3b` model pulled (used locally and as the fallback):
  ```bash
  ollama pull llama3.2:3b
  ```
- A [Groq](https://console.groq.com) API key (optional; without it, use `LLM_PROVIDER=ollama`)

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
LLM_PROVIDER=groq            # or ollama
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
JWT_SECRET_KEY=a_long_random_secret_string
```
Login tokens are valid for 7 days.

Set up the database:
```sql
CREATE DATABASE document_query_db;
\c document_query_db
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL
);

CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    source_file TEXT NOT NULL,
    page_number INTEGER,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(768),
    content_hash TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_chunks_fts ON document_chunks
    USING GIN (to_tsvector('english', chunk_text));
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

### MCP server (optional)

The MCP server lets an MCP client (for example Claude Desktop) use your knowledge base. Add it to the client's config, with a login token for your account:

```json
{
  "mcpServers": {
    "docquery": {
      "command": "path/to/venv/Scripts/python.exe",
      "args": ["path/to/backend/mcp_server.py"],
      "env": { "DOCQUERY_ACCESS_TOKEN": "your_jwt_token" }
    }
  }
}
```
It provides three tools: `list_my_documents`, `search_documents` and `summarize_document`.

---

## Usage

1. Open the app, sign up or log in, and upload one or more PDFs
2. Once processing finishes, ask questions in the chat box
3. Try things like:
   - A specific question ("What is the due date and payment term?")
   - A table request ("Show all items with quantity above 100")
   - "Summarize the story" or "Give me a short overview of all these documents"
   - "Which document talks about glaciers?"
   - "What documents do you have?"
   - A casual greeting ("Hi, how are you?")

---

## Known limitations

- Scanned (image-only) PDFs depend on OCR, which can misread digits. Text-based PDFs are read exactly.
- Very long tables can lose rows, and rows split across a page break may be read as two rows.
- Groq's free tier has limits (about 8,000 tokens per minute and 200,000 per day). When reached, search answers come from the smaller local model and table requests are unavailable until the limit resets.
- Conversation memory is limited to the last two turns.
- Chat history is stored in the browser, not on the server.
- Sources are shown by document, not by page.
- The MCP server supports one user token and searches directly, without the agent loop.
