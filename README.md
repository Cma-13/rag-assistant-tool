# DocQuery

DocQuery is a document question-answering assistant built on Retrieval-Augmented Generation (RAG). Upload PDF documents and ask questions in plain language: answers are generated only from your documents, and every answer links to the document and page it came from.

An AI agent decides how to handle each message and calls its tools through MCP (Model Context Protocol) servers, one for document search and one for an optional read-only Gmail connection.

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Known Limitations](#known-limitations)

## Features

**Documents**
- Upload text-based, scanned or table-heavy PDFs (up to 10 at a time, 50 MB each by default)
- Background processing with live status, and duplicate detection by file content
- Scanned pages read with Tesseract OCR, including Nepali (Devanagari) and legacy Preeti-font PDFs
- Document panel to open or delete files

**Answers**
- Grounded answers with clickable sources showing the document and page numbers
- Hybrid retrieval combining vector search and keyword search
- Table and key-value extraction with copy and CSV download
- Document summaries and multi-document overviews
- Questions and answers in English or Nepali
- Answer trace showing the intent, searches, passages and tools used for each answer

**Conversation and accounts**
- Follow-up requests such as "make it shorter" or "give an example"
- Chat history saved per user
- Email/password accounts and Google sign-in

**Gmail (optional)**
- Read-only Gmail connection through Google OAuth
- Search and read emails from the chat
- Import PDF attachments into your documents

## How It Works

```
PDF upload ──► Text extraction (PyMuPDF, pdfplumber, Tesseract OCR)
           ──► Chunking (500 characters, 50 overlap, page-aware)
           ──► Embeddings (multilingual-e5-base)
           ──► PostgreSQL + pgvector (stored per user)

Question  ──► Agent: intent routing
           ──► MCP tools (document search / Gmail)
           ──► Hybrid retrieval (vector + full-text, merged with Reciprocal Rank Fusion)
           ──► Language model (Groq, with Ollama fallback)
           ──► Answer with sources and pages
```

Each message is routed to one of the following actions:

| Action | Purpose |
|---|---|
| `SEARCH` | Answer a question from the documents using the agent search loop |
| `TABLE` | Extract a table or key-value pairs exactly as written |
| `SUMMARY` | Summarize one document |
| `OVERVIEW` | Give a short overview of several documents |
| `LIST` | List the uploaded documents |
| `EMAIL` | Work with Gmail through the Gmail MCP tools |
| `CHAT` | Reply to small talk without searching |

**Agent and MCP.** The agent is an MCP client connected to two MCP servers mounted in the backend over Streamable HTTP:

| Server | Path | Tools |
|---|---|---|
| DocQuery | `/mcp` | `search_documents`, `list_my_documents` |
| Gmail | `/gmail-mcp` | `search_emails`, `read_email`, `import_email_attachment` |

The first search is always required, so answers never come from the model's general knowledge. If the documents do not contain the answer, the assistant says so.

**Model fallback.** With `LLM_PROVIDER=groq`, answers come from Groq. On rate limits or errors, search answers fall back to the local Ollama model.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| PDF processing | PyMuPDF, pdfplumber, Tesseract OCR (pytesseract) |
| Text chunking | LangChain text splitters |
| Embeddings | Sentence Transformers, `intfloat/multilingual-e5-base` |
| Database | PostgreSQL, pgvector, PostgreSQL full-text search |
| Language models | Groq `openai/gpt-oss-120b`, Ollama `llama3.2:3b` |
| Agent tools | MCP Python SDK |
| Authentication | JWT (python-jose), bcrypt (passlib), Google sign-in (google-auth) |
| Gmail | Gmail API, Google OAuth 2.0, encrypted token storage (cryptography) |
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS 4 |
| Deployment | Docker, Docker Compose |

## Project Structure

```
rag-assistant-tool/
├── backend/
│   ├── main.py              # FastAPI app and endpoints; mounts the MCP servers
│   ├── pipeline.py          # Extraction, OCR, chunking, retrieval, agent routing and answers
│   ├── ingest.py            # PDF upload handling and background processing
│   ├── mcp_client.py        # MCP client used by the agent
│   ├── docquery_mcp.py      # DocQuery MCP server (document search)
│   ├── gmail_mcp.py         # Gmail MCP server
│   ├── gmail_account.py     # Gmail OAuth connection and Gmail API calls
│   ├── answer_trace.py      # Records the steps behind each answer
│   ├── llm.py               # Groq / Ollama provider with fallback
│   ├── auth.py              # Password hashing and JWT helpers
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app/
│   │   ├── page.tsx                     # Chat interface
│   │   ├── config.ts                    # Backend URL
│   │   └── components/
│   │       ├── AuthModal.tsx            # Login and sign-up
│   │       └── AnswerTrace.tsx          # Answer trace panel
│   ├── package.json
│   └── Dockerfile
├── db/
│   └── init.sql             # Database schema
├── docker-compose.yml
└── docker-compose.dev.yml   # Development override with live reload
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) on your PATH, with `nep` and `osd` language data for Nepali documents
- [Ollama](https://ollama.com) with the `llama3.2:3b` model
- A [Groq API key](https://console.groq.com) (optional; use `LLM_PROVIDER=ollama` without one)

### Option 1: Docker

1. Create `backend/.env` (see [Configuration](#configuration)).
2. Start the database, backend and frontend:
   ```bash
   docker compose --env-file backend/.env up --build
   ```
3. Open http://localhost:3000

The database is created from `db/init.sql` on first start. Ollama is not included in the containers; the backend uses the Ollama running on your machine.

For development with live reload:
```bash
docker compose --env-file backend/.env -f docker-compose.yml -f docker-compose.dev.yml up --build
```

### Option 2: Local setup

**1. Database**
```bash
psql -U postgres -c "CREATE DATABASE document_query_db;"
psql -U postgres -d document_query_db -f db/init.sql
```

**2. Backend**
```bash
cd backend
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
ollama pull llama3.2:3b
uvicorn main:app --reload
```
The API runs at http://127.0.0.1:8000, with interactive docs at http://127.0.0.1:8000/docs.

**3. Frontend**
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:3000.

## Configuration

### Backend (`backend/.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `DB_PASSWORD` | Yes | | PostgreSQL password |
| `JWT_SECRET_KEY` | Yes | | Secret for signing login tokens |
| `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` | No | `localhost` / `5432` / `document_query_db` / `postgres` | Database connection |
| `LLM_PROVIDER` | No | `ollama` | `groq` or `ollama` |
| `GROQ_API_KEY` | With Groq | | Groq API key |
| `GROQ_MODEL` | No | `openai/gpt-oss-120b` | Groq model |
| `OLLAMA_MODEL` | No | `llama3.2:3b` | Local and fallback model |
| `OCR_WORKERS` | No | CPU cores (max 4) | Pages processed by OCR in parallel |
| `NEPALI_OCR` | No | `1` | Set to `0` to read all scanned pages in English only |
| `MAX_UPLOAD_MB` | No | `50` | Maximum PDF size |
| `CORS_ORIGINS` | No | `http://localhost:3000` | Allowed frontend origins (comma separated) |
| `GOOGLE_CLIENT_ID` | For Google | | Google sign-in and Gmail |
| `GOOGLE_CLIENT_SECRET` | For Gmail | | Gmail connection |
| `GMAIL_REDIRECT_URI` | No | `http://127.0.0.1:8000/gmail/callback` | Gmail OAuth redirect |
| `FRONTEND_URL` | No | `http://localhost:3000` | Frontend address used after Gmail connection |

Example:
```env
DB_PASSWORD=your_postgres_password
JWT_SECRET_KEY=a_long_random_secret
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
```

### Frontend (`frontend/.env.local`)

| Variable | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:8000` | Backend URL |
| `NEXT_PUBLIC_GOOGLE_CLIENT_ID` | | Google OAuth client ID for sign-in |

### Google sign-in and Gmail (optional)

1. In [Google Cloud Console](https://console.cloud.google.com/apis/credentials), create an OAuth client ID of type **Web application**.
2. Add `http://localhost:3000` to **Authorized JavaScript origins**.
3. For Gmail: enable the **Gmail API**, add the `gmail.readonly` scope on the OAuth consent screen, add test users while the app is in testing, and add `http://127.0.0.1:8000/gmail/callback` to **Authorized redirect URIs**.
4. Set `GOOGLE_CLIENT_ID` (and `GOOGLE_CLIENT_SECRET` for Gmail) in `backend/.env`, and `NEXT_PUBLIC_GOOGLE_CLIENT_ID` in `frontend/.env.local`.

## Usage

1. Sign up or log in.
2. Attach PDFs with the **+** button or by dragging them onto the page.
3. Ask a question and press **Enter** (**Shift + Enter** for a new line).
4. Click a source under an answer to open the PDF at that page, or open **How this answer was found** to see the steps.
5. Use **My documents** to open or delete files, and the account menu to clear the chat or log out.

Example requests:
- `What is the total amount on the invoice?`
- `Show the items as a table`
- `Summarize this document`
- `Make it shorter`

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/signup`, `/login` | Create an account or log in |
| `POST` | `/auth/google` | Sign in with Google |
| `POST` | `/upload` | Upload a PDF |
| `GET` | `/upload/status` | Background processing status |
| `POST` | `/query` | Ask a question |
| `GET` | `/documents` | List documents |
| `GET` | `/documents/file` | Open a stored PDF |
| `DELETE` | `/documents` | Delete a document |
| `GET` / `PUT` / `DELETE` | `/chat` | Load, save or clear chat history |
| `GET` | `/gmail/status` | Gmail connection status |
| `POST` | `/gmail/connect` | Start the Gmail connection |
| `DELETE` | `/gmail` | Disconnect Gmail |

Full interactive documentation is available at `/docs` when the backend is running.

## Known Limitations

- OCR accuracy depends on scan quality; digits and some Nepali characters can be misread.
- Very long tables may lose rows, especially across page breaks.
- Groq free-tier usage limits apply; when reached, answers come from the smaller local model.
- While the Google OAuth app is in testing mode, Gmail connections expire after 7 days.
