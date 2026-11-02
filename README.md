# Document Query Tool

A document intelligence and question-answering system built with Retrieval-Augmented Generation (RAG). Upload PDF documents and ask natural-language questions across all of them. Answers are generated using only the content of your documents, and every answer shows which document and which page it came from. Table and key-detail requests are returned as clean tables.

The language model can run locally (Ollama) or on Groq, with an automatic fallback to the local model when Groq is unavailable.

This project was built to explore document intelligence, information retrieval, RAG, and AI agent architectures (tool calling and MCP) end-to-end.

---

## What it does

**Documents**
- **Upload** PDF documents (text-based, table-heavy or scanned) with the **+** button, or by dragging them into the page. Up to 10 PDFs at a time
- **Attach and ask in one message**: attach a PDF and type a question about it; the question is answered from the attached files
- **Background reading**: a PDF is saved right away and read in the background, so large files never block the app. A file card in the chat shows *Reading…* and changes to *Ready* (or a clear reason if the PDF could not be read)
- **My documents** panel: open any document, delete it (its stored PDF and all its data are removed), newest first
- **Stored PDFs**: the original PDF is kept, so a source can be opened from the chat at the exact page

**Questions and answers**
- **Ask questions** across all your uploaded documents and get grounded answers with **source chips showing the document and page numbers** (for example `report.pdf · p. 4, 7`). Clicking a chip opens the PDF at that page
- **Extract tables and key details** ("show the full invoice as a table", "items with quantity above 100"), view them as tables, copy them or download as CSV
- **List** your uploaded documents
- **Summarize** one document, or get a short **overview** of several documents
- **"These documents I just uploaded"**: the assistant knows which files were added last, so a request about "these two documents" or "the file I just uploaded" covers only those files
- Refuses to answer (rather than guess) when a question isn't covered by the documents

**Conversation**
- **Chat naturally**: greetings and small talk are handled without unnecessary document lookups
- **Conversation memory** within the saved chat: small details you state ("I am Seema", "my project is RAG") are remembered and used in later replies, and questions about the conversation itself ("what are we talking about?") are answered from the chat
- **Follow-up questions** are understood. "Make it shorter", "in simple words", "as bullet points" reword the last answer, while "make it longer" and "give an example" search the last real question again for a fuller answer or a concrete example
- **Chat history is saved on the server** for each user, so it comes back on any device, with a **Clear chat** button

**Accounts**
- **Sign up and log in** with email and password, or **continue with Google**
- Each user only sees their own documents and chat

**Other**
- **MCP server**: the knowledge base can be used from MCP-compatible apps such as Claude Desktop

---

## How it works

The system follows a RAG pipeline, with an agent layer on top that decides how to handle each message:

```
PDF Upload  (duplicate check by SHA-256, file stored, answered immediately)
   │
   ▼  background thread (one PDF at a time, the others wait their turn)
Extraction  ──►  Text layer (PyMuPDF) + tables (pdfplumber);
                 Tesseract OCR only for pages with no text layer
                 (picture-only pages are read several at a time)
   │
   ▼
Chunking (LangChain, heading-aware)  ──►  Overlapping segments, split at section headings,
   │                                      each chunk remembers its page number
   ▼
Embedding (multilingual-e5-base)  ──►  Each chunk converted into a vector
   │
   ▼
Storage (PostgreSQL + pgvector)  ──►  Chunks, vectors and pages stored per user
   │
   ▼
─────────────── on each question ───────────────
   │
   ▼
Agent (intent routing)  ──►  LIST / SUMMARY / OVERVIEW / TABLE / CHAT / SEARCH
   │                          (follow-ups and conversation memory are handled first)
   ▼
SEARCH: agent loop  ──►  The model calls a search tool (up to 4 times)
   │                      Hybrid retrieval: vector search + keyword search, merged
   ▼
Answer (Groq, or Ollama as fallback)  ──►  Grounded in the retrieved passages, with sources and pages
```

### The Agent layer

Every message is classified into one of six actions before anything else happens. Simple rules handle the clear cases, and the language model only decides between chat and search.

| Action | Trigger | Behavior |
|---|---|---|
| `SEARCH` | A specific question about the documents | Agent loop with hybrid retrieval, then a grounded answer with sources and pages |
| `TABLE` | "Show ... as a table", "items where ...", "list the rows" | Extracts a table or key-value pairs in a fixed JSON structure, exactly as written |
| `SUMMARY` | "Summarize the story / this document" | Reads the whole document in order and writes a short summary |
| `OVERVIEW` | "Overview of all documents", "summarize these two documents" | One or two sentences about each document |
| `LIST` | "What documents do you have?" | Returns the list of uploaded documents |
| `CHAT` | Greetings, small talk, facts about you, "what are we talking about?" | Responds conversationally without touching the documents |

**Agentic search.** In `SEARCH` the model is given a `search_documents` tool. The first search is always forced, so it can never answer from general knowledge. It sees which documents contain relevant passages and can search again inside each one, so a question touching several documents doesn't lose one. If the passages don't state the answer, the model says it couldn't find it.

**Hybrid retrieval.** Meaning-based (vector) search and keyword (full-text) search are merged by rank, so both paraphrased questions and exact words, numbers and IDs are found.

**Follow-ups and memory.** Short messages made only of follow-up words ("make it shorter", "give me an example then") are recognised and handled before routing. The assistant keeps simple notes of what you told it (rule-based, no extra model call) and the last few turns of the chat, which are sent along with each question.

**Model fallback.** Groq (`openai/gpt-oss-120b`) is used when `LLM_PROVIDER=groq`. If Groq hits a rate limit or errors, search requests fall back to the local Ollama model. Table requests don't fall back (the small model isn't reliable for tables) and show a short "usage limit reached" message.

**Background reading.** `POST /upload` saves the file and returns at once. A background thread reads it (one PDF at a time, the others queue). The page checks the status until the file is ready or has failed. The jobs are kept in memory, so a file that is still being read when the backend restarts has to be uploaded again.

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
| Authentication | JWT (python-jose), bcrypt (passlib), Google sign-in (google-auth) |
| Integration | MCP server (MCP Python SDK) |
| Frontend | Next.js (TypeScript, App Router, Tailwind CSS), `@react-oauth/google` |

---

## Project structure

```
rag-assistant-tool/
├── backend/
│   ├── main.py          # FastAPI app and endpoints, background upload jobs
│   ├── pipeline.py      # Core pipeline: extraction, chunking, embedding, retrieval,
│   │                    # agent routing, agent search loop, follow-ups, tables, summaries
│   ├── llm.py           # Groq / Ollama switch with automatic fallback
│   ├── auth.py          # Password hashing and JWT helpers
│   ├── mcp_server.py    # MCP server for the knowledge base
│   ├── requirements.txt
│   ├── uploads/         # Stored PDFs, one folder per user (created automatically, not in git)
│   └── .env             # Database password, model settings, API keys (not in git)
└── frontend/
    ├── app/page.tsx                   # Chat interface, tables, file cards, source chips, documents panel
    ├── app/globals.css                # Colours and small animations
    └── app/components/AuthModal.tsx   # Login, signup and Google sign-in window
```

### API endpoints

| Endpoint | Purpose |
|---|---|
| `POST /signup`, `POST /login` | Create an account / get a token |
| `POST /auth/google` | Sign in with a Google account (creates the account the first time) |
| `POST /upload` | Save a PDF and start reading it in the background (requires login) |
| `GET /upload/status` | How the background reading is going (one entry per file) |
| `POST /query` | Ask a question |
| `GET /documents` | List the signed-in user's documents, newest first |
| `DELETE /documents?name=` | Delete a document and its stored PDF |
| `GET /documents/file?name=` | Open the stored PDF |
| `GET /chat`, `PUT /chat`, `DELETE /chat` | Load, save and clear the saved chat |

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
- A Google OAuth client ID (optional; only needed for "Continue with Google", see below)

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
GOOGLE_CLIENT_ID=your_google_oauth_client_id.apps.googleusercontent.com
OCR_WORKERS=4                # optional: how many picture-only pages are read at the same time
```
Login tokens are valid for 7 days. `OCR_WORKERS` defaults to the number of CPU cores, up to 4.

Set up the database:
```sql
CREATE DATABASE document_query_db;
\c document_query_db
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT              -- empty for accounts created with Google
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

CREATE TABLE chat_history (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    messages JSONB NOT NULL DEFAULT '[]',
    updated_at TIMESTAMP DEFAULT NOW()
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

### Google sign-in (optional)

1. In the [Google Cloud console](https://console.cloud.google.com/apis/credentials), create an **OAuth client ID** of type *Web application*
2. Add `http://localhost:3000` under *Authorized JavaScript origins*
3. Put the client ID in `backend/.env` as `GOOGLE_CLIENT_ID`, and in `frontend/.env.local` as `NEXT_PUBLIC_GOOGLE_CLIENT_ID` (the same value; the public client ID is not a secret)

The backend checks the Google token with `google-auth`. An account created with Google has no password, so it signs in with Google only.

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

1. Open the app, sign up or log in (or continue with Google)
2. Click **+** to attach one or more PDFs. Send them with a question, or with no question and wait for the file card to show **Ready**
3. Ask questions in the chat box. Try things like:
   - A specific question ("What is the due date and payment term?")
   - A table request ("Show all items with quantity above 100")
   - "Summarize the story" or "Summarize these two documents I just uploaded"
   - "Which document talks about glaciers?"
   - "What documents do you have?"
   - A follow-up: "make it shorter", "give me an example", "as bullet points"
   - A casual message ("Hi, I am Seema"), and later "who am I?"
4. Click a source chip under an answer to open the PDF at that page. Use **My documents** to open or delete a document, and the account menu to clear the chat or log out

---

## Known limitations

- Scanned (image-only) PDFs depend on OCR, which can misread digits. Text-based PDFs are read exactly. Picture-only pages in a very large PDF take longer to read
- Very long tables can lose rows, and rows split across a page break may be read as two rows
- Groq's free tier has limits (about 8,000 tokens per minute and 200,000 per day). When reached, search answers come from the smaller local model and table requests are unavailable until the limit resets
- Conversation memory is rule-based and keeps simple facts the user states, not everything said
- Background reading jobs are kept in memory: if the backend restarts while a file is being read, the file has to be uploaded again
- Counting questions ("how many times does X appear?") depend on the passages found and may be incomplete
- The MCP server supports one user token and searches directly, without the agent loop
