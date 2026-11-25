# DocQuery

**Ask questions about your PDFs and get answers that point to the exact page they came from.**

DocQuery is a document question-answering assistant built on Retrieval-Augmented Generation (RAG). Upload PDF documents (text-based, scanned, table-heavy, in English or Nepali) and ask questions in your own words across all of them. Answers come only from your documents, every answer shows its sources with page numbers, and each answer can show step by step how it was found.

An AI agent decides how to handle each message and reaches its tools through MCP servers: one for document search and one for an optional, read-only Gmail connection. The language model runs on Groq, with an automatic fallback to a local Ollama model.

Built as a final-year internship project at Treeleaf Technologies to explore document intelligence, information retrieval, RAG and AI agent architectures (tool calling and MCP) end to end.

---

## Contents

- [Highlights](#highlights)
- [Features](#features)
- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
  - [Option A: Docker](#option-a-docker)
  - [Option B: Run locally](#option-b-run-locally)
  - [Environment variables](#environment-variables)
  - [Google sign-in (optional)](#google-sign-in-optional)
  - [Gmail (optional)](#gmail-optional)
- [Usage](#usage)
- [API reference](#api-reference)
- [Security and privacy](#security-and-privacy)
- [Known limitations](#known-limitations)

---

## Highlights

- **Grounded answers with page-level sources**: click a source to open the PDF at that page
- **Hybrid search**: meaning-based (vector) and keyword search merged, so paraphrases and exact numbers or IDs are both found
- **Agentic search**: the model can search several times, once per document, before answering, and says so when the documents don't answer the question
- **Nepali support**: scanned Nepali pages are read with Nepali OCR, old Preeti-font PDFs are handled, and questions can be asked and answered in Nepali or English
- **Tables and key details** extracted exactly as written, with copy and CSV download
- **MCP inside the app**: document search and Gmail are MCP servers that the chat's agent connects to
- **Answer trace**: "How this answer was found" shows the intent, the searches, the passages used, the tools called and the time each step took

---

## Features

### Documents
- **Upload** PDFs with the **+** button or by dragging them onto the page, up to 10 at a time (50 MB each by default)
- **Attach and ask in one message**: the question is answered from the attached files only
- **Background reading**: a PDF is saved at once and read in the background; a card in the chat shows *Reading…* and then *Ready*, or a clear reason if the file could not be read
- **Duplicate detection**: the same file (by content) is never stored twice
- **My documents** panel: open or delete any document (its stored PDF and all its data are removed), newest first

### Questions and answers
- **Ask across all documents** and get answers with **source chips** showing the document and pages (for example `report.pdf · p. 4, 7`)
- **Tables and key details** ("show the invoice items as a table", "items with quantity above 100"): shown as tables, with copy and CSV download
- **Summaries** of one document, and a short **overview** of several
- **List** your documents, or ask "which document talks about glaciers?"
- **"The documents I just uploaded"** is understood and covers only those files
- **Honest answers**: when the documents don't contain the answer, it says so instead of guessing

### Nepali documents
- **Scanned Nepali pages** are read with Nepali + English OCR; English pages are not affected
- **Old Preeti-font PDFs**, whose copied text reads as `g]kfn` instead of `नेपाल`, are read with OCR instead
- **Nepali keyword search** handles word endings (`विद्यार्थी` also finds `विद्यार्थीहरूको`)
- **Answers follow the language of the question**, and an English question about a Nepali document (or the other way round) searches in both languages

### Conversation
- **Small talk** is answered without searching the documents
- **Follow-ups**: "make it shorter", "in simple words", "as bullet points" reword the last answer; "make it longer" and "give an example" search again for more
- **Conversation memory** within the chat: details you state ("I am Seema") are remembered
- **Saved chat history** per user, available on any device, with **Clear chat** (asks before deleting)

### Accounts
- Sign up and log in with email and password, or **continue with Google**
- Each user only sees their own documents, chat and Gmail

### Gmail (optional)
- **Connect Gmail** (read-only) from the header, then ask: "emails from Ashok this week", "read the latest invoice email"
- Emails are shown as **email cards** with dates in your own time zone
- **Import a PDF attachment** into your documents and ask questions about it like any uploaded file

---

## How it works

DocQuery follows a RAG pipeline, with an agent layer that decides how to handle each message.

```
PDF upload  (duplicate check by SHA-256, file stored, answered immediately)
   │
   ▼  background thread (one PDF at a time, the others wait their turn)
Extraction  ──►  Text layer (PyMuPDF) + tables (pdfplumber)
                 Tesseract OCR for pages with no readable text layer
                 (several pages at a time; Devanagari pages read as Nepali + English)
   │
   ▼
Chunking (LangChain, heading-aware)  ──►  Overlapping segments, each remembers its page
   │
   ▼
Embedding (multilingual-e5-base)  ──►  Each chunk becomes a 768-number vector
   │
   ▼
Storage (PostgreSQL + pgvector)  ──►  Chunks, vectors and pages, stored per user
   │
   ▼
──────────────────── on each question ────────────────────
   │
   ▼
Agent (intent routing)  ──►  SEARCH / TABLE / SUMMARY / OVERVIEW / LIST / EMAIL / CHAT
   │                          (follow-ups and conversation memory are handled first)
   ▼
MCP tools  ──►  search_documents (DocQuery server) · Gmail tools (Gmail server)
   │             Hybrid retrieval: vector + keyword search, merged by rank (RRF)
   ▼
Answer (Groq, Ollama as fallback)  ──►  Grounded in the passages, with sources and pages
```

### Intent routing

Every message is classified before anything else happens. Simple rules handle the clear cases, and the language model only decides between chat and search.

| Action | Example | What happens |
|---|---|---|
| `SEARCH` | "What is the due date?" | Agent loop with hybrid retrieval, then a grounded answer with sources |
| `TABLE` | "Show the items as a table" | Extracts a table or key-value pairs exactly as written |
| `SUMMARY` | "Summarize this document" | Reads the whole document in order and writes a short summary |
| `OVERVIEW` | "Summarize these two documents" | One or two sentences about each document |
| `LIST` | "What documents do you have?" | Lists the uploaded documents |
| `EMAIL` | "Emails from Ashok this week" | Agent loop with the Gmail MCP tools |
| `CHAT` | "Hi, how are you?" | A conversational reply, no document search |

### MCP tools

The agent is an MCP client (`mcp_client.py`). It connects to two MCP servers mounted in the backend over Streamable HTTP, asks each for its tools and gives them to the model:

| Server | Address | Tools |
|---|---|---|
| DocQuery | `/mcp` | `search_documents`, `list_my_documents` |
| Gmail | `/gmail-mcp` | `search_emails`, `read_email`, `import_email_attachment` |

Only the tools offered for that request can run, and every call carries the user's login token, so a tool only sees that user's documents and Gmail.

### Agentic search and hybrid retrieval

The first search is always forced, so the model never answers from general knowledge. It sees which documents hold relevant passages and can search inside each one again (up to 3 searches), so a question touching several documents doesn't lose one. Vector search and PostgreSQL full-text search are merged with Reciprocal Rank Fusion (k = 60). The model lists the passages it used, and those become the answer's sources.

### Nepali OCR

Before a scanned page is read, Tesseract checks its script. Devanagari pages are read with Nepali + English, and all other pages with English only, so English documents give exactly the same text as before. Pages typed in old Nepali fonts (Preeti, Kantipur and similar) are read with OCR instead of from their text layer. Without Tesseract's Nepali data, or with `NEPALI_OCR=0`, every page is read in English.

### Answer trace

Each answer records its steps: the model calls, the intent, the tools called through MCP, the searches (with the vector, keyword and merged rankings), the passages given to the model and how each document was read (text layer or OCR, and which pages as Nepali). The chat shows this under **How this answer was found**.

### Model fallback

With `LLM_PROVIDER=groq`, Groq (`openai/gpt-oss-120b`) answers. If Groq hits a rate limit or an error, search answers fall back to the local Ollama model (`llama3.2:3b`). Table requests don't fall back (the small model isn't reliable for tables) and show a short "usage limit reached" message instead.

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend API | Python, FastAPI |
| PDF reading | PyMuPDF, pdfplumber, Tesseract OCR (pytesseract), English and Nepali |
| Text chunking | LangChain text splitters |
| Embeddings | `intfloat/multilingual-e5-base` (Sentence Transformers) |
| Storage and search | PostgreSQL, pgvector, PostgreSQL full-text search |
| Language models | Groq `openai/gpt-oss-120b`, Ollama `llama3.2:3b` |
| Agent tools | MCP Python SDK (MCP servers and client, Streamable HTTP) |
| Authentication | JWT (python-jose), bcrypt (passlib), Google sign-in (google-auth) |
| Gmail | Gmail API, Google OAuth 2.0 (read-only), Fernet encryption |
| Frontend | Next.js, TypeScript, Tailwind CSS, `@react-oauth/google` |
| Deployment | Docker, Docker Compose |

---

## Project structure

```
rag-assistant-tool/
├── backend/
│   ├── main.py             # FastAPI app and endpoints; mounts the MCP servers
│   ├── pipeline.py         # Extraction, OCR, chunking, embedding, retrieval,
│   │                       # intent routing, agent loops, follow-ups, tables, summaries
│   ├── ingest.py           # Adding a PDF: saving it and reading it in the background
│   ├── mcp_client.py       # The agent's MCP client: lists and calls the servers' tools
│   ├── docquery_mcp.py     # DocQuery MCP server (document search)
│   ├── gmail_mcp.py        # Gmail MCP server (search, read, import attachment)
│   ├── gmail_account.py    # Connecting Gmail (Google OAuth) and calling the Gmail API
│   ├── answer_trace.py     # Records how each answer was made
│   ├── llm.py              # Groq / Ollama switch with automatic fallback
│   ├── auth.py             # Password hashing and JWT helpers
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── uploads/            # Stored PDFs, one folder per user (created automatically, not in git)
│   └── .env                # Your settings and keys (not in git)
├── frontend/
│   ├── app/page.tsx                    # Chat, file cards, source chips, tables, documents panel
│   ├── app/components/AuthModal.tsx    # Login, sign-up and Google sign-in window
│   ├── app/components/AnswerTrace.tsx  # "How this answer was found" panel
│   ├── app/config.ts                   # Where the backend runs
│   ├── app/globals.css                 # Colours and small animations
│   └── Dockerfile
├── db/
│   └── init.sql            # Database tables (used by Docker and for a local setup)
├── docker-compose.yml      # Database, backend and frontend
└── docker-compose.dev.yml  # Development override (live reload)
```

---

## Getting started

### Option A: Docker

The quickest way to run everything (database, backend and frontend). You need [Docker Desktop](https://www.docker.com/products/docker-desktop/) and, for the local fallback model, [Ollama](https://ollama.com) running on your computer.

1. Create `backend/.env` (see [Environment variables](#environment-variables)). At least `DB_PASSWORD` and `JWT_SECRET_KEY` are needed.
2. From the project folder:
   ```bash
   docker compose --env-file backend/.env up --build
   ```
3. Open `http://localhost:3000`

Docker uses its own empty database, created from `db/init.sql` the first time it starts. The image includes Tesseract with the English and Nepali data. The first start downloads the embedding model (about 1 GB), which is then kept in a Docker volume.

To work on the code inside Docker with live reload:
```bash
docker compose --env-file backend/.env -f docker-compose.yml -f docker-compose.dev.yml up --build
```

### Option B: Run locally

#### Prerequisites

- Python 3.10 or newer
- Node.js 18 or newer
- PostgreSQL 15 or newer with the [pgvector](https://github.com/pgvector/pgvector) extension
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) on your system PATH, with the **Nepali** (`nep`) and **OSD** (`osd`) language data for Nepali documents. Check with:
  ```bash
  tesseract --list-langs
  ```
- [Ollama](https://ollama.com) with the local model:
  ```bash
  ollama pull llama3.2:3b
  ```
- A [Groq API key](https://console.groq.com) (optional; without one, set `LLM_PROVIDER=ollama`)

#### 1. Database

```bash
psql -U postgres -c "CREATE DATABASE document_query_db;"
psql -U postgres -d document_query_db -f db/init.sql
```

`db/init.sql` creates the pgvector extension and every table: `users`, `document_chunks`, `document_info`, `chat_history` and `gmail_accounts`.

#### 2. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Create `backend/.env`:
```
DB_PASSWORD=your_postgres_password
JWT_SECRET_KEY=a_long_random_secret_string
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
```

Start the API:
```bash
uvicorn main:app --reload
```
The API runs at `http://127.0.0.1:8000`, with interactive docs at `http://127.0.0.1:8000/docs`.

#### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000`.

### Environment variables

Backend settings go in `backend/.env`.

| Variable | Required | Default | What it does |
|---|---|---|---|
| `DB_PASSWORD` | Yes | | PostgreSQL password |
| `JWT_SECRET_KEY` | Yes | | Secret used to sign login tokens (also protects the stored Gmail key). Changing it logs everyone out |
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER` | No | `localhost`, `5432`, `document_query_db`, `postgres` | Database connection |
| `LLM_PROVIDER` | No | `ollama` | `groq` or `ollama` |
| `GROQ_API_KEY` | With Groq | | Groq API key |
| `GROQ_MODEL` | No | `openai/gpt-oss-120b` | Groq model |
| `OLLAMA_MODEL` | No | `llama3.2:3b` | Local model, also the fallback |
| `OCR_WORKERS` | No | CPU cores, up to 4 | How many scanned pages are read at the same time |
| `NEPALI_OCR` | No | `1` | `0` reads every scanned page in English only |
| `MAX_UPLOAD_MB` | No | `50` | Largest PDF accepted |
| `CORS_ORIGINS` | No | `http://localhost:3000` | Frontend addresses allowed to call the API (comma separated) |
| `GOOGLE_CLIENT_ID` | For Google | | Google sign-in and Gmail |
| `GOOGLE_CLIENT_SECRET` | For Gmail | | Gmail connection |
| `GMAIL_REDIRECT_URI` | No | `http://127.0.0.1:8000/gmail/callback` | Where Google returns after allowing Gmail |
| `FRONTEND_URL` | No | `http://localhost:3000` | Where the backend sends the browser back after connecting Gmail |

Frontend settings go in `frontend/.env.local`.

| Variable | Default | What it does |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:8000` | Where the backend runs |
| `NEXT_PUBLIC_GOOGLE_CLIENT_ID` | | Google sign-in (same value as `GOOGLE_CLIENT_ID`) |

Login tokens are valid for 7 days.

### Google sign-in (optional)

1. In the [Google Cloud console](https://console.cloud.google.com/apis/credentials), create an **OAuth client ID** of type *Web application*
2. Add `http://localhost:3000` under *Authorized JavaScript origins*
3. Put the client ID in `backend/.env` as `GOOGLE_CLIENT_ID` and in `frontend/.env.local` as `NEXT_PUBLIC_GOOGLE_CLIENT_ID` (the client ID is public, not a secret)

The backend verifies the Google token with `google-auth`. An account created with Google has no password, so it signs in with Google only.

### Gmail (optional)

1. In the same Google Cloud project, enable the **Gmail API**
2. On the **OAuth consent screen**, add the `gmail.readonly` scope. While the app is in *Testing*, add each Gmail address that will be used under **Test users** (up to 100)
3. In the OAuth client, add `http://127.0.0.1:8000/gmail/callback` under *Authorized redirect URIs*, and copy the client secret
4. Add to `backend/.env`:
   ```
   GOOGLE_CLIENT_SECRET=your_google_oauth_client_secret
   GMAIL_REDIRECT_URI=http://127.0.0.1:8000/gmail/callback
   FRONTEND_URL=http://localhost:3000
   ```
5. Restart the backend and click **Connect Gmail** in the header

---

## Usage

1. Open the app and sign up, log in or continue with Google
2. Click **+** (or drag PDFs onto the page) to attach documents. Send them with a question, or on their own and wait for **Ready**
3. Type a question and press **Enter** to send (**Shift + Enter** starts a new line). Try:
   - A direct question: "What is the due date and the payment term?"
   - A table: "Show all items with quantity above 100"
   - "Summarize the story" or "Summarize these two documents I just uploaded"
   - "Which document talks about glaciers?"
   - In Nepali: "छात्रवृत्ति पाउने विद्यार्थीहरूको नाम के हो?"
   - A follow-up: "make it shorter", "give me an example", "as bullet points"
   - With Gmail connected: "emails with a PDF attached this week", then "import that PDF"
4. Click a source chip to open the PDF at that page, and **How this answer was found** to see the steps
5. Use **My documents** to open or delete a document, and the account menu to clear the chat or log out

---

## API reference

| Endpoint | Login | Purpose |
|---|---|---|
| `POST /signup`, `POST /login` | No | Create an account / log in; returns a token |
| `POST /auth/google` | No | Sign in with Google (creates the account the first time) |
| `POST /upload` | Yes | Save a PDF and start reading it in the background |
| `GET /upload/status?ids=` | Yes | Progress of the background reading |
| `POST /query` | Optional | Ask a question (documents need a login; small talk works without) |
| `GET /documents` | Yes | The user's documents, newest first |
| `GET /documents/file?name=` | Yes | Open the stored PDF |
| `DELETE /documents?name=` | Yes | Delete a document and its stored PDF |
| `GET /chat`, `PUT /chat`, `DELETE /chat` | Yes | Load, save and clear the saved chat |
| `GET /gmail/status`, `POST /gmail/connect`, `DELETE /gmail` | Yes | Check, start and remove the Gmail connection |
| `GET /gmail/callback` | No | Where Google returns after the permission screen |
| `/mcp`, `/gmail-mcp` | Token | The DocQuery and Gmail MCP servers used by the agent |

---

## Security and privacy

- Passwords are hashed with bcrypt; logins use signed JWT tokens that expire after 7 days
- Every document, chunk, stored PDF, saved chat and Gmail connection belongs to one user, and every query is filtered by that user
- MCP tools take the user from the login token sent with each call, never from what the model writes
- Gmail access is **read-only** (`gmail.readonly`): DocQuery can never send, change or delete emails. The Google refresh token is stored encrypted, and **Disconnect Gmail** also cancels the access at Google
- Email text is treated as data, never as instructions to the assistant
- `backend/.env`, `frontend/.env.local` and `backend/uploads/` are kept out of git

---

## Known limitations

- **OCR accuracy on scanned documents.** Results depend on scan quality: digits and some joined Nepali letters can be misread, including certain digits in Preeti-font documents. Text-based PDFs are read exactly.
- **Very short scanned pages.** A page with only a few words may be too short for script detection and is then read in English.
- **Long or complex tables.** Very long tables may lose rows, and a row that continues on the next page may be read as two rows.
- **Groq usage limits.** On the free tier (about 8,000 tokens per minute and 200,000 per day), search answers fall back to the local model and table extraction is paused until the limit resets.
- **Conversation memory.** Only simple facts the user states, such as their name, are remembered, not the full conversation.
- **Gmail in testing mode.** While the Google app is unverified, a connection expires after 7 days and only registered test users (up to 100) can connect.
