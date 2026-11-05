import shutil
import os
import json
import hashlib
import threading
import time
import uuid
import re
import traceback
from psycopg2.extras import Json
from psycopg2.errors import UniqueViolation
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pipeline import ocr_pdf, chunk_text_with_pages, embed_chunks, store_chunks, retrieve_chunks, generate_answer, get_connection, db_cursor, list_documents
from fastapi.middleware.cors import CORSMiddleware
from pipeline import agent_query
from fastapi import Depends, HTTPException, Header
from llm import LLMBusyError
from auth import hash_password, verify_password, create_access_token, decode_access_token
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests
from dotenv import load_dotenv

load_dotenv()
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
# Websites allowed to call this API (comma separated). Default: the local frontend.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]

app = FastAPI()

# Uploaded PDFs are kept here (one folder per user, named by the file's hash),
# so a source can be opened from the chat later.
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")


def pdf_path(user_id, content_hash):
    return os.path.join(UPLOAD_DIR, str(user_id), f"{content_hash}.pdf")

class SignupRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ChatSaveRequest(BaseModel):
    messages: list[dict]


class GoogleLoginRequest(BaseModel):
    credential: str  # the ID token Google gives the browser after the user picks an account


def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = authorization.split(" ")[1]
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return {"user_id": payload["user_id"], "email": payload["email"]}


MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_BYTES = 72  # bcrypt only uses the first 72 bytes of a password


def email_problem(email):
    """Why this (already trimmed and lower-cased) email cannot be used to sign up, or None
    when it is fine. The sign-up form in the browser checks the same rules."""
    example = "like name@example.com"
    if not email:
        return f"Please enter your email, {example}."
    if re.search(r"\s", email):
        return "An email can't contain spaces."
    if len(email) > 254:
        return "This email is too long (the limit is 254 characters)."
    parts = email.split("@")
    if len(parts) < 2:
        return f"Your email needs an @ sign, {example}."
    if len(parts) > 2:
        return "An email can only have one @ sign."
    local, domain = parts
    if not local:
        return f"Please type something before the @, {example}."
    if len(local) > 64:
        return "The part before the @ is too long (the limit is 64 characters)."
    if not re.fullmatch(r"[a-z0-9._%+-]+", local):
        return "Before the @, use only letters, numbers and . _ % + -"
    if local.startswith(".") or local.endswith("."):
        return "The part before the @ can't start or end with a dot."
    if ".." in local:
        return "The part before the @ can't have two dots in a row."
    if len(domain) < 3:
        return f"Please type your email provider after the @, {example}."
    if len(domain) > 255:
        return "The part after the @ is too long."
    if not re.fullmatch(r"[a-z0-9.-]+", domain):
        return "After the @, use only letters, numbers, hyphens and dots."
    labels = domain.split(".")
    if len(labels) < 2:
        return "The part after @ should look like example.com."
    if any(label == "" or len(label) > 63 for label in labels):
        return "The part after @ should look like example.com."
    if any(label.startswith("-") or label.endswith("-") for label in labels):
        return "A part of the domain can't start or end with a hyphen."
    if not re.fullmatch(r"[a-z]{2,}", labels[-1]):
        return "The ending (like .com) must be at least 2 letters."
    return None


def clean_email(email):
    """Emails are stored in lower case, so Seema@x.com and seema@x.com are one account."""
    return (email or "").strip().lower()


@app.post("/signup")
def signup(request: SignupRequest):
    email = clean_email(request.email)
    problem = email_problem(email)
    if problem:
        raise HTTPException(status_code=400, detail=problem)
    if len(request.password) < MIN_PASSWORD_LENGTH:
        raise HTTPException(status_code=400, detail=f"Your password must be at least {MIN_PASSWORD_LENGTH} characters long.")
    if len(request.password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise HTTPException(status_code=400, detail=f"Your password is too long (the limit is {MAX_PASSWORD_BYTES} characters).")

    already = "An account with this email already exists."
    try:
        with db_cursor(commit=True) as cur:
            # LOWER() so accounts created before emails were lower-cased are found too
            cur.execute("SELECT id FROM users WHERE LOWER(email) = %s", (email,))
            if cur.fetchone():
                raise HTTPException(status_code=400, detail=already)
            cur.execute(
                "INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id",
                (email, hash_password(request.password))
            )
            user_id = cur.fetchone()[0]
    except UniqueViolation:
        # two sign-ups with the same email at the very same moment
        raise HTTPException(status_code=400, detail=already)

    token = create_access_token({"user_id": user_id, "email": email})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/login")
def login(request: LoginRequest):
    email = clean_email(request.email)
    with db_cursor() as cur:
        cur.execute("SELECT id, password_hash, email FROM users WHERE LOWER(email) = %s ORDER BY id LIMIT 1", (email,))
        row = cur.fetchone()

    if row and not row[1]:
        raise HTTPException(status_code=400, detail="This account uses Google sign-in. Please use 'Continue with Google'.")

    if not row or not verify_password(request.password, row[1]):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    token = create_access_token({"user_id": row[0], "email": row[2]})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/auth/google")
def google_login(request: GoogleLoginRequest):
    """Sign in (or sign up) with Google. The browser sends the ID token from Google;
    we check it with Google, then find or create the user by email and
    return our own login token, exactly like a normal login."""
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Google sign-in is not set up on the server.")

    try:
        info = google_id_token.verify_oauth2_token(
            request.credential, google_requests.Request(), GOOGLE_CLIENT_ID
        )
    except ValueError:
        raise HTTPException(status_code=401, detail="Google sign-in failed. Please try again.")

    email = clean_email(info.get("email"))
    if not email or not info.get("email_verified"):
        raise HTTPException(status_code=401, detail="Your Google email is not verified.")

    with db_cursor(commit=True) as cur:
        cur.execute("SELECT id, email FROM users WHERE LOWER(email) = %s ORDER BY id LIMIT 1", (email,))
        row = cur.fetchone()
        if row:
            user_id, email = row  # keep the email exactly as it is stored
        else:
            # Google users have no password, so password_hash is left empty (NULL)
            cur.execute("INSERT INTO users (email, password_hash) VALUES (%s, NULL) RETURNING id", (email,))
            user_id = cur.fetchone()[0]

    token = create_access_token({"user_id": user_id, "email": email})
    return {"access_token": token, "token_type": "bearer", "email": email}


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Document Query Tool is running"}


def get_unique_filename(cur, filename, user_id):
    """If filename is already used by different content, append (1), (2), etc.
    until we find a name not currently in the KB scoped to this user only,
    since different users can have files with the same name."""
    cur.execute("SELECT DISTINCT source_file FROM document_chunks WHERE user_id = %s;", (user_id,))
    existing_names = {row[0] for row in cur.fetchall()}

    if filename not in existing_names:
        return filename

    name, ext = os.path.splitext(filename)
    counter = 1
    while True:
        candidate = f"{name}({counter}){ext}"
        if candidate not in existing_names:
            return candidate
        counter += 1


def get_optional_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        return None

    token = authorization.split(" ")[1]
    payload = decode_access_token(token)
    if not payload:
        return None

    return {"user_id": payload["user_id"], "email": payload["email"]}

# ── Background reading of uploaded PDFs ───────────────────────────────────
# /upload only saves the file and answers straight away. The slow part (reading
# the text, OCR, embedding) runs in a background thread, so the server stays
# responsive and the browser never waits on a big PDF. The jobs are kept in
# memory: if the server restarts while a file is still being read, that job is
# lost and the file simply has to be uploaded again.
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))  # biggest PDF accepted
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
UPLOAD_JOBS = {}              # job_id -> {user_id, filename, hash, status, chunks, error, created}
JOBS_LOCK = threading.Lock()  # protects UPLOAD_JOBS
READ_SLOT = threading.Semaphore(1)  # only one PDF is read at a time; the others wait their turn
JOB_KEEP_SECONDS = 3600       # finished jobs are forgotten after an hour


class UploadReadError(Exception):
    """A problem with the PDF itself that the user can understand (shown as it is)."""


def _job_view(job_id, job):
    return {"job_id": job_id, "filename": job["filename"], "status": job["status"],
            "chunks": job["chunks"], "error": job["error"]}


def _forget_old_jobs():
    now = time.time()
    with JOBS_LOCK:
        for job_id in [j for j, v in UPLOAD_JOBS.items()
                       if v["status"] != "reading" and now - v["created"] > JOB_KEEP_SECONDS]:
            del UPLOAD_JOBS[job_id]


def _read_pdf_in_background(job_id, saved_path, storage_filename, content_hash, user_id):
    """Runs in its own thread: read the PDF, split it, embed it and store it."""
    try:
        with READ_SLOT:
            text = ocr_pdf(saved_path)
            chunk_pairs = chunk_text_with_pages(text)
            if not chunk_pairs:
                raise UploadReadError("No readable text was found in this PDF.")
            chunks = [c for c, _ in chunk_pairs]
            pages = [pg for _, pg in chunk_pairs]
            embeddings = embed_chunks(chunks)
            store_chunks(storage_filename, chunks, embeddings, content_hash, user_id, pages=pages)
        with JOBS_LOCK:
            UPLOAD_JOBS[job_id].update(status="done", chunks=len(chunks))
    except Exception as e:
        print(f"UPLOAD: reading '{storage_filename}' failed ({type(e).__name__}): {e}")
        traceback.print_exc()
        # don't keep a file that never made it into the knowledge base
        if os.path.exists(saved_path):
            os.remove(saved_path)
        message = str(e) if isinstance(e, UploadReadError) else "This PDF could not be read."
        with JOBS_LOCK:
            UPLOAD_JOBS[job_id].update(status="failed", error=message)


@app.post("/upload")
async def upload_document(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    upload_name = os.path.basename((file.filename or "").replace("\\", "/")) or "document.pdf"
    # read one byte more than the limit, so a too-big file is noticed without loading all of it
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"This file is too large. The limit is {MAX_UPLOAD_MB} MB.")
    if not content:
        raise HTTPException(status_code=400, detail="This file is empty.")
    if b"%PDF-" not in content[:1024]:
        raise HTTPException(status_code=400, detail="This is not a PDF file. Only PDF files can be uploaded.")
    content_hash = hashlib.sha256(content).hexdigest()
    _forget_old_jobs()

    # 0. The same file is already being read for this user -> just follow that job
    with JOBS_LOCK:
        for job_id, job in UPLOAD_JOBS.items():
            if job["user_id"] == user_id and job["hash"] == content_hash and job["status"] == "reading":
                return {"status": "reading", "job_id": job_id, "filename": job["filename"]}
        reserved = {job["filename"] for job in UPLOAD_JOBS.values()
                    if job["user_id"] == user_id and job["status"] == "reading"}

    with db_cursor() as cur:
        # 1. Same content already present for THIS user -> skip reprocessing entirely
        cur.execute(
            "SELECT COUNT(*) FROM document_chunks WHERE content_hash = %s AND user_id = %s",
            (content_hash, user_id)
        )
        already_exists = cur.fetchone()[0] > 0

        if already_exists:
            # the name it is stored under may differ from the name of this upload
            cur.execute(
                "SELECT source_file FROM document_chunks WHERE content_hash = %s AND user_id = %s LIMIT 1",
                (content_hash, user_id)
            )
            stored_name = cur.fetchone()[0]
            return {
                "status": "done",
                "filename": stored_name,
                "chunks_stored": 0,
                "message": "This document (or an identical copy) is already in your knowledge base."
            }

        storage_filename = get_unique_filename(cur, upload_name, user_id)

    # a name another file of this user is about to take (it is still being read)
    if storage_filename in reserved:
        name, ext = os.path.splitext(upload_name)
        counter = 1
        while storage_filename in reserved or _name_is_taken(storage_filename, user_id):
            storage_filename = f"{name}({counter}){ext}"
            counter += 1

    saved_path = pdf_path(user_id, content_hash)
    os.makedirs(os.path.dirname(saved_path), exist_ok=True)
    with open(saved_path, "wb") as buffer:
        buffer.write(content)

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        UPLOAD_JOBS[job_id] = {"user_id": user_id, "filename": storage_filename, "hash": content_hash,
                               "status": "reading", "chunks": 0, "error": None, "created": time.time()}
    threading.Thread(
        target=_read_pdf_in_background,
        args=(job_id, saved_path, storage_filename, content_hash, user_id),
        daemon=True,
    ).start()

    return {"status": "reading", "job_id": job_id, "filename": storage_filename}


def _name_is_taken(filename, user_id):
    """True if this user already has a stored document with this exact name."""
    with db_cursor() as cur:
        cur.execute("SELECT 1 FROM document_chunks WHERE source_file = %s AND user_id = %s LIMIT 1;", (filename, user_id))
        taken = cur.fetchone() is not None
    return taken


@app.get("/upload/status")
def upload_status(ids: str | None = None, current_user: dict = Depends(get_current_user)):
    """How the background reading is going. With ids (comma separated job ids): those jobs.
    Without ids: all of this user's files that are still being read."""
    user_id = current_user["user_id"]
    wanted = [i for i in (ids or "").split(",") if i]
    with JOBS_LOCK:
        if wanted:
            jobs = {}
            for job_id in wanted:
                job = UPLOAD_JOBS.get(job_id)
                if job and job["user_id"] == user_id:
                    jobs[job_id] = _job_view(job_id, job)
                else:
                    # unknown: the server was restarted, or the job is too old
                    jobs[job_id] = {"job_id": job_id, "status": "unknown", "filename": None,
                                    "chunks": 0, "error": "This upload was interrupted. Please upload the PDF again."}
            return {"jobs": jobs}
        return {"jobs": {j: _job_view(j, v) for j, v in UPLOAD_JOBS.items()
                         if v["user_id"] == user_id and v["status"] == "reading"}}


class QueryRequest(BaseModel):
    question: str
    source_file: str | None = None  # optional — omit to search the whole knowledge base
    source_files: list[str] | None = None  # optional — limit the question to these documents (e.g. files just attached)
    history: list[dict] | None = None  # recent [{"question":..., "answer":...}, ...] for follow-ups
    recent_uploads: list[str] | None = None  # the files added most recently, so "these two documents" can be understood
    chat_user_messages: list[str] | None = None  # everything the user typed in this chat, so small details they stated can be remembered

@app.post("/query")
def query_documents(request: QueryRequest, current_user: dict | None = Depends(get_optional_user)):
    user_id = current_user["user_id"] if current_user else None
    scope = request.source_files or request.source_file
    try:
        result = agent_query(request.question, user_id, source_file=scope, history=request.history,
                             user_messages=request.chat_user_messages, recent_uploads=request.recent_uploads)
    except Exception as e:
        # Never show the user a server error: say what happened in plain words.
        if isinstance(e, LLMBusyError):
            message = "The AI service is busy right now, so I couldn't finish this answer. Please try again in a minute."
        else:
            print(f"QUERY: answering failed ({type(e).__name__}): {e}")
            traceback.print_exc()
            message = "Sorry, something went wrong while preparing the answer. Please try again."
        return {"question": request.question, "answer": message, "sources": [], "source_refs": None,
                "action_taken": "ERROR", "structured": None}
    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"],
        "source_refs": result.get("source_refs"),  # [{"file": ..., "pages": [3]}] for SEARCH answers, else null
        "action_taken": result["action_taken"],
        "structured": result.get("structured"),  # {"pairs": [...], "tables": [...], "note": "..."} for TABLE answers, else null
    }
    
@app.get("/documents")
def get_documents(current_user: dict = Depends(get_current_user)):
    return {"documents": list_documents(current_user["user_id"])}


@app.delete("/documents")
def delete_document(name: str, current_user: dict = Depends(get_current_user)):
    """Remove one of the signed-in user's documents (all its chunks).
    The user_id check means nobody can delete another user's file."""
    user_id = current_user["user_id"]
    with db_cursor(commit=True) as cur:
        cur.execute(
            "SELECT content_hash FROM document_chunks WHERE source_file = %s AND user_id = %s LIMIT 1",
            (name, user_id)
        )
        row = cur.fetchone()
        cur.execute(
            "DELETE FROM document_chunks WHERE source_file = %s AND user_id = %s",
            (name, user_id)
        )
        deleted = cur.rowcount

    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found.")

    # also remove the stored PDF (older documents have none)
    if row and row[0]:
        try:
            os.remove(pdf_path(user_id, row[0]))
        except OSError:
            pass

    return {"deleted": name, "chunks_removed": deleted}


@app.get("/documents/file")
def get_document_file(name: str, current_user: dict = Depends(get_current_user)):
    """Send back the stored PDF of one of the signed-in user's documents."""
    user_id = current_user["user_id"]
    with db_cursor() as cur:
        cur.execute(
            "SELECT content_hash FROM document_chunks WHERE source_file = %s AND user_id = %s LIMIT 1",
            (name, user_id)
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Document not found.")

    path = pdf_path(user_id, row[0]) if row[0] else None
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="The original file isn't stored for this document. Delete it and upload it again to enable opening.")

    return FileResponse(path, media_type="application/pdf", filename=name, content_disposition_type="inline")


MAX_SAVED_MESSAGES = 100
MAX_SAVED_CHAT_BYTES = 2_000_000


@app.get("/chat")
def get_chat(current_user: dict = Depends(get_current_user)):
    """The signed-in user's saved chat (empty list if there is none)."""
    with db_cursor() as cur:
        cur.execute("SELECT messages FROM chat_history WHERE user_id = %s", (current_user["user_id"],))
        row = cur.fetchone()
    return {"messages": row[0] if row else []}


@app.put("/chat")
def save_chat(request: ChatSaveRequest, current_user: dict = Depends(get_current_user)):
    """Save the user's chat, replacing the previous copy. Only the latest
    messages are kept, and the oldest are dropped if the chat gets too big."""
    messages = request.messages[-MAX_SAVED_MESSAGES:]
    while len(messages) > 1 and len(json.dumps(messages)) > MAX_SAVED_CHAT_BYTES:
        messages = messages[1:]

    with db_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO chat_history (user_id, messages, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (user_id) DO UPDATE SET messages = EXCLUDED.messages, updated_at = NOW()
            """,
            (current_user["user_id"], Json(messages))
        )
    return {"saved": len(messages)}


@app.delete("/chat")
def clear_chat(current_user: dict = Depends(get_current_user)):
    """Delete the user's saved chat."""
    with db_cursor(commit=True) as cur:
        cur.execute("DELETE FROM chat_history WHERE user_id = %s", (current_user["user_id"],))
    return {"cleared": True}