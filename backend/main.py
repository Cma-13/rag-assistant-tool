import shutil
import os
import json
import hashlib
from psycopg2.extras import Json
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pipeline import ocr_pdf, chunk_text_with_pages, embed_chunks, store_chunks, retrieve_chunks, generate_answer, get_connection, list_documents
from fastapi.middleware.cors import CORSMiddleware
from pipeline import agent_query
from fastapi import Depends, HTTPException, Header
from auth import hash_password, verify_password, create_access_token, decode_access_token
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests
from dotenv import load_dotenv

load_dotenv()
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

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


@app.post("/signup")
def signup(request: SignupRequest):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT id FROM users WHERE email = %s", (request.email,))
    if cur.fetchone():
        cur.close()
        conn.close()
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    hashed = hash_password(request.password)
    cur.execute(
        "INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id",
        (request.email, hashed)
    )
    user_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    token = create_access_token({"user_id": user_id, "email": request.email})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/login")
def login(request: LoginRequest):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT id, password_hash FROM users WHERE email = %s", (request.email,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row and not row[1]:
        raise HTTPException(status_code=400, detail="This account uses Google sign-in. Please use 'Continue with Google'.")

    if not row or not verify_password(request.password, row[1]):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    user_id = row[0]
    token = create_access_token({"user_id": user_id, "email": request.email})
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

    email = info.get("email")
    if not email or not info.get("email_verified"):
        raise HTTPException(status_code=401, detail="Your Google email is not verified.")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE email = %s", (email,))
    row = cur.fetchone()
    if row:
        user_id = row[0]
    else:
        # Google users have no password, so password_hash is left empty (NULL)
        cur.execute("INSERT INTO users (email, password_hash) VALUES (%s, NULL) RETURNING id", (email,))
        user_id = cur.fetchone()[0]
        conn.commit()
    cur.close()
    conn.close()

    token = create_access_token({"user_id": user_id, "email": email})
    return {"access_token": token, "token_type": "bearer", "email": email}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
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

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    content = await file.read()
    content_hash = hashlib.sha256(content).hexdigest()

    conn = get_connection()
    cur = conn.cursor()

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
        cur.close()
        conn.close()
        return {
            "filename": stored_name,
            "chunks_stored": 0,
            "message": "This document (or an identical copy) is already in your knowledge base."
        }

    storage_filename = get_unique_filename(cur, file.filename, user_id)
    cur.close()
    conn.close()

    saved_path = pdf_path(user_id, content_hash)
    os.makedirs(os.path.dirname(saved_path), exist_ok=True)
    with open(saved_path, "wb") as buffer:
        buffer.write(content)

    try:
        text = ocr_pdf(saved_path)
        chunk_pairs = chunk_text_with_pages(text)
        chunks = [c for c, _ in chunk_pairs]
        pages = [pg for _, pg in chunk_pairs]
        embeddings = embed_chunks(chunks)
        store_chunks(storage_filename, chunks, embeddings, content_hash, user_id, pages=pages)
    except Exception:
        # don't keep a file that never made it into the knowledge base
        if os.path.exists(saved_path):
            os.remove(saved_path)
        raise

    return {"filename": storage_filename, "chunks_stored": len(chunks)}


class QueryRequest(BaseModel):
    question: str
    source_file: str | None = None  # optional — omit to search the whole knowledge base
    source_files: list[str] | None = None  # optional — limit the question to these documents (e.g. files just attached)
    history: list[dict] | None = None  # recent [{"question":..., "answer":...}, ...] for follow-ups

@app.post("/query")
def query_documents(request: QueryRequest, current_user: dict | None = Depends(get_optional_user)):
    user_id = current_user["user_id"] if current_user else None
    scope = request.source_files or request.source_file
    result = agent_query(request.question, user_id, source_file=scope, history=request.history)
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
    conn = get_connection()
    cur = conn.cursor()
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
    conn.commit()
    cur.close()
    conn.close()

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
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT content_hash FROM document_chunks WHERE source_file = %s AND user_id = %s LIMIT 1",
        (name, user_id)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

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
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT messages FROM chat_history WHERE user_id = %s", (current_user["user_id"],))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return {"messages": row[0] if row else []}


@app.put("/chat")
def save_chat(request: ChatSaveRequest, current_user: dict = Depends(get_current_user)):
    """Save the user's chat, replacing the previous copy. Only the latest
    messages are kept, and the oldest are dropped if the chat gets too big."""
    messages = request.messages[-MAX_SAVED_MESSAGES:]
    while len(messages) > 1 and len(json.dumps(messages)) > MAX_SAVED_CHAT_BYTES:
        messages = messages[1:]

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO chat_history (user_id, messages, updated_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (user_id) DO UPDATE SET messages = EXCLUDED.messages, updated_at = NOW()
        """,
        (current_user["user_id"], Json(messages))
    )
    conn.commit()
    cur.close()
    conn.close()
    return {"saved": len(messages)}


@app.delete("/chat")
def clear_chat(current_user: dict = Depends(get_current_user)):
    """Delete the user's saved chat."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM chat_history WHERE user_id = %s", (current_user["user_id"],))
    conn.commit()
    cur.close()
    conn.close()
    return {"cleared": True}