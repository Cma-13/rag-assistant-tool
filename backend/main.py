import contextlib
import shutil
import os
import json
import re
import traceback
from urllib.parse import urlencode
from psycopg2.extras import Json
from psycopg2.errors import UniqueViolation
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from pipeline import retrieve_chunks, generate_answer, get_connection, db_cursor, list_documents
from fastapi.middleware.cors import CORSMiddleware
from pipeline import agent_query, forget_document_info
import docquery_mcp
import gmail_account
import gmail_mcp
import ingest
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

# The two MCP servers the chat's agent connects to, over Streamable HTTP (stateless:
# each call is a separate request): DocQuery's own at /mcp/ and Gmail at /gmail-mcp/.
_mcp_app = docquery_mcp.server.streamable_http_app(streamable_http_path="/", stateless_http=True, json_response=True)
_gmail_mcp_app = gmail_mcp.server.streamable_http_app(streamable_http_path="/", stateless_http=True, json_response=True)
# Where to send the browser back to after connecting Gmail.
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000").rstrip("/")


@contextlib.asynccontextmanager
async def _lifespan(app):
    # Each MCP server needs its session manager running for as long as the app runs.
    async with docquery_mcp.server.session_manager.run(), gmail_mcp.server.session_manager.run():
        yield


app = FastAPI(lifespan=_lifespan)

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


EMAIL_ERROR = "Please enter a valid email address."


def email_problem(email):
    """Why this (already trimmed and lower-cased) email cannot be used to sign up, or None
    when it is fine. The sign-up form in the browser checks the same rules."""
    if not email:
        return EMAIL_ERROR
    if re.search(r"\s", email):
        return EMAIL_ERROR
    if len(email) > 254:
        return EMAIL_ERROR
    parts = email.split("@")
    if len(parts) < 2:
        return EMAIL_ERROR
    if len(parts) > 2:
        return EMAIL_ERROR
    local, domain = parts
    if not local:
        return EMAIL_ERROR
    if len(local) > 64:
        return EMAIL_ERROR
    if not re.fullmatch(r"[a-z0-9._%+-]+", local):
        return EMAIL_ERROR
    if local.startswith(".") or local.endswith("."):
        return EMAIL_ERROR
    if ".." in local:
        return EMAIL_ERROR
    if len(domain) < 3:
        return EMAIL_ERROR
    if len(domain) > 255:
        return EMAIL_ERROR
    if not re.fullmatch(r"[a-z0-9.-]+", domain):
        return EMAIL_ERROR
    labels = domain.split(".")
    if len(labels) < 2:
        return EMAIL_ERROR
    if any(label == "" or len(label) > 63 for label in labels):
        return EMAIL_ERROR
    if any(label.startswith("-") or label.endswith("-") for label in labels):
        return EMAIL_ERROR
    if not re.fullmatch(r"[a-z]{2,}", labels[-1]):
        return EMAIL_ERROR
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
    # read one byte more than the limit, so a too-big file is noticed without loading all of it
    content = await file.read(ingest.MAX_UPLOAD_BYTES + 1)
    try:
        return ingest.add_pdf(content, file.filename, current_user["user_id"])
    except ingest.PdfRejected as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@app.get("/upload/status")
def upload_status(ids: str | None = None, current_user: dict = Depends(get_current_user)):
    """How the background reading is going. With ids (comma separated job ids): those jobs.
    Without ids: all of this user's files that are still being read."""
    wanted = [i for i in (ids or "").split(",") if i]
    return {"jobs": ingest.jobs_status(current_user["user_id"], wanted)}


class QueryRequest(BaseModel):
    question: str
    source_file: str | None = None  # optional — omit to search the whole knowledge base
    source_files: list[str] | None = None  # optional — limit the question to these documents (e.g. files just attached)
    history: list[dict] | None = None  # recent [{"question":..., "answer":...}, ...] for follow-ups
    recent_uploads: list[str] | None = None  # the files added most recently, so "these two documents" can be understood
    chat_user_messages: list[str] | None = None  # everything the user typed in this chat, so small details they stated can be remembered
    timezone: str | None = None  # the user's time zone from the browser (e.g. Asia/Kathmandu), for email times

@app.post("/query")
def query_documents(request: QueryRequest, current_user: dict | None = Depends(get_optional_user)):
    user_id = current_user["user_id"] if current_user else None
    scope = request.source_files or request.source_file
    try:
        result = agent_query(request.question, user_id, source_file=scope, history=request.history,
                             user_messages=request.chat_user_messages, recent_uploads=request.recent_uploads,
                             timezone=request.timezone)
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
        "trace": result.get("trace"),  # the steps taken to find the answer, shown under it in the app
        "imported": result.get("imported"),  # PDFs added from Gmail by this answer: [{"job_id", "filename"}], else null
        "emails": result.get("emails"),  # emails the answer is about, shown as cards: [{"from", "date", "subject", "preview", "attachments"}]
    }
    
# ── Gmail connection ─────────────────────────────────────────────────────
@app.get("/gmail/status")
def gmail_status(current_user: dict = Depends(get_current_user)):
    """Whether this user has connected Gmail, and which address."""
    return gmail_account.status(current_user["user_id"])


@app.post("/gmail/connect")
def gmail_connect(current_user: dict = Depends(get_current_user)):
    """The Google page to send the browser to, where the user allows read-only Gmail access."""
    try:
        return {"url": gmail_account.connect_url(current_user["user_id"])}
    except gmail_account.GmailError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/gmail/callback")
def gmail_callback(code: str | None = None, state: str | None = None, error: str | None = None):
    """Google sends the browser back here after the permission screen. The result is
    passed on to the app in the address (?gmail=connected or ?gmail_error=...)."""
    if error or not code:
        # "access_denied" when the user pressed Cancel on Google's screen
        message = "Gmail was not connected." if error == "access_denied" else "Google did not finish the connection."
        return RedirectResponse(f"{FRONTEND_URL}/?{urlencode({'gmail_error': message})}")
    try:
        gmail_account.finish_connect(code, state)
    except gmail_account.GmailError as e:
        return RedirectResponse(f"{FRONTEND_URL}/?{urlencode({'gmail_error': str(e)})}")
    return RedirectResponse(f"{FRONTEND_URL}/?gmail=connected")


@app.delete("/gmail")
def gmail_disconnect(current_user: dict = Depends(get_current_user)):
    """Disconnect Gmail (DocQuery's access is also cancelled at Google)."""
    gmail_account.disconnect(current_user["user_id"])
    return {"connected": False}


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
    try:
        forget_document_info(name, user_id)
    except Exception as e:
        print(f"DELETE: could not remove page details for '{name}' ({type(e).__name__}): {e}")

    # also remove the stored PDF (older documents have none)
    if row and row[0]:
        try:
            os.remove(ingest.pdf_path(user_id, row[0]))
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

    path = ingest.pdf_path(user_id, row[0]) if row[0] else None
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


# Mounted last so all the normal routes above are matched first.
app.mount("/mcp", _mcp_app)
app.mount("/gmail-mcp", _gmail_mcp_app)
