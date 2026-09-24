import shutil
import os
import hashlib
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from pipeline import ocr_pdf, chunk_text, embed_chunks, store_chunks, retrieve_chunks, generate_answer, get_connection
from fastapi.middleware.cors import CORSMiddleware
from pipeline import agent_query
from fastapi import Depends, HTTPException, Header
from auth import hash_password, verify_password, create_access_token, decode_access_token

app = FastAPI()

class SignupRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


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

    if not row or not verify_password(request.password, row[1]):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    user_id = row[0]
    token = create_access_token({"user_id": user_id, "email": request.email})
    return {"access_token": token, "token_type": "bearer"}

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
    until we find a name not currently in the KB — scoped to this user only,
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
        cur.close()
        conn.close()
        return {
            "filename": file.filename,
            "chunks_stored": 0,
            "message": "This document (or an identical copy) is already in your knowledge base."
        }

    storage_filename = get_unique_filename(cur, file.filename, user_id)
    cur.close()
    conn.close()

    temp_path = f"temp_{storage_filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(content)

    text = ocr_pdf(temp_path)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    store_chunks(storage_filename, chunks, embeddings, content_hash, user_id)

    os.remove(temp_path)

    return {"filename": storage_filename, "chunks_stored": len(chunks)}


class QueryRequest(BaseModel):
    question: str
    source_file: str | None = None  # optional — omit to search the whole knowledge base


@app.post("/query")
def query_documents(request: QueryRequest, current_user: dict | None = Depends(get_optional_user)):
    user_id = current_user["user_id"] if current_user else None
    result = agent_query(request.question, user_id, source_file=request.source_file)
    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"],
        "action_taken": result["action_taken"]
    }
    
@app.get("/documents")
def get_documents(current_user: dict = Depends(get_current_user)):
    return {"documents": list_documents(current_user["user_id"])}