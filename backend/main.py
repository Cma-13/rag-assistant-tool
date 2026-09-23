import shutil
import os
import hashlib
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from pipeline import ocr_pdf, chunk_text, embed_chunks, store_chunks, retrieve_chunks, generate_answer, get_connection
from fastapi.middleware.cors import CORSMiddleware
from pipeline import agent_query

app = FastAPI()

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


def get_unique_filename(cur, filename):
    """If filename is already used by different content, append (1), (2), etc.
    until we find a name not currently in the knowledge base."""
    cur.execute("SELECT DISTINCT source_file FROM document_chunks;")
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


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    content_hash = hashlib.sha256(content).hexdigest()

    conn = get_connection()
    cur = conn.cursor()

    # 1. Same content already present (under any filename) -> skip reprocessing entirely
    cur.execute("SELECT COUNT(*) FROM document_chunks WHERE content_hash = %s", (content_hash,))
    already_exists = cur.fetchone()[0] > 0

    if already_exists:
        cur.close()
        conn.close()
        return {
            "filename": file.filename,
            "chunks_stored": 0,
            "message": "This document (or an identical copy) is already in the knowledge base."
        }

    # 2. Not identical content — if this filename is already used by different content,
    #    keep both by storing the new upload under a unique variant of the name instead
    #    of deleting the existing document. Nothing in the knowledge base gets lost.
    storage_filename = get_unique_filename(cur, file.filename)
    cur.close()
    conn.close()

    temp_path = f"temp_{storage_filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(content)

    text = ocr_pdf(temp_path)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    store_chunks(storage_filename, chunks, embeddings, content_hash)

    os.remove(temp_path)

    return {"filename": storage_filename, "chunks_stored": len(chunks)}


class QueryRequest(BaseModel):
    question: str
    source_file: str | None = None  # optional — omit to search the whole knowledge base


@app.post("/query")
def query_documents(request: QueryRequest):
    result = agent_query(request.question, source_file=request.source_file)
    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"],
        "action_taken": result["action_taken"]
    }
    
@app.get("/documents")
def get_documents():
    from pipeline import list_documents
    return {"documents": list_documents()}