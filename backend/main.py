import shutil
import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from pipeline import ocr_pdf, chunk_text, embed_chunks, store_chunks, retrieve_chunks, generate_answer
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


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    from pipeline import get_connection
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM document_chunks WHERE source_file = %s", (file.filename,))
    conn.commit()
    cur.close()
    conn.close()

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = ocr_pdf(temp_path)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    store_chunks(file.filename, chunks, embeddings)

    os.remove(temp_path)

    return {"filename": file.filename, "chunks_stored": len(chunks)}


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