import shutil
import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from pipeline import ocr_pdf, chunk_text, embed_chunks, store_chunks, retrieve_chunks, generate_answer

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Document Query Tool is running"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run the full ingestion pipeline
    text = ocr_pdf(temp_path)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    store_chunks(file.filename, chunks, embeddings)

    # Clean up the temp file
    os.remove(temp_path)

    return {
        "filename": file.filename,
        "chunks_stored": len(chunks)
    }


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query_documents(request: QueryRequest):
    results = retrieve_chunks(request.question)
    answer = generate_answer(request.question, results)

    return {
        "question": request.question,
        "answer": answer,
        "sources": [source_file for (_, source_file, _, _) in results]
    }