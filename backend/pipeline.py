import os
import io
import pymupdf as fitz
import pytesseract
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import psycopg2
import ollama
from dotenv import load_dotenv

load_dotenv()

# --- Load model once, reused everywhere ---
model = SentenceTransformer('intfloat/multilingual-e5-base')

DB_PASSWORD = os.getenv("DB_PASSWORD")


def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="document_query_db",
        user="postgres",
        password=DB_PASSWORD
    )


def ocr_pdf(pdf_path, lang="eng+nep"):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        text = pytesseract.image_to_string(image, lang=lang)
        full_text += f"\n--- Page {page_num + 1} ---\n{text}"
    return full_text


def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "। ", " ", ""]
    )
    return splitter.split_text(text)


def embed_chunks(chunks):
    prefixed_chunks = [f"passage: {chunk}" for chunk in chunks]
    return model.encode(prefixed_chunks)


def store_chunks(source_file, chunks, embeddings):
    conn = get_connection()
    cur = conn.cursor()
    for chunk, embedding in zip(chunks, embeddings):
        cur.execute(
            """
            INSERT INTO document_chunks (source_file, chunk_text, embedding)
            VALUES (%s, %s, %s)
            """,
            (source_file, chunk, embedding.tolist())
        )
    conn.commit()
    cur.close()
    conn.close()


def retrieve_chunks(query, top_k=5, source_file=None):
    query_embedding = model.encode(f"query: {query}").tolist()
    conn = get_connection()
    cur = conn.cursor()

    if source_file:
        cur.execute(
            """
            SELECT id, source_file, chunk_text, embedding <=> %s::vector AS distance
            FROM document_chunks
            WHERE source_file = %s
            ORDER BY distance ASC
            LIMIT %s
            """,
            (query_embedding, source_file, top_k)
        )
    else:
        cur.execute(
            """
            SELECT id, source_file, chunk_text, embedding <=> %s::vector AS distance
            FROM document_chunks
            ORDER BY distance ASC
            LIMIT %s
            """,
            (query_embedding, top_k)
        )

    results = cur.fetchall()
    cur.close()
    conn.close()
    return results


def generate_answer(query, retrieved_chunks):
    context = "\n\n".join([chunk_text for (_, _, chunk_text, _) in retrieved_chunks])
    prompt = f"""You are a helpful assistant answering questions based only on the provided context.
    If the answer isn't in the context, respond with "I don't know" (in English) or "मलाई थाहा छैन" (in Nepali) — matching the question's language. Never respond in Hindi under any circumstance.
    IMPORTANT: Always respond in the SAME language as the question — English question gets English answer, Nepali question gets Nepali answer. Never use Hindi.

Context:
{context}

Question: {query}

Answer:"""

    response = ollama.generate(
        model='llama3.2:3b',
        prompt=prompt,
        options={'num_predict': 200}  # allow up to 200 tokens in the response
)
    return response['response']