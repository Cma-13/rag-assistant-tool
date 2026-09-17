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
Provide a complete, informative answer in at least one full sentence — do not just repeat the question's key term.
If the answer isn't in the context, say you don't know — do not make up information.

Context:
{context}

Question: {query}

Answer:"""
    response = ollama.generate(
        model='llama3.2:3b',
        prompt=prompt,
        options={'num_predict': 200}
    )
    return response['response']

def list_documents():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT source_file FROM document_chunks;")
    results = cur.fetchall()
    cur.close()
    conn.close()
    return [row[0] for row in results]

def agent_decide_action(query):
    """Decide the action: LIST via simple keyword rules, CHAT vs SEARCH via LLM."""
    query_lower = query.lower()

    # Rule-based LIST detection — reliable, no LLM needed
    list_keywords = ["what documents", "which documents", "what files", "which files",
                      "list documents", "list files", "documents have you", "documents do you"]
    if any(keyword in query_lower for keyword in list_keywords):
        return "LIST"

    # LLM-based CHAT vs SEARCH classification for everything else
    prompt = f"""You are an assistant that decides how to handle a user's message.
Respond with ONLY one word — no explanation, no punctuation:
- "SEARCH" if the message is a question that likely needs looking up information in uploaded documents.
- "CHAT" if the message is a greeting, small talk, or doesn't require document lookup at all.

Message: {query}

Answer:"""
    response = ollama.generate(model='llama3.2:3b', prompt=prompt, options={'num_predict': 10})
    decision = response['response'].strip().upper()

    if "CHAT" in decision:
        return "CHAT"
    else:
        return "SEARCH"  # default/fallback


def check_relevance(query, context):
    """Ask the LLM to verify if the context is relevant enough to attempt an answer."""
    prompt = f"""Does the context below contain information related to the question? Answer loosely - if there's any relevant connection, say YES.
Respond with ONLY one word: YES or NO.

Context:
{context}

Question: {query}

Answer:"""
    response = ollama.generate(model='llama3.2:3b', prompt=prompt, options={'num_predict': 10})
    decision = response['response'].strip().upper()
    return "NO" not in decision  # default to relevant unless explicitly told NO

def agent_query(query, source_file=None):
    action = agent_decide_action(query)

    if action == "LIST":
        docs = list_documents()
        if docs:
            answer = "Here are the documents you've uploaded: " + ", ".join(docs)
        else:
            answer = "No documents have been uploaded yet."
        return {"answer": answer, "sources": [], "action_taken": "LIST"}

    elif action == "CHAT":
        response = ollama.generate(
            model='llama3.2:3b',
            prompt=f"Respond naturally and briefly to this message: {query}",
            options={'num_predict': 100}
        )
        return {"answer": response['response'], "sources": [], "action_taken": "CHAT"}

    else:  # SEARCH
        results = retrieve_chunks(query, top_k=5, source_file=source_file)
        context = "\n\n".join([chunk_text for (_, _, chunk_text, _) in results])

        if not results or not check_relevance(query, context):
            return {
                "answer": "I don't know this doesn't appear to be covered in the uploaded document.",
                "sources": [],
                "action_taken": "SEARCH_NO_MATCH"
            }

        answer = generate_answer(query, results)
        sources = [source_file for (_, source_file, _, _) in results]
        return {"answer": answer, "sources": sources, "action_taken": "SEARCH"}