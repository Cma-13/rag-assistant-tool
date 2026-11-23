"""Adding a PDF to a user's knowledge base.

Used by the upload button (main.py, /upload) and by the Gmail MCP server's
import_email_attachment tool, so a PDF imported from an email is read, split,
embedded and stored exactly like an uploaded one.

Saving the file is quick; the slow part (reading the text, OCR, embedding) runs
in a background thread, so the server stays responsive and nobody waits on a big
PDF. The jobs are kept in memory: if the server restarts while a file is still
being read, that job is lost and the file simply has to be added again.
"""

import hashlib
import os
import threading
import time
import traceback
import uuid

from pipeline import (chunk_text_with_pages, db_cursor, embed_chunks, ocr_pdf,
                      save_document_info, store_chunks)

# Uploaded PDFs are kept here (one folder per user, named by the file's hash),
# so a source can be opened from the chat later.
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))  # biggest PDF accepted
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
UPLOAD_JOBS = {}              # job_id -> {user_id, filename, hash, status, chunks, error, created}
JOBS_LOCK = threading.Lock()  # protects UPLOAD_JOBS
READ_SLOT = threading.Semaphore(1)  # only one PDF is read at a time; the others wait their turn
JOB_KEEP_SECONDS = 3600       # finished jobs are forgotten after an hour


class UploadReadError(Exception):
    """A problem with the PDF itself that the user can understand (shown as it is)."""


class PdfRejected(Exception):
    """The file can't be added at all (too big, empty, not a PDF). status_code is
    the HTTP status the upload endpoint answers with."""

    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


def pdf_path(user_id, content_hash):
    return os.path.join(UPLOAD_DIR, str(user_id), f"{content_hash}.pdf")


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


def _name_is_taken(filename, user_id):
    """True if this user already has a stored document with this exact name."""
    with db_cursor() as cur:
        cur.execute("SELECT 1 FROM document_chunks WHERE source_file = %s AND user_id = %s LIMIT 1;", (filename, user_id))
        taken = cur.fetchone() is not None
    return taken


def job_view(job_id, job):
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
            page_stats = {}  # how each page was read (text layer or OCR), shown in the answer trace
            text = ocr_pdf(saved_path, stats=page_stats)
            chunk_pairs = chunk_text_with_pages(text)
            if not chunk_pairs:
                raise UploadReadError("No readable text was found in this PDF.")
            chunks = [c for c, _ in chunk_pairs]
            pages = [pg for _, pg in chunk_pairs]
            embeddings = embed_chunks(chunks)
            store_chunks(storage_filename, chunks, embeddings, content_hash, user_id, pages=pages)
            try:
                save_document_info(storage_filename, user_id, page_stats)
            except Exception as e:  # the document is stored; only the trace details are missing
                print(f"UPLOAD: could not save page details for '{storage_filename}' ({type(e).__name__}): {e}")
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


def add_pdf(content, upload_name, user_id):
    """Add one PDF (its bytes) to the user's knowledge base.

    Returns right away:
      {"status": "reading", "job_id", "filename"}  - being read in the background
      {"status": "done", "filename", "chunks_stored": 0, "message"}  - the same file is already there
    Raises PdfRejected if the file is too big, empty or not a PDF."""
    upload_name = os.path.basename((upload_name or "").replace("\\", "/")) or "document.pdf"
    if len(content) > MAX_UPLOAD_BYTES:
        raise PdfRejected(413, f"This file is too large. The limit is {MAX_UPLOAD_MB} MB.")
    if not content:
        raise PdfRejected(400, "This file is empty.")
    if b"%PDF-" not in content[:1024]:
        raise PdfRejected(400, "This is not a PDF file. Only PDF files can be uploaded.")
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
            "SELECT source_file FROM document_chunks WHERE content_hash = %s AND user_id = %s LIMIT 1",
            (content_hash, user_id)
        )
        row = cur.fetchone()
        if row:
            # the name it is stored under may differ from the name of this upload
            return {
                "status": "done",
                "filename": row[0],
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


def jobs_status(user_id, wanted=None):
    """How the background reading is going: the jobs asked for, or (no ids) all of
    this user's files that are still being read."""
    with JOBS_LOCK:
        if wanted:
            jobs = {}
            for job_id in wanted:
                job = UPLOAD_JOBS.get(job_id)
                if job and job["user_id"] == user_id:
                    jobs[job_id] = job_view(job_id, job)
                else:
                    # unknown: the server was restarted, or the job is too old
                    jobs[job_id] = {"job_id": job_id, "status": "unknown", "filename": None,
                                    "chunks": 0, "error": "This upload was interrupted. Please upload the PDF again."}
            return jobs
        return {j: job_view(j, v) for j, v in UPLOAD_JOBS.items()
                if v["user_id"] == user_id and v["status"] == "reading"}
