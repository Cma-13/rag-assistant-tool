import json
import os
import io
import pymupdf as fitz
import pytesseract
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import psycopg2
from dotenv import load_dotenv
import re
from concurrent.futures import ThreadPoolExecutor, Future
from llm import generate as llm_generate, groq_chat, generate_json, LLMBusyError, PROVIDER as LLM_PROVIDER
import pdfplumber

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


_NUMERIC_PIECE = re.compile(r"^[\d,.\-%]+$")


def _clean_cell(value):
    """One table cell as a single clean line. pdfplumber keeps the line breaks
    of wrapped text, so '81,900.0' + '0' is joined back into '81,900.00' and
    'Qt' + 'y' into 'Qt y'."""
    if value is None:
        return ""
    pieces = [p.strip() for p in str(value).split("\n") if p.strip()]
    if not pieces:
        return ""
    if len(pieces) > 1 and all(_NUMERIC_PIECE.match(p) for p in pieces):
        return "".join(pieces)
    return " ".join(pieces)


def _format_table(table):
    """Render a pdfplumber table as clean, pipe-delimited text so each
    number stays correctly tied to its row and column, rather than relying
    on OCR to visually reconstruct table alignment from a flattened image.

    Word/PDF tables with merged cells come out of pdfplumber with many empty
    cells and repeated labels. Columns that are empty in every row are dropped,
    rows that are completely empty are skipped, and repeated neighbours are
    collapsed, so the stored text is short and tidy."""
    grid = [[_clean_cell(c) for c in row] for row in table]
    grid = [row for row in grid if any(row)]
    if not grid:
        return ""

    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]
    keep = [j for j in range(width) if any(r[j] for r in grid)]

    # "Dense" rows are the real data rows (most cells filled). A column that is
    # empty in nearly all of them is a leftover of merged cells (a spacer), so
    # it is dropped from those rows, which keeps every value under the right
    # column. Short rows (titles, notes, the invoice-info block) are untouched.
    dense = [r for r in grid if sum(1 for j in keep if r[j]) >= 0.6 * len(keep)]
    spacers = set()
    if len(dense) >= 3:
        for j in keep:
            if sum(1 for r in dense if not r[j]) >= 0.9 * len(dense):
                spacers.add(j)
    dense_ids = {id(r) for r in dense}

    lines = []
    for r in grid:
        use = [j for j in keep if not (id(r) in dense_ids and j in spacers)]
        cells = [r[j] for j in use]
        filled = [c for c in cells if c]
        if len(filled) * 2 < len(cells):
            # mostly-empty row (title or merged heading): drop empties and repeats
            compact = []
            for c in filled:
                if not compact or compact[-1] != c:
                    compact.append(c)
            cells = compact
        lines.append(" | ".join(cells))
    return "\n".join(lines)


# Scanned/image pages are rendered to about this many pixels wide before OCR.
# Tesseract reads best at roughly this size: a fixed 4x zoom made a large page
# so huge that digits were misread (3 -> 5, 6 -> 0) and whole lines were lost,
# while a small A4 page still needs about 4x to reach it.
OCR_TARGET_WIDTH_PX = 2400
# Picture-only pages are read with OCR several at a time (Tesseract runs as a separate
# program, so the pages really are read side by side). Set OCR_WORKERS in .env to change it.
OCR_WORKERS = max(1, int(os.getenv("OCR_WORKERS", min(4, os.cpu_count() or 1))))


def _ocr_png(png_bytes, lang):
    """Read the text on one page picture."""
    return pytesseract.image_to_string(Image.open(io.BytesIO(png_bytes)), lang=lang)


def _page_text_outside_tables(page, boxes):
    """The page's text layer, leaving out words that sit inside a table (those
    cells are stored separately, cleanly, from the table itself). With no
    tables this is just the normal page text."""
    if not boxes:
        return page.get_text().strip()

    def inside(w):
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        return any(x0 <= cx <= x1 and top <= cy <= bottom for (x0, top, x1, bottom) in boxes)

    lines = {}
    for w in page.get_text("words"):
        if not inside(w):
            lines.setdefault((w[5], w[6]), []).append(w[4])
    return "\n".join(" ".join(ws) for _, ws in sorted(lines.items())).strip()


def _title_above(page, box, other_boxes):
    """The heading line sitting just above a table (for example 'Semester
    Summary'), so the table keeps its name after the page text is separated
    from the table cells. Empty if there is none."""
    x0, top, x1, bottom = box

    def in_any_table(w):
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        return any(a <= cx <= c and b <= cy <= d for (a, b, c, d) in other_boxes)

    lines = {}
    for w in page.get_text("words"):
        if w[3] <= top + 2 and w[3] >= top - 45 and not in_any_table(w):
            lines.setdefault((w[5], w[6]), []).append(w)
    if not lines:
        return ""
    nearest = max(lines.values(), key=lambda ws: max(w[3] for w in ws))
    title = " ".join(w[4] for w in sorted(nearest, key=lambda w: w[0]))
    return title if len(title) <= 80 else ""


_UTF16_RUN = re.compile(r"(?:[\x00-\x07][\x00-\xff])+")


def _clean_extracted_text(text):
    """Some PDFs store text as 2-byte characters (every letter has a hidden 0x00 in front,
    like "\\x00M\\x00i\\x00l"). Decode those runs back into normal text, so letters such as
    "ć" survive, and drop any NUL (0x00) that is left, because the database cannot store it."""
    def decode(match):
        run = match.group(0)
        try:
            return run.encode("latin-1").decode("utf-16-be")
        except UnicodeError:
            return run
    return _UTF16_RUN.sub(decode, text).replace("\x00", "")


def ocr_pdf(pdf_path, lang="eng"):
    """Read a scanned PDF (pages that are only images) by recognising the text with OCR."""
    doc = fitz.open(pdf_path)
    full_text = ""

    # Extract tables directly from the PDF's structure (not from an OCR'd
    # image), since this preserves exact row/column alignment even when
    # visual spacing is inconsistent. The table's position on the page is kept
    # too, so the same cells are not stored a second time as messy page text.
    tables_by_page = {}
    boxes_by_page = {}
    try:
        with pdfplumber.open(pdf_path) as pl_doc:
            for i, page in enumerate(pl_doc.pages):
                found = page.find_tables()
                tables = [t.extract() for t in found]
                good = [(t, f.bbox) for t, f in zip(tables, found) if t and any(any(c for c in row if c) for row in t)]
                if good:
                    tables_by_page[i] = [t for t, _ in good]
                    boxes_by_page[i] = [bb for _, bb in good]
    except Exception:
        tables_by_page, boxes_by_page = {}, {}

    # Pages with a real text layer are read directly. Picture-only pages go to OCR, several at
    # a time; each result is put back in its page position afterwards.
    if OCR_WORKERS > 1:
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # one Tesseract thread per page, the pages run side by side
    page_parts = []  # (page_num, text or Future, table text)
    pending = []
    with ThreadPoolExecutor(max_workers=OCR_WORKERS) as pool:
        for page_num, page in enumerate(doc):
            has_text_layer = bool(page.get_text().strip())
            native_text = _page_text_outside_tables(page, boxes_by_page.get(page_num, []))

            if has_text_layer:
                # (a page that is entirely a table leaves no loose text; that is fine,
                # it must not be sent to OCR)
                # Digitally-generated page with a real text layer: use it
                # directly. This is exact, unlike OCR, which reconstructs text
                # from pixels and can drop or scramble content.
                text = native_text
            else:
                # No embedded text (a genuinely scanned page) -> fall back to OCR.
                zoom = min(max(OCR_TARGET_WIDTH_PX / page.rect.width, 1), 4)
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                # Default page-segmentation mode: it also reads light text on
                # coloured bands (table headers, footers), which --psm 6 skipped.
                text = pool.submit(_ocr_png, pix.tobytes("png"), lang)
                pending.append(text)
                if len(pending) >= OCR_WORKERS * 2:
                    pending.pop(0).result()  # wait for the oldest, so pictures do not pile up in memory

            table_text = ""
            if page_num in tables_by_page:
                boxes = boxes_by_page[page_num]
                for t_idx, table in enumerate(tables_by_page[page_num]):
                    title = _title_above(page, boxes[t_idx], boxes) if has_text_layer else ""
                    label = f"Table {t_idx + 1} on Page {page_num + 1}" + (f" - {title}" if title else "")
                    table_text += f"\n\n[{label}]\n{_format_table(table)}"
            page_parts.append((page_num, text, table_text))

    full_text = ""
    for page_num, text, table_text in page_parts:
        if isinstance(text, Future):
            text = text.result()
        full_text += f"\n--- Page {page_num + 1} ---\n{text}{table_text}"

    return _clean_extracted_text(full_text)


_HEADING_RE = re.compile(r"^[A-Z][A-Z &/,\-]{3,40}$")


def _heading_of(line):
    """Return the heading text if this line looks like an ALL-CAPS section
    heading (ignoring stray bullet characters like '. ' from OCR)."""
    core = re.sub(r"^[^A-Za-z0-9]+", "", line.strip())
    return core if _HEADING_RE.match(core) else None


_PAGE_MARKER = re.compile(r"^--- Page (\d+) ---$")


def chunk_text_with_pages(text, chunk_size=500, chunk_overlap=50):
    """Split a document into small overlapping pieces, keeping each section heading with its text, so they can be searched.
    Returns a list of (chunk_text, page_number): the page that holds most of the chunk,
    read from the '--- Page N ---' markers that ocr_pdf puts in the text."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "\u0964 ", " ", ""]
    )

    # 1. Split the document into sections at heading lines, remembering which
    #    page each section starts on
    sections, heading, body = [], None, []
    current_page = 1
    section_page = 1
    for line in text.split("\n"):
        m = _PAGE_MARKER.match(line.strip())
        if m:
            current_page = int(m.group(1))
        h = _heading_of(line)
        if h:
            sections.append((heading, "\n".join(body), section_page))
            heading, body = h, []
            section_page = current_page
        else:
            body.append(line)
    sections.append((heading, "\n".join(body), section_page))

    # 2. Chunk each section on its own, keeping the heading with every chunk
    chunks = []
    for heading, body, start_page in sections:
        body = body.strip()
        if not body:
            continue

        # where each page marker sits inside this section's text
        markers, offset = [], 0
        for line in body.split("\n"):
            m = _PAGE_MARKER.match(line.strip())
            if m:
                markers.append((offset, int(m.group(1))))
            offset += len(line) + 1

        search_from = 0
        for piece in splitter.split_text(body):
            pos = body.find(piece, search_from)
            if pos == -1:
                pos = search_from
            else:
                search_from = pos + 1
            # A chunk can run over a page break: use the page that holds most of it.
            end = pos + len(piece)
            share = {}
            seg_start, seg_page = pos, start_page
            for marker_pos, marker_page in markers:
                if marker_pos <= pos:
                    seg_page = marker_page
                elif marker_pos < end:
                    share[seg_page] = share.get(seg_page, 0) + (marker_pos - seg_start)
                    seg_start, seg_page = marker_pos, marker_page
            share[seg_page] = share.get(seg_page, 0) + (end - seg_start)
            page = max(share, key=lambda pg: (share[pg], -pg))
            chunks.append((f"{heading}\n{piece}" if heading else piece, page))
    return chunks


def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """Same as chunk_text_with_pages, but only the chunk texts."""
    return [c for c, _ in chunk_text_with_pages(text, chunk_size, chunk_overlap)]


def embed_chunks(chunks):
    prefixed_chunks = [f"passage: {chunk}" for chunk in chunks]
    return model.encode(prefixed_chunks)


def store_chunks(source_file, chunks, embeddings, content_hash, user_id, pages=None):
    """Save the chunks. `pages` (optional) is the page number of each chunk."""
    if pages is None:
        pages = [None] * len(chunks)
    conn = get_connection()
    cur = conn.cursor()
    for chunk, embedding, page in zip(chunks, embeddings, pages):
        cur.execute(
            """
            INSERT INTO document_chunks (source_file, page_number, chunk_text, embedding, content_hash, user_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (source_file, page, chunk, embedding.tolist(), content_hash, user_id)
        )
    conn.commit()
    cur.close()
    conn.close()


def _keyword_query(text):
    """Turn a question into an OR-style keyword query, e.g.
    'What languages does Ramesh speak?' -> 'what | languages | does | ramesh | speak'.
    Postgres drops common stop words (what, does...) itself."""
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    words = [w for w in words if len(w) > 2]
    return " | ".join(dict.fromkeys(words))


def _as_scope(source_file):
    """A single file name, a list of names, or nothing -> a list of names (or None)."""
    if not source_file:
        return None
    if isinstance(source_file, str):
        return [source_file]
    return [f for f in source_file if f] or None


def retrieve_chunks(query, user_id, top_k=5, source_file=None, candidate_k=20):
    """Hybrid retrieval: semantic (vector) search + keyword (full-text) search,
    merged with Reciprocal Rank Fusion. Every returned row keeps its real
    vector distance, so the relevance thresholds in agent_query still work."""
    query_embedding = model.encode(f"query: {query}").tolist()
    keyword_q = _keyword_query(query)

    scope_sql = "user_id = %s"
    scope_params = [user_id]
    scope = _as_scope(source_file)  # one file name, or several
    if scope:
        scope_sql += " AND source_file = ANY(%s)"
        scope_params.append(scope)

    conn = get_connection()
    cur = conn.cursor()

    # 1. Semantic candidates (by meaning)
    cur.execute(
        f"""
        SELECT id, source_file, chunk_text, embedding <=> %s::vector AS distance
        FROM document_chunks
        WHERE {scope_sql}
        ORDER BY distance ASC
        LIMIT %s
        """,
        [query_embedding] + scope_params + [candidate_k]
    )
    semantic = cur.fetchall()

    # 2. Keyword candidates (by actual words)
    keyword = []
    if keyword_q:
        cur.execute(
            f"""
            SELECT id, source_file, chunk_text, embedding <=> %s::vector AS distance
            FROM document_chunks
            WHERE {scope_sql}
              AND to_tsvector('english', chunk_text) @@ to_tsquery('english', %s)
            ORDER BY ts_rank(to_tsvector('english', chunk_text), to_tsquery('english', %s)) DESC
            LIMIT %s
            """,
            [query_embedding] + scope_params + [keyword_q, keyword_q, candidate_k]
        )
        keyword = cur.fetchall()

    cur.close()
    conn.close()

    # 3. Merge the two ranked lists (Reciprocal Rank Fusion)
    RRF_K = 60
    scores = {}
    rows = {}
    for rank, row in enumerate(semantic):
        scores[row[0]] = scores.get(row[0], 0) + 1 / (RRF_K + rank + 1)
        rows[row[0]] = row
    for rank, row in enumerate(keyword):
        scores[row[0]] = scores.get(row[0], 0) + 1 / (RRF_K + rank + 1)
        rows[row[0]] = row

    best_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [rows[i] for i in best_ids]

_MORE_DETAIL = ("The user wants a fuller answer than before: give a more detailed answer in several sentences, "
                "covering every relevant detail in the context.")
_AN_EXAMPLE = ("The user wants an example: give one concrete example taken from the context (a specific fact, figure or case), "
               "in two or three sentences.")


def generate_answer(query, retrieved_chunks, elaborate=False):
    """Write the answer to a question using only the retrieved document passages."""
    # Group chunks by document (each document's chunks stay in original order)
    # and label each group, so the model can tell which document a fact is from.
    by_doc = {}
    for (chunk_id, source_file, chunk_text, _) in sorted(retrieved_chunks, key=lambda x: x[0]):
        by_doc.setdefault(source_file, []).append(chunk_text)

    context = "\n\n".join(
        f"[Document: {doc}]\n" + "\n\n".join(chunks)
        for doc, chunks in by_doc.items()
    )

    more_detail = ((elaborate if isinstance(elaborate, str) else _MORE_DETAIL) + "\n\n") if elaborate else ""

    prompt = f"""You are a helpful assistant answering questions based only on the provided context.
The context is split into labeled documents. Details from one document must never be mixed into facts from another.
Only state facts that are explicitly and directly written in the context. Do not infer causes, combine unrelated events, or guess at relationships between events that aren't clearly stated.
Provide a complete, informative answer in at least one full sentence - do not just repeat the question's key term.
If the answer isn't clearly stated in the context, say you don't know - do not make up information.
Copy numbers, IDs, dates and account numbers exactly as they are written in the context - never reformat them, add spaces to them, or recalculate them.
If the user asks about you (who you are, what you can do), say briefly that you are a document Q&A assistant that answers questions from the PDFs they upload. Never describe what the documents say as your own skills, knowledge or experience.
Never mention labels such as 'Table 1 on Page 1' in your answer. If the question needs adding up or comparing values from several rows or tables and that total is not written in the context, say it is not stated. The same goes for counting: if the question asks how many items or rows there are and that number is not written in the context, say it is not stated.
Answer only if the context directly states the thing asked about. Do not guess from related details: a phone number of the seller is not the buyer's, and a black telephone does not tell you the colour of a door. If the context is about a different person, company or thing than the one asked about, say it is not stated.
Write currency exactly as it appears in the context (for example Rs. or NPR); never change it to another symbol such as the rupee sign or a dollar sign.

{more_detail}Context:
{context}

Question: {query}

Answer:"""
    return llm_generate(prompt, num_predict=350 if elaborate else 200, temperature=0.1)


def list_documents(user_id):
    """List the names of the PDFs this user has uploaded, the newest first."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT source_file FROM document_chunks WHERE user_id = %s "
        "GROUP BY source_file ORDER BY MAX(id) DESC;",
        (user_id,),
    )
    results = cur.fetchall()
    cur.close()
    conn.close()
    return [row[0] for row in results]


def get_document_chunks_in_order(source_file, user_id, limit=25):
    """Fetch chunks in original document order, for summarization (not similarity-based)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT chunk_text FROM document_chunks
        WHERE source_file = %s AND user_id = %s
        ORDER BY id ASC
        LIMIT %s
        """,
        (source_file, user_id, limit)
    )
    results = cur.fetchall()
    cur.close()
    conn.close()
    return [row[0] for row in results]


def generate_summary(source_file, user_id):
    """Generate a broad summary using chunks spread across the document."""
    chunks = get_document_chunks_in_order(source_file, user_id, limit=40)
    if not chunks:
        return "I don't have any content to summarize yet please upload a document first."

    context = "\n\n".join(chunks)[:18000]
    prompt = f"""Summarize the following document content in 3-5 sentences, covering its main topics and purpose.
Use only facts written in the text. Do not guess who people are or how they are related unless the text says so.

Document content:
{context}

Summary:"""
    return llm_generate(prompt, num_predict=300, temperature=0.2)


# ---------------------------------------------------------------------------
# Structured output: tables and key-value pairs
# ---------------------------------------------------------------------------

# Explicit requests for table / key-value style output. Kept deliberately
# narrow so a normal question like "What is a table in PostgreSQL?" still goes
# to SEARCH. To make the tool recognise more phrasings, add them here.
_TABLE_REQUEST = re.compile(
    r"\b(tabular|key[\s\-]?value)\b"
    r"|\b(in|as|into)\s+(a\s+|the\s+)?(table|tabular)\b"
    r"|\b(show|give|display|present|format|put|extract|convert|turn|return|list|get|pull|fetch|generate|create|make|build|produce|prepare|print|output|tabulate)\b.{0,40}\b(table|tables|columns?|rows?)\b"
    r"|\b(complete|full|whole|entire)\s+table\b"
    r"|\b(show|list|display|give|get|find|fetch)\b(?!.{0,40}\b(documents?|files?|pdfs?)\b).{0,60}\b(where|whose|with|having)\b"
    r"|\b(show|list|display|give|get|find|fetch)\b(?!.{0,40}\b(documents?|files?|pdfs?|answers?|explain\w*|summar\w*)\b)"
    r".{0,60}\b(only|and above|and below|or above|or below|more than|less than|greater than|higher than|lower than|at least|at most|above|below|between)\b"
    r"|\bextract\b.{0,40}\b(data|details|fields|values|information)\b"
    r"|\b(invoice|bill|receipt|statement)\s+(details|data|summary|fields)\b",
    re.IGNORECASE)

# The shape Groq is forced to answer in. Values are strings on purpose, so
# numbers keep their exact original formatting (commas, decimals, symbols).
_STRUCTURED_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
                "additionalProperties": False,
            },
        },
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                },
                "required": ["title", "columns", "rows"],
                "additionalProperties": False,
            },
        },
        "note": {"type": "string"},
    },
    "required": ["pairs", "tables", "note"],
    "additionalProperties": False,
}

_STRUCTURED_PROMPT = """You extract structured data from a document and return it as JSON.

The JSON has three parts:
- "pairs": a list of {"key": ..., "value": ...} for single labeled facts (for example Invoice Number, Date, Total).
- "tables": a list of tables. Each table has a "title", a list of "columns" (the headers) and "rows" (each row is a list of cell values, in the same order as the columns).
- "note": one short sentence only if something is missing, unclear, or nothing relevant was found. Otherwise an empty string.

Rules:
1. Use ONLY information written in the document below. Never use outside knowledge and never invent values.
2. Copy every value EXACTLY as written, including currency symbols, commas, decimals, percentages and units. Never calculate, round, convert or "fix" a number. If a cell is empty or missing, use "".
3. Follow what the user asked for:
   - Whole table(s) requested: return the complete table with every row and every column.
   - Only certain rows, columns or values requested: return ONLY those, keeping the original column names.
   - Key-value pairs, details or fields requested: put them in "pairs". Also include a table when there is a list of items (for example line items).
   - A general request such as "extract the data": return the main details in "pairs" and every table in "tables".
4. Blocks that start with [Table N on Page M - heading] hold cells separated by " | " and are the most reliable source for table structure, and the heading (when present) is the table's name, so use it to find the table the user means. The other text can be less tidy, and some text can repeat where the document was split into pieces. Ignore the repeats.
5. If nothing relevant is found, return empty "pairs" and empty "tables" and explain in "note". When the user names a particular table or topic (for example "consumption history" or "slab calculation") and this text has no such table, return nothing - never return a different table instead.
5b. Give every table a short, plain title that says what it holds (for example "Invoice Items"). Never use a title like "Table 2 on Page 1".
6. Use an empty list for "pairs" or "tables" when it is not needed.
6b. Never give the same data twice. If you put values in "pairs", do not repeat them in a table, and the other way round.
7. Return ONLY what was asked. If the user asked for certain rows or columns, or for a table of items, "pairs" must be empty and there must be no extra summaries or other tables. Never put the file name in a title.
7b. Unless the user asked for specific columns, keep EVERY column of the original table, using the document's own column names. Every row must have a value for every column, in the right place.
8. Leave "note" empty when you found what was asked for. This text may be only one part of a longer document, so never say that something is missing just because it is not in this part.

<<COLUMNS>>Document:
<<DOCUMENT>>

User request: <<REQUEST>>"""

# Groq's free tier refuses any single request over 8,000 tokens (reading + the
# room kept for the answer) and allows about 8,000 tokens per minute in total.
# So each piece of a document is kept small, and the answer is capped too.
MAX_TABLE_CONTEXT_CHARS = 8000
MAX_TABLE_ANSWER_TOKENS = 3000


# Words that say what KIND of file it is, not which file. A document called
# "invoice.pdf" must not be picked just because the question says "invoice".
_GENERIC_NAME_WORDS = {
    "invoice", "invoices", "bill", "bills", "receipt", "receipts", "report",
    "document", "documents", "doc", "docs", "file", "files", "pdf", "copy",
    "final", "new", "sample", "data", "page", "table", "the", "and", "for",
}


def _documents_named_in_query(query, docs):
    """Documents whose name the user actually typed: the full filename, or the
    name without .pdf (unless that name is only a generic word like 'invoice')."""
    q = query.lower()
    named = []
    for doc in docs:
        stem = os.path.splitext(doc)[0].lower().strip()
        words = re.findall(r"[a-z0-9]+", stem)
        generic_only = all(w in _GENERIC_NAME_WORDS for w in words)
        if doc.lower() in q or (len(stem) >= 3 and not generic_only and stem in q):
            named.append(doc)
    return named


_TABLE_WORDS = set("""show me give list display extract get put make as a an the table tables key details detail values
value numbers number all of in into rows row columns column from this that document file pdf please can you""".split())


def _table_topic(query):
    """The topic of a table request ('temperatures, CO2 levels, sea rise'), without the
    words that only ask for a table. The search uses this, so the word 'table' does
    not pull in documents that merely contain tables."""
    words = [w for w in re.findall(r"[A-Za-z0-9]+", query) if w.lower() not in _TABLE_WORDS]
    return " ".join(words) if len(words) >= 2 else query


def _pick_documents_for_query(query, user_id, source_file, max_docs=3, recent_doc=None):
    """Work out which document(s) the user means: an explicit selection first,
    then filenames typed in the question, then the documents the search finds
    the most relevant chunks in. When the question could fit several documents
    (for example two invoices), return up to max_docs so each gets its own
    labelled result instead of silently guessing one."""
    scope = _as_scope(source_file)
    if scope:
        return scope

    named = _documents_named_in_query(query, list_documents(user_id))
    if named:
        return named[:max_docs]

    results = retrieve_chunks(_table_topic(query), user_id, top_k=8)
    if not results:
        return []

    order = list(dict.fromkeys(row[1] for row in results))  # best-ranked first
    counts = {}
    for row in results:
        counts[row[1]] = counts.get(row[1], 0) + 1
    if recent_doc in counts and counts[recent_doc] >= max(counts.values()):
        return [recent_doc]  # the document we are talking about wins ties
    ranked = sorted(order, key=lambda d: -counts[d])  # stable: ties keep search order
    # Keep documents with more than half as many matching chunks as the best one
    return [d for d in ranked if counts[d] > counts[ranked[0]] / 2][:max_docs]


MAX_TABLE_WINDOWS = 8


def _document_windows(source_file, user_id):
    """The document's text in original order, cut into pieces that each fit one
    request. A short document is a single piece. A long one (like a 120-row
    table) is read piece by piece and the results are combined afterwards, so
    nothing is skipped just because the document is big.

    Returns (pieces, was_cut_short)."""
    texts = get_document_chunks_in_order(source_file, user_id, limit=5000)
    if not texts:
        return [], False

    # Later pieces would not contain the table's header row, so remind the
    # model of it (the first line after the first table tag).
    header = ""
    for t in texts:
        if "[Table " in t:
            after = t.split("]", 1)[-1].strip().split("\n")
            header = after[0] if after else ""
            break

    pieces, current, used = [], [], 0
    for t in texts:
        if current and used + len(t) > MAX_TABLE_CONTEXT_CHARS:
            pieces.append("\n\n".join(current))
            current, used = [], 0
        current.append(t)
        used += len(t)
    if current:
        pieces.append("\n\n".join(current))

    cut_short = len(pieces) > MAX_TABLE_WINDOWS
    pieces = pieces[:MAX_TABLE_WINDOWS]
    if header and len(pieces) > 1:
        pieces = [pieces[0]] + [f"[The table's header row is: {header}]\n\n{x}" for x in pieces[1:]]
    return pieces, cut_short


def _merge_structured(parts):
    """Combine the results from several pieces of one document: tables with the
    same number of columns are joined into one table (repeated rows removed),
    and repeated key-value pairs are removed."""
    pairs, seen_pairs = [], set()
    tables = []
    notes = []
    for part in parts:
        for pr in part["pairs"]:
            k = (pr["key"].lower(), pr["value"])
            if k not in seen_pairs:
                seen_pairs.add(k)
                pairs.append(pr)
        for t in part["tables"]:
            match = next((m for m in tables if len(m["columns"]) == len(t["columns"])), None)
            if match is None:
                tables.append({"title": t["title"], "columns": list(t["columns"]), "rows": [], "_seen": set()})
                match = tables[-1]
            for row in t["rows"]:
                if tuple(row) not in match["_seen"]:
                    match["_seen"].add(tuple(row))
                    match["rows"].append(row)
        if part["note"] and part["note"] not in notes:
            notes.append(part["note"])
    for t in tables:
        t.pop("_seen")
    # A "nothing found" note from one piece is noise when another piece found the data.
    note = "" if (pairs or tables) else " ".join(notes)
    return {"pairs": pairs, "tables": tables, "note": note}


def _clean_structured(data):
    """Tidy the model's JSON: strip whitespace, make every row exactly as long
    as its header, and drop empty rows/pairs, so the display never breaks."""
    if not isinstance(data, dict):
        raise ValueError("structured output was not a JSON object")

    pairs = []
    for p in data.get("pairs") or []:
        key = str(p.get("key", "")).strip()
        value = str(p.get("value", "")).strip()
        if key or value:
            pairs.append({"key": key, "value": value})

    tables = []
    for t in data.get("tables") or []:
        columns = [str(c).strip() for c in (t.get("columns") or [])]
        rows = []
        for r in t.get("rows") or []:
            cells = [str(c).strip() for c in r]
            if columns:
                cells = (cells + [""] * len(columns))[:len(columns)]
            if any(cells):
                rows.append(cells)
        if columns or rows:
            tables.append({"title": str(t.get("title", "")).strip(), "columns": columns, "rows": rows})

    return {"pairs": pairs, "tables": tables, "note": str(data.get("note") or "").strip()}


def extract_structured(query, document_text, columns_hint=None):
    """Ask the LLM to pull key-value pairs and/or tables out of the document,
    shaped by what the user asked for."""
    hint = ""
    if columns_hint:
        hint = ("This text is one part of a longer document. If you return a table, use EXACTLY these column names "
                "in this order, so the parts can be joined: " + " | ".join(columns_hint) + "\n\n")
    prompt = (_STRUCTURED_PROMPT.replace("<<COLUMNS>>", hint)
              .replace("<<DOCUMENT>>", document_text).replace("<<REQUEST>>", query))
    return _clean_structured(generate_json(prompt, _STRUCTURED_SCHEMA, max_tokens=MAX_TABLE_ANSWER_TOKENS))


def _structured_to_text(structured):
    """Plain-text version of the structured result (key: value lines and
    markdown-style tables), used as the chat answer text."""
    parts = []
    if structured["pairs"]:
        parts.append("\n".join(f"{p['key']}: {p['value']}" for p in structured["pairs"]))

    for t in structured["tables"]:
        lines = []
        if t["title"]:
            lines.append(t["title"])
        if t["columns"]:
            lines.append("| " + " | ".join(t["columns"]) + " |")
            lines.append("| " + " | ".join("---" for _ in t["columns"]) + " |")
        for row in t["rows"]:
            lines.append("| " + " | ".join(row) + " |")
        parts.append("\n".join(lines))

    if structured["note"]:
        parts.append(structured["note"])

    if not parts:
        return "I couldn't find any table or structured data for that in the document."
    return "\n\n".join(parts)


# Questions about the assistant itself ("who are you?", "what can you do?").
# Searching the PDFs would only mix document contents into the answer, so these
# are treated as plain conversation (CHAT) instead. To catch another phrasing,
# add it here.
_ABOUT_ASSISTANT = re.compile(
    r"\bwhat\s+(all\s+)?(can|could)\s+you\s+(do|help)\b"
    r"|\bwhat\b.{0,30}\byou\s+(can|could|are\s+able\s+to|are\s+capable\s+of)\s+(do|help)\b"
    r"|\bthings\s+(that\s+)?you\s+(can|could)\b"
    r"|\bwhat\s+are\s+you(r)?\s+(capabilit\w+|features?|abilit\w+|able\s+to|purpose)\b"
    r"|\byour\s+(capabilit\w+|features?|abilit\w+|purpose)\b"
    r"|\bwhat\s+(capabilit\w+|features?|abilit\w+)\s+(do\s+you|does\s+this\s+(tool|assistant|app|system))\s+have\b"
    r"|\bwhat\s+(do|can)\s+you\s+(offer|provide)\b"
    r"|\bwho\s+are\s+you\b|\bwhat\s+are\s+you\b"
    r"|\bwhat\s+(do|does)\s+(you|this\s+(tool|assistant|app|system))\s+do\b(?!\s+for\b)"
    r"|\bwhat\s+(can|could)\s+(this|the)\s+(tool|assistant|app|system)\s+do\b"
    r"|\bwhat\s+is\s+(this|your)\s+(tool|assistant|app|system)\b"
    r"|\btell\s+me\s+about\s+(yourself|this\s+(tool|assistant|app|system))\b"
    r"|\bintroduce\s+yourself\b|\bhow\s+can\s+you\s+help\b",
    re.IGNORECASE)

# What the assistant knows about itself when chatting. Edit this line when the
# tool gains a big new ability.
ASSISTANT_BACKGROUND = ("You are DocQuery, a document Q&A tool. You answer questions using the PDFs the user uploads "
                        "(only what is written in them), you can read scanned PDFs, you can pull tables and key details "
                        "out of documents and show them as tables, and you show which document an answer came from.")


GREETING_KEYWORDS = {
    "hi", "hey", "hello", "hellooo", "hii", "yo", "sup",
    "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "ok", "okay", "bye", "goodbye",
}


# "list all documents", "show my files" (the keyword list above misses these).
_LIST_REQUEST = re.compile(
    r"\b(list|show)\s+(me\s+)?(all\s+)?(the\s+|my\s+|uploaded\s+)*(documents|files|pdfs)\b", re.IGNORECASE)

# "summarize the story / this document / the pdf" - a summary of ONE whole file.
# A topic summary like "summarize the Rana regime" is NOT matched; it is a normal question.
_SUMMARY_REQUEST = re.compile(
    r"\b(summari[sz]e|summary\s+of|overview\s+of|gist\s+of)\b.{0,25}"
    r"\b(document|pdf|file|story|article|report|text|paper|manual|guide|agreement|interview|invoice|bill)\b",
    re.IGNORECASE)

# "give me an overview of all these documents" - one short summary per file.
_OVERVIEW_ALL = re.compile(
    r"\b(overview|summary|summari[sz]e|describe|tell\s+me\s+about)\b.{0,40}\b(all|each|every)\b.{0,20}\b(documents?|files?|pdfs?)\b",
    re.IGNORECASE)


# "these two documents", "the file I just uploaded": the files added most recently.
_ABOUT_UPLOADS = re.compile(
    r"\b(?:these|those|both)\s+(?:(?:two|three|four|five|six|\d+)\s+)?(?:new\s+)?(?:documents?|docs?|files?|pdfs?)\b"
    r"|\bjust\s+(?:uploaded|added|attached|gave|given|give|sent|shared|provided|put)\b"
    r"|\b(?:i|we)(?:\s+have|\s+had|'ve)?\s+(?:just\s+)?(?:uploaded|attached|gave|given|sent|shared|provided|added)\b"
    r"|\bnew(?:ly)?\s+(?:uploaded\s+)?(?:documents?|files?|pdfs?)\b",
    re.IGNORECASE)
_SUMMARY_WORDS = re.compile(r"\b(summari[sz]e|summary|overview|gist)\b", re.IGNORECASE)


def agent_decide_action(query):
    """Decide the action: LIST/SUMMARY/TABLE/CHAT via keyword rules first, CHAT vs SEARCH via LLM otherwise."""
    query_lower = query.lower()

    if query_lower.strip("!.? ") in GREETING_KEYWORDS:
        return "CHAT"

    if _ABOUT_ASSISTANT.search(query):
        return "CHAT"

    # "my name is Seema" / "what's my name?" are about the conversation, not the documents
    if (_is_user_fact(query) or _ABOUT_CONVERSATION.search(query)
            or (_ABOUT_ME.search(query) and not _DOC_WORDS.search(query))):
        return "CHAT"

    list_keywords = ["what documents", "which documents", "what files", "which files",
                      "list documents", "list files", "documents have you", "documents do you"]
    if any(keyword in query_lower for keyword in list_keywords) or _LIST_REQUEST.search(query):
        return "LIST"

    if _OVERVIEW_ALL.search(query):
        return "OVERVIEW"

    summary_keywords = ["tell me about this document", "tell me about the document",
                         "what is this document about", "what's this document about",
                         "summarize this document", "summarize the document",
                         "give me a summary", "what does this document cover",
                         "tell me about the pdf", "tell me about this pdf"]
    if any(keyword in query_lower for keyword in summary_keywords) or _SUMMARY_REQUEST.search(query):
        return "SUMMARY"

    if _TABLE_REQUEST.search(query):
        return "TABLE"

    prompt = f"""You are a router deciding whether a user's message needs the uploaded documents, or is just conversation.
Respond with ONLY one word - no explanation, no punctuation.

Rules:
- Any question asking for a definition, fact, explanation, or a "what/who/when/where/why/how" question about any topic is SEARCH - even if you personally already know the answer. The user wants it grounded in THEIR documents, not your own knowledge.
- CHAT is ONLY for greetings, small talk, thanks, or messages that are not really questions at all.

Examples:
"What is PostgreSQL used for?" -> SEARCH
"How's it going?" -> CHAT
"What is RAG?" -> SEARCH
"Thanks so much!" -> CHAT

Message: {query}

Answer:"""
    decision = llm_generate(prompt, num_predict=10, temperature=0.1).strip().upper()

    if "CHAT" in decision:
        return "CHAT"
    else:
        return "SEARCH"


def check_relevance(query, context):
    """Ask the LLM to verify if the context is relevant enough to attempt an answer."""
    prompt = f"""Does the context below contain information related to the question? Answer loosely - if there's any relevant connection, say YES.
Respond with ONLY one word: YES or NO.

Context:
{context}

Question: {query}

Answer:"""
    decision = llm_generate(prompt, num_predict=10, temperature=0.1).strip().upper()
    return "NO" not in decision  # default to relevant unless explicitly told NO

# Messages this short/simple are almost never a follow-up needing rewriting,
# and sending them through the rewrite model risks corrupting a harmless
# greeting into something that gets misrouted. Skip rewriting for these.
SKIP_REWRITE_PATTERNS = [
    "hi", "hey", "hello", "hellooo", "hii", "yo", "sup",
    "thanks", "thank you", "ok", "okay", "bye", "goodbye",
]


# Only messages containing a reference word (it, that, this, he, his, "shorter",
# "and ...", etc.) can be follow-ups that need rewriting. Anything else is
# already standalone, and rewriting it only risks pulling in the previous topic.
_FOLLOWUP_HINTS = re.compile(
    r"\b(it|its|that|this|these|those|they|them|their|he|him|his|she|her|hers|"
    r"the above|the previous|the same|same one|former|latter|"
    r"shorter|longer|simpler|briefer|elaborate|rephrase|again|more|less|"
    r"i mean|i meant)\b"
    r"|^(and|also|what about|how about)\b",
    re.IGNORECASE)

# Messages that are clearly follow-up requests about the previous answer.
# The CHAT/SEARCH classifier can't see the topic in these, so it calls them CHAT.
_FOLLOWUP_REQUESTS = re.compile(
    r"\b(explain|simplif\w*|shorter|longer|elaborate|rephrase|summari[sz]e|clarify|"
    r"more detail|in one sentence|i mean|i meant|why|tell me more|go on|continue|an example|for example)\b"
    r"|^(and|also|what about|how about)\b",
    re.IGNORECASE)


def rewrite_query(query, history):
    """Rewrite a follow-up into a standalone question using the most recent
    turns. Messages with no follow-up reference words are returned unchanged."""
    if not history:
        return query

    stripped = query.strip().lower().strip("!.? ")
    if stripped in SKIP_REWRITE_PATTERNS or len(stripped) <= 2:
        return query

    # Already standalone (no vague reference) -> leave it alone
    if not (_FOLLOWUP_HINTS.search(query.strip()) or _SHORT_FOLLOWUP.match(stripped)):
        return query

    recent = history[-2:]
    history_text = "\n".join(
        f"Q: {turn['question']}\nA: {turn['answer']}" for turn in recent
    )

    prompt = f"""Given the MOST RECENT exchange below and a new message, rewrite the new message into a fully standalone question, using ONLY the most recent exchange to resolve vague words (like "that", "it", "this", "he", "she"). Vague references point to the LAST topic discussed, not anything earlier.
Change ONLY the vague words. Never add topics, names, or qualifiers that are not in the new message.
If the new message is already standalone, a greeting, or doesn't reference anything earlier, return it EXACTLY unchanged.
Respond with ONLY the rewritten (or unchanged) question - no explanation, no quotes.

Most recent exchange:
{history_text}

New message: {query}

Standalone question:"""

    rewritten = llm_generate(prompt, num_predict=60, temperature=0.1).strip().strip('"')
    return rewritten if rewritten else query

# ── Conversation memory ──────────────────────────────────────────────────
# Small facts the user states about themselves in the chat ("my name is Seema",
# "I only care about the bills", "call me Sam") are picked up with simple rules,
# so the assistant can use them later in the same chat. Nothing is stored
# anywhere else: they are re-read from the saved chat on every question.
_USER_FACT = re.compile(
    r"^\s*(?:please\s+)?(?:"
    r"remember\b|note\s+that\b|keep\s+in\s+mind\b|from\s+now\s+on\b|call\s+me\b"
    r"|my\s+\w+(?:\s+\w+)?\s+(?:is|are|was)\b"
    r"|i(?:\s+am|'m)\s+called\b|i(?:\s+am|'m)\s+(?!(?:looking|trying|searching|asking|wondering|going|getting|having|unable|not|here|confused)\b)\w+"
    r"|i\s+(?:work|study|live|teach|prefer|like|love|use)\b"
    r"|i\s+(?:only\s+)?care\s+(?:about|for)\b|i\s+only\b|i(?:\s+am|'m)\s+focus\w*\b"
    r")",
    re.IGNORECASE)

# Questions about the conversation itself ("what's my name?", "what did I ask?").
_ABOUT_CONVERSATION = re.compile(
    r"\bwho\s+am\s+i\b|\bwho\s+i\s+am\b|\bwhat\s+do\s+you\s+know\s+about\s+me\b"
    r"|\b(?:were|are|did|do)\s+you\s+(?:able\s+to\s+)?(?:answer|understand)\b"
    r"|\bwhat\s+(?:are|were)\s+we\s+(?:talking|discussing|speaking)\s+about\b"
    r"|\bwhich\s+(?:document|file|pdf)\s+(?:are|were)\s+we\b"
    r"|\bdo\s+you\s+(?:remember|know)\s+(?:my|me|what\s+i)\b"
    r"|\bwhat\s+did\s+(?:i|we)\s+(?:say|ask|talk|discuss|mention)\b"
    r"|\bwhat\s+(?:did\s+you|have\s+you)\s+(?:say|said|tell|told)\b"
    r"|\bdid\s+i\s+(?:tell|mention|say|ask)\b"
    r"|\bwhat\s+(?:have\s+)?we\s+(?:talked|discussed|covered)\b"
    r"|\bwhat\s+was\s+my\s+(?:last|first|previous)\s+(?:question|message)\b"
    r"|\bremind\s+me\s+what\b",
    re.IGNORECASE)


# "my name", "about me": about the user, unless the question is about a document
# ("what is my name on the marksheet?").
_ABOUT_ME = re.compile(r"\bmy\s+name\b|\babout\s+me\b", re.IGNORECASE)
_DOC_WORDS = re.compile(r"\b(document|doc|pdf|file|invoice|bill|marksheet|certificate|resume|cv|report|page)\b", re.IGNORECASE)


def _is_user_fact(text):
    t = (text or "").strip()
    return bool(t) and len(t) <= 200 and not t.endswith("?") and bool(_USER_FACT.match(t))


def extract_notes(user_messages, max_notes=8):
    """The facts the user stated about themselves, newest last (at most max_notes)."""
    notes = []
    for text in user_messages or []:
        if _is_user_fact(text):
            note = " ".join(str(text).split())[:150]
            if note not in notes:
                notes.append(note)
    return notes[-max_notes:]


def _trim_history(history, max_turns=6):
    """Keep the last few exchanges, with long questions/answers cut short so the
    prompts stay small (the Groq free tier has a small token limit)."""
    trimmed = []
    for turn in (history or [])[-max_turns:]:
        item = dict(turn)
        item["question"] = str(turn.get("question", ""))[:300]
        item["answer"] = str(turn.get("answer", ""))[:600]
        trimmed.append(item)
    return trimmed


def _chat_context(history, notes):
    """The earlier conversation, as text for a prompt."""
    parts = []
    if notes:
        parts.append("Things the user told you earlier in this conversation:\n- " + "\n- ".join(notes))
    if history:
        turns = "\n".join(f"User: {t['question']}\nAssistant: {t['answer'][:300]}" for t in history[-4:])
        parts.append("Recent conversation:\n" + turns)
        used = list(dict.fromkeys(src for t in history[-4:] for src in (t.get("sources") or [])))
        if used:
            parts.append("Documents used in the recent answers: " + ", ".join(used))
    return "\n\n".join(parts)


def _recent_source_doc(history):
    """The one document the recent answers came from, or None when it is unclear."""
    for turn in reversed((history or [])[-4:]):
        sources = list(dict.fromkeys(turn.get("sources") or []))
        if sources:
            return sources[0] if len(sources) == 1 else None
    return None


# "this document", "this one", "the file we are talking about" ...
_DEICTIC = re.compile(
    r"\b(?:this|that|the|current|same)\s+(?:\w+\s+)?(?:document|doc|file|pdf|one|story)\b"
    r"|\bwe\s+(?:are|were)\s+(?:currently\s+)?(?:talking|discussing)\b|\bcurrently\b",
    re.IGNORECASE)

_ASK_WHICH = "Which document would you like summarized?"


# ── Follow-ups ─────────────────────────────────────────────────────────────
# "make it shorter", "in simple words", "as bullet points": change the FORM of the
# previous answer. No new search is needed, only the previous answer is reworded.
_REWORK = re.compile(
    r"\b(shorter|shorten|briefer|brief(?:ly)?|simpler|simplify|simple\s+(?:words|terms|english|language)|"
    r"in\s+(?:one|a)\s+(?:sentence|line|word)|bullet(?:s|\s+points?)?|in\s+points|rephrase|reword|rewrite|"
    r"say\s+(?:that|it)\s+again|translate)\b", re.IGNORECASE)

# "make it longer", "tell me more", "give an example": a fuller answer is wanted,
# so the previous question is searched again, asking for more detail.
_ELABORATE = re.compile(
    r"\b(longer|more\s+detail(?:s|ed)?|in\s+detail|detailed|elaborate|expand|tell\s+me\s+more|"
    r"explain\s+(?:it\s+)?more|more\s+about\s+(?:it|that|this)|go\s+on|continue|an?\s+example|for\s+example)\b",
    re.IGNORECASE)

_REFERENCE = re.compile(r"\b(it|its|that|this|these|those|them|answer|response|previous|above|last|again)\b", re.IGNORECASE)

# One- or two-word follow-ups that need the last topic ("why?", "how so?").
_SHORT_FOLLOWUP = re.compile(
    r"^(why|why so|how|how so|how come|when|where|who|which|which one|what else|anything else|and then|really|so)$",
    re.IGNORECASE)


# Words that can appear in a follow-up like "make it shorter" or "give me an example then".
# If a message has any other word (a topic such as "RAG" or "the penalty clause"), it is a
# new question and is never treated as a follow-up.
_FOLLOWUP_VOCAB = set("""make it that this those these them its shorter shorten longer simpler simplify simple more detail
details detailed in one a an the sentence line word words bullet bullets point points please can you could give me example
for tell explain elaborate expand go on continue then now again rephrase reword rewrite translate into to nepali english hindi
briefly brief short so also and just little bit i want need would like be use say put as with some another of your answer
response previous above last what said about is was thing things pls plz kindly thanks thank okay ok yes yeah sure hmm well but now""".split())


def _followup_kind(query, history):
    """'REWORK' (change the form of the last answer), 'ELABORATE' (a fuller answer
    to the last question) or None. Only short messages made of follow-up words
    count, so a normal new question is never mistaken for one."""
    if not history:
        return None
    if len(query.split()) > 12:
        return None
    words = re.findall(r"[a-z']+", query.lower())
    if not words or any(w not in _FOLLOWUP_VOCAB for w in words):
        return None
    if _REWORK.search(query):
        return "REWORK"
    if _ELABORATE.search(query):
        return "ELABORATE"
    return None


def _rework_instruction(query):
    q = query.lower()
    if re.search(r"\bone\s+(?:sentence|line|word)\b|\ba\s+(?:sentence|line|word)\b", q):
        return "Write it as exactly one short sentence."
    if re.search(r"\bshorter|\bshorten|\bbrief", q):
        return "Make it clearly shorter: at most two sentences, keeping only the key facts."
    if re.search(r"\bsimpl", q):
        return "Use very simple words and short sentences, as if explaining to a child."
    if re.search(r"\bbullet|\bin\s+points\b", q):
        return "Write it as a short bullet list."
    if "translate" in q:
        return f"Translate it as asked ({query}). Keep numbers and names the same."
    return f"Do what the user asks: {query}"


def rework_answer(query, last_turn):
    """Re-write the previous answer the way the user asks (shorter, simpler, bullets...),
    using only what the previous answer already says."""
    prompt = f"""Here is the previous question and the answer you gave.

Previous question: {last_turn['question']}
Previous answer: {str(last_turn['answer'])[:3000]}

The user now says: {query}
Instruction: {_rework_instruction(query)}

Rewrite the previous answer that way. Use ONLY the facts in the previous answer - do not add anything new. Keep numbers, names and dates exactly as written.

Rewritten answer:"""
    return llm_generate(prompt, num_predict=300, temperature=0.1).strip()


_SOURCE_STOPWORDS = {
    "which", "their", "there", "about", "would", "could", "should", "these", "those",
    "other", "where", "while", "being", "using", "based", "document", "documents",
    "provided", "context", "information", "mentioned", "stated", "written",
}


_NO_ANSWER = re.compile(
    r"couldn'?t find|could not find|can'?t find|don'?t know|do not know|no information|"
    r"not (mentioned|stated|found|covered|present|available)|isn'?t (mentioned|stated|covered)|"
    r"(doesn'?t|does not|don'?t|do not) (contain|mention|appear|include|have|provide|say|state|specify)|"
    r"wasn'?t able|unable to find|no (mention|details|data)|not (specified|provided|listed)", re.IGNORECASE)


def _answer_tokens(answer):
    """The words and numbers of an answer that are worth looking for in documents."""
    # The model often writes dates and numbers with special non-breaking
    # hyphens/spaces, which would never match the document text.
    answer = re.sub(r"[\u2010-\u2015\u2212]", "-", answer)
    answer = re.sub(r"[\u00a0\u202f\u2009]", " ", answer)
    tokens = {t for t in re.findall(r"[a-z0-9][a-z0-9.,\-]{3,}", answer.lower())}
    tokens = {t.strip(".,-") for t in tokens} - _SOURCE_STOPWORDS
    return {t for t in tokens if len(t) >= 4 or any(c.isdigit() for c in t)}


def _sources_used(answer, chunks, max_sources=3):
    """Which documents did the answer really come from? Retrieval pulls chunks
    from many documents, but the answer usually uses only one or two. A
    document counts when words/numbers from the answer appear in its retrieved
    text. Words found in every document count for little, words found in only
    one document count for a lot.
    `chunks` is a list of (document_name, chunk_text)."""
    # The model often writes curly apostrophes (don\u2019t), which would not match
    # the refusal phrases below.
    answer = answer.replace("\u2019", "'").replace("\u2018", "'")
    # A short "I couldn't find / I don't know" answer used no document at all.
    if len(answer) < 250 and _NO_ANSWER.search(answer):
        return []

    texts = {}
    for doc, text in chunks:
        texts[doc] = texts.get(doc, "") + " " + text.lower()
    if not texts:
        return []

    tokens = _answer_tokens(answer)

    scores = {}
    for doc, text in texts.items():
        scores[doc] = 0.0
    for t in tokens:
        holders = [d for d, text in texts.items() if t in text]
        for d in holders:
            scores[d] += 1.0 / len(holders)

    best = max(scores.values())
    if best <= 0:
        return []
    keep = [d for d, sc in sorted(scores.items(), key=lambda x: -x[1]) if sc >= 0.4 * best]
    return keep[:max_sources]


def _page_numbers(chunk_ids):
    """{chunk id: page number} for chunks that have one stored."""
    ids = list({i for i in chunk_ids if i is not None})
    if not ids:
        return {}
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, page_number FROM document_chunks WHERE id = ANY(%s) AND page_number IS NOT NULL", (ids,))
    found = {row[0]: row[1] for row in cur.fetchall()}
    cur.close()
    conn.close()
    return found


def _source_refs(answer, chunks, docs, max_pages=3):
    """For each source document, the page(s) the answer most likely came from.
    `chunks` is a list of (chunk_id, document, text); `docs` the names chosen by
    _sources_used. A chunk counts for the pages in proportion to how many
    answer words it contains; only the best-matching chunk(s) of each document
    are used. Documents without stored page numbers just get no pages."""
    tokens = _answer_tokens(answer.replace("\u2019", "'").replace("\u2018", "'"))
    try:
        page_of = _page_numbers([c[0] for c in chunks])
    except Exception:
        page_of = {}

    refs = []
    for doc in docs:
        scored = []
        for cid, d, text in chunks:
            if d != doc or cid not in page_of:
                continue
            low = text.lower()
            score = sum(1 for t in tokens if t in low)
            scored.append((score, page_of[cid]))
        pages = []
        if scored:
            best = max(sc for sc, _ in scored)
            if best > 0:
                pages = sorted({pg for sc, pg in scored if sc >= 0.6 * best})[:max_pages]
        refs.append({"file": doc, "pages": pages})
    return refs


SEARCH_TOOL = [{
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Search the user's uploaded documents for information relevant to a query. Optionally restrict the search to one specific document by its exact filename.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
                "source_file": {"type": ["string", "null"], "description": "Optional exact filename to search only within that one document. Pass null or omit this if searching all documents."},
            },
            "required": ["query"],
        },
    },
}]


def agent_search_loop(query, user_id, history, max_steps=4, scope=None, notes=None, elaborate=False):
    """Agentic SEARCH: lets the model call search_documents more than once,
    e.g. once per document, so a question relevant to several documents
    doesn't lose one of them to a single top-k retrieval pass."""
    docs = list_documents(user_id)
    scope = _as_scope(scope)
    if scope:
        docs = [d for d in docs if d in scope]  # the user picked these files: stay inside them
    doc_list_str = ", ".join(docs) if docs else "no documents uploaded yet"

    system_prompt = f"""You are a helpful assistant answering questions using the user's uploaded documents.
Available documents: {doc_list_str}

Rules you must follow:
1. Always call search_documents at least once before answering. Never answer from your own general knowledge.
2. Your FIRST search should NOT set source_file — search across all documents first, so you can see which ones actually contain relevant content.
3. If that first search returns chunks from more than one document, run a follow-up search scoped to each of those specific documents (using source_file) to gather more detail from each, rather than guessing at document names.
4. Only include facts that search_documents actually returned. If a document's search found nothing relevant, don't mention that document.
5. When combining facts from more than one document, you may mention which document a fact is from in plain natural language if it's genuinely helpful for the reader (e.g. "your resume mentions..."), but do NOT use bracket-style citations like [filename.pdf] or 【filename.pdf】 in your answer. Write like a normal, natural assistant.
6. If nothing relevant is found after searching, say you don't know.
7. Copy numbers, IDs, dates and account numbers exactly as they appear in the search results - never reformat them, add spaces to them, or recalculate them. If a value (such as a total) is not actually written in the results, do not work it out yourself; say it isn't stated.
8. If the user asks about you (who you are, what you can do), say briefly that you are a document Q&A assistant that answers questions from the PDFs they upload. Never describe what the documents say as your own skills, knowledge or experience.
9. Answer only if the search results directly state the thing asked about. Do not guess from related details: a phone number of the seller is not the buyer's, and a black telephone does not tell you the colour of a door. If the results are about a different person, company or thing than the one asked about, say you couldn't find it.
10. If the user asks how many items or rows there are and that number is not written in the results, say it isn't stated - do not count them yourself.
11. Write currency exactly as it appears in the results (for example Rs. or NPR); never change it to another symbol such as the rupee sign or a dollar sign."""
    if elaborate:
        system_prompt += "\n12. " + (elaborate if isinstance(elaborate, str) else _MORE_DETAIL)
    if notes:
        system_prompt += ("\n\nThings the user told you earlier in this chat (about themselves - use them only if they help, "
                          "and never as a source of facts about the documents):\n- " + "\n- ".join(notes))

    messages = [{"role": "system", "content": system_prompt}]
    for turn in (history or [])[-2:]:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": query})

    all_sources = []
    all_chunks = []  # (document, text) of everything the searches returned
    all_hits = []    # (chunk id, document, text), used to find page numbers
    for step in range(max_steps):
        choice = "required" if step == 0 else ("none" if step == max_steps - 1 else "auto")
        try:
            response = groq_chat(messages, tools=SEARCH_TOOL, tool_choice=choice)
            msg = response.choices[0].message
        except Exception as e:
            # Groq sometimes answers in text instead of calling the forced tool and then
            # returns an error. On the first step we simply do the search ourselves below.
            if step == 0 and "tool_use_failed" in str(e):
                msg = type("NoToolReply", (), {"content": "", "tool_calls": None})()
            else:
                raise

        # Groq doesn't always honor tool_choice="required" on the first turn.
        # If it skipped the tool on step 0, force one direct unscoped search
        # ourselves rather than letting the model answer ungrounded.
        if step == 0 and not msg.tool_calls:
            results = retrieve_chunks(query, user_id, top_k=8, source_file=scope)
            results = add_following_chunks(results, user_id)
            all_sources.extend(r[1] for r in results)
            all_chunks.extend((r[1], r[2]) for r in results)
            all_hits.extend((r[0], r[1], r[2]) for r in results)

            snippet = "\n\n".join(f"[{r[1]}] {r[2]}" for r in results) or "No relevant results found."
            messages.append({"role": "assistant", "content": msg.content or ""})
            messages.append({"role": "user", "content": f"Search results for your reference (use these to answer, don't just repeat them):\n\n{snippet}"})
            continue

        if not msg.tool_calls:
            docs_used = _sources_used(msg.content or "", all_chunks)
            return {
                "answer": msg.content,
                "sources": docs_used,
                "source_refs": _source_refs(msg.content or "", all_hits, docs_used),
                "action_taken": "AGENT_SEARCH",
            }

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            wanted = args.get("source_file")
            if scope:
                # never search outside the files the user picked
                wanted = [wanted] if wanted in scope else scope
            results = retrieve_chunks(
                args.get("query", query), user_id, top_k=5,
                source_file=wanted
            )
            results = add_following_chunks(results, user_id)
            print(f"AGENT SEARCH: query={args.get('query')!r} source_file={args.get('source_file')!r} -> {len(results)} chunks from {sorted(set(r[1] for r in results))}")


            all_sources.extend(r[1] for r in results)
            all_chunks.extend((r[1], r[2]) for r in results)
            all_hits.extend((r[0], r[1], r[2]) for r in results)
            snippet = "\n\n".join(f"[{r[1]}] {r[2]}" for r in results) or "No relevant results found."
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": snippet,
            })

    return {
        "answer": "I wasn't able to fully resolve this after several searches — could you rephrase or narrow the question?",
        "sources": [],
        "action_taken": "AGENT_SEARCH_INCOMPLETE",
    }


def _resolve_summary_doc(query, user_id, source_file, history=None):
    """Which document does 'summarize the story' mean? An explicit selection,
    a filename typed in the question, the only document, or the document the
    search finds clearly best. Returns None when it is not clear."""
    scope = _as_scope(source_file)
    if scope:
        return scope[0] if len(scope) == 1 else None  # several selected: ask which one
    docs = list_documents(user_id)
    if len(docs) == 1:
        return docs[0]
    named = _documents_named_in_query(query, docs)
    if len(named) == 1:
        return named[0]
    # "summarize this document" while talking about one document means that one
    recent = _recent_source_doc(history)
    if recent in docs and _DEICTIC.search(query):
        return recent
    results = retrieve_chunks(query, user_id, top_k=8)
    if not results:
        return None
    counts = {}
    for row in results:
        counts[row[1]] = counts.get(row[1], 0) + 1
    ranked = sorted(counts, key=lambda d: -counts[d])
    if len(ranked) == 1 or counts[ranked[0]] >= 2 * counts[ranked[1]]:
        return ranked[0]
    return None


def generate_overview(user_id, max_docs=8, only=None):
    """One or two sentences about each uploaded document, using only the start of each.
    `only` (optional) limits it to those document names."""
    lines = []
    docs = list_documents(user_id)
    if only:
        docs = [d for d in docs if d in only]
    for doc in docs[:max_docs]:
        chunks = get_document_chunks_in_order(doc, user_id, limit=6)
        if not chunks:
            continue
        text = "\n\n".join(chunks)[:3000]
        prompt = f"""In one or two short sentences, say what this document is about. Use only the text below.

Text:
{text}

Answer:"""
        lines.append(f"- {doc}: " + llm_generate(prompt, num_predict=100, temperature=0.1).strip())
    return "\n".join(lines) if lines else "No documents have been uploaded yet."


def add_following_chunks(results, user_id, max_total=10):
    """A fact can sit just after the point where a passage was cut. When the
    results come from one document, also fetch the chunk right after each hit."""
    if not results or len({r[1] for r in results}) != 1:
        return results
    have = {r[0] for r in results}
    wanted = sorted({r[0] + 1 for r in results} - have)
    if not wanted:
        return results
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, source_file, chunk_text FROM document_chunks WHERE id = ANY(%s) AND user_id = %s",
        (wanted, user_id))
    extra = [(i, src, text, 1.0) for (i, src, text) in cur.fetchall() if src == results[0][1]]
    cur.close()
    conn.close()
    combined = sorted(list(results) + extra, key=lambda r: r[0])
    return combined if len(combined) <= max_total else list(results)


def agent_query(query, user_id, source_file=None, history=None, user_messages=None, recent_uploads=None):
    """Entry point for every question: decides what kind of request it is (list documents, summarise, table or key-value extraction, chat, or document search) and answers it."""
    full_history = history or []
    history = _trim_history(history)  # long answers cut short, so prompts stay small
    notes = extract_notes((user_messages or []) + [query])  # facts the user stated earlier in this chat
    action = agent_decide_action(query)
    scope = _as_scope(source_file)  # files the user picked for this question (none = all documents)

    # "Summarize these two documents I just gave you": the files added most recently.
    if not scope and recent_uploads and user_id is not None and _ABOUT_UPLOADS.search(query):
        existing = set(list_documents(user_id))
        picked = [d for d in recent_uploads if d in existing]
        if picked:
            scope = picked
    # Several documents to summarise: one short summary per document.
    if len(scope or []) >= 2 and action in ("SUMMARY", "SEARCH", "CHAT") and _SUMMARY_WORDS.search(query):
        action = "OVERVIEW"

    # The reply to "Which document would you like summarized?": a document name, or "this one".
    if (user_id is not None and history and str(history[-1].get("answer", "")).startswith(_ASK_WHICH)
            and action in ("CHAT", "SEARCH")):
        all_docs = list_documents(user_id)
        named = _documents_named_in_query(query, all_docs)
        recent = _recent_source_doc(history)
        pick = named[0] if len(named) == 1 else (recent if recent in all_docs and _DEICTIC.search(query) else None)
        if pick:
            action, scope = "SUMMARY", [pick]

    # "make it shorter" / "make it longer" / "give an example" point back at the last answer.
    followup = _followup_kind(query, history) if action in ("CHAT", "SEARCH") else None
    elaborate = False
    if followup == "REWORK":
        last = full_history[-1]
        return {
            "answer": rework_answer(query, last),
            "sources": last.get("sources") or [],
            "source_refs": last.get("source_refs"),
            "action_taken": "REWORK",
        }
    if followup == "ELABORATE":
        # the last REAL question (not an earlier "make it shorter" or "why?")
        earlier = [t["question"] for t in history
                   if len(t["question"].split()) >= 3
                   and _followup_kind(t["question"], [t]) is None
                   and not _SHORT_FOLLOWUP.match(t["question"].strip().lower().strip("!.? "))]
        if earlier:
            wants_example = bool(re.search(r"example", query, re.IGNORECASE))
            # "an example" / "more detail" after a table makes no sense as another table, so the
            # search uses the table's topic (without the word "table") and asks for sentences.
            last_q = earlier[-1]
            query = _table_topic(last_q) if _TABLE_REQUEST.search(last_q) else last_q
            action = "SEARCH"  # search that question again
            elaborate = (_AN_EXAMPLE if wants_example else _MORE_DETAIL) + " Answer in plain sentences, not a table."


    # A follow-up like "explain that more simply" or "no, I mean RAG" has no
    # topic of its own, so the classifier says CHAT and it gets answered with no
    # grounding. With history present, treat clear follow-up requests as SEARCH
    # so they get rewritten and answered from the documents.
    if action == "CHAT" and history and _FOLLOWUP_REQUESTS.search(query.strip()):
        action = "SEARCH"

    # Only rewrite for SEARCH — LIST/SUMMARY/CHAT should act on what the user
    # actually typed, since rewriting a short command like "what documents
    # do you have?" against unrelated prior topics can corrupt it.
    if action == "SEARCH" and not elaborate:
        query = rewrite_query(query, history)


    if action != "CHAT" and user_id is None:
        return {
            "answer": "I don't have general knowledge to draw from yet. I only answer based on documents you upload. Sign up or log in to get started.",
            "sources": [],
            "action_taken": "AUTH_REQUIRED"
        }

    if action == "LIST":
        docs = list_documents(user_id)
        if docs:
            answer = "Here are the documents you've uploaded: " + ", ".join(docs)
        else:
            answer = "No documents have been uploaded yet."
        return {"answer": answer, "sources": [], "action_taken": "LIST"}

    elif action == "OVERVIEW":
        return {"answer": generate_overview(user_id, only=scope), "sources": [], "action_taken": "OVERVIEW"}

    elif action == "SUMMARY":
        target = _resolve_summary_doc(query, user_id, scope, history)
        if not target:
            docs = scope or list_documents(user_id)
            answer = f"{_ASK_WHICH} You have: " + ", ".join(docs) if docs else "No documents have been uploaded yet."
            return {"answer": answer, "sources": [], "action_taken": "SUMMARY"}
        return {"answer": generate_summary(target, user_id), "sources": [target], "action_taken": "SUMMARY"}

    elif action == "TABLE":
        docs = _pick_documents_for_query(query, user_id, scope, recent_doc=_recent_source_doc(history))
        if not docs:
            return {
                "answer": "I couldn't find a document with that information. Try naming the document, for example: \"show the table in invoice.pdf\".",
                "sources": [],
                "action_taken": "TABLE_NO_MATCH"
            }

        recent = _recent_source_doc(history)
        explicit = (bool(scope) or bool(_documents_named_in_query(query, list_documents(user_id)))
                    or (recent is not None and docs == [recent]))  # the document we are talking about

        results = []
        sections = {}
        for doc in docs:
            try:
                pieces, cut_short = _document_windows(doc, user_id)
                sections[doc] = list(dict.fromkeys(
                    re.findall(r"\[Table \d+ on Page \d+ - ([^\]]+)\]", "\n".join(pieces))))
                has_tables = any("[Table " in x for x in pieces)
                # A long document with no tables that the user did not ask for
                # by name is not worth reading piece by piece.
                if not explicit and len(pieces) > 1 and not has_tables:
                    print(f"TABLE: skipping '{doc}' (long, no tables, not named)")
                    continue
                print(f"TABLE: using '{doc}' ({len(pieces)} piece(s), {sum(len(x) for x in pieces)} characters)")
                parts, failed, columns = [], 0, None
                for piece in pieces:
                    try:
                        part = extract_structured(query, piece, columns)
                        if columns is None and part["tables"]:
                            columns = part["tables"][0]["columns"]
                        parts.append(part)
                    except LLMBusyError:
                        raise
                    except Exception as e:
                        failed += 1
                        print(f"  one piece of {doc} failed ({type(e).__name__}): {e}")
                if not parts:
                    raise RuntimeError("every piece failed")
                structured = _merge_structured(parts)
                extra = []
                if failed:
                    extra.append(f"Part of this document could not be read this time ({failed} of {len(pieces)} pieces), so the result may be incomplete. Please try again.")
                if cut_short:
                    extra.append("This document is very long, so only the first part was read.")
                if extra:
                    structured["note"] = " ".join([structured["note"]] + extra).strip()
            except LLMBusyError:
                return {
                    "answer": "Groq's usage limit has been reached for now, so I can't build a reliable table right now. Please try again in a few minutes.",
                    "sources": [],
                    "action_taken": "TABLE_BUSY"
                }
            except Exception as e:
                print(f"Structured extraction failed for {doc} ({type(e).__name__}): {e}")
                continue
            results.append({"document": doc, **structured})

        if not results:
            return {
                "answer": "I couldn't turn that into a table this time. Please try again or rephrase.",
                "sources": [],
                "action_taken": "TABLE_ERROR"
            }

        # Only show documents where something was actually found.
        shown = [r for r in results if r["pairs"] or r["tables"]]
        if not shown:
            first = results[0]["document"]
            answer = f"I couldn't find that in {first}."
            # Suggest at most 3 short section names (long headings are skipped).
            short = [s for s in sections.get(first, []) if len(s) <= 30][:3]
            if short:
                answer += " Try asking about: " + ", ".join(short) + "."
            return {"answer": answer, "sources": [], "action_taken": "TABLE_NOT_FOUND"}
        if len(shown) == 1:
            answer = _structured_to_text(shown[0])
        else:
            answer = "\n\n".join(f"From {r['document']}:\n{_structured_to_text(r)}" for r in shown)

        return {
            "answer": answer,
            "sources": [r["document"] for r in shown],
            "action_taken": "TABLE",
            "structured": {"results": shown},
        }

    elif action == "CHAT":
        if _ABOUT_ASSISTANT.search(query):
            # A question about the assistant itself: answer from its background line.
            answer = llm_generate(
                f"{ASSISTANT_BACKGROUND}\nAnswer the question below briefly (1 to 3 sentences), only from the description above. "
                f"Do not make up abilities.\n\nQuestion: {query}",
                num_predict=150)
        else:
            # Ordinary small talk: just chat, without bringing up PDFs. The earlier
            # conversation is included, so "what's my name?" can be answered.
            context = _chat_context(history, notes)
            answer = llm_generate(
                "Respond naturally and briefly to the message below. Use the conversation notes only for what the user "
                "told you or asked earlier (such as their name). Never invent facts about the user; if they ask about "
                "something that was not said, say you don't remember it being mentioned. You cannot see the user's "
                "documents in this reply, so never describe what a document says: if they want something from a "
                "document, ask them to put it as a question.\n\n"
                + (context + "\n\n" if context else "") + f"Message: {query}", num_predict=120)
        return {"answer": answer, "sources": [], "action_taken": "CHAT"}

    else:  # SEARCH
        if LLM_PROVIDER == "groq":
            try:
                return agent_search_loop(query, user_id, history, scope=scope, notes=notes, elaborate=elaborate)
            except Exception as e:
                print(f"Agent loop failed ({type(e).__name__}): {e}")

        results = retrieve_chunks(query, user_id, top_k=8, source_file=scope)


        if not results:
            return {
                "answer": "I don't know — this doesn't appear to be covered in the knowledge base.",
                "sources": [],
                "action_taken": "SEARCH_NO_MATCH"
            }

        best_distance = min(row[3] for row in results)
        context = "\n\n".join([chunk_text for (_, _, chunk_text, _) in results])

        STRONG_THRESHOLD = 0.25
        WEAK_THRESHOLD = 0.45

        if best_distance <= STRONG_THRESHOLD:
            is_relevant = True
        elif best_distance >= WEAK_THRESHOLD:
            is_relevant = False
        else:
            is_relevant = check_relevance(query, context)

        if not is_relevant:
            return {
                "answer": "I don't know — this doesn't appear to be covered in the knowledge base.",
                "sources": [],
                "action_taken": "SEARCH_NO_MATCH"
            }

        answer = generate_answer(query, results, elaborate=elaborate)
        sources = _sources_used(answer, [(src, text) for (_, src, text, _) in results])
        source_refs = _source_refs(answer, [(cid, src, text) for (cid, src, text, _) in results], sources)
        return {"answer": answer, "sources": sources, "source_refs": source_refs, "action_taken": "SEARCH"}