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
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            # Default page-segmentation mode: it also reads light text on
            # coloured bands (table headers, footers), which --psm 6 skipped.
            text = pytesseract.image_to_string(image, lang=lang)

        full_text += f"\n--- Page {page_num + 1} ---\n{text}"

        if page_num in tables_by_page:
            boxes = boxes_by_page[page_num]
            for t_idx, table in enumerate(tables_by_page[page_num]):
                title = _title_above(page, boxes[t_idx], boxes) if has_text_layer else ""
                label = f"Table {t_idx + 1} on Page {page_num + 1}" + (f" - {title}" if title else "")
                full_text += f"\n\n[{label}]\n{_format_table(table)}"

    return full_text


_HEADING_RE = re.compile(r"^[A-Z][A-Z &/,\-]{3,40}$")


def _heading_of(line):
    """Return the heading text if this line looks like an ALL-CAPS section
    heading (ignoring stray bullet characters like '. ' from OCR)."""
    core = re.sub(r"^[^A-Za-z0-9]+", "", line.strip())
    return core if _HEADING_RE.match(core) else None


def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """Split a document into small overlapping pieces, keeping each section heading with its text, so they can be searched."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "। ", " ", ""]
    )

    # 1. Split the document into sections at heading lines
    sections, heading, body = [], None, []
    for line in text.split("\n"):
        h = _heading_of(line)
        if h:
            sections.append((heading, "\n".join(body)))
            heading, body = h, []
        else:
            body.append(line)
    sections.append((heading, "\n".join(body)))

    # 2. Chunk each section on its own, keeping the heading with every chunk
    chunks = []
    for heading, body in sections:
        body = body.strip()
        if not body:
            continue
        for piece in splitter.split_text(body):
            chunks.append(f"{heading}\n{piece}" if heading else piece)
    return chunks


def embed_chunks(chunks):
    prefixed_chunks = [f"passage: {chunk}" for chunk in chunks]
    return model.encode(prefixed_chunks)


def store_chunks(source_file, chunks, embeddings, content_hash, user_id):
    conn = get_connection()
    cur = conn.cursor()
    for chunk, embedding in zip(chunks, embeddings):
        cur.execute(
            """
            INSERT INTO document_chunks (source_file, chunk_text, embedding, content_hash, user_id)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (source_file, chunk, embedding.tolist(), content_hash, user_id)
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


def retrieve_chunks(query, user_id, top_k=5, source_file=None, candidate_k=20):
    """Hybrid retrieval: semantic (vector) search + keyword (full-text) search,
    merged with Reciprocal Rank Fusion. Every returned row keeps its real
    vector distance, so the relevance thresholds in agent_query still work."""
    query_embedding = model.encode(f"query: {query}").tolist()
    keyword_q = _keyword_query(query)

    scope_sql = "user_id = %s"
    scope_params = [user_id]
    if source_file:
        scope_sql += " AND source_file = %s"
        scope_params.append(source_file)

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

def generate_answer(query, retrieved_chunks):
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

    prompt = f"""You are a helpful assistant answering questions based only on the provided context.
The context is split into labeled documents. Details from one document must never be mixed into facts from another.
Only state facts that are explicitly and directly written in the context. Do not infer causes, combine unrelated events, or guess at relationships between events that aren't clearly stated.
Provide a complete, informative answer in at least one full sentence - do not just repeat the question's key term.
If the answer isn't clearly stated in the context, say you don't know - do not make up information.
Copy numbers, IDs, dates and account numbers exactly as they are written in the context - never reformat them, add spaces to them, or recalculate them.
If the user asks about you (who you are, what you can do), say briefly that you are a document Q&A assistant that answers questions from the PDFs they upload. Never describe what the documents say as your own skills, knowledge or experience.
Never mention labels such as 'Table 1 on Page 1' in your answer. If the question needs adding up or comparing values from several rows or tables and that total is not written in the context, say it is not stated.

Context:
{context}

Question: {query}

Answer:"""
    return llm_generate(prompt, num_predict=200, temperature=0.1)


def list_documents(user_id):
    """List the names of the PDFs this user has uploaded."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT source_file FROM document_chunks WHERE user_id = %s;", (user_id,))
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
    chunks = get_document_chunks_in_order(source_file, user_id, limit=25)
    if not chunks:
        return "I don't have any content to summarize yet please upload a document first."

    context = "\n\n".join(chunks)
    prompt = f"""Summarize the following document content in 3-5 sentences, covering its main topics and purpose.

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
5. If nothing relevant is found, return empty "pairs" and empty "tables" and explain in "note".
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


def _pick_documents_for_query(query, user_id, source_file, max_docs=3):
    """Work out which document(s) the user means: an explicit selection first,
    then filenames typed in the question, then the documents the search finds
    the most relevant chunks in. When the question could fit several documents
    (for example two invoices), return up to max_docs so each gets its own
    labelled result instead of silently guessing one."""
    if source_file:
        return [source_file]

    named = _documents_named_in_query(query, list_documents(user_id))
    if named:
        return named[:max_docs]

    results = retrieve_chunks(query, user_id, top_k=8)
    if not results:
        return []

    order = list(dict.fromkeys(row[1] for row in results))  # best-ranked first
    counts = {}
    for row in results:
        counts[row[1]] = counts.get(row[1], 0) + 1
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


def agent_decide_action(query):
    """Decide the action: LIST/SUMMARY/TABLE/CHAT via keyword rules first, CHAT vs SEARCH via LLM otherwise."""
    query_lower = query.lower()

    if query_lower.strip("!.? ") in GREETING_KEYWORDS:
        return "CHAT"

    if _ABOUT_ASSISTANT.search(query):
        return "CHAT"

    list_keywords = ["what documents", "which documents", "what files", "which files",
                      "list documents", "list files", "documents have you", "documents do you"]
    if any(keyword in query_lower for keyword in list_keywords):
        return "LIST"

    summary_keywords = ["tell me about this document", "tell me about the document",
                         "what is this document about", "what's this document about",
                         "summarize this document", "summarize the document",
                         "give me a summary", "what does this document cover",
                         "tell me about the pdf", "tell me about this pdf"]
    if any(keyword in query_lower for keyword in summary_keywords):
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
    r"more detail|in one sentence|i mean|i meant)\b"
    r"|^(and|also|what about|how about)\b",
    re.IGNORECASE)


def rewrite_query(query, history):
    """Rewrite a follow-up into a standalone question using the most recent
    turns. Messages with no follow-up reference words are returned unchanged."""
    if not history:
        return query

    stripped = query.strip().lower().strip("!.? ")
    if stripped in SKIP_REWRITE_PATTERNS or len(stripped) <= 3:
        return query

    # Already standalone (no vague reference) -> leave it alone
    if not _FOLLOWUP_HINTS.search(query.strip()):
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

_SOURCE_STOPWORDS = {
    "which", "their", "there", "about", "would", "could", "should", "these", "those",
    "other", "where", "while", "being", "using", "based", "document", "documents",
    "provided", "context", "information", "mentioned", "stated", "written",
}


_NO_ANSWER = re.compile(
    r"couldn'?t find|could not find|can'?t find|don'?t know|do not know|no information|"
    r"not (mentioned|stated|found|covered|present|available)|isn'?t (mentioned|stated|covered)|"
    r"doesn'?t (contain|mention|appear)", re.IGNORECASE)


def _sources_used(answer, chunks, max_sources=3):
    """Which documents did the answer really come from? Retrieval pulls chunks
    from many documents, but the answer usually uses only one or two. A
    document counts when words/numbers from the answer appear in its retrieved
    text. Words found in every document count for little, words found in only
    one document count for a lot.
    `chunks` is a list of (document_name, chunk_text)."""
    # A short "I couldn't find / I don't know" answer used no document at all.
    if len(answer) < 250 and _NO_ANSWER.search(answer):
        return []

    texts = {}
    for doc, text in chunks:
        texts[doc] = texts.get(doc, "") + " " + text.lower()
    if not texts:
        return []

    tokens = {t for t in re.findall(r"[a-z0-9][a-z0-9.,\-]{3,}", answer.lower())}
    tokens = {t.strip(".,-") for t in tokens} - _SOURCE_STOPWORDS
    tokens = {t for t in tokens if len(t) >= 4 or any(c.isdigit() for c in t)}

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


def agent_search_loop(query, user_id, history, max_steps=4):
    """Agentic SEARCH: lets the model call search_documents more than once,
    e.g. once per document, so a question relevant to several documents
    doesn't lose one of them to a single top-k retrieval pass."""
    docs = list_documents(user_id)
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
8. If the user asks about you (who you are, what you can do), say briefly that you are a document Q&A assistant that answers questions from the PDFs they upload. Never describe what the documents say as your own skills, knowledge or experience."""

    messages = [{"role": "system", "content": system_prompt}]
    for turn in (history or [])[-2:]:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": query})

    all_sources = []
    all_chunks = []  # (document, text) of everything the searches returned
    for step in range(max_steps):
        choice = "required" if step == 0 else "auto"
        response = groq_chat(messages, tools=SEARCH_TOOL, tool_choice=choice)
        msg = response.choices[0].message

        # Groq doesn't always honor tool_choice="required" on the first turn.
        # If it skipped the tool on step 0, force one direct unscoped search
        # ourselves rather than letting the model answer ungrounded.
        if step == 0 and not msg.tool_calls:
            results = retrieve_chunks(query, user_id, top_k=8)
            all_sources.extend(r[1] for r in results)
            all_chunks.extend((r[1], r[2]) for r in results)

            snippet = "\n\n".join(f"[{r[1]}] {r[2]}" for r in results) or "No relevant results found."
            messages.append({"role": "assistant", "content": msg.content or ""})
            messages.append({"role": "user", "content": f"Search results for your reference (use these to answer, don't just repeat them):\n\n{snippet}"})
            continue

        if not msg.tool_calls:
            return {
                "answer": msg.content,
                "sources": _sources_used(msg.content or "", all_chunks),
                "action_taken": "AGENT_SEARCH",
            }

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            results = retrieve_chunks(
                args.get("query", query), user_id, top_k=5,
                source_file=args.get("source_file")
            )
            print(f"AGENT SEARCH: query={args.get('query')!r} source_file={args.get('source_file')!r} -> {len(results)} chunks from {sorted(set(r[1] for r in results))}")


            all_sources.extend(r[1] for r in results)
            all_chunks.extend((r[1], r[2]) for r in results)
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


def agent_query(query, user_id, source_file=None, history=None):
    """Entry point for every question: decides what kind of request it is (list documents, summarise, table or key-value extraction, chat, or document search) and answers it."""
    action = agent_decide_action(query)


    # A follow-up like "explain that more simply" or "no, I mean RAG" has no
    # topic of its own, so the classifier says CHAT and it gets answered with no
    # grounding. With history present, treat clear follow-up requests as SEARCH
    # so they get rewritten and answered from the documents.
    if action == "CHAT" and history and _FOLLOWUP_REQUESTS.search(query.strip()):
        action = "SEARCH"

    # Only rewrite for SEARCH — LIST/SUMMARY/CHAT should act on what the user
    # actually typed, since rewriting a short command like "what documents
    # do you have?" against unrelated prior topics can corrupt it.
    if action == "SEARCH":
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

    elif action == "SUMMARY":
        if not source_file:
            answer = "Please tell me which document you'd like summarized."
        else:
            answer = generate_summary(source_file, user_id)
        return {"answer": answer, "sources": [source_file] if source_file else [], "action_taken": "SUMMARY"}

    elif action == "TABLE":
        docs = _pick_documents_for_query(query, user_id, source_file)
        if not docs:
            return {
                "answer": "I couldn't find a document with that information. Try naming the document, for example: \"show the table in invoice.pdf\".",
                "sources": [],
                "action_taken": "TABLE_NO_MATCH"
            }

        explicit = bool(source_file) or bool(_documents_named_in_query(query, list_documents(user_id)))

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
                    "sources": docs,
                    "action_taken": "TABLE_BUSY"
                }
            except Exception as e:
                print(f"Structured extraction failed for {doc} ({type(e).__name__}): {e}")
                continue
            results.append({"document": doc, **structured})

        if not results:
            return {
                "answer": "I couldn't turn that into a table this time. Please try again or rephrase.",
                "sources": docs,
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
            return {"answer": answer, "sources": [first], "action_taken": "TABLE_NOT_FOUND"}
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
            # Ordinary small talk: just chat, without bringing up PDFs.
            answer = llm_generate(f"Respond naturally and briefly to this message: {query}", num_predict=100)
        return {"answer": answer, "sources": [], "action_taken": "CHAT"}

    else:  # SEARCH
        if LLM_PROVIDER == "groq":
            try:
                return agent_search_loop(query, user_id, history)
            except Exception as e:
                print(f"Agent loop failed ({type(e).__name__}): {e}")

        results = retrieve_chunks(query, user_id, top_k=8, source_file=source_file)


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

        answer = generate_answer(query, results)
        sources = _sources_used(answer, [(src, text) for (_, src, text, _) in results])
        return {"answer": answer, "sources": sources, "action_taken": "SEARCH"}