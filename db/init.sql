-- Runs once, the first time the database container starts with an empty data volume.
-- (The database itself, document_query_db, is created by the container settings.)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT              -- empty for accounts created with Google
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    source_file TEXT NOT NULL,
    page_number INTEGER,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(768),
    content_hash TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_fts ON document_chunks
    USING GIN (to_tsvector('english', chunk_text));

-- How each page of a document was read at upload (text layer or OCR); shown in the answer trace.
-- The backend also creates this table itself if an older database does not have it.
CREATE TABLE IF NOT EXISTS document_info (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source_file TEXT NOT NULL,
    pages INTEGER,
    text_pages INTEGER[],
    ocr_pages INTEGER[],
    nepali_pages INTEGER[],          -- scanned pages read as Nepali
    table_pages INTEGER[],
    tables INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (user_id, source_file)
);

CREATE TABLE IF NOT EXISTS chat_history (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    messages JSONB NOT NULL DEFAULT '[]',
    updated_at TIMESTAMP DEFAULT NOW()
);

-- A user's connected Gmail (read-only access). The refresh token is stored encrypted.
-- The backend also creates this table itself if an older database does not have it.
CREATE TABLE IF NOT EXISTS gmail_accounts (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    refresh_token TEXT NOT NULL,
    connected_at TIMESTAMP DEFAULT NOW()
);
