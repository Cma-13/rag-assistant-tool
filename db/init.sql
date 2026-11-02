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

CREATE TABLE IF NOT EXISTS chat_history (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    messages JSONB NOT NULL DEFAULT '[]',
    updated_at TIMESTAMP DEFAULT NOW()
);
