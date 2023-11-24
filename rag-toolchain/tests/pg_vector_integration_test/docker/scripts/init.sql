CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS embeddings (
    id serial primary key,
    content TEXT,
    embedding vector(3)
);