-- Runs once when the pilot database is first created.
-- pgvector backs the optional postgres vector store; pg_trgm helps keyword search fallbacks.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
