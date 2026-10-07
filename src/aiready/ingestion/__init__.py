"""RAG ingestion platform: connectors -> change detection -> parse -> clean/classify -> chunk -> enrich ->
embed (bulk, via the gateway) -> index (via the search service) -> verify, with lineage, a document registry,
retries, a dead-letter queue and scheduled / event-driven runs.

Kept import-light: heavy modules are imported where used.
"""
