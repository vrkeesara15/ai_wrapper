"""Legacy integration layer: the per-client adapters that let the AI platform use an existing application
without modifying it (slide "Legacy integration architecture").

* ``identity_bridge`` / ``bridge_api`` - turns an existing app session (or IdP token) into a short-lived,
  signed AI-Ready token (port 8150).
* ``facade`` / ``facade_api`` - clean, declared endpoints over existing functions; each becomes an MCP tool
  (port 8151).
* ``change_capture`` - replica reader / change capture feeding the ingestion work queue.
* ``file_crawler`` - reads files from shares with their access lists.

Nothing is imported eagerly so that a process running only one adapter loads only what it needs.
"""
