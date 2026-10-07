"""RAG service: grounded answers with sources (port 8110, wiring name ``rag``).

``connect("rag")`` returns the in-process ``RagServiceImpl`` or, when ``services.rag.url`` is set, a
``RagClient``. Both implement ``core.interfaces.RagService`` plus ``answer_stream`` (Mode B).
"""

from __future__ import annotations

import os
from typing import Any

from aiready.core.wiring import provider, service_token


@provider("rag", "local")
def _local(cfg: dict[str, Any]) -> Any:
    from aiready.rag.service import RagServiceImpl

    return RagServiceImpl.from_config()


@provider("rag", "http")
def _http(cfg: dict[str, Any]) -> Any:
    from aiready.rag.client import RagClient

    return RagClient(cfg["url"], service_token=service_token(os.environ.get("AIREADY_SERVICE", "rag-client")),
                     timeout=float(cfg.get("timeout", 60.0)))
