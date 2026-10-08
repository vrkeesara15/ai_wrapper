"""PostgreSQL full-text keyword index. Runs only when AIREADY_TEST_PG_URL points at a disposable database:
    docker run -d --rm -p 55432:5432 -e POSTGRES_PASSWORD=t postgres:16-alpine
    AIREADY_TEST_PG_URL=postgresql+psycopg://postgres:t@localhost:55432/postgres pytest tests/search
"""

import os
import uuid

import pytest

from aiready.core.types import Chunk
from aiready.search.filters import SearchFilter

PG = os.environ.get("AIREADY_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not PG, reason="set AIREADY_TEST_PG_URL to run against PostgreSQL")


def _chunk(i: int, text: str, tags: list[str], doc: str = "d") -> Chunk:
    return Chunk(id=f"c{i}", doc_id=f"{doc}{i}", text=text, access_tags=tags, metadata={"kind": "policy"})


def test_postgres_keyword_index_filters_access_first(tmp_path):
    from aiready.search.keyword import PostgresKeywordIndex

    idx = PostgresKeywordIndex(PG, table=f"kw_{uuid.uuid4().hex[:8]}")
    col = "kb__app__t"
    idx.upsert(col, [_chunk(1, "Escape of water excess is 350 EUR", ["public"]),
                     _chunk(2, "Settlement authority above 5000 EUR needs a senior adjuster", ["claims-team"]),
                     _chunk(3, "Fraud indicators: staged water damage", ["fraud"])])
    assert idx.count(col) == 3
    pub = idx.search(col, "water excess", SearchFilter(access_groups=["public"]), 5)
    assert [c.id for c, _ in pub] == ["c1"]
    team = idx.search(col, "water settlement", SearchFilter(access_groups=["public", "claims-team"]), 5)
    assert {c.id for c, _ in team} == {"c1", "c2"}            # never c3 (fraud)
    assert idx.search(col, "water", SearchFilter(access_groups=[]), 5) == []
    snap = idx.snapshot(col, str(tmp_path))
    assert idx.delete(col, doc_ids=["d1"]) == 1 and idx.count(col) == 2
    assert idx.restore(col, snap) == 3
