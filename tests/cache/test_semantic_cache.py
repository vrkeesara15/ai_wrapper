from aiready.cache import CacheScope, SemanticCache
from aiready.core.textutil import hash_embed


async def _embed(t):
    return hash_embed(t)


async def test_hit_same_scope_only():
    c = SemanticCache(embed=_embed, threshold=0.8)
    s1 = CacheScope(app="a", access_groups=["g1"])
    await c.store("How do I file a claim?", {"answer": "Use form X"}, s1, source_doc_ids=["d1"])
    assert (await c.lookup("how do I file a claim", s1)).answer["answer"] == "Use form X"
    assert await c.lookup("How do I file a claim?", CacheScope(app="a", access_groups=["g2"])) is None


async def test_never_cache_and_invalidate():
    c = SemanticCache(embed=_embed, threshold=0.8)
    s = CacheScope()
    assert await c.store("q", {"a": 1}, s, contains_personal_data=True) is None
    await c.store("refund policy", {"a": 1}, s, source_doc_ids=["doc9"])
    assert c.invalidate_document("doc9") == 1
    assert await c.lookup("refund policy", s) is None
