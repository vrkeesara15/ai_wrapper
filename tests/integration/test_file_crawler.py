"""File crawler: every file is read with its access list; files without one are skipped (fail closed)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aiready.core.audit import get_audit
from aiready.core.types import DataClass
from aiready.ingestion.documents import ListSink
from aiready.integration.file_crawler import FileCrawler

REPO = Path(__file__).resolve().parents[2]


def write(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


@pytest.fixture
def share(tmp_path: Path) -> Path:
    root = tmp_path / "share"
    write(root / "claims" / "handbook.md", "# Handbook")
    write(root / "claims" / ".acl.json", json.dumps({"groups": ["claims-team"], "owner": "ops"}))
    write(root / "claims" / "fraud" / "indicators.md", "# Indicators")
    write(root / "claims" / "fraud" / ".acl.json", json.dumps({"groups": ["fraud"], "data_class": "confidential"}))
    write(root / "claims" / "fraud" / "shared.md", "# Shared")
    write(root / "claims" / "fraud" / "shared.md.acl.json", json.dumps({"users": ["carol"], "inherit": True}))
    write(root / "public" / "faq.md", "# FAQ")
    write(root / "loose.txt", "no access list")
    return root


def docs(crawler: FileCrawler) -> dict[str, object]:
    return {d.metadata["path"]: d for d in crawler.crawl()}


def test_sidecars_nearest_wins_and_inherit(share):
    c = FileCrawler(str(share), source="files", acl=["sidecar", {"type": "path_map", "map": {"public/**": ["public"]}}])
    found = docs(c)
    assert found["claims/handbook.md"].access_tags == ["claims-team"] and found["claims/handbook.md"].owner == "ops"
    ind = found["claims/fraud/indicators.md"]
    assert ind.access_tags == ["fraud"] and ind.metadata["data_class"] == DataClass.CONFIDENTIAL.value
    # file sidecar with inherit merges with the directory chain until a non-inheriting sidecar
    assert found["claims/fraud/shared.md"].access_tags == ["fraud", "user:carol"]
    assert found["public/faq.md"].access_tags == ["public"]
    # fail closed + sidecars themselves are never documents
    assert "loose.txt" not in found and not any(p.endswith(".acl.json") for p in found)
    assert c.stats.skipped_no_acl == 1 and c.stats.emitted == 4
    d = found["claims/handbook.md"]
    assert d.id == "files:claims/handbook.md" and d.mime_type == "text/markdown" and d.checksum
    assert d.uri.startswith("file://") and d.content == b"# Handbook"


def test_default_tags_include_exclude_and_size_limit(share):
    write(share / "big.txt", "x" * 3000)
    c = FileCrawler(str(share), acl=["sidecar"], default_tags=["staff"], include=["*.txt", "*.md"],
                    exclude=["claims/fraud/*"], max_file_mb=0.002)
    found = docs(c)
    assert found["loose.txt"].access_tags == ["staff"]
    assert not any(p.startswith("claims/fraud/") for p in found)
    assert "big.txt" not in found and c.stats.skipped_too_large == 1


def test_access_list_change_changes_checksum(share):
    c = FileCrawler(str(share), acl=["sidecar"])
    before = docs(c)["claims/handbook.md"].checksum
    write(share / "claims" / ".acl.json", json.dumps({"groups": ["claims-team", "auditors"]}))
    after = docs(c)["claims/handbook.md"]
    assert after.checksum != before and after.access_tags == ["auditors", "claims-team"]


def test_broken_sidecar_grants_nothing(share):
    write(share / "claims" / ".acl.json", "{not json")
    found = docs(FileCrawler(str(share), acl=["sidecar"]))
    assert "claims/handbook.md" not in found


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_posix_resolver_maps_owner_group_and_world(share):
    f = share / "loose.txt"
    os.chmod(f, 0o644)
    st = f.stat()
    c = FileCrawler(str(share), acl=[{"type": "posix", "world_tag": "public", "uid_map": {st.st_uid: "svc-owner"},
                                      "gid_map": {st.st_gid: "finance"}}])
    d = docs(c)["loose.txt"]
    assert d.access_tags == ["finance", "public", "user:svc-owner"] and d.owner == "svc-owner"
    os.chmod(f, 0o600)
    assert docs(c)["loose.txt"].access_tags == ["user:svc-owner"]


def test_symlinks_outside_root_are_not_followed(share, tmp_path):
    secret = write(tmp_path / "outside" / "secret.md", "# secret")
    write(tmp_path / "outside" / ".acl.json", json.dumps({"groups": ["everyone"]}))
    os.symlink(secret, share / "claims" / "link.md")
    assert "claims/link.md" not in docs(FileCrawler(str(share), acl=["sidecar"]))
    with pytest.raises(PermissionError):
        FileCrawler(str(share)).backend.read_bytes("../outside/secret.md")


async def test_run_emits_to_sink_and_audits(share):
    sink = ListSink()
    stats = await FileCrawler(str(share), source="files", acl=["sidecar"]).run(sink, run_id="r1")
    assert stats.emitted == len(sink.items) == 3
    rec = [r for r in get_audit().read() if r.event == "integration.file_crawl"][-1]
    assert rec.details["emitted"] == 3 and rec.details["run_id"] == "r1"


def test_sample_app_share_has_access_lists():
    c = FileCrawler(str(REPO / "examples" / "legacy_app" / "docs"), acl=[
        "sidecar", {"type": "path_map", "map": {"public/**": ["public"]}}])
    found = docs(c)
    assert found["fraud/fraud-indicators.html"].access_tags == ["fraud"]
    assert found["claims/settlement-limits.csv"].access_tags == ["claims-team"]
    assert c.stats.skipped_no_acl == 0


class FakeS3:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects

    def get_paginator(self, name):
        objs = self.objects

        class P:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k, "Size": len(v)} for k, v in objs.items() if k.startswith(Prefix)]}
        return P()

    def get_object(self, Bucket, Key):
        import io

        if Key not in self.objects:
            raise KeyError(Key)
        return {"Body": io.BytesIO(self.objects[Key])}


def test_s3_backend_with_sidecars():
    s3 = FakeS3({"docs/a.md": b"# A", "docs/.acl.json": b'{"groups": ["team"]}', "docs/b/c.md": b"# C"})
    c = FileCrawler(source="s3docs", backend="s3", bucket="bkt", prefix="docs", s3_client=s3, acl=["sidecar"])
    found = docs(c)
    assert found["a.md"].access_tags == ["team"] and found["b/c.md"].access_tags == ["team"]
    assert found["a.md"].uri == "s3://bkt/docs/a.md"
