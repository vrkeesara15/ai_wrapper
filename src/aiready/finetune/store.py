"""Versioned dataset store: immutable versions with a manifest and checksums.

Every adapter must be traceable to the exact examples it was trained on (and tested on). A version is
written once - ``data/finetune/datasets/<name>/v<N>/{train,val,heldout}.jsonl`` + ``manifest.json`` - and its
checksums are verified on every load, so a silently edited file is detected before it reaches training or
the release gate. The manifest is also kept in the DocStore for listing without touching the files.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aiready.core.errors import NotFound, ValidationFailed
from aiready.core.stores import DocStore, get_docstore
from aiready.finetune.dataset import SPLITS, BuildReport, DatasetExample, DatasetSplits

NS = "finetune.datasets"


class DatasetManifest(BaseModel):
    name: str
    version: int
    task: str | None = None
    created_at: float = Field(default_factory=time.time)
    created_by: str | None = None
    files: dict[str, str]  # split -> relative path
    checksums: dict[str, str]  # split -> sha256
    counts: dict[str, int]
    content_hash: str  # over all checksums: one id for the whole version
    report: BuildReport
    params: dict[str, Any] = Field(default_factory=dict)

    @property
    def ref(self) -> str:
        return f"{self.name}@v{self.version}"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


class DatasetStore:
    def __init__(self, root: str | Path = "data/finetune/datasets", docstore: DocStore | None = None):
        self.root = Path(root)
        self._docstore = docstore

    @property
    def docstore(self) -> DocStore:
        return self._docstore or get_docstore()

    def versions(self, name: str) -> list[DatasetManifest]:
        return sorted((DatasetManifest.model_validate(v) for v in self.docstore.values(NS, prefix=f"{name}@")),
                      key=lambda m: m.version)

    def save(self, name: str, splits: DatasetSplits, *, task: str | None = None, created_by: str | None = None,
             params: dict[str, Any] | None = None) -> DatasetManifest:
        existing = self.versions(name)
        version = (existing[-1].version + 1) if existing else 1
        d = self.root / name / f"v{version}"
        if d.exists():
            raise ValidationFailed(f"dataset version directory {d} already exists; versions are immutable")
        d.mkdir(parents=True)
        files, sums = {}, {}
        for split in SPLITS:
            p = d / f"{split}.jsonl"
            with p.open("w", encoding="utf-8") as f:
                for ex in splits.split(split):
                    f.write(ex.model_dump_json() + "\n")
            files[split] = str(p.relative_to(self.root))
            sums[split] = _sha(p)
        content = hashlib.sha256(json.dumps(sums, sort_keys=True).encode()).hexdigest()
        m = DatasetManifest(name=name, version=version, task=task, created_by=created_by, files=files, checksums=sums,
                            counts={s: len(splits.split(s)) for s in SPLITS}, content_hash=content,
                            report=splits.report, params=params or {})
        (d / "manifest.json").write_text(m.model_dump_json(indent=2), encoding="utf-8")
        for p in d.iterdir():  # read-only on disk as a second guard against edits
            p.chmod(0o444)
        self.docstore.put(NS, f"{name}@{version:06d}", m.model_dump())
        return m

    def manifest(self, name: str, version: int | None = None) -> DatasetManifest:
        vs = self.versions(name)
        if not vs:
            raise NotFound(f"dataset {name} not found")
        if version is None:
            return vs[-1]
        for m in vs:
            if m.version == version:
                return m
        raise NotFound(f"dataset {name}@v{version} not found")

    def verify(self, name: str, version: int | None = None) -> DatasetManifest:
        m = self.manifest(name, version)
        for split, rel in m.files.items():
            p = self.root / rel
            if not p.exists():
                raise ValidationFailed(f"{m.ref}: {split} file missing")
            if _sha(p) != m.checksums[split]:
                raise ValidationFailed(f"{m.ref}: checksum mismatch on {split} - the file was modified")
        return m

    def load(self, name: str, split: str, version: int | None = None) -> list[DatasetExample]:
        m = self.verify(name, version)
        p = self.root / m.files[split]
        return [DatasetExample.model_validate_json(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    def path(self, name: str, split: str, version: int | None = None) -> Path:
        m = self.manifest(name, version)
        return self.root / m.files[split]
