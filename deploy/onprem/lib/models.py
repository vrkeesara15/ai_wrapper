#!/usr/bin/env python3
"""Pack, verify and unpack model weights for air-gapped release bundles.

Why a separate helper: weights are large, many-file trees. Hashing every file (not just the archive)
lets the import host and each GPU node prove that the exact bytes approved in the model registry are
the ones being served, independent of how the archive travelled.

    models.py pack    --models models.yaml --out <bundle>/models
    models.py verify  --dir <bundle>/models
    models.py unpack  --dir <bundle>/models --dest <model-store-path>

Standard library only, except PyYAML for ``pack`` (run on the connected build host).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
import time
from pathlib import Path

CHUNK = 4 * 1024 * 1024
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _sha256_stream(stream) -> str:
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(CHUNK), b""):
        h.update(block)
    return h.hexdigest()


def _sha256_path(path: Path) -> str:
    with path.open("rb") as f:
        return _sha256_stream(f)


def pack(models_file: Path, out: Path) -> int:
    import yaml  # build host only

    spec = yaml.safe_load(models_file.read_text()) or {}
    entries = spec.get("models") or []
    if not entries:
        print(f"no models listed in {models_file}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"format": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "models": []}
    for m in entries:
        name, source, revision = m.get("name"), m.get("source"), str(m.get("revision", ""))
        if not name or not NAME_RE.match(name) or not revision or not NAME_RE.match(revision):
            print(f"invalid model entry (name/revision must match {NAME_RE.pattern}): {m}", file=sys.stderr)
            return 1
        src = Path(os.path.expanduser(str(source)))
        if not src.is_dir():
            print(f"model {name}: source directory not found: {src}", file=sys.stderr)
            return 1
        files = []
        for p in sorted(x for x in src.rglob("*") if x.is_file()):
            if p.is_symlink():
                # HF caches use symlinks into blobs/; resolve so the archive holds real bytes.
                p_real = p.resolve()
            else:
                p_real = p
            rel = p.relative_to(src).as_posix()
            files.append({"path": rel, "sha256": _sha256_path(p_real), "size": p_real.stat().st_size})
        archive = out / f"{name}-{revision}.tar"
        with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as tar:
            for f in files:
                real = (src / f["path"]).resolve()
                info = tar.gettarinfo(str(real), arcname=f["path"])
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mode = 0o444
                with real.open("rb") as fh:
                    tar.addfile(info, fh)
        manifest["models"].append({
            "name": name,
            "revision": revision,
            "licence": m.get("licence", "unknown"),
            "archive": archive.name,
            "archive_sha256": _sha256_path(archive),
            "total_bytes": sum(f["size"] for f in files),
            "files": files,
        })
        print(f"packed {name}@{revision}: {len(files)} files", file=sys.stderr)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


def verify(models_dir: Path) -> int:
    manifest = json.loads((models_dir / "manifest.json").read_text())
    failures = 0
    for m in manifest.get("models", []):
        archive = models_dir / m["archive"]
        if not archive.is_file():
            print(f"MISSING archive {archive.name}", file=sys.stderr)
            failures += 1
            continue
        if _sha256_path(archive) != m["archive_sha256"]:
            print(f"MISMATCH archive {archive.name}", file=sys.stderr)
            failures += 1
            continue
        expected = {f["path"]: f["sha256"] for f in m["files"]}
        seen: set[str] = set()
        with tarfile.open(archive, "r") as tar:
            for member in tar:
                if not member.isfile():
                    print(f"UNEXPECTED non-file member {member.name} in {archive.name}", file=sys.stderr)
                    failures += 1
                    continue
                if member.name not in expected:
                    print(f"UNEXPECTED file {member.name} in {archive.name}", file=sys.stderr)
                    failures += 1
                    continue
                stream = tar.extractfile(member)
                assert stream is not None
                if _sha256_stream(stream) != expected[member.name]:
                    print(f"MISMATCH {m['name']}:{member.name}", file=sys.stderr)
                    failures += 1
                seen.add(member.name)
        for missing in sorted(set(expected) - seen):
            print(f"MISSING {m['name']}:{missing}", file=sys.stderr)
            failures += 1
        if not failures:
            print(f"ok {m['name']}@{m['revision']} ({len(expected)} files)", file=sys.stderr)
    return 1 if failures else 0


def unpack(models_dir: Path, dest: Path) -> int:
    """Extract each model to <dest>/<name>/<revision>/ with its own manifest.json beside the weights."""
    manifest = json.loads((models_dir / "manifest.json").read_text())
    for m in manifest.get("models", []):
        target = dest / m["name"] / m["revision"]
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(models_dir / m["archive"], "r") as tar:
            for member in tar:
                # Refuse anything that could escape the target directory.
                if not member.isfile() or member.name.startswith("/") or ".." in Path(member.name).parts:
                    print(f"refusing unsafe member {member.name}", file=sys.stderr)
                    return 1
                extra = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
                tar.extract(member, target, set_attrs=False, **extra)
        model_manifest = {k: v for k, v in m.items() if k != "archive"}
        (target / "manifest.json").write_text(json.dumps(model_manifest, indent=2, sort_keys=True) + "\n")
        # sha256sum-compatible list so a busybox init container can run `sha256sum -c` on the node.
        (target / "SHA256SUMS").write_text("".join(f"{f['sha256']}  {f['path']}\n" for f in m["files"]))
        print(f"unpacked {m['name']}@{m['revision']} -> {target}", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--models", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    v = sub.add_parser("verify")
    v.add_argument("--dir", required=True, type=Path)
    u = sub.add_parser("unpack")
    u.add_argument("--dir", required=True, type=Path)
    u.add_argument("--dest", required=True, type=Path)
    a = ap.parse_args()
    if a.cmd == "pack":
        return pack(a.models, a.out)
    if a.cmd == "verify":
        return verify(a.dir)
    return unpack(a.dir, a.dest)


if __name__ == "__main__":
    sys.exit(main())
