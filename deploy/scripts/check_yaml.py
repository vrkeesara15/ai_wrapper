#!/usr/bin/env python3
"""Parse every plain YAML/JSON deployment asset so a broken file fails `make lint` and CI.

Helm templates (deploy/helm/*/templates) are Go templates, not YAML, and are checked by `helm lint` /
`helm template` instead. Multi-document files are fully parsed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
GLOBS = ["deploy/**/*.yaml", "deploy/**/*.yml", "deploy/**/*.json", ".github/workflows/*.yml", "config/**/*.yaml"]


def main() -> int:
    failures = 0
    checked = 0
    for pattern in GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            rel = path.relative_to(ROOT)
            if "templates" in rel.parts or path.is_symlink() or any(p.is_symlink() for p in path.parents):
                continue
            checked += 1
            try:
                text = path.read_text()
                if path.suffix == ".json":
                    json.loads(text)
                else:
                    list(yaml.safe_load_all(text))
            except (yaml.YAMLError, json.JSONDecodeError) as exc:
                failures += 1
                print(f"FAIL {rel}: {exc}", file=sys.stderr)
    print(f"checked {checked} files, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
