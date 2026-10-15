#!/usr/bin/env python3
"""Verify that all relative Markdown links in tutorials/ resolve to real files.

Usage: python3 tools/check_links.py [dir ...]   (default: tutorials)
Exits non-zero and prints every broken target if any link is dangling.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK_RE = re.compile(r"\]\(([^)#\s]+?)(?:#[^)\s]*)?\)")


def main(argv: list[str]) -> int:
    dirs = [Path(d) for d in argv[1:]] or [ROOT / "tutorials"]
    broken: list[tuple[Path, str]] = []
    checked = 0
    for d in dirs:
        for md in sorted(Path(d).rglob("*.md")):
            text = md.read_text(encoding="utf-8")
            for m in LINK_RE.finditer(text):
                target = m.group(1)
                if target.startswith(("http://", "https://", "mailto:")):
                    continue
                checked += 1
                resolved = (md.parent / target).resolve()
                if not resolved.exists():
                    broken.append((md, target))
    for md, target in broken:
        print(f"BROKEN {md.relative_to(ROOT)} → {target}")
    print(f"checked {checked} links, {len(broken)} broken")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
