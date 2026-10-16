#!/usr/bin/env python3
"""Verify that all relative Markdown links in tutorials/ resolve to real files.

Usage: python3 tools/check_links.py [path ...]   (default: tutorials; files or directories)
Exits non-zero and prints every broken target if any link is dangling.

Every path segment is matched case-sensitively: Windows and macOS file systems are
case-insensitive, so a link to `solution.cs` resolves locally even though it is broken
on Linux (where CI runs) and on GitHub.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK_RE = re.compile(r"\]\(([^)#\s]+?)(?:#[^)\s]*)?\)")
# Fenced code blocks are not Markdown, so `inequality[j](point)` in pseudocode is not a link.
FENCE_RE = re.compile(r"^[ \t]*```[^\n]*\n.*?^[ \t]*```[ \t]*$", re.DOTALL | re.MULTILINE)


class Resolver:
    """Resolves link targets segment by segment, requiring an exact-case match."""

    def __init__(self) -> None:
        self._listings: dict[Path, dict[str, str]] = {}

    def _listing(self, directory: Path) -> dict[str, str]:
        names = self._listings.get(directory)
        if names is None:
            names = {}
            try:
                for entry in directory.iterdir():
                    names[entry.name] = entry.name
            except OSError:
                pass
            self._listings[directory] = names
        return names

    def resolve(self, start: Path, target: str) -> tuple[bool, str | None]:
        """Return (ok, wrong_case_segment) for `target` relative to `start`."""
        current = start
        for part in target.replace("\\", "/").split("/"):
            if part in ("", "."):
                continue
            if part == "..":
                current = current.parent
                continue
            names = self._listing(current)
            if part in names:
                current = current / part
                continue
            for name in names:
                if name.lower() == part.lower():
                    return False, part
            return False, None
        return (current.exists(), None)


def markdown_files(targets: list[Path]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        if target.is_dir():
            files += sorted(target.rglob("*.md"))
        elif target.is_file() and target.suffix == ".md":
            files.append(target)
    return files


def main(argv: list[str]) -> int:
    targets = [Path(t) for t in argv[1:]] or [ROOT / "tutorials"]
    broken: list[tuple[Path, str, str | None]] = []
    checked = 0
    resolver = Resolver()
    for md in markdown_files(targets):
        text = md.read_text(encoding="utf-8")
        for m in LINK_RE.finditer(FENCE_RE.sub("", text)):
            target = m.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            checked += 1
            ok, wrong_case = resolver.resolve(md.parent.resolve(), target)
            if not ok:
                broken.append((md.resolve(), target, wrong_case))
    for md, target, wrong_case in broken:
        try:
            where = md.relative_to(ROOT).as_posix()
        except ValueError:
            where = str(md)
        if wrong_case:
            print(f"BROKEN {where} -> {target} (wrong case: {wrong_case!r} does not match)")
        else:
            print(f"BROKEN {where} -> {target}")
    print(f"checked {checked} links, {len(broken)} broken")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
