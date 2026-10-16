#!/usr/bin/env python3
"""Validate the mermaid blocks and structure of tutorial Markdown files.

Usage:
    python3 tools/check_mermaid.py [path ...]        (default: tutorials)

Checks, per file under the given paths (files or directories, recursively):
  * every ```mermaid fence is closed, and no stray ``` inside a block
  * the first significant line of a block is a known diagram keyword
  * no tab characters (tabs break mermaid parsing in some renderers)
  * raw '<' or '>' outside of HTML-ish tags inside a label (must be &lt; / &gt;)
  * node ids do not collide with mermaid keywords (end, class, style, ...)
  * (for tutorials/tasks/**) the required section headings, the link block and 3+ diagrams

Exits non-zero and prints every problem.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DIAGRAM_KEYWORDS = (
    "flowchart", "flowchart-v2", "graph", "sequenceDiagram", "classDiagram",
    "classDiagram-v2", "stateDiagram", "stateDiagram-v2", "erDiagram", "mindmap",
    "journey", "timeline", "quadrantChart", "pie", "gitGraph", "requirementDiagram",
    "sankey-beta", "xychart-beta", "block-beta", "C4Context", "C4Container",
)

RESERVED_IDS = {"end", "class", "classDef", "style", "click", "subgraph", "direction",
                "linkStyle", "default", "graph", "flowchart", "sequenceDiagram"}

REQUIRED_SECTIONS = (
    "## 1. Problem restatement",
    "## 2. Intuition",
    "## 3. Algorithm",
    "## 4. Worked example",
    "## 5. Complexity",
    "## 6. Pitfalls and edge cases",
    "## 7. Variants and follow-ups",
    "## 8. Check yourself",
)

# A quoted label: "..." ; an unquoted label is whatever follows the node id.
LABEL_RE = re.compile(r'("(?:[^"\\]|\\.)*")|(\[[^\]\n]*\])|(\{[^}\n]*\})|(\([^)\n]*\))')
NODE_ID_RE = re.compile(r"(?:^|\s|-->|---|->|==>|-\.->)([A-Za-z_][A-Za-z0-9_]*)\s*(\[|\{|\()")
RAW_ANGLE_RE = re.compile(r"[<>]")
ALLOWED_ANGLE_RE = re.compile(r"</?(br|b|i|em|strong|code|small|sub|sup|span|div)\s*/?>",
                              re.IGNORECASE)
# Link/arrow tokens contain '-' and '>' but are not labels; strip them before the
# angle-bracket check so '-->' is not reported as a raw '>'.
LINK_RE = re.compile(
    r"<<-->>|<<->>|<-->|-->>|->>|--x|--o|--\)|-\)|-\.->|-\.-|-->|---|==>|===|"
    r"~~~|<\|\.\.|--\|>|<\|--|\|>|o--|\*--|\.\.>|\.\.\|>|x--x|o--o|<\.\.|\.\.|--|->"
)
# Class-diagram annotations such as <<abstract>> / <<interface>> are valid mermaid.
ANNOTATION_RE = re.compile(r"<<[^<>\n]*>>")


def iter_markdown(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(p.rglob("*.md"))
        elif p.suffix == ".md":
            files.append(p)
    return files


class Report:
    """Collects problems; a file under tutorials/tasks/ is an error, anything else a warning."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def add(self, md: Path, lineno: int | None, message: str) -> None:
        where = f"{md}:{lineno}" if lineno else str(md)
        entry = f"{where}: {message}"
        try:
            is_tutorial = str(md.relative_to(ROOT)).replace("\\", "/").startswith("tutorials/tasks/")
        except ValueError:
            is_tutorial = False
        (self.errors if is_tutorial else self.warnings).append(entry)


def check_mermaid_blocks(md: Path, text: str, report: Report) -> int:
    """Return the number of mermaid blocks."""
    lines = text.splitlines()
    blocks = 0
    in_block = False
    body: list[str] = []
    start_line = 0

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not in_block:
            if stripped.startswith("```"):
                info = stripped[3:].strip()
                if info == "mermaid":
                    in_block = True
                    blocks += 1
                    start_line = i
                    body = []
            continue

        if stripped.startswith("```"):
            in_block = False
            if stripped != "```":
                report.add(md, i, f"mermaid block closed with info string {stripped!r}")
            _check_block(md, start_line, body, report)
            continue
        body.append(line)

    if in_block:
        report.add(md, start_line, "mermaid block is never closed")
    return blocks


def _check_block(md: Path, start: int, body: list[str], report: Report) -> None:
    if not body:
        report.add(md, start, "empty mermaid block")
        return

    significant = [ln for ln in body if ln.strip() and not ln.strip().startswith("%%")]
    if not significant:
        report.add(md, start, "mermaid block has no diagram")
        return

    first = significant[0].strip()
    keyword = first.split()[0]
    if keyword not in DIAGRAM_KEYWORDS:
        report.add(md, start, f"first line is not a diagram keyword: {first[:60]!r}")

    for offset, line in enumerate(body):
        lineno = start + 1 + offset
        if "\t" in line:
            report.add(md, lineno, "tab character inside a mermaid block")
        code = line.split("%%")[0] if line.strip().startswith("%%") else line
        code = LINK_RE.sub(" ", code)
        code = ANNOTATION_RE.sub(" ", code)
        if RAW_ANGLE_RE.search(code):
            leftover = ALLOWED_ANGLE_RE.sub("", code)
            if RAW_ANGLE_RE.search(leftover):
                report.add(md, lineno, "raw '<' or '>' - escape as &lt; / &gt;")
        for m in NODE_ID_RE.finditer(code):
            if m.group(1) in RESERVED_IDS:
                report.add(md, lineno, f"node id {m.group(1)!r} collides with a mermaid keyword")
        # unquoted labels with punctuation are the most common parse failure
        for m in LABEL_RE.finditer(code):
            quoted, bracket, brace, paren = m.groups()
            if quoted or brace is None:
                continue
            inner = bracket or paren or ""
            core = inner[1:-1].strip()
            if core and not (core[0] == '"' and core[-1] == '"'):
                if any(ch in core for ch in ",;:(){}[]"):
                    report.add(md, lineno, f"unquoted label with punctuation: {inner!r}")


def check_tutorial_structure(md: Path, text: str, blocks: int, report: Report) -> None:
    for section in REQUIRED_SECTIONS:
        if section not in text:
            report.add(md, None, f"missing required section {section!r}")
    if blocks < 3:
        report.add(md, None, f"expected at least 3 mermaid diagrams, found {blocks}")
    if not re.search(r"^# \d{3} — ", text, re.MULTILINE):
        report.add(md, None, "first heading must be '# NNN — <title>'")
    if "../../../solutions/" not in text:
        report.add(md, None, "missing links to the solutions")
    if "../../../tasks/" not in text:
        report.add(md, None, "missing link to the task spec")


def main(argv: list[str]) -> int:
    targets = [Path(a) for a in argv[1:]] or [ROOT / "tutorials"]
    files = iter_markdown(targets)
    report = Report()
    total_blocks = 0
    for md in files:
        try:
            text = md.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            report.add(md, None, f"not valid UTF-8 ({exc})")
            continue
        blocks = check_mermaid_blocks(md, text, report)
        total_blocks += blocks
        try:
            rel = md.relative_to(ROOT).as_posix()
        except ValueError:
            rel = str(md)
        if rel.startswith("tutorials/tasks/"):
            check_tutorial_structure(md, text, blocks, report)

    for p in report.errors[:400]:
        print(f"PROBLEM {p}")
    if len(report.errors) > 400:
        print(f"... and {len(report.errors) - 400} more")
    if report.warnings:
        print(f"({len(report.warnings)} pre-existing warning(s) outside tutorials/tasks)")
        for w in report.warnings[:15]:
            print(f"  warning {w}")
    print(f"checked {len(files)} markdown files, {total_blocks} mermaid diagrams, "
          f"{len(report.errors)} problems, {len(report.warnings)} warnings")
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
