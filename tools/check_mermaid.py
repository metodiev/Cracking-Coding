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
  * sequence diagrams: actor names may not be reserved words; message and note text may
    not contain HTML entities, ';', or any angle bracket at all
  * state diagrams: state ids may not be reserved words, and may not be quoted
  * class diagrams: labels may not contain ';'
  * (for tutorials/tasks/**) the required section headings, the link block and 3+ diagrams

These rules are the ones tools/check_mermaid_render.mjs (the real mermaid parser) rejects;
keeping them here lets the cheap check catch them without a Node toolchain.

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

# Sequence diagrams parse their message/note text with their own lexer, which is why
# these words cannot be used as actor names and why '&lt;' (but not '<') is a parse error.
SEQ_RESERVED_IDS = {"loop", "end", "opt", "alt", "else", "par", "and", "rect", "critical",
                    "option", "break", "note", "over", "activate", "deactivate",
                    "autonumber", "participant", "actor", "box", "destroy", "create",
                    "link", "links", "properties", "details", "off"}
STATE_RESERVED_IDS = {"state", "class", "note", "style", "classdef", "click", "scale"}

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

# Sequence-diagram actor names, message text and notes.
SEQ_ACTOR_RE = re.compile(r"(?:^|\s)(?:participant|actor)\s+([A-Za-z_][A-Za-z0-9_]*)")
SEQ_ARROW_ACTOR_RE = re.compile(
    r"(?:^|\s)([A-Za-z_][A-Za-z0-9_]*)\s*(?:-->>|->>|-->|->|--x|--\)|-\))")
SEQ_TEXT_RE = re.compile(
    r"^\s*(?:"
    r"[A-Za-z_][A-Za-z0-9_]*\s*(?:-->>|->>|-->|->|--x|--\)|-\))[^:]*"
    r"|[Nn]ote\s+(?:over|left of|right of)\s+[^:]*"
    r"):(.*)$")
ENTITY_RE = re.compile(r"&[A-Za-z][A-Za-z0-9]*;")

# State-diagram transitions: both the source and the target must be plain ids.
_STATE_TOKEN = r'"[^"]*"|[A-Za-z_][A-Za-z0-9_]*|\[\*\]'
STATE_TRANSITION_RE = re.compile(r"^\s*(" + _STATE_TOKEN + r")\s*-->\s*(" + _STATE_TOKEN + r")")
# Class-diagram relation labels use ':' and cannot contain ';'.
CLASS_LABEL_RE = re.compile(r":.*;")


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
            rel = md.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            rel = md.as_posix()
        (self.errors if rel.startswith("tutorials/tasks/") else self.warnings).append(entry)


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

    in_sequence = keyword == "sequenceDiagram"
    in_state = keyword in ("stateDiagram", "stateDiagram-v2")
    in_class = keyword in ("classDiagram", "classDiagram-v2")

    for offset, line in enumerate(body):
        lineno = start + 1 + offset
        if "\t" in line:
            report.add(md, lineno, "tab character inside a mermaid block")
        code = line.split("%%")[0] if line.strip().startswith("%%") else line
        if in_sequence:
            _check_sequence_line(md, lineno, code, report)

        labelled = LINK_RE.sub(" ", code)
        labelled = ANNOTATION_RE.sub(" ", labelled)
        # Sequence message text is the one place where '<' and '>' are legal and '&lt;'
        # is not, so it is checked separately (above) instead of by the generic rule.
        if not in_sequence and RAW_ANGLE_RE.search(labelled):
            leftover = ALLOWED_ANGLE_RE.sub("", labelled)
            if RAW_ANGLE_RE.search(leftover):
                report.add(md, lineno, "raw '<' or '>' - escape as &lt; / &gt;")
        for m in NODE_ID_RE.finditer(labelled):
            if m.group(1) in RESERVED_IDS:
                report.add(md, lineno, f"node id {m.group(1)!r} collides with a mermaid keyword")

        if in_state:
            m = STATE_TRANSITION_RE.match(code)
            for side in (m.group(1), m.group(2)) if m else ():
                if side.startswith('"'):
                    report.add(md, lineno,
                               'quoted state id - declare it as: state "Label" as s0')
                elif side.lower() in STATE_RESERVED_IDS:
                    report.add(md, lineno,
                               f"state id {side!r} collides with a mermaid keyword")

        if in_class and CLASS_LABEL_RE.search(code):
            report.add(md, lineno, "';' inside a class-diagram label is a parse error")

        # unquoted labels with punctuation are the most common parse failure
        if in_sequence:
            continue
        for m in LABEL_RE.finditer(labelled):
            quoted, bracket, brace, paren = m.groups()
            if quoted or brace is None:
                continue
            inner = bracket or paren or ""
            core = inner[1:-1].strip()
            if core and not (core[0] == '"' and core[-1] == '"'):
                if any(ch in core for ch in ",;:(){}[]"):
                    report.add(md, lineno, f"unquoted label with punctuation: {inner!r}")


def _check_sequence_line(md: Path, lineno: int, code: str, report: Report) -> None:
    """Sequence diagrams have their own lexer, so they have their own traps.

    Reserved words cannot be actor names, and message/note text may not contain ';',
    HTML entities, or any angle bracket at all (mermaid accepts a raw '<' here but the
    renderer may swallow it, and the natural "fix" of escaping it is a parse error).
    """
    for pattern in (SEQ_ACTOR_RE, SEQ_ARROW_ACTOR_RE):
        for m in pattern.finditer(code):
            if m.group(1).lower() in SEQ_RESERVED_IDS:
                report.add(md, lineno,
                           f"actor name {m.group(1)!r} collides with a mermaid keyword")

    m = SEQ_TEXT_RE.match(code)
    if not m:
        return
    text = m.group(1)
    if ENTITY_RE.search(text):
        report.add(md, lineno,
                   "named HTML entity in sequence message text is a parse error - "
                   "use a tag such as <br/> or plain words")
    if ";" in text:
        report.add(md, lineno, "';' in sequence message text is a parse error - use a comma")
    if RAW_ANGLE_RE.search(ALLOWED_ANGLE_RE.sub("", text)):
        report.add(md, lineno,
                   "avoid '<' and '>' in sequence message text - write 'less than' / "
                   "'at least' (only HTML tags such as <br/> may use them)")


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
