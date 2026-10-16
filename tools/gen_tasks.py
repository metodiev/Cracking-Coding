#!/usr/bin/env python3
"""Generate per-task spec files under tasks/ from README.md.

Parses the 998 numbered tasks in README.md, groups them by tier and writes
tasks/<tier>/<NNN-slug>.md with metadata and links to existing solutions.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TIERS = ["beginner", "intermediate", "advanced", "expert", "master"]
TIER_LABELS = {
    "beginner": "Beginner",
    "intermediate": "Intermediate",
    "advanced": "Advanced",
    "expert": "Expert",
    "master": "Master / Challenge",
}
LANGS = ["python", "java", "javascript", "csharp"]
LANG_FILES = {
    "python": "solution.py",
    "java": "Solution.java",
    "javascript": "solution.js",
    "csharp": "Solution.cs",
}


def slugify(title: str) -> str:
    s = title.strip().rstrip(".")
    s = re.sub(r"[`*_]", "", s)
    s = s.lower().replace("\u2019", "'")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:64].strip("-")


def parse_readme() -> list[dict]:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    tasks: list[dict] = []
    tier: str | None = None
    for line in text.splitlines():
        h = re.match(r"^#\s+(.+?)\s*$", line)
        if h and not re.match(r"^\d+\.\s", line):
            name = re.sub(r"[^a-z]", "", h.group(1).lower())
            if name == "beginner":
                tier = "beginner"
            elif name == "intermediate":
                tier = "intermediate"
            elif name == "advanced":
                tier = "advanced"
            elif name == "expert":
                tier = "expert"
            elif name in ("masterchallenge", "master"):
                tier = "master"
            continue
        m = re.match(r"^(\d+)\.\s+(.+?)\s*$", line)
        if m and tier is not None:
            tasks.append({
                "number": int(m.group(1)),
                "title": m.group(2),
                "tier": tier,
            })
    return tasks


def chapter_of(number: int) -> str:
    mapping_file = ROOT / "docs" / "ctci-mapping.json"
    if mapping_file.exists():
        data = json.loads(mapping_file.read_text(encoding="utf-8"))
        return data.get(str(number), "—")
    return "—"


def rel_target(tier: str, num: int, slug: str, lang: str) -> str:
    """Relative path from tasks/<tier>/NNN-slug.md to a language solution file."""
    return f"../../solutions/{lang}/{tier}/{num:03d}-{slug}/{LANG_FILES[lang]}"


def rel_tutorial(tier: str, num: int, slug: str) -> str:
    """Relative path from tasks/<tier>/NNN-slug.md to its tutorial."""
    return f"../../tutorials/tasks/{tier}/{num:03d}-{slug}.md"


def main() -> None:
    tasks = parse_readme()
    assert len(tasks) == 998, f"expected 998 tasks, parsed {len(tasks)}"
    nums = [t["number"] for t in tasks]
    assert nums == list(range(1, 999)), "task numbering is not a contiguous 1..998"

    for tier in TIERS:
        (ROOT / "tasks" / tier).mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for t in tasks:
        slug = slugify(t["title"])
        tier = t["tier"]
        counts[tier] = counts.get(tier, 0) + 1
        num = t["number"]
        tutorial = ROOT / "tutorials" / "tasks" / tier / f"{num:03d}-{slug}.md"
        lines = [
            f"# {num:03d} — {t['title']}",
            "",
            f"- **Number:** {num}",
            f"- **Tier:** {TIER_LABELS[tier]}",
            f"- **CTCI chapter:** {chapter_of(num)}",
        ]
        if tutorial.exists():
            lines.append(f"- **Tutorial:** [{num:03d}-{slug}.md]({rel_tutorial(tier, num, slug)})")
        lines += [
            "",
            "## Solutions",
            "",
        ]
        any_solution = False
        for lang in LANGS:
            folder = ROOT / "solutions" / lang / tier / f"{num:03d}-{slug}"
            if (folder / LANG_FILES[lang]).exists():
                lines.append(
                    f"- {lang.capitalize()}: "
                    f"[{LANG_FILES[lang]}]({rel_target(tier, num, slug, lang)})"
                )
                any_solution = True
        if not any_solution:
            lines.append("- _Not implemented yet — see the staged plan in [README.md](../../README.md)._")
        lines.append("")
        out = ROOT / "tasks" / tier / f"{num:03d}-{slug}.md"
        out.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {len(tasks)} task specs")
    for tier in TIERS:
        print(f"  {tier}: {counts.get(tier, 0)}")


if __name__ == "__main__":
    main()
