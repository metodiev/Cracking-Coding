"""Task 994 - Implement editorial writing assistant that drafts explanations per solution.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A template-and-lint assistant: from a solution's metadata (title, approach, complexities,
walkthrough steps, pitfalls, test count) it drafts the five sections of an editorial write-up,
then lints its own draft against the house style - per-section word budgets, banned filler
phrases, mandatory mention of both complexities, a walkthrough step count that must match the
code, no empty section and no duplicated sentence.  The lint result is what makes the output
publishable: a draft that violates any rule is reported instead of being shipped.

Simplified versus production: drafting is rule-based text assembly (no language model) and the
style guide is a small constant table.  Verified: a clean publish for complete metadata,
budget acceptance exactly at the limit and rejection one word over, banned-phrase and
missing-complexity detection, step-count and duplicate-sentence detection on an independent
re-count, determinism of draft/digest, and the metadata-missing and no-walkthrough boundaries.
"""
import hashlib
import re

SECTIONS = ("summary", "approach", "complexity", "pitfalls", "verification")
BUDGETS = {"summary": 40, "approach": 120, "complexity": 40, "pitfalls": 80, "verification": 40}
BANNED = ("obviously", "trivially", "everyone knows", "just ")
REQUIRED = ("title", "approach", "time_complexity", "space_complexity", "steps", "tests", "pitfalls")
WORD = re.compile(r"[A-Za-z0-9'\-]+")
SENTENCE = re.compile(r"[^.;]+[.;]")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_error(action, needle, error_type=ValueError):
    try:
        action()
    except error_type as exc:
        check(needle in str(exc), "error %r does not mention %r" % (str(exc), needle))
        return
    raise AssertionError("expected %s mentioning %r" % (error_type.__name__, needle))


def words(text):
    return WORD.findall(text)


def sentences(text):
    return [found.strip().lower() for found in SENTENCE.findall(text)]


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class WritingAssistant:
    """Drafts the five editorial sections, then lints them against the style guide."""

    def draft(self, meta):
        for field in REQUIRED:
            if field not in meta:
                raise ValueError("solution metadata is missing %r" % (field,))
        return {
            "summary": "Solution for %s: %s." % (meta["title"], meta["approach"]),
            "approach": " ".join("%d. %s" % (number, step)
                                 for number, step in enumerate(meta["steps"], start=1)),
            "complexity": "Runs in %s time and uses %s extra space."
                          % (meta["time_complexity"], meta["space_complexity"]),
            "pitfalls": " ".join("- %s" % pitfall for pitfall in meta["pitfalls"]),
            "verification": "Covered by %d scenario tests." % meta["tests"],
        }

    def lint(self, draft, meta):
        """Return every style-guide violation; an empty list means the draft may be published."""
        problems = []
        for section in SECTIONS:
            if section not in draft:
                problems.append("section %s is missing" % section)
                continue
            text = draft[section]
            if not text.strip():
                problems.append("section %s is empty" % section)
            if len(words(text)) > BUDGETS[section]:
                problems.append("section %s exceeds its %d-word budget (%d words)"
                                % (section, BUDGETS[section], len(words(text))))
            lowered = text.lower()
            for phrase in BANNED:
                if phrase in lowered:
                    problems.append("section %s uses the banned phrase %r" % (section, phrase.strip()))
        complexity = draft.get("complexity", "")
        for required in (meta["time_complexity"], meta["space_complexity"]):
            if required not in complexity:
                problems.append("complexity section omits %s" % required)
        listed = re.findall(r"(?:^| )(\d+)\. ", draft.get("approach", ""))
        if len(listed) != len(meta["steps"]):
            problems.append("walkthrough lists %d steps but the solution has %d"
                            % (len(listed), len(meta["steps"])))
        found = sentences(" ".join(draft[section] for section in SECTIONS if section in draft))
        for sentence in sorted(set(found)):
            if found.count(sentence) > 1:
                problems.append("duplicate sentence %r" % sentence)
        return problems

    def publish(self, meta):
        draft = self.draft(meta)
        problems = self.lint(draft, meta)
        return {"draft": draft, "violations": problems, "ok": not problems,
                "digest": digest(self.render(draft))}

    def render(self, draft):
        return "\n".join("## %s\n%s" % (section.title(), draft[section]) for section in SECTIONS)


def solution_meta():
    return {"title": "Two Sum", "approach": "scan once with a hash map",
            "time_complexity": "O(n)", "space_complexity": "O(n)",
            "steps": ["Build a hash map of values", "Look up the complement of each element",
                      "Return the pair of indices"],
            "pitfalls": ["Do not reuse one index twice", "Duplicates must still be matched"],
            "tests": 12}


def main():
    assistant = WritingAssistant()
    meta = solution_meta()
    published = assistant.publish(meta)
    draft = published["draft"]

    # Happy path: every section is produced and the draft clears the style guide.
    check(sorted(draft) == sorted(SECTIONS), "the draft must contain exactly the five sections")
    check(published["ok"] and published["violations"] == [],
          "clean metadata must publish without violations, got %r" % (published["violations"],))
    check(draft["approach"].startswith("1. Build a hash map") and draft["approach"].count(". ") == 3,
          "the walkthrough must enumerate every step once")
    check("O(n) time" in draft["complexity"] and "O(n) extra space" in draft["complexity"],
          "the complexity section must name both complexities")
    check(published["digest"] == digest(assistant.render(draft)), "the digest must cover the render")

    # Independent oracle: count each step's occurrences directly in the rendered draft.
    rendered = assistant.render(draft)
    for step in meta["steps"]:
        check(rendered.count(step) == 1, "step %r must appear exactly once, not %d times"
              % (step, rendered.count(step)))
    check(len(rendered.splitlines()) == 2 * len(SECTIONS), "each section must be a heading plus a body")

    # Determinism: drafting twice is byte-identical, and a changed step changes the digest.
    check(assistant.publish(meta) == published, "drafting must be deterministic")
    edited = dict(meta, steps=meta["steps"][:2] + ["Return the pair of indices, reversed"])
    check(assistant.publish(edited)["digest"] != published["digest"],
          "editing a step must change the published digest")

    # Word-budget boundary: exactly at the limit passes, one word over is reported.
    edges = dict(draft, summary=" ".join(["pad"] * BUDGETS["summary"]))
    check(not any("summary exceeds" in problem for problem in assistant.lint(edges, meta)),
          "a section exactly at its budget must be accepted")
    over = dict(draft, summary=" ".join(["pad"] * (BUDGETS["summary"] + 1)))
    check(any("summary exceeds" in problem for problem in assistant.lint(over, meta)),
          "one word over budget must be reported")

    # Style violations on an otherwise valid draft: filler phrase and a missing complexity.
    filler = dict(draft, pitfalls=draft["pitfalls"] + " - obviously this is fast")
    check(any("banned phrase 'obviously'" in problem for problem in assistant.lint(filler, meta)),
          "the filler phrase must be flagged")
    vague = dict(draft, complexity="Runs fast.")
    check(any("omits O(n)" in problem for problem in assistant.lint(vague, meta)),
          "a complexity section that omits a complexity must be flagged")

    # Walkthrough drift: dropping a step, and repeating a sentence, are both detected.
    dropped = dict(draft, approach="1. Build a hash map of values")
    check(any("walkthrough lists 1 steps" in problem for problem in assistant.lint(dropped, meta)),
          "a walkthrough that lost a step must be flagged")
    repeated = dict(draft, pitfalls="- Match duplicates. - Match duplicates.")
    check(any("duplicate sentence" in problem for problem in assistant.lint(repeated, meta)),
          "a duplicated sentence must be flagged")
    check(not any("duplicate sentence" in problem for problem in published["violations"]),
          "the clean draft must not report duplicates")

    # Boundaries: missing metadata raises, and a solution with no walkthrough cannot publish.
    expect_error(lambda: assistant.draft({key: value for key, value in meta.items() if key != "steps"}),
                 "steps")
    bare = {"title": "Solo", "approach": "single pass", "time_complexity": "O(1)",
            "space_complexity": "O(1)", "steps": [], "pitfalls": [], "tests": 1}
    unpublished = assistant.publish(bare)
    check(not unpublished["ok"] and any("approach is empty" in problem
                                       for problem in unpublished["violations"]),
          "a solution with no walkthrough must be withheld: %r" % (unpublished["violations"],))
    summary_only = dict(draft, verification="Covered by 12 scenario tests.")
    check(not any("summary" in problem for problem in assistant.lint(summary_only, meta)),
          "a section left unchanged must stay clean")

    print("sections=%d words=%d digest=%s ok=%s"
          % (len(draft), sum(len(words(draft[section])) for section in SECTIONS),
             published["digest"][:16], published["ok"]))
    print("PASS")


if __name__ == "__main__":
    main()
