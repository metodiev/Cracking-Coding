"""Task 953 - Implement contributor guide and templates for new problems, tests, and solutions.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

The guide is executable.  `validate` is the checklist a contribution must satisfy and returns one
violation per broken rule in the shape the CI comment bot posts; `TEMPLATES` renders the three
artefacts a contributor submits (spec, solution, test).  Rendering is strict - a placeholder without
a value is an error - and idempotent, so a rendered artefact can be re-rendered by a later tool.

Simplified: the "repository" is an in-memory bundle instead of a checkout, and submitted tests are
shape-checked rather than executed (count and the presence of the check helper are what the guide
can police without running user code).  Verified: the acceptance path, every rule's violation code,
slug-length and test-count boundaries, strict and idempotent rendering, the empty-bundle case and
agreement with an independently written predicate audit.
"""
import re

LANGUAGES = ("python", "java", "javascript", "csharp")
SLUG_RE = re.compile(r"\d{3}-[a-z0-9]+(?:-[a-z0-9]+)*")
PLACEHOLDER = re.compile(r"\{\{\s*([a-z_]+)\s*\}\}")
SLUG_MAX = 64
MIN_SOLUTION_LINES = 60
MIN_TESTS = 3
RULE_CODES = ("E001", "E002", "E003", "E004", "E005", "E006", "E007", "E008", "E009")
EMPTY_CODES = ("E001", "E002", "E004", "E005", "E009")  # rules needing empty/absent content

TEMPLATES = {
    "spec": "# {{number}} - {{title}}\n\n- **Number:** {{number}}\n- **Tier:** {{tier}}\n"
            "- **Slug:** {{slug}}\n\n## Prompt\n\n{{prompt}}\n",
    "solution": '"""Task {{number}} - {{title}}.  (Tier: {{tier}})\n\n{{summary}}\n"""\n'
                "{{body}}\n",
    "test": "# Generated cases for task {{number}}.\nCASES = [\n{{cases}}\n]\n",
}


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class MissingValue(Exception):
    """Raised when a template placeholder has no value."""


def render(template, values):
    """Strict placeholder substitution: every placeholder must have a value."""
    missing = sorted({name for name in PLACEHOLDER.findall(template) if name not in values})
    if missing:
        raise MissingValue(f"template needs {', '.join(missing)}")
    return PLACEHOLDER.sub(lambda match: str(values[match.group(1)]), template)


def scaffold(number, tier, slug, title, prompt, summary, body, cases):
    """Render the three artefacts a contributor submits, from TEMPLATES."""
    values = {"number": number, "tier": tier, "slug": slug, "title": title, "prompt": prompt,
              "summary": summary, "body": body,
              "cases": "\n".join(f"    ({case!r}, {expected!r})," for case, expected in cases)}
    return {name: render(template, values) for name, template in TEMPLATES.items()}


def validate(bundle):
    """One violation per broken guide rule, ordered by (code, field)."""
    violations = []
    number = bundle.get("number")
    slug = str(bundle.get("slug", ""))
    solutions = bundle.get("solutions", {})
    spec = bundle.get("spec", "")

    def report(code, field, detail):
        violations.append({"code": code, "field": field, "detail": str(detail)})

    if not isinstance(number, int) or not 1 <= number <= 998:
        report("E001", "number", number)
    prefix = f"{number:03d}-" if isinstance(number, int) else ""
    if len(slug) > SLUG_MAX or not SLUG_RE.fullmatch(slug) or not slug.startswith(prefix):
        report("E002", "slug", slug)
    if bundle.get("title") != bundle.get("readme_title"):
        report("E003", "title", bundle.get("title"))
    heading = spec.splitlines()[:1]
    if not (heading and (heading[0].startswith(f"# {number} -")
                         or heading[0].startswith(f"# {number} —"))) or "- **Tier:**" not in spec:
        report("E004", "spec", "heading or tier line")
    missing = sorted(set(LANGUAGES) - set(solutions))
    if missing:
        report("E005", "solutions", ",".join(missing))
    extra = sorted(set(solutions) - set(LANGUAGES))
    if extra:
        report("E006", "solutions", ",".join(extra))
    for language in sorted(set(solutions) & set(LANGUAGES)):
        text = solutions[language]
        if len(text.splitlines()) < MIN_SOLUTION_LINES:
            report("E007", f"solutions.{language}", len(text.splitlines()))
        if "check(" not in text:
            report("E008", f"solutions.{language}", "no check helper")
    if len(bundle.get("tests", [])) < MIN_TESTS:
        report("E009", "tests", len(bundle.get("tests", [])))
    return sorted(violations, key=lambda row: (row["code"], row["field"]))


def audit(bundle):
    """Independent audit: the same rules as plain predicates, returning a set of codes."""
    found = set()
    number = bundle.get("number")
    slug = str(bundle.get("slug", ""))
    solutions = bundle.get("solutions", {})
    spec = bundle.get("spec", "")
    if not isinstance(number, int) or not 1 <= number <= 998:
        found.add("E001")
    prefix = f"{number:03d}-" if isinstance(number, int) else ""
    if len(slug) > SLUG_MAX or not re.fullmatch(r"\d{3}-[a-z0-9]+(-[a-z0-9]+)*", slug) \
            or slug[:4] != prefix:
        found.add("E002")
    if bundle.get("title") != bundle.get("readme_title"):
        found.add("E003")
    first = spec.splitlines()[:1]
    if (not (first and first[0][:2] == "# " and first[0][2:2 + len(str(number))] == str(number))
            or "- **Tier:**" not in spec):
        found.add("E004")
    if set(LANGUAGES) - set(solutions):
        found.add("E005")
    if set(solutions) - set(LANGUAGES):
        found.add("E006")
    for text in solutions.values():
        if text.count("\n") + 1 < MIN_SOLUTION_LINES:
            found.add("E007")
        if "check(" not in text:
            found.add("E008")
    if len(bundle.get("tests", [])) < MIN_TESTS:
        found.add("E009")
    return found


def solution_text(marker=""):
    """A guide-compliant solution body: long enough and carrying a check helper."""
    body = "\n".join(f"    step_{index} = {index}  # {marker}"
                     for index in range(MIN_SOLUTION_LINES))
    return ("def check(ok, message):\n    if not ok:\n        raise AssertionError(message)\n"
            + body + "\n")


def good_bundle(number=953, slug=None, tests=None):
    slug = slug or f"{number:03d}-implement-contributor-guide-and-templates-for-new-problems"
    cases = list(tests) if tests is not None else [(1, 1), (2, 2), (3, 3)]
    title = "Implement contributor guide and templates for new problems, tests, and solutions."
    return {"number": number, "tier": "master", "slug": slug, "title": title,
            "readme_title": title,
            "spec": scaffold(number, "master", slug, title, "Prompt", "Summary", "b = 1",
                             cases)["spec"],
            "solutions": {language: solution_text(language) for language in LANGUAGES},
            "tests": cases}


def main():
    bundle = good_bundle()

    # 1) happy path: a compliant contribution validates clean and scaffolds without leftovers.
    scaffolded = scaffold(953, "master", bundle["slug"], bundle["title"], "Prompt", "Summary",
                          "b = 1", bundle["tests"])
    check(validate(bundle) == [] and sorted(scaffolded) == ["solution", "spec", "test"]
          and not any("{{" in text for text in scaffolded.values()),
          f"a compliant bundle must validate clean: {validate(bundle)}")

    # 2) every rule fires on its own mutation (one folded check over all nine codes).
    mutations = {
        "E001": good_bundle(number=1200),
        "E002": good_bundle(slug="953-Implement-Guide"),
        "E003": dict(good_bundle(), title="A different title."),
        "E004": dict(good_bundle(), spec="no heading at all"),
        "E005": dict(good_bundle(), solutions={k: v for k, v in bundle["solutions"].items()
                                               if k != "java"}),
        "E006": dict(good_bundle(), solutions=dict(bundle["solutions"], rust=solution_text())),
        "E007": dict(good_bundle(), solutions=dict(bundle["solutions"], python="def check(): ...")),
        "E008": dict(good_bundle(), solutions=dict(bundle["solutions"],
                                                   python="\n".join(["x = 1"] * 70))),
        "E009": good_bundle(tests=[(1, 1)]),
    }
    fired = {code: sorted({row["code"] for row in validate(mutated)})
             for code, mutated in mutations.items()}
    check(all(code in fired[code] for code in RULE_CODES),
          f"every rule must be reachable: {fired}")

    # 3) independent oracle: the predicate-style audit agrees with validate everywhere.
    disagreements = [code for code, mutated in mutations.items()
                     if audit(mutated) != {row["code"] for row in validate(mutated)}]
    check(not disagreements and audit(bundle) == set(),
          f"the independent audit disagreed on {disagreements}")

    # 4) boundaries: a 64-character slug and exactly three tests pass, 65 characters and two fail.
    at_limit = good_bundle(slug=f"953-{'a' * 60}")
    too_long = good_bundle(slug=f"953-{'a' * 61}")
    check(len(at_limit["slug"]) == 64 and validate(at_limit) == []
          and [row["code"] for row in validate(too_long)] == ["E002"]
          and validate(good_bundle(tests=[(1, 1), (2, 2), (3, 3)])) == []
          and [row["code"] for row in validate(good_bundle(tests=[(1, 1), (2, 2)]))] == ["E009"],
          f"boundaries: {[row['code'] for row in validate(too_long)]} "
          f"{[row['code'] for row in validate(good_bundle(tests=[(1, 1)]))]}")

    # 5) strictness (negative check): a missing value names the placeholder, extras are ignored.
    try:
        render(TEMPLATES["test"], {"number": 953})
        raise AssertionError("rendering without `cases` must fail")
    except MissingValue as error:
        check("cases" in str(error), f"the error must name the missing placeholder: {error}")
    check(render("{{number}}-{{number}}", {"number": 953, "unused": "x"}) == "953-953",
          "values that no placeholder uses must be ignored")

    # 6) rendering is idempotent and byte-stable across calls.
    values = {"number": 1, "title": "T", "tier": "master", "slug": "001-t", "prompt": "P"}
    once = render(TEMPLATES["spec"], values)
    check(render(once, {}) == once and once == render(TEMPLATES["spec"], values)
          and "- **Slug:** 001-t" in once,
          "rendered artefacts must be idempotent and byte-stable")

    # 7) boundary: an empty bundle reports every rule that does not depend on submitted content.
    empty = validate({})
    check([row["code"] for row in empty] == list(EMPTY_CODES)
          and empty == sorted(empty, key=lambda row: (row["code"], row["field"]))
          and audit({}) == set(EMPTY_CODES),
          f"an empty bundle must report {EMPTY_CODES}: {[row['code'] for row in empty]}")

    # 8) determinism, and the missing-language detail lists the gap in sorted order.
    partial = mutations["E005"]
    detail = next(row["detail"] for row in validate(partial) if row["code"] == "E005")
    check(validate(partial) == validate(partial) and detail == "java"
          and sorted(partial["solutions"]) == ["csharp", "javascript", "python"]
          and validate(bundle) == [] and audit(partial) == {"E005"},
          f"validation must be deterministic: {detail}")

    print(f"rules={len(RULE_CODES)} empty_codes={len(empty)} slug_lengths="
          f"{len(at_limit['slug'])}/{len(too_long['slug'])} templates={sorted(TEMPLATES)}")
    print(f"fired={sorted(fired)} missing_language_detail={detail}")
    print("PASS")


if __name__ == "__main__":
    main()
