"""Task 963 - Implement docs site generator that publishes exercises, editorial notes, solutions.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

Generator model: an in-memory site model (exercise metadata, editorial prose, per-language solution
code) is rendered into a set of HTML pages - an index plus one page per exercise with prev/next
navigation and one code block per language.  Every interpolated value is HTML-escaped, ids are
slug-ified and unique, and the generated page set is checked for dangling links, which is the
property a static-site publish step actually has to guarantee.

Simplified: pages are plain strings (no templating engine, no asset pipeline) and the "serve" step is
the link audit.  Verified: index completeness and ordering, link integrity against the generated page
set, escaping of a hostile payload, code-block language classes, prev/next at both ends, the empty
site boundary, id uniqueness, and byte-for-byte determinism.
"""
import re
from html import escape

LANGUAGES = ("python", "java", "javascript", "csharp")
SLUG_RE = re.compile(r"[^a-z0-9]+")
HREF_RE = re.compile(r'href="([^"]+)"')
NAV_RE = re.compile(r'<nav class="pager">(.*?)</nav>')
EXERCISES = (
    {"number": 1, "title": "Reverse a string", "tier": "beginner",
     "editorial": "Two pointers from both ends; watch the <odd> middle.",
     "solutions": {"python": "def reverse(s):\n    return s[::-1]", "java": "class S {}",
                   "javascript": "const reverse = s => [...s].reverse().join('');",
                   "csharp": "public static class S {}"}},
    {"number": 2, "title": "Fizz & <Buzz>", "tier": "beginner",
     "editorial": "Modulo arithmetic; <script>alert(1)</script> is escaped, not executed.",
     "solutions": {"python": "for i in range(1, 101):\n    pass", "java": "for (int i = 1; i <= 100; i++) {}",
                   "javascript": "for (let i = 1; i <= 100; i++) {}",
                   "csharp": "for (int i = 1; i <= 100; i++) { }"}},
    {"number": 3, "title": "Merge intervals", "tier": "intermediate",
     "editorial": "Sort by start, then scan while the next start is reached.",
     "solutions": {"python": "def merge(xs):\n    return xs", "java": "class M {}",
                   "javascript": "const merge = xs => xs;", "csharp": "public static class M {}"}},
)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def slugify(text):
    return SLUG_RE.sub("-", text.strip().lower()).strip("-") or "page"


def page_id(number, title):
    return f"{number:03d}-{slugify(title)}"


def code_block(language, code):
    """A language-tagged, escaped code block; the fence carries the class the highlighter needs."""
    return (f'<pre class="code"><code class="language-{escape(language)}">'
            f"{escape(code)}</code></pre>")


def exercise_page(exercise, previous, following):
    """One exercise page with navigation to its neighbours."""
    lines = ["<!doctype html>", "<html><head><meta charset=\"utf-8\">",
             f"<title>{escape(exercise['title'])}</title></head>", "<body>",
             f"<h1 id=\"{page_id(exercise['number'], exercise['title'])}\">"
             f"{escape(exercise['title'])}</h1>",
             f"<p class=\"tier\">Tier: {escape(exercise['tier'])}</p>",
             f"<section class=\"editorial\">{escape(exercise['editorial'])}</section>"]
    for language in LANGUAGES:
        code = exercise["solutions"].get(language)
        if code is not None:
            lines.append(f"<h2>Solution: {escape(language)}</h2>")
            lines.append(code_block(language, code))
    navigation = []
    if previous is not None:
        navigation.append(f'<a href="{previous["file"]}">prev</a>')
    if following is not None:
        navigation.append(f'<a href="{following["file"]}">next</a>')
    lines.append(f"<nav class=\"pager\">{' '.join(navigation) or 'no neighbours'}</nav>")
    lines.append('<p><a href="index.html">index</a></p>')
    lines.append("</body></html>")
    return "\n".join(lines) + "\n"


def build_site(exercises=EXERCISES):
    """Render every page; returns file -> html, with exercises in numeric order."""
    ordered = sorted(exercises, key=lambda item: item["number"])
    pages = {}
    for index, exercise in enumerate(ordered):
        previous = None
        following = None
        if index > 0:
            previous = {"number": ordered[index - 1]["number"], "title": ordered[index - 1]["title"],
                        "file": f"{page_id(ordered[index - 1]['number'], ordered[index - 1]['title'])}.html"}
        if index + 1 < len(ordered):
            following = {"number": ordered[index + 1]["number"],
                         "title": ordered[index + 1]["title"],
                         "file": f"{page_id(ordered[index + 1]['number'], ordered[index + 1]['title'])}.html"}
        pages[f"{page_id(exercise['number'], exercise['title'])}.html"] = \
            exercise_page(exercise, previous, following)
    pages["index.html"] = index_page(ordered, pages)
    return pages


def index_page(ordered, pages):
    """The index lists every exercise exactly once, in numeric order."""
    lines = ["<!doctype html>", "<html><head><meta charset=\"utf-8\"><title>Exercises</title></head>",
             "<body>", "<h1 id=\"exercise-index\">Exercises</h1>", "<ol class=\"exercise-list\">"]
    for exercise in ordered:
        target = f"{page_id(exercise['number'], exercise['title'])}.html"
        lines.append(f'<li><a href="{target}">{escape(exercise["title"])}</a> '
                     f'<span class="tier">{escape(exercise["tier"])}</span></li>')
    lines.append("</ol>")
    lines.append('<p><a href="index.html">index</a></p>')
    lines.append("</body></html>")
    return "\n".join(lines) + "\n"


def links(pages):
    """Every href in the site, in page order, with the page it came from."""
    return [(page, href) for page in sorted(pages) for href in HREF_RE.findall(pages[page])]


def dangling(pages):
    """hrefs that do not resolve to a generated page or an anchor inside one."""
    return [(page, href) for page, href in links(pages)
            if not href.startswith("#") and href not in pages]


def code_classes(pages):
    return sorted({match for page in pages for match in
                   re.findall(r'class="language-([a-z]+)"', pages[page])})


def pager(html):
    """The navigation block only - the surrounding prose may mention 'next' for other reasons."""
    match = NAV_RE.search(html)
    return match.group(1) if match else ""


def main():
    pages = build_site()

    # 1) happy path: one page per exercise plus the index, and the index lists every exercise once
    #    in numeric order.
    index = pages["index.html"]
    titles = [exercise["title"] for exercise in sorted(EXERCISES, key=lambda item: item["number"])]
    check(len(pages) == len(EXERCISES) + 1
          and [index.count(f">{escape(title)}</a>") for title in titles] == [1, 1, 1]
          and index.index("001-reverse-a-string") < index.index("002-fizz-buzz")
          < index.index("003-merge-intervals"),
          f"index must list every exercise once in order: {sorted(pages)}")

    # 2) link integrity: an independent href sweep finds no dangling target, and the pager links do
    #    exist in the page set.
    check(dangling(pages) == [] and len(links(pages)) == 3 * len(EXERCISES) + 2
          and all(href in pages for _, href in links(pages) if not href.startswith("#")),
          f"dangling links: {dangling(pages)}")

    # 3) escaping: the hostile payload in the editorial appears escaped and never as live markup.
    check("&lt;script&gt;alert(1)&lt;/script&gt;" in pages["002-fizz-buzz.html"]
          and "<script>" not in pages["002-fizz-buzz.html"]
          and "&lt;Buzz&gt;" in index and "<Buzz>" not in index
          and "&lt;odd&gt;" in pages["001-reverse-a-string.html"],
          "interpolated prose must be escaped, not executed")

    # 4) code blocks: one per available language, fenced with the language class the highlighter
    #    looks for, and the code itself escaped.
    classes = code_classes(pages)
    check(classes == ["csharp", "java", "javascript", "python"]
          and pages["001-reverse-a-string.html"].count('class="code"')
          == len(EXERCISES[0]["solutions"]) == 4,
          f"code blocks wrong: {classes}")

    # 5) navigation boundaries: the first page has no prev, the last has no next, and the middle
    #    page points at both neighbours.
    first, middle, last = (pages["001-reverse-a-string.html"], pages["002-fizz-buzz.html"],
                           pages["003-merge-intervals.html"])
    check(NAV_RE.search(first) and NAV_RE.search(middle) and NAV_RE.search(last)
          and "prev" not in pager(first) and "next" in pager(first)
          and "prev" in pager(last) and "next" not in pager(last)
          and "001-reverse-a-string.html" in pager(middle)
          and "003-merge-intervals.html" in pager(middle)
          and NAV_RE.search(first) and NAV_RE.search(last),
          f"prev/next wrong: {pager(first)!r} {pager(middle)!r} {pager(last)!r}")

    # 6) ids are unique: no page repeats an element id.
    repeated = [page for page in pages
                if len(re.findall(r'id="([^"]+)"', pages[page]))
                != len(set(re.findall(r'id="([^"]+)"', pages[page])))]
    check(repeated == [] and page_id(3, "Fizz & <Buzz>") == "003-fizz-buzz"
          and len(re.findall(r'id="([^"]+)"', pages["003-merge-intervals.html"])) == 1,
          f"element ids must be unique per page: {repeated}")

    # 7) boundary: an empty site still publishes a valid index with no list entries and no links.
    empty = build_site(())
    check(sorted(empty) == ["index.html"] and "exercise-list" in empty["index.html"]
          and dangling(empty) == [] and links(empty) == [("index.html", "index.html")]
          and "<li>" not in empty["index.html"],
          f"the empty site must still render an index: {sorted(empty)}")

    # 8) slugify boundaries: punctuation, casing and whitespace collapse to a single-dash slug.
    check(slugify("  Merge   Intervals!! ") == "merge-intervals"
          and slugify("Fizz & <Buzz>") == "fizz-buzz" and slugify("!!!") == "page"
          and page_id(7, "A/B") == "007-a-b",
          f"slugify boundaries wrong: {slugify('!!!')} / {slugify('Fizz & <Buzz>')}")

    # 9) determinism: rebuilding produces byte-identical pages, and the exercise input order does
    #    not matter.
    reordered = build_site(tuple(reversed(EXERCISES)))
    check(build_site() == pages and reordered == pages,
          "the generator must be deterministic and order independent")

    print(f"pages={len(pages)} links={len(links(pages))} dangling={len(dangling(pages))} "
          f"languages={classes}")
    print(f"index_entries={index.count('<li>')} first_has_next={'next' in first} "
          f"last_has_prev={'prev' in last} empty_site={sorted(empty)}")
    print("PASS")


if __name__ == "__main__":
    main()
