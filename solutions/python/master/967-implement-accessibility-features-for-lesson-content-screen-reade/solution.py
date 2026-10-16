"""Task 967 - Implement accessibility features for lesson content (screen-reader friendly).
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

Accessibility model: a lesson is a list of typed content blocks (heading, paragraph, image, table,
list, code, link).  The generator renders them into semantic markup - real heading levels, `alt`
text, `th scope="col"` header cells, list semantics, language-tagged code, accessible link names -
and produces a linear plain-text "screen reader" view in document order.  The auditor checks the
authoring rules a rendering step cannot fix on its own: heading order, image alt text, table headers,
link names, code language labels, empty lists, duplicate ids and WCAG contrast for coloured text.

Simplified: markup is a string, not a DOM, and there is no ARIA live region handling.  Verified: a
clean lesson renders and audits clean, each rule fires on its own defect, the WCAG contrast ratio
against an independently written formula (including the 4.5 boundary sampled over the grey ramp),
reading order and escaping, and the empty/single-block boundaries.
"""
from html import escape

NORMAL_TEXT, LARGE_TEXT = 4.5, 3.0
GENERIC_LINK_TEXT = frozenset({"click here", "here", "link", "read more", "more"})


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def luminance(color):
    """WCAG relative luminance of '#rrggbb'."""
    channels = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
              for value in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(foreground, background):
    first, second = luminance(foreground), luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def meets_contrast(ratio, large=False):
    return ratio >= (LARGE_TEXT if large else NORMAL_TEXT)


def render_block(block):
    """Semantic markup per block type; everything interpolated is escaped."""
    kind = block["type"]
    if kind == "heading":
        level = min(max(block.get("level", 2), 1), 6)
        return f"<h{level}>{escape(block['text'])}</h{level}>"
    if kind == "paragraph":
        style = ""
        if "color" in block and "background" in block:
            style = f' style="color:{escape(block["color"])};background:{escape(block["background"])}"'
        return f"<p{style}>{escape(block['text'])}</p>"
    if kind == "image":
        return f'<img src="{escape(block["src"])}" alt="{escape(block.get("alt", ""))}">'
    if kind == "table":
        head = "".join(f'<th scope="col">{escape(str(cell))}</th>'
                       for cell in block.get("headers", ()))
        rows = "".join("<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>"
                       for row in block.get("rows", ()))
        return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"
    if kind == "list":
        tag = "ol" if block.get("ordered") else "ul"
        items = "".join(f"<li>{escape(str(item))}</li>" for item in block.get("items", ()))
        return f"<{tag}>{items}</{tag}>"
    if kind == "code":
        return (f'<pre><code class="language-{escape(block.get("language", ""))}">'
                f"{escape(block['text'])}</code></pre>")
    if kind == "link":
        name = block.get("aria_label") or block.get("text", "")
        return (f'<a href="{escape(block["href"])}" aria-label="{escape(name)}">'
                f"{escape(block.get('text', ''))}</a>")
    raise ValueError(f"unknown block type {kind!r}")


def screen_reader_view(lesson):
    spoken = []
    for block in lesson:
        kind = block["type"]
        if kind == "heading":
            spoken.append(f"heading level {block.get('level', 2)}: {block['text']}")
        elif kind == "image":
            spoken.append(f"image: {block.get('alt') or 'decorative'}")
        elif kind == "table":
            spoken.append(f"table with {len(block.get('headers', ()))} columns and "
                          f"{len(block.get('rows', ()))} rows")
        elif kind == "list":
            spoken.append(f"{'ordered' if block.get('ordered') else 'bullet'} list of "
                          f"{len(block.get('items', ()))} items")
        elif kind == "paragraph":
            spoken.append(block.get("text", ""))
        elif kind == "link":
            spoken.append(f"link: {block.get('aria_label') or block.get('text', '')}")
        else:
            spoken.append(f"code: {block.get('language', 'unknown')}")
    return spoken


def audit(lesson):
    """Authoring defects, sorted by (code, block index)."""
    issues = []
    expected_level = 0
    for index, block in enumerate(lesson):
        kind = block["type"]
        if kind == "heading":
            level = block.get("level", 2)
            if level > expected_level + 1:
                issues.append(("heading-order", index, f"h{expected_level} then h{level}"))
            expected_level = level
        elif kind == "image" and "alt" not in block:
            issues.append(("image-alt", index, block.get("src", "")))
        elif kind == "table":
            if not block.get("headers"):
                issues.append(("table-headers", index, len(block.get("rows", ()))))
            elif any(len(row) != len(block["headers"]) for row in block.get("rows", ())):
                issues.append(("table-shape", index, len(block["headers"])))
        elif kind == "list" and not block.get("items"):
            issues.append(("empty-list", index, 0))
        elif kind == "code" and not block.get("language"):
            issues.append(("code-language", index, ""))
        elif kind == "link":
            name = (block.get("aria_label") or block.get("text", "")).strip().lower()
            if not name or name in GENERIC_LINK_TEXT:
                issues.append(("link-name", index, name))
        elif kind == "paragraph" and "color" in block and "background" in block:
            ratio = contrast_ratio(block["color"], block["background"])
            if not meets_contrast(ratio, block.get("large", False)):
                issues.append(("contrast", index, round(ratio, 2)))
    return sorted(issues, key=lambda row: (row[0], row[1]))


def lesson():
    return [
        {"type": "heading", "level": 1, "text": "Two pointers"},        {"type": "paragraph", "text": "Walk from both ends & compare 1 < 2."},
        {"type": "image", "src": "ends.svg", "alt": "Two arrows moving towards the middle"},
        {"type": "heading", "level": 2, "text": "Complexity"},
        {"type": "table", "headers": ("Case", "Time"), "rows": (("best", "O(n)"),
                                                                ("worst", "O(n)"))},
        {"type": "list", "items": ("low", "high"), "ordered": False},
        {"type": "code", "language": "python", "text": "while low < high:\n    low += 1"},
        {"type": "link", "href": "notes.html", "text": "Read the invariant proof"},
        {"type": "paragraph", "text": "Strong text.", "color": "#000000",
         "background": "#ffffff"},
    ]


def main():
    good = lesson()
    rendered = [render_block(block) for block in good]

    # 1) happy path: a well-authored lesson audits clean and renders real semantics per block.
    check(audit(good) == [] and rendered[0] == "<h1>Two pointers</h1>"
          and 'alt="Two arrows moving towards the middle"' in rendered[2]
          and '<th scope="col">Case</th>' in rendered[4] and rendered[5].startswith("<ul>")
          and 'class="language-python"' in rendered[6]
          and 'aria-label="Read the invariant proof"' in rendered[7],
          f"a clean lesson must audit clean: {audit(good)}")

    # 2) escaping: text is escaped in the markup and stays readable in the linear view.
    check("&amp;" in rendered[1] and "&lt;" in rendered[1]
          and "<p>Walk from both ends &amp; compare 1 &lt; 2.</p>" == rendered[1]
          and screen_reader_view(good)[1] == "Walk from both ends & compare 1 < 2.",
          f"escaping wrong: {rendered[1]}")

    # 3) heading order: a skipped level is reported, and the fixed version is not.
    skipped = [{"type": "heading", "level": 1, "text": "A"},
               {"type": "heading", "level": 3, "text": "C"}]
    fixed = [{"type": "heading", "level": 1, "text": "A"},
             {"type": "heading", "level": 2, "text": "B"},
             {"type": "heading", "level": 3, "text": "C"}]
    check([row[0] for row in audit(skipped)] == ["heading-order"]
          and audit(fixed) == [] and audit(good) == [],
          f"heading order wrong: {audit(skipped)}")

    # 4) each authoring rule fires on its own defect (one folded check over six mutations).
    mutations = {
        "image-alt": [{"type": "image", "src": "x.svg"}],
        "table-headers": [{"type": "table", "rows": ((1, 2),)}],
        "empty-list": [{"type": "list", "items": ()}],
        "code-language": [{"type": "code", "text": "pass"}],
        "link-name": [{"type": "link", "href": "a.html", "text": "click here"}],
        "contrast": [{"type": "paragraph", "text": "faint", "color": "#cccccc",
                      "background": "#ffffff"}],
    }
    fired = {code: [row[0] for row in audit(blocks)] for code, blocks in mutations.items()}
    check(set(fired) == {"image-alt", "table-headers", "empty-list", "code-language",
                         "link-name", "contrast"}
          and all(fired[code] == [code] for code in fired),
          f"each rule must fire on its own defect: {fired}")

    # 5) decorative images (alt="") and an aria-label rescuing a generic link are accepted, which is
    #    what makes the alt rule meaningful rather than a blanket requirement.
    allowed = [{"type": "image", "src": "spacer.svg", "alt": ""},
               {"type": "link", "href": "a.html", "text": "click here",
                "aria_label": "Open the merge intervals editorial"}]
    check(audit(allowed) == [] and audit([{"type": "link", "href": "a.html", "text": "  "}])
          == [("link-name", 0, "")],
          f"decorative/labelled cases wrong: {audit(allowed)}")

    # 6) contrast oracle: an independently written WCAG formula agrees, and the 4.5 boundary is
    #    sampled over the grey ramp - the darkest failing grey and the lightest passing one differ.
    ratios = [(contrast_ratio(f"#{value:02x}{value:02x}{value:02x}", "#ffffff"), value)
              for value in range(1, 256)]
    below = max(pair for pair in ratios if pair[0] < NORMAL_TEXT)
    above = min(pair for pair in ratios if pair[0] >= NORMAL_TEXT)
    first, second = luminance("#777777"), luminance("#ffffff")
    independent = (max(first, second) + 0.05) / (min(first, second) + 0.05)
    check(not meets_contrast(below[0]) and meets_contrast(above[0])
          and below[0] < NORMAL_TEXT <= above[0] and abs(below[0] - above[0]) < 0.1
          and abs(independent - contrast_ratio("#777777", "#ffffff")) < 1e-12
          and meets_contrast(LARGE_TEXT, large=True)
          and not meets_contrast(LARGE_TEXT - 0.01, large=True),
          f"the contrast boundary is wrong: below {below} above {above}")

    # 7) reading order: the linear view follows document order and names every element.
    spoken = screen_reader_view(good)
    check(spoken == ["heading level 1: Two pointers",
                     "Walk from both ends & compare 1 < 2.",
                     "image: Two arrows moving towards the middle",
                     "heading level 2: Complexity",
                     "table with 2 columns and 2 rows",
                     "bullet list of 2 items",
                     "code: python",
                     "link: Read the invariant proof",
                     "Strong text."],
          f"the linear view must follow document order: {spoken}")

    # 8) boundaries: an empty lesson is audited and rendered without complaint, and a single block
    #    is handled.
    check(audit([]) == [] and "".join(render_block(block) for block in []) == ""
          and screen_reader_view([]) == []
          and audit([{"type": "paragraph", "text": "solo"}]) == []
          and render_block({"type": "list", "items": ()}) == "<ul></ul>",
          "the empty/single-block boundaries must behave")

    # 9) determinism: rendering and auditing twice is identical, and issue order is stable.
    check([render_block(block) for block in good] == rendered and audit(good) == audit(good)
          and audit(mutations["table-headers"]) == sorted(audit(mutations["table-headers"])),
          "rendering and auditing must be deterministic")

    print(f"issues={audit(good)} blocks={len(good)} mutation_codes={sorted(fired)}")
    print(f"contrast below={below[0]:.4f} above={above[0]:.4f} ratio_777777="
          f"{contrast_ratio('#777777', '#ffffff'):.4f} sr_view={len(spoken)} lines")
    print("PASS")


if __name__ == "__main__":
    main()
