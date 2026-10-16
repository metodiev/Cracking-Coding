"""Task 993 - Implement academic citation generator for editorial solutions with references.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A bibliography engine for editorial solutions.  Records are validated (title, year, venue and
at least one given/family author are mandatory) and de-duplicated by DOI, then rendered in four
styles - APA, MLA, IEEE and BibTeX - from one canonical ordering (author surname, year, title)
so the printed workbook and the machine-readable export never disagree.  IEEE numbering and
BibTeX citation keys follow that same order, and a key gets an a/b/c suffix only when one
author publishes more than once in a year.

Simplified versus production: four styles are implemented and author lists print in full
(no et-al truncation).  Verified: exact rendered bytes for every style, the ordering oracle,
DOI de-duplication, mandatory-field and author-shape rejections, citation-key disambiguation,
brace/quote sanitisation, and the single-record and empty-bibliography boundaries.
"""
import hashlib

STYLES = ("APA", "MLA", "IEEE", "BibTeX")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_error(action, needle):
    try:
        action()
    except ValueError as exc:
        check(needle in str(exc), "error %r does not mention %r" % (str(exc), needle))
        return
    raise AssertionError("expected a ValueError mentioning %r" % needle)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sanitize(text):
    """Neutralise characters that would break the quoting rules of the rendered styles."""
    return text.replace("{", "(").replace("}", ")").replace('"', "'").strip()


def initials(given):
    return " ".join(part[0] + "." for part in given.replace("-", " ").split())


def validate_record(record):
    for field in ("title", "year", "venue", "authors"):
        if not record.get(field):
            raise ValueError("record %r is missing %s" % (record.get("title", "<untitled>"), field))
    for author in record["authors"]:
        if not author.get("family") or not author.get("given"):
            raise ValueError("author %r needs both given and family names" % (author,))


def apa_authors(authors):
    parts = ["%s, %s" % (sanitize(a["family"]), initials(a["given"])) for a in authors]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + ", & " + parts[-1]


def mla_authors(authors):
    head = "%s, %s" % (sanitize(authors[0]["family"]), authors[0]["given"])
    rest = ["%s %s" % (a["given"], sanitize(a["family"])) for a in authors[1:]]
    return head + (", and " + " and ".join(rest) if rest else "")


def ieee_authors(authors):
    names = ["%s %s" % (initials(a["given"]), sanitize(a["family"])) for a in authors]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def apa_entry(record):
    tail = ", ".join(str(item) for item in (record.get("volume"), record.get("pages")) if item)
    text = "%s (%d). %s. %s%s." % (apa_authors(record["authors"]), record["year"],
                                   sanitize(record["title"]), sanitize(record["venue"]),
                                   ", " + tail if tail else "")
    return text + (" https://doi.org/%s" % record["doi"] if record.get("doi") else "")


def mla_entry(record):
    text = '%s. "%s." %s' % (mla_authors(record["authors"]), sanitize(record["title"]),
                             sanitize(record["venue"]))
    text += ", vol. %s" % record["volume"] if record.get("volume") else ""
    text += ", %d" % record["year"]
    text += ", pp. %s" % record["pages"] if record.get("pages") else ""
    return text + "."


def ieee_entry(record, number):
    text = '[%d] %s, "%s," %s' % (number, ieee_authors(record["authors"]),
                                  sanitize(record["title"]), sanitize(record["venue"]))
    text += ", vol. %s" % record["volume"] if record.get("volume") else ""
    text += ", pp. %s" % record["pages"] if record.get("pages") else ""
    return text + ", %d." % record["year"]


def bibtex_entry(record, key):
    fields = {"author": " and ".join("%s, %s" % (a["family"], a["given"]) for a in record["authors"]),
              "title": sanitize(record["title"]), "journal": sanitize(record["venue"]),
              "year": str(record["year"])}
    for optional in ("volume", "pages", "doi"):
        if record.get(optional):
            fields[optional] = str(record[optional])
    body = ",\n".join("  %s = {%s}" % (name, fields[name]) for name in sorted(fields))
    return "@article{%s,\n%s\n}" % (key, body)


class Bibliography:
    def __init__(self):
        self.records = []
        self.duplicates = []

    def add(self, record):
        """Validate, then de-duplicate by DOI: a repeat is reported instead of stored twice."""
        validate_record(record)
        doi = (record.get("doi") or "").lower()
        if doi and any((stored.get("doi") or "").lower() == doi for stored in self.records):
            self.duplicates.append(doi)
            return False
        self.records.append(record)
        return True

    def ordered(self):
        return sorted(self.records, key=lambda item: (item["authors"][0]["family"].lower(),
                                                     item["year"], item["title"].lower()))

    def cite_keys(self):
        ordered = self.ordered()
        bases = ["%s%d" % (item["authors"][0]["family"].lower(), item["year"]) for item in ordered]
        seen, keys = {}, {}
        for record, base in zip(ordered, bases):
            if bases.count(base) == 1:
                keys[record["title"]] = base
                continue
            index = seen.get(base, 0)
            seen[base] = index + 1
            keys[record["title"]] = "%s%s" % (base, chr(ord("a") + index))
        return keys

    def render(self, style):
        if style not in STYLES:
            raise ValueError("unsupported style %r" % (style,))
        keys, entries = self.cite_keys(), []
        for number, item in enumerate(self.ordered(), start=1):
            if style == "APA":
                entries.append(apa_entry(item))
            elif style == "MLA":
                entries.append(mla_entry(item))
            elif style == "IEEE":
                entries.append(ieee_entry(item, number))
            else:
                entries.append(bibtex_entry(item, keys[item["title"]]))
        return "\n".join(entries)

    def digest(self):
        return sha256_text("\n".join("%s|%s" % (style, self.render(style)) for style in STYLES))


def record(title, family, given, year=1843, doi=None):
    return {"title": title, "year": year, "venue": "Scientific Memoirs", "volume": "3",
            "pages": "666-731", "doi": doi, "authors": [{"given": given, "family": family}]}


def main():
    primary = {"title": "Notes on the Analytical Engine", "year": 1843, "venue": "Scientific Memoirs",
               "volume": "3", "pages": "666-731", "doi": "10.1/abc",
               "authors": [{"given": "Ada", "family": "Lovelace"},
                           {"given": "Grace", "family": "Hopper"}]}
    library = Bibliography()
    check(library.add(primary), "a valid record must be accepted")

    # Exact rendering for every style, hand-written straight from the citation rules.
    check(library.render("APA") == "Lovelace, A., & Hopper, G. (1843). Notes on the Analytical "
          "Engine. Scientific Memoirs, 3, 666-731. https://doi.org/10.1/abc",
          "APA rendering drifted: %r" % (library.render("APA"),))
    check(library.render("MLA") == 'Lovelace, Ada, and Grace Hopper. "Notes on the Analytical '
          'Engine." Scientific Memoirs, vol. 3, 1843, pp. 666-731.',
          "MLA rendering drifted: %r" % (library.render("MLA"),))
    check(library.render("IEEE") == '[1] A. Lovelace and G. Hopper, "Notes on the Analytical '
          'Engine," Scientific Memoirs, vol. 3, pp. 666-731, 1843.',
          "IEEE rendering drifted: %r" % (library.render("IEEE"),))
    check(library.render("BibTeX") == "@article{lovelace1843,\n"
          "  author = {Lovelace, Ada and Hopper, Grace},\n  doi = {10.1/abc},\n"
          "  journal = {Scientific Memoirs},\n  pages = {666-731},\n"
          "  title = {Notes on the Analytical Engine},\n  volume = {3},\n  year = {1843}\n}",
          "BibTeX rendering drifted: %r" % (library.render("BibTeX"),))
    expect_error(lambda: library.render("Chicago"), "unsupported style")

    # Ordering oracle: brute-force sort of (surname, year, title) over a fixed library.
    alpha = Bibliography()
    for index, (family, year, title) in enumerate(
            [("Zuse", 1941, "Plankalkuel"), ("Babbage", 1837, "On Engines"),
             ("Lovelace", 1842, "Sketch"), ("Lovelace", 1842, "Another Sketch"),
             ("Babbage", 1832, "Difference Engine"), ("Hopper", 1952, "Compiling Routines")]):
        alpha.add(record(title, family, "Author", year=year, doi="10.9/%d" % index))
    brute = sorted(alpha.records, key=lambda item: (item["authors"][0]["family"].lower(),
                                                   item["year"], item["title"].lower()))
    check([item["title"] for item in alpha.ordered()] == [item["title"] for item in brute],
          "canonical ordering disagrees with the brute-force sort")
    check(alpha.ordered()[0]["title"] == "Difference Engine"
          and alpha.render("IEEE").splitlines()[0].startswith("[1] A. Babbage,"),
          "the earliest entry must lead both the order and the IEEE numbering")

    # Citation keys: same author twice in one year gets an a/b suffix, unique bases stay bare.
    keys = alpha.cite_keys()
    expected_keys = {"Difference Engine": "babbage1832", "On Engines": "babbage1837",
                     "Plankalkuel": "zuse1941", "Compiling Routines": "hopper1952",
                     "Another Sketch": "lovelace1842a", "Sketch": "lovelace1842b"}
    check(keys == expected_keys, "citation keys drifted: %r" % (keys,))
    # DOI de-duplication: the repeat is reported and never rendered.
    before = len(library.records)
    duplicate = dict(primary, title="A Verbatim Reprint", doi="10.1/ABC")
    check(not library.add(duplicate), "a repeated DOI must be rejected")
    check(library.duplicates == ["10.1/abc"] and len(library.records) == before
          and "A Verbatim Reprint" not in library.render("APA"),
          "the duplicate must be logged and excluded from every style")

    # Mandatory fields and author shape.
    expect_error(lambda: library.add({"title": "No Year", "venue": "X",
                                      "authors": [{"given": "A", "family": "B"}]}), "year")
    expect_error(lambda: library.add({"title": "Bad Author", "year": 1900, "venue": "X",
                                      "authors": [{"family": "B"}]}), "given and family")
    expect_error(lambda: library.add({"title": "No Authors", "year": 1900, "venue": "X",
                                      "authors": []}), "authors")
    # Sanitisation: braces balanced for BibTeX, quotes neutralised for MLA.
    messy = Bibliography()
    messy.add(record('Curly {braces} and "quotes"', "Knuth", "Donald", year=1974, doi="10.8/1"))
    check(messy.render("BibTeX").count("{") == messy.render("BibTeX").count("}")
          and "(braces)" in messy.render("BibTeX") and "'quotes'" in messy.render("MLA")
          and '"quotes"' not in messy.render("MLA"),
          "braces and quotes inside a title must be neutralised")

    # Boundary: a single record is numbered [1]; an empty bibliography renders nothing.
    solo = Bibliography()
    solo.add(record("Only Paper", "Hopper", "Grace", year=1952, doi="10.5/1"))
    empty = Bibliography()
    empty_digest = hashlib.sha256("\n".join("%s|" % style for style in STYLES).encode("utf-8")).hexdigest()
    check(solo.render("IEEE").startswith("[1] G. Hopper,") and empty.render("APA") == ""
          and empty.render("BibTeX") == "" and empty.cite_keys() == {}
          and empty.digest() == empty_digest,
          "a lone record must be [1]; an empty bibliography renders nothing but still digests")

    print("records=%d styles=%s duplicates=%d digest=%s"
          % (len(alpha.records), len(STYLES), len(library.duplicates), alpha.digest()[:16]))
    print("PASS")


if __name__ == "__main__":
    main()
