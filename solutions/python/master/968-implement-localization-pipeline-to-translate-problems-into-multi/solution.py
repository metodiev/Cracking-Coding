"""Task 968 - Implement localization pipeline to translate problems into multiple languages.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

An in-memory localization pipeline.  A source catalog holds the English problem strings plus one pluralized
message; locale bundles hold the translations.  A lookup for "pt-BR" walks a fallback chain - "pt-BR", then
"pt", then the source locale - so a partially translated locale still renders.  Lookups interpolate
{placeholders}, pluralized messages pick a CLDR-style category (English one/other, Russian one/few/many/other,
Japanese other only), and the pipeline reports per-locale coverage, every fallback taken, and every translation
whose placeholder set differs from the source - the check that catches "translated, but broke the format string".

Simplified versus gettext/Fluent: no files, no catalogs on disk, no gender or ordinal categories.  Verified:
fallback order, coverage arithmetic on the extracted key set, placeholder parity and interpolation errors,
plural categories at the 0/1/2/11/21/22/101 boundaries, pseudo-localization expansion with placeholders
preserved, unused/unknown key detection, empty catalogs, and reproducibility of the reports.
"""
import re

SOURCE_LOCALE = "en"
PLACEHOLDER_RE = re.compile(r"\{([^{}]*)\}")
CATALOG = {
    "problem.two-sum.title": "Two Sum",
    "problem.two-sum.body": "Return the indices of the two numbers in {array} summing to {target}.",
    "problem.two-sum.hint": "Hash each value; it needs {target} minus itself.",
    "ui.attempts": "Recorded {count} attempts by learner {learner}.",
    "ui.progress": "Solved {solved} of {total} problems.",
}
# "de" deliberately breaks a placeholder; "pt-BR" is partial so the fallback chain has work to do.
BUNDLES = {
    "en": dict(CATALOG),
    "de": {"problem.two-sum.title": "Zwei Summe",
           "problem.two-sum.body": "Gib die Indizes der zwei Zahlen aus {array} zurueck.",
           "ui.progress": "Geloest: {solved} von {total} Aufgaben."},
    "pt": {"problem.two-sum.title": "Soma de Dois",
           "problem.two-sum.body": "Devolve os indices dos dois numeros em {array} com soma {target}.",
           "ui.progress": "Resolveste {solved} de {total} problemas."},
    "pt-BR": {"problem.two-sum.title": "Soma de Dois Numeros"},
    "ru": {"problem.two-sum.title": "Сумма двух", "ui.progress": "Решено {solved} из {total} задач."},
}
PLURAL_FORMS = {
    "en": {"problem.passed": {"one": "You passed {count} test", "other": "You passed {count} tests"}},
    "ru": {"problem.passed": {"one": "Пройден {count} тест", "few": "Пройдено {count} теста",
                              "many": "Пройдено {count} тестов", "other": "Пройдено {count} теста"}},
    "ja": {"problem.passed": {"other": "{count}件のテストに合格しました"}},
}
# keys a problem page really references - everything else is unused inventory
EXTRACTED_KEYS = ("problem.two-sum.title", "problem.two-sum.body", "problem.two-sum.hint",
                  "ui.progress", "problem.passed")
ACCENTS = str.maketrans(dict(zip("abcdefghijklmnopqrstuvwxyz", "áƀçðéƒĝĥíĵķļɱñóþɋŕšţúṽŵẋýž")))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect(error, call, message):
    """Asserts that `call` raises `error`; used for the negative (rejected input) checks."""
    try:
        call()
    except error:
        return
    raise AssertionError(message)


def parse_locale(tag):
    """'pt-BR' -> ('pt', 'BR'); underscores and @-extensions are folded away."""
    parts = tag.split("@")[0].replace("_", "-").split("-")
    region = next((part.upper() for part in parts[1:] if len(part) == 2), "")
    return parts[0].lower(), region


def fallback_chain(tag):
    """Most specific first: the full tag, then its language, then the source locale."""
    language, region = parse_locale(tag)
    return ([f"{language}-{region}"] if region else []) + [language] \
        + ([SOURCE_LOCALE] if language != SOURCE_LOCALE else [])


def placeholders(template):
    return frozenset(PLACEHOLDER_RE.findall(template))


def interpolate(template, params):
    if template.count("{") != template.count("}"):
        raise ValueError(f"unbalanced placeholder braces in {template!r}")
    return PLACEHOLDER_RE.sub(lambda m: _value(m, params, template), template)


def _value(match, params, template):
    name = match.group(1)
    if name not in params:
        raise KeyError(f"missing value for placeholder {name!r} in {template!r}")
    return str(params[name])


def plural_category(language, count):
    """CLDR integer categories for the languages this demo ships."""
    mod10, mod100 = abs(count) % 10, abs(count) % 100
    if language == "ru":
        if mod10 == 1 and mod100 != 11:
            return "one"
        if 2 <= mod10 <= 4 and not 12 <= mod100 <= 14:
            return "few"
        if mod10 == 0 or 5 <= mod10 <= 9 or 11 <= mod100 <= 14:
            return "many"
        return "other"
    return "other" if language == "ja" or abs(count) != 1 else "one"


class Translator:
    """Resolves keys through the fallback chain, recording every fallback and every failure."""

    def __init__(self, bundles, plural_forms):
        self.bundles, self.plural_forms, self.fallbacks, self.failures = bundles, plural_forms, [], []

    def _body(self, key, locale, via_plural):
        for candidate in fallback_chain(locale):
            language = parse_locale(candidate)[0]
            body = self.bundles.get(candidate, {}).get(key)
            if body is None and via_plural and candidate == language:
                body = self.plural_forms.get(language, {}).get(key)
            if body:
                if candidate != locale:
                    self.fallbacks.append((locale, key, candidate))
                return body
        self.failures.append((locale, key))
        raise KeyError(f"key {key!r} is not available for {locale!r}")

    def translate(self, key, locale, **params):
        return interpolate(self._body(key, locale, False), params)

    def translate_plural(self, key, locale, count, **params):
        forms = self._body(key, locale, True)
        template = forms.get(plural_category(parse_locale(locale)[0], count)) or forms["other"]
        return interpolate(template, dict(params, count=count))


def placeholder_parity(bundles, catalog):
    """(locale, key, missing, extra) for every translation that disagrees with the source."""
    return sorted((locale, key, tuple(sorted(placeholders(catalog[key]) - placeholders(body))),
                   tuple(sorted(placeholders(body) - placeholders(catalog[key]))))
                  for locale, bundle in bundles.items() if locale != SOURCE_LOCALE
                  for key, body in bundle.items() if key in catalog
                  and placeholders(body) != placeholders(catalog[key]))


def coverage(bundles, keys, plural_forms):
    """locale -> (translated, total, ratio), plus the sorted missing-key list per locale."""
    report, missing = {}, {}
    for locale, bundle in sorted(bundles.items()):
        forms = plural_forms.get(parse_locale(locale)[0], {})
        absent = sorted(key for key in keys if key not in bundle and not forms.get(key))
        report[locale] = (len(keys) - len(absent), len(keys), (len(keys) - len(absent)) / len(keys))
        missing[locale] = absent
    return report, missing


def key_hygiene(catalog, extracted):
    return (sorted(k for k in catalog if k not in extracted), sorted(k for k in extracted if k not in catalog))


def pseudo_localize(text):
    """Test locale: accented letters, bracket padding, >=30% expansion, placeholders intact."""
    parts = PLACEHOLDER_RE.split(text)
    accented = "".join("{" + part + "}" if index % 2 else part.translate(ACCENTS)
                       for index, part in enumerate(parts))
    return "[" + accented.ljust(int(len(accented) * 1.35) + 4, "·") + "]"


def main():
    translator = Translator(BUNDLES, PLURAL_FORMS)
    # 1) happy path: a locale renders its own string, interpolated, without touching the fallback chain.
    check(translator.translate("problem.two-sum.title", "de") == "Zwei Summe"
          and translator.translate("ui.progress", "pt", solved=3, total=7) == "Resolveste 3 de 7 problemas."
          and translator.fallbacks == [], "direct hits must render the locale's own translation")
    # 2) fallback chain: pt-BR -> pt for the body, pt-BR -> pt -> en for the hint.  Oracle: chain order.
    check(translator.translate("problem.two-sum.body", "pt-BR", array="[2,7,11]", target=9)
          == "Devolve os indices dos dois numeros em [2,7,11] com soma 9."
          and translator.translate("problem.two-sum.hint", "pt-BR", target=9).startswith("Hash")
          and fallback_chain("pt-BR") == ["pt-BR", "pt", "en"] and fallback_chain("ja") == ["ja", "en"]
          and translator.fallbacks == [("pt-BR", "problem.two-sum.body", "pt"),
                                       ("pt-BR", "problem.two-sum.hint", "en")],
          f"fallback resolution is wrong: {translator.fallbacks}")
    # 3) boundary: an unknown locale degrades to the source locale; an unknown key fails and is recorded.
    check(translator.translate("problem.two-sum.title", "xx-YY") == "Two Sum"
          and fallback_chain("xx-YY") == ["xx-YY", "xx", "en"], "an unknown locale must degrade to the source")
    expect(KeyError, lambda: translator.translate("problem.does-not-exist", "de"), "unknown key must not resolve")
    check(translator.failures == [("de", "problem.does-not-exist")], f"unresolved: {translator.failures}")
    # 4) interpolation contract: a missing value and an unbalanced brace are both rejected.
    expect(KeyError, lambda: translator.translate("problem.two-sum.body", "pt", target=9), "missing value")
    expect(ValueError, lambda: interpolate("value {a", {"a": 1}), "unbalanced braces")
    check(interpolate("{a}+{b}", {"a": 2, "b": 3}) == "2+3" and placeholders("x {a} {b}") == {"a", "b"}
          and placeholders("plain") == frozenset(), "interpolation and extraction are inconsistent")
    # 5) plural categories at the boundaries - English, Russian (11/21/22/25/101) and Japanese.
    ru = [plural_category("ru", n) for n in (0, 1, 2, 5, 11, 21, 22, 25, 101)]
    check(ru == ["many", "one", "few", "many", "many", "one", "few", "many", "one"]
          and [plural_category("en", n) for n in (0, 1, 2)] == ["other", "one", "other"]
          and plural_category("ja", 1) == "other" and plural_category("ru", -1) == "one",
          f"plural categories are wrong: {ru}")
    check(translator.translate_plural("problem.passed", "en", 1) == "You passed 1 test"
          and translator.translate_plural("problem.passed", "en", 2) == "You passed 2 tests"
          and translator.translate_plural("problem.passed", "ru", 22) == "Пройдено 22 теста"
          and translator.translate_plural("problem.passed", "ja", 5) == "5件のテストに合格しました",
          "plural rendering must follow the category of the locale")
    # 6) the parity scan finds exactly the planted German defect (oracle: hand-counted defects).
    defects = placeholder_parity(BUNDLES, CATALOG)
    check(defects == [("de", "problem.two-sum.body", ("target",), ())], f"parity defects: {defects}")
    # 7) coverage arithmetic, and each locale's missing list, against the extracted key set.
    report, missing = coverage(BUNDLES, EXTRACTED_KEYS, PLURAL_FORMS)
    check(report["en"] == (5, 5, 1.0) and report["pt"][:2] == (3, 5)
          and missing["de"] == ["problem.passed", "problem.two-sum.hint"]
          and missing["pt-BR"] == ["problem.passed", "problem.two-sum.body", "problem.two-sum.hint",
                                   "ui.progress"]
          and sum(row[0] for row in report.values()) == 15, f"coverage wrong: {report} / {missing}")
    # 8) key hygiene: the plural-only key is unknown inventory, the unused UI string is unused.
    unused, unknown = key_hygiene(CATALOG, EXTRACTED_KEYS)
    check(unused == ["ui.attempts"] and unknown == ["problem.passed"] and key_hygiene({}, ()) == ([], []),
          f"unused {unused} / unknown {unknown} key detection is wrong")
    # 9) pseudo-localization: bracketed, non-ASCII, expanded >=30%, placeholders preserved.
    sample, pseudo = CATALOG["ui.attempts"], pseudo_localize(CATALOG["ui.attempts"])
    rendered = interpolate(pseudo, {"count": 2, "learner": "ada"})
    check(pseudo.startswith("[") and pseudo.endswith("]") and any(ord(c) > 127 for c in pseudo)
          and len(pseudo) >= 1.3 * len(sample) and placeholders(pseudo) == placeholders(sample)
          and "ada" in rendered and not placeholders(rendered) and pseudo_localize(sample) == pseudo,
          f"pseudo-localization must expand and preserve placeholders: {pseudo!r}")
    # 10) empty-catalog boundaries, plus reproducibility of the whole report pipeline.
    check(coverage({}, EXTRACTED_KEYS, {})[0] == {} and placeholder_parity({}, CATALOG) == []
          and pseudo_localize("") == "[····]" and coverage(BUNDLES, EXTRACTED_KEYS, PLURAL_FORMS)[0] == report,
          "empty catalogs must not break the pipeline")
    summary = ", ".join(f"{loc}:{row[0]}/{row[1]}" for loc, row in report.items())
    print(f"locales={len(BUNDLES)} keys={len(EXTRACTED_KEYS)} coverage={{{summary}}}")
    print(f"fallbacks={len(translator.fallbacks)} parity_defects={len(defects)} unused={unused}")
    print("PASS")


if __name__ == "__main__":
    main()
