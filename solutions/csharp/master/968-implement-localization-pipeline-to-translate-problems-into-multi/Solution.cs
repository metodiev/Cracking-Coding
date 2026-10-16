// Task 968 - Implement localization pipeline to translate problems into multiple languages.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// A BCP-47 style tag ("pt-BR") is parsed and expanded into a fallback chain (pt-BR -> pt -> en).
// Message keys are resolved through that chain; when the key carries a count the CLDR plural
// category of the resolved language is appended (items.one / items.few / items.many), and
// {0}/{1} placeholders are interpolated from the call arguments.
//
// Verified: English, Russian and Polish plural selection against a hand-written CLDR table (the
// independent oracle) plus a seeded random cross-check over 40 counts, the fallback-chain order,
// the "return the key itself when nothing matches" negative path, deterministic interpolation and
// the empty / unknown locale boundary.
// Simplification: in-memory catalogues for five locales with only the plural categories exercised
// below (no ICU, no file or network loading); non-English strings are ASCII transliterations.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

public static class Solution
{
    private const string DefaultLocale = "en";

    private static ulong _state;

    private static ulong NextUInt64()
    {
        unchecked
        {
            _state += 0x9E3779B97F4A7C15UL;
            ulong z = _state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }

    private static double NextUnit() => (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private sealed class Locale
    {
        public readonly string Language, Region;

        public Locale(string language, string region)
        {
            Language = language;
            Region = region;
        }

        public override string ToString() => Region.Length == 0 ? Language : Language + "-" + Region;
    }

    private static Locale ParseTag(string tag)
    {
        if (string.IsNullOrWhiteSpace(tag)) return new Locale(DefaultLocale, "");
        string[] parts = tag.Trim().Replace('_', '-').Split('-');
        string language = parts[0].ToLowerInvariant();
        string region = parts.Length > 1 ? parts[1].ToUpperInvariant() : "";
        return new Locale(language, region);
    }

    /** pt-BR -> pt -> en; the default locale is only appended when it is not already present. */
    private static List<string> FallbackChain(string tag)
    {
        Locale locale = ParseTag(tag);
        var chain = new List<string> { locale.ToString() };
        if (locale.Region.Length > 0) chain.Add(locale.Language);
        if (!chain.Contains(DefaultLocale)) chain.Add(DefaultLocale);
        return chain;
    }

    private static string PluralCategory(string language, long n)
    {
        switch (language)
        {
            case "ru":
                if (n % 10 == 1 && n % 100 != 11) return "one";
                if (n % 10 >= 2 && n % 10 <= 4 && (n % 100 < 12 || n % 100 > 14)) return "few";
                return "many";
            case "pl":
                if (n == 1) return "one";
                if (n % 10 >= 2 && n % 10 <= 4 && (n % 100 < 12 || n % 100 > 14)) return "few";
                return "many";
            default:
                return n == 1 ? "one" : "other";
        }
    }

    private static bool IsNumeric(object value) => value is int || value is long || value is short || value is byte;

    private static List<string> CandidateKeys(string key, string language, object[] args)
    {
        var candidates = new List<string>();
        if (args.Length > 0 && IsNumeric(args[0]))
        {
            long n = Math.Abs(Convert.ToInt64(args[0], CultureInfo.InvariantCulture));
            candidates.Add(key + "." + PluralCategory(language, n));
            if (language != DefaultLocale) candidates.Add(key + ".other");
        }
        candidates.Add(key);
        return candidates;
    }

    private static string Translate(
        Dictionary<string, Dictionary<string, string>> catalog, string tag, string key, params object[] args)
    {
        foreach (string locale in FallbackChain(tag))
        {
            if (!catalog.TryGetValue(locale, out Dictionary<string, string> table)) continue;
            string language = ParseTag(locale).Language;
            foreach (string candidate in CandidateKeys(key, language, args))
                if (table.TryGetValue(candidate, out string pattern)) return Interpolate(pattern, args);
        }
        return key;
    }

    private static string Interpolate(string pattern, object[] args)
    {
        var builder = new StringBuilder();
        for (int i = 0; i < pattern.Length; i++)
        {
            if (pattern[i] == '{')
            {
                int close = pattern.IndexOf('}', i);
                string slot = close > i ? pattern.Substring(i + 1, close - i - 1) : null;
                if (slot != null
                    && int.TryParse(slot, NumberStyles.Integer, CultureInfo.InvariantCulture, out int index)
                    && index >= 0 && index < args.Length)
                {
                    builder.Append(Convert.ToString(args[index], CultureInfo.InvariantCulture));
                    i = close;
                    continue;
                }
            }
            builder.Append(pattern[i]);
        }
        return builder.ToString();
    }

    private static Dictionary<string, Dictionary<string, string>> BuildCatalog()
    {
        var catalog = new Dictionary<string, Dictionary<string, string>>(StringComparer.Ordinal);
        catalog["en"] = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            { "items.one", "{0} item" }, { "items.other", "{0} items" },
            { "hint", "Hint {1}: {0}" },
            { "downloads.one", "{0} download" }, { "downloads.other", "{0} downloads" },
        };
        catalog["pt"] = new Dictionary<string, string>(StringComparer.Ordinal)
        { { "items.one", "{0} item (pt)" }, { "items.other", "{0} itens (pt)" } };
        catalog["pt-BR"] = new Dictionary<string, string>(StringComparer.Ordinal)
        { { "items.one", "{0} item (pt-BR)" }, { "items.other", "{0} itens (pt-BR)" } };
        catalog["ru"] = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            { "items.one", "{0} predmet" }, { "items.few", "{0} predmeta" }, { "items.many", "{0} predmetov" },
            { "downloads.one", "{0} zagruzka" }, { "downloads.few", "{0} zagruzki" }, { "downloads.many", "{0} zagruzok" },
        };
        catalog["pl"] = new Dictionary<string, string>(StringComparer.Ordinal)
        { { "items.one", "{0} element" }, { "items.few", "{0} elementy" }, { "items.many", "{0} elementow" } };
        return catalog;
    }

    public static void Main()
    {
        var catalog = BuildCatalog();

        // Happy path: English plural selection and ordered placeholder interpolation.
        Check(Translate(catalog, "en", "items", 1) == "1 item" && Translate(catalog, "en", "items", 7) == "7 items",
            "singular and plural counts select the English 'one' and 'other' forms");
        Check(Translate(catalog, "en", "hint", "binary search", 3) == "Hint 3: binary search",
            "placeholders {0} and {1} are filled from the argument list in order");

        // Boundary: region beats language beats default, and unknown locales resolve to English.
        Check(Translate(catalog, "pt-BR", "items", 1) == "1 item (pt-BR)"
              && Translate(catalog, "pt-PT", "items", 1) == "1 item (pt)",
            "a region locale is preferred, and an untranslated region falls back to its language");
        Check(Translate(catalog, "fr", "items", 2) == "2 items" && FallbackChain("pt-BR").Count == 3,
            "an unknown locale falls back to the default, and the chain is pt-BR -> pt -> en");        Check(string.Join(",", FallbackChain("pt-BR")) == "pt-BR,pt,en"
              && string.Join(",", FallbackChain("en")) == "en",
            "the chain is pt-BR -> pt -> en, and the default locale never appears twice in its own chain");

        // Negative: nothing matches anywhere, so the key itself is returned instead of throwing.
        Check(Translate(catalog, "zz", "no.such.key", 5) == "no.such.key", "a missing key degrades to the key itself");

        // Oracle: hand-written CLDR categories for n = 1..30, independent of the rule implementation.
        string[] ruExpected =
        {
            "one", "few", "few", "few", "many", "many", "many", "many", "many", "many", "many", "many", "many", "many", "many",
            "many", "many", "many", "many", "many", "one", "few", "few", "few", "many", "many", "many", "many", "many", "many",
        };
        string[] plExpected =
        {
            "one", "few", "few", "few", "many", "many", "many", "many", "many", "many", "many", "many", "many", "many", "many",
            "many", "many", "many", "many", "many", "many", "few", "few", "few", "many", "many", "many", "many", "many", "many",
        };
        int mismatches = 0;
        for (int n = 1; n <= 30; n++)
        {
            if (PluralCategory("ru", n) != ruExpected[n - 1]) mismatches++;
            if (PluralCategory("pl", n) != plExpected[n - 1]) mismatches++;
            if (PluralCategory("en", n) != (n == 1 ? "one" : "other")) mismatches++;
        }
        Check(mismatches == 0, "all 30 hand-written CLDR categories match for ru, pl and en (found " + mismatches + ")");
        Check(PluralCategory("ru", 11) == "many" && PluralCategory("ru", 21) == "one" && PluralCategory("ru", 111) == "many",
            "the Russian 11/21/111 boundary cases are classified many/one/many");

        // Seeded random cross-check: the resolved pattern must agree with the plural category.
        _state = 968UL;
        int one = 0, few = 0, many = 0;
        for (int i = 0; i < 40; i++)
        {
            long n = (long)(NextUnit() * 200.0);
            string category = PluralCategory("ru", n);
            string text = Translate(catalog, "ru", "downloads", n);
            string expected = "zagruz" + (category == "one" ? "ka" : category == "few" ? "ki" : "ok");
            Check(text == n + " " + expected, "random count " + n + " renders the " + category + " form");
            if (category == "one") one++;
            else if (category == "few") few++;
            else many++;
        }
        Check(one > 0 && few > 0 && many > 0, "the sampled counts reach all three Russian categories");

        int translated = 0, probes = 0;
        foreach (string tag in new[] { "en", "pt-BR", "ru", "pl", "fr" })
            foreach (string key in new[] { "items", "hint" })
                foreach (int count in new[] { 1, 5 })
                {
                    probes++;
                    if (Translate(catalog, tag, key, count) != key) translated++;
                }
        Check(translated == probes, "every (locale,key) probe resolves through the fallback chain ("
            + translated + "/" + probes + ")");
        Check(Translate(catalog, "ru", "runtime", 3) == "runtime",
            "a key absent from every catalogue stays unresolved even for a known locale");

        Check(catalog.Count == 5 && FallbackChain("pt-BR").Count == 3, "five catalogues load and the chain has three entries");
        Console.WriteLine("locales: " + catalog.Count + ", plural categories verified: ru/pl/en over n=1..30 plus 40 random counts, probes resolved "
            + translated + "/" + probes);
        Console.WriteLine("PASS");
    }
}
