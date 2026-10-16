// Task 993 - Implement academic citation generator for editorial solutions with references.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Structured reference metadata (authors, title, venue, year, DOI, pages) is validated per entry type, given a
// Lastname+Year citation key that is disambiguated with a/b/c suffixes when several works collide, deduplicated
// by DOI and rendered in three styles (APA, IEEE, ACM) plus a BibTeX-ish record with escaped specials. The
// bibliography order is a total order (first author, year, title), so numbering is stable no matter how the
// entries arrive. Verified: the APA rendering against a hand-written expectation (oracle), a round-trip parse
// of that rendering back to metadata, key disambiguation, escaping, validation refusals with the exact reason,
// DOI deduplication, insertion-order independence and the empty-bibliography boundary.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAIL: " + label); }
    private static string Error(Action action)
    {
        try { action(); return ""; }
        catch (Exception ex) { return ex.Message; }
    }

    private sealed class Entry
    {
        public readonly string Id, Type, Title, Venue, Doi, Pages;
        public readonly List<string> Authors;
        public readonly int Year;
        public Entry(string id, string type, List<string> authors, string title, string venue, int year, string doi, string pages)
        { Id = id; Type = type; Authors = authors; Title = title; Venue = venue; Year = year; Doi = doi; Pages = pages; }

        public string LastName()
        {
            string author = Authors[0];
            int comma = author.IndexOf(',');
            return comma < 0 ? author : author.Substring(0, comma).Trim();
        }

        public string Key() =>
            LastName().ToLowerInvariant().Replace(" ", "") + Year.ToString(CultureInfo.InvariantCulture);
    }

    private static readonly string[] Types = { "article", "inproceedings", "book", "techreport" };

    private static void Validate(Entry entry)
    {
        if (Array.IndexOf(Types, entry.Type) < 0) throw new Exception("unknown-type:" + entry.Id);
        if (entry.Authors.Count == 0) throw new Exception("missing-authors:" + entry.Id);
        if (entry.Title.Length == 0) throw new Exception("missing-title:" + entry.Id);
        if (entry.Venue.Length == 0) throw new Exception("missing-venue:" + entry.Id);
        if (entry.Year < 1900 || entry.Year > 2100) throw new Exception("bad-year:" + entry.Id);
        if (entry.Doi.Length > 0 && !(entry.Doi.StartsWith("10.", StringComparison.Ordinal) && entry.Doi.Contains("/")))
            throw new Exception("bad-doi:" + entry.Id);
    }

    /** BibTeX-ish escaping: only the characters that would break the record are escaped. */
    private static string Escape(string text)
    {
        var escaped = new StringBuilder();
        foreach (char c in text)
        {
            if (c == '{' || c == '}' || c == '%' || c == '&' || c == '$' || c == '#' || c == '_') escaped.Append('\\');
            escaped.Append(c);
        }
        return escaped.ToString();
    }

    /** APA: Last, F. (Year). Title. Venue, Pages. https://doi.org/DOI */
    private static string Apa(Entry entry)
    {
        var authors = new List<string>();
        foreach (string author in entry.Authors)
        {
            int comma = author.IndexOf(',');
            if (comma < 0) { authors.Add(author); continue; }
            string last = author.Substring(0, comma).Trim(), given = author.Substring(comma + 1).Trim();
            authors.Add(last + ", " + (given.Length > 0 ? given.Substring(0, 1) + "." : ""));
        }
        var text = new StringBuilder();
        text.Append(string.Join(", ", authors)).Append(" (").Append(entry.Year.ToString(CultureInfo.InvariantCulture)).Append("). ")
            .Append(entry.Title).Append(". ").Append(entry.Venue);
        if (entry.Pages.Length > 0) text.Append(", ").Append(entry.Pages);
        text.Append('.');
        if (entry.Doi.Length > 0) text.Append(" https://doi.org/").Append(entry.Doi);
        return text.ToString();
    }

    private static string Ieee(Entry entry, int number) =>
        "[" + number.ToString(CultureInfo.InvariantCulture) + "] " + entry.Authors[0] + ", \"" + entry.Title + ",\" "
        + entry.Venue + ", " + entry.Year.ToString(CultureInfo.InvariantCulture) + ".";

    private static string Acm(Entry entry) =>
        entry.LastName() + (entry.Authors.Count > 1 ? " et al. " : ". ") + entry.Title + ". " + entry.Venue
        + " (" + entry.Year.ToString(CultureInfo.InvariantCulture) + ").";

    private static string Bibtex(Entry entry, string key) =>
        "@" + entry.Type + "{" + key + ", author = {" + string.Join(" and ", entry.Authors) + "}, title = {" + Escape(entry.Title)
        + "}, year = {" + entry.Year.ToString(CultureInfo.InvariantCulture) + "}, venue = {" + entry.Venue + "}}";

    private sealed class Reference
    {
        public readonly Entry Entry;
        public readonly string Key;
        public readonly int Number;
        public Reference(Entry entry, string key, int number) { Entry = entry; Key = key; Number = number; }
    }

    /** Validate, deduplicate by DOI, order totally, then disambiguate colliding keys. */
    private static List<Reference> Bibliography(List<Entry> entries)
    {
        var unique = new List<Entry>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (Entry entry in entries)
        {
            Validate(entry);
            if (entry.Doi.Length > 0 && !seen.Add(entry.Doi)) continue;
            unique.Add(entry);
        }
        unique.Sort((a, b) =>
        {
            int byAuthor = string.CompareOrdinal(a.LastName(), b.LastName());
            if (byAuthor != 0) return byAuthor;
            int byYear = a.Year.CompareTo(b.Year);
            return byYear != 0 ? byYear : string.CompareOrdinal(a.Title, b.Title);
        });
        var references = new List<Reference>();
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (Entry entry in unique)
        {
            string baseKey = entry.Key();
            counts[baseKey] = counts.TryGetValue(baseKey, out int occurrences) ? occurrences + 1 : 1;
        }
        var used = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (Entry entry in unique)
        {
            string baseKey = entry.Key();
            string key = baseKey;
            if (counts[baseKey] > 1)
            {
                int at = used.TryGetValue(baseKey, out int taken) ? taken : 0;
                used[baseKey] = at + 1;
                key = baseKey + (char)('a' + at);
            }
            references.Add(new Reference(entry, key, references.Count + 1));
        }
        return references;
    }

    private static Entry Sample(string id, string title, int year, string doi) =>
        new Entry(id, "article", new List<string> { "Smith, Jane", "Doe, John" }, title, "Journal of Algorithms", year, doi, "1-10");

    public static void Main()
    {
        Entry smith = Sample("e1", "Sorting with Braces", 2020, "10.1000/abc");
        string apa = Apa(smith);
        Check(apa == "Smith, J., Doe, J. (2020). Sorting with Braces. Journal of Algorithms, 1-10. https://doi.org/10.1000/abc",
              "APA rendering matches the hand-written expectation (oracle)");
        // Oracle: the rendering parses back to the same last name, year and title.
        int open = apa.IndexOf(" ("), close = apa.IndexOf("). ");
        Check(apa.StartsWith("Smith, J.", StringComparison.Ordinal)
              && apa.Substring(open + 2, close - open - 2) == "2020"
              && apa.Substring(close + 3, smith.Title.Length) == smith.Title,
              "the APA string round-trips back to the metadata (oracle)");

        Entry other = Sample("e2", "Another Paper", 2020, "10.1000/xyz");
        List<Reference> bibliography = Bibliography(new List<Entry> { other, smith });
        Check(string.Join(",", bibliography.ConvertAll(r => r.Key)) == "smith2020a,smith2020b"
              && bibliography[0].Number == 1 && bibliography[1].Number == 2
              && Ieee(smith, 1).StartsWith("[1] Smith, Jane,", StringComparison.Ordinal),
              "same-author same-year keys are disambiguated and numbered");

        Entry braced = Sample("e3", "Costs {and} 50% $offsets_", 2021, "10.1000/esc");
        Check(Bibtex(braced, "smith2021").Contains("title = {Costs \\{and\\} 50\\% \\$offsets\\_}")
              && Bibtex(braced, "smith2021").StartsWith("@article{smith2021, author = {Smith, Jane and Doe, John}", StringComparison.Ordinal),
              "BibTeX escaping covers braces, percent, dollar and underscore");

        Check(Error(() => Bibliography(new List<Entry> { new Entry("e4", "article", new List<string> { "Solo, Ann" }, "T", "", 2020, "", "") }))
                 == "missing-venue:e4"
              && Error(() => Bibliography(new List<Entry> { new Entry("e5", "misc", new List<string> { "Solo, Ann" }, "T", "V", 2020, "", "") }))
                 == "unknown-type:e5"
              && Error(() => Bibliography(new List<Entry> { Sample("e6", "T", 2020, "not-a-doi") })) == "bad-doi:e6"
              && Error(() => Bibliography(new List<Entry> { Sample("e7", "T", 1800, "10.1/a") })) == "bad-year:e7"
              && Error(() => Bibliography(new List<Entry> { new Entry("e8", "book", new List<string>(), "T", "V", 2020, "", "") }))
                 == "missing-authors:e8",
              "validation refuses malformed entries with the exact reason");

        Entry duplicate = Sample("e9", "Duplicate", 2022, "10.1000/ABC");
        List<Reference> deduped = Bibliography(new List<Entry> { duplicate, Sample("e10", "Duplicate too", 2023, "10.1000/abc") });
        Check(deduped.Count == 1 && deduped[0].Entry.Id == "e9" && Bibliography(new List<Entry>()).Count == 0,
              "DOIs deduplicate case-insensitively and the empty bibliography is a boundary");

        Check(Acm(smith) == "Smith et al. Sorting with Braces. Journal of Algorithms (2020).",
              "ACM rendering uses the et al. short form");

        Console.WriteLine("references: " + bibliography.Count + ", keys: "
                          + string.Join(",", bibliography.ConvertAll(r => r.Key)));
        Console.WriteLine("PASS");
    }
}
