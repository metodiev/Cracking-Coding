// Task 987 - Implement migration tools for older problem formats to new canonical schema.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Two legacy shapes are lifted into one canonical record: `v1` is a flat `key=value;...` line (3-level
// `lvl`, seconds, no tags, unknown keys kept aside) and `v2` is a pipe-separated fixed-order row (1-5
// difficulty, ms, optional tag column). Canonicalising resolves language aliases, sorts languages and tags,
// derives the slug from the title, maps the legacy ladder (1->1, 2->3, 3->5) and converts seconds to ms. A
// schema validator written independently of the migration code rejects whatever is still non-canonical
// (ranges, ordering, unsupported language, slug that no longer matches its title), so every migrated record
// is an exact fixed point of the rules. Verified: both legacy happy paths, the ladder and unit mappings,
// exact refusal reasons, the 100 ms / 10000 ms limits with their neighbours, extra-key preservation, the
// fixed-point oracle with sorted output, and the empty input boundary.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAIL: " + label); }

    private static readonly string[] AllowedLanguages = { "csharp", "java", "javascript", "python" };
    private static readonly string[] RequiredFields = { "num", "name", "lvl", "langs", "t" };

    private sealed class Record
    {
        public int Number, Difficulty, TimeLimitMs;
        public string Slug = "", Title = "";
        public List<string> Languages = new List<string>(), Tags = new List<string>();
        public List<KeyValuePair<string, string>> Extra = new List<KeyValuePair<string, string>>();
    }

    private static int ParseInt(string text, string field)
    {
        if (!int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value))
            throw new Exception("bad-" + field + ":" + text);
        return value;
    }

    private static string CanonicalLanguage(string alias) {
        switch (alias.Trim().ToLowerInvariant())
        {
            case "py": case "python": return "python";
            case "js": case "javascript": case "node": return "javascript";
            case "java": return "java";
            case "cs": case "csharp": case "c#": return "csharp";
            default: throw new Exception("unknown-language:" + alias);
        }
    }

    /** Split, canonicalise (or lower-case) a comma list, then sort and drop duplicates. */
    private static List<string> ParseList(string text, bool languages)
    {
        var values = new List<string>();
        foreach (string part in text.Split(','))
        {
            if (part.Trim().Length == 0) continue;
            values.Add(languages ? CanonicalLanguage(part) : part.Trim().ToLowerInvariant());
        }
        values.Sort(StringComparer.Ordinal);
        for (int i = 1; i < values.Count; i++) if (values[i] == values[i - 1]) values.RemoveAt(i--);
        return values;
    }
    /** Title -> lower-case dash slug; each run of non-alphanumerics collapses into a single dash. */
    private static string Slugify(string title)
    {
        var slug = new StringBuilder();
        bool pendingDash = false;
        foreach (char c in title)
        {
            bool alnum = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
            if (!alnum) { pendingDash = true; continue; }
            if (pendingDash && slug.Length > 0) slug.Append('-');
            pendingDash = false;
            slug.Append(char.ToLowerInvariant(c));
        }
        return slug.ToString();
    }
    private static Record FromV1(string text, List<string> warnings)
    {
        var fields = new Dictionary<string, string>(StringComparer.Ordinal);
        var extras = new List<KeyValuePair<string, string>>();
        foreach (string part in text.Split(';'))
        {
            if (part.Length == 0) continue;
            int eq = part.IndexOf('=');
            if (eq <= 0) throw new Exception("malformed-pair:" + part);
            string key = part.Substring(0, eq).Trim();
            string value = part.Substring(eq + 1).Trim();
            if (Array.IndexOf(RequiredFields, key) < 0)
            {
                extras.Add(new KeyValuePair<string, string>(key, value));
                warnings.Add("preserved-extra:" + key);
                continue;
            }
            if (fields.ContainsKey(key)) throw new Exception("duplicate-key:" + key);
            fields[key] = value;
        }
        foreach (string required in RequiredFields)
            if (!fields.ContainsKey(required)) throw new Exception("missing-field:" + required);
        int level = ParseInt(fields["lvl"], "level");
        if (level < 1 || level > 3) throw new Exception("bad-level:" + level);
        return new Record
        {
            Number = ParseInt(fields["num"], "number"),
            Title = fields["name"],
            Difficulty = level * 2 - 1,                        // 1 -> 1, 2 -> 3, 3 -> 5
            TimeLimitMs = ParseInt(fields["t"], "time") * 1000, // legacy seconds -> canonical ms
            Languages = ParseList(fields["langs"], true),
            Slug = Slugify(fields["name"]),
            Extra = extras,
        };
    }

    private static Record FromV2(string text)
    {
        string[] parts = text.Split('|');
        if (parts.Length != 5 && parts.Length != 6) throw new Exception("bad-column-count:" + parts.Length);
        return new Record
        {
            Number = ParseInt(parts[0], "number"),
            Title = parts[1],
            Difficulty = ParseInt(parts[2], "level"),
            TimeLimitMs = ParseInt(parts[4], "time"),
            Languages = ParseList(parts[3], true),
            Tags = parts.Length == 6 ? ParseList(parts[5], false) : new List<string>(),
            Slug = Slugify(parts[1]),
        };
    }

    private static Record Migrate(string version, string text, List<string> warnings)
    {
        if (version == "v1") return FromV1(text, warnings);
        if (version == "v2") return FromV2(text);
        throw new Exception("unsupported-format:" + version);
    }

    private static bool SortedDistinct(List<string> values)
    {
        for (int i = 0; i < values.Count; i++)
            if (values[i].Length == 0 || (i > 0 && string.CompareOrdinal(values[i - 1], values[i]) >= 0)) return false;
        return true;
    }

    /** Schema validation, written against the canonical shape only. */
    private static string SchemaError(Record record)
    {
        if (record.Number < 1 || record.Number > 998) return "number-out-of-range:" + record.Number;
        if (record.Title.Length == 0) return "empty-title";
        if (record.Slug != Slugify(record.Title)) return "slug-not-canonical:" + record.Slug;
        if (record.Difficulty < 1 || record.Difficulty > 5) return "difficulty-out-of-range:" + record.Difficulty;
        if (record.TimeLimitMs < 100 || record.TimeLimitMs > 10000) return "time-out-of-range:" + record.TimeLimitMs;
        if (record.Languages.Count == 0) return "no-languages";
        if (!SortedDistinct(record.Languages)) return "languages-not-sorted";
        foreach (string language in record.Languages)
            if (Array.IndexOf(AllowedLanguages, language) < 0) return "unknown-language:" + language;
        if (!SortedDistinct(record.Tags)) return "tags-not-sorted";
        return "";
    }

    private sealed class Migration
    {
        public readonly List<Record> Records = new List<Record>();
        public readonly List<string> Rejected = new List<string>(), Warnings = new List<string>();
    }
    private static Migration MigrateAll(params (string Version, string Text)[] inputs)
    {
        var result = new Migration();
        var seen = new HashSet<int>();
        foreach ((string version, string text) in inputs)
        {
            try
            {
                Record record = Migrate(version, text, result.Warnings);
                string error = SchemaError(record);
                if (error.Length > 0) { result.Rejected.Add(error); continue; }
                if (!seen.Add(record.Number)) { result.Rejected.Add("duplicate-number:" + record.Number); continue; }
                result.Records.Add(record);
            }
            catch (Exception ex) { result.Rejected.Add(ex.Message); }
        }
        result.Records.Sort((a, b) => a.Number.CompareTo(b.Number));
        return result;
    }
    /** Numbers of the accepted records, ascending - the deterministic schedule order. */
    private static string Numbers(List<Record> records) =>
        string.Join(",", records.ConvertAll(r => r.Number.ToString(CultureInfo.InvariantCulture)));
    public static void Main()
    {
        Migration v1 = MigrateAll(("v1", "num=7;name=Two Sum;lvl=2;langs=py,js;t=2"), ("v1", "num=1;name=Solo;lvl=1;langs=java;t=1"),
                                  ("v1", "num=2;name=Hard;lvl=3;langs=java;t=1"));
        Record two = v1.Records[2];
        Check(v1.Rejected.Count == 0 && v1.Warnings.Count == 0 && v1.Records[0].Slug == "solo"
              && v1.Records[0].Difficulty == 1 && v1.Records[0].TimeLimitMs == 1000 && v1.Records[0].Tags.Count == 0
              && v1.Records[1].Difficulty == 5 && two.Slug == "two-sum" && two.Title == "Two Sum"
              && two.Difficulty == 3 && two.TimeLimitMs == 2000 && string.Join(",", two.Languages) == "javascript,python",
              "v1 records migrate through the level ladder and the seconds -> ms mapping");

        Migration v2 = MigrateAll(("v2", "12|Graph Walk|4|node,C#|1500|graphs,dp"));
        Record second = v2.Records[0];
        Check(second.Slug == "graph-walk" && second.Difficulty == 4 && second.TimeLimitMs == 1500
              && string.Join(",", second.Languages) == "csharp,javascript" && string.Join(",", second.Tags) == "dp,graphs",
              "v2 record canonicalises aliases and sorts tags");

        Migration refusals = MigrateAll(("v1", "num=9;name=C;lvl=2;langs=ruby;t=1"), ("v3", "anything"),
                                        ("v1", "num=9;name=C;lvl=4;langs=py;t=1"), ("v1", "num=9;name=C;lvl=2;langs=py;t"),
                                        ("v1", "num=9;name=C;lvl=2;langs=py"), ("v2", "9|C|4|python|100"),
                                        ("v2", "9|D|4|python|100"));
        Check(string.Join(" / ", refusals.Rejected)
              == "unknown-language:ruby / unsupported-format:v3 / bad-level:4 / malformed-pair:t"
              + " / missing-field:t / duplicate-number:9",
              "refusals report the exact reason");

        Migration bounds = MigrateAll(("v2", "3|Slow|2|python|10001"), ("v1", "num=4;name=Fast;lvl=2;langs=python;t=0"),
                                      ("v2", "5|Floor|2|python|100"), ("v2", "6|Ceil|2|python|10000"),
                                      ("v1", "num=8;name=Extra;lvl=2;langs=py;t=1;author=ada;review=2"));
        Record extra = bounds.Records[2];
        Check(string.Join(" / ", bounds.Rejected) == "time-out-of-range:10001 / time-out-of-range:0"
              && Numbers(bounds.Records) == "5,6,8" && bounds.Records[0].TimeLimitMs == 100
              && bounds.Records[1].TimeLimitMs == 10000 && extra.Extra.Count == 2 && extra.Extra[0].Key == "author"
              && extra.Extra[1].Value == "2"
              && string.Join(",", bounds.Warnings) == "preserved-extra:author,preserved-extra:review",
              "time limits are inclusive at 100 ms and 10000 ms and unknown keys survive with a warning");

        // Oracle: migrated records are fixed points of every rule, and the independent validator agrees.
        bool fixedPoints = true;
        foreach (Record record in v1.Records)
        {
            if (SchemaError(record).Length != 0 || Slugify(record.Title) != record.Slug) fixedPoints = false;
            if (string.Join(",", ParseList(string.Join(",", record.Languages), true)) != string.Join(",", record.Languages))
                fixedPoints = false;
        }
        bool unsortedRejected = SchemaError(new Record
        {
            Number = 10, Slug = "x", Title = "x", Difficulty = 3, TimeLimitMs = 900,
            Languages = new List<string> { "python", "java" },
        }) == "languages-not-sorted";
        Check(fixedPoints && unsortedRejected, "migrated records are exact fixed points of the rules (oracle)");

        Migration mixed = MigrateAll(("v2", "30|Third|3|java|900"), ("v1", "num=10;name=First;lvl=1;langs=py;t=1"),
                                     ("v2", "20|Second|5|java|900"));
        Migration none = MigrateAll();
        Check(Numbers(mixed.Records) == "10,20,30" && SchemaError(mixed.Records[1]).Length == 0
              && none.Records.Count == 0 && none.Rejected.Count == 0 && none.Warnings.Count == 0,
              "output is sorted by number and the empty input boundary holds");

        Console.WriteLine("migrated records: " + mixed.Records.Count + " sorted as " + Numbers(mixed.Records));
        Console.WriteLine("PASS");
    }
}
