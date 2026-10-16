// Task 994 - Implement editorial writing assistant that drafts explanations per solution.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// The assistant drafts an editorial from a solution's verified facts (complexity, check count, edge cases,
// data structures) by filling section templates, choosing a phrasing variant per section from a seeded
// splitmix64 stream so identical facts always produce the same draft. Every template declares the fact ids it
// relies on and only variants whose citations are all present in the facts are eligible; when no variant of a
// section is eligible the draft is refused as an unsupported claim rather than written. The draft must stay
// inside a word budget and quotes the measured complexity verbatim. Verified: all five sections present, no
// mention of a fact that was not provided, refusal of an unsupported claim and of an unknown approach, verbatim
// quoting of complexity / check count / edges / structures, the minimal-facts skeleton, budget enforcement and
// reproducible regeneration.
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

    private sealed class Stream
    {
        private ulong _state;
        public Stream(ulong seed) { _state = seed; }
        public ulong Next()
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
    }

    private sealed class Facts
    {
        public string Number = "", Title = "", Approach = "", Time = "", Space = "";
        public int Checks;
        public List<string> Edges = new List<string>(), Structures = new List<string>();

        public bool Has(string id)
        {
            switch (id)
            {
                case "number": return Number.Length > 0;
                case "title": return Title.Length > 0;
                case "approach": return Approach.Length > 0;
                case "time": return Time.Length > 0;
                case "space": return Space.Length > 0;
                case "checks": return Checks > 0;
                case "edges": return Edges.Count > 0;
                case "structures": return Structures.Count > 0;
                default: return false;
            }
        }
    }

    /** A section template plus the fact ids it claims; only fully satisfied templates are eligible. */
    private sealed class Template
    {
        public readonly string Section, Text;
        public readonly string[] Cites;
        public Template(string section, string text, params string[] cites)
        { Section = section; Text = text; Cites = cites; }
    }

    private static readonly string[] Sections = { "introduction", "approach", "complexity", "correctness", "edge-cases" };
    private static readonly string[] Approaches = { "two-pointers", "hash-map", "dynamic-programming", "graph-search" };

    private static readonly List<Template> Script = new List<Template>
    {
        new Template("introduction", "Problem {number} - {title} - is documented here with its verified facts.", "number", "title"),
        new Template("introduction", "This editorial covers problem {number}, {title}.", "number", "title"),
        new Template("approach", "The {approach} approach is used, keeping the invariant explicit at every step.", "approach"),
        new Template("approach", "An editorial solution based on {approach} is presented.", "approach"),
        new Template("complexity", "Time complexity is {time} and the extra space is {space}.", "time", "space"),
        new Template("complexity", "The running time is {time}, with {space} auxiliary space.", "time", "space"),
        new Template("correctness", "The implementation is exercised by {checks} checks, including boundary cases.", "checks"),
        new Template("correctness", "Verification covers {checks} checks over generated and adversarial inputs.", "checks"),
        new Template("edge-cases", "Edge cases covered: {edges}; data structures: {structures}.", "edges", "structures"),
        new Template("edge-cases", "The draft lists these edges: {edges}, relying on {structures}.", "edges", "structures"),
        new Template("edge-cases", "No edge cases beyond the {checks} checks were recorded.", "checks"),
    };

    private static string Fill(Template template, Facts facts)
    {
        string edges = facts.Edges.Count == 0 ? "none" : string.Join(", ", facts.Edges);
        string structures = facts.Structures.Count == 0 ? "none" : string.Join(", ", facts.Structures);
        return template.Text.Replace("{number}", facts.Number).Replace("{title}", facts.Title)
            .Replace("{approach}", facts.Approach).Replace("{time}", facts.Time).Replace("{space}", facts.Space)
            .Replace("{checks}", facts.Checks.ToString(CultureInfo.InvariantCulture))
            .Replace("{edges}", edges).Replace("{structures}", structures);
    }

    private static int Words(string text)
    {
        int words = 0;
        foreach (string part in text.Split(' ', '\n')) if (part.Trim().Length > 0) words++;
        return words;
    }

    /** Draft the editorial; refuses an unknown approach, an unsupported claim, or a budget overrun. */
    private static string Draft(Facts facts, ulong seed, int budget)
    {
        if (Array.IndexOf(Approaches, facts.Approach) < 0) throw new Exception("unknown-approach:" + facts.Approach);
        var stream = new Stream(seed);
        var chosen = new Dictionary<string, Template>(StringComparer.Ordinal);
        foreach (string section in Sections)
        {
            var candidates = new List<Template>();
            foreach (Template candidate in Script) if (candidate.Section == section) candidates.Add(candidate);
            var eligible = new List<Template>();
            foreach (Template candidate in candidates)
            {
                bool satisfied = true;
                foreach (string cite in candidate.Cites) if (!facts.Has(cite)) satisfied = false;
                if (satisfied) eligible.Add(candidate);
            }
            if (eligible.Count == 0)
                foreach (string cite in candidates[0].Cites)
                    if (!facts.Has(cite)) throw new Exception("unsupported-claim:" + cite);
            chosen[section] = eligible[(int)(stream.Next() % (ulong)eligible.Count)];
        }
        var draft = new StringBuilder();
        foreach (string section in Sections)
            draft.Append("## ").Append(section).Append('\n').Append(Fill(chosen[section], facts)).Append('\n');
        string text = draft.ToString();
        if (Words(text) > budget) throw new Exception("budget-exceeded:" + Words(text));
        return text;
    }

    private static Facts Sample() => new Facts
    {
        Number = "431", Title = "Minimum Window Sort", Approach = "two-pointers", Time = "O(n log n)", Space = "O(1)",
        Checks = 9, Edges = new List<string> { "empty input", "single element", "already sorted" },
        Structures = new List<string> { "array", "two indices" },
    };

    public static void Main()
    {
        Facts facts = Sample();
        string draft = Draft(facts, 7UL, 400);
        bool sections = true;
        foreach (string section in Sections) if (!draft.Contains("## " + section)) sections = false;
        Check(sections && draft.Split('\n', StringSplitOptions.RemoveEmptyEntries).Length == 10 && Words(draft) > 0,
              "all five sections are present and non-empty");

        Facts sparse = Sample();
        sparse.Edges = new List<string>();
        sparse.Structures = new List<string>();
        string sparseDraft = Draft(sparse, 7UL, 400);
        Check(sparseDraft.Contains("No edge cases beyond the 9 checks were recorded.") && !sparseDraft.Contains("Edge cases covered")
              && !sparseDraft.Contains("empty input"),
              "a fact that was not provided is never claimed (variant selection guard)");

        Facts missingSpace = Sample();
        missingSpace.Space = "";
        Check(Error(() => Draft(missingSpace, 7UL, 400)) == "unsupported-claim:space"
              && Error(() => Draft(new Facts { Number = "1", Title = "T", Approach = "brute-force", Time = "O(n)", Checks = 6 }, 1UL, 400))
                 == "unknown-approach:brute-force",
              "an unsupported claim and an unknown approach are refused");

        Check(draft.Contains(facts.Time) && draft.Contains(facts.Space) && draft.Contains("9 checks")
              && draft.Contains("empty input, single element, already sorted") && draft.Contains("array, two indices")
              && !draft.Contains("{") && !draft.Contains("}"),
              "complexity, check count, edges and structures are quoted verbatim with no leftover placeholders");

        Facts minimal = new Facts { Number = "1", Title = "Trivial", Approach = "hash-map", Time = "O(n)", Space = "O(1)", Checks = 6 };
        string minimalDraft = Draft(minimal, 3UL, 400);
        Check(minimalDraft.Contains("## introduction") && minimalDraft.Contains("## edge-cases")
              && minimalDraft.Contains("No edge cases beyond the 6 checks were recorded.")
              && minimalDraft.Split('\n', StringSplitOptions.RemoveEmptyEntries).Length == 10,
              "minimal facts still produce the full skeleton with an explicit no-edges sentence");

        Check(Error(() => Draft(facts, 7UL, 20)).StartsWith("budget-exceeded:", StringComparison.Ordinal)
              && Words(draft) <= 200 && Draft(facts, 7UL, 400) == draft && Draft(facts, 99UL, 400) == Draft(facts, 99UL, 400),
              "the word budget is enforced and drafting is reproducible for a seed");

        Console.WriteLine("sections: " + Sections.Length + ", draft words: " + Words(draft) + ", checks: " + facts.Checks);
        Console.WriteLine("PASS");
    }
}
