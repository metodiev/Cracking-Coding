// Task 975 - Implement natural-language problem paraphrasing generator for diverse practice.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// A problem record (id, signature, constraint text, subject/action/object wording) is rewritten by
// one of four sentence templates whose slots are filled from synonym classes. Template choice and
// synonym choice are both derived from a splitmix64 hash of the problem id, so a problem always
// paraphrases the same way while different problems spread across templates and wordings.
//
// Verified: the payload invariance that every digit run of the source survives into the paraphrase
// (oracle: an independent character-scanning extraction compared as a sorted multiset, not with
// regex), the verbatim signature and constraint clause, synonym-class membership, template diversity
// over 200 ids, per-id determinism, absence of unresolved placeholders, and the empty template-class
// boundaries.
// Simplification: wording is template and synonym substitution over a fixed four-template catalogue;
// there is no parser, no grammar model and no semantics, only the payload-preservation invariant.
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    private static readonly string[] Templates =
    {
        "{verb} the {noun} in the {subject} for {signature}. Constraints: {limits}.",
        "Given {signature} over a {subject}, {verb} the {noun}. Constraints: {limits}.",
        "Your task is to {verb} the {noun} of the {subject} described by {signature}, where {limits}.",
        "{verb} the {noun} within the {subject} ({signature}) subject to {limits}.",
    };

    private static readonly string[] ActionSynonyms = { "find", "determine", "locate" };
    private static readonly string[] SubjectSynonyms = { "array", "list", "sequence" };
    private static readonly string[] ObjectSynonyms = { "index", "position", "subscript" };

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private sealed class Problem
    {
        public readonly string Id;
        public readonly string Signature;
        public readonly string Limits;
        public readonly string Action;
        public readonly string Subject;
        public readonly string Object;

        public Problem(string id, string signature, string limits, string action, string subject, string obj)
        {
            Id = id;
            Signature = signature;
            Limits = limits;
            Action = action;
            Subject = subject;
            Object = obj;
        }

        public string Source =>
            Action + " the " + Object + " in the " + Subject + " using " + Signature + " where " + Limits;
    }

    private static ulong Mix(ulong value)
    {
        unchecked
        {
            ulong z = value + 0x9E3779B97F4A7C15UL;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }

    private static ulong Hash(string text, ulong seed)
    {
        ulong hash = seed ^ 0xCBF29CE484222325UL;
        foreach (char character in text) hash = Mix(hash ^ character);
        return hash;
    }

    private static string Pick(string[] options, string fallback, ulong hash, int slot)
    {
        if (options == null || options.Length == 0) return fallback;
        ulong mixed = Mix(hash + (ulong)slot * 0x9E3779B97F4A7C15UL);
        return options[(int)(mixed % (ulong)options.Length)];
    }

    private static string Generate(
        Problem problem, string[] templates, string[] verbs, string[] subjects, string[] nouns, ulong seed)
    {
        ulong hash = Hash(problem.Id, seed);
        if (templates == null || templates.Length == 0) return problem.Source;
        string template = templates[(int)(Mix(hash ^ 0x51ED2701UL) % (ulong)templates.Length)];
        return template
            .Replace("{verb}", Pick(verbs, problem.Action, hash, 0))
            .Replace("{noun}", Pick(nouns, problem.Object, hash, 1))
            .Replace("{subject}", Pick(subjects, problem.Subject, hash, 2))
            .Replace("{signature}", problem.Signature)
            .Replace("{limits}", problem.Limits);
    }

    private static string Paraphrase(Problem problem, ulong seed)
        => Generate(problem, Templates, ActionSynonyms, SubjectSynonyms, ObjectSynonyms, seed);

    /** Oracle: digit runs pulled out by a character scan, so it shares no code with the generator. */
    private static List<string> DigitRuns(string text)
    {
        var runs = new List<string>();
        var current = new StringBuilder();
        foreach (char character in text)
        {
            if (character >= '0' && character <= '9') current.Append(character);
            else if (current.Length > 0)
            {
                runs.Add(current.ToString());
                current.Clear();
            }
        }
        if (current.Length > 0) runs.Add(current.ToString());
        runs.Sort(StringComparer.Ordinal);
        return runs;
    }

    /** Oracle: template class inferred from the rendered text, not from the generator's own choice. */
    private static string TemplateClass(string text)
    {
        if (text.Contains("Given ")) return "given";
        if (text.Contains("Your task")) return "task";
        if (text.Contains("within")) return "within";
        return "plain";
    }

    private static List<Problem> Catalogue()
    {
        var problems = new List<Problem>();
        string[] ids = { "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8" };
        string[] signatures = { "solve(nums, k)", "solve(text, pattern)", "solve(grid, start)", "solve(edges, n)" };
        string[] limits = { "1 <= n <= 100000", "0 <= nums[i] <= 10000", "1 <= len <= 1000", "-10000 <= value <= 10000" };
        for (int i = 0; i < ids.Length; i++)
            problems.Add(new Problem(ids[i], signatures[i % signatures.Length], limits[i % limits.Length],
                "find", "array", "index"));
        return problems;
    }

    public static void Main()
    {
        List<Problem> problems = Catalogue();
        Problem sample = problems[0];
        string paraphrase = Paraphrase(sample, 975UL);

        Check(paraphrase.Contains(sample.Signature), "the signature survives verbatim into the paraphrase");
        Check(paraphrase.Contains(sample.Limits), "the constraint clause survives verbatim into the paraphrase");
        Check(!paraphrase.Contains("{") && !paraphrase.Contains("}"),
            "no template placeholder is left unresolved (got '" + paraphrase + "')");
        Check(paraphrase != sample.Source && paraphrase.Length > 40, "the paraphrase differs from the source wording");

        // Oracle: digit-run multisets of source and paraphrase must be identical.
        Check(string.Join("|", DigitRuns(sample.Source)) == string.Join("|", DigitRuns(paraphrase)),
            "every digit run of the source is reproduced in the paraphrase (oracle by character scan)");
        Check(DigitRuns(paraphrase).Count == 2, "the sample carries exactly the two numeric bounds of its constraint clause");

        // Synonym classes: the chosen verb must come from the declared class.
        int foreign = 0;
        for (int i = 0; i < 60; i++)
        {
            string text = Paraphrase(problems[i % problems.Count], (ulong)(i + 1));
            bool found = false;
            foreach (string candidate in ActionSynonyms) found |= text.StartsWith(candidate + " the")
                || text.Contains(", " + candidate + " the") || text.Contains("to " + candidate + " the");
            if (!found) foreign++;
        }
        Check(foreign == 0, "every generated sentence still uses a verb from the declared synonym class");

        // Diversity: eight problems must not collapse onto one template, and all four are reachable.
        var classes = new HashSet<string>(StringComparer.Ordinal);
        var texts = new HashSet<string>(StringComparer.Ordinal);
        foreach (Problem problem in problems)
        {
            string text = Paraphrase(problem, 975UL);
            classes.Add(TemplateClass(text));
            texts.Add(text);
        }
        Check(classes.Count >= 3, "eight problems spread over at least three templates (got " + classes.Count + ")");
        Check(texts.Count == problems.Count, "each problem id gets its own paraphrase");
        var reachable = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < 200; i++) reachable.Add(TemplateClass(Paraphrase(problems[0], (ulong)(i + 1))));
        Check(reachable.Count == 4, "over 200 seeds every template class is reachable (got " + reachable.Count + ")");

        // Determinism and seed sensitivity.
        Check(Paraphrase(sample, 975UL) == paraphrase, "the same problem id always paraphrases identically");
        Check(Paraphrase(sample, 976UL) != paraphrase || Paraphrase(sample, 977UL) != paraphrase,
            "changing the seed changes at least one rendering");

        // Boundaries: empty template catalogue and empty synonym class fall back instead of failing.
        Check(Generate(sample, new string[0], ActionSynonyms, SubjectSynonyms, ObjectSynonyms, 1UL) == sample.Source,
            "an empty template catalogue returns the original wording unchanged");
        string verbless = Generate(sample, new[] { "{verb} the {noun} of a {subject} near {signature}." },
            new string[0], SubjectSynonyms, ObjectSynonyms, 1UL);
        Check(verbless.StartsWith(sample.Action + " the") && !verbless.Contains("{"),
            "an empty synonym class falls back to the problem's own verb");

        Console.WriteLine("sample paraphrase: " + paraphrase);
        Console.WriteLine("templates seen over 8 problems: " + classes.Count + ", reachable classes over 200 seeds: "
            + reachable.Count);
        Console.WriteLine("digit runs preserved: " + string.Join(",", DigitRuns(paraphrase)));
        Console.WriteLine("PASS");
    }
}
