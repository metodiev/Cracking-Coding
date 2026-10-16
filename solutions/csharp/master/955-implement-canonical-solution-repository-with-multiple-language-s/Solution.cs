// Task 955 - Implement canonical solution repository with multiple language solutions and performance baselines.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// The repository is modelled as problems, each carrying one entry per canonical language (python, java,
// javascript, csharp) with the file path, a stored baseline cost and a fresh candidate cost.  A lint
// pass checks the four invariants a canonical repo must satisfy - every language present exactly once,
// the folder named `<3-digit number>-<slug>`, a baseline per language, and no performance regression
// beyond tolerance - and additionally flags a language that diverges from the fastest one by more than
// a factor, which is how a bad port shows up.
//
// Verified: a clean repository lints clean, missing/duplicated/folder-mismatched entries and baseline
// regressions are reported, the divergence outlier fires only above the factor, the coverage counter is
// recomputed independently, insertion order does not change the corpus fingerprint, and the empty /
// zero-solution boundaries hold.  Simplification: costs are numbers in memory; nothing is timed.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private static readonly string[] Canonical = { "python", "java", "javascript", "csharp" };

    private sealed class Entry
    {
        public readonly string Language;
        public readonly string Path;
        public readonly long BaselineUs;
        public readonly long CandidateUs;

        public Entry(string language, string path, long baselineUs, long candidateUs)
        {
            Language = language;
            Path = path;
            BaselineUs = baselineUs;
            CandidateUs = candidateUs;
        }
    }

    private sealed class Problem
    {
        public readonly int Number;
        public readonly string Slug;
        public readonly List<Entry> Entries = new List<Entry>();

        public Problem(int number, string slug)
        {
            Number = number;
            Slug = slug;
        }
    }

    private static string Folder(Problem problem) => problem.Number.ToString("D3", CultureInfo.InvariantCulture) + "-" + problem.Slug;

    private static Entry MakeEntry(string language, Problem problem, long baselineUs, long candidateUs)
    {
        string file = language == "python" ? "solution.py" : language == "java" ? "Solution.java" :
            language == "javascript" ? "solution.js" : "Solution.cs";
        return new Entry(language, "solutions/" + language + "/master/" + Folder(problem) + "/" + file, baselineUs, candidateUs);
    }

    private static Problem BuildProblem(int number, string slug, long baselineUs, params long[] candidateUs)
    {
        var problem = new Problem(number, slug);
        for (int i = 0; i < Canonical.Length; i++)
        {
            long candidate = i < candidateUs.Length ? candidateUs[i] : baselineUs;
            problem.Entries.Add(MakeEntry(Canonical[i], problem, baselineUs, candidate));
        }
        return problem;
    }

    private static List<string> Validate(List<Problem> problems, double tolerance, double divergenceFactor)
    {
        var issues = new List<string>();
        foreach (Problem problem in problems)
        {
            string folder = Folder(problem);
            var seen = new SortedSet<string>(StringComparer.Ordinal);
            foreach (Entry entry in problem.Entries)
            {
                if (!seen.Add(entry.Language)) issues.Add("duplicate " + folder + " " + entry.Language);
            }
            foreach (string language in Canonical)
            {
                if (!problem.Entries.Exists(e => e.Language == language)) issues.Add("missing " + folder + " " + language);
            }
            long fastest = long.MaxValue, slowest = 0;
            foreach (Entry entry in problem.Entries)
            {
                if (!entry.Path.Contains("/" + folder + "/", StringComparison.Ordinal))
                    issues.Add("folder mismatch " + entry.Path + " (expected " + folder + ")");
                if (entry.BaselineUs <= 0)
                    issues.Add("no baseline " + entry.Path);
                else if (entry.CandidateUs > entry.BaselineUs * (1.0 + tolerance))
                    issues.Add("regression " + entry.Path + " +" +
                        ((entry.CandidateUs / (double)entry.BaselineUs - 1.0) * 100.0).ToString("F1", CultureInfo.InvariantCulture) + "%");
                fastest = Math.Min(fastest, entry.CandidateUs);
                slowest = Math.Max(slowest, entry.CandidateUs);
            }
            if (problem.Entries.Count > 0 && fastest > 0 && slowest > fastest * divergenceFactor)
                issues.Add("outlier " + folder + " slowest/fastest = " +
                    (slowest / (double)fastest).ToString("F2", CultureInfo.InvariantCulture));
        }
        issues.Sort(StringComparer.Ordinal);
        return issues;
    }

    /** Independent oracle for the coverage report: count (problem, canonical language) pairs present. */
    private static int Coverage(List<Problem> problems)
    {
        int covered = 0;
        foreach (Problem problem in problems)
        {
            int here = 0;
            foreach (string language in Canonical)
            {
                if (problem.Entries.Exists(e => e.Language == language)) here++;
            }
            covered += here;
        }
        return covered;
    }

    /** Order-insensitive corpus fingerprint: sorted entry descriptors hashed with FNV-1a. */
    private static ulong CorpusFingerprint(List<Problem> problems)
    {
        var descriptors = new List<string>();
        foreach (Problem problem in problems)
        {
            foreach (Entry entry in problem.Entries)
                descriptors.Add(entry.Path + ":" + entry.BaselineUs.ToString(CultureInfo.InvariantCulture) + ":" +
                    entry.CandidateUs.ToString(CultureInfo.InvariantCulture));
        }
        descriptors.Sort(StringComparer.Ordinal);
        var sb = new StringBuilder();
        foreach (string descriptor in descriptors) sb.Append(descriptor).Append('\n');
        unchecked
        {
            ulong hash = 14695981039346656037UL;
            foreach (char ch in sb.ToString())
            {
                hash ^= ch;
                hash *= 1099511628211UL;
            }
            return hash;
        }
    }

    public static void Main()
    {
        var clean = new List<Problem>
        {
            BuildProblem(951, "reproducible-benchmarking-harness", 12_000, 11_500, 12_400, 12_900, 12_100),
            BuildProblem(952, "autograder-with-complexity", 8_000, 8_100, 8_000, 7_900, 8_200),
        };
        Check(Validate(clean, 0.10, 3.0).Count == 0, "a canonical repository lints clean");
        Check(Coverage(clean) == clean.Count * Canonical.Length, "coverage equals problems x canonical languages");
        int recount = 0;
        foreach (Problem problem in clean)
        {
            foreach (string language in Canonical)
            {
                if (problem.Entries.Exists(e => e.Language == language)) recount++;
            }
        }
        Check(recount == Coverage(clean) && recount == 8, "an independent recount agrees with the coverage oracle (" + recount + " entries)");

        var missing = new List<Problem> { BuildProblem(953, "contributor-templates", 1_000, 1_000, 1_000, 1_000, 1_000) };
        missing[0].Entries.RemoveAll(e => e.Language == "javascript");
        List<string> missingIssues = Validate(missing, 0.10, 3.0);
        Check(missingIssues.Count == 1 && missingIssues[0].StartsWith("missing ", StringComparison.Ordinal) &&
            missingIssues[0].EndsWith("javascript", StringComparison.Ordinal), "a missing language is reported once, by name");

        var mismatched = new List<Problem> { BuildProblem(954, "ci-pipeline-matrix", 2_000, 2_000, 2_000, 2_000, 2_000) };
        mismatched[0].Entries.RemoveAt(0);
        mismatched[0].Entries.Insert(0, new Entry("python", "solutions/python/master/955-wrong-folder/solution.py", 2_000, 2_000));
        Check(Validate(mismatched, 0.10, 3.0).Exists(i => i.StartsWith("folder mismatch", StringComparison.Ordinal)),
            "an entry stored under the wrong folder is flagged");

        var duplicated = new List<Problem> { BuildProblem(956, "grading-rubric", 3_000, 3_000, 3_000, 3_000, 3_000) };
        duplicated[0].Entries.Add(MakeEntry("csharp", duplicated[0], 3_000, 3_000));
        Check(Validate(duplicated, 0.10, 3.0).Exists(i => i.StartsWith("duplicate ", StringComparison.Ordinal)),
            "a second solution in the same language is flagged as a duplicate");

        var regressed = new List<Problem> { BuildProblem(957, "randomized-oracles", 10_000, 12_000, 10_000, 10_000, 10_000) };
        Check(Validate(regressed, 0.10, 3.0).Exists(i => i.StartsWith("regression ", StringComparison.Ordinal)),
            "a 20% slowdown in one language is reported against a 10% tolerance");
        var tolerable = new List<Problem> { BuildProblem(957, "randomized-oracles", 10_000, 10_500, 10_000, 10_000, 10_000) };
        Check(Validate(tolerable, 0.10, 3.0).Count == 0, "a 5% drift stays inside the 10% tolerance");

        var divergent = new List<Problem> { BuildProblem(958, "plagiarism-detector", 5_000, 5_000, 5_000, 5_000, 20_000) };
        Check(Validate(divergent, 0.10, 3.0).Exists(i => i.StartsWith("outlier ", StringComparison.Ordinal)),
            "a 4x-slower port is flagged as a divergence outlier");
        var mildDivergence = new List<Problem> { BuildProblem(958, "plagiarism-detector", 10_000, 10_000, 5_000, 5_000, 10_000) };
        Check(Validate(mildDivergence, 0.10, 3.0).Count == 0, "a 2x spread stays below the divergence factor and inside tolerance");

        var brokenBaseline = new List<Problem> { BuildProblem(959, "leaderboard-verified", 4_000, 4_000, 4_000, 4_000, 4_000) };
        brokenBaseline[0].Entries[2] = MakeEntry("javascript", brokenBaseline[0], 0, 4_000);
        Check(Validate(brokenBaseline, 0.10, 3.0).Exists(i => i.StartsWith("no baseline", StringComparison.Ordinal)),
            "an entry without a stored baseline is flagged");

        var emptyProblem = new List<Problem> { new Problem(960, "gamification-badges") };
        List<string> emptyIssues = Validate(emptyProblem, 0.10, 3.0);
        Check(emptyIssues.Count == Canonical.Length, "a problem with zero solutions reports four missing languages (" + emptyIssues.Count + ")");
        List<string> nothing = Validate(new List<Problem>(), 0.10, 3.0);
        Check(nothing.Count == 0 && Coverage(new List<Problem>()) == 0, "an empty repository is clean and has zero coverage");

        var cleanAgain = new List<Problem>
        {
            BuildProblem(952, "autograder-with-complexity", 8_000, 8_100, 8_000, 7_900, 8_200),
            BuildProblem(951, "reproducible-benchmarking-harness", 12_000, 11_500, 12_400, 12_900, 12_100),
        };
        Check(CorpusFingerprint(clean) == CorpusFingerprint(cleanAgain),
            "the corpus fingerprint ignores insertion order (identical sets hash identically)");
        Check(Validate(clean, 0.10, 3.0).Count == 0 && Validate(cleanAgain, 0.10, 3.0).Count == 0,
            "re-validating the same corpus is deterministic");

        Console.WriteLine("corpus: " + clean.Count + " problems, coverage " + Coverage(clean) + "/" +
            (clean.Count * Canonical.Length) + " entries, fingerprint " + CorpusFingerprint(clean).ToString("x16", CultureInfo.InvariantCulture));
        Console.WriteLine("lint: missing/duplicate/folder-mismatch/baseline/regression/outlier classes all exercised");
        Console.WriteLine("tolerances: regression at 10%, divergence factor at 3.0x");
        Console.WriteLine("PASS");
    }
}
