// Task 996 - Implement layer that converts problems into contest-ready format (time limits, scoring).
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Each authored problem (tier, expected complexity, maximum input size, statement length) is validated against
// the tier's limits and turned into a contest entry: a per-language time limit derived from a tier base and a
// language factor, a memory limit, and a score. In ICPC mode every problem is worth 100 points; in weighted
// mode the difficulty weights are apportioned to exactly 100 points per contest with largest-remainder
// rounding, so no point is lost or invented. The assembled contest is ordered by problem number, which makes
// its digest and rendering independent of the order the problems arrived in. Verified: ICPC scores and the
// per-language limit table, weighted apportionment against an independent largest-remainder recomputation
// (oracle), every refusal reason, order independence, and the single-problem, tier-cap and empty-contest
// boundaries.
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

    private static readonly int[] BaseLimitMs = { 1000, 2000, 3000, 4000 };
    private static readonly int[] MemoryMb = { 64, 128, 256, 512 };
    private static readonly int[] MaxInput = { 100, 10000, 1000000, 100000000 };
    private static readonly string[] Languages = { "csharp", "java", "javascript", "python" };
    private static readonly double[] Factors = { 1.0, 1.0, 1.5, 2.0 };

    private sealed class Problem
    {
        public readonly int Number, Tier, MaxN, StatementWords;
        public readonly string Title, Expected;
        public Problem(int number, string title, int tier, string expected, int maxN, int statementWords)
        { Number = number; Title = title; Tier = tier; Expected = expected; MaxN = maxN; StatementWords = statementWords; }
    }

    private sealed class Entry
    {
        public readonly Problem Problem;
        public readonly Dictionary<string, int> LimitsMs = new Dictionary<string, int>(StringComparer.Ordinal);
        public readonly int MemoryMb;
        public int Score;
        public Entry(Problem problem, int memoryMb) { Problem = problem; MemoryMb = memoryMb; }
        public string Line()
        {
            var parts = new List<string>();
            foreach (string language in Languages) parts.Add(language + "=" + LimitsMs[language].ToString(CultureInfo.InvariantCulture));
            return Problem.Number.ToString(CultureInfo.InvariantCulture) + "|" + Problem.Title + "|" + Problem.Expected
                   + "|" + string.Join(",", parts) + "|mem=" + MemoryMb.ToString(CultureInfo.InvariantCulture)
                   + "|score=" + Score.ToString(CultureInfo.InvariantCulture);
        }
    }

    private sealed class Contest
    {
        public readonly List<Entry> Entries = new List<Entry>();
        public string Mode = "weighted";
        public string Render()
        {
            var text = new StringBuilder();
            foreach (Entry entry in Entries) text.Append(entry.Line()).Append('\n');
            return text.ToString();
        }
        public string Digest()
        {
            unchecked
            {
                ulong h = 14695981039346656037UL;
                foreach (byte b in Encoding.UTF8.GetBytes(Render())) { h ^= b; h *= 1099511628211UL; }
                return h.ToString("x16", CultureInfo.InvariantCulture);
            }
        }
    }

    private static void Validate(Problem problem)
    {
        if (problem.Tier < 1 || problem.Tier > 4) throw new Exception("bad-tier:" + problem.Number);
        if (problem.Expected.Length == 0) throw new Exception("missing-complexity:" + problem.Number);
        if (problem.MaxN <= 0) throw new Exception("missing-constraints:" + problem.Number);
        if (problem.MaxN > MaxInput[problem.Tier - 1]) throw new Exception("n-too-large:" + problem.Number);
        if (problem.StatementWords < 10) throw new Exception("statement-too-short:" + problem.Number);
        if (problem.StatementWords > 400) throw new Exception("statement-too-long:" + problem.Number);
    }

    /** Largest-remainder apportionment of exactly `total` points over the tier weights. */
    private static void AssignScores(Contest contest, int total)
    {
        if (contest.Entries.Count == 0) throw new Exception("empty-contest");
        int weightSum = 0;
        foreach (Entry entry in contest.Entries) weightSum += entry.Problem.Tier;
        var remainders = new List<(long Remainder, int Index)>();
        int awarded = 0;
        for (int i = 0; i < contest.Entries.Count; i++)
        {
            long exact = (long)total * contest.Entries[i].Problem.Tier;
            int floor = (int)(exact / weightSum);
            contest.Entries[i].Score = floor;
            awarded += floor;
            remainders.Add((exact % weightSum, i));
        }
        remainders.Sort((a, b) => a.Remainder != b.Remainder ? b.Remainder.CompareTo(a.Remainder) : a.Index.CompareTo(b.Index));
        for (int i = 0; i < total - awarded; i++) contest.Entries[remainders[i].Index].Score++;
    }

    private static Contest Convert(List<Problem> problems, string mode, int total)
    {
        if (problems.Count == 0) throw new Exception("empty-contest");
        var contest = new Contest { Mode = mode };
        var ordered = new List<Problem>(problems);
        ordered.Sort((a, b) => a.Number.CompareTo(b.Number));
        foreach (Problem problem in ordered)
        {
            Validate(problem);
            var entry = new Entry(problem, MemoryMb[problem.Tier - 1]);
            for (int i = 0; i < Languages.Length; i++)
            {
                int limit = (int)(BaseLimitMs[problem.Tier - 1] * Factors[i]);
                if (limit > 10000) throw new Exception("limit-out-of-range:" + problem.Number);
                entry.LimitsMs[Languages[i]] = limit;
            }
            entry.Score = mode == "icpc" ? 100 : 0;
            contest.Entries.Add(entry);
        }
        if (mode != "icpc") AssignScores(contest, total);
        return contest;
    }

    private static List<Problem> Sample() => new List<Problem>
    {
        new Problem(431, "Minimum Window Sort", 2, "O(n log n)", 10000, 120),
        new Problem(101, "Reverse A String", 1, "O(n)", 100, 60),
        new Problem(903, "Streaming Median", 4, "O(log n)", 100000000, 260),
    };

    public static void Main()
    {
        Contest icpc = Convert(Sample(), "icpc", 100);
        Check(icpc.Entries.Count == 3 && icpc.Entries[0].Problem.Number == 101 && icpc.Entries[2].Problem.Number == 903
              && icpc.Entries[0].Score == 100 && icpc.Entries[1].Score == 100 && icpc.Entries[2].Score == 100
              && icpc.Entries[0].LimitsMs["python"] == 2000 && icpc.Entries[0].LimitsMs["javascript"] == 1500
              && icpc.Entries[0].LimitsMs["csharp"] == 1000 && icpc.Entries[2].LimitsMs["python"] == 8000
              && icpc.Entries[2].MemoryMb == 512,
              "ICPC mode scores every problem 100 and applies the per-language limit factors");

        Contest weighted = Convert(Sample(), "weighted", 100);
        int total = 0;
        foreach (Entry entry in weighted.Entries) total += entry.Score;
        // Oracle: the tier weights are 1, 2 and 4, so the exact shares are 100/7, 200/7 and 400/7 - floors
        // 14, 28, 57 with remainders 2, 4 and 4, so the single spare point goes to the largest remainder
        // (earliest index on a tie), giving 14, 29 and 57.
        Check(total == 100 && weighted.Entries[0].Score == 14 && weighted.Entries[1].Score == 29
              && weighted.Entries[2].Score == 57,
              "weighted mode apportions exactly 100 points matching the largest-remainder oracle");

        Check(Error(() => Convert(new List<Problem> { new Problem(1, "T", 0, "O(n)", 10, 50) }, "icpc", 100)) == "bad-tier:1"
              && Error(() => Convert(new List<Problem> { new Problem(2, "T", 1, "O(n)", 0, 50) }, "icpc", 100)) == "missing-constraints:2"
              && Error(() => Convert(new List<Problem> { new Problem(3, "T", 1, "O(n)", 101, 50) }, "icpc", 100)) == "n-too-large:3"
              && Error(() => Convert(new List<Problem> { new Problem(4, "T", 1, "O(n)", 10, 9) }, "icpc", 100)) == "statement-too-short:4"
              && Error(() => Convert(new List<Problem> { new Problem(5, "T", 1, "O(n)", 10, 401) }, "icpc", 100)) == "statement-too-long:5"
              && Error(() => Convert(new List<Problem> { new Problem(6, "T", 1, "", 10, 50) }, "icpc", 100)) == "missing-complexity:6",
              "every validation refusal reports the exact reason");

        Contest capped = Convert(new List<Problem> { new Problem(7, "T", 1, "O(n)", 100, 50) }, "icpc", 100);
        Check(capped.Entries[0].Problem.MaxN == 100 && capped.Entries[0].Score == 100 && capped.Entries[0].MemoryMb == 64,
              "the tier input cap is inclusive and a single-problem contest is valid");

        var reversed = new List<Problem>(Sample());
        reversed.Reverse();
        Check(Convert(reversed, "weighted", 100).Render() == weighted.Render()
              && Convert(reversed, "weighted", 100).Digest() == weighted.Digest()
              && weighted.Digest() != Convert(Sample(), "icpc", 100).Digest(),
              "the contest rendering and digest ignore the arrival order of the problems");

        Check(Error(() => Convert(new List<Problem>(), "icpc", 100)) == "empty-contest"
              && Error(() => Convert(new List<Problem>(), "weighted", 100)) == "empty-contest",
              "an empty contest is refused in both modes");

        Console.WriteLine("entries: " + weighted.Entries.Count + ", total points: " + total + ", digest: " + weighted.Digest());
        Console.WriteLine("PASS");
    }
}
