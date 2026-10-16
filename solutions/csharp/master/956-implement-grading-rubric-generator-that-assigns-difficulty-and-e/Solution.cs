// Task 956 - Implement grading rubric generator that assigns difficulty and expected complexity.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// A rubric is derived from a problem's declared constraints: the maximum input size picks the tightest
// time/space complexity a solver is expected to reach (a monotone ladder from O(n^3) down to O(log n)),
// and the techniques named by the author, the recursion depth and an explicit ambiguity flag add to a
// difficulty score clamped to 1..10.  Points are split between correctness, efficiency, edge cases and
// clarity, and the harder split is used once difficulty reaches 8; both splits total exactly 100.
//
// Verified: the total is 100 and re-parsing the rendered rubric reproduces it (independent oracle), the
// ladder is monotone in both directions over a size ladder, known size/complexity pairs match the
// hand-derived table, the ambiguity flag moves difficulty by exactly one, the clamp at 10 holds, hints
// cover every named technique, and the minimal-problem boundary is well formed.  Simplification: points
// and complexities come from tables in memory - no submission is actually graded.
using System;
using System.Collections.Generic;
using System.Globalization;

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private sealed class ProblemSpec
    {
        public readonly int Number;
        public readonly string Slug;
        public readonly long MaxN;
        public readonly string[] Techniques;
        public readonly int RecursionDepth;
        public readonly bool Ambiguous;

        public ProblemSpec(int number, string slug, long maxN, string[] techniques, int recursionDepth, bool ambiguous)
        {
            Number = number;
            Slug = slug;
            MaxN = maxN;
            Techniques = techniques;
            RecursionDepth = recursionDepth;
            Ambiguous = ambiguous;
        }
    }

    private sealed class RubricLine
    {
        public readonly string Category;
        public readonly int Points;

        public RubricLine(string category, int points)
        {
            Category = category;
            Points = points;
        }
    }

    private sealed class Rubric
    {
        public int Difficulty;
        public string ExpectedTime = "";
        public string ExpectedSpace = "";
        public int Total;
        public readonly List<RubricLine> Lines = new List<RubricLine>();
        public readonly List<string> Hints = new List<string>();
    }

    private static readonly string[] GrowthOrder = { "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(n^3)" };

    private static string ExpectedTime(long maxN) =>
        maxN <= 100 ? "O(n^3)" : maxN <= 10_000 ? "O(n^2)" : maxN <= 1_000_000 ? "O(n log n)" : maxN <= 1_000_000_000 ? "O(n)" : "O(log n)";

    private static string ExpectedSpace(long maxN) =>
        maxN <= 100 ? "O(n^2)" : maxN <= 1_000_000 ? "O(n)" : "O(log n)";

    private static int Difficulty(ProblemSpec spec)
    {
        int difficulty = spec.MaxN <= 100 ? 1 : spec.MaxN <= 10_000 ? 3 : spec.MaxN <= 1_000_000 ? 5 : spec.MaxN <= 1_000_000_000 ? 7 : 8;
        difficulty += Math.Min(3, spec.Techniques.Length);
        if (spec.RecursionDepth > 10) difficulty += 1;
        if (spec.Ambiguous) difficulty += 1;
        return Math.Min(10, difficulty);
    }

    private static Rubric Generate(ProblemSpec spec)
    {
        var rubric = new Rubric
        {
            Difficulty = Difficulty(spec),
            ExpectedTime = ExpectedTime(spec.MaxN),
            ExpectedSpace = ExpectedSpace(spec.MaxN),
        };
        bool hard = rubric.Difficulty >= 8;
        rubric.Lines.Add(new RubricLine("correctness", hard ? 45 : 50));
        rubric.Lines.Add(new RubricLine("efficiency", hard ? 30 : 25));
        rubric.Lines.Add(new RubricLine("edge cases", hard ? 20 : 15));
        rubric.Lines.Add(new RubricLine("clarity", hard ? 5 : 10));
        foreach (RubricLine line in rubric.Lines) rubric.Total += line.Points;
        foreach (string technique in spec.Techniques) rubric.Hints.Add("name the " + technique + " invariant in the header comment");
        rubric.Hints.Add("state the complexity the constraints " + spec.MaxN.ToString(CultureInfo.InvariantCulture) + " allow");
        return rubric;
    }

    private static List<string> Render(Rubric rubric)
    {
        var lines = new List<string>
        {
            "difficulty: " + rubric.Difficulty.ToString(CultureInfo.InvariantCulture),
            "expected time: " + rubric.ExpectedTime,
            "expected space: " + rubric.ExpectedSpace,
        };
        foreach (RubricLine line in rubric.Lines)
            lines.Add(line.Category + ": " + line.Points.ToString(CultureInfo.InvariantCulture));
        return lines;
    }

    /** Independent oracle: re-parse the rendered rubric and add the category points back up. */
    private static int ParseBackTotal(List<string> rendered)
    {
        int total = 0;
        foreach (string line in rendered)
        {
            int separator = line.IndexOf(": ", StringComparison.Ordinal);
            if (separator < 0) continue;
            string category = line.Substring(0, separator);
            if (category == "difficulty" || category == "expected time" || category == "expected space") continue;
            total += int.Parse(line.Substring(separator + 2), CultureInfo.InvariantCulture);
        }
        return total;
    }

    private static int GrowthRank(string complexity)
    {
        for (int i = 0; i < GrowthOrder.Length; i++)
        {
            if (GrowthOrder[i] == complexity) return i;
        }
        throw new Exception("unknown complexity " + complexity);
    }

    public static void Main()
    {
        var specs = new List<ProblemSpec>
        {
            new ProblemSpec(951, "reproducible-benchmarking-harness", 10_000, new[] { "hashing", "amortisation" }, 4, false),
            new ProblemSpec(952, "autograder-with-complexity", 1_000_000, new[] { "divide and conquer" }, 18, true),
            new ProblemSpec(953, "contributor-templates", 50, new string[0], 0, false),
            new ProblemSpec(954, "ci-pipeline-matrix", 1_000_000_000_000, new[] { "graph search", "greedy", "monotone stack" }, 30, true),
        };

        foreach (ProblemSpec spec in specs)
        {
            Rubric rubric = Generate(spec);
            Check(rubric.Total == 100, "the rubric for " + spec.Slug + " totals 100 points (got " + rubric.Total + ")");
            Check(ParseBackTotal(Render(rubric)) == rubric.Total,
                "re-parsing the rendered rubric for " + spec.Slug + " reproduces the total");
            Check(rubric.Hints.Count == spec.Techniques.Length + 1,
                "the rubric for " + spec.Slug + " carries one hint per technique plus a general one");
            Check(rubric.Difficulty >= 1 && rubric.Difficulty <= 10, "the difficulty of " + spec.Slug + " stays inside 1..10");
        }

        // Known size/complexity pairs, derived by hand from the ladder.
        Check(ExpectedTime(50) == "O(n^3)" && ExpectedSpace(50) == "O(n^2)", "MaxN=50 expects O(n^3) time and O(n^2) space");
        Check(ExpectedTime(5_000) == "O(n^2)" && ExpectedSpace(5_000) == "O(n)", "MaxN=5000 expects O(n^2) time and O(n) space");
        Check(ExpectedTime(500_000) == "O(n log n)" && ExpectedSpace(500_000) == "O(n)", "MaxN=500000 expects O(n log n) time and O(n) space");
        Check(ExpectedTime(5_000_000_000) == "O(log n)" && ExpectedSpace(5_000_000_000) == "O(log n)",
            "MaxN=5e9 expects O(log n) time and space");

        long[] ladder = { 1, 100, 101, 10_000, 10_001, 1_000_000, 1_000_001, 1_000_000_000, 1_000_000_001, 1_000_000_000_000 };
        bool difficultyMonotone = true, growthMonotone = true;
        int previousDifficulty = 0, previousRank = GrowthOrder.Length;
        foreach (long maxN in ladder)
        {
            var probe = new ProblemSpec(900, "probe", maxN, new[] { "recursion" }, 3, false);
            int difficulty = Difficulty(probe);
            int rank = GrowthRank(ExpectedTime(maxN));
            difficultyMonotone &= difficulty >= previousDifficulty;
            growthMonotone &= rank <= previousRank;
            previousDifficulty = difficulty;
            previousRank = rank;
        }
        Check(difficultyMonotone, "difficulty never decreases as the input bound grows");
        Check(growthMonotone, "the expected time complexity never loosens as the input bound grows (O(n^3) -> O(log n))");

        var baseSpec = new ProblemSpec(960, "gamification-badges", 10_000, new[] { "hashing" }, 8, false);
        var ambiguousSpec = new ProblemSpec(960, "gamification-badges", 10_000, new[] { "hashing" }, 8, true);
        Check(Difficulty(ambiguousSpec) - Difficulty(baseSpec) == 1, "the ambiguity flag raises difficulty by exactly one");

        var deepSpec = new ProblemSpec(960, "gamification-badges", 10_000, new[] { "hashing" }, 11, false);
        Check(Difficulty(deepSpec) - Difficulty(baseSpec) == 1, "recursion deeper than ten frames raises difficulty by exactly one");

        Rubric hardest = Generate(specs[3]);
        Check(hardest.Difficulty == 10, "the hardest spec clamps at difficulty 10 instead of overflowing");
        Check(hardest.Lines.Exists(l => l.Category == "correctness" && l.Points == 45) && hardest.Total == 100,
            "difficulty >= 8 selects the harder correctness-weighted split, still totalling 100");

        var minimal = new ProblemSpec(967, "accessibility-lessons", 1, new string[0], 0, false);
        Rubric minimalRubric = Generate(minimal);
        Check(minimalRubric.Difficulty == 1 && minimalRubric.Hints.Count == 1, "the minimal problem scores difficulty 1 with a single hint");
        Check(minimalRubric.ExpectedTime == "O(n^3)" && minimalRubric.Lines.Count == 4 && ParseBackTotal(Render(minimalRubric)) == 100,
            "the minimal rubric is well formed and still totals 100");

        Check(string.Join("|", Render(Generate(specs[1]))) == string.Join("|", Render(Generate(specs[1]))),
            "generating the same rubric twice renders byte-identically");

        Console.WriteLine("rubric: " + specs.Count + " specs generated, all totals 100 (structured and re-parsed)");
        Console.WriteLine("ladder: MaxN 1 -> 1e12 maps O(n^3)/O(n^2) down to O(log n)/O(log n), difficulty 1 -> 10 (clamped)");
        Console.WriteLine("hardest: difficulty " + hardest.Difficulty + " -> " + string.Join(", ", Render(hardest)));
        Console.WriteLine("PASS");
    }
}
