// Task 287 - Combination sum with repetition via backtracking. (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    private static int[] numbers = Array.Empty<int>();
    private static List<List<int>> results = new List<List<int>>();
    private static List<int> current = new List<int>();

    /// <summary>
    /// All unique combinations of positive candidates that sum to target, each candidate
    /// reusable. Candidates are walked in increasing order and the DFS continues from the
    /// same index after taking a value, so every multiset appears exactly once.
    /// </summary>
    public static List<List<int>> CombinationSum(int[] candidates, int target)
    {
        numbers = (int[])candidates.Clone();
        Array.Sort(numbers);
        results = new List<List<int>>();
        current = new List<int>();
        Backtrack(0, target);
        results.Sort(CompareCombinations);
        return results;
    }

    private static void Backtrack(int start, int remaining)
    {
        if (remaining == 0)
        {
            results.Add(new List<int>(current));
            return;
        }
        for (int i = start; i < numbers.Length; i++)
        {
            if (numbers[i] > remaining) break;
            current.Add(numbers[i]);
            Backtrack(i, remaining - numbers[i]);   // same index: repetition allowed
            current.RemoveAt(current.Count - 1);
        }
    }

    private static int CompareCombinations(List<int> a, List<int> b)
    {
        int size = Math.Min(a.Count, b.Count);
        for (int i = 0; i < size; i++)
        {
            if (a[i] != b[i]) return a[i].CompareTo(b[i]);
        }
        return a.Count.CompareTo(b.Count);
    }

    /// <summary>Oracle: choose how many copies of each candidate to take (counts per candidate).</summary>
    public static List<List<int>> BruteForceCombinations(int[] candidates, int target)
    {
        var sorted = (int[])candidates.Clone();
        Array.Sort(sorted);
        var found = new List<List<int>>();
        Build(sorted, 0, target, new List<int>(), found);
        found.Sort(CompareCombinations);
        return found;
    }

    private static void Build(int[] sorted, int index, int remaining, List<int> chosen, List<List<int>> found)
    {
        if (index == sorted.Length)
        {
            if (remaining == 0) found.Add(new List<int>(chosen));
            return;
        }
        int value = sorted[index];
        for (int copies = 0; value * copies <= remaining; copies++)
        {
            var next = new List<int>(chosen);
            for (int i = 0; i < copies; i++) next.Add(value);
            Build(sorted, index + 1, remaining - value * copies, next, found);
        }
    }

    private static int seed = 13579;

    private static int NextSeed()
    {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    private static string Key(List<List<int>> combinations)
    {
        var parts = combinations.Select(c => string.Join(",", c)).ToList();
        parts.Sort(StringComparer.Ordinal);
        return string.Join("|", parts);
    }

    public static void Main()
    {
        Check(Key(CombinationSum(new[] { 2, 3, 6, 7 }, 7)) == "2,2,3|7", "classic seven");
        Check(Key(CombinationSum(new[] { 2, 3, 5 }, 8)) == "2,2,2,2|2,3,3|3,5", "classic eight");
        Check(Key(CombinationSum(new[] { 7, 2, 3, 6 }, 7)) == "2,2,3|7", "unsorted input");

        var zero = CombinationSum(new[] { 2, 3 }, 0);
        Check(zero.Count == 1 && zero[0].Count == 0, "target zero");
        Check(CombinationSum(new[] { 2, 3 }, -1).Count == 0, "negative target");
        Check(CombinationSum(new[] { 2, 4 }, 7).Count == 0, "odd target even values");
        Check(CombinationSum(new[] { 2 }, 1).Count == 0, "below every candidate");
        Check(Key(CombinationSum(new[] { 5 }, 10)) == "5,5", "repetition");
        Check(CombinationSum(new[] { 8 }, 4).Count == 0, "single too large");

        for (int trial = 0; trial < 200; trial++)
        {
            int size = 1 + NextSeed() % 4;
            var picked = new List<int>();
            while (picked.Count < size)
            {
                int value = 1 + NextSeed() % 8;
                if (!picked.Contains(value)) picked.Add(value);
            }
            int target = NextSeed() % 13;

            var found = CombinationSum(picked.ToArray(), target);
            Check(Key(found) == Key(BruteForceCombinations(picked.ToArray(), target)), "oracle " + trial);
            var unique = new HashSet<string>();
            foreach (var combination in found)
            {
                int sum = 0;
                int previous = int.MinValue;
                foreach (int value in combination)
                {
                    sum += value;
                    Check(picked.Contains(value), "value in candidates " + trial);
                    Check(value >= previous, "sorted combination " + trial);
                    previous = value;
                }
                Check(sum == target, "sum " + trial);
                Check(unique.Add(string.Join(",", combination)), "duplicate " + trial);
            }
        }

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
