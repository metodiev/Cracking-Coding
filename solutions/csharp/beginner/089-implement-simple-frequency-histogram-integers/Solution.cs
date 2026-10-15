// Task 089 - Implement simple frequency histogram (integers). (Tier: Beginner, CTCI Ch 1)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static SortedDictionary<int, int> HistogramCounts(int[] values)
    {
        var counts = new SortedDictionary<int, int>();
        foreach (int value in values)
        {
            counts.TryGetValue(value, out int current);
            counts[value] = current + 1;
        }
        return counts;
    }

    public static List<string> RenderHistogram(SortedDictionary<int, int> counts)
    {
        var lines = new List<string>();
        foreach (var pair in counts)
        {
            lines.Add(pair.Key + ": " + new string('#', pair.Value));
        }
        return lines;
    }

    public static void Main()
    {
        var counts = HistogramCounts(new[] { 1, 2, 2, 3, 3, 3 });
        Check(counts.Count == 3 && counts[1] == 1 && counts[2] == 2 && counts[3] == 3, "counts");
        var lines = RenderHistogram(counts);
        Check(lines.Count == 3, "one line per distinct value");
        Check(string.Join("|", lines) == "1: #|2: ##|3: ###", "rendering");

        var empty = HistogramCounts(new int[0]);
        Check(empty.Count == 0 && RenderHistogram(empty).Count == 0, "empty");

        var negatives = HistogramCounts(new[] { -1, -1, 0, 2, 2, 2 });
        Check(negatives.Count == 3 && negatives[-1] == 2 && negatives[0] == 1 && negatives[2] == 3, "negatives counts");
        Check(string.Join("|", RenderHistogram(negatives)) == "-1: ##|0: #|2: ###", "negatives rendering");

        var single = HistogramCounts(new[] { 7 });
        Check(single.Count == 1 && single[7] == 1, "single");
        Check(string.Join("|", RenderHistogram(single)) == "7: #", "single rendering");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
