// Task 165 - Reservoir sampling for k items from stream. (Tier: Beginner, CTCI Ch 6)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    /// <summary>Algorithm R with an injected Random, so tests are reproducible.</summary>
    public static List<int> ReservoirSample(int[] stream, int k, Random rng)
    {
        var sample = new List<int>();
        for (int i = 0; i < stream.Length; i++)
        {
            if (i < k)
            {
                sample.Add(stream[i]);
            }
            else
            {
                int j = rng.Next(i + 1);
                if (j < k) sample[j] = stream[i];
            }
        }
        return sample;
    }

    public static void Main()
    {
        var stream = Enumerable.Range(0, 100).ToArray();

        var sample = ReservoirSample(stream, 5, new Random(7));
        Check(sample.Count == 5, "sample size");
        Check(sample.Distinct().Count() == 5, "distinct picks");
        Check(string.Join(",", sample) == string.Join(",", ReservoirSample(stream, 5, new Random(7))),
            "reproducible with same seed");
        Check(string.Join(",", sample) != string.Join(",", ReservoirSample(stream, 5, new Random(8))),
            "differs with another seed");

        Check(string.Join(",", ReservoirSample(new[] { 1, 2, 3 }, 5, new Random(1))) == "1,2,3", "k >= n keeps all");
        Check(ReservoirSample(new int[0], 3, new Random(1)).Count == 0, "empty stream");

        int trials = 4000, hits = 0;
        var rng = new Random(123);
        for (int t = 0; t < trials; t++)
            if (ReservoirSample(stream, 5, rng).Contains(0)) hits++;
        double ratio = hits / (double)trials;
        Check(ratio > 0.03 && ratio < 0.08, "inclusion ratio ~0.05, got " + ratio.ToString("F3"));
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
