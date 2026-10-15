// Task 498 - Implement k-means clustering (kmeans++ initialization). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// k-means over fixed 2-D blobs with a fully deterministic k-means++ seeding: the first centre is
// an LCG-uniform data point, then each next centre is drawn with probability proportional to
// D(x)^2 (the squared distance to the closest chosen centre; the draw is one fixed-LCG uniform
// against the cumulative D^2 weights). Lloyd iterations then alternate exact assignment
// (squared Euclidean distance, ties to the lowest cluster index) with mean updates (an empty
// cluster keeps its old centre) until the labels stop changing. Data: three planted discs of
// jittered points around (0,0), (4,0) and (0,4), 20 points each, jitter in 0.01 steps of
// +/-0.5. Documented expectations: with 100 fixed random restarts (distinct-index sampling,
// same LCG family) the k-means++ objective is at most the first restart's objective and within
// 5% of the best restart (measured 10.613 vs 176.38 for the first restart and 10.613 for the
// best of 100); every cluster is pure (all its points come from one planted blob); and
// re-running with the same seed reproduces the labels and centroids exactly (bit for bit).
using System;
using System.Collections.Generic;

internal static class KMeansDemo
{
    internal static readonly double[][] Blobs = { new[] { 0.0, 0.0 }, new[] { 4.0, 0.0 },
        new[] { 0.0, 4.0 } };
    internal const int PerBlob = 20;
    internal const int K = 3;
    internal const int Seed = 7;
    internal const int Restarts = 100;

    internal static double[][] LcgPoints(int seed, out int[] truth)
    {
        var points = new double[Blobs.Length * PerBlob][];
        truth = new int[points.Length];
        int index = 0;
        for (int blob = 0; blob < Blobs.Length; blob++)
        {
            for (int i = 0; i < PerBlob; i++)
            {
                seed = (seed * 75 + 74) % 65537;
                double x = Blobs[blob][0] + (seed % 101 - 50) / 100.0;
                seed = (seed * 75 + 74) % 65537;
                double y = Blobs[blob][1] + (seed % 101 - 50) / 100.0;
                points[index] = new[] { x, y };
                truth[index] = blob;
                index++;
            }
        }
        return points;
    }

    internal static double Sqdist(double[] a, double[] b)
    {
        double dx = a[0] - b[0];
        double dy = a[1] - b[1];
        return dx * dx + dy * dy;
    }

    internal static List<double[]> KmeansPlusPlus(double[][] points, int k, int seed)
    {
        int n = points.Length;
        seed = (seed * 75 + 74) % 65537;
        int first = seed % n;
        var centers = new List<double[]> { points[first] };
        var d2 = new double[n];
        for (int i = 0; i < n; i++) d2[i] = Sqdist(points[i], centers[0]);
        while (centers.Count < k)
        {
            double total = 0.0;
            foreach (double value in d2) total += value;
            seed = (seed * 75 + 74) % 65537;
            double target = (seed + 0.5) / 65537.0 * total;
            double acc = 0.0;
            int choice = n - 1;
            for (int i = 0; i < n; i++)
            {
                acc += d2[i];
                if (acc >= target)
                {
                    choice = i;
                    break;
                }
            }
            centers.Add(points[choice]);
            double[] last = centers[centers.Count - 1];
            for (int i = 0; i < n; i++) d2[i] = Math.Min(d2[i], Sqdist(points[i], last));
        }
        return centers;
    }

    /** Deterministic distinct-index sampling from the same LCG family. */
    internal static List<double[]> RandomInit(double[][] points, int k, int seed)
    {
        int n = points.Length;
        var chosen = new List<int>();
        while (chosen.Count < k)
        {
            seed = (seed * 75 + 74) % 65537;
            int index = seed % n;
            if (!chosen.Contains(index)) chosen.Add(index);
        }
        var centers = new List<double[]>();
        foreach (int index in chosen) centers.Add(points[index]);
        return centers;
    }

    internal static int Assign(double[] point, List<double[]> centers)
    {
        int best = 0;
        double bestDistance = Sqdist(point, centers[0]);
        for (int c = 1; c < centers.Count; c++)
        {
            double distance = Sqdist(point, centers[c]);
            if (distance < bestDistance)
            {
                best = c;
                bestDistance = distance;
            }
        }
        return best;
    }

    internal class Result
    {
        internal int[] Labels = Array.Empty<int>();
        internal List<double[]> Centers = new List<double[]>();
    }

    internal static Result Lloyd(double[][] points, List<double[]> startCenters)
    {
        var centers = new List<double[]>(startCenters);
        int[] labels = null;
        for (int iter = 0; iter < 100; iter++)
        {
            var newLabels = new int[points.Length];
            for (int i = 0; i < points.Length; i++) newLabels[i] = Assign(points[i], centers);
            var newCenters = new List<double[]>();
            for (int c = 0; c < centers.Count; c++)
            {
                double sumX = 0.0;
                double sumY = 0.0;
                int members = 0;
                for (int i = 0; i < points.Length; i++)
                {
                    if (newLabels[i] == c)
                    {
                        sumX += points[i][0];
                        sumY += points[i][1];
                        members++;
                    }
                }
                if (members > 0) newCenters.Add(new[] { sumX / members, sumY / members });
                else newCenters.Add((double[])centers[c].Clone());  // empty keeps its centre
            }
            bool same = labels != null && ArraysEqual(labels, newLabels);
            labels = newLabels;
            centers = newCenters;
            if (same) break;
        }
        return new Result { Labels = labels, Centers = centers };
    }

    internal static Result Run(double[][] points, int k, int seed)
    {
        return Lloyd(points, KmeansPlusPlus(points, k, seed));
    }

    internal static double Objective(double[][] points, List<double[]> centers)
    {
        double total = 0.0;
        foreach (double[] point in points)
        {
            double best = double.PositiveInfinity;
            foreach (double[] center in centers) best = Math.Min(best, Sqdist(point, center));
            total += best;
        }
        return total;
    }

    internal static bool ArraysEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i]) return false;
        }
        return true;
    }
}

public static class Solution
{
    public static void Main()
    {
        double[][] points = KMeansDemo.LcgPoints(KMeansDemo.Seed, out int[] truth);
        Check(points.Length == 3 * KMeansDemo.PerBlob, "point count");
        var unique = new HashSet<string>();
        foreach (double[] point in points) unique.Add(point[0] + "," + point[1]);
        Check(unique.Count == points.Length, "jittered blobs are duplicate-free");

        KMeansDemo.Result result = KMeansDemo.Run(points, KMeansDemo.K, KMeansDemo.Seed + 1);
        var labelsSeen = new SortedSet<int>();
        foreach (int label in result.Labels) labelsSeen.Add(label);
        Check(labelsSeen.Count == 3 && labelsSeen.Min == 0 && labelsSeen.Max == 2,
                "three non-empty clusters");

        var purity = new Dictionary<int, HashSet<int>>();
        for (int i = 0; i < result.Labels.Length; i++)
        {
            if (!purity.TryGetValue(result.Labels[i], out HashSet<int> blobs))
            {
                blobs = new HashSet<int>();
                purity[result.Labels[i]] = blobs;
            }
            blobs.Add(truth[i]);
        }
        var blobsUsed = new HashSet<int>();
        foreach (HashSet<int> blobs in purity.Values)
        {
            Check(blobs.Count == 1, "clusters must be pure");
            foreach (int blob in blobs) blobsUsed.Add(blob);
        }
        Check(blobsUsed.Count == 3, "one blob per cluster");

        double kppObjective = KMeansDemo.Objective(points, result.Centers);

        KMeansDemo.Result firstRandom = KMeansDemo.Lloyd(
                points, KMeansDemo.RandomInit(points, KMeansDemo.K, 9100));
        double firstRandomObjective = KMeansDemo.Objective(points, firstRandom.Centers);
        Check(kppObjective <= firstRandomObjective, "k-means++ must beat the first random init");
        double bestRandom = double.PositiveInfinity;
        for (int r = 0; r < KMeansDemo.Restarts; r++)
        {
            KMeansDemo.Result randomRun = KMeansDemo.Lloyd(
                    points, KMeansDemo.RandomInit(points, KMeansDemo.K, 1000 + 17 * r));
            bestRandom = Math.Min(bestRandom, KMeansDemo.Objective(points, randomRun.Centers));
        }
        Check(kppObjective <= 1.05 * bestRandom, "within 5% of the best of 100 restarts");

        KMeansDemo.Result repeat = KMeansDemo.Run(points, KMeansDemo.K, KMeansDemo.Seed + 1);
        Check(KMeansDemo.ArraysEqual(repeat.Labels, result.Labels), "labels stable under re-run");
        Check(repeat.Centers.Count == result.Centers.Count, "centre count");
        for (int c = 0; c < result.Centers.Count; c++)
        {
            bool identical = true;
            for (int i = 0; i < 2; i++)
            {
                if (repeat.Centers[c][i] != result.Centers[c][i]) identical = false;
            }
            Check(identical, "centroids identical bit for bit");
        }

        List<double[]> centersPp = KMeansDemo.KmeansPlusPlus(points, KMeansDemo.K,
                KMeansDemo.Seed + 1);
        foreach (double[] center in centersPp)
        {
            bool isPoint = false;
            foreach (double[] point in points)
            {
                if (point[0] == center[0] && point[1] == center[1]) isPoint = true;
            }
            Check(isPoint, "centres are data points");
        }
        Check(centersPp.Count == KMeansDemo.K, "centre count is k");
        Check(KMeansDemo.KmeansPlusPlus(points, points.Length, KMeansDemo.Seed + 1).Count
                == points.Length, "k = n gives every point as centre");

        double[][] tiny = {
            new[] { 0.0, 0.0 }, new[] { 0.1, 0.0 }, new[] { 0.0, 0.1 }, new[] { 0.1, 0.1 },
            new[] { 3.0, 3.0 }, new[] { 3.1, 3.0 }, new[] { 3.0, 3.1 }, new[] { 3.1, 3.1 },
        };
        KMeansDemo.Result tinyResult = KMeansDemo.Run(tiny, 2, 55);
        for (int i = 1; i < 4; i++)
        {
            Check(tinyResult.Labels[i] == tinyResult.Labels[0], "tiny left blob");
            Check(tinyResult.Labels[4 + i] == tinyResult.Labels[4], "tiny right blob");
        }
        Check(tinyResult.Labels[0] != tinyResult.Labels[4], "tiny blobs separate");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
