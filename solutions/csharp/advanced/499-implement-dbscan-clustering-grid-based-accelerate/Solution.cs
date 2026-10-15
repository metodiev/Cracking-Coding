// Task 499 - Implement DBSCAN clustering (grid-based accelerate). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// Density-based clustering (eps-neighbourhoods count points with squared distance <= eps^2,
// minPts core threshold) with two interchangeable neighbourhood engines: the naive one scans
// all n points, the accelerated one hashes points into a grid of cell size eps and probes only
// the 3x3 block of cells around a query - every point within eps must lie in that block, so the
// results are identical while far fewer distances are evaluated (the self-test counts both and
// asserts the grid strictly wins). Clusters are grown by the textbook queue expansion in index
// order, which fixes the processing order of ambiguous border points, so the two engines must
// produce element-wise identical labels (noise = -1); the comparison is also made
// order-insensitively (sets of point-index sets per cluster plus the noise set). Test sets are
// LCG-generated 2-D integer points, including duplicate-heavy sets, a mostly-noise sparse set,
// a hand-checked chain, and points exactly eps apart (the <= boundary is inclusive).
using System;
using System.Collections.Generic;
using System.Text;

internal abstract class RegionEngine
{
    internal readonly double[][] Points;
    internal readonly double Eps2;
    internal readonly double Eps;
    internal int Checks;

    protected RegionEngine(double[][] points, double eps)
    {
        Points = points;
        Eps2 = eps * eps;
        Eps = eps;
    }

    internal abstract int[] Region(int index);

    internal static int[] ToArray(List<int> list)
    {
        var result = new int[list.Count];
        for (int i = 0; i < result.Length; i++) result[i] = list[i];
        return result;
    }
}

internal class NaiveEngine : RegionEngine
{
    internal NaiveEngine(double[][] points, double eps) : base(points, eps) { }

    internal override int[] Region(int index)
    {
        var result = new List<int>();
        double px = Points[index][0];
        double py = Points[index][1];
        for (int j = 0; j < Points.Length; j++)
        {
            Checks++;
            double dx = px - Points[j][0];
            double dy = py - Points[j][1];
            if (dx * dx + dy * dy <= Eps2) result.Add(j);
        }
        return ToArray(result);
    }
}

internal class GridEngine : RegionEngine
{
    private readonly Dictionary<long, List<int>> cells = new Dictionary<long, List<int>>();

    internal GridEngine(double[][] points, double eps) : base(points, eps)
    {
        for (int j = 0; j < points.Length; j++)
        {
            long key = Key((int)Math.Floor(points[j][0] / eps), (int)Math.Floor(points[j][1] / eps));
            if (!cells.TryGetValue(key, out List<int> bucket))
            {
                bucket = new List<int>();
                cells[key] = bucket;
            }
            bucket.Add(j);
        }
    }

    private static long Key(int gx, int gy) => ((long)(gx + 2048) << 13) | (gy + 2048);

    internal override int[] Region(int index)
    {
        double px = Points[index][0];
        double py = Points[index][1];
        int cx = (int)Math.Floor(px / Eps);
        int cy = (int)Math.Floor(py / Eps);
        var result = new List<int>();
        for (int gx = cx - 1; gx <= cx + 1; gx++)
        {
            for (int gy = cy - 1; gy <= cy + 1; gy++)
            {
                if (!cells.TryGetValue(Key(gx, gy), out List<int> bucket)) continue;
                foreach (int j in bucket)
                {
                    Checks++;
                    double dx = px - Points[j][0];
                    double dy = py - Points[j][1];
                    if (dx * dx + dy * dy <= Eps2) result.Add(j);
                }
            }
        }
        int[] array = ToArray(result);
        Array.Sort(array);
        return array;
    }
}

public static class Solution
{
    private const int Noise = -1;

    private static double[][] LcgPoints(int n, int seed, int limit)
    {
        var points = new double[n][];
        for (int i = 0; i < n; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            double x = seed % limit;
            seed = (seed * 75 + 74) % 65537;
            double y = seed % limit;
            points[i] = new[] { x, y };
        }
        return points;
    }

    private static int[] Dbscan(double[][] points, int minPts, RegionEngine engine)
    {
        int n = points.Length;
        var labels = new int[n];
        for (int i = 0; i < n; i++) labels[i] = Noise;
        var visited = new bool[n];
        int cluster = 0;
        for (int start = 0; start < n; start++)
        {
            if (visited[start]) continue;
            visited[start] = true;
            int[] neighbours = engine.Region(start);
            if (neighbours.Length < minPts) continue;   // provisional noise; may become border
            labels[start] = cluster;
            var queue = new List<int>();
            foreach (int q in neighbours)
            {
                if (q != start) queue.Add(q);
            }
            for (int head = 0; head < queue.Count; head++)
            {
                int current = queue[head];
                if (!visited[current])
                {
                    visited[current] = true;
                    int[] currentNeighbours = engine.Region(current);
                    if (currentNeighbours.Length >= minPts)
                    {
                        foreach (int q in currentNeighbours)
                        {
                            if (!visited[q]) queue.Add(q);
                        }
                    }
                }
                if (labels[current] == Noise) labels[current] = cluster;    // border point
            }
            cluster += 1;
        }
        return labels;
    }

    /** Renumber clusters by first appearance so two runs compare label-for-label. */
    private static int[] Canonical(int[] labels)
    {
        var mapping = new Dictionary<int, int>();
        var result = new int[labels.Length];
        for (int i = 0; i < labels.Length; i++)
        {
            if (labels[i] == Noise)
            {
                result[i] = Noise;
            }
            else
            {
                if (!mapping.TryGetValue(labels[i], out int mapped))
                {
                    mapped = mapping.Count;
                    mapping[labels[i]] = mapped;
                }
                result[i] = mapped;
            }
        }
        return result;
    }

    /** Order-insensitive partition: per-cluster index sets (sorted keys) plus the noise set. */
    private static string Partition(int[] labels)
    {
        var clusters = new SortedDictionary<int, List<int>>();
        var noise = new List<int>();
        for (int i = 0; i < labels.Length; i++)
        {
            if (labels[i] == Noise)
            {
                noise.Add(i);
            }
            else
            {
                if (!clusters.TryGetValue(labels[i], out List<int> members))
                {
                    members = new List<int>();
                    clusters[labels[i]] = members;
                }
                members.Add(i);
            }
        }
        var parts = new List<string>();
        foreach (List<int> members in clusters.Values)
        {
            var builder = new StringBuilder();
            foreach (int member in members) builder.Append(member).Append(' ');
            parts.Add(builder.ToString());
        }
        parts.Sort(StringComparer.Ordinal);
        var combined = new StringBuilder();
        foreach (string part in parts) combined.Append(part).Append('|');
        combined.Append('#');
        foreach (int member in noise) combined.Append(member).Append(' ');
        return combined.ToString();
    }

    private sealed class Report
    {
        internal int[] Labels = Array.Empty<int>();
        internal int NaiveChecks;
        internal int GridChecks;
    }

    private static Report CompareSet(double[][] points, double eps, int minPts, string label)
    {
        var naive = new NaiveEngine(points, eps);
        var grid = new GridEngine(points, eps);
        int[] naiveLabels = Dbscan(points, minPts, naive);
        int[] gridLabels = Dbscan(points, minPts, grid);
        int[] naiveCanonical = Canonical(naiveLabels);
        int[] gridCanonical = Canonical(gridLabels);
        for (int i = 0; i < naiveCanonical.Length; i++)
        {
            Check(naiveCanonical[i] == gridCanonical[i], label + ": engines disagree");
        }
        Check(Partition(naiveLabels) == Partition(gridLabels), label + ": partitions differ");
        Check(grid.Checks < naive.Checks, label + ": grid must evaluate fewer distances");
        return new Report { Labels = naiveLabels, NaiveChecks = naive.Checks, GridChecks = grid.Checks };
    }

    public static void Main()
    {
        double[][] chain = { new[] { 0.0, 0.0 }, new[] { 1.0, 0.0 }, new[] { 2.0, 0.0 },
            new[] { 10.0, 10.0 } };
        int[] chainLabels = Dbscan(chain, 2, new NaiveEngine(chain, 1.2));
        Check(chainLabels[0] == 0 && chainLabels[1] == 0 && chainLabels[2] == 0
                && chainLabels[3] == Noise, "chain at minPts = 2");
        chainLabels = Dbscan(chain, 3, new NaiveEngine(chain, 1.2));
        Check(chainLabels[0] == 0 && chainLabels[1] == 0 && chainLabels[2] == 0
                && chainLabels[3] == Noise, "all three points are core at minPts = 3");

        double[][] pair = { new[] { 0.0, 0.0 }, new[] { 2.0, 0.0 }, new[] { 20.0, 20.0 } };
        int[] pairLabels = Dbscan(pair, 2, new NaiveEngine(pair, 2.0));
        Check(pairLabels[0] == 0 && pairLabels[1] == 0 && pairLabels[2] == Noise,
                "exact eps neighbours");
        pairLabels = Dbscan(pair, 2, new NaiveEngine(pair, 1.999));
        Check(pairLabels[0] == Noise && pairLabels[1] == Noise && pairLabels[2] == Noise,
                "just inside eps everything is noise");

        string[] names = { "dense", "fine", "coarse", "sparse", "duplicates" };
        double[][][] sets = {
            LcgPoints(60, 11, 40), LcgPoints(80, 23, 40), LcgPoints(50, 37, 40),
            LcgPoints(40, 53, 100), LcgPoints(30, 67, 10),
        };
        double[] epsilons = { 3.0, 2.0, 5.0, 1.5, 2.0 };
        int[] minPtsArr = { 3, 2, 4, 5, 3 };
        var clusterCounts = new int[sets.Length];
        var noiseCounts = new int[sets.Length];
        var sizes = new int[sets.Length];
        for (int s = 0; s < sets.Length; s++)
        {
            Report report = CompareSet(sets[s], epsilons[s], minPtsArr[s], names[s]);
            sizes[s] = sets[s].Length;
            var clusters = new HashSet<int>();
            int noise = 0;
            foreach (int label in report.Labels)
            {
                if (label == Noise) noise++;
                else clusters.Add(label);
            }
            clusterCounts[s] = clusters.Count;
            noiseCounts[s] = noise;
            Check(report.GridChecks <= report.NaiveChecks / 2, names[s] + ": the grid saves half");
        }

        Check(clusterCounts[0] >= 1 && clusterCounts[0] <= 12, "dense set: a handful of clusters");
        Check(noiseCounts[0] >= 0 && noiseCounts[0] < sizes[0], "dense set: not all noise");

        double[][] lonely = LcgPoints(12, 5, 1000);
        int[] lonelyLabels = Dbscan(lonely, 3, new GridEngine(lonely, 1.0));
        foreach (int label in lonelyLabels) Check(label == Noise, "scattered points are all noise");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
