// Task 460 - Implement small epsilon-approximate nearest neighbor in high-dim (LSH idea). (Tier: Advanced, CTCI Beyond)
using System;
using System.Collections.Generic;

internal static class AnnDemo
{
    public const int Dim = 10;
    public const int Tables = 6;
    public const int BucketWidth = 700;
    public const int NumPoints = 40;

    private static int seed = 4601;

    private static int Lcg()
    {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    public static int NextSeed() => Lcg();

    public static int[][] MakePoints(int count)
    {
        var points = new int[count][];
        for (int p = 0; p < count; p++)
        {
            points[p] = new int[Dim];
            for (int j = 0; j < Dim; j++) points[p][j] = Lcg() % 61 - 30;
        }
        return points;
    }

    public class Table
    {
        public readonly int[] Projection = new int[Dim];
        public int Offset;
        public readonly Dictionary<int, List<int>> Buckets = new Dictionary<int, List<int>>();

        public int HashOf(int[] point)
        {
            int total = Offset;
            for (int j = 0; j < Dim; j++) total += Projection[j] * point[j];
            return FloorDiv(total, BucketWidth);
        }
    }

    private static int FloorDiv(int num, int den)
        => num >= 0 ? num / den : -((-num + den - 1) / den);

    public static Table[] MakeTablesFromSeed(int tableSeed)
    {
        var tables = new Table[Tables];
        for (int t = 0; t < Tables; t++)
        {
            var table = new Table();
            for (int j = 0; j < Dim; j++)
            {
                tableSeed = (tableSeed * 75 + 74) % 65537;
                table.Projection[j] = 1 + tableSeed % 20;
            }
            tableSeed = (tableSeed * 75 + 74) % 65537;
            table.Offset = tableSeed % BucketWidth;
            tables[t] = table;
        }
        return tables;
    }

    public static Table[] MakeTables()
    {
        int tableSeed = seed;
        for (int i = 0; i < Tables * (Dim + 1); i++) Lcg();
        return MakeTablesFromSeed(tableSeed);
    }

    public static void BuildIndex(int[][] points, Table[] tables)
    {
        foreach (var table in tables)
        {
            for (int idx = 0; idx < points.Length; idx++)
            {
                int h = table.HashOf(points[idx]);
                if (!table.Buckets.TryGetValue(h, out var bucket))
                {
                    bucket = new List<int>();
                    table.Buckets[h] = bucket;
                }
                bucket.Add(idx);
            }
        }
    }

    public static int SquaredDistance(int[] p, int[] q)
    {
        int total = 0;
        for (int j = 0; j < p.Length; j++)
        {
            int diff = p[j] - q[j];
            total += diff * diff;
        }
        return total;
    }

    /// <summary>Exact nearest neighbour (the epsilon-oracle). Returns (index, squared dist).</summary>
    public static (int index, int dist) BruteForceNn(int[][] points, int[] query)
    {
        int bestIdx = 0;
        int bestDist = SquaredDistance(points[0], query);
        for (int idx = 1; idx < points.Length; idx++)
        {
            int dist = SquaredDistance(points[idx], query);
            if (dist < bestDist)
            {
                bestIdx = idx;
                bestDist = dist;
            }
        }
        return (bestIdx, bestDist);
    }

    /// <summary>Returns (index, squared dist, usedBucket); falls back to exact when buckets empty.</summary>
    public static (int index, int dist, bool usedBucket) LshApproxNn(int[][] points, Table[] tables, int[] query)
    {
        var candidates = new SortedSet<int>();
        foreach (var table in tables)
        {
            if (table.Buckets.TryGetValue(table.HashOf(query), out var bucket))
            {
                foreach (int idx in bucket) candidates.Add(idx);
            }
        }
        if (candidates.Count == 0)
        {
            var exact = BruteForceNn(points, query);
            return (exact.index, exact.dist, false);
        }
        int bestIdx = -1;
        int bestDist = int.MaxValue;
        foreach (int idx in candidates)
        {
            int dist = SquaredDistance(points[idx], query);
            if (dist < bestDist)
            {
                bestIdx = idx;
                bestDist = dist;
            }
        }
        return (bestIdx, bestDist, true);
    }

    public static (int trueIdx, int trueDist, int approxIdx, int approxDist, bool usedBucket) CheckQuery(
        int[][] points, Table[] tables, int[] query, int[] counters, string label)
    {
        var exact = BruteForceNn(points, query);
        var approx = LshApproxNn(points, tables, query);
        if (approx.dist < exact.dist) throw new Exception("FAILED: " + label + " never closer than true NN");
        if (4 * approx.dist > 9 * exact.dist) throw new Exception("FAILED: " + label + " epsilon bound");
        if (SquaredDistance(points[approx.index], query) != approx.dist)
            throw new Exception("FAILED: " + label + " distance matches");
        counters[approx.usedBucket ? 1 : 0]++;
        return (exact.index, exact.dist, approx.index, approx.dist, approx.usedBucket);
    }
}

public static class Solution
{
    public static void Main()
    {
        int[][] points = AnnDemo.MakePoints(AnnDemo.NumPoints);
        var tables = AnnDemo.MakeTables();
        AnnDemo.BuildIndex(points, tables);

        int bucketCount = 0;
        int maxBucket = 0;
        foreach (var table in tables)
        {
            foreach (var bucket in table.Buckets.Values)
            {
                bucketCount++;
                maxBucket = Math.Max(maxBucket, bucket.Count);
            }
        }
        Check(bucketCount > 0, "buckets exist");
        Check(maxBucket >= 2, "collisions exist");

        var counters = new int[2];
        for (int q = 0; q < 12; q++)
        {
            var query = new int[AnnDemo.Dim];
            for (int j = 0; j < AnnDemo.Dim; j++) query[j] = (AnnDemo.NextSeed() % 61) - 30;
            AnnDemo.CheckQuery(points, tables, query, counters, "LCG query " + q);
        }

        var outlier = new int[AnnDemo.Dim];
        for (int j = 0; j < AnnDemo.Dim; j++) outlier[j] = 10000;
        var result = AnnDemo.CheckQuery(points, tables, outlier, counters, "outlier");
        Check(!result.usedBucket, "outlier falls back");
        Check(result.approxIdx == result.trueIdx && result.approxDist == result.trueDist, "fallback is exact");

        var handPoints = new int[3][];
        for (int p = 0; p < 3; p++) handPoints[p] = new int[AnnDemo.Dim];
        handPoints[1][0] = 10;
        handPoints[2][1] = 10;
        var handTables = AnnDemo.MakeTablesFromSeed(99999);
        AnnDemo.BuildIndex(handPoints, handTables);
        var handQuery = new int[AnnDemo.Dim];
        handQuery[0] = 9;
        var handExact = AnnDemo.BruteForceNn(handPoints, handQuery);
        Check(handExact.index == 1 && handExact.dist == 1, "hand true NN");
        var handApprox = AnnDemo.LshApproxNn(handPoints, handTables, handQuery);
        Check(handApprox.dist >= 1, "hand never closer");
        Check(4 * handApprox.dist <= 9 * 1, "hand epsilon bound");
        Check(AnnDemo.SquaredDistance(handPoints[handApprox.index], handQuery) == handApprox.dist, "hand distance matches");

        Check(counters[1] >= 1, "bucket path exercised");
        Check(counters[0] >= 1, "fallback path exercised");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
