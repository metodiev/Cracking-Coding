// Task 458 - Implement dynamic programming on trees with knapsack per node. (Tier: Advanced, CTCI Ch 4)
using System;
using System.Collections.Generic;

internal static class TreeKnapsack
{
    private const int NegInf = -1_000_000_000;

    /// <summary>Best sum of <= k chosen vertices with the parent-before-child closure rule.</summary>
    public static int Solve(int n, int[] parent, int[] weight, int k)
    {
        if (k <= 0 || n == 0) return 0;
        var children = new List<List<int>>();
        for (int i = 0; i < n; i++) children.Add(new List<int>());
        for (int v = 1; v < n; v++) children[parent[v]].Add(v);
        var (dp, _) = Dfs(0, k, children, weight);
        int best = 0;
        for (int j = 1; j <= k; j++) best = Math.Max(best, dp[j]);
        return best;
    }

    private static (int[] dp, int size) Dfs(int u, int k, List<List<int>> children, int[] weight)
    {
        int subtreeSize = 1;
        var dp = new int[k + 1];
        for (int i = 0; i <= k; i++) dp[i] = NegInf;
        dp[1] = weight[u];
        foreach (int child in children[u])
        {
            var (cdp, childSize) = Dfs(child, k, children, weight);
            var newDp = (int[])dp.Clone();
            for (int i = 1; i <= Math.Min(k, subtreeSize); i++)
            {
                if (dp[i] == NegInf) continue;
                for (int j = 1; j <= Math.Min(childSize, k - i); j++)
                {
                    if (cdp[j] == NegInf) continue;
                    newDp[i + j] = Math.Max(newDp[i + j], dp[i] + cdp[j]);
                }
            }
            dp = newDp;
            subtreeSize += childSize;
        }
        return (dp, subtreeSize);
    }

    /// <summary>Exhaustive over all parent-closed subsets of size <= k (empty set allowed).</summary>
    public static int Brute(int n, int[] parent, int[] weight, int k)
    {
        int best = 0;
        for (int mask = 0; mask < (1 << n); mask++)
        {
            int size = 0;
            for (int v = 0; v < n; v++) size += (mask >> v) & 1;
            if (size > k) continue;
            bool ok = true;
            for (int v = 1; v < n; v++)
            {
                if (((mask >> v) & 1) == 1 && ((mask >> parent[v]) & 1) == 0)
                {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            int total = 0;
            for (int v = 0; v < n; v++) if (((mask >> v) & 1) == 1) total += weight[v];
            best = Math.Max(best, total);
        }
        return best;
    }

    public static List<(int n, int[] parent, int[] weight)> LcgTrees(int count, int seed)
    {
        var trees = new List<(int, int[], int[])>();
        for (int g = 0; g < count; g++)
        {
            seed = (seed * 75 + 74) % 65537;
            int n = 6 + seed % 4;
            var parent = new int[n];
            var weight = new int[n];
            parent[0] = -1;
            seed = (seed * 75 + 74) % 65537;
            weight[0] = seed % 31 - 15;
            for (int v = 1; v < n; v++)
            {
                seed = (seed * 75 + 74) % 65537;
                parent[v] = seed % v;
                seed = (seed * 75 + 74) % 65537;
                weight[v] = seed % 31 - 15;
            }
            trees.Add((n, parent, weight));
        }
        return trees;
    }
}

public static class Solution
{
    public static void Main()
    {
        var p1 = new[] { -1, 0, 0, 1 };
        var w1 = new[] { 5, 3, 4, -10 };
        Check(TreeKnapsack.Solve(4, p1, w1, 1) == 5, "tree1 K=1");
        Check(TreeKnapsack.Solve(4, p1, w1, 2) == 9, "tree1 K=2");
        Check(TreeKnapsack.Solve(4, p1, w1, 3) == 12, "tree1 K=3");
        Check(TreeKnapsack.Solve(4, p1, w1, 4) == 12, "tree1 K=4");
        Check(TreeKnapsack.Solve(4, p1, w1, 0) == 0, "tree1 K=0");
        for (int k = 0; k <= 4; k++)
        {
            Check(TreeKnapsack.Solve(4, p1, w1, k) == TreeKnapsack.Brute(4, p1, w1, k), "tree1 brute " + k);
        }

        var p2 = new[] { -1, 0, 0, 1, 1 };
        var w2 = new[] { 10, 8, 5, 7, 6 };
        Check(TreeKnapsack.Solve(5, p2, w2, 3) == 25, "tree2 K=3");
        for (int k = 0; k <= 5; k++)
        {
            Check(TreeKnapsack.Solve(5, p2, w2, k) == TreeKnapsack.Brute(5, p2, w2, k), "tree2 brute " + k);
        }

        var p3 = new[] { -1, 0 };
        var w3 = new[] { -2, -3 };
        Check(TreeKnapsack.Solve(2, p3, w3, 2) == 0, "negative K=2");
        Check(TreeKnapsack.Solve(2, p3, w3, 1) == 0, "negative K=1");

        var p4 = new[] { -1, 0, 1 };
        var w4 = new[] { 1, -5, 10 };
        Check(TreeKnapsack.Solve(3, p4, w4, 1) == 1, "chain K=1");
        Check(TreeKnapsack.Solve(3, p4, w4, 2) == 1, "chain K=2");
        Check(TreeKnapsack.Solve(3, p4, w4, 3) == 6, "chain K=3");
        for (int k = 0; k <= 3; k++)
        {
            Check(TreeKnapsack.Solve(3, p4, w4, k) == TreeKnapsack.Brute(3, p4, w4, k), "chain brute " + k);
        }

        int checkedCount = 0, comparisons = 0, expectedComparisons = 0;
        foreach (int seed in new[] { 4581, 4582, 4583 })
        {
            foreach (var (n, parent, weight) in TreeKnapsack.LcgTrees(5, seed))
            {
                int previous = 0;
                for (int k = 0; k <= n; k++)
                {
                    int got = TreeKnapsack.Solve(n, parent, weight, k);
                    int want = TreeKnapsack.Brute(n, parent, weight, k);
                    Check(got == want, "LCG tree K=" + k);
                    Check(got >= previous, "monotone in K");
                    previous = got;
                    comparisons++;
                }
                expectedComparisons += n + 1;
                checkedCount++;
            }
        }
        Check(checkedCount == 15, "tree count");
        Check(comparisons == expectedComparisons, "comparison count");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
