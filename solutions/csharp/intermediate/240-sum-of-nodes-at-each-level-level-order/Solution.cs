// Task 240 - Sum of nodes at each level (level order). (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;

public class TreeNode
{
    public int Val;
    public TreeNode? Left, Right;
    public TreeNode(int v) { Val = v; }
}

public static class Solution
{
    /// <summary>Build a binary tree from level-order values (null marks a missing child).</summary>
    public static TreeNode? BuildTree(params int?[] values)
    {
        if (values.Length == 0 || values[0] == null) return null;
        var root = new TreeNode(values[0]!.Value);
        var q = new Queue<TreeNode>();
        q.Enqueue(root);
        int i = 1;
        while (q.Count > 0 && i < values.Length)
        {
            var node = q.Dequeue();
            if (i < values.Length && values[i] != null)
            {
                node.Left = new TreeNode(values[i]!.Value);
                q.Enqueue(node.Left);
            }
            i++;
            if (i < values.Length && values[i] != null)
            {
                node.Right = new TreeNode(values[i]!.Value);
                q.Enqueue(node.Right);
            }
            i++;
        }
        return root;
    }

    /// <summary>
    /// Level-order BFS with a per-level snapshot sums each level independently:
    /// one pass over the tree, O(n) time. The empty tree yields [].
    /// </summary>
    public static List<int> LevelSums(TreeNode? root)
    {
        var sums = new List<int>();
        if (root == null) return sums;
        var q = new Queue<TreeNode>();
        q.Enqueue(root);
        while (q.Count > 0)
        {
            int total = 0;
            for (int size = q.Count; size > 0; size--)   // snapshot of the level size
            {
                var node = q.Dequeue();
                total += node.Val;
                if (node.Left != null) q.Enqueue(node.Left);
                if (node.Right != null) q.Enqueue(node.Right);
            }
            sums.Add(total);
        }
        return sums;
    }

    /// <summary>Independent helper: accumulate sums by depth with a DFS.</summary>
    public static List<int> LevelSumsDfs(TreeNode? root)
    {
        var sums = new List<int>();
        Dfs(root, 0, sums);
        return sums;
    }

    private static void Dfs(TreeNode? node, int depth, List<int> sums)
    {
        if (node == null) return;
        if (depth == sums.Count) sums.Add(0);
        sums[depth] += node.Val;
        Dfs(node.Left, depth + 1, sums);
        Dfs(node.Right, depth + 1, sums);
    }

    private static TreeNode? Prune(TreeNode? node, Random rng)
    {
        if (node == null || rng.NextDouble() < 0.25) return null;
        node.Left = Prune(node.Left, rng);
        node.Right = Prune(node.Right, rng);
        return node;
    }

    public static void Main()
    {
        CheckGroup(LevelSums(BuildTree(1, 2, 3, 4, 5, null, 6)), new[] { 1, 5, 15 },
            "classic: {1}, {2,3}, {4,5,6}");
        CheckGroup(LevelSums(BuildTree(1, 2, 3)), new[] { 1, 5 }, "two levels");
        CheckGroup(LevelSums(BuildTree(1)), new[] { 1 }, "single node");
        Check(LevelSums(null).Count == 0, "empty");
        CheckGroup(LevelSums(BuildTree(1, 2, null, 3, null, 4)), new[] { 1, 2, 3, 4 }, "left chain");
        CheckGroup(LevelSums(BuildTree(-1, -2, -3)), new[] { -1, -5 }, "negatives");
        CheckGroup(LevelSums(BuildTree(0, 0, 0, 0)), new[] { 0, 0, 0 }, "zeros");

        var rng = new Random(240);
        for (int test = 0; test < 200; test++)      // BFS result == DFS depth accumulation
        {
            int n = rng.Next(15);
            var nodes = new TreeNode[n];
            for (int i = 0; i < n; i++) nodes[i] = new TreeNode(rng.Next(41) - 20);
            for (int i = 0; i < n; i++)
            {
                if (2 * i + 1 < n) nodes[i].Left = nodes[2 * i + 1];
                if (2 * i + 2 < n) nodes[i].Right = nodes[2 * i + 2];
            }
            TreeNode? root = n == 0 ? null : Prune(nodes[0], rng);
            CheckGroup(LevelSums(root), LevelSumsDfs(root).ToArray(), "random agreement");
        }
        Console.WriteLine("PASS");
    }

    private static void CheckGroup(List<int> actual, int[] expected, string label)
    {
        bool ok = actual.Count == expected.Length;
        for (int i = 0; ok && i < expected.Length; i++) ok = actual[i] == expected[i];
        Check(ok, label);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
