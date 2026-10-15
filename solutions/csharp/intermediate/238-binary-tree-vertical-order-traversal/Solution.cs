// Task 238 - Binary tree vertical order traversal. (Tier: Intermediate, CTCI Ch 4)
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
    /// Columns from the root (root = 0, left = -1, right = +1). BFS with columns reads
    /// each column top-to-bottom and visits same-row nodes left-to-right (tie rule);
    /// a sorted dictionary emits the columns in increasing order. O(n log n) time.
    /// </summary>
    public static List<List<int>> VerticalOrder(TreeNode? root)
    {
        var result = new List<List<int>>();
        if (root == null) return result;
        var columns = new SortedDictionary<int, List<int>>();
        var q = new Queue<(TreeNode Node, int Col)>();
        q.Enqueue((root, 0));
        while (q.Count > 0)
        {
            var (node, col) = q.Dequeue();
            if (!columns.TryGetValue(col, out var list))
            {
                list = new List<int>();
                columns[col] = list;
            }
            list.Add(node.Val);
            if (node.Left != null) q.Enqueue((node.Left, col - 1));
            if (node.Right != null) q.Enqueue((node.Right, col + 1));
        }
        foreach (var list in columns.Values) result.Add(list);
        return result;
    }

    /// <summary>Independent helper: sort (column, row, preorder index) then group by column.</summary>
    public static List<List<int>> VerticalOrderSorted(TreeNode? root)
    {
        var items = new List<(int Col, int Row, int Seq, int Val)>();
        Dfs(root, 0, 0, items);
        items.Sort((a, b) =>
        {
            if (a.Col != b.Col) return a.Col.CompareTo(b.Col);
            if (a.Row != b.Row) return a.Row.CompareTo(b.Row);
            return a.Seq.CompareTo(b.Seq);
        });
        var result = new List<List<int>>();
        int lastCol = int.MinValue;
        foreach (var item in items)
        {
            if (result.Count == 0 || item.Col != lastCol)
            {
                result.Add(new List<int>());
                lastCol = item.Col;
            }
            result[result.Count - 1].Add(item.Val);
        }
        return result;
    }

    private static void Dfs(TreeNode? node, int col, int row, List<(int, int, int, int)> items)
    {
        if (node == null) return;
        items.Add((col, row, items.Count, node.Val));
        Dfs(node.Left, col - 1, row + 1, items);
        Dfs(node.Right, col + 1, row + 1, items);
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
        var classic = BuildTree(3, 9, 20, null, null, 15, 7);
        CheckGroups(VerticalOrder(classic),
            new[] { new[] { 9 }, new[] { 3, 15 }, new[] { 20 }, new[] { 7 } }, "classic");
        Check(SameGroups(VerticalOrderSorted(classic), VerticalOrder(classic)), "classic sorted agrees");

        var perfect = BuildTree(1, 2, 3, 4, 5, 6, 7);
        CheckGroups(VerticalOrder(perfect),
            new[] { new[] { 4 }, new[] { 2 }, new[] { 1, 5, 6 }, new[] { 3 }, new[] { 7 } },
            "perfect tree with left-to-right tie");
        Check(SameGroups(VerticalOrderSorted(perfect), VerticalOrder(perfect)), "perfect sorted agrees");

        var deeper = BuildTree(3, 9, 20, null, null, 15, 7, 8, 11);
        CheckGroups(VerticalOrder(deeper),
            new[] { new[] { 9, 8 }, new[] { 3, 15 }, new[] { 20, 11 }, new[] { 7 } }, "stacked columns");
        Check(SameGroups(VerticalOrderSorted(deeper), VerticalOrder(deeper)), "deeper sorted agrees");

        var chain = BuildTree(1, 2, null, 3, null, 4);
        CheckGroups(VerticalOrder(chain),
            new[] { new[] { 4 }, new[] { 3 }, new[] { 2 }, new[] { 1 } }, "left chain");
        CheckGroups(VerticalOrder(BuildTree(5)), new[] { new[] { 5 } }, "single node");
        Check(VerticalOrder(null).Count == 0, "empty");

        var rng = new Random(238);
        for (int test = 0; test < 200; test++)      // BFS order == sorted-DFS order
        {
            int n = rng.Next(15);
            var nodes = new TreeNode[n];
            for (int i = 0; i < n; i++) nodes[i] = new TreeNode(i + 1);
            for (int i = 0; i < n; i++)
            {
                if (2 * i + 1 < n) nodes[i].Left = nodes[2 * i + 1];
                if (2 * i + 2 < n) nodes[i].Right = nodes[2 * i + 2];
            }
            TreeNode? root = n == 0 ? null : Prune(nodes[0], rng);
            Check(SameGroups(VerticalOrder(root), VerticalOrderSorted(root)), "random agreement");
        }
        Console.WriteLine("PASS");
    }

    private static bool SameGroups(List<List<int>> a, List<List<int>> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++)
        {
            if (a[i].Count != b[i].Count) return false;
            for (int j = 0; j < a[i].Count; j++)
                if (a[i][j] != b[i][j]) return false;
        }
        return true;
    }

    private static void CheckGroups(List<List<int>> actual, int[][] expected, string label)
    {
        bool ok = actual.Count == expected.Length;
        for (int i = 0; ok && i < expected.Length; i++)
        {
            ok = actual[i].Count == expected[i].Length;
            for (int j = 0; ok && j < expected[i].Length; j++)
                ok = actual[i][j] == expected[i][j];
        }
        Check(ok, label);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
