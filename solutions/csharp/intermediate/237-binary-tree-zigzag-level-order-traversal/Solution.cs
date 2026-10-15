// Task 237 - Binary tree zigzag level order traversal. (Tier: Intermediate, CTCI Ch 4)
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
    /// Level-order BFS with a per-level snapshot; odd levels (0-based) are reversed so
    /// the output alternates left->right / right->left. An empty tree yields []. O(n).
    /// </summary>
    public static List<List<int>> ZigzagLevelOrder(TreeNode? root)
    {
        var levels = new List<List<int>>();
        if (root == null) return levels;
        var q = new Queue<TreeNode>();
        q.Enqueue(root);
        bool leftToRight = true;
        while (q.Count > 0)
        {
            var level = new List<int>();
            for (int size = q.Count; size > 0; size--)   // snapshot of the level size
            {
                var node = q.Dequeue();
                level.Add(node.Val);
                if (node.Left != null) q.Enqueue(node.Left);
                if (node.Right != null) q.Enqueue(node.Right);
            }
            if (!leftToRight) level.Reverse();
            levels.Add(level);
            leftToRight = !leftToRight;
        }
        return levels;
    }

    public static void Main()
    {
        CheckGroups(ZigzagLevelOrder(BuildTree(3, 9, 20, null, null, 15, 7)),
            new[] { new[] { 3 }, new[] { 20, 9 }, new[] { 15, 7 } }, "classic");
        CheckGroups(ZigzagLevelOrder(BuildTree(1, 2, 3, 4, 5, 6, 7)),
            new[] { new[] { 1 }, new[] { 3, 2 }, new[] { 4, 5, 6, 7 } }, "perfect tree");
        CheckGroups(ZigzagLevelOrder(BuildTree(1, 2, 3, 4, null, null, 5)),
            new[] { new[] { 1 }, new[] { 3, 2 }, new[] { 4, 5 } }, "ragged last level");
        CheckGroups(ZigzagLevelOrder(BuildTree(1)), new[] { new[] { 1 } }, "single node");
        CheckGroups(ZigzagLevelOrder(BuildTree(1, 2)), new[] { new[] { 1 }, new[] { 2 } }, "two nodes");
        Check(ZigzagLevelOrder(null).Count == 0, "empty");
        Console.WriteLine("PASS");
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
