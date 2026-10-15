// Task 153 - Count nodes in complete binary tree faster than O(n). (Tier: Beginner, CTCI Ch 4)
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

    private static int LeftHeight(TreeNode? node)
    {
        int h = 0;
        while (node != null) { h++; node = node.Left; }
        return h;
    }

    private static int RightHeight(TreeNode? node)
    {
        int h = 0;
        while (node != null) { h++; node = node.Right; }
        return h;
    }

    /// <summary>
    /// O(log^2 n) node count for a complete binary tree. If the left and right
    /// spine heights match, the subtree is perfect and holds 2^h - 1 nodes.
    /// </summary>
    public static int CountNodes(TreeNode? root)
    {
        if (root == null) return 0;
        int lh = LeftHeight(root), rh = RightHeight(root);
        if (lh == rh) return (1 << lh) - 1;
        return 1 + CountNodes(root.Left) + CountNodes(root.Right);
    }

    /// <summary>Plain O(n) count used only to cross-check CountNodes.</summary>
    public static int NaiveCount(TreeNode? node)
    {
        if (node == null) return 0;
        return 1 + NaiveCount(node.Left) + NaiveCount(node.Right);
    }

    public static void Main()
    {
        Check(CountNodes(null) == 0, "empty");
        Check(CountNodes(BuildTree(1)) == 1, "single");

        int[] sizes = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 16, 31, 63, 64, 65, 100, 1000 };
        foreach (int n in sizes)
        {
            var values = new int?[n];
            for (int i = 0; i < n; i++) values[i] = i;
            var root = BuildTree(values);   // no gaps -> complete tree
            Check(CountNodes(root) == n, "count == " + n);
            Check(CountNodes(root) == NaiveCount(root), "fast == naive for n=" + n);
        }

        Check(CountNodes(BuildTree(1, 2, 3, 4, 5, 6, 7)) == 7, "perfect tree");
        var partial = BuildTree(1, 2, 3, 4, 5, 6, null);   // last level partially filled
        Check(CountNodes(partial) == 6 && NaiveCount(partial) == 6, "partial last level");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
