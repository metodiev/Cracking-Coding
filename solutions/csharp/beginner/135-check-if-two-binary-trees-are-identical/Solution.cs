// Task 135 - Check if two binary trees are identical. (Tier: Beginner, CTCI Ch 4)
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

    public static bool AreIdentical(TreeNode? a, TreeNode? b)
    {
        if (a == null && b == null) return true;
        if (a == null || b == null) return false;
        return a.Val == b.Val
            && AreIdentical(a.Left, b.Left)
            && AreIdentical(a.Right, b.Right);
    }

    public static void Main()
    {
        Check(AreIdentical(BuildTree(1, 2, 3, null, 4), BuildTree(1, 2, 3, null, 4)), "equal trees");
        Check(!AreIdentical(BuildTree(1, 2, 3, null, 4), BuildTree(1, 2, 3, 4)), "different shape");
        Check(!AreIdentical(BuildTree(1, 2), BuildTree(1, 3)), "different values");
        Check(AreIdentical(null, null), "both empty");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
