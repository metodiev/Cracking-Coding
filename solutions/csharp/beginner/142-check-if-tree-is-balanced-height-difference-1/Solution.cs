// Task 142 - Check if tree is balanced (height difference <= 1). (Tier: Beginner, CTCI Ch 4)
//
// Balanced = at EVERY node the two subtree heights differ by at most 1.
// The bottom-up pass returns a height, or -1 as soon as a node fails: O(n).
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

    /// <summary>Number of nodes on the longest downward path (empty tree = 0).</summary>
    public static int Height(TreeNode? node)
    {
        if (node == null) return 0;
        return 1 + Math.Max(Height(node.Left), Height(node.Right));
    }

    /// <summary>Subtree height, or -1 when some node inside the subtree is unbalanced.</summary>
    public static int HeightOrFailure(TreeNode? node)
    {
        if (node == null) return 0;
        int left = HeightOrFailure(node.Left);
        if (left == -1) return -1;
        int right = HeightOrFailure(node.Right);
        if (right == -1) return -1;
        if (Math.Abs(left - right) > 1) return -1;
        return 1 + Math.Max(left, right);
    }

    /// <summary>O(n) bottom-up: the -1 sentinel short-circuits everything above it.</summary>
    public static bool IsBalanced(TreeNode? root) => HeightOrFailure(root) != -1;

    /// <summary>O(n^2) contrast: recompute both subtree heights at every node.</summary>
    public static bool IsBalancedNaive(TreeNode? root)
    {
        if (root == null) return true;
        if (Math.Abs(Height(root.Left) - Height(root.Right)) > 1) return false;
        return IsBalancedNaive(root.Left) && IsBalancedNaive(root.Right);
    }

    public static void Main()
    {
        Check(IsBalanced(null) && IsBalancedNaive(null), "empty");
        Check(IsBalanced(BuildTree(1)) && IsBalanced(BuildTree(1, 2)), "tiny");
        Check(IsBalanced(BuildTree(3, 9, 20, null, null, 15, 7)), "balanced sample");
        Check(IsBalanced(BuildTree(1, 2, 3, 4, 5)), "balanced deeper");
        Check(!IsBalanced(BuildTree(1, 2, 2, 3, 3, null, null, 4, 4)), "unbalanced sample");
        Check(!IsBalanced(BuildTree(1, 2, null, 3, null, 4)), "left chain");
        Check(!IsBalanced(BuildTree(1, null, 2, null, 3)), "right chain");
        Check(Height(BuildTree(1, 2, 3, 4, 5, null, 6)) == 3, "height of sample");
        Check(IsBalanced(BuildTree(1, 2, 2, 3, 3, null, null, 4, 4)) ==
              IsBalancedNaive(BuildTree(1, 2, 2, 3, 3, null, null, 4, 4)), "bottom-up == naive");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
