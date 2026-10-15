// Task 143 - Symmetric tree check (mirror). (Tier: Beginner, CTCI Ch 4)
//
// A tree is symmetric when its left and right subtrees are mirror images.
// Traversal order alone is not enough: compare a.Left with b.Right and
// a.Right with b.Left.
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

    /// <summary>Two subtrees mirror each other when their mirrored children match.</summary>
    public static bool IsMirror(TreeNode? a, TreeNode? b)
    {
        if (a == null && b == null) return true;
        if (a == null || b == null) return false;
        return a.Val == b.Val && IsMirror(a.Left, b.Right) && IsMirror(a.Right, b.Left);
    }

    /// <summary>Recursive: root is symmetric when its two subtrees mirror each other.</summary>
    public static bool IsSymmetric(TreeNode? root)
    {
        if (root == null) return true;
        return IsMirror(root.Left, root.Right);
    }

    /// <summary>Iterative variant: stack of mirrored (a, b) pairs.</summary>
    public static bool IsSymmetricIterative(TreeNode? root)
    {
        if (root == null) return true;
        var stack = new Stack<(TreeNode? A, TreeNode? B)>();
        stack.Push((root.Left, root.Right));
        while (stack.Count > 0)
        {
            var (a, b) = stack.Pop();
            if (a == null && b == null) continue;
            if (a == null || b == null || a.Val != b.Val) return false;
            stack.Push((a.Left, b.Right));
            stack.Push((a.Right, b.Left));
        }
        return true;
    }

    public static void Main()
    {
        Check(IsSymmetric(null) && IsSymmetricIterative(null), "empty");
        Check(IsSymmetric(BuildTree(1)), "single node");
        Check(IsSymmetric(BuildTree(1, 2, 2)), "two mirrors");
        Check(IsSymmetric(BuildTree(1, 2, 2, 3, 4, 4, 3)), "classic symmetric");
        // both 3s hang as RIGHT children, so the mirrored pairing fails
        Check(!IsSymmetric(BuildTree(1, 2, 2, null, 3, null, 3)), "mirrored pairing fails");
        Check(IsSymmetric(BuildTree(1, 2, 2, null, 3, 3, null)), "mirrored placement");
        Check(!IsSymmetric(BuildTree(1, 2, 3)), "different values");
        Check(!IsSymmetricIterative(BuildTree(1, 2, 2, null, 3, null, 3)), "iterative mismatch");
        Check(IsSymmetricIterative(BuildTree(1, 2, 2, null, 3, 3, null)), "iterative match");
        Check(IsSymmetric(BuildTree(1, 2, 2, 3, null, null, 3)) &&
              IsSymmetricIterative(BuildTree(1, 2, 2, 3, null, null, 3)), "deeper symmetric");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
