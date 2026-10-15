// Task 145 - Validate BST property (inorder). (Tier: Beginner, CTCI Ch 4)
//
// Two equivalent O(n) routes:
//   1. min/max range recursion - every node must stay inside the open interval
//      inherited from its ancestors (long bounds, not int bounds, so
//      int.MinValue / int.MaxValue values are still legal);
//   2. inorder must be strictly increasing (see IsValidBstInorder below).
// Comparing only a node with its own children is NOT enough:
// [5, 1, 4, null, null, 3, 6] passes that weaker test although 3 < 5.
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

    /// <summary>Range recursion: each node carries the (lo, hi) window it must fit in.</summary>
    public static bool IsValidBst(TreeNode? root) => ValidRange(root, long.MinValue, long.MaxValue);

    private static bool ValidRange(TreeNode? node, long lo, long hi)
    {
        if (node == null) return true;
        if (node.Val <= lo || node.Val >= hi) return false;
        return ValidRange(node.Left, lo, node.Val) && ValidRange(node.Right, node.Val, hi);
    }

    /// <summary>Alternative route: walk the tree in order and require a strict increase.</summary>
    public static bool IsValidBstInorder(TreeNode? root)
    {
        long previous = long.MinValue;
        var stack = new Stack<TreeNode>();
        var node = root;
        while (stack.Count > 0 || node != null)
        {
            while (node != null)
            {
                stack.Push(node);
                node = node.Left;
            }
            node = stack.Pop();
            if (node.Val <= previous) return false;
            previous = node.Val;
            node = node.Right;
        }
        return true;
    }

    public static void Main()
    {
        Check(IsValidBst(null) && IsValidBstInorder(null), "empty");
        Check(IsValidBst(BuildTree(2, 1, 3)), "2,1,3");
        Check(IsValidBst(BuildTree(7)), "single");
        Check(IsValidBst(BuildTree(3, 2, null, 1)), "left chain");
        // classic pitfall: 4 sits in the right subtree of 5 but is smaller than 5
        Check(!IsValidBst(BuildTree(5, 1, 4, null, null, 3, 6)), "weak test would pass");
        Check(!IsValidBst(BuildTree(10, 5, 15, null, null, 6, 20)), "deep violation");
        Check(!IsValidBst(BuildTree(5, 4, 6, null, null, 3, 7)), "3 under the right subtree");
        Check(!IsValidBst(BuildTree(1, 1)), "duplicates");
        // long bounds: int extremes are legal node values
        Check(IsValidBst(BuildTree(int.MaxValue, int.MinValue, null)), "int extremes");
        var trap = BuildTree(5, 1, 4, null, null, 3, 6);
        Check(IsValidBst(trap) == IsValidBstInorder(trap), "both routes reject the trap");
        Check(IsValidBst(BuildTree(2, 1, 3)) == IsValidBstInorder(BuildTree(2, 1, 3)),
              "both routes accept a real BST");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
