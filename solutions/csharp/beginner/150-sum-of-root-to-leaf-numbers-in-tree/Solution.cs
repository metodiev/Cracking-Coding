// Task 150 - Sum of root-to-leaf numbers in tree. (Tier: Beginner, CTCI Ch 4)
//
// Every root-to-leaf path spells a decimal number (1 -> 2 -> 3 means 123). The DFS
// carries the prefix built so far: acc * 10 + node.val, and adds the accumulated
// value whenever a leaf closes a path.
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

    /// <summary>Recursive DFS with the running prefix; an empty tree contributes 0.</summary>
    public static int SumNumbers(TreeNode? root) => Walk(root, 0);

    private static int Walk(TreeNode? node, int acc)
    {
        if (node == null) return 0;
        acc = acc * 10 + node.Val;
        if (node.Left == null && node.Right == null) return acc;   // leaf closes one number
        return Walk(node.Left, acc) + Walk(node.Right, acc);
    }

    /// <summary>Stack of (node, prefix) pairs; a leaf adds its prefix to the total.</summary>
    public static int SumNumbersIterative(TreeNode? root)
    {
        if (root == null) return 0;
        int total = 0;
        var stack = new Stack<(TreeNode Node, int Acc)>();
        stack.Push((root, root.Val));
        while (stack.Count > 0)
        {
            var (node, acc) = stack.Pop();
            if (node.Left == null && node.Right == null)
            {
                total += acc;
                continue;
            }
            if (node.Left != null) stack.Push((node.Left, acc * 10 + node.Left.Val));
            if (node.Right != null) stack.Push((node.Right, acc * 10 + node.Right.Val));
        }
        return total;
    }

    public static void Main()
    {
        Check(SumNumbers(null) == 0 && SumNumbersIterative(null) == 0, "empty tree");
        Check(SumNumbers(BuildTree(7)) == 7, "single node");
        Check(SumNumbers(BuildTree(1, 2, 3)) == 25, "12 + 13");
        Check(SumNumbers(BuildTree(4, 9, 0, 5, 1)) == 1026, "495 + 491 + 40");
        Check(SumNumbers(BuildTree(1, 0, 0)) == 20, "zeros inside the path");
        Check(SumNumbers(BuildTree(1, 2, null, 3)) == 123, "one path only");
        Check(SumNumbers(BuildTree(1, 2, 3, 4, 5)) == 262, "three paths");
        var cases = new[]
        {
            new int?[] { }, new int?[] { 7 }, new int?[] { 1, 2, 3 }, new int?[] { 4, 9, 0, 5, 1 },
            new int?[] { 1, 0, 0 }, new int?[] { 1, 2, null, 3 }, new int?[] { 1, 2, 3, 4, 5 },
            new int?[] { 1, 9, 9, null, 1 }
        };
        foreach (var data in cases)
        {
            var tree = BuildTree(data);
            Check(SumNumbers(tree) == SumNumbersIterative(tree), "recursive == iterative");
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
