// Task 241 - Maximum path sum in binary tree (any path). (Tier: Intermediate, CTCI Ch 4)
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
            if (i < values.Length && values[i] != null) { node.Left = new TreeNode(values[i]!.Value); q.Enqueue(node.Left); }
            i++;
            if (i < values.Length && values[i] != null) { node.Right = new TreeNode(values[i]!.Value); q.Enqueue(node.Right); }
            i++;
        }
        return root;
    }

    public static List<int> Preorder(TreeNode? node)
    {
        var outList = new List<int>();
        Pre(node, outList);
        return outList;
    }

    private static void Pre(TreeNode? node, List<int> outList)
    {
        if (node == null) return;
        outList.Add(node.Val);
        Pre(node.Left, outList);
        Pre(node.Right, outList);
    }

    public static List<int> Inorder(TreeNode? node)
    {
        var outList = new List<int>();
        In(node, outList);
        return outList;
    }

    private static void In(TreeNode? node, List<int> outList)
    {
        if (node == null) return;
        In(node.Left, outList);
        outList.Add(node.Val);
        In(node.Right, outList);
    }

    private static long best;

    public static long MaxPathSum(TreeNode? root)
    {
        if (root == null) throw new ArgumentException("empty tree has no path");
        best = long.MinValue;
        Gain(root);
        return best;
    }

    private static long Gain(TreeNode? node)
    {
        if (node == null) return 0;
        long left = Math.Max(0, Gain(node.Left));
        long right = Math.Max(0, Gain(node.Right));
        best = Math.Max(best, node.Val + left + right);
        return node.Val + Math.Max(left, right);
    }

    public static void Main()
    {
        Check(MaxPathSum(BuildTree(1, 2, 3)) == 6, "small tree");
        Check(MaxPathSum(BuildTree(-10, 9, 20, null, null, 15, 7)) == 42, "classic");
        Check(MaxPathSum(BuildTree(-3)) == -3, "single negative");
        Check(MaxPathSum(BuildTree(2, -1)) == 2, "skip negative child");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}

