// Task 243 - Reconstruct binary tree from inorder and preorder. (Tier: Intermediate, CTCI Ch 4)
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

    private static int prePos;
    private static Dictionary<int, int> inIndex = new Dictionary<int, int>();

    public static TreeNode? BuildFromPreIn(int[] preorder, int[] inorder)
    {
        if (preorder.Length == 0 || preorder.Length != inorder.Length)
            throw new ArgumentException("preorder and inorder must be non-empty and equal-length");
        prePos = 0;
        inIndex = new Dictionary<int, int>();
        for (int i = 0; i < inorder.Length; i++) inIndex[inorder[i]] = i;
        return Build(preorder, 0, inorder.Length - 1);
    }

    private static TreeNode? Build(int[] preorder, int lo, int hi)
    {
        if (lo > hi) return null;
        int rootVal = preorder[prePos++];
        var node = new TreeNode(rootVal);
        int mid = inIndex[rootVal];
        node.Left = Build(preorder, lo, mid - 1);
        node.Right = Build(preorder, mid + 1, hi);
        return node;
    }

    public static void Main()
    {
        var root = BuildFromPreIn(new[] { 3, 9, 20, 15, 7 }, new[] { 9, 3, 15, 20, 7 });
        Check(string.Join(",", Preorder(root)) == "3,9,20,15,7", "preorder round trip");
        Check(string.Join(",", Inorder(root)) == "9,3,15,20,7", "inorder round trip");
        var spine = BuildFromPreIn(new[] { 1, 2, 3 }, new[] { 3, 2, 1 });
        Check(string.Join(",", Preorder(spine)) == "1,2,3" && string.Join(",", Inorder(spine)) == "3,2,1", "left spine");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}

