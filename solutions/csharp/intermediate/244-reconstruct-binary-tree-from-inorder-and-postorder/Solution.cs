// Task 244 - Reconstruct binary tree from inorder and postorder. (Tier: Intermediate, CTCI Ch 4)
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

    private static int postPos;
    private static Dictionary<int, int> inIndex = new Dictionary<int, int>();

    public static TreeNode? BuildFromInPost(int[] inorder, int[] postorder)
    {
        if (postorder.Length == 0 || postorder.Length != inorder.Length)
            throw new ArgumentException("inorder and postorder must be non-empty and equal-length");
        postPos = postorder.Length - 1;
        inIndex = new Dictionary<int, int>();
        for (int i = 0; i < inorder.Length; i++) inIndex[inorder[i]] = i;
        return Build(postorder, 0, inorder.Length - 1);
    }

    private static TreeNode? Build(int[] postorder, int lo, int hi)
    {
        if (lo > hi) return null;
        int rootVal = postorder[postPos--];
        var node = new TreeNode(rootVal);
        int mid = inIndex[rootVal];
        node.Right = Build(postorder, mid + 1, hi);
        node.Left = Build(postorder, lo, mid - 1);
        return node;
    }

    public static void Main()
    {
        var root = BuildFromInPost(new[] { 9, 3, 15, 20, 7 }, new[] { 9, 15, 7, 20, 3 });
        Check(string.Join(",", Inorder(root)) == "9,3,15,20,7", "inorder round trip");
        Check(string.Join(",", Postorder(root)) == "9,15,7,20,3", "postorder round trip");
        Console.WriteLine("PASS");
    }

    private static List<int> Postorder(TreeNode? node)
    {
        var outList = new List<int>();
        Post(node, outList);
        return outList;
    }

    private static void Post(TreeNode? node, List<int> outList)
    {
        if (node == null) return;
        Post(node.Left, outList);
        Post(node.Right, outList);
        outList.Add(node.Val);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}

