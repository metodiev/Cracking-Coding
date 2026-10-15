// Task 136 - Tree preorder traversal recursive. (Tier: Beginner, CTCI Ch 4)
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

    public static List<int> Preorder(TreeNode? node)
    {
        var outList = new List<int>();
        Preorder(node, outList);
        return outList;
    }

    private static void Preorder(TreeNode? node, List<int> outList)
    {
        if (node == null) return;
        outList.Add(node.Val);
        Preorder(node.Left, outList);
        Preorder(node.Right, outList);
    }

    public static void Main()
    {
        var t = BuildTree(1, 2, 3, 4, 5, null, 6);
        Check(string.Join(",", Preorder(t)) == "1,2,4,5,3,6", "traversal order");
        Check(Preorder(null).Count == 0, "empty tree");
        Check(string.Join(",", Preorder(BuildTree(7))) == "7", "single node");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
