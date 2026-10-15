// Task 245 - Reconstruct binary tree from preorder with null markers. (Tier: Intermediate, CTCI Ch 4)
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

    private static int tokenPos;

    /// <summary>tokens like "1,2,#,#,3,#,#" where "#" marks a missing child.</summary>
    public static TreeNode? BuildFromPreorderTokens(string csv)
    {
        if (csv.Length == 0) return null;
        var tokens = csv.Split(',');
        tokenPos = 0;
        return Build(tokens);
    }

    private static TreeNode? Build(string[] tokens)
    {
        if (tokenPos >= tokens.Length || tokens[tokenPos] == "#")
        {
            tokenPos++;
            return null;
        }
        var node = new TreeNode(int.Parse(tokens[tokenPos++]));
        node.Left = Build(tokens);
        node.Right = Build(tokens);
        return node;
    }

    public static void Main()
    {
        var root = BuildFromPreorderTokens("1,2,#,#,3,#,#");
        Check(string.Join(",", Preorder(root)) == "1,2,3", "three nodes preorder");
        Check(string.Join(",", Inorder(root)) == "2,1,3", "three nodes inorder");

        var root2 = BuildFromPreorderTokens("1,2,4,#,#,5,#,#,3,#,6,#,#");
        Check(string.Join(",", Preorder(root2)) == "1,2,4,5,3,6", "six nodes preorder");
        Check(string.Join(",", Inorder(root2)) == "4,2,5,1,3,6", "six nodes inorder");

        Check(BuildFromPreorderTokens("#") == null, "all-null tree");
        var single = BuildFromPreorderTokens("9,#,#");
        Check(single!.Val == 9 && single.Left == null && single.Right == null, "single node");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}

