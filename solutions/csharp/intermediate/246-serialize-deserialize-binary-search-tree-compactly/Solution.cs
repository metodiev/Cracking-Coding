// Task 246 - Serialize/deserialize binary search tree compactly. (Tier: Intermediate, CTCI Ch 4)
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

    public static string Serialize(TreeNode? root) => string.Join(",", Preorder(root));

    /// <summary>Naive form kept for comparison: explicit '#' markers for missing children.</summary>
    public static string SerializeWithNulls(TreeNode? node, List<string> tokens)
    {
        if (node == null) { tokens.Add("#"); return string.Join(",", tokens); }
        tokens.Add(node.Val.ToString());
        SerializeWithNulls(node.Left, tokens);
        SerializeWithNulls(node.Right, tokens);
        return string.Join(",", tokens);
    }

    private static int[] values = Array.Empty<int>();
    private static int pos;

    public static TreeNode? Deserialize(string data)
    {
        if (data.Length == 0) return null;
        var parts = data.Split(',');
        values = new int[parts.Length];
        for (int i = 0; i < parts.Length; i++) values[i] = int.Parse(parts[i]);
        pos = 0;
        return Build(long.MinValue, long.MaxValue);
    }

    private static TreeNode? Build(long lo, long hi)
    {
        if (pos >= values.Length) return null;
        long value = values[pos];
        if (value < lo || value > hi) return null;
        pos++;
        var node = new TreeNode((int)value);
        node.Left = Build(lo, value - 1);
        node.Right = Build(value + 1, hi);
        return node;
    }

    public static TreeNode? BuildBst(int[] input)
    {
        TreeNode? root = null;
        foreach (int value in input)
        {
            if (root == null) { root = new TreeNode(value); continue; }
            var node = root;
            while (true)
            {
                if (value < node.Val)
                {
                    if (node.Left == null) { node.Left = new TreeNode(value); break; }
                    node = node.Left;
                }
                else
                {
                    if (node.Right == null) { node.Right = new TreeNode(value); break; }
                    node = node.Right;
                }
            }
        }
        return root;
    }

    public static void Main()
    {
        int[] vals = { 5, 3, 8, 2, 4, 7, 9 };
        var root = BuildBst(vals);
        string data = Serialize(root);
        Check(data == "5,3,2,4,8,7,9", "preorder serialization");
        Check(SerializeWithNulls(root, new List<string>()) == "5,3,2,#,#,4,#,#,8,7,#,#,9,#,#", "null-marker form");

        var clone = Deserialize(data);
        Check(string.Join(",", Preorder(clone)) == "5,3,2,4,8,7,9", "clone preorder");
        Check(string.Join(",", Inorder(clone)) == "2,3,4,5,7,8,9", "clone inorder sorted");
        Check(data.Split(',').Length < SerializeWithNulls(root, new List<string>()).Split(',').Length, "compact form is shorter");
        Check(Deserialize("") == null, "empty string");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
