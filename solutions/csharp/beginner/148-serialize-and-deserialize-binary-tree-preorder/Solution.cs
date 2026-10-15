// Task 148 - Serialize and deserialize binary tree (preorder). (Tier: Beginner, CTCI Ch 4)
//
// Format (identical in all four languages): preorder values separated by commas,
// with "#" marking every missing child, e.g. "1,2,#,#,3,#,#". An empty tree is "#"
// and a single node is "7,#,#", so every tree with n nodes uses exactly 2n + 1
// tokens and the stream can be rebuilt by a single left-to-right pass.
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

    /// <summary>Preorder walk that also emits a sentinel for each missing child.</summary>
    public static string Serialize(TreeNode? root)
    {
        var parts = new List<string>();
        SerializeInto(root, parts);
        return string.Join(",", parts);
    }

    private static void SerializeInto(TreeNode? node, List<string> parts)
    {
        if (node == null)
        {
            parts.Add("#");
            return;
        }
        parts.Add(node.Val.ToString());
        SerializeInto(node.Left, parts);
        SerializeInto(node.Right, parts);
    }

    /// <summary>Rebuild the tree from the preorder stream produced by Serialize.</summary>
    public static TreeNode? Deserialize(string data)
    {
        if (data.Length == 0) return null;
        int pos = 0;                                     // stream cursor shared with Build
        return Build(data.Split(','), ref pos);
    }

    private static TreeNode? Build(string[] tokens, ref int pos)
    {
        string token = tokens[pos++];
        if (token == "#") return null;
        var node = new TreeNode(int.Parse(token));
        node.Left = Build(tokens, ref pos);              // the left subtree comes next
        node.Right = Build(tokens, ref pos);
        return node;
    }

    /// <summary>Preorder values only (the sentinels are dropped).</summary>
    public static List<int> PreorderValues(TreeNode? root)
    {
        var values = new List<int>();
        PreorderInto(root, values);
        return values;
    }

    private static void PreorderInto(TreeNode? node, List<int> values)
    {
        if (node == null) return;
        values.Add(node.Val);
        PreorderInto(node.Left, values);
        PreorderInto(node.Right, values);
    }

    /// <summary>Round-trip check: same shape and same values everywhere.</summary>
    public static bool AreIdentical(TreeNode? a, TreeNode? b)
    {
        if (a == null && b == null) return true;
        if (a == null || b == null) return false;
        return a.Val == b.Val && AreIdentical(a.Left, b.Left) && AreIdentical(a.Right, b.Right);
    }

    public static void Main()
    {
        Check(Serialize(null) == "#", "empty tree");
        Check(Deserialize("#") == null && Deserialize("") == null, "deserialize empty");
        Check(Serialize(BuildTree(7)) == "7,#,#", "single node");
        Check(Serialize(BuildTree(1, 2, null, 3)) == "1,2,3,#,#,#,#", "left chain");
        Check(Serialize(BuildTree(1, 2, 5, 3, 4, null, 6)) == "1,2,3,#,#,4,#,#,5,#,6,#,#", "sample");
        Check(Serialize(BuildTree(1, 2, 3)).Split(',').Length == 7, "2n + 1 tokens");
        var cases = new[]
        {
            new int?[] { }, new int?[] { 7 }, new int?[] { 1, 2, null, 3 },
            new int?[] { 1, 2, 5, 3, 4, null, 6 }, new int?[] { 1, 2, 3, 4, 5, null, 6 }
        };
        foreach (var data in cases)
        {
            var tree = BuildTree(data);
            string text = Serialize(tree);
            var withoutSentinels = new List<string>();
            foreach (var token in text.Split(','))
            {
                if (token != "#") withoutSentinels.Add(token);
            }
            Check(string.Join(",", PreorderValues(tree)) == string.Join(",", withoutSentinels),
                  "preorder matches the stream");
            var rebuilt = Deserialize(text);
            Check(AreIdentical(tree, rebuilt), "round trip keeps shape and values");
            Check(Serialize(rebuilt) == text, "string round trip");
            if (tree != null)
            {
                Check(!ReferenceEquals(rebuilt, tree), "deserialize builds a fresh structure");
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
