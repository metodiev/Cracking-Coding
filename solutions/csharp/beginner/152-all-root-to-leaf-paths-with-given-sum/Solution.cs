// Task 152 - All root-to-leaf paths with given sum. (Tier: Beginner, CTCI Ch 8)
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

    /// <summary>All root-to-leaf paths summing to target, left subtree before right.</summary>
    public static List<List<int>> PathSumPaths(TreeNode? root, int target)
    {
        var result = new List<List<int>>();
        Dfs(root, target, new List<int>(), result);
        return result;
    }

    private static void Dfs(TreeNode? node, int remaining, List<int> path, List<List<int>> result)
    {
        if (node == null) return;
        path.Add(node.Val);
        if (node.Left == null && node.Right == null)
        {
            if (remaining == node.Val) result.Add(new List<int>(path));
        }
        else
        {
            Dfs(node.Left, remaining - node.Val, path, result);
            Dfs(node.Right, remaining - node.Val, path, result);
        }
        path.RemoveAt(path.Count - 1);
    }

    private static string Fmt(List<List<int>> paths) =>
        "[" + string.Join(", ", paths.ConvertAll(p => "[" + string.Join(", ", p) + "]")) + "]";

    public static void Main()
    {
        var root = BuildTree(5, 4, 8, 11, null, 13, 4, 7, 2, null, null, 5, 1);
        Check(Fmt(PathSumPaths(root, 22)) == "[[5, 4, 11, 2], [5, 8, 4, 5]]", "two paths");
        Check(Fmt(PathSumPaths(root, 26)) == "[[5, 8, 13]]", "single path");
        Check(Fmt(PathSumPaths(root, 27)) == "[[5, 4, 11, 7]]", "left-most path");
        Check(PathSumPaths(root, 100).Count == 0, "no matching path");
        Check(PathSumPaths(null, 0).Count == 0, "empty tree");
        Check(Fmt(PathSumPaths(BuildTree(7), 7)) == "[[7]]", "single node match");
        Check(PathSumPaths(BuildTree(7), 1).Count == 0, "single node miss");
        Check(PathSumPaths(BuildTree(1, 2), 1).Count == 0, "one missing child is not an end");
        Check(Fmt(PathSumPaths(BuildTree(1, 2), 3)) == "[[1, 2]]", "one-child path");
        Check(Fmt(PathSumPaths(BuildTree(-1, -2, -3), -3)) == "[[-1, -2]]", "negatives");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
