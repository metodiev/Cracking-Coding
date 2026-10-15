// Task 139 - Level order traversal (BFS) of tree. (Tier: Beginner, CTCI Ch 4)
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

    public static List<List<int>> LevelOrder(TreeNode? root)
    {
        var levels = new List<List<int>>();
        if (root == null) return levels;
        var q = new Queue<TreeNode>();
        q.Enqueue(root);
        while (q.Count > 0)
        {
            int size = q.Count;
            var level = new List<int>();
            for (int i = 0; i < size; i++)
            {
                var node = q.Dequeue();
                level.Add(node.Val);
                if (node.Left != null) q.Enqueue(node.Left);
                if (node.Right != null) q.Enqueue(node.Right);
            }
            levels.Add(level);
        }
        return levels;
    }

    public static void Main()
    {
        var levels = LevelOrder(BuildTree(1, 2, 3, 4, 5, null, 6));
        Check(levels.Count == 3, "three levels");
        Check(string.Join(",", levels[0]) == "1" && string.Join(",", levels[2]) == "4,5,6", "level contents");
        Check(LevelOrder(null).Count == 0, "empty");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
