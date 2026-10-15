// Task 239 - Connect next pointers in each node (perfect binary tree). (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;

public class Node
{
    public int Val;
    public Node? Left, Right, Next;
    public Node(int v) { Val = v; }
}

public static class Solution
{
    /// <summary>Build a binary tree from level-order values (null marks a missing child).</summary>
    public static Node? BuildTree(params int?[] values)
    {
        if (values.Length == 0 || values[0] == null) return null;
        var root = new Node(values[0]!.Value);
        var q = new Queue<Node>();
        q.Enqueue(root);
        int i = 1;
        while (q.Count > 0 && i < values.Length)
        {
            var node = q.Dequeue();
            if (i < values.Length && values[i] != null)
            {
                node.Left = new Node(values[i]!.Value);
                q.Enqueue(node.Left);
            }
            i++;
            if (i < values.Length && values[i] != null)
            {
                node.Right = new Node(values[i]!.Value);
                q.Enqueue(node.Right);
            }
            i++;
        }
        return root;
    }

    /// <summary>
    /// Every node gets Next -> its right neighbour on the same level (null at level
    /// ends). O(1) extra space: descend level by level through the already-set Next
    /// pointers of the parent level; for a parent p, p.Left.Next = p.Right and, when
    /// p.Next exists, p.Right.Next = p.Next.Left. Returns the root.
    /// </summary>
    public static Node? Connect(Node? root)
    {
        if (root == null) return null;
        for (Node? leftmost = root; leftmost.Left != null; leftmost = leftmost.Left)
        {
            for (Node? node = leftmost; node != null; node = node.Next)
            {
                node.Left!.Next = node.Right;
                if (node.Next != null) node.Right!.Next = node.Next.Left;
            }
        }
        return root;
    }

    /// <summary>Read the levels through the Next pointers (fails fast if a chain loops).</summary>
    public static List<List<int>> LevelsViaNext(Node? root)
    {
        var levels = new List<List<int>>();
        for (Node? leftmost = root; leftmost != null; leftmost = leftmost.Left)
        {
            var level = new List<int>();
            for (Node? node = leftmost; node != null; node = node.Next)
            {
                level.Add(node.Val);
                if (level.Count > 1000) throw new Exception("Next chain longer than expected");
            }
            levels.Add(level);
        }
        return levels;
    }

    public static void Main()
    {
        var root = Connect(BuildTree(1, 2, 3, 4, 5, 6, 7))!;
        CheckGroups(LevelsViaNext(root),
            new[] { new[] { 1 }, new[] { 2, 3 }, new[] { 4, 5, 6, 7 } }, "classic levels");
        Check(root.Next == null, "root boundary");
        Check(root.Left!.Next == root.Right, "2 -> 3");
        Check(root.Right!.Next == null, "3 boundary");
        Check(root.Left.Left!.Next == root.Left.Right, "4 -> 5");
        Check(root.Left.Right!.Next == root.Right.Left, "5 -> 6");
        Check(root.Right.Left!.Next == root.Right.Right, "6 -> 7");
        Check(root.Right.Right!.Next == null, "7 boundary");

        var single = Connect(BuildTree(1))!;
        CheckGroups(LevelsViaNext(single), new[] { new[] { 1 } }, "single level");
        Check(single.Next == null, "single boundary");
        Check(Connect(null) == null, "empty");

        var big = Connect(BuildTree(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))!;
        CheckGroups(LevelsViaNext(big),
            new[]
            {
                new[] { 1 },
                new[] { 2, 3 },
                new[] { 4, 5, 6, 7 },
                new[] { 8, 9, 10, 11, 12, 13, 14, 15 },
            }, "four levels");

        var again = Connect(root);                  // connecting twice changes nothing
        Check(again == root, "idempotent identity");
        CheckGroups(LevelsViaNext(again),
            new[] { new[] { 1 }, new[] { 2, 3 }, new[] { 4, 5, 6, 7 } }, "idempotent levels");
        Console.WriteLine("PASS");
    }

    private static void CheckGroups(List<List<int>> actual, int[][] expected, string label)
    {
        bool ok = actual.Count == expected.Length;
        for (int i = 0; ok && i < expected.Length; i++)
        {
            ok = actual[i].Count == expected[i].Length;
            for (int j = 0; ok && j < expected[i].Length; j++)
                ok = actual[i][j] == expected[i][j];
        }
        Check(ok, label);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
