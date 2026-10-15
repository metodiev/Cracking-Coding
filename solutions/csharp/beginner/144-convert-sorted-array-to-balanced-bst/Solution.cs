// Task 144 - Convert sorted array to balanced BST. (Tier: Beginner, CTCI Ch 4)
//
// The middle element becomes the root, so both halves stay equally deep and the
// resulting tree has the minimal possible height for those values.
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

    /// <summary>Height-balanced BST from ascending nums; null for an empty array.</summary>
    public static TreeNode? SortedArrayToBst(int[] nums) => BuildBst(nums, 0, nums.Length - 1);

    private static TreeNode? BuildBst(int[] nums, int lo, int hi)
    {
        if (lo > hi) return null;
        int mid = (lo + hi) / 2;                 // left-biased middle keeps it stable
        var node = new TreeNode(nums[mid]);
        node.Left = BuildBst(nums, lo, mid - 1);
        node.Right = BuildBst(nums, mid + 1, hi);
        return node;
    }

    /// <summary>Left - node - right values; sorted output proves the BST ordering.</summary>
    public static List<int> Inorder(TreeNode? root)
    {
        var values = new List<int>();
        InorderInto(root, values);
        return values;
    }

    private static void InorderInto(TreeNode? node, List<int> values)
    {
        if (node == null) return;
        InorderInto(node.Left, values);
        values.Add(node.Val);
        InorderInto(node.Right, values);
    }

    public static int Height(TreeNode? node)
    {
        if (node == null) return 0;
        return 1 + Math.Max(Height(node.Left), Height(node.Right));
    }

    /// <summary>Check the requirement directly: every node's subtree heights differ by <= 1.</summary>
    public static bool IsHeightBalanced(TreeNode? root)
    {
        if (root == null) return true;
        return Math.Abs(Height(root.Left) - Height(root.Right)) <= 1
               && IsHeightBalanced(root.Left) && IsHeightBalanced(root.Right);
    }

    public static void Main()
    {
        Check(SortedArrayToBst(new int[] { }) == null, "empty array");
        Check(SortedArrayToBst(new int[] { 5 })!.Val == 5, "single element");
        var small = SortedArrayToBst(new int[] { 1, 2, 3 })!;
        Check(small.Val == 2 && small.Left!.Val == 1 && small.Right!.Val == 3, "1,2,3");
        var cases = new[]
        {
            new int[] { }, new int[] { 1 }, new int[] { 1, 2, 3 }, new int[] { 1, 2, 3, 4, 5, 6 },
            new int[] { -10, -3, 0, 5, 9 }, new int[] { 1, 2, 3, 4, 5, 6, 7 },
            new int[] { 1, 1, 1 }, new int[] { 0, 1000000 }
        };
        foreach (var nums in cases)
        {
            var tree = SortedArrayToBst(nums);
            Check(string.Join(",", Inorder(tree)) == string.Join(",", nums),
                  "inorder equals input of length " + nums.Length);
            Check(IsHeightBalanced(tree), "balanced for length " + nums.Length);
            if (nums.Length > 0)
            {
                Check(tree!.Val == nums[(nums.Length - 1) / 2], "middle element is the root");
            }
        }
        Check(Height(SortedArrayToBst(new int[] { 1, 2, 3, 4, 5, 6, 7 })) == 3, "height 3 for 7 values");
        Check(Height(SortedArrayToBst(new int[] { -10, -3, 0, 5, 9 })) == 3, "height 3 for 5 values");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
