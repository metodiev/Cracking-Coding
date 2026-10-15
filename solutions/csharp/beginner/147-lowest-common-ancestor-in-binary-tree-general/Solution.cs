// Task 147 - Lowest common ancestor in binary tree (general). (Tier: Beginner, CTCI Ch 4)
//
// No BST ordering is available, so a postorder search is used: a subtree reports
// the first key it finds from below; the node where both subtrees report a key is
// the split point and therefore the LCA. When either key is not part of the tree
// at all, the answer is null.
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

    /// <summary>Plain depth-first search: a general tree has no ordering to exploit.</summary>
    public static TreeNode? Find(TreeNode? root, int val)
    {
        if (root == null || root.Val == val) return root;
        var hit = Find(root.Left, val);
        return hit ?? Find(root.Right, val);
    }

    /// <summary>Identity search: is that very node object part of this tree?</summary>
    public static bool Contains(TreeNode? root, TreeNode? target)
    {
        if (root == null) return false;
        return ReferenceEquals(root, target) || Contains(root.Left, target) || Contains(root.Right, target);
    }

    /// <summary>Postorder LCA; null when either node is missing from the tree.</summary>
    public static TreeNode? LowestCommonAncestor(TreeNode? root, TreeNode? p, TreeNode? q)
    {
        if (root == null || p == null || q == null) return null;
        if (!Contains(root, p) || !Contains(root, q)) return null;
        return Search(root, p, q);
    }

    /// <summary>Classic postorder: report a target from below, else report the split node.</summary>
    private static TreeNode? Search(TreeNode? node, TreeNode p, TreeNode q)
    {
        if (node == null || ReferenceEquals(node, p) || ReferenceEquals(node, q)) return node;
        var left = Search(node.Left, p, q);
        var right = Search(node.Right, p, q);
        if (left != null && right != null) return node;   // p and q in different subtrees
        return left ?? right;
    }

    public static void Main()
    {
        var tree = BuildTree(3, 5, 1, 6, 2, 0, 8, null, null, 7, 4);
        Check(LowestCommonAncestor(tree, Find(tree, 5), Find(tree, 1))!.Val == 3, "root splits them");
        Check(LowestCommonAncestor(tree, Find(tree, 5), Find(tree, 4))!.Val == 5, "one key is an ancestor");
        Check(LowestCommonAncestor(tree, Find(tree, 6), Find(tree, 7))!.Val == 5, "both in the left branch");
        Check(LowestCommonAncestor(tree, Find(tree, 6), Find(tree, 4))!.Val == 5, "deeper split");
        Check(LowestCommonAncestor(tree, Find(tree, 7), Find(tree, 4))!.Val == 2, "siblings under 2");
        Check(LowestCommonAncestor(tree, Find(tree, 0), Find(tree, 4))!.Val == 3, "different subtrees");
        Check(LowestCommonAncestor(tree, Find(tree, 3), Find(tree, 5))!.Val == 3, "one key is the root");
        Check(LowestCommonAncestor(tree, Find(tree, 7), Find(tree, 7))!.Val == 7, "node with itself");
        Check(LowestCommonAncestor(tree, Find(tree, 0), Find(tree, 99)) == null, "missing key");
        Check(LowestCommonAncestor(tree, null, Find(tree, 5)) == null, "null key");
        Check(LowestCommonAncestor(null, Find(tree, 5), Find(tree, 1)) == null, "no tree");
        // same value, but a different node object is not part of this tree
        Check(LowestCommonAncestor(tree, Find(tree, 3), new TreeNode(5)) == null, "detached node");
        var chain = BuildTree(1, null, 2, null, 3);
        Check(LowestCommonAncestor(chain, Find(chain, 1), Find(chain, 3))!.Val == 1, "right chain");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
