// Task 146 - Lowest common ancestor in BST (iterative). (Tier: Beginner, CTCI Ch 4)
//
// The BST ordering turns the search into a walk: while both keys sit on the same
// side of the current node, step that way. The first node where the keys split
// (or where one key IS the node) is the lowest common ancestor.
// O(h) steps, O(1) extra space.
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

    /// <summary>Locate a node by value using the BST ordering; null when it is absent.</summary>
    public static TreeNode? Find(TreeNode? root, int val)
    {
        while (root != null && root.Val != val)
        {
            root = val < root.Val ? root.Left : root.Right;
        }
        return root;
    }

    /// <summary>Iterative split-point walk; null when a key is missing from this BST.</summary>
    public static TreeNode? LowestCommonAncestor(TreeNode? root, TreeNode? p, TreeNode? q)
    {
        if (root == null || p == null || q == null) return null;
        var node = root;
        while (node != null)
        {
            if (p.Val < node.Val && q.Val < node.Val)
            {
                node = node.Left;                 // both keys live in the left subtree
            }
            else if (p.Val > node.Val && q.Val > node.Val)
            {
                node = node.Right;                // both keys live in the right subtree
            }
            else
            {
                return node;                      // split point, or one key is node itself
            }
        }
        return null;
    }

    public static void Main()
    {
        var tree = BuildTree(6, 2, 8, 0, 4, 7, 9, null, null, 3, 5);
        Check(LowestCommonAncestor(tree, Find(tree, 2), Find(tree, 8))!.Val == 6, "split at the root");
        Check(LowestCommonAncestor(tree, Find(tree, 2), Find(tree, 4))!.Val == 2, "ancestor == key");
        Check(LowestCommonAncestor(tree, Find(tree, 0), Find(tree, 5))!.Val == 2, "split on the left");
        Check(LowestCommonAncestor(tree, Find(tree, 4), Find(tree, 5))!.Val == 4, "both left, then right");
        Check(LowestCommonAncestor(tree, Find(tree, 7), Find(tree, 9))!.Val == 8, "both right of the root");
        Check(LowestCommonAncestor(tree, Find(tree, 2), Find(tree, 2))!.Val == 2, "node with itself");
        Check(LowestCommonAncestor(tree, Find(tree, 0), Find(tree, 0))!.Val == 0, "leaf with itself");
        Check(LowestCommonAncestor(tree, Find(tree, 0), Find(tree, 3))!.Val == 2, "both keys left");
        Check(LowestCommonAncestor(tree, Find(tree, 0), Find(tree, 99)) == null, "missing key");
        Check(LowestCommonAncestor(tree, null, Find(tree, 8)) == null, "null key");
        Check(LowestCommonAncestor(null, Find(tree, 2), Find(tree, 8)) == null, "no tree");
        // 2 and 4 sit in opposite subtrees of the root 3; 5 is 4's ancestor
        var chain = BuildTree(3, 1, 5, null, 2, 4);
        Check(LowestCommonAncestor(chain, Find(chain, 2), Find(chain, 4))!.Val == 3, "different subtrees");
        Check(LowestCommonAncestor(chain, Find(chain, 1), Find(chain, 2))!.Val == 1, "both below 1");
        Check(LowestCommonAncestor(chain, Find(chain, 4), Find(chain, 5))!.Val == 5, "ancestor == key");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
