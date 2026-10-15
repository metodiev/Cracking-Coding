// Task 248 - Implement red-black tree insert (basic). (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;

public class RBNode
{
    public int Key;
    public bool Red = true;
    public RBNode? Left, Right, Parent;
    public RBNode(int key) { Key = key; }
}

public class RedBlackTree
{
    public RBNode? Root;

    private void RotateLeft(RBNode x)
    {
        var y = x.Right!;
        x.Right = y.Left;
        if (y.Left != null) y.Left.Parent = x;
        y.Parent = x.Parent;
        if (x.Parent == null) Root = y;
        else if (x == x.Parent.Left) x.Parent.Left = y;
        else x.Parent.Right = y;
        y.Left = x;
        x.Parent = y;
    }

    private void RotateRight(RBNode x)
    {
        var y = x.Left!;
        x.Left = y.Right;
        if (y.Right != null) y.Right.Parent = x;
        y.Parent = x.Parent;
        if (x.Parent == null) Root = y;
        else if (x == x.Parent.Right) x.Parent.Right = y;
        else x.Parent.Left = y;
        y.Right = x;
        x.Parent = y;
    }

    public bool Insert(int key)
    {
        RBNode? parent = null;
        var node = Root;
        while (node != null)
        {
            parent = node;
            if (key < node.Key) node = node.Left;
            else if (key > node.Key) node = node.Right;
            else return false;
        }
        var z = new RBNode(key) { Parent = parent };
        if (parent == null) Root = z;
        else if (key < parent.Key) parent.Left = z;
        else parent.Right = z;
        Fixup(z);
        return true;
    }

    private void Fixup(RBNode z)
    {
        while (z.Parent != null && z.Parent.Red)
        {
            var parent = z.Parent;
            var grand = parent.Parent!;
            if (parent == grand.Left)
            {
                var uncle = grand.Right;
                if (uncle != null && uncle.Red)
                {
                    parent.Red = false;
                    uncle.Red = false;
                    grand.Red = true;
                    z = grand;
                }
                else
                {
                    if (z == parent.Right)
                    {
                        z = parent;
                        RotateLeft(z);
                        parent = z.Parent!;
                        grand = parent.Parent!;
                    }
                    parent.Red = false;
                    grand.Red = true;
                    RotateRight(grand);
                }
            }
            else
            {
                var uncle = grand.Left;
                if (uncle != null && uncle.Red)
                {
                    parent.Red = false;
                    uncle.Red = false;
                    grand.Red = true;
                    z = grand;
                }
                else
                {
                    if (z == parent.Left)
                    {
                        z = parent;
                        RotateRight(z);
                        parent = z.Parent!;
                        grand = parent.Parent!;
                    }
                    parent.Red = false;
                    grand.Red = true;
                    RotateLeft(grand);
                }
            }
        }
        Root!.Red = false;
    }
}

public static class Solution
{
    public static List<int> Inorder(RBNode? node)
    {
        var outList = new List<int>();
        In(node, outList);
        return outList;
    }

    private static void In(RBNode? node, List<int> outList)
    {
        if (node == null) return;
        In(node.Left, outList);
        outList.Add(node.Key);
        In(node.Right, outList);
    }

    public static bool NoRedRed(RBNode? node)
    {
        if (node == null) return true;
        if (node.Red)
        {
            if (node.Left != null && node.Left.Red) return false;
            if (node.Right != null && node.Right.Red) return false;
        }
        return NoRedRed(node.Left) && NoRedRed(node.Right);
    }

    public static HashSet<int> BlackHeights(RBNode? node, int measure)
    {
        var result = new HashSet<int>();
        if (node == null) { result.Add(measure); return result; }
        int extra = node.Red ? 0 : 1;
        foreach (int value in BlackHeights(node.Left, measure)) result.Add(value + extra);
        foreach (int value in BlackHeights(node.Right, measure)) result.Add(value + extra);
        return result;
    }

    public static void Main()
    {
        var tree = new RedBlackTree();
        foreach (int key in new[] { 10, 20, 30 }) Check(tree.Insert(key), "insert " + key);
        Check(tree.Root!.Key == 20 && !tree.Root.Red, "root 20 black");
        Check(tree.Root.Left!.Key == 10 && tree.Root.Left.Red, "left 10 red");
        Check(tree.Root.Right!.Key == 30 && tree.Root.Right.Red, "right 30 red");

        int[][] orders = {
            new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
            new[] { 7, 3, 9, 1, 5, 8, 10, 2, 4, 6 },
            new[] { 50, 20, 60, 10, 30, 55, 70, 5, 15, 25, 35 },
        };
        foreach (var order in orders)
        {
            tree = new RedBlackTree();
            foreach (int key in order) Check(tree.Insert(key), "insert " + key);
            var sorted = (int[])order.Clone();
            Array.Sort(sorted);
            Check(string.Join(",", Inorder(tree.Root)) == string.Join(",", sorted), "BST order");
            Check(!tree.Root!.Red, "root is black");
            Check(NoRedRed(tree.Root), "no red-red violation");
            Check(BlackHeights(tree.Root, 0).Count == 1, "uniform black height");
        }
        Check(!tree.Insert(50), "duplicate rejected");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
