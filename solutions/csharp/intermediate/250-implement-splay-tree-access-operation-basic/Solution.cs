// Task 250 - Implement splay tree access operation (basic). (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;

public class SplayNode
{
    public int Key;
    public SplayNode? Left, Right, Parent;
    public SplayNode(int key) { Key = key; }
}

public class SplayTree
{
    public SplayNode? Root;

    private void ReplaceChild(SplayNode? parent, SplayNode old, SplayNode? fresh)
    {
        if (parent == null) Root = fresh;
        else if (parent.Left == old) parent.Left = fresh;
        else parent.Right = fresh;
        if (fresh != null) fresh.Parent = parent;
    }

    private void RotateLeft(SplayNode x)
    {
        var y = x.Right!;
        x.Right = y.Left;
        if (y.Left != null) y.Left.Parent = x;
        var parent = x.Parent;
        ReplaceChild(parent, x, y);
        y.Left = x;
        x.Parent = y;
    }

    private void RotateRight(SplayNode x)
    {
        var y = x.Left!;
        x.Left = y.Right;
        if (y.Right != null) y.Right.Parent = x;
        var parent = x.Parent;
        ReplaceChild(parent, x, y);
        y.Right = x;
        x.Parent = y;
    }

    public void Splay(SplayNode x)
    {
        while (x.Parent != null)
        {
            var parent = x.Parent;
            var grand = parent.Parent;
            if (grand == null)
            {
                if (x == parent.Left) RotateRight(parent); else RotateLeft(parent);
            }
            else if (x == parent.Left && parent == grand.Left)
            {
                RotateRight(grand);
                RotateRight(parent);
            }
            else if (x == parent.Right && parent == grand.Right)
            {
                RotateLeft(grand);
                RotateLeft(parent);
            }
            else if (x == parent.Right && parent == grand.Left)
            {
                RotateLeft(parent);
                RotateRight(grand);
            }
            else
            {
                RotateRight(parent);
                RotateLeft(grand);
            }
        }
        Root = x;
    }

    public bool Insert(int key)
    {
        SplayNode? parent = null;
        var node = Root;
        while (node != null)
        {
            parent = node;
            if (key < node.Key) node = node.Left;
            else if (key > node.Key) node = node.Right;
            else { Splay(node); return false; }
        }
        var fresh = new SplayNode(key) { Parent = parent };
        if (parent == null) Root = fresh;
        else if (key < parent.Key) parent.Left = fresh;
        else parent.Right = fresh;
        Splay(fresh);
        return true;
    }

    /// <summary>Splays key to the root if it exists; returns whether it was found.</summary>
    public bool Access(int key)
    {
        var node = Root;
        while (node != null)
        {
            if (key == node.Key) { Splay(node); return true; }
            node = key < node.Key ? node.Left : node.Right;
        }
        return false;
    }

    public List<int> Inorder()
    {
        var outList = new List<int>();
        Walk(Root, outList);
        return outList;
    }

    private static void Walk(SplayNode? node, List<int> outList)
    {
        if (node == null) return;
        Walk(node.Left, outList);
        outList.Add(node.Key);
        Walk(node.Right, outList);
    }

    public bool IsBst()
    {
        var keys = Inorder();
        for (int i = 1; i < keys.Count; i++) if (keys[i - 1] >= keys[i]) return false;
        return ParentsConsistent(Root);
    }

    private static bool ParentsConsistent(SplayNode? node)
    {
        if (node == null) return true;
        if (node.Left != null && node.Left.Parent != node) return false;
        if (node.Right != null && node.Right.Parent != node) return false;
        return ParentsConsistent(node.Left) && ParentsConsistent(node.Right);
    }
}

public static class Solution
{
    public static void Main()
    {
        var tree = new SplayTree();
        foreach (int key in new[] { 50, 30, 70, 20, 40, 60, 80 }) Check(tree.Insert(key), "insert " + key);
        Check(tree.Root!.Key == 80, "last insert at root");
        Check(string.Join(",", tree.Inorder()) == "20,30,40,50,60,70,80", "sorted");
        Check(tree.IsBst(), "bst + parent links");

        Check(tree.Access(20) && tree.Root!.Key == 20, "zig-zig to root");
        Check(tree.IsBst(), "still bst after splay 20");

        Check(tree.Access(80) && tree.Root!.Key == 80, "mirror zig-zig");
        Check(tree.IsBst(), "still bst after splay 80");

        Check(tree.Access(50) && tree.Root!.Key == 50, "zig-zag to root");
        Check(string.Join(",", tree.Inorder()) == "20,30,40,50,60,70,80", "sorted after splays");
        Check(tree.IsBst(), "bst + parent links after splays");

        Check(!tree.Access(999) && tree.Root!.Key == 50, "miss leaves root alone");
        Check(!tree.Insert(50) && tree.Root!.Key == 50, "duplicate splayed, not inserted");
        Check(string.Join(",", tree.Inorder()) == "20,30,40,50,60,70,80", "no duplicate stored");

        var single = new SplayTree();
        Check(single.Insert(7), "single insert");
        Check(single.Access(7) && single.Root!.Key == 7 && single.Root.Parent == null, "single access");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
