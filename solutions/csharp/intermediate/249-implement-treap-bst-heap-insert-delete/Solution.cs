// Task 249 - Implement treap (BST + heap) insert/delete. (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;

public class TreapNode
{
    public int Key, Pri;
    public TreapNode? Left, Right;
    public TreapNode(int key, int pri) { Key = key; Pri = pri; }
}

public static class Solution
{
    private static TreapNode RotateRight(TreapNode y)
    {
        var x = y.Left!;
        y.Left = x.Right;
        x.Right = y;
        return x;
    }

    private static TreapNode RotateLeft(TreapNode x)
    {
        var y = x.Right!;
        x.Right = y.Left;
        y.Left = x;
        return y;
    }

    public static TreapNode? Insert(TreapNode? node, int key, int pri, ref bool inserted)
    {
        if (node == null) { inserted = true; return new TreapNode(key, pri); }
        if (key == node.Key) { inserted = false; return node; }
        if (key < node.Key)
        {
            node.Left = Insert(node.Left, key, pri, ref inserted);
            if (node.Left!.Pri > node.Pri) node = RotateRight(node);
        }
        else
        {
            node.Right = Insert(node.Right, key, pri, ref inserted);
            if (node.Right!.Pri > node.Pri) node = RotateLeft(node);
        }
        return node;
    }

    public static TreapNode? Erase(TreapNode? node, int key, ref bool removed)
    {
        if (node == null) { removed = false; return null; }
        if (key < node.Key)
        {
            node.Left = Erase(node.Left, key, ref removed);
        }
        else if (key > node.Key)
        {
            node.Right = Erase(node.Right, key, ref removed);
        }
        else
        {
            removed = true;
            if (node.Left == null) return node.Right;
            if (node.Right == null) return node.Left;
            if (node.Left.Pri > node.Right.Pri)
            {
                node = RotateRight(node);
                node.Right = Erase(node.Right, key, ref removed);
            }
            else
            {
                node = RotateLeft(node);
                node.Left = Erase(node.Left, key, ref removed);
            }
        }
        return node;
    }

    public static TreapNode? Build(int[][] entries)
    {
        TreapNode? root = null;
        foreach (var entry in entries)
        {
            bool inserted = false;
            root = Insert(root, entry[0], entry[1], ref inserted);
        }
        return root;
    }

    public static List<int> Inorder(TreapNode? node)
    {
        var outList = new List<int>();
        In(node, outList);
        return outList;
    }

    private static void In(TreapNode? node, List<int> outList)
    {
        if (node == null) return;
        In(node.Left, outList);
        outList.Add(node.Key);
        In(node.Right, outList);
    }

    public static List<int> Preorder(TreapNode? node)
    {
        var outList = new List<int>();
        Pre(node, outList);
        return outList;
    }

    private static void Pre(TreapNode? node, List<int> outList)
    {
        if (node == null) return;
        outList.Add(node.Key);
        Pre(node.Left, outList);
        Pre(node.Right, outList);
    }

    public static bool IsHeap(TreapNode? node)
    {
        if (node == null) return true;
        if (node.Left != null && node.Left.Pri > node.Pri) return false;
        if (node.Right != null && node.Right.Pri > node.Pri) return false;
        return IsHeap(node.Left) && IsHeap(node.Right);
    }

    public static bool Contains(TreapNode? node, int key)
    {
        while (node != null)
        {
            if (key == node.Key) return true;
            node = key < node.Key ? node.Left : node.Right;
        }
        return false;
    }

    public static void Main()
    {
        var entries = new[]
        {
            new[] { 50, 40 }, new[] { 30, 25 }, new[] { 70, 60 }, new[] { 20, 15 },
            new[] { 40, 35 }, new[] { 60, 50 }, new[] { 80, 70 },
        };
        var root = Build(entries);
        Check(root!.Key == 80 && root.Pri == 70, "highest priority at root");
        Check(string.Join(",", Preorder(root)) == "80,70,60,50,40,30,20", "preorder shape");
        Check(string.Join(",", Inorder(root)) == "20,30,40,50,60,70,80", "inorder sorted");
        Check(IsHeap(root), "max-heap on priorities");

        bool inserted = false;
        Insert(Build(new[] { new[] { 1, 5 } }), 1, 9, ref inserted);
        Check(!inserted, "duplicate rejected");

        bool removed = false;
        root = Erase(root, 30, ref removed);
        Check(removed && string.Join(",", Inorder(root)) == "20,40,50,60,70,80" && IsHeap(root), "erase leaf");
        root = Erase(root, 80, ref removed);
        Check(removed && root!.Key == 70 && IsHeap(root), "erase root");
        root = Erase(root, 999, ref removed);
        Check(!removed && string.Join(",", Inorder(root)) == "20,40,50,60,70", "erase missing");
        foreach (int key in new[] { 20, 40, 50, 60, 70 }) Check(Contains(root, key), "contains " + key);
        Check(!Contains(root, 30), "30 gone");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
