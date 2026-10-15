// Task 247 - Implement AVL tree insert (rotations + balance). (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;

public class AVLNode
{
    public int Key;
    public int Height = 1;
    public AVLNode? Left, Right;
    public AVLNode(int key) { Key = key; }
}

public static class Solution
{
    private static int Height(AVLNode? node) => node == null ? 0 : node.Height;

    private static void Update(AVLNode node) => node.Height = 1 + Math.Max(Height(node.Left), Height(node.Right));

    private static int BalanceFactor(AVLNode node) => Height(node.Left) - Height(node.Right);

    private static AVLNode RotateRight(AVLNode y)
    {
        var x = y.Left!;
        y.Left = x.Right;
        x.Right = y;
        Update(y);
        Update(x);
        return x;
    }

    private static AVLNode RotateLeft(AVLNode x)
    {
        var y = x.Right!;
        x.Right = y.Left;
        y.Left = x;
        Update(x);
        Update(y);
        return y;
    }

    public static AVLNode? Insert(AVLNode? node, int key)
    {
        if (node == null) return new AVLNode(key);
        if (key < node.Key) node.Left = Insert(node.Left, key);
        else if (key > node.Key) node.Right = Insert(node.Right, key);
        else return node;

        Update(node);
        int bf = BalanceFactor(node);
        if (bf > 1 && key < node.Left!.Key) return RotateRight(node);
        if (bf < -1 && key > node.Right!.Key) return RotateLeft(node);
        if (bf > 1 && key > node.Left!.Key) { node.Left = RotateLeft(node.Left); return RotateRight(node); }
        if (bf < -1 && key < node.Right!.Key) { node.Right = RotateRight(node.Right); return RotateLeft(node); }
        return node;
    }

    public static AVLNode? Build(params int[] keys)
    {
        AVLNode? root = null;
        foreach (int key in keys) root = Insert(root, key);
        return root;
    }

    public static List<int> Inorder(AVLNode? node)
    {
        var outList = new List<int>();
        In(node, outList);
        return outList;
    }

    private static void In(AVLNode? node, List<int> outList)
    {
        if (node == null) return;
        In(node.Left, outList);
        outList.Add(node.Key);
        In(node.Right, outList);
    }

    public static bool IsBalanced(AVLNode? node)
    {
        if (node == null) return true;
        if (Math.Abs(BalanceFactor(node)) > 1) return false;
        return IsBalanced(node.Left) && IsBalanced(node.Right);
    }

    public static int Count(AVLNode? node) => node == null ? 0 : 1 + Count(node.Left) + Count(node.Right);

    public static void Main()
    {
        var root = Build(10, 20, 30);
        Check(root!.Key == 20 && root.Left!.Key == 10 && root.Right!.Key == 30, "LL rotation");
        root = Build(30, 20, 10);
        Check(root!.Key == 20 && root.Left!.Key == 10 && root.Right!.Key == 30, "RR rotation");
        root = Build(30, 10, 20);
        Check(root!.Key == 20 && root.Left!.Key == 10 && root.Right!.Key == 30, "LR rotation");
        root = Build(10, 30, 20);
        Check(root!.Key == 20 && root.Left!.Key == 10 && root.Right!.Key == 30, "RL rotation");

        var asc = Build(1, 2, 3, 4, 5, 6, 7);
        Check(string.Join(",", Inorder(asc)) == "1,2,3,4,5,6,7", "inorder sorted");
        Check(IsBalanced(asc) && asc!.Height == 3, "height 3 for 7 ascending keys");

        var ten = Build(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        Check(Count(ten) == 10 && IsBalanced(ten) && ten!.Height == 4, "height 4 for 10 ascending keys");
        Check(Count(Build(5, 5, 5)) == 1, "duplicates ignored");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
