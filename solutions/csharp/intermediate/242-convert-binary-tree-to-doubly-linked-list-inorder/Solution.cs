// Task 242 - Convert binary tree to doubly linked list (inorder). (Tier: Intermediate, CTCI Ch 4)
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
            if (i < values.Length && values[i] != null) { node.Left = new TreeNode(values[i]!.Value); q.Enqueue(node.Left); }
            i++;
            if (i < values.Length && values[i] != null) { node.Right = new TreeNode(values[i]!.Value); q.Enqueue(node.Right); }
            i++;
        }
        return root;
    }

    public static List<int> Preorder(TreeNode? node)
    {
        var outList = new List<int>();
        Pre(node, outList);
        return outList;
    }

    private static void Pre(TreeNode? node, List<int> outList)
    {
        if (node == null) return;
        outList.Add(node.Val);
        Pre(node.Left, outList);
        Pre(node.Right, outList);
    }

    public static List<int> Inorder(TreeNode? node)
    {
        var outList = new List<int>();
        In(node, outList);
        return outList;
    }

    private static void In(TreeNode? node, List<int> outList)
    {
        if (node == null) return;
        In(node.Left, outList);
        outList.Add(node.Val);
        In(node.Right, outList);
    }

    private static TreeNode? last, head;

    public static TreeNode? ToDoublyLinkedList(TreeNode? root)
    {
        last = null;
        head = null;
        Visit(root);
        return head;
    }

    private static void Visit(TreeNode? node)
    {
        if (node == null) return;
        Visit(node.Left);
        node.Left = last;                       // prev
        if (last != null) last.Right = node;    // next of predecessor
        else head = node;
        last = node;
        Visit(node.Right);
    }

    public static List<int> ToValues(TreeNode? head)
    {
        var outList = new List<int>();
        for (var node = head; node != null; node = node.Right) outList.Add(node.Val);
        return outList;
    }

    public static List<int> ToValuesReverse(TreeNode? head)
    {
        var outList = new List<int>();
        if (head == null) return outList;
        var node = head;
        while (node.Right != null) node = node.Right;
        for (; node != null; node = node.Left) outList.Add(node.Val);
        return outList;
    }

    public static void Main()
    {
        var head = ToDoublyLinkedList(BuildTree(4, 2, 5, 1, 3));
        Check(string.Join(",", ToValues(head)) == "1,2,3,4,5", "inorder order");
        Check(string.Join(",", ToValuesReverse(head)) == "5,4,3,2,1", "prev links");
        Check(ToDoublyLinkedList(null) == null, "empty");
        var single = ToDoublyLinkedList(BuildTree(7));
        Check(ToValues(single).Count == 1 && ToValues(single)[0] == 7, "single");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}

