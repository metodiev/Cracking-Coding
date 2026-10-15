// Task 063 - Implement doubly linked list insert/delete. (Tier: Beginner, CTCI Ch 2)
using System;
using System.Collections.Generic;

public class DNode
{
    public int Val;
    public DNode? Prev;
    public DNode? Next;
    public DNode(int v) { Val = v; }
}

public class DoublyLinkedList
{
    public DNode? Head;
    public DNode? Tail;

    public DNode Append(int val)
    {
        var node = new DNode(val);
        if (Tail == null) Head = Tail = node;
        else
        {
            node.Prev = Tail;
            Tail.Next = node;
            Tail = node;
        }
        return node;
    }

    public DNode InsertBefore(DNode node, int val)
    {
        var fresh = new DNode(val) { Prev = node.Prev, Next = node };
        if (node.Prev != null) node.Prev.Next = fresh;
        else Head = fresh;
        node.Prev = fresh;
        return fresh;
    }

    public void Delete(DNode node)
    {
        if (node.Prev != null) node.Prev.Next = node.Next;
        else Head = node.Next;
        if (node.Next != null) node.Next.Prev = node.Prev;
        else Tail = node.Prev;
        node.Prev = node.Next = null;
    }

    public List<int> ToList()
    {
        var outList = new List<int>();
        for (var p = Head; p != null; p = p.Next) outList.Add(p.Val);
        return outList;
    }

    public List<int> ToListReversed()
    {
        var outList = new List<int>();
        for (var p = Tail; p != null; p = p.Prev) outList.Add(p.Val);
        return outList;
    }
}

public static class Solution
{
    public static void Main()
    {
        var d = new DoublyLinkedList();
        var a = d.Append(1);
        var b = d.Append(2);
        var c = d.Append(3);
        Check(string.Join(",", d.ToList()) == "1,2,3", "append");
        Check(string.Join(",", d.ToListReversed()) == "3,2,1", "reversed links");

        d.InsertBefore(b, 9);
        Check(string.Join(",", d.ToList()) == "1,9,2,3", "insert before middle");
        d.InsertBefore(a, 0);
        Check(string.Join(",", d.ToList()) == "0,1,9,2,3", "insert before head");
        Check(string.Join(",", d.ToListReversed()) == "3,2,9,1,0", "reverse after inserts");

        d.Delete(a);
        Check(string.Join(",", d.ToList()) == "0,9,2,3", "delete middle");
        d.Delete(c);
        Check(string.Join(",", d.ToList()) == "0,9,2", "delete tail");
        d.Delete(b);
        Check(string.Join(",", d.ToList()) == "0,9", "delete node");
        Check(string.Join(",", d.ToListReversed()) == "9,0", "reverse after deletes");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
