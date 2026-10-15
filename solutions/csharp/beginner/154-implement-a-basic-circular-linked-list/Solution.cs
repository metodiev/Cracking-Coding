// Task 154 - Implement a basic circular linked list. (Tier: Beginner, CTCI Ch 2)
using System;
using System.Collections.Generic;

public class CircularNode
{
    public int Value;
    public CircularNode? Next;
    public CircularNode(int value) { Value = value; }
}

/// <summary>Singly linked circular list: the tail always points back at the head.</summary>
public class CircularLinkedList
{
    public CircularNode? Head;
    public int Size;

    /// <summary>Append at the tail; the new tail points back at the head.</summary>
    public void Append(int value)
    {
        var node = new CircularNode(value);
        if (Head == null)
        {
            node.Next = node;
            Head = node;
        }
        else
        {
            var tail = Tail()!;
            tail.Next = node;
            node.Next = Head;
        }
        Size++;
    }

    /// <summary>Insert before position `index` (0 &lt;= index &lt;= size); size appends.</summary>
    public void Insert(int index, int value)
    {
        if (index < 0 || index > Size) throw new ArgumentOutOfRangeException(nameof(index), "insert index out of range");
        if (index == Size) { Append(value); return; }
        var node = new CircularNode(value);
        if (index == 0)
        {
            var tail = Tail()!;
            node.Next = Head;
            tail.Next = node;
            Head = node;
        }
        else
        {
            var prev = Head!;
            for (int i = 0; i < index - 1; i++) prev = prev.Next!;
            node.Next = prev.Next;
            prev.Next = node;
        }
        Size++;
    }

    /// <summary>Delete the first node holding `value`; returns true if one was removed.</summary>
    public bool Delete(int value)
    {
        if (Head == null) return false;
        if (Head.Value == value)
        {
            if (Size == 1)
            {
                Head = null;
            }
            else
            {
                var tail = Tail()!;
                Head = Head.Next;
                tail.Next = Head;
            }
            Size--;
            return true;
        }
        var prev = Head;
        while (prev.Next != Head && prev.Next!.Value != value) prev = prev.Next!;
        if (prev.Next == Head) return false;
        prev.Next = prev.Next!.Next;
        Size--;
        return true;
    }

    /// <summary>The node whose next is the head (null for an empty list).</summary>
    public CircularNode? Tail()
    {
        if (Head == null) return null;
        var node = Head;
        while (node.Next != Head) node = node.Next!;
        return node;
    }

    /// <summary>One lap: exactly `size` steps starting at the head (terminates).</summary>
    public List<int> ToList()
    {
        var outList = new List<int>();
        var node = Head;
        for (int i = 0; i < Size; i++)
        {
            outList.Add(node!.Value);
            node = node.Next;
        }
        return outList;
    }
}

public static class Solution
{
    public static void Main()
    {
        var lst = new CircularLinkedList();
        Check(lst.ToList().Count == 0 && lst.Head == null && lst.Size == 0, "empty list");

        foreach (int v in new[] { 1, 2, 3 }) lst.Append(v);
        Check(Fmt(lst.ToList()) == "[1, 2, 3]" && lst.Size == 3, "appended three");
        Check(lst.Tail()!.Next == lst.Head, "tail wraps to head");
        Check(lst.Head!.Next!.Next!.Next == lst.Head, "third node wraps to head");

        lst.Insert(0, 0);
        Check(Fmt(lst.ToList()) == "[0, 1, 2, 3]" && lst.Head!.Value == 0, "insert at head");
        lst.Insert(2, 99);
        Check(Fmt(lst.ToList()) == "[0, 1, 99, 2, 3]", "insert in middle");
        lst.Insert(lst.Size, 4);
        Check(Fmt(lst.ToList()) == "[0, 1, 99, 2, 3, 4]", "insert at tail");
        Check(lst.Tail()!.Value == 4 && lst.Tail()!.Next == lst.Head, "tail still wraps");

        bool threw = false;
        try { lst.Insert(99, 0); } catch (ArgumentOutOfRangeException) { threw = true; }
        Check(threw, "insert out of range throws");

        Check(lst.Delete(99) && Fmt(lst.ToList()) == "[0, 1, 2, 3, 4]", "delete middle");
        Check(lst.Delete(0) && lst.Head!.Value == 1 && Fmt(lst.ToList()) == "[1, 2, 3, 4]", "delete head");
        Check(lst.Tail()!.Next == lst.Head, "ring still closed after head delete");
        Check(lst.Delete(4) && Fmt(lst.ToList()) == "[1, 2, 3]", "delete tail");
        Check(!lst.Delete(42) && Fmt(lst.ToList()) == "[1, 2, 3]", "missing value");

        Check(lst.Delete(1) && lst.Delete(2) && lst.Delete(3), "drain to empty");
        Check(lst.ToList().Count == 0 && lst.Size == 0 && lst.Head == null && lst.Tail() == null, "empty again");
        Check(!lst.Delete(1), "delete on empty");
        lst.Append(7);
        Check(Fmt(lst.ToList()) == "[7]" && lst.Head!.Next == lst.Head, "single node self-loop");
        Check(lst.Delete(7) && lst.ToList().Count == 0 && lst.Head == null, "delete last node");
        Console.WriteLine("PASS");
    }

    private static string Fmt(List<int> values) => "[" + string.Join(", ", values) + "]";

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
