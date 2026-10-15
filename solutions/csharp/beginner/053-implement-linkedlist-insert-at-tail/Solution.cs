// Task 053 - Implement LinkedList insert at tail. (Tier: Beginner, CTCI Ch 2)
using System;
using System.Collections.Generic;

public class ListNode
{
    public int Val;
    public ListNode? Next;
    public ListNode(int v) { Val = v; }
}

public static class Solution
{
    public static ListNode FromList(params int[] vals)
    {
        ListNode? head = null;
        for (int i = vals.Length - 1; i >= 0; i--) head = new ListNode(vals[i]) { Next = head };
        return head!;
    }

    public static List<int> ToList(ListNode? head)
    {
        var outList = new List<int>();
        for (var p = head; p != null; p = p.Next) outList.Add(p.Val);
        return outList;
    }

    public static ListNode InsertTail(ListNode? head, int value)
    {
        var node = new ListNode(value);
        if (head == null) return node;
        var p = head;
        while (p.Next != null) p = p.Next;
        p.Next = node;
        return head;
    }

    public static void Main()
    {
        ListNode? head = null;
        foreach (var v in new[] { 1, 2, 3 }) head = InsertTail(head, v);
        Check(string.Join(",", ToList(head)) == "1,2,3", "appended");
        Check(string.Join(",", ToList(InsertTail(null, 7))) == "7", "into empty");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
