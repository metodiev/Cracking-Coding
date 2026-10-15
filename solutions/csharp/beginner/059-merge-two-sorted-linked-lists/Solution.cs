// Task 059 - Merge two sorted linked lists. (Tier: Beginner, CTCI Ch 2)
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

    public static ListNode? MergeSorted(ListNode? a, ListNode? b)
    {
        var dummy = new ListNode(0);
        var tail = dummy;
        while (a != null && b != null)
        {
            if (a.Val <= b.Val) { tail.Next = a; a = a.Next; }
            else                { tail.Next = b; b = b.Next; }
            tail = tail.Next;
        }
        tail.Next = a ?? b;
        return dummy.Next;
    }

    public static void Main()
    {
        Check(string.Join(",", ToList(MergeSorted(FromList(1, 3, 5), FromList(2, 4, 6)))) == "1,2,3,4,5,6", "interleaved");
        Check(string.Join(",", ToList(MergeSorted(null, FromList(1)))) == "1", "one empty");
        Check(ToList(MergeSorted(null, null)).Count == 0, "both empty");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
