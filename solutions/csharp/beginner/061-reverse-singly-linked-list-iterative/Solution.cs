// Task 061 - Reverse singly linked list (iterative). (Tier: Beginner, CTCI Ch 2)
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

    public static ListNode? Reverse(ListNode? head)
    {
        ListNode? prev = null;
        while (head != null)
        {
            var next = head.Next;
            head.Next = prev;
            prev = head;
            head = next;
        }
        return prev;
    }

    public static void Main()
    {
        Check(string.Join(",", ToList(Reverse(FromList(1, 2, 3)))) == "3,2,1", "three");
        Check(string.Join(",", ToList(Reverse(FromList(1)))) == "1", "single");
        Check(ToList(Reverse(null)).Count == 0, "empty");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
