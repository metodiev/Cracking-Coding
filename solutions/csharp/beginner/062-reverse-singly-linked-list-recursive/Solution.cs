// Task 062 - Reverse singly linked list (recursive). (Tier: Beginner, CTCI Ch 2)
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

    public static ListNode? ReverseRecursive(ListNode? head)
    {
        if (head == null || head.Next == null) return head;
        var newHead = ReverseRecursive(head.Next);
        head.Next!.Next = head;
        head.Next = null;
        return newHead;
    }

    public static void Main()
    {
        Check(string.Join(",", ToList(ReverseRecursive(FromList(1, 2, 3, 4)))) == "4,3,2,1", "four");
        Check(string.Join(",", ToList(ReverseRecursive(FromList(9)))) == "9", "single");
        Check(ToList(ReverseRecursive(null)).Count == 0, "empty");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
