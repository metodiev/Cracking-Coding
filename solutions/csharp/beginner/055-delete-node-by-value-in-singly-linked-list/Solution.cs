// Task 055 - Delete node by value in singly linked list. (Tier: Beginner, CTCI Ch 2)
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

    public static ListNode? DeleteValue(ListNode? head, int value)
    {
        var dummy = new ListNode(0) { Next = head };
        var prev = dummy;
        while (prev.Next != null)
        {
            if (prev.Next.Val == value)
            {
                prev.Next = prev.Next.Next;
                break;
            }
            prev = prev.Next;
        }
        return dummy.Next;
    }

    public static void Main()
    {
        Check(string.Join(",", ToList(DeleteValue(FromList(1, 2, 3), 2))) == "1,3", "middle");
        Check(string.Join(",", ToList(DeleteValue(FromList(1, 2, 3), 1))) == "2,3", "head");
        Check(ToList(DeleteValue(FromList(1), 1)).Count == 0, "only node");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
