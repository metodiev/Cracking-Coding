// Task 060 - Remove nth node from end of linked list. (Tier: Beginner, CTCI Ch 2)
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

    public static ListNode? RemoveNthFromEnd(ListNode? head, int n)
    {
        var dummy = new ListNode(0) { Next = head };
        var slow = dummy;
        var fast = dummy;
        for (int i = 0; i <= n; i++) fast = fast.Next;
        while (fast != null) { slow = slow.Next; fast = fast.Next; }
        slow.Next = slow.Next!.Next;
        return dummy.Next;
    }

    public static void Main()
    {
        Check(string.Join(",", ToList(RemoveNthFromEnd(FromList(1, 2, 3, 4, 5), 2))) == "1,2,3,5", "middle");
        Check(string.Join(",", ToList(RemoveNthFromEnd(FromList(1, 2, 3, 4, 5), 5))) == "2,3,4,5", "head");
        Check(ToList(RemoveNthFromEnd(FromList(1), 1)).Count == 0, "single");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
