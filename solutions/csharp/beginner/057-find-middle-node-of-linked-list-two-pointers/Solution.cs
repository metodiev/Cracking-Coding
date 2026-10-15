// Task 057 - Find middle node of linked list (two pointers). (Tier: Beginner, CTCI Ch 2)
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

    public static ListNode? Middle(ListNode? head)
    {
        var slow = head;
        var fast = head;
        while (fast != null && fast.Next != null)
        {
            slow = slow!.Next;
            fast = fast.Next.Next;
        }
        return slow;
    }

    public static void Main()
    {
        Check(Middle(FromList(1, 2, 3, 4, 5))!.Val == 3, "odd");
        Check(Middle(FromList(1, 2, 3, 4))!.Val == 3, "even -> second middle");
        Check(Middle(null) == null, "empty");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
