// Task 058 - Detect cycle in linked list (Floyd). (Tier: Beginner, CTCI Ch 2)
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

    public static bool HasCycle(ListNode? head)
    {
        var slow = head;
        var fast = head;
        while (fast != null && fast.Next != null)
        {
            slow = slow!.Next;
            fast = fast.Next.Next;
            if (ReferenceEquals(slow, fast)) return true;
        }
        return false;
    }

    public static void Main()
    {
        var a = FromList(1, 2, 3, 4);
        Check(!HasCycle(a), "no cycle");
        a.Next!.Next!.Next!.Next = a.Next;    // 4 -> 2
        Check(HasCycle(a), "cycle detected");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
