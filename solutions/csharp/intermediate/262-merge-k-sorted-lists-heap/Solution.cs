// Task 262 - Merge K sorted lists (heap). (Tier: Intermediate, CTCI Ch 10)
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
    public static ListNode? BuildList(params int[] values)
    {
        ListNode? head = null;
        for (int i = values.Length - 1; i >= 0; i--) head = new ListNode(values[i]) { Next = head };
        return head;
    }

    public static List<int> ToArray(ListNode? head)
    {
        var outList = new List<int>();
        for (var p = head; p != null; p = p.Next) outList.Add(p.Val);
        return outList;
    }

    /// <summary>
    /// Min-heap over the current heads: each of the n nodes enters the heap once, so
    /// O(n log k) time / O(k) space; nodes are reused, tails rewired in place. Returns
    /// null for k = 0 and the SAME head object when k = 1. Priorities are
    /// (value, insertion serial) so value ties are consumed deterministically.
    /// </summary>
    public static ListNode? MergeKLists(ListNode?[] lists)
    {
        var pq = new PriorityQueue<ListNode, (int Val, int Serial)>();
        int serial = 0;
        foreach (var node in lists)
        {
            if (node != null)
            {
                pq.Enqueue(node, (node.Val, serial));
                serial++;
            }
        }
        var dummy = new ListNode(0);
        var tail = dummy;
        while (pq.Count > 0)
        {
            var node = pq.Dequeue();
            tail.Next = node;
            tail = node;
            if (node.Next != null)
            {
                pq.Enqueue(node.Next, (node.Next.Val, serial));
                serial++;
            }
        }
        tail.Next = null;
        return dummy.Next;
    }

    /// <summary>Flat sort reference used only for cross-checks; call BEFORE merging (merging rewires tails).</summary>
    public static int[] MergeOracle(ListNode?[] lists)
    {
        var values = new List<int>();
        foreach (var node in lists) values.AddRange(ToArray(node));
        values.Sort();
        return values.ToArray();
    }

    private static void CheckList(ListNode? head, int[] expected, string label)
    {
        var got = ToArray(head);
        bool ok = got.Count == expected.Length;
        for (int i = 0; ok && i < expected.Length; i++) ok = got[i] == expected[i];
        Check(ok, label + " got=[" + string.Join(", ", got) + "]");
    }

    public static void Main()
    {
        var merged = MergeKLists(new ListNode?[] { BuildList(1, 4, 5), BuildList(1, 3, 4), BuildList(2, 6) });
        CheckList(merged, new[] { 1, 1, 2, 3, 4, 4, 5, 6 }, "classic");

        Check(MergeKLists(new ListNode?[0]) == null, "k = 0 -> empty");
        Check(MergeKLists(new ListNode?[] { null, null }) == null, "only empty lists");

        var single = BuildList(1, 2, 3);
        var same = MergeKLists(new ListNode?[] { single });
        Check(ReferenceEquals(same, single), "k = 1 returns the same list");
        CheckList(same, new[] { 1, 2, 3 }, "single list kept");

        CheckList(MergeKLists(new ListNode?[] { BuildList(1), BuildList(2, 3, 4, 5), BuildList(6) }),
            new[] { 1, 2, 3, 4, 5, 6 }, "unequal lengths");
        CheckList(MergeKLists(new ListNode?[] { BuildList(1, 1), BuildList(1, 1) }),
            new[] { 1, 1, 1, 1 }, "duplicates");
        CheckList(MergeKLists(new ListNode?[] { null, BuildList(1), null, BuildList() }),
            new[] { 1 }, "empty lists mixed in");
        CheckList(MergeKLists(new ListNode?[] { BuildList(-3, -1), BuildList(-2, 0) }),
            new[] { -3, -2, -1, 0 }, "negatives");
        CheckList(MergeKLists(new ListNode?[] { BuildList(5, 5), BuildList(5), BuildList() }),
            new[] { 5, 5, 5 }, "repeats with an empty tail");

        var rng = new Random(262);
        for (int test = 0; test < 300; test++)          // cross-check vs the flat-sort oracle
        {
            int k = rng.Next(6);
            var lists = new ListNode?[k];
            for (int i = 0; i < k; i++)
            {
                int len = rng.Next(6);
                var values = new int[len];
                for (int j = 0; j < len; j++) values[j] = rng.Next(13) - 6;
                Array.Sort(values);
                lists[i] = BuildList(values);
            }
            var expected = MergeOracle(lists);
            CheckList(MergeKLists(lists), expected, "random agreement");
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
