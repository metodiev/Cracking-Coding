// Task 065 - Implement priority queue using built-in heap. (Tier: Beginner, CTCI Ch 3)
// .NET 6+ ships PriorityQueue<TElement, TPriority> (array-based min-heap).
using System;
using System.Collections.Generic;

public static class Solution
{
    public static void Main()
    {
        var minPq = new PriorityQueue<string, int>();
        minPq.Enqueue("low", 5);
        minPq.Enqueue("high", 1);
        minPq.Enqueue("mid", 3);
        Check(minPq.Peek() == "high", "peek min priority");
        Check(minPq.Dequeue() == "high", "pop 1");
        Check(minPq.Dequeue() == "mid", "pop 2");
        Check(minPq.Dequeue() == "low", "pop 3");
        Check(minPq.Count == 0, "emptied");

        // Max-priority: invert the priority when enqueueing.
        var maxPq = new PriorityQueue<string, int>();
        maxPq.Enqueue("a", -2);
        maxPq.Enqueue("b", -9);
        Check(maxPq.Dequeue() == "b", "max priority via negation");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
