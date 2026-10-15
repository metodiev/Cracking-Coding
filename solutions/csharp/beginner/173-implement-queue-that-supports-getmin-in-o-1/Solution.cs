// Task 173 - Implement queue that supports getMin in O(1). (Tier: Beginner, CTCI Ch 3)
using System;
using System.Collections.Generic;

public static class Solution
{
    private class MinStackInner
    {
        public readonly Stack<int> Data = new Stack<int>();
        public readonly Stack<int> Mins = new Stack<int>();

        public void Push(int value)
        {
            Data.Push(value);
            Mins.Push(Mins.Count == 0 ? value : Math.Min(value, Mins.Peek()));
        }

        public int Pop() { Mins.Pop(); return Data.Pop(); }
        public int Peek() => Data.Peek();
        public int Min() => Mins.Peek();
        public int Size => Data.Count;
    }

    private class MinQueue
    {
        private readonly MinStackInner inbox = new MinStackInner();
        private readonly MinStackInner outbox = new MinStackInner();

        public void Enqueue(int value) => inbox.Push(value);

        public int Dequeue()
        {
            if (IsEmpty) throw new InvalidOperationException("queue empty");
            if (outbox.Size == 0) while (inbox.Size > 0) outbox.Push(inbox.Pop());
            return outbox.Pop();
        }

        public int Min()
        {
            if (IsEmpty) throw new InvalidOperationException("queue empty");
            if (inbox.Size == 0) return outbox.Min();
            if (outbox.Size == 0) return inbox.Min();
            return Math.Min(inbox.Min(), outbox.Min());
        }

        public bool IsEmpty => inbox.Size == 0 && outbox.Size == 0;
    }

    public static void Main()
    {
        var q = new MinQueue();
        foreach (var v in new[] { 3, 1, 4, 1, 5 }) q.Enqueue(v);
        Check(q.Min() == 1, "min after enqueues");
        Check(q.Dequeue() == 3 && q.Min() == 1, "dequeue 3");
        Check(q.Dequeue() == 1 && q.Min() == 1, "duplicate min kept");
        Check(q.Dequeue() == 4 && q.Min() == 1, "dequeue 4");
        Check(q.Dequeue() == 1 && q.Min() == 5, "min moves to 5");
        q.Enqueue(0);
        Check(q.Min() == 0, "new min");
        Check(q.Dequeue() == 5 && q.Dequeue() == 0 && q.IsEmpty, "drained");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
