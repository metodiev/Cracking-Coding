// Task 064 - Implement circular queue (ring buffer). (Tier: Beginner, CTCI Ch 3)
using System;

public class CircularQueue
{
    private readonly int?[] data;
    private int head = 0, count = 0;

    public CircularQueue(int capacity)
    {
        if (capacity <= 0) throw new ArgumentException("capacity must be positive");
        data = new int?[capacity];
    }

    public void Enqueue(int value)
    {
        if (IsFull) throw new InvalidOperationException("queue full");
        data[(head + count) % data.Length] = value;
        count++;
    }

    public int Dequeue()
    {
        if (IsEmpty) throw new InvalidOperationException("queue empty");
        int value = data[head]!.Value;
        data[head] = null;
        head = (head + 1) % data.Length;
        count--;
        return value;
    }

    public int Front()
    {
        if (IsEmpty) throw new InvalidOperationException("queue empty");
        return data[head]!.Value;
    }

    public int Rear()
    {
        if (IsEmpty) throw new InvalidOperationException("queue empty");
        return data[(head + count - 1) % data.Length]!.Value;
    }

    public bool IsEmpty => count == 0;
    public bool IsFull => count == data.Length;
    public int Size => count;
}

public static class Solution
{
    public static void Main()
    {
        var q = new CircularQueue(3);
        Check(q.IsEmpty && !q.IsFull, "starts empty");
        q.Enqueue(1); q.Enqueue(2); q.Enqueue(3);
        Check(q.IsFull && q.Size == 3, "fills up");
        Check(q.Front() == 1 && q.Rear() == 3, "front/rear");
        Check(q.Dequeue() == 1, "dequeue");
        q.Enqueue(4);
        Check(q.Front() == 2 && q.Rear() == 4, "wrap-around");
        Check(q.Dequeue() == 2 && q.Dequeue() == 3 && q.Dequeue() == 4, "drain in order");
        Check(q.IsEmpty, "empty at end");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
