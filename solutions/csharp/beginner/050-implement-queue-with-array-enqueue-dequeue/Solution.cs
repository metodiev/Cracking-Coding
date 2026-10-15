// Task 050 - Implement queue with array (enqueue/dequeue). (Tier: Beginner, CTCI Ch 3)
using System;
using System.Collections.Generic;
using System.Text;

public class QueueArray
{
    private readonly int[] data;
    private int head = 0, count = 0;

    public QueueArray(int capacity) { data = new int[capacity]; }

    public void Enqueue(int value)
    {
        if (count == data.Length) throw new InvalidOperationException("queue full");
        data[(head + count) % data.Length] = value;
        count++;
    }

    public int Dequeue()
    {
        if (IsEmpty) throw new InvalidOperationException("queue empty");
        int value = data[head];
        head = (head + 1) % data.Length;
        count--;
        return value;
    }

    public int Front()
    {
        if (IsEmpty) throw new InvalidOperationException("queue empty");
        return data[head];
    }

    public bool IsEmpty => count == 0;
    public int Size => count;
}

public static class Solution
{
    public static void Main()
    {
        var q = new QueueArray(3);
        q.Enqueue(1); q.Enqueue(2); q.Enqueue(3);
        Check(q.Front() == 1, "front");
        Check(q.Dequeue() == 1, "dequeue 1");
        q.Enqueue(4);
        Check(q.Dequeue() == 2 && q.Dequeue() == 3 && q.Dequeue() == 4, "FIFO order");
        Check(q.IsEmpty, "empty at end");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
