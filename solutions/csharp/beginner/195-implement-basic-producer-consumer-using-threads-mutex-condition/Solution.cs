// Task 195 - Implement basic producer-consumer using threads (mutex/condition). (Tier: Beginner, CTCI Ch 15)
using System;
using System.Collections.Generic;
using System.Threading;

public static class Solution
{
    private class BoundedBuffer
    {
        private readonly object gate = new object();
        private readonly int capacity;
        private readonly Queue<int> items = new Queue<int>();
        private bool closed;
        private int maxSeen;

        public BoundedBuffer(int capacity)
        {
            if (capacity <= 0) throw new ArgumentException("capacity must be positive");
            this.capacity = capacity;
        }

        public void Put(int item)
        {
            lock (gate)
            {
                while (!closed && items.Count >= capacity) Monitor.Wait(gate);   // block while full
                if (closed) return;
                items.Enqueue(item);
                if (items.Count > maxSeen) maxSeen = items.Count;
                Monitor.PulseAll(gate);
            }
        }

        public int? Take()
        {
            lock (gate)
            {
                while (!closed && items.Count == 0) Monitor.Wait(gate);          // block while empty
                if (items.Count == 0) return null;
                int item = items.Dequeue();
                Monitor.PulseAll(gate);
                return item;
            }
        }

        public void Close()
        {
            lock (gate)
            {
                closed = true;
                Monitor.PulseAll(gate);
            }
        }

        public int MaxSeen
        {
            get { lock (gate) { return maxSeen; } }
        }
    }

    public static void Main()
    {
        const int items = 1000;
        const int capacity = 7;
        var buffer = new BoundedBuffer(capacity);
        var consumedA = new List<int>();
        var consumedB = new List<int>();

        var consumerA = new Thread(() => ConsumeAll(buffer, consumedA));
        var consumerB = new Thread(() => ConsumeAll(buffer, consumedB));
        var producerA = new Thread(() => Produce(buffer, 0, items / 2));
        var producerB = new Thread(() => Produce(buffer, items / 2, items));

        consumerA.Start();
        consumerB.Start();
        producerA.Start();
        producerB.Start();

        // Timeout sanity: every thread must terminate, otherwise the run fails.
        Check(producerA.Join(10000), "producer A terminated");
        Check(producerB.Join(10000), "producer B terminated");
        buffer.Close();
        Check(consumerA.Join(10000), "consumer A terminated");
        Check(consumerB.Join(10000), "consumer B terminated");

        var consumed = new List<int>();
        consumed.AddRange(consumedA);
        consumed.AddRange(consumedB);
        Check(consumed.Count == items, "no items lost or duplicated");
        consumed.Sort();
        var expected = new List<int>();
        for (int i = 0; i < items; i++) expected.Add(i);
        Check(ListsEqual(consumed, expected), "every item consumed exactly once");
        Check(buffer.MaxSeen <= capacity, "capacity never exceeded");
        Check(buffer.MaxSeen >= 1, "buffer was actually used");
        Console.WriteLine("PASS");
    }

    private static void Produce(BoundedBuffer buffer, int from, int to)
    {
        for (int item = from; item < to; item++) buffer.Put(item);
    }

    private static void ConsumeAll(BoundedBuffer buffer, List<int> outItems)
    {
        while (true)
        {
            int? item = buffer.Take();
            if (item == null) break;
            outItems.Add(item.Value);
        }
    }

    private static bool ListsEqual(List<int> a, List<int> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++)
        {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
