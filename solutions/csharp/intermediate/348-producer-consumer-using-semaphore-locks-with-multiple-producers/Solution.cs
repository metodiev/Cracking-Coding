// Task 348 - Multi-producer/multi-consumer with counting semaphores. (Tier: Intermediate, CTCI Ch 15)
using System;
using System.Collections.Generic;
using System.Threading;

public static class Solution
{
    /// <summary>
    /// Bounded buffer of capacity 7: a SemaphoreSlim(1) mutex plus the counting semaphores `empty`
    /// (free slots) and `full` (queued items).  Two producers fill disjoint halves of 0..999, two
    /// consumers drain.  After the producers finish, the main thread reserves slot permits and
    /// appends one -1 sentinel per consumer, so consumers stop deterministically with no timing
    /// guess.  Assertions: every item consumed exactly once, per-producer order preserved, capacity
    /// never exceeded, all threads terminate (background threads + bounded joins).
    /// </summary>
    private static void Produce(SemaphoreSlim empty, SemaphoreSlim full, SemaphoreSlim mutex, Queue<int> buffer,
                                int[] maxSeen, int from, int to)
    {
        for (int item = from; item < to; item++)
        {
            empty.Wait();
            mutex.Wait();
            try
            {
                buffer.Enqueue(item);
                if (buffer.Count > maxSeen[0]) maxSeen[0] = buffer.Count;
            }
            finally
            {
                mutex.Release();
            }
            full.Release();
        }
    }

    private static void Consume(SemaphoreSlim empty, SemaphoreSlim full, SemaphoreSlim mutex, Queue<int> buffer,
                                List<int[]> consumed, int sentinel)
    {
        while (true)
        {
            full.Wait();
            mutex.Wait();
            int item = buffer.Dequeue();
            if (item == sentinel)
            {
                mutex.Release();
                break;                     // the sentinel's slot permit stays consumed
            }
            consumed.Add(new[] { consumed.Count, item });   // global order recorded under the mutex
            mutex.Release();
            empty.Release();
        }
    }

    public static void Main()
    {
        const int capacity = 7;
        const int items = 1000;
        const int sentinel = -1;
        var empty = new SemaphoreSlim(capacity);
        var full = new SemaphoreSlim(0);
        var mutex = new SemaphoreSlim(1);
        var buffer = new Queue<int>();
        var consumed = new List<int[]>();
        var maxSeen = new int[1];

        var producerA = new Thread(() => Produce(empty, full, mutex, buffer, maxSeen, 0, items / 2)) { IsBackground = true };
        var producerB = new Thread(() => Produce(empty, full, mutex, buffer, maxSeen, items / 2, items)) { IsBackground = true };
        var consumerA = new Thread(() => Consume(empty, full, mutex, buffer, consumed, sentinel)) { IsBackground = true };
        var consumerB = new Thread(() => Consume(empty, full, mutex, buffer, consumed, sentinel)) { IsBackground = true };
        consumerA.Start();
        consumerB.Start();
        producerA.Start();
        producerB.Start();

        Check(producerA.Join(TimeSpan.FromSeconds(15)) && producerB.Join(TimeSpan.FromSeconds(15)), "producers finished");

        for (int i = 0; i < 2; i++)        // reserve a slot and append one sentinel per consumer
        {
            empty.Wait();
            mutex.Wait();
            buffer.Enqueue(sentinel);
            mutex.Release();
            full.Release();
        }

        Check(consumerA.Join(TimeSpan.FromSeconds(15)) && consumerB.Join(TimeSpan.FromSeconds(15)), "consumers finished");

        var ordered = new int[consumed.Count];
        for (int i = 0; i < ordered.Length; i++)
        {
            Check(consumed[i][0] == i, "consumption index " + i);
            ordered[i] = consumed[i][1];
        }
        var sorted = (int[])ordered.Clone();
        Array.Sort(sorted);
        Check(sorted.Length == items, "no item lost or duplicated");
        for (int i = 0; i < items; i++) Check(sorted[i] == i, "every item exactly once: " + i);
        Check(maxSeen[0] <= capacity, "capacity never exceeded");
        Check(maxSeen[0] >= 1, "buffer was actually used");

        var producerAItems = new List<int>();
        var producerBItems = new List<int>();
        foreach (int item in ordered)
        {
            if (item < items / 2) producerAItems.Add(item);
            else producerBItems.Add(item);
        }
        var expectedA = new List<int>();
        for (int i = 0; i < items / 2; i++) expectedA.Add(i);
        var expectedB = new List<int>();
        for (int i = items / 2; i < items; i++) expectedB.Add(i);
        Check(string.Join(",", producerAItems) == string.Join(",", expectedA), "producer A order preserved");
        Check(string.Join(",", producerBItems) == string.Join(",", expectedB), "producer B order preserved");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
