// Task 492 - Implement circular buffer with lock-free design (atomic). (Tier: Advanced, CTCI Ch 15)
//
// Bounded MPMC ring buffer in the Vyukov style: capacity slots each carry a sequence number
// (slot i starts at i), head and tail are plain longs advanced only by Interlocked
// compare-and-swap, and every operation retries when its CAS loses the race. A slot is
// publishable when seq == pos and consumable when seq == pos + 1; the claimer overwrites the
// slot and then stores seq = pos + 1 (enqueue) or seq = pos + capacity (dequeue), which
// re-opens the slot one lap later. One extra occupancy check (tail - head >= capacity => full)
// is required because for capacity 1 the final lap's free and published sequence values
// coincide. TryEnqueue never blocks: it reports false (full); TryDequeue reports null (empty),
// so callers spin outside. The self-test asserts: single-threaded wraparound/full/empty
// semantics, exact global FIFO for one producer / one consumer across REAL threads (bounded
// capacity forces both to spin), exactly-once delivery plus per-producer FIFO for many
// producers / consumers - the position claimed by the head CAS acts as the linearization
// witness - and a deterministic hook that lets a competing producer claim the position just
// before the outer CAS, so the retry path is exercised and must fail exactly once. All worker
// threads join with a timeout and are checked to have terminated.
using System;
using System.Collections.Generic;
using System.Threading;

internal class LockFreeCircularBuffer
{
    internal delegate void CasHook(LockFreeCircularBuffer buffer, long pos);

    private readonly int capacity;
    private readonly object?[] slots;
    private readonly long[] seq;
    private long head;
    private long tail;
    private long casAttempts;
    private long casFailures;
    private long enqueueRetries;
    private long dequeueRetries;
    private CasHook? casHook;

    internal LockFreeCircularBuffer(int capacity, CasHook? hook)
    {
        if (capacity <= 0) throw new ArgumentException("capacity must be positive");
        this.capacity = capacity;
        slots = new object?[capacity];
        seq = new long[capacity];
        for (int i = 0; i < capacity; i++) seq[i] = i;      // slot i is free at position i
        casHook = hook;
    }

    internal long Head => Volatile.Read(ref head);
    internal long Tail => Volatile.Read(ref tail);
    internal long CasAttempts => Volatile.Read(ref casAttempts);
    internal long CasFailures => Volatile.Read(ref casFailures);
    internal long EnqueueRetries => Volatile.Read(ref enqueueRetries);
    internal long DequeueRetries => Volatile.Read(ref dequeueRetries);

    /** True when value was enqueued, false when the queue is full. */
    internal bool TryEnqueue(object value)
    {
        while (true)
        {
            long pos = Volatile.Read(ref tail);
            if (pos - Volatile.Read(ref head) >= capacity)
            {
                return false;                       // occupancy guard (needed for capacity 1)
            }
            int slot = (int)(pos % capacity);
            long slotSeq = Volatile.Read(ref seq[slot]);
            if (slotSeq == pos)
            {
                CasHook? hook = casHook;
                if (hook != null)
                {
                    casHook = null;                 // fires once
                    hook(this, pos);
                }
                Interlocked.Increment(ref casAttempts);
                if (Interlocked.CompareExchange(ref tail, pos + 1, pos) == pos)
                {
                    Volatile.Write(ref slots[slot], value);
                    Volatile.Write(ref seq[slot], pos + 1);     // publish (release store)
                    return true;
                }
                Interlocked.Increment(ref casFailures);         // lost: reread and retry
            }
            else if (slotSeq < pos)
            {
                return false;                       // the previous lap has not caught up
            }
            Interlocked.Increment(ref enqueueRetries);
        }
    }

    /** Oldest value, or null when empty. When claimSink is given, claimSink[0] receives the
     * position claimed by the head CAS (the linearization witness used by the self-test). */
    internal object? TryDequeue(long[]? claimSink)
    {
        while (true)
        {
            long pos = Volatile.Read(ref head);
            int slot = (int)(pos % capacity);
            long slotSeq = Volatile.Read(ref seq[slot]);
            if (slotSeq == pos + 1)
            {
                Interlocked.Increment(ref casAttempts);
                if (Interlocked.CompareExchange(ref head, pos + 1, pos) == pos)
                {
                    if (claimSink != null) claimSink[0] = pos;
                    object? value = Volatile.Read(ref slots[slot]);
                    Volatile.Write(ref seq[slot], pos + capacity);   // reopen one lap later
                    return value;
                }
                Interlocked.Increment(ref casFailures);
            }
            else if (slotSeq < pos + 1)
            {
                return null;                        // nothing published yet: empty
            }
            Interlocked.Increment(ref dequeueRetries);
        }
    }
}

public static class Solution
{
    private static void Join(Thread thread, int timeoutMillis)
    {
        thread.Join(timeoutMillis);
    }

    public static void Main()
    {
        // Single-threaded semantics: wraparound, full, empty, capacity 1.
        var buf = new LockFreeCircularBuffer(4, null);
        for (int value = 1; value <= 4; value++) Check(buf.TryEnqueue(value), "enqueue " + value);
        Check(!buf.TryEnqueue(5), "full");
        for (int value = 1; value <= 4; value++)
        {
            Check(Equals(buf.TryDequeue(null), value), "dequeue " + value);
        }
        Check(buf.TryDequeue(null) == null, "empty");
        Check(buf.TryEnqueue(5) && buf.TryEnqueue(6), "after wrap");
        Check(Equals(buf.TryDequeue(null), 5) && Equals(buf.TryDequeue(null), 6), "wrapped twice");
        Check(buf.TryDequeue(null) == null, "empty again");

        var single = new LockFreeCircularBuffer(1, null);
        Check(single.TryEnqueue("a"), "cap1 push");
        Check(!single.TryEnqueue("b"), "cap1 full");
        Check(Equals(single.TryDequeue(null), "a"), "cap1 pop");
        Check(single.TryEnqueue("b"), "cap1 push again");
        Check(Equals(single.TryDequeue(null), "b"), "cap1 pop again");

        // Deterministic retry-path demonstration: the hook, running just before the outer CAS,
        // lets a "concurrent" producer claim the same position; the outer CAS must fail exactly
        // once, reread the tail and succeed on the next lap, hook's item first in FIFO order.
        var forced = new LockFreeCircularBuffer(8, (buffer, pos) =>
        {
            Check(buffer.TryEnqueue("stealer"), "hook enqueue");
            Check(buffer.Head == 0 && buffer.Tail == pos + 1, "hook claim");
        });
        Check(forced.TryEnqueue("mine"), "outer enqueue");
        Check(forced.CasFailures == 1, "stale CAS fails exactly once");
        Check(forced.EnqueueRetries == 1, "exactly one retry lap");
        Check(Equals(forced.TryDequeue(null), "stealer"), "hook item first");
        Check(Equals(forced.TryDequeue(null), "mine"), "retried item second");
        Check(forced.TryDequeue(null) == null, "forced empty");

        // SPSC across real threads: bounded capacity forces the producer to spin on FULL and
        // the consumer to spin on EMPTY; the dequeued stream must be exactly 0, 1, ..., total-1.
        int total = 20000;
        var spsc = new LockFreeCircularBuffer(8, null);
        var received = new List<int>();
        var producer = new Thread(() =>
        {
            for (int value = 0; value < total; value++)
            {
                while (!spsc.TryEnqueue(value)) Thread.Yield();
            }
        });
        var consumer = new Thread(() =>
        {
            while (received.Count < total)
            {
                object? value = spsc.TryDequeue(null);
                if (value == null) Thread.Yield();
                else received.Add((int)value);
            }
        });
        producer.Start();
        consumer.Start();
        Join(producer, 30000);
        Join(consumer, 30000);
        Check(!producer.IsAlive && !consumer.IsAlive, "SPSC threads terminate");
        Check(received.Count == total, "SPSC count");
        for (int i = 0; i < total; i++) Check(received[i] == i, "SPSC FIFO at " + i);

        // MPMC accounting: 4 producers x 300 tagged items, 3 consumers, capacity 16. Consumers
        // record the claimed position, so the merged record reproduces the dequeue order:
        // positions 0..total-1 exactly once, and each producer's items in its enqueue order.
        int producers = 4;
        int perProducer = 300;
        int consumers = 3;
        int grand = producers * perProducer;
        var mpmc = new LockFreeCircularBuffer(16, null);
        var records = new List<long[]>[consumers];
        for (int c = 0; c < consumers; c++) records[c] = new List<long[]>();
        int[] tally = { 0 };

        var workers = new Thread[producers + consumers];
        for (int p = 0; p < producers; p++)
        {
            int producerIndex = p;
            workers[p] = new Thread(() =>
            {
                for (int i = 0; i < perProducer; i++)
                {
                    int[] item = { producerIndex, i };
                    while (!mpmc.TryEnqueue(item)) Thread.Yield();
                }
            });
        }
        for (int c = 0; c < consumers; c++)
        {
            int consumerIndex = c;
            workers[producers + c] = new Thread(() =>
            {
                List<long[]> local = records[consumerIndex];
                while (Volatile.Read(ref tally[0]) < grand)
                {
                    long[] claim = new long[1];
                    object? value = mpmc.TryDequeue(claim);
                    if (value == null)
                    {
                        Thread.Yield();
                        continue;
                    }
                    var item = (int[])value;
                    local.Add(new[] { claim[0], item[0], item[1] });
                    Interlocked.Increment(ref tally[0]);
                }
            });
        }
        foreach (Thread worker in workers) worker.Start();
        foreach (Thread worker in workers) Join(worker, 30000);
        foreach (Thread worker in workers) Check(!worker.IsAlive, "MPMC worker terminates");

        Check(Volatile.Read(ref tally[0]) == grand, "exactly-once delivery");
        var merged = new List<long[]>();
        foreach (List<long[]> local in records) merged.AddRange(local);
        merged.Sort((a, b) => a[0].CompareTo(b[0]));
        Check(merged.Count == grand, "merged size");
        for (int i = 0; i < grand; i++) Check(merged[i][0] == i, "positions permutation at " + i);
        var order = new List<long>[producers];
        for (int p = 0; p < producers; p++) order[p] = new List<long>();
        foreach (long[] record in merged) order[record[1]].Add(record[2]);
        for (int p = 0; p < producers; p++)
        {
            for (int i = 0; i < perProducer; i++)
            {
                Check(order[p][i] == i, "per-producer FIFO producer " + p + " item " + i);
            }
        }
        Check(mpmc.CasAttempts >= grand, "every operation used the CAS path");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
