// Task 625 - Implement wait-free queue algorithm (Kogan-Petrank style outline). (Tier: Expert, CTCI Ch 15 - Threads and Locks)
//
// The Kogan-Petrank (KP) queue is wait-free: enqueue reserves a unique, immutable index with a
// fetch-and-add on EnqIdx (the only contended primitive) and later writes the item into the slot
// that index selects; dequeue does the same on DeqIdx. Indices are monotone, so a slot is never
// reused for the same index and no ABA problem exists. Storage is a chain of fixed-size array
// segments, so the queue grows without copying and the dequeue side can recycle the segment that
// both sides have drained, which keeps the footprint bounded.
//
//   * Reserve()         = the enqueue's FAA step; returns the reserved index together with the
//                         segment that owns it (the real algorithm derives the segment from the
//                         index; carrying it explicitly is the documented simplification below).
//                         The drained last segment is recycled here rather than reallocated.
//   * Complete(token,v) = the enqueue's write step: publish the item and set the slot's flag.
//   * TryDequeue        = Empty when no index was ever reserved, Pending when an enqueue reserved
//                         the index but has not written yet, Ok otherwise. A dequeuer never skips
//                         a reserved index, so the dequeue order is the RESERVATION order, which
//                         is exactly what makes the queue linearizable when two enqueuers' writes
//                         land out of order.
//
// What is simplified (documented per the authoring contract): concurrency is simulated
// deterministically - one process, no threads, explicit Reserve/Complete/TryDequeue steps - and
// the in-flight-segment "helping" of the real algorithm is reduced to deterministic segment
// creation. The dequeuer does not consume its index while the queue is empty, and an enqueuer
// that crashed between Reserve and Complete would leave a permanent hole. Verified invariants:
// FIFO in reservation order, no item lost or duplicated, bounded steps per operation (the
// wait-freedom witness) and a bounded number of live segments under a steady workload.
using System;
using System.Collections.Generic;

#nullable enable

internal enum DequeueStatus { Ok, Empty, Pending }

internal sealed class KpSegment
{
    public int Id;
    public long Start;
    public readonly int[] Items;
    public readonly bool[] Written;
    public readonly int Capacity;
    public KpSegment? Next;

    public KpSegment(int id, long start, int capacity)
    {
        Id = id;
        Start = start;
        Capacity = capacity;
        Items = new int[capacity];
        Written = new bool[capacity];
    }

    public long UpperBound => Start + Capacity;

    public void Rebase(long start)
    {
        Start = start;
        for (int i = 0; i < Capacity; i++) Written[i] = false;
    }
}

internal readonly struct EnqueueToken
{
    public readonly long Index;
    public readonly KpSegment Segment;

    public EnqueueToken(long index, KpSegment segment)
    {
        Index = index;
        Segment = segment;
    }
}

internal sealed class KoganPetrankQueue
{
    private readonly int capacity;
    private KpSegment head;   // segment holding DeqIdx
    private KpSegment tail;   // segment owning the highest reserved index

    public long EnqIdx;
    public long DeqIdx;
    public int SegmentsAllocated;
    public int SegmentsReused;
    public int MaxOpSteps;
    public long TotalSteps;
    public long Operations;

    public KoganPetrankQueue(int capacityBits)
    {
        capacity = 1 << capacityBits;
        head = new KpSegment(0, 0, capacity);
        tail = head;
        SegmentsAllocated = 1;
    }

    public int Capacity => capacity;

    private void Charge(int steps)
    {
        Operations++;
        TotalSteps += steps;
        if (steps > MaxOpSteps) MaxOpSteps = steps;
    }

    /// <summary>Fetch-and-add on EnqIdx (the enqueue reservation); bounded, non-blocking steps.</summary>
    public EnqueueToken Reserve()
    {
        int steps = 1;
        long idx = EnqIdx++;
        while (idx >= tail.UpperBound)
        {
            steps++;
            if (tail.Next != null)
            {
                tail = tail.Next;
                continue;
            }
            // A single fully drained segment is recycled instead of being replaced.
            if (ReferenceEquals(head, tail) && DeqIdx >= tail.UpperBound)
            {
                tail.Rebase(tail.Start + capacity);
                SegmentsReused++;
                steps++;
                continue;
            }
            tail.Next = new KpSegment(SegmentsAllocated, tail.UpperBound, capacity);
            SegmentsAllocated++;
            tail = tail.Next;
        }
        Charge(steps);
        return new EnqueueToken(idx, tail);
    }

    /// <summary>Publish the reserved item: the enqueue's write step, O(1) thanks to the token.</summary>
    public void Complete(EnqueueToken token, int value)
    {
        int slot = (int)(token.Index - token.Segment.Start);
        token.Segment.Items[slot] = value;
        token.Segment.Written[slot] = true;
        Charge(2);
    }

    public void Enqueue(int value) => Complete(Reserve(), value);

    public DequeueStatus TryDequeue(out int value)
    {
        value = 0;
        int steps = 1;
        long idx = DeqIdx;
        if (idx >= EnqIdx) { Charge(steps); return DequeueStatus.Empty; }
        int walk = 0;
        KpSegment seg = head;
        while (idx >= seg.UpperBound)
        {
            seg = seg.Next ?? throw new Exception("segment chain broken at index " + idx);
            walk++;
        }
        steps += walk;
        int slot = (int)(idx - seg.Start);
        if (!seg.Written[slot]) { Charge(steps); return DequeueStatus.Pending; }
        value = seg.Items[slot];
        seg.Written[slot] = false;
        head = seg;
        DeqIdx = idx + 1;
        Charge(steps + 2);
        return DequeueStatus.Ok;
    }

    public int SegmentFootprint()
    {
        int count = 0;
        for (KpSegment? s = head; s != null; s = s.Next) count++;
        return count;
    }
}

internal sealed class InFlightOp
{
    public EnqueueToken Token;
    public long ModelIndex;
}

/// <summary>Reference FIFO indexed by reservation order: the model of what the queue must return.</summary>
internal sealed class ModelQueue
{
    private readonly List<int> slots = new List<int>();
    private readonly List<bool> written = new List<bool>();
    private int head;
    private long nextIndex;

    public long Reserve()
    {
        slots.Add(0);
        written.Add(false);
        return nextIndex++;
    }

    public void Complete(long idx, int value)
    {
        slots[(int)idx] = value;
        written[(int)idx] = true;
    }

    public bool ReservedAtHead => head < slots.Count;

    public bool TryDequeue(out int value)
    {
        value = 0;
        if (!ReservedAtHead || !written[head]) return false;
        value = slots[head];
        head++;
        return true;
    }

    public int Remaining()
    {
        int count = 0;
        for (int i = head; i < slots.Count; i++) if (written[i]) count++;
        return count;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static int NextSeed(int seed) => (seed * 1103515245 + 12345) & 0x7FFFFFFF;

    private static void SingleThreadedFifo()
    {
        var q = new KoganPetrankQueue(2);
        Check(q.TryDequeue(out _) == DequeueStatus.Empty, "empty dequeue on a fresh queue");
        for (int i = 1; i <= 10; i++) q.Enqueue(i * 10);
        var got = new List<int>();
        while (q.TryDequeue(out int v) == DequeueStatus.Ok) got.Add(v);
        Check(string.Join(",", got) == "10,20,30,40,50,60,70,80,90,100", "FIFO order, got " + string.Join(",", got));
        Check(q.TryDequeue(out _) == DequeueStatus.Empty, "empty again after draining");
        Check(q.EnqIdx == q.DeqIdx, "reservation and dequeue indices meet after draining");
    }

    private static void SegmentGrowthAndReuse()
    {
        var q = new KoganPetrankQueue(2);          // four slots per segment
        for (int i = 0; i < 10; i++) q.Enqueue(i);
        Check(q.SegmentsAllocated == 3, "ten items over four-slot segments need three segments, got " + q.SegmentsAllocated);
        Check(q.SegmentFootprint() == 3, "the chain has three live segments");
        var got = new List<int>();
        while (q.TryDequeue(out int v) == DequeueStatus.Ok) got.Add(v);
        Check(string.Join(",", got) == "0,1,2,3,4,5,6,7,8,9", "drained across segment boundaries");
        Check(q.SegmentFootprint() == 1, "the chain collapses to the last segment once drained");

        // Realign to a segment boundary, then run the steady state: four items in, four out, so
        // the queue empties exactly when the tail segment is exhausted and must be recycled.
        q.Enqueue(100);
        q.Enqueue(101);
        while (q.TryDequeue(out _) == DequeueStatus.Ok) { }
        Check(q.EnqIdx == q.DeqIdx && q.EnqIdx % q.Capacity == 0, "index realigned to a segment boundary");
        Check(q.SegmentsAllocated == 3, "still three segments before the steady state, got " + q.SegmentsAllocated);

        for (int round = 0; round < 25; round++)
        {
            for (int i = 0; i < q.Capacity; i++) q.Enqueue(round * 100 + i);
            for (int i = 0; i < q.Capacity; i++)
            {
                DequeueStatus status = q.TryDequeue(out int v);
                Check(status == DequeueStatus.Ok && v == round * 100 + i, "steady-state FIFO round " + round);
            }
        }
        Check(q.SegmentsReused == 25, "each drained segment was recycled: " + q.SegmentsReused);
        Check(q.SegmentsAllocated == 3, "footprint stayed bounded, allocated=" + q.SegmentsAllocated);
        Check(q.SegmentFootprint() == 1, "one live segment in steady state");
    }

    private static void PendingReservation()
    {
        var q = new KoganPetrankQueue(2);
        EnqueueToken token = q.Reserve();
        Check(token.Index == 0, "first reservation is index 0");
        Check(q.TryDequeue(out _) == DequeueStatus.Pending, "reserved but unwritten slot reads as pending");
        q.Complete(token, 42);
        Check((q.TryDequeue(out int v) == DequeueStatus.Ok) && v == 42, "the item appears once the write lands");
        Check(q.TryDequeue(out _) == DequeueStatus.Empty, "queue is empty afterwards");
    }

    private static void ReservationOrderBeatsWriteOrder()
    {
        var q = new KoganPetrankQueue(3);
        EnqueueToken first = q.Reserve();
        EnqueueToken second = q.Reserve();
        Check(second.Index == first.Index + 1, "reservations are contiguous");
        q.Complete(second, 200);                 // the later reservation writes first
        Check(q.TryDequeue(out _) == DequeueStatus.Pending, "dequeue waits for the earlier reservation");
        q.Complete(first, 100);
        Check((q.TryDequeue(out int a) == DequeueStatus.Ok) && a == 100, "FIFO follows reservation order, not write order");
        Check((q.TryDequeue(out int b) == DequeueStatus.Ok) && b == 200, "second item next");
        Check(q.TryDequeue(out _) == DequeueStatus.Empty, "empty after both");
    }

    private static void InterleavedFifoAgainstModel()
    {
        var q = new KoganPetrankQueue(2);
        var model = new ModelQueue();
        var unwritten = new List<InFlightOp>();
        int produced = 0, consumed = 0, pendingSeen = 0, emptySeen = 0;
        int seed = 31337;

        for (int step = 0; step < 4000; step++)
        {
            seed = NextSeed(seed);
            int choice = seed % 10;
            if (choice < 4)
            {
                var op = new InFlightOp { Token = q.Reserve(), ModelIndex = model.Reserve() };
                Check(op.Token.Index == op.ModelIndex, "reservation indices agree");
                unwritten.Add(op);
            }
            else if (choice < 7 && unwritten.Count > 0)
            {
                int pick = (seed / 7) % unwritten.Count;
                InFlightOp op = unwritten[pick];
                unwritten.RemoveAt(pick);
                int value = 7000 + produced++;
                q.Complete(op.Token, value);
                model.Complete(op.ModelIndex, value);
            }
            else
            {
                DequeueStatus status = q.TryDequeue(out int v);
                if (status == DequeueStatus.Ok)
                {
                    Check(model.TryDequeue(out int mv), "model has an item whenever the queue does");
                    Check(v == mv, "dequeued value matches the model: " + v + " vs " + mv);
                    consumed++;
                }
                else if (status == DequeueStatus.Pending)
                {
                    pendingSeen++;
                    Check(model.ReservedAtHead && !model.TryDequeue(out _), "model also reports an unwritten head");
                }
                else
                {
                    emptySeen++;
                    Check(!model.ReservedAtHead, "model also reports nothing reserved");
                }
            }
            Check(q.EnqIdx - q.DeqIdx == (long)unwritten.Count + model.Remaining(),
                "outstanding items are exactly reservations minus dequeues");
        }

        while (unwritten.Count > 0)
        {
            InFlightOp op = unwritten[0];
            unwritten.RemoveAt(0);
            int value = 7000 + produced++;
            q.Complete(op.Token, value);
            model.Complete(op.ModelIndex, value);
        }
        while (q.TryDequeue(out int v) == DequeueStatus.Ok)
        {
            Check(model.TryDequeue(out int mv) && mv == v, "drain matches the model");
            consumed++;
        }
        Check(consumed == produced, "every produced item was consumed: " + consumed + " of " + produced);
        Check(model.Remaining() == 0, "model drained");
        Check(pendingSeen > 0, "the interleaving really produced pending observations: " + pendingSeen);
        Check(emptySeen > 0, "the interleaving really produced empty observations: " + emptySeen);
        Check(q.TryDequeue(out _) == DequeueStatus.Empty, "pipe is empty at the end");
        Check(q.MaxOpSteps <= 4, "wait-free step bound per operation, max=" + q.MaxOpSteps);
        Console.WriteLine("kp queue: ops=" + q.Operations + " total-steps=" + q.TotalSteps
            + " max-steps=" + q.MaxOpSteps + " segments=" + q.SegmentsAllocated
            + " reused=" + q.SegmentsReused + " pending=" + pendingSeen + " empty=" + emptySeen);
    }

    private static void WaitFreedomBound()
    {
        var q = new KoganPetrankQueue(1);      // two slots per segment: worst case for walking
        var tokens = new List<EnqueueToken>();
        for (int i = 0; i < 40; i++) tokens.Add(q.Reserve());
        Check(tokens.Count == 40 && tokens[39].Index == 39, "forty contiguous reservations");
        Check(q.SegmentsAllocated >= 20, "two-slot segments were created on demand: " + q.SegmentsAllocated);
        for (int i = 39; i >= 0; i--) q.Complete(tokens[i], i);
        q.MaxOpSteps = 0;
        for (int i = 0; i < 40; i++)
        {
            Check((q.TryDequeue(out int v) == DequeueStatus.Ok) && v == i, "FIFO across twenty segments");
            Check(q.MaxOpSteps <= 4, "bounded steps even across a long chain, got " + q.MaxOpSteps);
        }
        Check(q.SegmentFootprint() <= 2, "footprint stays tiny while draining");
        Check(q.SegmentsReused == 0, "no reuse while the queue never fully drained");
    }

    public static void Main()
    {
        SingleThreadedFifo();
        SegmentGrowthAndReuse();
        PendingReservation();
        ReservationOrderBeatsWriteOrder();
        InterleavedFifoAgainstModel();
        WaitFreedomBound();
        Console.WriteLine("PASS");
    }
}
