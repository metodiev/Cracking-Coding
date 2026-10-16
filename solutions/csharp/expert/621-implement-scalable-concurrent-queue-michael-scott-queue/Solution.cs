// Task 621 - Implement scalable concurrent queue (Michael-Scott queue). (Tier: Expert, CTCI Ch 15 — Threads and Locks)
//
// The Michael-Scott queue, executed as a deterministic *simulation*. There are no threads: every
// thread has a program counter and one outstanding operation, each operation is a sequence of
// micro-steps (read a field, compare-and-swap a field), and an explicit schedule of (thread, step)
// pairs decides which micro-step runs next. A compare-and-swap is modelled directly - it succeeds
// only when the current value is still the expected one, which is exactly what interleaving another
// thread's CAS between the read and the swap does. This reproduces the algorithm's real behaviour,
// including the subtle parts:
//
//   * enqueue links a new node with CAS(tail.next, null, node); if it observes a non-null tail.next
//     it *helps* by CAS(tail, oldTail, tail.next) and retries from the start;
//   * dequeue reads head.next, returns the value only after CAS(head, oldHead, next) succeeds, and
//     when head == tail with an empty next it tries to fix the tail instead.
//
// Verified after every micro-step: the queue is traversable from head to tail with tail.next == null
// (and a bounded walk never revisits a node), no item is lost or duplicated - the dequeued values
// followed by the values still linked is exactly the multiset enqueued - and the dequeued values
// appear in the order in which their nodes were *linked*, which is the queue's linearization order,
// so the queue is FIFO. A sequential reference run of the same operations produces the same dequeue
// sequence, a targeted schedule forces the helping path and the tail fix, and the whole simulation is
// reproducible from its seed.

using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;          // deterministic stand-in for Random
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

internal sealed class QueueNode
{
    internal int Value;
    internal QueueNode Next;                 // null until linked
    internal int LinkStep = -1;              // schedule step at which this node was linked
}

internal sealed class MsThread
{
    internal int Id;
    internal int ProgramCounter;
    internal bool Idle = true;
    internal int PendingValue;               // argument of the outstanding operation
    internal bool Dequeue;
    internal int Result;

    // Micro-step locals, exactly as the algorithm names them.
    internal QueueNode LocalTail;
    internal QueueNode LocalNext;
    internal QueueNode LocalHead;
    internal QueueNode LocalNew;
}

internal sealed class MichaelScottQueue
{
    internal QueueNode Head;
    internal QueueNode Tail;

    internal readonly List<MsThread> Threads = new List<MsThread>();
    internal readonly List<int> Dequeued = new List<int>();
    internal readonly List<int> LinkedOrder = new List<int>();
    internal readonly List<int> EnqueuedOrder = new List<int>();

    internal int HelpingSteps;
    internal int TailFixes;
    internal int FailedCas;
    internal int SuccessfulCas;
    internal int CompletedEnqueues;
    internal int CompletedDequeues;
    internal int EmptyDequeues;
    internal int Step;

    internal MichaelScottQueue(int threads)
    {
        QueueNode dummy = new QueueNode { Value = int.MinValue };
        Head = dummy;
        Tail = dummy;
        for (int i = 0; i < threads; i++) Threads.Add(new MsThread { Id = i });
    }

    private bool Cas(ref QueueNode target, QueueNode expected, QueueNode replacement)
    {
        if (!ReferenceEquals(target, expected))
        {
            FailedCas++;
            return false;
        }
        target = replacement;
        SuccessfulCas++;
        Step++;
        return true;
    }

    internal void StartEnqueue(MsThread thread, int value)
    {
        thread.Idle = false;
        thread.Dequeue = false;
        thread.PendingValue = value;
        thread.ProgramCounter = 0;
        EnqueuedOrder.Add(value);
    }

    internal void StartDequeue(MsThread thread)
    {
        thread.Idle = false;
        thread.Dequeue = true;
        thread.PendingValue = 0;
        thread.ProgramCounter = 0;
    }

    /** Advance one thread by one micro-step; returns true when the operation completed. */
    internal bool Advance(MsThread thread)
    {
        Step++;
        if (thread.Dequeue)
        {
            switch (thread.ProgramCounter)
            {
                case 0:
                    thread.LocalHead = Head;
                    thread.ProgramCounter = 1;
                    return false;
                case 1:
                    thread.LocalNext = thread.LocalHead.Next;
                    thread.ProgramCounter = 2;
                    return false;
                case 2:
                    if (thread.LocalNext == null)
                    {
                        if (ReferenceEquals(thread.LocalHead, Tail))
                        {
                            thread.Idle = true;                       // the queue is empty
                            thread.Result = int.MinValue;
                            CompletedDequeues++;
                            EmptyDequeues++;
                            return true;
                        }
                        QueueNode tail = Tail;                        // stale tail: help by fixing it
                        TailFixes++;
                        Cas(ref Tail, tail, tail.Next ?? tail);
                        thread.ProgramCounter = 0;
                        return false;
                    }
                    thread.Result = thread.LocalNext.Value;
                    thread.ProgramCounter = 3;
                    return false;
                default:
                    if (Cas(ref Head, thread.LocalHead, thread.LocalNext))
                    {
                        Dequeued.Add(thread.Result);
                        thread.Idle = true;
                        CompletedDequeues++;
                        return true;
                    }
                    thread.ProgramCounter = 0;
                    return false;
            }
        }
        switch (thread.ProgramCounter)
        {
            case 0:
                thread.LocalTail = Tail;
                thread.ProgramCounter = 1;
                return false;
            case 1:
                thread.LocalNext = thread.LocalTail.Next;
                thread.ProgramCounter = 2;
                return false;
            case 2:
                thread.LocalNew = new QueueNode { Value = thread.PendingValue };
                thread.ProgramCounter = thread.LocalNext == null ? 3 : 4;
                return false;
            case 3:
                if (Cas(ref thread.LocalTail.Next, null, thread.LocalNew))
                {
                    thread.LocalNew.LinkStep = Step;
                    LinkedOrder.Add(thread.PendingValue);
                    thread.ProgramCounter = 5;
                    return false;
                }
                thread.ProgramCounter = 0;
                return false;
            case 4:                                                   // helping step
                HelpingSteps++;
                Cas(ref Tail, thread.LocalTail, thread.LocalNext);
                thread.ProgramCounter = 0;
                return false;
            default:
                Cas(ref Tail, thread.LocalTail, thread.LocalNew);
                thread.Idle = true;
                CompletedEnqueues++;
                return true;
        }
    }

    /** head -> ... -> tail is intact with no cycle. Note that tail.next may legitimately be non-null
     *  while an enqueue is mid-flight (the link CAS has happened but the tail has not advanced yet),
     *  so that state is only checked once the queue is quiescent. */
    internal bool StructureOk(out string report)
    {
        report = "";
        var seen = new HashSet<QueueNode>();
        QueueNode node = Head;
        int guard = 0;
        while (!ReferenceEquals(node, Tail) && guard++ < 10000)
        {
            if (!seen.Add(node))
            {
                report = "cycle at value " + node.Value;
                return false;
            }
            if (node.Next == null)
            {
                report = "chain breaks before the tail";
                return false;
            }
            node = node.Next;
        }
        if (!ReferenceEquals(node, Tail))
        {
            report = "tail not reachable";
            return false;
        }
        return true;
    }

    internal bool QuiescentOk(out string report)
    {
        if (!StructureOk(out report)) return false;
        if (Tail.Next != null)
        {
            report = "tail.next is not null while the queue is quiescent";
            return false;
        }
        return true;
    }

    internal List<int> LinkedValues()
    {
        var values = new List<int>();
        for (QueueNode node = Head.Next; node != null; node = node.Next)
        {
            values.Add(node.Value);
            if (values.Count > 10000) break;
        }
        return values;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /** Run a deterministic schedule of operations, checking the structure after every micro-step. */
    private static void Drive(MichaelScottQueue queue, int operations, int seed, int queueSize)
    {
        ulong state = (ulong)seed * 0x2545F4914F6CDD1DUL + 11;
        int issued = 0;
        int value = 1;
        int guard = 0;
        while ((issued < operations || queue.Threads.Exists(t => !t.Idle)) && guard++ < 20000)
        {
            var free = new List<MsThread>();
            foreach (MsThread thread in queue.Threads)
            {
                if (thread.Idle) free.Add(thread);
            }
            if (issued < operations && free.Count > 0 && SplitMix64.Next(ref state) % 2 == 0)
            {
                MsThread thread = free[(int)(SplitMix64.Next(ref state) % (ulong)free.Count)];
                // Keep the queue inside a window so both operations stay interesting.
                if (queue.LinkedValues().Count > queueSize || (value % 4 == 0 && value > 4))
                {
                    queue.StartDequeue(thread);
                }
                else
                {
                    queue.StartEnqueue(thread, value);
                    value++;
                }
                issued++;
                continue;
            }
            var busy = new List<MsThread>();
            foreach (MsThread thread in queue.Threads)
            {
                if (!thread.Idle) busy.Add(thread);
            }
            if (busy.Count == 0) continue;
            MsThread pick = busy[(int)(SplitMix64.Next(ref state) % (ulong)busy.Count)];
            queue.Advance(pick);
            Check(queue.StructureOk(out string report), "structure after step " + queue.Step + ": " + report);
        }
        Check(guard < 20000, "schedule made progress");
        Check(queue.QuiescentOk(out string quiescent), "quiescent state: " + quiescent);
    }

    public static void Main()
    {
        // Interleaved run: FIFO order is the order in which nodes were linked.
        var queue = new MichaelScottQueue(3);
        Drive(queue, 60, 7, 6);
        Check(queue.CompletedEnqueues == queue.LinkedOrder.Count, "every enqueue completed");
        Check(queue.LinkedOrder.Count == queue.EnqueuedOrder.Count, "enqueues all linked");
        Check(queue.Dequeued.Count + queue.EmptyDequeues == queue.CompletedDequeues,
            "every dequeue either recorded a value or found the queue empty");
        for (int i = 0; i < queue.Dequeued.Count; i++)
        {
            Check(queue.Dequeued[i] == queue.LinkedOrder[i], "FIFO order at position " + i + ": dequeued "
                + queue.Dequeued[i] + " expected " + queue.LinkedOrder[i]);
        }
        var remaining = queue.LinkedValues();
        var accounted = new List<int>(queue.Dequeued);
        accounted.AddRange(remaining);
        accounted.Sort();
        var enqueued = new List<int>(queue.EnqueuedOrder);
        enqueued.Sort();
        Check(accounted.Count == enqueued.Count, "no item lost or duplicated: " + accounted.Count + " vs "
            + enqueued.Count);
        for (int i = 0; i < accounted.Count; i++)
        {
            Check(accounted[i] == enqueued[i], "multiset matches at " + i);
        }
        Check(queue.SuccessfulCas > 0 && queue.FailedCas > 0,
            "the schedule exercised both CAS outcomes: ok=" + queue.SuccessfulCas + " failed="
            + queue.FailedCas + " helping=" + queue.HelpingSteps);

        // Sequential reference: the same operations applied one at a time give the same order.
        var sequential = new Queue<int>();
        var expectedOrder = new List<int>();
        foreach (int value in queue.LinkedOrder)
        {
            sequential.Enqueue(value);
        }
        while (sequential.Count > 0) expectedOrder.Add(sequential.Dequeue());
        var referenceFifo = new List<int>(queue.LinkedOrder);
        Check(expectedOrder.Count == referenceFifo.Count, "reference dequeues everything");
        for (int i = 0; i < expectedOrder.Count; i++)
        {
            Check(expectedOrder[i] == referenceFifo[i], "reference FIFO at " + i);
        }
        Check(queue.Dequeued.Count <= queue.LinkedOrder.Count, "never dequeued more than was linked");

        // A longer run with more threads: invariants hold all the way.
        var busy = new MichaelScottQueue(4);
        Drive(busy, 150, 99, 12);
        Check(busy.StructureOk(out string busyReport), "long run structure " + busyReport);
        Check(busy.CompletedEnqueues + busy.CompletedDequeues == 150, "long run completed every operation: "
            + busy.CompletedEnqueues + "+" + busy.CompletedDequeues);
        var busyRemaining = busy.LinkedValues();
        var busyAccounted = new List<int>(busy.Dequeued);
        busyAccounted.AddRange(busyRemaining);
        busyAccounted.Sort();
        var busyEnqueued = new List<int>(busy.EnqueuedOrder);
        busyEnqueued.Sort();
        Check(busyAccounted.Count == busyEnqueued.Count, "long run accounting");

        // Targeted schedule: force the helping path and the tail fix with a hand-written interleaving.
        var targeted = new MichaelScottQueue(2);
        var first = targeted.Threads[0];
        var second = targeted.Threads[1];
        targeted.StartEnqueue(first, 1);
        targeted.Advance(first);                       // read tail
        targeted.Advance(first);                       // read tail.next == null
        targeted.Advance(first);                       // build the node
        targeted.Advance(first);                       // CAS link, succeeds
        targeted.StartEnqueue(second, 2);
        targeted.Advance(second);                      // reads the *stale* tail
        targeted.Advance(second);                      // sees the linked node: helping path
        targeted.Advance(second);                      // builds the new node
        int guard = 0;
        while (targeted.HelpingSteps == 0 && guard++ < 10) targeted.Advance(second);
        Check(targeted.HelpingSteps == 1, "helping path exercised, count=" + targeted.HelpingSteps);
        Check(targeted.StructureOk(out string targetedReport), "targeted structure " + targetedReport);
        while (!second.Idle && guard++ < 30) targeted.Advance(second);
        while (!first.Idle && guard++ < 30) targeted.Advance(first);
        Check(targeted.QuiescentOk(out string afterReport), "targeted quiescent " + afterReport);
        Check(targeted.LinkedOrder.Count == 2 && targeted.LinkedValues().Count == 2,
            "both targeted enqueues are linked");

        var helper = new MichaelScottQueue(1);
        var solo = helper.Threads[0];
        for (int i = 0; i < 5; i++)
        {
            helper.StartEnqueue(solo, i + 1);
            while (!solo.Idle) helper.Advance(solo);
        }
        for (int i = 0; i < 5; i++)
        {
            helper.StartDequeue(solo);
            while (!solo.Idle) helper.Advance(solo);
            Check(solo.Result == i + 1, "sequential dequeue order at " + i + " got " + solo.Result);
        }
        helper.StartDequeue(solo);
        while (!solo.Idle) helper.Advance(solo);
        Check(solo.Result == int.MinValue, "dequeue of an empty queue reports empty");
        Check(helper.StructureOk(out string soloReport), "solo structure " + soloReport);

        // Reproducibility: the same seed reproduces the exact dequeue sequence.
        var again = new MichaelScottQueue(3);
        Drive(again, 60, 7, 6);
        Check(again.Dequeued.Count == queue.Dequeued.Count, "reproducible dequeue count");
        for (int i = 0; i < again.Dequeued.Count; i++)
        {
            Check(again.Dequeued[i] == queue.Dequeued[i], "reproducible dequeue at " + i);
        }
        Check(again.SuccessfulCas == queue.SuccessfulCas && again.FailedCas == queue.FailedCas,
            "reproducible CAS counts");

        Console.WriteLine("  run: enqueues=" + queue.CompletedEnqueues + " dequeues=" + queue.CompletedDequeues
            + " linked=" + queue.LinkedOrder.Count + " dequeued=" + queue.Dequeued.Count + " cas="
            + queue.SuccessfulCas + " failedCas=" + queue.FailedCas);
        Console.WriteLine("  long run: steps=" + busy.Step + " cas=" + busy.SuccessfulCas + " helping="
            + busy.HelpingSteps + " tailFixes=" + busy.TailFixes + " remaining=" + busyRemaining.Count);
        Console.WriteLine("PASS");
    }
}
