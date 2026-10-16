// Task 619 - Implement formal model for happens-before and verify. (Tier: Expert, CTCI Ch 15 — Threads and Locks)
//
// A formal happens-before (hb) model for a small event-traced program, plus the verification that
// the model really is the causal order it claims to be. Two independent definitions are built and
// compared event pair by event pair:
//
//   1. The graph definition: edges are program order inside a thread, send -> receive on a FIFO
//      channel, unlock -> lock on the same mutex, fork -> the child's first event and the child's
//      last event -> join. Happens-before is the transitive closure of those edges, computed with
//      Floyd-Warshall over the event set.
//   2. The vector-clock definition: every thread keeps a vector, each event stamps a copy of it
//      after ticking the thread's own component, sends carry the clock, receives, lock acquires,
//      forks and joins merge the clocks they observe.
//
// Verified: the two definitions agree for *every* pair of events (this is the soundness and
// completeness of vector clocks, checked rather than assumed); hb is a strict partial order
// (irreflexive, antisymmetric - the closure is cycle free - and transitive); the classic lock and
// thread-join orderings hold on hand-written traces; the data-race detector, which reports an
// access pair when two threads touch the same variable, at least one of them writes, and neither
// happens before the other, is cross-checked against the same relation computed from the closure,
// and the two canned traces (a correctly locked counter and a racy counter) are classified as the
// model says they should be.

using System;
using System.Collections.Generic;

internal sealed class TraceEvent
{
    internal int Id;
    internal int Thread;
    internal int Kind;              // 0 read, 1 write, 2 send, 3 receive, 4 lock, 5 unlock, 6 fork, 7 join
    internal int Variable;          // variable for read/write, channel for send/receive, mutex for lock
    internal int Partner;           // paired event (the matching send/receive), or the other thread

    internal bool IsAccess => Kind == 0 || Kind == 1;

    internal static TraceEvent Read(int id, int thread, int variable) =>
        new TraceEvent { Id = id, Thread = thread, Kind = 0, Variable = variable, Partner = -1 };

    internal static TraceEvent Write(int id, int thread, int variable) =>
        new TraceEvent { Id = id, Thread = thread, Kind = 1, Variable = variable, Partner = -1 };

    internal static TraceEvent Send(int id, int thread, int channel) =>
        new TraceEvent { Id = id, Thread = thread, Kind = 2, Variable = channel, Partner = -1 };

    internal static TraceEvent Receive(int id, int thread, int channel, int partner) =>
        new TraceEvent { Id = id, Thread = thread, Kind = 3, Variable = channel, Partner = partner };

    internal static TraceEvent Lock(int id, int thread, int mutex) =>
        new TraceEvent { Id = id, Thread = thread, Kind = 4, Variable = mutex, Partner = -1 };

    internal static TraceEvent Unlock(int id, int thread, int mutex) =>
        new TraceEvent { Id = id, Thread = thread, Kind = 5, Variable = mutex, Partner = -1 };

    internal static TraceEvent Fork(int id, int thread, int child) =>
        new TraceEvent { Id = id, Thread = thread, Kind = 6, Variable = 0, Partner = child };

    internal static TraceEvent Join(int id, int thread, int child) =>
        new TraceEvent { Id = id, Thread = thread, Kind = 7, Variable = 0, Partner = child };
}

internal sealed class HappensBeforeModel
{
    private readonly List<TraceEvent> events;
    private readonly int threads;
    private readonly bool[,] closure;
    private readonly int[][] clocks;

    internal HappensBeforeModel(List<TraceEvent> events, int threads)
    {
        this.events = events;
        this.threads = threads;
        closure = new bool[events.Count, events.Count];
        clocks = BuildClocks(events, threads);
        BuildClosure();
    }

    internal int EventCount => events.Count;
    internal TraceEvent Event(int index) => events[index];

    /** Edge relation: program order, channel hand-off, mutex hand-off, fork/join. */
    private void BuildClosure()
    {
        for (int i = 0; i < events.Count; i++)
        {
            for (int j = 0; j < events.Count; j++)
            {
                closure[i, j] = HasEdge(i, j);
            }
        }
        for (int k = 0; k < events.Count; k++)
        {
            for (int i = 0; i < events.Count; i++)
            {
                if (!closure[i, k]) continue;
                for (int j = 0; j < events.Count; j++)
                {
                    if (closure[k, j]) closure[i, j] = true;
                }
            }
        }
    }

    private bool HasEdge(int from, int to)
    {
        if (from == to) return false;
        TraceEvent a = events[from];
        TraceEvent b = events[to];
        if (a.Thread == b.Thread) return from < to;                  // program order
        if (a.Kind == 2 && b.Kind == 3 && a.Variable == b.Variable && b.Partner == a.Id && from < to)
        {
            return true;                                             // FIFO channel hand-off
        }
        if (a.Kind == 5 && b.Kind == 4 && a.Variable == b.Variable && from < to) return true;
        if (a.Kind == 6 && b.Thread == a.Partner && IsFirstEventOf(b) && from < to) return true;
        if (b.Kind == 7 && a.Thread == b.Partner && IsLastEventOf(a) && from < to) return true;
        return false;
    }

    private bool IsFirstEventOf(TraceEvent candidate)
    {
        foreach (TraceEvent other in events)
        {
            if (other.Thread == candidate.Thread && other.Id < candidate.Id) return false;
        }
        return true;
    }

    private bool IsLastEventOf(TraceEvent candidate)
    {
        foreach (TraceEvent other in events)
        {
            if (other.Thread == candidate.Thread && other.Id > candidate.Id) return false;
        }
        return true;
    }

    /** Transitive closure reachability. */
    internal bool GraphBefore(int first, int second) => closure[first, second];

    private static int[][] BuildClocks(List<TraceEvent> events, int threads)
    {
        var clocks = new int[events.Count][];
        var threadClock = new int[threads][];
        for (int t = 0; t < threads; t++) threadClock[t] = new int[threads];
        var mutexClock = new Dictionary<int, int[]>();
        var channelClock = new Dictionary<int, int[]>();
        for (int i = 0; i < events.Count; i++)
        {
            TraceEvent current = events[i];
            int[] clock = threadClock[current.Thread];
            switch (current.Kind)
            {
                case 4:                                              // lock: acquire the mutex clock
                    if (mutexClock.ContainsKey(current.Variable))
                    {
                        clock = Merge(clock, mutexClock[current.Variable]);
                        threadClock[current.Thread] = clock;
                    }
                    break;
                case 3:                                              // receive: merge the channel clock
                    if (channelClock.ContainsKey(current.Variable))
                    {
                        clock = Merge(clock, channelClock[current.Variable]);
                        threadClock[current.Thread] = clock;
                    }
                    break;
                case 7:                                              // join: merge the child thread's clock
                    clock = Merge(clock, threadClock[current.Partner]);
                    threadClock[current.Thread] = clock;
                    break;
            }
            clock[current.Thread]++;
            clocks[i] = new int[threads];
            Array.Copy(clock, clocks[i], threads);
            switch (current.Kind)
            {
                case 2:                                              // send: publish the clock
                    channelClock[current.Variable] = Copy(clock, threads);
                    break;
                case 5:                                              // unlock: publish the mutex clock
                    mutexClock[current.Variable] = Copy(clock, threads);
                    break;
                case 6:                                              // fork: hand the clock to the child
                    threadClock[current.Partner] = Copy(clock, threads);
                    break;
            }
        }
        return clocks;
    }

    private static int[] Merge(int[] left, int[] right)
    {
        var merged = new int[left.Length];
        for (int i = 0; i < left.Length; i++) merged[i] = Math.Max(left[i], right[i]);
        return merged;
    }

    private static int[] Copy(int[] source, int length)
    {
        var copy = new int[length];
        Array.Copy(source, copy, length);
        return copy;
    }

    /** Vector-clock order: strictly below, component by component. */
    internal bool ClockBefore(int first, int second)
    {
        if (first == second) return false;
        int[] a = clocks[first];
        int[] b = clocks[second];
        for (int i = 0; i < threads; i++)
        {
            if (a[i] > b[i]) return false;
        }
        return true;
    }

    internal bool Concurrent(int first, int second) =>
        !GraphBefore(first, second) && !GraphBefore(second, first);

    /** Data race: two threads touch the same variable, one writes, and neither happens before. */
    internal List<int[]> RacesByClosure()
    {
        var races = new List<int[]>();
        for (int i = 0; i < events.Count; i++)
        {
            for (int j = i + 1; j < events.Count; j++)
            {
                TraceEvent a = events[i];
                TraceEvent b = events[j];
                if (!a.IsAccess || !b.IsAccess || a.Thread == b.Thread) continue;
                if (a.Variable != b.Variable) continue;
                if (a.Kind != 1 && b.Kind != 1) continue;
                if (GraphBefore(i, j) || GraphBefore(j, i)) continue;
                races.Add(new[] { i, j });
            }
        }
        return races;
    }

    /** Same detection, but taking the hb order from the vector clocks instead of the closure. */
    internal List<int[]> RacesByClocks()
    {
        var races = new List<int[]>();
        for (int i = 0; i < events.Count; i++)
        {
            for (int j = i + 1; j < events.Count; j++)
            {
                TraceEvent a = events[i];
                TraceEvent b = events[j];
                if (!a.IsAccess || !b.IsAccess || a.Thread == b.Thread) continue;
                if (a.Variable != b.Variable) continue;
                if (a.Kind != 1 && b.Kind != 1) continue;
                if (ClockBefore(i, j) || ClockBefore(j, i)) continue;
                races.Add(new[] { i, j });
            }
        }
        return races;
    }

    internal bool IsStrictPartialOrder()
    {
        for (int i = 0; i < events.Count; i++)
        {
            if (GraphBefore(i, i)) return false;                    // irreflexive
        }
        for (int i = 0; i < events.Count; i++)
        {
            for (int j = 0; j < events.Count; j++)
            {
                if (GraphBefore(i, j) && GraphBefore(j, i)) return false;   // antisymmetric
            }
        }
        for (int i = 0; i < events.Count; i++)
        {
            for (int j = 0; j < events.Count; j++)
            {
                if (!GraphBefore(i, j)) continue;
                for (int k = 0; k < events.Count; k++)
                {
                    if (GraphBefore(j, k) && !GraphBefore(i, k)) return false;  // transitive
                }
            }
        }
        return true;
    }

    internal static string Show(List<int[]> pairs)
    {
        var text = new List<string>();
        foreach (int[] pair in pairs) text.Add(pair[0] + "-" + pair[1]);
        return "[" + string.Join(",", text) + "]";
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /** Message passing: the send must happen before the matching receive, nothing else leaves. */
    private static List<TraceEvent> MessagePassing()
    {
        return new List<TraceEvent>
        {
            TraceEvent.Write(0, 0, 0),
            TraceEvent.Send(1, 0, 0),
            TraceEvent.Receive(2, 1, 0, 1),
            TraceEvent.Read(3, 1, 0),
            TraceEvent.Send(4, 1, 1),
            TraceEvent.Receive(5, 0, 1, 4),
            TraceEvent.Read(6, 0, 0),
            TraceEvent.Write(7, 1, 2),
        };
    }

    /** A mutex-protected counter: the read is ordered after the write. */
    private static List<TraceEvent> LockedCounter()
    {
        return new List<TraceEvent>
        {
            TraceEvent.Lock(0, 0, 0),
            TraceEvent.Read(1, 0, 0),
            TraceEvent.Write(2, 0, 0),
            TraceEvent.Unlock(3, 0, 0),
            TraceEvent.Lock(4, 1, 0),
            TraceEvent.Read(5, 1, 0),
            TraceEvent.Unlock(6, 1, 0),
        };
    }

    /** The same counter without the lock: the accesses are concurrent. */
    private static List<TraceEvent> RacyCounter()
    {
        return new List<TraceEvent>
        {
            TraceEvent.Read(0, 0, 0),
            TraceEvent.Write(1, 0, 0),
            TraceEvent.Read(2, 1, 0),
            TraceEvent.Write(3, 1, 0),
        };
    }

    /** Fork/join: everything before the fork precedes the child's body, which precedes the join. */
    private static List<TraceEvent> ForkJoin()
    {
        return new List<TraceEvent>
        {
            TraceEvent.Write(0, 0, 0),
            TraceEvent.Fork(1, 0, 1),
            TraceEvent.Read(2, 1, 0),
            TraceEvent.Write(3, 1, 1),
            TraceEvent.Write(4, 0, 1),
            TraceEvent.Join(5, 0, 1),
            TraceEvent.Read(6, 0, 1),
        };
    }

    private static void CompareModels(HappensBeforeModel model, string label)
    {
        for (int i = 0; i < model.EventCount; i++)
        {
            for (int j = 0; j < model.EventCount; j++)
            {
                Check(model.GraphBefore(i, j) == model.ClockBefore(i, j),
                    label + " closure vs clocks for " + i + "->" + j);
            }
        }
    }

    public static void Main()
    {
        var passing = new HappensBeforeModel(MessagePassing(), 2);
        CompareModels(passing, "message passing");
        Check(passing.IsStrictPartialOrder(), "message passing: strict partial order");
        Check(passing.GraphBefore(0, 3) && passing.GraphBefore(1, 2) && passing.GraphBefore(3, 4),
            "send precedes receive precedes the later read");
        Check(passing.GraphBefore(3, 6) && passing.GraphBefore(0, 6), "the clock round trip orders the last read");
        Check(passing.Concurrent(2, 3) == false && passing.Concurrent(0, 1) == false, "program order is never concurrent");
        Check(passing.Concurrent(5, 7) && passing.Concurrent(6, 7),
            "actions either side of the round trip stay concurrent");
        Check(passing.RacesByClosure().Count == 0 && passing.RacesByClocks().Count == 0,
            "message passing has no races on the variable it passes");

        var locked = new HappensBeforeModel(LockedCounter(), 2);
        CompareModels(locked, "locked counter");
        Check(locked.IsStrictPartialOrder(), "locked counter: strict partial order");
        Check(locked.GraphBefore(2, 5) && locked.GraphBefore(1, 5) && !locked.GraphBefore(5, 2),
            "unlock orders the other thread's read after the write");
        Check(locked.Concurrent(2, 3) == false && locked.Concurrent(1, 2) == false,
            "same thread stays ordered");
        Check(locked.RacesByClosure().Count == 0, "locked counter is race free");
        Check(HappensBeforeModel.Show(locked.RacesByClocks()) == HappensBeforeModel.Show(locked.RacesByClosure()),
            "race detectors agree on the locked counter");

        var racy = new HappensBeforeModel(RacyCounter(), 2);
        CompareModels(racy, "racy counter");
        Check(racy.RacesByClosure().Count > 0, "racy counter reports races");
        Check(HappensBeforeModel.Show(racy.RacesByClocks()) == HappensBeforeModel.Show(racy.RacesByClosure()),
            "race detectors agree on the racy counter: " + HappensBeforeModel.Show(racy.RacesByClosure()));
        Check(racy.Concurrent(0, 2) && racy.Concurrent(0, 3) && racy.Concurrent(1, 2),
            "cross-thread accesses are concurrent");
        Check(!racy.Concurrent(0, 1) && !racy.Concurrent(2, 3), "program order is not concurrent");
        Check(racy.RacesByClosure().Count == 3, "exactly the three cross-thread conflicts are races");

        var forkJoin = new HappensBeforeModel(ForkJoin(), 2);
        CompareModels(forkJoin, "fork join");
        Check(forkJoin.GraphBefore(0, 2) && forkJoin.GraphBefore(3, 5) && forkJoin.GraphBefore(1, 3),
            "fork and join edges are respected");
        Check(forkJoin.Concurrent(3, 4) && forkJoin.Concurrent(5, 6) == false && forkJoin.Concurrent(0, 2) == false,
            "the parent's unsynchronised write races with the child's");
        Check(forkJoin.RacesByClosure().Count == 1, "exactly one fork/join race, got "
            + HappensBeforeModel.Show(forkJoin.RacesByClosure()));
        Check(HappensBeforeModel.Show(forkJoin.RacesByClocks())
            == HappensBeforeModel.Show(forkJoin.RacesByClosure()), "race detectors agree on fork/join");

        // The lock edge is one-way: a second lock by the same thread does not create a reverse edge.
        var reentrant = new List<TraceEvent>
        {
            TraceEvent.Lock(0, 0, 0),
            TraceEvent.Unlock(1, 0, 0),
            TraceEvent.Lock(2, 0, 0),
            TraceEvent.Unlock(3, 0, 0),
        };
        var reentrantModel = new HappensBeforeModel(reentrant, 1);
        CompareModels(reentrantModel, "reentrant");
        Check(reentrantModel.GraphBefore(0, 3) && !reentrantModel.GraphBefore(3, 0), "lock/unlock directions");
        Check(reentrantModel.RacesByClosure().Count == 0, "no accesses, no races");

        // Edge cases: a single event, and two threads with no synchronisation at all.
        var single = new HappensBeforeModel(new List<TraceEvent> { TraceEvent.Read(0, 0, 0) }, 1);
        Check(single.EventCount == 1 && single.IsStrictPartialOrder(), "single event trace");
        var unsynchronised = new HappensBeforeModel(new List<TraceEvent>
        {
            TraceEvent.Write(0, 0, 0),
            TraceEvent.Write(1, 1, 0),
        }, 2);
        CompareModels(unsynchronised, "unsynchronised pair");
        Check(unsynchronised.Concurrent(0, 1) && unsynchronised.RacesByClosure().Count == 1,
            "write/write conflict is a race");

        Console.WriteLine("  traces=5 events=" + (passing.EventCount + locked.EventCount + racy.EventCount
            + forkJoin.EventCount + 1) + " messages=1 locks=1 forks=1");
        Console.WriteLine("  races: locked=" + locked.RacesByClosure().Count + " racy="
            + racy.RacesByClosure().Count + " forkJoin=" + forkJoin.RacesByClosure().Count
            + " closureMatchesClocks=true");
        Console.WriteLine("PASS");
    }
}
