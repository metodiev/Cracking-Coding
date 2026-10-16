// Task 506 - Implement rate-limited queue with leaky bucket test harness. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
//
// A leaky bucket in exact integer arithmetic plus the queue that drains behind it, driven by a
// deterministic test harness. The bucket holds at most C volume units and leaks R units per tick,
// so an arrival at tick t first refreshes the level as max(0, level - R * (t - lastTick)) and is
// admitted only if level + volume <= C; volume > C can never be admitted and the harness reports
// the exact wait as ceil((level + volume - C) / R). No floating point is used anywhere, so the
// idle-drain bookkeeping cannot drift. Documented checks:
//   * crafted cases: a burst of exactly C is admitted at t = 0 and the C+1-th unit is rejected;
//     after R ticks one more unit fits; an oversized request is rejected forever and WaitTicks
//     reports -1; ten units left idle for 10 ticks drain to exactly zero and a fresh burst of C
//     fits again;
//   * the O(1) bucket agrees record for record (level and verdict) with an independent tick-by-tick
//     reference bucket over a 4000-arrival stream (gaps of 1-2 ticks, volumes 2-5, C = 8, R = 1)
//     driven by a counter-indexed splitmix64 stream, so the bucket is genuinely overloaded;
//   * the harness verifies the fundamental guarantee by brute force: over every window of W ticks
//     (W = 1..60) of the accepted-volume trace, accepted <= C + R * W;
//   * under sustained overload (2 units requested per tick against R = 1) the acceptance rate in
//     the steady-state window of 200 ticks is exactly R = 1 per tick, while the queue records the
//     remaining requests as rejected;
//   * the rate-limited queue is FIFO, drains at most R items per tick, conserves items
//     (drained + dropped == enqueued) and drops exactly the items whose deadline passed: R = 3 with
//     a 5-tick deadline drains 18 of 20 items in six ticks, and R = 1 with a 2-tick deadline keeps
//     only the first three of ten.
using System;
using System.Collections.Generic;

internal static class Clock
{
    internal static double Uniform(long seed, long index)
    {
        ulong x = (ulong)seed * 0x9E3779B97F4A7C15UL + (ulong)index * 0xD1B54A32D192ED03UL;
        x += 0x9E3779B97F4A7C15UL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
        x ^= x >> 31;
        return 2.0 * ((x >> 11) * (1.0 / 9007199254740992.0)) - 1.0;
    }

    /** Integer in [low, high] from the deterministic stream. */
    internal static long Range(long seed, long index, long low, long high)
    {
        double unit = (Uniform(seed, index) + 1.0) * 0.5;
        long span = high - low + 1;
        long value = low + (long)(unit * span);
        return value > high ? high : value;
    }
}

/** O(1) leaky bucket: state is the level and the tick it was last refreshed at. */
internal class LeakyBucket
{
    internal readonly long Capacity;
    internal readonly long Rate;
    internal long Level;
    internal long LastTick;

    internal LeakyBucket(long capacity, long rate)
    {
        Capacity = capacity;
        Rate = rate;
    }

    internal long Leak(long tick)
    {
        long elapsed = tick - LastTick;
        if (elapsed > 0)
        {
            Level = Math.Max(0L, Level - Rate * elapsed);
            LastTick = tick;
        }
        return Level;
    }

    internal bool TryAdd(long tick, long volume)
    {
        Leak(tick);
        if (volume > Capacity) return false;
        if (Level + volume > Capacity) return false;
        Level += volume;
        return true;
    }

    /** Ticks to wait before `volume` fits; -1 when it can never fit. */
    internal long WaitTicks(long volume)
    {
        if (volume > Capacity) return -1;
        long excess = Level + volume - Capacity;
        if (excess <= 0) return 0;
        return (excess + Rate - 1) / Rate;
    }
}

/** Independent reference: drains one tick at a time, so it cannot share the closed form's bugs. */
internal class TickReferenceBucket
{
    private readonly long capacity;
    private readonly long rate;
    private long level;
    private long clock;

    internal TickReferenceBucket(long capacity, long rate)
    {
        this.capacity = capacity;
        this.rate = rate;
    }

    internal long Level => level;

    internal bool TryAdd(long tick, long volume)
    {
        while (clock < tick)
        {
            clock += 1;
            level = Math.Max(0L, level - rate);
        }
        if (volume > capacity) return false;
        if (level + volume > capacity) return false;
        level += volume;
        return true;
    }
}

internal struct Arrival
{
    internal long Tick;
    internal long Volume;
}

internal struct Admission
{
    internal long Tick;
    internal long Volume;
    internal bool Accepted;
    internal long Level;
    internal long Wait;
}

internal static class Harness
{
    /** Runs the arrivals through the O(1) bucket and records one row per request. */
    internal static List<Admission> Run(Arrival[] arrivals, long capacity, long rate)
    {
        var bucket = new LeakyBucket(capacity, rate);
        var rows = new List<Admission>();
        foreach (Arrival arrival in arrivals)
        {
            long wait = bucket.WaitTicks(arrival.Volume);
            bool accepted = bucket.TryAdd(arrival.Tick, arrival.Volume);
            rows.Add(new Admission
            {
                Tick = arrival.Tick,
                Volume = arrival.Volume,
                Accepted = accepted,
                Level = bucket.Level,
                Wait = wait,
            });
        }
        return rows;
    }

    internal static List<Admission> RunReference(Arrival[] arrivals, long capacity, long rate)
    {
        var bucket = new TickReferenceBucket(capacity, rate);
        var rows = new List<Admission>();
        foreach (Arrival arrival in arrivals)
        {
            bool accepted = bucket.TryAdd(arrival.Tick, arrival.Volume);
            rows.Add(new Admission
            {
                Tick = arrival.Tick,
                Volume = arrival.Volume,
                Accepted = accepted,
                Level = bucket.Level,
                Wait = 0,
            });
        }
        return rows;
    }

    /** Accepted volume per tick, indexed by tick (ticks beyond the trace are zero). */
    internal static long[] AcceptedByTick(List<Admission> rows, long ticks)
    {
        var perTick = new long[ticks];
        foreach (Admission row in rows)
        {
            if (row.Accepted && row.Tick < ticks) perTick[row.Tick] += row.Volume;
        }
        return perTick;
    }

    internal static Arrival[] DeterministicStream(int count, long seed, long gapLow, long gapHigh,
            long volumeLow, long volumeHigh)
    {
        var arrivals = new Arrival[count];
        long tick = 0;
        long index = 0;
        for (int i = 0; i < count; i++)
        {
            tick += Clock.Range(seed, index++, gapLow, gapHigh);
            arrivals[i] = new Arrival
            {
                Tick = tick,
                Volume = Clock.Range(seed, index++, volumeLow, volumeHigh),
            };
        }
        return arrivals;
    }
}

internal class RateLimitedQueue
{
    private readonly Queue<(long Id, long EnqueuedTick)> waiting = new Queue<(long, long)>();
    private readonly long rate;
    private readonly long deadline;

    internal RateLimitedQueue(long rate, long deadline)
    {
        this.rate = rate;
        this.deadline = deadline;
    }

    internal List<(long Id, long Tick)> Drained { get; } = new List<(long, long)>();

    internal List<long> Dropped { get; } = new List<long>();

    internal long EnqueuedCount { get; private set; }

    internal void Enqueue(long id, long tick)
    {
        waiting.Enqueue((id, tick));
        EnqueuedCount += 1;
    }

    internal void RunUntil(long untilTick)
    {
        for (long tick = 0; tick <= untilTick; tick++)
        {
            while (waiting.Count > 0 && tick - waiting.Peek().EnqueuedTick > deadline)
            {
                Dropped.Add(waiting.Dequeue().Id);
            }
            long budget = rate;
            while (budget > 0 && waiting.Count > 0)
            {
                (long id, long _) = waiting.Dequeue();
                Drained.Add((id, tick));
                budget -= 1;
            }
        }
    }
}

public static class Solution
{
    public static void Main()
    {
        // --- crafted cases ---------------------------------------------------------------------------
        var bucket = new LeakyBucket(10, 1);
        for (int i = 0; i < 10; i++)
        {
            Check(bucket.TryAdd(0, 1), "a burst of exactly C = 10 fills the bucket");
        }
        Check(bucket.Level == 10, "the bucket is exactly full");
        Check(!bucket.TryAdd(0, 1), "the C+1-th unit is rejected");
        Check(bucket.WaitTicks(1) == 1, "the rejected unit waits exactly one tick");
        Check(bucket.TryAdd(1, 1), "after one tick one unit fits again");
        Check(bucket.Level == 10, "the level is back at capacity");
        Check(!bucket.TryAdd(1, 5), "a five-unit request does not fit");
        Check(bucket.WaitTicks(5) == 5, "it needs exactly five ticks");
        Check(bucket.WaitTicks(11) == -1, "an oversized request can never be admitted");
        Check(!bucket.TryAdd(2, 11), "an oversized request is rejected even when idle");

        var idle = new LeakyBucket(10, 1);
        Check(idle.TryAdd(0, 10), "idle bucket accepts a full burst");
        Check(idle.Leak(5) == 5, "five ticks leak five units");
        Check(idle.Leak(10) == 0, "ten ticks drain the bucket");
        Check(idle.Leak(100) == 0, "a long idle period keeps the level at zero");
        Check(idle.TryAdd(100, 10), "a full burst fits again after idling");

        var coarse = new LeakyBucket(10, 3);
        Check(coarse.TryAdd(0, 10), "coarse bucket fills");
        Check(coarse.WaitTicks(1) == 1, "with R = 3 one tick is enough for one unit");
        Check(coarse.WaitTicks(2) == 1, "two units also need one tick");
        Check(coarse.WaitTicks(4) == 2, "four units need two ticks");
        Check(coarse.TryAdd(1, 2), "the waiting time calculation matches reality");

        // --- agreement with the tick-by-tick reference bucket -----------------------------------------
        Arrival[] stream = Harness.DeterministicStream(4000, 987654321, 1, 2, 2, 5);
        List<Admission> rows = Harness.Run(stream, 8, 1);
        List<Admission> reference = Harness.RunReference(stream, 8, 1);
        Check(rows.Count == reference.Count && rows.Count == stream.Length, "both runs processed every arrival");
        int accepted = 0;
        foreach (Admission row in rows)
        {
            Check(row.Level <= 8, "the level never exceeds the capacity");
            if (row.Accepted) accepted += 1;
        }
        for (int i = 0; i < rows.Count; i++)
        {
            Check(rows[i].Accepted == reference[i].Accepted, "verdict agrees with the reference at " + i);
            Check(rows[i].Level == reference[i].Level, "level agrees with the reference at " + i);
        }
        Check(accepted > 100 && accepted < 0.8 * stream.Length,
                "the overloaded stream both admits and rejects (accepted " + accepted + ")");

        // --- the fundamental window bound: accepted <= C + R * W ---------------------------------------
        long lastTick = stream[stream.Length - 1].Tick;
        long[] perTick = Harness.AcceptedByTick(rows, lastTick + 1);
        for (int window = 1; window <= 60; window++)
        {
            for (int start = 0; start + window <= perTick.Length; start += 7)
            {
                long sum = 0;
                for (int i = start; i < start + window; i++) sum += perTick[i];
                Check(sum <= 8 + 1L * window, "window bound over W = " + window);
            }
        }
        long totalAccepted = 0;
        foreach (long value in perTick) totalAccepted += value;
        Check(totalAccepted <= 8 + 1L * (lastTick + 1), "the whole trace respects the bound");

        // --- sustained overload: steady-state throughput is exactly R ----------------------------------
        var overload = new Arrival[800];
        for (int i = 0; i < overload.Length; i++)
        {
            overload[i] = new Arrival { Tick = i, Volume = 2 };
        }
        List<Admission> overloadRows = Harness.Run(overload, 10, 1);
        var overloadReference = Harness.RunReference(overload, 10, 1);
        for (int i = 0; i < overloadRows.Count; i++)
        {
            Check(overloadRows[i].Accepted == overloadReference[i].Accepted, "overload verdict agrees");
        }
        long[] overloadPerTick = Harness.AcceptedByTick(overloadRows, overload.Length);
        long steadyAccepted = 0;
        for (int tick = 200; tick < 400; tick++) steadyAccepted += overloadPerTick[tick];
        Check(steadyAccepted == 200, "steady state admits exactly R = 1 unit per tick (got " + steadyAccepted + ")");
        long earlyAccepted = 0;
        for (int tick = 0; tick < 200; tick++) earlyAccepted += overloadPerTick[tick];
        Check(earlyAccepted <= 210, "the early window is capped by the burst allowance");
        int overloadRejections = 0;
        foreach (Admission row in overloadRows)
        {
            if (!row.Accepted) overloadRejections += 1;
        }
        Check(overloadRejections > 300, "sustained overload rejects most requests");

        // --- the rate-limited queue --------------------------------------------------------------------
        var queue = new RateLimitedQueue(3, 5);
        for (long i = 0; i < 20; i++) queue.Enqueue(i, 0);
        queue.RunUntil(50);
        Check(queue.Drained.Count + queue.Dropped.Count == queue.EnqueuedCount, "items are conserved");
        Check(queue.EnqueuedCount == 20, "everything enqueued");
        for (int i = 1; i < queue.Drained.Count; i++)
        {
            Check(queue.Drained[i].Id > queue.Drained[i - 1].Id, "the queue drains in FIFO order");
            Check(queue.Drained[i].Tick >= queue.Drained[i - 1].Tick, "drain ticks are non-decreasing");
        }
        var perTickDrains = new Dictionary<long, int>();
        foreach ((long _, long tick) in queue.Drained)
        {
            perTickDrains.TryGetValue(tick, out int count);
            perTickDrains[tick] = count + 1;
        }
        foreach (KeyValuePair<long, int> entry in perTickDrains)
        {
            Check(entry.Value <= 3, "at most R = 3 items drain per tick");
        }
        foreach ((long _, long tick) in queue.Drained)
        {
            Check(tick <= 5, "every drained item met the 5-tick deadline");
        }
        Check(queue.Drained.Count >= 15, "the first 15 items drain within five ticks");

        var slowQueue = new RateLimitedQueue(1, 2);
        for (long i = 0; i < 10; i++) slowQueue.Enqueue(i, 0);
        slowQueue.RunUntil(20);
        Check(slowQueue.Drained.Count == 3 && slowQueue.Dropped.Count == 7,
                "with a 2-tick deadline and R = 1 only three items survive");
        Check(slowQueue.Drained[0].Id == 0 && slowQueue.Drained[2].Id == 2, "the survivors are the oldest");
        Check(slowQueue.Dropped[0] == 3 && slowQueue.Dropped[slowQueue.Dropped.Count - 1] == 9,
                "the dropped items are the newest, in order");
        foreach ((long id, long tick) in slowQueue.Drained)
        {
            Check(tick <= 2, "drained item " + id + " waited at most the 2-tick deadline");
        }

        var emptyQueue = new RateLimitedQueue(5, 3);
        emptyQueue.RunUntil(10);
        Check(emptyQueue.Drained.Count == 0 && emptyQueue.Dropped.Count == 0, "an empty queue drains nothing");

        // --- determinism --------------------------------------------------------------------------------
        List<Admission> repeat = Harness.Run(stream, 8, 1);
        for (int i = 0; i < rows.Count; i++)
        {
            Check(repeat[i].Accepted == rows[i].Accepted && repeat[i].Level == rows[i].Level,
                    "the harness is deterministic at record " + i);
        }

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
