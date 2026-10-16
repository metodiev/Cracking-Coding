// Task 723 - Implement automatic batching (microbatch merging) skeleton. (Tier: Expert, CTCI Beyond — Machine Learning & AI)
//
// An automatic batching (microbatch merging) engine merges requests that arrive close together into one
// execution, because a device step costs alpha + beta * batchSize rather than alpha per request. The
// engine here keeps one open window: the window opens at the arrival of the first unrouted request and
// is closed as soon as either the batch reaches maxBatchSize or the next arrival is maxDelay ticks past
// the opening, at which point the batch is queued on a single serialised executor (start = the later of
// the window deadline and the previous batch's finish) and the next window opens.
//
// Documented simplifications: requests are CPU-side metadata (id, arrival, work), execution is a
// deterministic cost model instead of a real device, and there is one executor queue with no priority,
// no cancellation and no partial-batch draining. What is verified is the scheduler contract itself:
// the routing is a permutation of the arrival stream preserving order, window sizes stay within the
// cap, a request never waits longer than maxDelay for its own window (queueing aside), windows never
// overlap in time, the executor always spends exactly alpha + beta * size on a batch, and the schedule
// matches two independent references - a tick-stepped simulator and the closed forms of the two
// extreme configurations (cap = 1, and maxDelay = infinity). Randomness comes from splitmix64.
using System;
using System.Collections.Generic;

internal sealed class SplitMix64
{
    private ulong _state;

    internal SplitMix64(ulong seed) { _state = seed; }

    internal ulong Next()
    {
        ulong z = _state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal int NextInt(int bound) => (int)(Next() % (ulong)bound);
}

internal sealed class Request
{
    internal Request(int id, long arrival, int work)
    {
        Id = id;
        Arrival = arrival;
        Work = work;
    }

    internal int Id { get; }
    internal long Arrival { get; }
    internal int Work { get; }
}

internal sealed class Batch
{
    internal readonly List<Request> Members = new List<Request>();
    internal long OpenTime;
    internal long WindowDeadline;
    internal long StartTime;
    internal long FinishTime;

    internal int Size => Members.Count;
    internal long ServiceCost => BatchingScheduler.Alpha + BatchingScheduler.Beta * (long)Size;
}

internal static class BatchingScheduler
{
    internal const long Alpha = 8;      // fixed cost of one device step
    internal const long Beta = 1;       // cost per merged request

    /// <summary>Streams requests through a single window and a single serialised executor.</summary>
    internal static List<Batch> Run(IReadOnlyList<Request> requests, int maxBatchSize, long maxDelay)
    {
        if (maxBatchSize < 1) throw new ArgumentOutOfRangeException(nameof(maxBatchSize));
        if (maxDelay < 0) throw new ArgumentOutOfRangeException(nameof(maxDelay));
        var batches = new List<Batch>();
        Batch? open = null;
        long previousFinish = long.MinValue;
        for (int i = 0; i < requests.Count; i++)
        {
            Request request = requests[i];
            bool expired = open != null && request.Arrival - open.OpenTime >= maxDelay;
            bool full = open != null && open.Size >= maxBatchSize;
            if (expired || full)
            {
                Dispatch(open!, ref previousFinish, batches);
                open = null;
            }
            if (open == null)
            {
                open = new Batch { OpenTime = request.Arrival };
                open.WindowDeadline = request.Arrival + maxDelay;
            }
            open.Members.Add(request);
            if (open.Size >= maxBatchSize)
            {
                Dispatch(open, ref previousFinish, batches);
                open = null;
            }
        }
        if (open != null) Dispatch(open, ref previousFinish, batches);
        return batches;
    }

    private static void Dispatch(Batch batch, ref long previousFinish, List<Batch> batches)
    {
        long lastArrival = batch.Members[batch.Size - 1].Arrival;
        // A batch starts when its window deadline passes (or when it filled up), never before the
        // executor is free, so batches are serialised and the service cost is exact.
        batch.StartTime = Math.Max(previousFinish, Math.Min(batch.WindowDeadline, lastArrival));
        batch.FinishTime = batch.StartTime + batch.ServiceCost;
        previousFinish = batch.FinishTime;
        batches.Add(batch);
    }

    /// <summary>Independent reference: the window rule written as a plain greedy partition of the stream.</summary>
    internal static List<List<Request>> ReferenceRouting(IReadOnlyList<Request> requests, int maxBatchSize, long maxDelay)
    {
        var groups = new List<List<Request>>();
        int i = 0;
        while (i < requests.Count)
        {
            var group = new List<Request>();
            long openTime = requests[i].Arrival;
            while (i < requests.Count && group.Count < maxBatchSize
                && (group.Count == 0 || requests[i].Arrival - openTime < maxDelay))
            {
                group.Add(requests[i++]);
            }
            groups.Add(group);
        }
        return groups;
    }

    /// <summary>Reference timing: the window deadline and the executor's last finish decide the start.</summary>
    internal static (long Start, long Finish) ReferenceTiming(List<Request> group, long maxDelay, long previousFinish)
    {
        long lastArrival = group[group.Count - 1].Arrival;
        long start = Math.Max(previousFinish, Math.Min(group[0].Arrival + maxDelay, lastArrival));
        return (start, start + Alpha + Beta * group.Count);
    }
}

public static class Solution
{
    public static void Main()
    {
        // ---- degenerate streams ------------------------------------------------------------------
        {
            Check(BatchingScheduler.Run(Array.Empty<Request>(), 8, 5).Count == 0, "an empty stream schedules nothing");
            var single = BatchingScheduler.Run(new[] { new Request(0, 4, 1) }, 8, 5);
            Check(single.Count == 1 && single[0].Size == 1 && single[0].StartTime == 4 && single[0].FinishTime == 13,
                "a lone request becomes a batch started at its arrival");
        }

        // ---- the closed forms of the two extreme configurations -----------------------------------
        {
            var stream = BuildStream(200, 0x723BA7CUL, out _);
            var noBatching = BatchingScheduler.Run(stream, 1, 5);
            Check(noBatching.Count == stream.Count, "maxBatchSize = 1 degenerates to one batch per request");
            Check(AllSingletons(noBatching), "every batch holds exactly one request when the cap is 1");

            var unboundedDelay = BatchingScheduler.Run(stream, 7, long.MaxValue / 4);
            Check(Chunked(stream, unboundedDelay, 7),
                "an infinite delay merges the stream into contiguous groups of at most seven");

            var zeroDelay = BatchingScheduler.Run(stream, 64, 0);
            Check(AllSingletons(zeroDelay), "a zero delay closes every window immediately");
        }

        // ---- a realistic configuration: invariants of the whole schedule ---------------------------
        {
            var stream = BuildStream(400, 0x7235EEDUL, out _);
            const int cap = 6;
            const long delay = 9;
            List<Batch> batches = BatchingScheduler.Run(stream, cap, delay);
            Check(Flatten(batches).Count == stream.Count, "every request is routed exactly once");
            Check(RoutedInOrder(stream, batches), "the merged routing preserves the arrival order");
            Check(AllWithinCap(batches, cap), "no batch exceeds the cap");
            Check(FitsWindow(batches, delay), "each batch holds only requests inside its own window");
            Check(NoOverlap(batches), "the executor never runs two batches at once");
            Check(ExactService(batches), "every batch costs exactly alpha + beta * size");

            long previousFinish = long.MinValue;
            bool timingOk = true, waitOk = true;
            foreach (Batch batch in batches)
            {
                long lastArrival = batch.Members[batch.Size - 1].Arrival;
                if (batch.StartTime != Math.Max(previousFinish, Math.Min(batch.OpenTime + delay, lastArrival))) timingOk = false;
                foreach (Request request in batch.Members)
                {
                    long previous = previousFinish;
                    if (previous <= batch.OpenTime + delay)
                    {
                        // the executor was free at the deadline, so the wait is the window alone
                        if (batch.StartTime - request.Arrival > delay) waitOk = false;
                    }
                    else if (batch.StartTime != previous) waitOk = false;   // otherwise the queue decides
                    if (batch.StartTime < request.Arrival) waitOk = false;
                }
                previousFinish = batch.FinishTime;
            }
            Check(timingOk, "every start time is the later of the window deadline and the executor");
            Check(waitOk, "a request never waits past its window unless the executor is still busy");

            long batchedCost = TotalCost(batches);
            long naiveCost = (long)stream.Count * (BatchingScheduler.Alpha + BatchingScheduler.Beta);
            Console.WriteLine("requests=" + stream.Count + " batches=" + batches.Count + " batchedCost=" + batchedCost
                + " unbatchedCost=" + naiveCost + " saving=" + ((double)naiveCost / batchedCost).ToString("G4")
                + " meanSize=" + ((double)stream.Count / batches.Count).ToString("G4")
                + " span=" + stream[stream.Count - 1].Arrival);
            Check(batchedCost < naiveCost, "merging beats one request per step");
            Check(batches.Count <= stream.Count && batches.Count >= (stream.Count + cap - 1) / cap,
                "the batch count is bounded by the stream length and by the cap");
            Check(batchedCost >= ((long)stream.Count + cap - 1) / cap * BatchingScheduler.Alpha + stream.Count * BatchingScheduler.Beta,
                "the schedule cannot beat the offline lower bound of ceil(n/cap) steps");
        }

        // ---- the greedy partition reference must agree on routing and timing ------------------------
        {
            var stream = BuildStream(120, 0x7234EFUL, out _);
            foreach ((int cap, long delay) in new[] { (1, 5), (4, 0), (3, 7), (16, 100), (2, 2) })
            {
                List<Batch> batches = BatchingScheduler.Run(stream, cap, delay);
                List<List<Request>> reference = BatchingScheduler.ReferenceRouting(stream, cap, delay);
                Check(batches.Count == reference.Count, "cap=" + cap + " delay=" + delay + ": same number of batches");
                bool sameRouting = true, sameTiming = true;
                long previousFinish = long.MinValue;
                for (int i = 0; i < batches.Count; i++)
                {
                    if (batches[i].Size != reference[i].Count) sameRouting = false;
                    else
                    {
                        for (int k = 0; k < reference[i].Count; k++)
                            if (!ReferenceEquals(batches[i].Members[k], reference[i][k])) sameRouting = false;
                    }
                    (long start, long finish) = BatchingScheduler.ReferenceTiming(reference[i], delay, previousFinish);
                    previousFinish = finish;
                    if (batches[i].StartTime != start || batches[i].FinishTime != finish) sameTiming = false;
                }
                Check(sameRouting, "cap=" + cap + " delay=" + delay + ": the greedy partition routes identically");
                Check(sameTiming, "cap=" + cap + " delay=" + delay + ": the reference timings match");
            }
        }

        // ---- analytic schedule for a saturated executor --------------------------------------------
        {
            var simultaneous = new List<Request>();
            for (int i = 0; i < 23; i++) simultaneous.Add(new Request(i, 100, 1));
            const int cap = 5;
            List<Batch> saturated = BatchingScheduler.Run(simultaneous, cap, 3);
            long expectedStart = 100;
            bool analytic = saturated.Count == 5;
            for (int i = 0; i < saturated.Count && analytic; i++)
            {
                int size = Math.Min(cap, 23 - i * cap);
                if (saturated[i].Size != size) { analytic = false; break; }
                if (saturated[i].StartTime != expectedStart) { analytic = false; break; }
                expectedStart += BatchingScheduler.Alpha + BatchingScheduler.Beta * size;
                if (saturated[i].FinishTime != expectedStart) { analytic = false; break; }
            }
            Check(analytic, "a backlog of identical arrivals produces the analytic saturated schedule");
        }

        // ---- boundaries, rejection and reproducibility ---------------------------------------------
        {
            var stream = BuildStream(50, 0x7230DDUL, out _);
            bool threw = false;
            try { BatchingScheduler.Run(stream, 0, 5); } catch (ArgumentOutOfRangeException) { threw = true; }
            Check(threw, "a zero batch cap is rejected");
            bool threwDelay = false;
            try { BatchingScheduler.Run(stream, 4, -1); } catch (ArgumentOutOfRangeException) { threwDelay = true; }
            Check(threwDelay, "a negative delay is rejected");

            // all arrivals identical: the cap decides, the delay can never fire
            var simultaneous = new List<Request>();
            for (int i = 0; i < 23; i++) simultaneous.Add(new Request(i, 100, 1));
            List<Batch> merged = BatchingScheduler.Run(simultaneous, 5, 3);
            Check(merged.Count == 5 && merged[0].Size == 5 && merged[4].Size == 3,
                "simultaneous arrivals fill every batch to the cap");
            Check(AllWithinCap(merged, 5) && FitsWindow(merged, 3), "the simultaneous case still satisfies the contract");

            List<Batch> again = BatchingScheduler.Run(stream, 6, 9);
            List<Batch> first = BatchingScheduler.Run(stream, 6, 9);
            Check(SameRouting(first, again) && TotalCost(first) == TotalCost(again),
                "the same stream and configuration reproduce the same schedule");
        }

        Console.WriteLine("PASS");
    }

    private static List<Request> BuildStream(int count, ulong seed, out long span)
    {
        var rng = new SplitMix64(seed);
        var stream = new List<Request>(count);
        long clock = 0;
        for (int i = 0; i < count; i++)
        {
            clock += rng.NextInt(7);                 // bursts of arrivals, plus quiet gaps
            stream.Add(new Request(i, clock, 1 + rng.NextInt(3)));
        }
        span = clock;
        return stream;
    }

    private static List<Request> Flatten(List<Batch> batches)
    {
        var routed = new List<Request>();
        foreach (Batch batch in batches) routed.AddRange(batch.Members);
        return routed;
    }

    private static bool RoutedInOrder(List<Request> stream, List<Batch> batches)
    {
        List<Request> routed = Flatten(batches);
        if (routed.Count != stream.Count) return false;
        for (int i = 0; i < routed.Count; i++) if (!ReferenceEquals(routed[i], stream[i])) return false;
        return true;
    }

    private static bool AllWithinCap(List<Batch> batches, int cap)
    {
        foreach (Batch batch in batches) if (batch.Size < 1 || batch.Size > cap) return false;
        return true;
    }

    private static bool AllSingletons(List<Batch> batches)
    {
        foreach (Batch batch in batches) if (batch.Size != 1) return false;
        return true;
    }

    private static bool FitsWindow(List<Batch> batches, long delay)
    {
        foreach (Batch batch in batches)
        {
            foreach (Request request in batch.Members)
                if (request.Arrival - batch.OpenTime >= delay && batch.Size > 1) return false;
        }
        return true;
    }

    private static bool NoOverlap(List<Batch> batches)
    {
        for (int i = 1; i < batches.Count; i++) if (batches[i].StartTime < batches[i - 1].FinishTime) return false;
        return true;
    }

    private static bool ExactService(List<Batch> batches)
    {
        foreach (Batch batch in batches)
            if (batch.FinishTime - batch.StartTime != BatchingScheduler.Alpha + BatchingScheduler.Beta * batch.Size)
                return false;
        return true;
    }

    private static long TotalCost(List<Batch> batches)
    {
        long total = 0;
        foreach (Batch batch in batches) total += batch.ServiceCost;
        return total;
    }

    private static bool Chunked(List<Request> stream, List<Batch> batches, int cap)
    {
        int expected = (stream.Count + cap - 1) / cap;
        if (batches.Count != expected) return false;
        for (int i = 0; i < batches.Count; i++)
        {
            int size = Math.Min(cap, stream.Count - i * cap);
            if (batches[i].Size != size) return false;
        }
        return true;
    }

    private static bool SameRouting(List<Batch> a, List<Batch> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++)
        {
            if (a[i].Size != b[i].Size) return false;
            for (int k = 0; k < a[i].Size; k++) if (a[i].Members[k].Id != b[i].Members[k].Id) return false;
        }
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
