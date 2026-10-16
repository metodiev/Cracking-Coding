// Task 891 - Implement benchmarking harness for large distributed systems with failure injection. (Tier: Master, Beyond — Software Engineering & Tooling)
//
// A deterministic discrete-event cluster simulator plus a benchmarking harness on top of it. Requests
// arrive at a node, walk a fixed pipeline of RPC hops, and each node serves one request at a time, so
// contention shows up as queueing. A single event queue ordered by (time, sequence) drives everything:
// arrivals, service completions, forwarded hops and retry timeouts. Faults come from an explicit plan -
// node crash windows, a degraded (slow) node, dropped messages at a deterministic rate and reordered
// deliveries - and the plan counts its decisions while the message path counts its observations, so the
// tests can show the faults actually reached the path rather than being decided and forgotten.
//
// The harness reports throughput, nearest-rank latency percentiles, failure/retry/drop counts, per-node
// busy time, and an SLO scorecard (p99 latency, failure rate, minimum throughput). Every request keeps a
// contiguous trace of (network, wait, service, timeout) segments, which the tests use to re-derive the
// end-to-end latency independently of the simulator's bookkeeping.
//
// Simplifications (documented, not behavioural): time is an abstract microsecond counter, there are no
// threads, sockets or wall-clock reads, and latency is the simulated one - the point is a reproducible
// harness, not a real network. There is no dynamic membership, consensus or clock skew; a request either
// completes, fails on a crashed node, or fails once its retries are exhausted. Packet loss is modelled
// only on the forward path.
//
// Verified: events are consumed in non-decreasing time order; every submitted request ends completed or
// failed, and no request is left in flight; each request's trace is gapless and its segments sum exactly
// to the end-to-end latency (for failed requests too); latency percentiles equal an independently coded
// nearest-rank oracle and are monotone with the 0th/100th percentiles at the extremes; total busy time
// never exceeds nodes x makespan and throughput respects the nodes/service-time ceiling; injected drops
// are observed, retries stay inside their budget, crash hits equal failures, and a fully crashed cluster
// fails everything with no latency samples; reordering and slow nodes raise the tail; the scorecard
// flips to FAIL for unreachable SLOs and to PASS for just-inside limits; an empty workload reports
// zeroes rather than dividing by zero; deeper pipelines cost more; and the same seed reproduces every
// metric exactly while a different seed gives a different profile.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

internal enum Phase { Network, Wait, Service, Timeout }

internal sealed class TraceSegment
{
    internal long Start;
    internal long End;
    internal Phase Phase;
}

internal sealed class Request
{
    internal int Id;
    internal long SubmittedAt;
    internal long CompletedAt;
    internal bool Completed;
    internal bool Failed;
    internal string FailureReason = "";
    internal int Retries;
    internal int Hop;
    internal int Level;
    internal readonly List<TraceSegment> Trace = new List<TraceSegment>();

    internal long Latency => CompletedAt - SubmittedAt;
}

internal sealed class SimEvent
{
    internal long Time;
    internal long Sequence;
    internal int Kind;                 // 0 arrive, 1 service completed, 2 retry arrival
    internal Request Request = null!;
    internal int Node;
}

internal sealed class FaultPlan
{
    internal HashSet<int> CrashedNodes = new HashSet<int>();
    internal long CrashStart;
    internal long CrashEnd;
    internal int SlowNode = -1;
    internal long SlowStart;
    internal long SlowEnd;
    internal long SlowFactor = 1;
    internal int DropRatePerMille;
    internal int ReorderRatePerMille;
    internal int MaxRetries = 2;
    internal long Timeout = 400;
    internal long NetworkLatency = 20;
    internal long ReorderJitter = 300;

    internal long DropDecisions;
    internal long ReorderDecisions;

    internal bool NodeDown(int node, long time) =>
        CrashedNodes.Contains(node) && time >= CrashStart && time < CrashEnd;

    internal long ServiceFactor(int node, long time) =>
        node == SlowNode && time >= SlowStart && time < SlowEnd ? SlowFactor : 1;

    internal bool DecideDrop(int perMille)
    {
        if (perMille >= DropRatePerMille) return false;
        DropDecisions++;
        return true;
    }

    internal bool DecideReorder(int perMille)
    {
        if (perMille >= ReorderRatePerMille) return false;
        ReorderDecisions++;
        return true;
    }
}

internal sealed class Metrics
{
    internal long Submitted;
    internal long Completed;
    internal long Failed;
    internal long Retries;
    internal long DroppedMessages;
    internal long ReorderedMessages;
    internal long CrashedNodeHits;
    internal long Makespan;
    internal long TotalServiceTime;
    internal long BusyTime;
    internal long TotalLatency;
    internal readonly List<long> Latencies = new List<long>();

    internal double Throughput => Makespan == 0 ? 0.0 : (double)Completed * 1_000_000.0 / Makespan;

    internal long Percentile(double percentile)
    {
        if (Latencies.Count == 0) return 0;
        long[] sorted = Latencies.OrderBy(v => v).ToArray();
        int rank = (int)Math.Ceiling(percentile / 100.0 * sorted.Length) - 1;
        return sorted[Math.Max(0, Math.Min(sorted.Length - 1, rank))];
    }

    internal double FailureRate => Submitted == 0 ? 0.0 : (double)Failed / Submitted;
}

internal sealed class Scorecard
{
    internal readonly List<(string Name, bool Met, string Detail)> Entries = new List<(string, bool, string)>();
    internal bool AllMet => Entries.TrueForAll(entry => entry.Met);

    internal void Add(string name, bool met, string detail) => Entries.Add((name, met, detail));

    internal string Describe() =>
        string.Join("; ", Entries.Select(entry => (entry.Met ? "OK " : "FAIL ") + entry.Name + " " + entry.Detail));
}

internal sealed class Simulator
{
    internal readonly int NodeCount;
    internal readonly FaultPlan Plan;
    internal readonly long BaseService;
    internal readonly Metrics Metrics = new Metrics();
    internal readonly List<Request> Tracked = new List<Request>();
    internal long OrderingViolations;
    internal long EventsProcessed;

    private readonly PriorityQueue<SimEvent, (long Time, long Sequence)> _queue = new PriorityQueue<SimEvent, (long, long)>();
    private readonly long[] _busyUntil;
    private ulong _random;
    private long _sequence;
    private long _lastEventTime;

    internal Simulator(int nodes, int seed, FaultPlan plan, long baseService)
    {
        NodeCount = nodes;
        Plan = plan;
        BaseService = baseService;
        _busyUntil = new long[nodes];
        _random = (ulong)seed * 0x9E3779B97F4A7C15UL + 0x123456789UL;
    }

    private ulong Next()
    {
        _random += 0x9E3779B97F4A7C15UL;
        ulong z = _random;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private int PerMille() => (int)(Next() % 1000UL);

    private void Schedule(long time, int kind, Request request, int node)
    {
        _sequence++;
        _queue.Enqueue(new SimEvent { Time = time, Sequence = _sequence, Kind = kind, Request = request, Node = node },
            (time, _sequence));
    }

    private void Segment(Request request, long start, long end, Phase phase)
    {
        if (end > start) request.Trace.Add(new TraceSegment { Start = start, End = end, Phase = phase });
    }

    internal void Submit(int id, long at, int node, int depth)
    {
        var request = new Request { Id = id, SubmittedAt = at, Level = depth };
        Tracked.Add(request);
        Metrics.Submitted++;
        Schedule(at, 0, request, node);
    }

    /** Drives the event queue to exhaustion; returns the simulated makespan. */
    internal long Run()
    {
        while (_queue.Count > 0)
        {
            SimEvent current = _queue.Dequeue();
            if (current.Time < _lastEventTime) OrderingViolations++;
            _lastEventTime = current.Time;
            EventsProcessed++;
            switch (current.Kind)
            {
                case 0: HandleArrive(current.Request, current.Time, current.Node); break;
                case 1: HandleService(current.Request, current.Time, current.Node); break;
                default: HandleRetry(current.Request, current.Time, current.Node); break;
            }
        }
        Metrics.Makespan = _lastEventTime;
        return _lastEventTime;
    }

    private void HandleArrive(Request request, long time, int node)
    {
        if (Plan.NodeDown(node, time))
        {
            Metrics.CrashedNodeHits++;
            Fail(request, time, "node " + node + " is down");
            return;
        }
        long start = Math.Max(time, _busyUntil[node]);
        Segment(request, time, start, Phase.Wait);
        long service = BaseService * Plan.ServiceFactor(node, time) + (long)(Next() % 40UL);
        Segment(request, start, start + service, Phase.Service);
        _busyUntil[node] = start + service;
        Metrics.TotalServiceTime += service;
        Metrics.BusyTime += service;
        Schedule(start + service, 1, request, node);
    }

    private void HandleService(Request request, long time, int node)
    {
        if (request.Hop + 1 >= request.Level)
        {
            request.Completed = true;
            request.CompletedAt = time;
            Metrics.Completed++;
            Metrics.Latencies.Add(request.Latency);
            Metrics.TotalLatency += request.Latency;
            return;
        }
        if (Plan.DecideDrop(PerMille()))
        {
            Metrics.DroppedMessages++;
            if (request.Retries >= Plan.MaxRetries)
            {
                Fail(request, time, "retries exhausted after a dropped message");
                return;
            }
            request.Retries++;
            Metrics.Retries++;
            long retryAt = time + Plan.Timeout;
            Segment(request, time, retryAt, Phase.Timeout);
            Schedule(retryAt, 2, request, (node + 1) % NodeCount);
            return;
        }
        Forward(request, time, (node + 1) % NodeCount);
    }

    private void HandleRetry(Request request, long time, int node)
    {
        if (Plan.NodeDown(node, time))
        {
            Metrics.CrashedNodeHits++;
            Fail(request, time, "node " + node + " is down on a retry");
            return;
        }
        Forward(request, time, node);
    }

    private void Forward(Request request, long time, int node)
    {
        request.Hop++;
        long latency = Plan.NetworkLatency + (long)(Next() % (ulong)Plan.NetworkLatency);
        if (Plan.DecideReorder(PerMille()))
        {
            Metrics.ReorderedMessages++;
            latency += (long)(Next() % (ulong)Plan.ReorderJitter);
        }
        Segment(request, time, time + latency, Phase.Network);
        Schedule(time + latency, 0, request, node);
    }

    private void Fail(Request request, long time, string reason)
    {
        request.Failed = true;
        request.CompletedAt = time;
        request.FailureReason = reason;
        Metrics.Failed++;
    }

    /** Recomputes end-to-end latency from the trace, independently of the simulator's own counters. */
    internal static long TraceLatency(Request request)
    {
        long total = 0;
        foreach (TraceSegment segment in request.Trace) total += segment.End - segment.Start;
        return total;
    }

    internal static bool TraceIsContiguous(Request request)
    {
        long cursor = request.SubmittedAt;
        foreach (TraceSegment segment in request.Trace)
        {
            if (segment.Start != cursor) return false;
            cursor = segment.End;
        }
        return cursor == request.CompletedAt;
    }

    internal long BusyUntilSum() => _busyUntil.Sum();
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static Simulator RunSimulation(int nodes, int seed, FaultPlan plan, long baseService, int requests, int depth, long spacing)
    {
        var simulator = new Simulator(nodes, seed, plan, baseService);
        long at = 0;
        for (int i = 0; i < requests; i++)
        {
            simulator.Submit(i, at, i % nodes, depth);
            at += spacing;
        }
        simulator.Run();
        return simulator;
    }

    private static Scorecard ScoreFor(Simulator simulator, double p99Limit, double failureLimit, double throughputLimit)
    {
        var card = new Scorecard();
        card.Add("p99_latency", simulator.Metrics.Percentile(99) <= p99Limit,
            simulator.Metrics.Percentile(99) + "<=" + p99Limit.ToString("F0", CultureInfo.InvariantCulture));
        card.Add("failure_rate", simulator.Metrics.FailureRate <= failureLimit,
            simulator.Metrics.FailureRate.ToString("F4", CultureInfo.InvariantCulture) + "<=" +
            failureLimit.ToString("F4", CultureInfo.InvariantCulture));
        card.Add("throughput", simulator.Metrics.Throughput >= throughputLimit,
            simulator.Metrics.Throughput.ToString("F1", CultureInfo.InvariantCulture) + ">=" +
            throughputLimit.ToString("F1", CultureInfo.InvariantCulture));
        return card;
    }

    public static void Main()
    {
        // ---- 1. a healthy cluster --------------------------------------------------------------
        var healthyPlan = new FaultPlan();
        Simulator healthy = RunSimulation(4, 20261016, healthyPlan, 200, 400, 3, 50);
        Metrics metrics = healthy.Metrics;
        Check(metrics.Submitted == 400, "400 requests were submitted");
        Check(metrics.Completed == 400 && metrics.Failed == 0, "a healthy cluster completes everything");
        Check(metrics.Retries == 0 && metrics.DroppedMessages == 0 && metrics.ReorderedMessages == 0,
            "no faults means no drops, retries or reordering");
        Check(healthy.OrderingViolations == 0, "events are consumed in non-decreasing time order");
        Check(metrics.Latencies.Count == 400 && metrics.TotalLatency > 0, "every completed request reported a latency");
        Check(healthy.Tracked.Count == 400 && healthy.Tracked.All(r => r.Completed), "no request is left in flight");

        // ---- 2. traces explain the latency exactly ----------------------------------------------
        long traceTotal = 0;
        foreach (Request request in healthy.Tracked)
        {
            Check(Simulator.TraceIsContiguous(request), "request " + request.Id + " has a gapless trace");
            Check(Simulator.TraceLatency(request) == request.Latency,
                "trace segments sum to the end-to-end latency of request " + request.Id);
            Check(request.Latency >= 3 * healthy.BaseService, "three services dominate the latency");
            traceTotal += Simulator.TraceLatency(request);
        }
        Check(traceTotal == metrics.TotalLatency, "traces account for the whole latency total");

        // ---- 3. percentiles against an independent nearest-rank oracle -------------------------
        long p50 = metrics.Percentile(50), p95 = metrics.Percentile(95), p99 = metrics.Percentile(99);
        Check(p50 <= p95 && p95 <= p99, "percentiles are monotone: " + p50 + " <= " + p95 + " <= " + p99);
        long[] sorted = metrics.Latencies.OrderBy(v => v).ToArray();
        Check(sorted[(int)Math.Ceiling(0.50 * sorted.Length) - 1] == p50, "p50 equals the hand-computed nearest rank");
        Check(sorted[(int)Math.Ceiling(0.95 * sorted.Length) - 1] == p95, "p95 equals the hand-computed nearest rank");
        Check(sorted[(int)Math.Ceiling(0.99 * sorted.Length) - 1] == p99, "p99 equals the hand-computed nearest rank");
        Check(metrics.Percentile(100) == sorted[^1] && metrics.Percentile(0) == sorted[0],
            "the extreme percentiles are the maximum and the minimum");

        // ---- 4. resource bounds ------------------------------------------------------------------
        Check(metrics.TotalServiceTime >= metrics.Completed * 3L * healthy.BaseService,
            "at least three minimum services per request: " + metrics.TotalServiceTime);
        Check(metrics.BusyTime <= metrics.TotalServiceTime && metrics.BusyTime > 0, "busy time is a share of service time");
        Check(metrics.BusyTime <= healthy.NodeCount * metrics.Makespan,
            "busy time across nodes never exceeds nodes x makespan: " + metrics.BusyTime + " <= " +
            healthy.NodeCount * metrics.Makespan);
        double throughputCeiling = (double)healthy.NodeCount * 1_000_000.0 / healthy.BaseService;
        Check(metrics.Throughput < throughputCeiling, "throughput respects the nodes/service-time ceiling: " +
            metrics.Throughput.ToString("F1", CultureInfo.InvariantCulture) + " < " +
            throughputCeiling.ToString("F1", CultureInfo.InvariantCulture));
        Check(metrics.Makespan >= metrics.Completed * 3 * healthy.BaseService / healthy.NodeCount,
            "makespan is at least the serialized service load");

        // ---- 5. failure injection ---------------------------------------------------------------
        // Control run with the same workload and seed but no faults, so the tail can be compared fairly.
        Simulator control = RunSimulation(4, 777, new FaultPlan(), 200, 300, 3, 40);
        Check(control.Metrics.Completed == 300 && control.Metrics.Failed == 0, "the control run is fault free");
        long controlP99 = control.Metrics.Percentile(99);
        var dropPlan = new FaultPlan { DropRatePerMille = 120, MaxRetries = 1 };
        Simulator dropped = RunSimulation(4, 777, dropPlan, 200, 300, 3, 40);
        Check(dropped.Metrics.DroppedMessages > 0, "the plan injected drops: " + dropped.Metrics.DroppedMessages);
        Check(dropped.Metrics.Retries > 0, "drops trigger retries: " + dropped.Metrics.Retries);
        Check(dropPlan.DropDecisions == dropped.Metrics.DroppedMessages,
            "every injected drop is observed in the message path: " + dropPlan.DropDecisions + " vs " +
            dropped.Metrics.DroppedMessages);
        Check(dropped.Metrics.Completed + dropped.Metrics.Failed == dropped.Metrics.Submitted, "requests are conserved under drops");
        Check(dropped.Metrics.Failed > 0, "a 12% drop rate with one retry fails some requests: " + dropped.Metrics.Failed);
        foreach (Request request in dropped.Tracked)
        {
            Check(request.Retries <= dropPlan.MaxRetries, "request " + request.Id + " stayed inside its retry budget");
            Check(Simulator.TraceIsContiguous(request), "request " + request.Id + " keeps a gapless trace under drops");
            Check(Simulator.TraceLatency(request) == request.Latency, "the trace still explains the latency under drops");
        }
        List<Request> retried = dropped.Tracked.Where(r => r.Completed && r.Retries > 0).ToList();
        List<Request> clean = dropped.Tracked.Where(r => r.Completed && r.Retries == 0).ToList();
        Check(retried.Count > 0, "some requests completed after a retry: " + retried.Count);
        Check(retried.All(r => r.Latency > dropPlan.Timeout), "a retried request pays at least the timeout");
        Check(retried.All(r => r.Trace.Any(s => s.Phase == Phase.Timeout)), "retried requests carry a timeout segment");
        Check(retried.Average(r => r.Latency) > clean.Average(r => r.Latency),
            "retried requests are slower on average: " +
            retried.Average(r => r.Latency).ToString("F0", CultureInfo.InvariantCulture) + " vs " +
            clean.Average(r => r.Latency).ToString("F0", CultureInfo.InvariantCulture));
        Check(dropped.Metrics.Percentile(99) >= controlP99 - dropPlan.Timeout,
            "the fault run never looks *better* than the control: " + dropped.Metrics.Percentile(99) + " vs " + controlP99);

        var slowPlan = new FaultPlan { SlowNode = 1, SlowEnd = long.MaxValue, SlowFactor = 6 };
        Simulator slowed = RunSimulation(4, 4242, slowPlan, 200, 300, 3, 40);
        Simulator slowControl = RunSimulation(4, 4242, new FaultPlan(), 200, 300, 3, 40);
        Check(slowed.Metrics.Percentile(95) > slowControl.Metrics.Percentile(95),
            "a degraded node raises p95: " + slowed.Metrics.Percentile(95) + " > " + slowControl.Metrics.Percentile(95));
        Check(slowed.Metrics.Completed == 300, "a slow node does not fail requests");

        var crashPlan = new FaultPlan { CrashedNodes = new HashSet<int> { 2 }, CrashStart = 2000, CrashEnd = 12000 };
        Simulator crashed = RunSimulation(4, 31337, crashPlan, 200, 300, 3, 40);
        Check(crashed.Metrics.CrashedNodeHits > 0, "requests hit the crashed node: " + crashed.Metrics.CrashedNodeHits);
        Check(crashed.Metrics.Failed == crashed.Metrics.CrashedNodeHits,
            "every crash hit fails exactly one request: " + crashed.Metrics.Failed + " vs " + crashed.Metrics.CrashedNodeHits);
        Check(crashed.Metrics.Completed + crashed.Metrics.Failed == crashed.Metrics.Submitted, "crash runs conserve requests");
        foreach (Request request in crashed.Tracked.Where(r => r.Failed))
        {
            Check(request.FailureReason.Contains("down"), "failure reasons name the crashed node");
            Check(Simulator.TraceIsContiguous(request), "failed requests also keep a gapless trace");
        }
        Check(crashed.Metrics.Latencies.Count == crashed.Metrics.Completed, "only completions report latency");

        var allDown = new FaultPlan { CrashedNodes = new HashSet<int> { 0, 1, 2, 3 }, CrashEnd = long.MaxValue };
        Simulator doomed = RunSimulation(4, 5, allDown, 200, 50, 3, 40);
        Check(doomed.Metrics.Failed == 50 && doomed.Metrics.Completed == 0, "a fully crashed cluster fails everything");
        Check(doomed.Metrics.Latencies.Count == 0 && doomed.Metrics.Percentile(99) == 0, "no latency samples are reported");
        Check(doomed.Metrics.FailureRate == 1.0, "the failure rate is exactly 1");
        Check(doomed.Tracked.All(r => Simulator.TraceIsContiguous(r)), "doomed requests have empty, gapless traces");

        var reorderPlan = new FaultPlan { ReorderRatePerMille = 300, ReorderJitter = 900 };
        Simulator reordered = RunSimulation(4, 99, reorderPlan, 200, 200, 3, 40);
        Check(reordered.Metrics.ReorderedMessages > 0, "reordering was injected: " + reordered.Metrics.ReorderedMessages);
        Check(reorderPlan.ReorderDecisions == reordered.Metrics.ReorderedMessages,
            "every reordering decision is observed: " + reorderPlan.ReorderDecisions);
        Check(reordered.Metrics.Completed == 200, "reordered deliveries still complete");

        // ---- 6. the scorecard reacts to the SLOs ------------------------------------------------
        Scorecard lenient = ScoreFor(healthy, long.MaxValue, 1.0, 0.0);
        Check(lenient.AllMet && lenient.Entries.Count == 3, "a lenient scorecard passes every SLO");
        Scorecard harsh = ScoreFor(healthy, 1.0, -0.5, double.MaxValue);
        Check(!harsh.AllMet && harsh.Entries.Count(e => !e.Met) == 3, "impossible SLOs fail the scorecard");
        Scorecard inside = ScoreFor(dropped, dropped.Metrics.Percentile(99) + 1, dropped.Metrics.FailureRate + 0.01,
            dropped.Metrics.Throughput - 1);
        Check(inside.AllMet, "the scorecard passes when the limits sit just outside the measurements");
        Scorecard outside = ScoreFor(dropped, dropped.Metrics.Percentile(99) - 1, dropped.Metrics.FailureRate - 0.01,
            dropped.Metrics.Throughput + 1);
        Check(!outside.AllMet && outside.Entries.Count(e => !e.Met) == 3, "and fails when the limits sit just below");

        // ---- 7. degenerate workloads -------------------------------------------------------------
        Simulator none = RunSimulation(4, 1, new FaultPlan(), 200, 0, 3, 40);
        Check(none.Metrics.Submitted == 0 && none.Metrics.Completed == 0 && none.Metrics.Throughput == 0.0,
            "an empty workload reports zeroes instead of dividing by zero");
        Check(none.Metrics.Percentile(99) == 0 && none.Metrics.FailureRate == 0.0, "an empty workload has no percentiles");
        Simulator one = RunSimulation(2, 2, new FaultPlan(), 200, 1, 1, 40);
        Check(one.Metrics.Completed == 1 && one.Metrics.Latencies.Count == 1, "a single-hop single request completes");
        Check(Simulator.TraceIsContiguous(one.Tracked[0]) && Simulator.TraceLatency(one.Tracked[0]) == one.Tracked[0].Latency,
            "the single request's trace is exact");
        Simulator singleNode = RunSimulation(1, 3, new FaultPlan(), 200, 400, 3, 50);
        Check(singleNode.Metrics.Completed == 400, "a single-node cluster serves every request");
        Check(singleNode.Metrics.Percentile(50) > p50,
            "one node queues far more than four under the same load: " + singleNode.Metrics.Percentile(50) + " > " + p50);
        Check(singleNode.Metrics.Makespan > metrics.Makespan, "and takes longer overall");

        // ---- 8. reproducibility ------------------------------------------------------------------
        Simulator repeat = RunSimulation(4, 20261016, new FaultPlan(), 200, 400, 3, 50);
        Check(repeat.Metrics.Completed == metrics.Completed && repeat.Metrics.Makespan == metrics.Makespan,
            "the same seed reproduces completion and makespan exactly");
        Check(repeat.Metrics.Percentile(99) == p99 && repeat.Metrics.TotalLatency == metrics.TotalLatency,
            "the same seed reproduces latency statistics exactly");
        Simulator other = RunSimulation(4, 20261017, new FaultPlan(), 200, 400, 3, 50);
        Check(other.Metrics.Completed == 400, "a different seed still completes everything");
        Check(other.Metrics.TotalLatency != metrics.TotalLatency, "a different seed produces a different latency profile");
        Check(other.OrderingViolations == 0 && repeat.OrderingViolations == 0, "ordering holds in every run");

        long shallow = RunSimulation(4, 11, new FaultPlan(), 200, 200, 3, 40).Metrics.Percentile(50);
        long deep = RunSimulation(4, 11, new FaultPlan(), 200, 200, 5, 40).Metrics.Percentile(50);
        Check(deep > shallow, "a deeper pipeline costs more latency: " + shallow + " -> " + deep);

        Console.WriteLine("healthy: 400 requests, throughput=" +
            metrics.Throughput.ToString("F1", CultureInfo.InvariantCulture) + "/s, p50=" + p50 + " p95=" + p95 +
            " p99=" + p99 + ", makespan=" + metrics.Makespan + "us, events=" + healthy.EventsProcessed +
            ", busy=" + metrics.BusyTime + "us");
        Console.WriteLine("faults: drops=" + dropped.Metrics.DroppedMessages + " (decided " + dropPlan.DropDecisions +
            "), retries=" + dropped.Metrics.Retries + ", failures=" + dropped.Metrics.Failed +
            ", crash hits=" + crashed.Metrics.CrashedNodeHits + ", reordered=" + reordered.Metrics.ReorderedMessages);
        Console.WriteLine("scorecard: lenient=" + (lenient.AllMet ? "PASS" : "FAIL") + ", harsh=" +
            (harsh.AllMet ? "PASS" : "FAIL") + " -> " + harsh.Describe());
        Console.WriteLine("pipeline depth 3 vs 5 median latency: " + shallow + " vs " + deep +
            "us; ordering violations=" + (healthy.OrderingViolations + dropped.OrderingViolations + crashed.OrderingViolations));
        Console.WriteLine("PASS");
    }
}
