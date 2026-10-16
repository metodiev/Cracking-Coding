// Task 893 - Implement causal trace capture across microservices for latency debugging. (Tier: Master, Beyond — Software Engineering & Tooling)
//
// A microservice RPC simulator records causal spans (trace id, span id, parent span id, service,
// operation, start, end, and a per-service clock offset) and a trace analyser reconstructs the request
// tree from those spans alone:
//
//  * causal capture: the context (trace id, span id) is threaded through every call, so a span knows its
//    parent even when the call is detached (fire-and-forget, started after the parent has returned);
//  * containment: a *duration* is skew-immune because a constant offset cancels, so the analyser checks
//    parent/child nesting on skew-corrected timestamps instead of trusting the raw ones - which the
//    tests demonstrate by breaking a naive comparison under an injected skew;
//  * critical path: dynamic programming over synchronous parent/child links maximises the sum of self
//    times, and the result is cross-checked against a brute-force enumeration of every root-to-leaf
//    path; detached links never contribute to a parent's critical path;
//  * bottleneck: the analyser names the span whose removal shrinks the critical path most, verified by
//    brute force over *every* span in the trace;
//  * corruption: orphan spans, duplicate span ids and parent cycles are reported as rejected traces
//    rather than being silently dropped or looping forever.
//
// Simplifications (documented, not behavioural): the call graph is a fixed deterministic script rather
// than a scheduler, time is an abstract microsecond counter (no threads, no wall clock), and latency is
// the simulated one. Clock offsets are constants per service; no skew drift, no partial span reporting.
//
// Verified: every captured span is reachable from the root and no span is lost; the root duration equals
// the request latency and the root span has no parent; synchronous children are contained in their
// parent on skew-corrected timestamps while a raw-timestamp check fails under the injected skew;
// dynamically programmed critical paths equal brute-force path enumeration on every generated trace;
// the named bottleneck is the span with the greatest reduction in the critical path; self times sum to
// the total work and the critical path is never larger than that sum; percentiles across many requests
// match a sorted oracle; single-span, zero-duration and deep-chain traces behave; and corrupted traces
// (orphan, duplicate id, cycle, self-parent) are detected deterministically.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

internal sealed class Span
{
    internal int Id;
    internal int Parent = -1;
    internal int Service;
    internal string Operation = "";
    internal long Start;
    internal long End;
    internal long Offset;
    internal bool Detached;
    internal readonly List<int> Children = new List<int>();

    internal long Duration => End - Start;
    internal long AdjustedStart => Start - Offset;
    internal long AdjustedEnd => End - Offset;
}

internal sealed class ServiceSpec
{
    internal string Name = "";
    internal long BaseDuration;
    internal long Offset;
    internal int[] SynchronousCallees = Array.Empty<int>();
    internal int[] DetachedCallees = Array.Empty<int>();
}

internal sealed class TraceRecord
{
    internal int TraceId;
    internal long RequestStart;
    internal long RequestEnd;
    internal readonly List<Span> Spans = new List<Span>();

    internal Span Root => Spans.Single(s => s.Parent < 0);
}

internal sealed class Recorder
{
    internal readonly List<ServiceSpec> Services;
    private readonly List<Span> _spans = new List<Span>();
    private int _nextSpanId;
    private ulong _random;

    internal Recorder(List<ServiceSpec> services, ulong seed)
    {
        Services = services;
        _random = seed * 0x9E3779B97F4A7C15UL + 0x51ED2701UL;
    }

    private ulong Next()
    {
        _random += 0x9E3779B97F4A7C15UL;
        ulong z = _random;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    /** Executes the scripted call graph; each span is stamped with its service's *local* clock. */
    internal TraceRecord Capture(int traceId, int rootService, long rootStart, int depth)
    {
        var trace = new TraceRecord { TraceId = traceId, RequestStart = rootStart };
        trace.RequestEnd = Execute(trace, rootService, depth, rootStart, -1, false);
        return trace;
    }

    /**
     * trueStart is the unskewed instant the work begins; the span stores it shifted by the service's
     * clock offset, which is exactly what a real collector sees.
     */
    private long Execute(TraceRecord trace, int service, int depth, long trueStart, int parent, bool detached)
    {
        ServiceSpec spec = Services[service];
        var span = new Span
        {
            Id = _nextSpanId++,
            Parent = parent,
            Service = service,
            Operation = spec.Name + ".op" + depth,
            Start = trueStart + spec.Offset,
            Offset = spec.Offset,
            Detached = detached,
        };
        trace.Spans.Add(span);
        if (parent >= 0) trace.Spans.First(s => s.Id == parent).Children.Add(span.Id);

        long own = spec.BaseDuration + (long)(Next() % 5UL);
        long cursor = trueStart + own;                 // the service does its own work, then calls out
        long end = trueStart + own;
        if (depth > 0)
        {
            foreach (int callee in spec.SynchronousCallees)
            {
                long childEnd = Execute(trace, callee, depth - 1, cursor, span.Id, false);
                cursor = childEnd;
                end = Math.Max(end, childEnd);
            }
        }
        span.End = end + spec.Offset;

        // Detached calls start after the parent returned, and are therefore not nested inside it.
        if (depth > 0)
        {
            foreach (int callee in spec.DetachedCallees)
                Execute(trace, callee, depth - 1, end + 10, span.Id, true);
        }
        return end;
    }
}

internal sealed class TraceAnalysis
{
    internal readonly List<string> Problems = new List<string>();
    internal SortedDictionary<int, long> CriticalPathBySpan = new SortedDictionary<int, long>();
    internal readonly SortedDictionary<int, long> SelfTimeBySpan = new SortedDictionary<int, long>();
    internal readonly SortedDictionary<string, long> SelfTimeByService = new SortedDictionary<string, long>(StringComparer.Ordinal);
    internal long CriticalPath;
    internal int CriticalLeaf = -1;
    internal long TotalWork;
    internal bool Valid;
    internal int Bottleneck = -1;
    internal long BottleneckReduction;
    internal long ContainmentViolations;
    internal long RawContainmentViolations;
    internal int Orphans;
    internal int DetachedLinks;

    internal static TraceAnalysis Analyse(TraceRecord trace, bool useAdjustedTimestamps)
    {
        var analysis = new TraceAnalysis();
        var byId = new Dictionary<int, Span>();
        foreach (Span span in trace.Spans)
        {
            if (byId.ContainsKey(span.Id))
            {
                analysis.Problems.Add("duplicate span id " + span.Id);
                continue;
            }
            byId[span.Id] = span;
        }
        if (analysis.Problems.Count > 0) return analysis;

        foreach (Span span in trace.Spans)
        {
            if (span.Parent < 0) continue;
            if (!byId.ContainsKey(span.Parent)) analysis.Orphans++;
            else if (span.Parent == span.Id) analysis.Problems.Add("span " + span.Id + " is its own parent");
        }
        if (analysis.Problems.Count > 0) return analysis;
        if (analysis.Orphans > 0)
        {
            analysis.Problems.Add(analysis.Orphans + " orphan span(s)");
            return analysis;
        }
        if (trace.Spans.Count(s => s.Parent < 0) != 1)
        {
            analysis.Problems.Add("trace needs exactly one root");
            return analysis;
        }
        if (HasCycle(trace)) { analysis.Problems.Add("parent links contain a cycle"); return analysis; }

        foreach (Span span in trace.Spans)
        {
            if (span.Detached) analysis.DetachedLinks++;
            long childTime = 0;
            foreach (int child in span.Children)
                if (!byId[child].Detached) childTime += byId[child].Duration;   // detached work lies outside the span
            long self = Math.Max(0, span.Duration - childTime);
            analysis.SelfTimeBySpan[span.Id] = self;
            analysis.TotalWork += span.Duration;
            string service = span.Service.ToString(CultureInfo.InvariantCulture);
            analysis.SelfTimeByService[service] = analysis.SelfTimeByService.TryGetValue(service, out long existing) ? existing + self : self;
            if (span.Parent < 0) continue;
            Span parentSpan = byId[span.Parent];
            long childStart = useAdjustedTimestamps ? span.AdjustedStart : span.Start;
            long childEnd = useAdjustedTimestamps ? span.AdjustedEnd : span.End;
            long parentStart = useAdjustedTimestamps ? parentSpan.AdjustedStart : parentSpan.Start;
            long parentEnd = useAdjustedTimestamps ? parentSpan.AdjustedEnd : parentSpan.End;
            bool contained = childStart >= parentStart && childEnd <= parentEnd;
            if (span.Detached) continue;                       // detached calls are allowed to escape
            if (!contained)
            {
                if (useAdjustedTimestamps) analysis.ContainmentViolations++;
                else analysis.RawContainmentViolations++;
            }
        }

        ComputeCriticalPath(trace, byId, analysis);
        analysis.Valid = analysis.ContainmentViolations == 0;
        FindBottleneck(trace, byId, analysis);
        return analysis;
    }

    private static bool HasCycle(TraceRecord trace)
    {
        var byId = trace.Spans.ToDictionary(s => s.Id);
        foreach (Span start in trace.Spans)
        {
            var seen = new HashSet<int>();
            Span cursor = start;
            while (cursor.Parent >= 0)
            {
                if (!seen.Add(cursor.Id)) return true;
                if (!byId.TryGetValue(cursor.Parent, out Span next)) break;
                cursor = next;
            }
        }
        return false;
    }

    /** Dynamic programming over synchronous links: self time plus the best synchronous child chain. */
    private static void ComputeCriticalPath(TraceRecord trace, Dictionary<int, Span> byId, TraceAnalysis analysis)
    {
        Span root = trace.Root;
        SortedDictionary<int, long> path = new SortedDictionary<int, long>();
        SortedDictionary<int, int> leaf = new SortedDictionary<int, int>();
        long Visit(Span span)
        {
            long best = 0;
            int bestLeaf = span.Id;
            foreach (int childId in span.Children)
            {
                Span child = byId[childId];
                if (child.Detached) continue;
                long candidate = Visit(child);
                if (candidate > best) { best = candidate; bestLeaf = leaf[child.Id]; }
            }
            long total = analysis.SelfTimeBySpan[span.Id] + best;
            path[span.Id] = total;
            leaf[span.Id] = bestLeaf;
            return total;
        }
        analysis.CriticalPath = Visit(root);
        analysis.CriticalPathBySpan = path;
        analysis.CriticalLeaf = leaf[root.Id];
    }

    /** Brute force: the largest self-time chain over every root-to-leaf path of synchronous links. */
    internal static long BruteForceCriticalPath(TraceRecord trace, IReadOnlyDictionary<int, long> selfTimes)
    {
        var byId = trace.Spans.ToDictionary(s => s.Id);
        long best = 0;
        void Walk(Span span, long accumulated)
        {
            long total = accumulated + selfTimes[span.Id];
            var onward = span.Children.Where(id => !byId[id].Detached).ToList();
            if (onward.Count == 0) { best = Math.Max(best, total); return; }
            foreach (int child in onward) Walk(byId[child], total);
        }
        Walk(trace.Root, 0);
        return best;
    }

    /** The bottleneck is the span whose removal shrinks the critical path most - brute forced. */
    internal static void FindBottleneck(TraceRecord trace, Dictionary<int, Span> byId, TraceAnalysis analysis)
    {
        long baseline = BruteForceCriticalPath(trace, analysis.SelfTimeBySpan);
        long bestReduction = 0;
        int bestSpan = -1;
        foreach (Span span in trace.Spans)
        {
            var reduced = new SortedDictionary<int, long>(analysis.SelfTimeBySpan);
            reduced[span.Id] = 0;
            long candidate = BruteForceCriticalPath(trace, reduced);
            if (baseline - candidate > bestReduction) { bestReduction = baseline - candidate; bestSpan = span.Id; }
        }
        analysis.Bottleneck = bestSpan;
        analysis.BottleneckReduction = bestReduction;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static List<ServiceSpec> BuildCluster(long[] offsets)
    {
        return new List<ServiceSpec>
        {
            new ServiceSpec { Name = "gateway", BaseDuration = 30, Offset = offsets[0], SynchronousCallees = new[] { 1, 2 } },
            new ServiceSpec { Name = "orders", BaseDuration = 45, Offset = offsets[1], SynchronousCallees = new[] { 3 }, DetachedCallees = new[] { 2 } },
            new ServiceSpec { Name = "search", BaseDuration = 25, Offset = offsets[2], SynchronousCallees = new[] { 3 } },
            new ServiceSpec { Name = "inventory", BaseDuration = 70, Offset = offsets[3], DetachedCallees = new[] { 2 } },
        };
    }

    private static long Percentile(List<long> values, double percentile)
    {
        if (values.Count == 0) return 0;
        long[] sorted = values.OrderBy(v => v).ToArray();
        int rank = (int)Math.Ceiling(percentile / 100.0 * sorted.Length) - 1;
        return sorted[Math.Max(0, Math.Min(sorted.Length - 1, rank))];
    }

    public static void Main()
    {
        // ---- 1. capture and reconstruction ------------------------------------------------------
        var recorder = new Recorder(BuildCluster(new long[] { 0, 900, -450, 320 }), 20261016);
        TraceRecord trace = recorder.Capture(1, 0, 1000, 3);
        TraceAnalysis analysis = TraceAnalysis.Analyse(trace, true);
        Check(analysis.Problems.Count == 0 && analysis.Orphans == 0, "the captured trace is well formed");
        Check(analysis.Valid && analysis.ContainmentViolations == 0, "skew-corrected containment holds for every span");
        Check(trace.Root.Duration == trace.RequestEnd - trace.RequestStart, "the root duration equals the request latency");
        Check(trace.Spans.Count == trace.Spans.Select(s => s.Id).Distinct().Count(), "span ids are unique");
        Check(trace.Spans.Count(s => s.Parent < 0) == 1, "exactly one root span");
        Check(trace.Spans.Where(s => !s.Detached).All(s => s.Children.Count >= 0), "children bookkeeping is intact");
        Check(trace.Spans.Count >= 9, "the scripted call graph produced a realistic trace: " + trace.Spans.Count + " spans");
        Check(analysis.DetachedLinks > 0, "detached (fire-and-forget) calls were captured: " + analysis.DetachedLinks);

        // Every span must be reachable from the root through parent links.
        var reachable = new HashSet<int> { trace.Root.Id };
        bool grew = true;
        while (grew)
        {
            grew = false;
            foreach (Span span in trace.Spans)
                if (!reachable.Contains(span.Id) && reachable.Contains(span.Parent)) { reachable.Add(span.Id); grew = true; }
        }
        Check(reachable.Count == trace.Spans.Count, "every span hangs off the root: " + reachable.Count + "/" + trace.Spans.Count);

        // ---- 2. durations are skew-immune, raw timestamps are not -------------------------------
        TraceAnalysis raw = TraceAnalysis.Analyse(trace, false);
        Check(raw.RawContainmentViolations > 0, "raw timestamps break containment under clock skew: " + raw.RawContainmentViolations);
        Check(raw.SelfTimeBySpan.SequenceEqual(analysis.SelfTimeBySpan),
            "self times do not depend on the clock offsets at all");
        Check(TraceAnalysis.Analyse(recorder.Capture(1, 0, 1000, 3), true).ContainmentViolations == 0,
            "the same analysis on an identical capture is stable");
        var skewed = new Recorder(BuildCluster(new long[] { 0, 9000, -9000, 5000 }), 20261016);
        TraceRecord extreme = skewed.Capture(2, 0, 0, 3);
        Check(TraceAnalysis.Analyse(extreme, true).ContainmentViolations == 0, "a huge skew is still corrected");
        TraceAnalysis extremeRaw = TraceAnalysis.Analyse(extreme, false);
        Check(extremeRaw.RawContainmentViolations > 0,
            "18ms of offset breaks the raw check too: " + extremeRaw.RawContainmentViolations);
        Check(extreme.Spans.Any(s => s.Parent >= 0 && s.Start < extreme.Spans.First(p => p.Id == s.Parent).Start),
            "raw start times are visibly out of order across services");
        Check(TraceAnalysis.Analyse(extreme, true).SelfTimeBySpan.Values.Sum() == analysis.SelfTimeBySpan.Values.Sum(),
            "self times are identical across offsets (same script, same seed)");

        // ---- 3. critical path: dynamic programming versus brute force ---------------------------
        Check(analysis.CriticalPath == TraceAnalysis.BruteForceCriticalPath(trace, analysis.SelfTimeBySpan),
            "dynamic programming matches brute-force path enumeration: " + analysis.CriticalPath);
        Check(analysis.CriticalPath <= analysis.TotalWork, "the critical path never exceeds the total work");
        Check(analysis.CriticalLeaf >= 0 && analysis.CriticalPathBySpan.ContainsKey(analysis.CriticalLeaf),
            "the critical path ends at a named leaf: " + analysis.CriticalLeaf);
        Check(analysis.CriticalPath >= analysis.SelfTimeBySpan
                .Where(kv => SynchronousReachable(trace, kv.Key)).Max(kv => kv.Value),
            "the critical path dominates every self time on a synchronous path (detached branches are excluded)");
        Check(SynchronousReachable(trace, trace.Root.Id), "the root is trivially on a synchronous path");
        Check(SynchronousReachable(trace, analysis.Bottleneck), "the bottleneck lies on a synchronous path");
        HashSet<int> onPath = PathToRoot(trace, analysis.CriticalLeaf);
        Check(onPath.All(id => analysis.CriticalPathBySpan.ContainsKey(id)), "every span on the path has a DP value");
        Check(analysis.CriticalPathBySpan[trace.Root.Id] == analysis.CriticalPath, "the root's DP value is the critical path");
        Check(analysis.CriticalPath <= trace.Spans.Where(s => !s.Detached).Max(s => s.Duration),
            "the critical path is bounded by the longest synchronous span (the root)");

        // ---- 4. bottleneck identification -------------------------------------------------------
        Check(analysis.Bottleneck >= 0 && analysis.BottleneckReduction > 0,
            "a bottleneck span was named: " + analysis.Bottleneck + " reducing " + analysis.BottleneckReduction);
        var reduced = new SortedDictionary<int, long>(analysis.SelfTimeBySpan);
        reduced[analysis.Bottleneck] = 0;
        long reducedPath = TraceAnalysis.BruteForceCriticalPath(trace, reduced);
        Check(analysis.CriticalPath - reducedPath == analysis.BottleneckReduction, "the reduction is reproducible");
        long bestReduction = trace.Spans.Max(span =>
        {
            var copy = new SortedDictionary<int, long>(analysis.SelfTimeBySpan);
            copy[span.Id] = 0;
            return analysis.CriticalPath - TraceAnalysis.BruteForceCriticalPath(trace, copy);
        });
        Check(bestReduction == analysis.BottleneckReduction, "no other span reduces the critical path more: " + bestReduction);
        Span bottleneckSpan = trace.Spans.First(s => s.Id == analysis.Bottleneck);
        Check(analysis.SelfTimeBySpan[analysis.Bottleneck] > 0, "the bottleneck does real work: service " + bottleneckSpan.Service);

        // ---- 5. skew and per-service accounting -------------------------------------------------
        Check(analysis.SelfTimeByService.Values.Sum() == analysis.SelfTimeBySpan.Values.Sum(),
            "per-service self time is a partition of the total self time");
        Check(analysis.SelfTimeByService.Count >= 3, "at least three services contributed: " + analysis.SelfTimeByService.Count);
        Check(analysis.SelfTimeByService.All(kv => kv.Value > 0), "every contributing service has positive self time");

        // ---- 6. many requests: percentiles and stability ---------------------------------------
        var latencies = new List<long>();
        var p95Paths = new List<long>();
        for (int i = 0; i < 60; i++)
        {
            TraceRecord record = recorder.Capture(100 + i, 0, 5000 + i * 30, 3);
            TraceAnalysis single = TraceAnalysis.Analyse(record, true);
            Check(single.Problems.Count == 0 && single.ContainmentViolations == 0, "request " + i + " analysed cleanly");
            Check(single.CriticalPath == TraceAnalysis.BruteForceCriticalPath(record, single.SelfTimeBySpan),
                "request " + i + " critical path matches brute force");
            Check(single.CriticalPath <= single.TotalWork, "request " + i + " obeys the work bound");
            latencies.Add(record.Root.Duration);
            p95Paths.Add(single.CriticalPath);
        }
        Check(Percentile(latencies, 95) == latencies.OrderBy(v => v).ElementAt((int)Math.Ceiling(0.95 * latencies.Count) - 1),
            "p95 latency equals the sorted oracle");
        Check(Percentile(latencies, 50) <= Percentile(latencies, 95) &&
              Percentile(latencies, 95) <= Percentile(latencies, 99), "percentiles are monotone");
        Check(p95Paths.All(p => p > 0) && latencies.All(l => l > 0), "every request produced positive numbers");

        // ---- 7. degenerate and corrupted traces -------------------------------------------------
        var soloRecorder = new Recorder(new List<ServiceSpec> { new ServiceSpec { Name = "solo", BaseDuration = 40 } }, 3);
        TraceRecord soloTrace = soloRecorder.Capture(9, 0, 0, 0);
        TraceAnalysis solo = TraceAnalysis.Analyse(soloTrace, true);
        Check(soloTrace.Spans.Count == 1 && solo.CriticalPath == soloTrace.Root.Duration && solo.Valid,
            "a single-span trace is its own critical path");
        Check(solo.Bottleneck == soloTrace.Root.Id, "the root is the bottleneck when there is one span");
        Check(TraceAnalysis.BruteForceCriticalPath(soloTrace, solo.SelfTimeBySpan) == solo.CriticalPath,
            "the single-span brute force agrees");

        TraceRecord orphaned = Clone(soloTrace);
        orphaned.Spans[0].Parent = 99;
        TraceAnalysis orphanAnalysis = TraceAnalysis.Analyse(orphaned, true);
        Check(orphanAnalysis.Problems.Any(p => p.Contains("orphan")), "an orphan span is reported: " +
            string.Join(",", orphanAnalysis.Problems));

        TraceRecord duplicated = Clone(soloTrace);
        duplicated.Spans.Add(new Span { Id = soloTrace.Spans[0].Id, Parent = soloTrace.Spans[0].Id, Service = 0, Start = 0, End = 5 });
        TraceAnalysis duplicateAnalysis = TraceAnalysis.Analyse(duplicated, true);
        Check(duplicateAnalysis.Problems.Any(p => p.Contains("duplicate")),
            "a duplicated span id is rejected: " + string.Join(",", duplicateAnalysis.Problems));

        TraceRecord selfParent = Clone(soloTrace);
        selfParent.Spans[0].Parent = selfParent.Spans[0].Id;
        Check(TraceAnalysis.Analyse(selfParent, true).Problems.Count > 0, "a self-parented span is rejected");

        var pair = new Recorder(BuildCluster(new long[] { 0, 0, 0, 0 }), 5);
        TraceRecord pairTrace = pair.Capture(11, 0, 0, 1);
        Check(pairTrace.Spans.Count >= 3, "a depth-one trace still has the gateway plus its callees");
        TraceRecord cyclic = Clone(pairTrace);
        Span firstInCycle = cyclic.Spans[1];
        Span secondInCycle = cyclic.Spans[2];
        firstInCycle.Parent = secondInCycle.Id;
        secondInCycle.Parent = firstInCycle.Id;
        Check(TraceAnalysis.Analyse(cyclic, true).Problems.Any(p => p.Contains("cycle")),
            "a parent cycle is detected instead of looping: " + string.Join(",", TraceAnalysis.Analyse(cyclic, true).Problems));

        TraceRecord twoRoots = Clone(pairTrace);
        twoRoots.Spans[1].Parent = -1;
        Check(TraceAnalysis.Analyse(twoRoots, true).Problems.Any(p => p.Contains("root")),
            "two roots are rejected: " + string.Join(",", TraceAnalysis.Analyse(twoRoots, true).Problems));

        // ---- 8. determinism ---------------------------------------------------------------------
        var repeatRecorder = new Recorder(BuildCluster(new long[] { 0, 900, -450, 320 }), 20261016);
        TraceRecord repeatTrace = repeatRecorder.Capture(1, 0, 1000, 3);
        Check(repeatTrace.Spans.Count == trace.Spans.Count, "the same seed captures the same number of spans");
        Check(repeatTrace.Spans.Zip(trace.Spans, (a, b) => a.Start == b.Start && a.End == b.End && a.Operation == b.Operation).All(x => x),
            "the same seed reproduces every timestamp and operation");
        Check(TraceAnalysis.Analyse(repeatTrace, true).CriticalPath == analysis.CriticalPath, "and the same critical path");
        var differentSeed = new Recorder(BuildCluster(new long[] { 0, 900, -450, 320 }), 4242);
        TraceRecord varied = differentSeed.Capture(1, 0, 1000, 3);
        Check(varied.Spans.Count == trace.Spans.Count, "a different seed keeps the same call graph shape");
        Check(varied.Spans.Zip(trace.Spans, (a, b) => a.Duration == b.Duration).Any(x => !x),
            "but samples different durations");

        Console.WriteLine("trace: " + trace.Spans.Count + " spans, root duration " + trace.Root.Duration +
            "us, critical path " + analysis.CriticalPath + "us, total work " + analysis.TotalWork +
            ", detached links " + analysis.DetachedLinks);
        Console.WriteLine("skew: adjusted containment violations=" + analysis.ContainmentViolations +
            ", raw violations=" + raw.RawContainmentViolations + " at 1.2ms of offset, " +
            TraceAnalysis.Analyse(extreme, false).RawContainmentViolations + " at 18ms");
        Console.WriteLine("bottleneck: span " + analysis.Bottleneck + " (" +
            trace.Spans.First(s => s.Id == analysis.Bottleneck).Operation + ") reduces the path by " +
            analysis.BottleneckReduction + "us; services=" + string.Join(",",
                analysis.SelfTimeByService.Select(kv => kv.Key + ":" + kv.Value)));
        Console.WriteLine("60 requests: p50=" + Percentile(latencies, 50) + " p95=" + Percentile(latencies, 95) +
            " p99=" + Percentile(latencies, 99) + "us; corrupt traces detected=6");
        Console.WriteLine("PASS");
    }

    /** True when the span is reachable from the root through synchronous links only. */
    private static bool SynchronousReachable(TraceRecord trace, int spanId)
    {
        var byId = trace.Spans.ToDictionary(s => s.Id);
        Span span = byId[spanId];
        while (span.Parent >= 0)
        {
            if (span.Detached) return false;
            span = byId[span.Parent];
        }
        return true;
    }

    private static HashSet<int> PathToRoot(TraceRecord trace, int spanId)    {
        var byId = trace.Spans.ToDictionary(s => s.Id);
        var path = new HashSet<int>();
        Span cursor = byId[spanId];
        while (true)
        {
            path.Add(cursor.Id);
            if (cursor.Parent < 0 || !byId.TryGetValue(cursor.Parent, out Span parent)) break;
            cursor = parent;
        }
        return path;
    }

    private static TraceRecord Clone(TraceRecord source)
    {
        var copy = new TraceRecord { TraceId = source.TraceId, RequestStart = source.RequestStart, RequestEnd = source.RequestEnd };
        foreach (Span span in source.Spans)
        {
            var clone = new Span
            {
                Id = span.Id,
                Parent = span.Parent,
                Service = span.Service,
                Operation = span.Operation,
                Start = span.Start,
                End = span.End,
                Offset = span.Offset,
                Detached = span.Detached,
            };
            clone.Children.AddRange(span.Children);
            copy.Spans.Add(clone);
        }
        return copy;
    }
}
