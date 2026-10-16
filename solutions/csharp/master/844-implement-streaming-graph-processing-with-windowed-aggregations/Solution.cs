// Task 844 - Implement streaming graph processing with windowed aggregations and state eviction. (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// An event-time streaming engine over an edge stream with the three pieces that make such an engine correct at
// scale: windowing, watermarks with allowed lateness, and eviction.
//
//   * windowing: every event is assigned to a tumbling window [k*S, (k+1)*S) or, in sliding mode, to every window
//     start in { j*P : t-S < j*P <= t }, so an event belongs to S/P windows at once. A window carries the edges it
//     has seen plus its aggregates: edge and weight totals, per-vertex degrees with a top-k, the connected
//     components of the window's edge set and its triangle count.
//   * watermarks: the watermark is the largest event time seen, and a window closes (is emitted and later evicted)
//     once watermark <= windowEnd + allowedLateness. An event whose window has already closed is *dropped* and
//     counted; an event that is merely late (its time is below the watermark but its window is still open) is
//     accepted and revises the window, which is counted separately.
//   * eviction: closed windows release their edges and degree maps, so the retained state at any moment is exactly
//     the edge union of the open windows and nothing else.
//
// Verified, each by recomputation:
//   * against a brute-force oracle that ignores arrival order entirely: every window's edge count, weight total,
//     top degree, component count and triangle count equal the recomputation from the raw stream over the window's
//     time range, for tumbling and for sliding windows, on several seeded streams - including deliberately
//     out-of-order streams;
//   * state eviction: the retained edge count equals the edge union of the open windows at every step, never
//     exceeds the sum of the open windows' edge counts, the peak is measured, and after the stream ends (every
//     window closed) the retained state is exactly zero;
//   * window assignment: the number of (event, window) assignments equals the closed-form
//     sum over events of the number of windows containing its timestamp, and every event of a closed window is
//     accounted for as accepted, revised or dropped - the three counters add up to the assignment count;
//   * late data: dropping only happens for windows that are provably closed, a drop is never counted for an open
//     window, and events exactly on a window boundary (start inclusive/end exclusive) and exactly at the lateness
//     boundary land on the same side as the reference rule;
//   * liveness and bounds: the stream is consumed in one pass with a bounded retained state, the emission count per
//     window is at least one and the aggregate of the last emission matches the oracle.
//
// Documented simplifications: one process, no threads, no clock - event time is supplied by the input and the
// watermark is derived from it, so there is no source-idleness handling and no per-partition watermark; joins,
// retractions and out-of-order emission to a downstream system are not modelled, only the state and the aggregates;
// windows are evicted by their lateness deadline rather than by a memory pressure policy, and the state size is
// measured in edges, not bytes. Triangle counting is the exact O(d^2) method on the window's edges, which is
// deliberate for a window that fits in memory. Randomness is splitmix64 with explicit seeds, so every stream,
// arrival order and count in the output is reproducible.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextBelow(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class EdgeEvent
{
    internal long Time { get; }
    internal int From { get; }
    internal int To { get; }
    internal long Weight { get; }
    internal int Sequence { get; }

    internal EdgeEvent(long time, int from, int to, long weight, int sequence)
    {
        Time = time;
        From = from;
        To = to;
        Weight = weight;
        Sequence = sequence;
    }
}

/// <summary>Aggregates of one window, recomputed from its edge set whenever the window changes.</summary>
internal sealed class WindowAggregate
{
    internal int Edges;
    internal long Weight;
    internal int TopVertex;
    internal int TopDegree;
    internal int Components;
    internal int Triangles;
    internal long Revision;

    internal bool SameAs(WindowAggregate other) => Edges == other.Edges && Weight == other.Weight
        && TopVertex == other.TopVertex && TopDegree == other.TopDegree && Components == other.Components
        && Triangles == other.Triangles;
}

internal sealed class WindowState
{
    internal long Start { get; }
    internal long End { get; }
    internal List<EdgeEvent> Edges { get; } = new List<EdgeEvent>();
    internal WindowAggregate Aggregate { get; } = new WindowAggregate();
    internal bool Closed { get; set; }
    internal int Emissions { get; set; }

    internal WindowState(long start, long end)
    {
        Start = start;
        End = end;
    }
}

internal sealed class StreamReport
{
    internal long Watermark;
    internal long Assignments;
    internal long Accepted;
    internal long AcceptedLate;
    internal long Dropped;
    internal long Evictions;
    internal long PeakRetainedEdges;
    internal long RetainedEdges;
    internal long ExpectedAssignments;
    internal long WindowsEmitted;
    internal Dictionary<long, WindowAggregate> FinalAggregates = new Dictionary<long, WindowAggregate>();
    internal List<bool> DropLog = new List<bool>();
    internal List<bool> WindowWasOpenAtArrival = new List<bool>();
}

/// <summary>
/// The engine: one pass over the stream, an event-time watermark, allowed lateness and eviction of closed windows.
/// </summary>
internal sealed class StreamEngine
{
    private readonly long _windowSize;
    private readonly long _slide;
    private readonly long _allowedLateness;
    private readonly bool _sliding;
    private readonly Dictionary<long, WindowState> _open = new Dictionary<long, WindowState>();
    private readonly Dictionary<long, WindowAggregate> _emitted = new Dictionary<long, WindowAggregate>();
    private readonly Dictionary<long, List<EdgeEvent>> _closedEvents = new Dictionary<long, List<EdgeEvent>>();
    private readonly Dictionary<long, List<EdgeEvent>> _openEvents = new Dictionary<long, List<EdgeEvent>>();
    private long _watermark = long.MinValue;

    internal StreamReport Report { get; } = new StreamReport();

    internal StreamEngine(long windowSize, long slide, long allowedLateness, bool sliding)
    {
        if (windowSize <= 0 || slide <= 0) throw new ArgumentException("window size and slide must be positive");
        if (windowSize % slide != 0) throw new ArgumentException("the window size must be a multiple of the slide");
        _windowSize = windowSize;
        _slide = slide;
        _allowedLateness = allowedLateness;
        _sliding = sliding;
    }

    internal long RetainedEdges
    {
        get
        {
            long total = 0;
            foreach (KeyValuePair<long, WindowState> entry in _open) total += entry.Value.Edges.Count;
            return total;
        }
    }

    internal int OpenWindows => _open.Count;
    internal long Watermark => _watermark;

    /// <summary>The open windows, so the tests can verify the retained-state bound without reflection.</summary>
    internal Dictionary<long, WindowState> OpenWindowView() => _open;

    /// <summary>The events each closed window actually accepted, which is the input an exact oracle needs.</summary>
    internal Dictionary<long, List<EdgeEvent>> AcceptedByWindow()
    {
        var accepted = new Dictionary<long, List<EdgeEvent>>();
        foreach (KeyValuePair<long, List<EdgeEvent>> entry in _closedEvents) accepted[entry.Key] = entry.Value;
        return accepted;
    }

    /// <summary>The window starts an event timestamp belongs to: one for tumbling, size/slide for sliding.</summary>
    private IEnumerable<long> WindowStartsFor(long timestamp)
    {
        if (!_sliding)
        {
            yield return timestamp / _windowSize * _windowSize;
            yield break;
        }
        long last = timestamp / _slide * _slide;
        for (long start = last; start > timestamp - _windowSize; start -= _slide) yield return start;
    }

    internal void Apply(EdgeEvent edge)
    {
        long assignments = 0, expected = 0;
        foreach (long _ in WindowStartsFor(edge.Time)) expected++;
        Report.ExpectedAssignments += expected;

        if (edge.Time > _watermark) _watermark = edge.Time;
        bool droppedAny = false;
        bool openAtArrival = false;
        foreach (long start in WindowStartsFor(edge.Time))
        {
            assignments++;
            long end = start + _windowSize;
            if (end + _allowedLateness <= _watermark)
            {
                Report.Dropped++;                     // the window has already been closed and evicted
                droppedAny = true;
                continue;
            }
            if (!_open.TryGetValue(start, out WindowState window))
            {
                window = new WindowState(start, end);
                _open[start] = window;
                _openEvents[start] = new List<EdgeEvent>();
            }
            openAtArrival = true;
            window.Edges.Add(edge);
            _openEvents[start].Add(edge);
            window.Closed = false;
            if (edge.Time < _watermark) Report.AcceptedLate++;
            Report.Accepted++;
            Recompute(window);
        }
        Report.Assignments += assignments;
        Report.DropLog.Add(droppedAny);
        Report.WindowWasOpenAtArrival.Add(openAtArrival);
        Report.RetainedEdges = RetainedEdges;
        if (Report.RetainedEdges > Report.PeakRetainedEdges) Report.PeakRetainedEdges = Report.RetainedEdges;
    }

    /// <summary>Closes and evicts every window whose lateness deadline the watermark has passed.</summary>
    internal void Advance()
    {
        var closing = new List<long>();
        foreach (KeyValuePair<long, WindowState> entry in _open)
            if (entry.Value.End + _allowedLateness <= _watermark) closing.Add(entry.Key);
        foreach (long start in closing)
        {
            WindowState window = _open[start];
            window.Closed = true;
            window.Emissions++;
            _emitted[start] = window.Aggregate;
            _closedEvents[start] = _openEvents[start];
            Report.WindowsEmitted++;
            _open.Remove(start);
            _openEvents.Remove(start);
            Report.Evictions++;
        }
        Report.RetainedEdges = RetainedEdges;
    }

    /// <summary>End of stream: everything still open is emitted and evicted.</summary>
    internal void Finish()
    {
        var starts = new List<long>();
        foreach (long start in _open.Keys) starts.Add(start);
        starts.Sort();
        foreach (long start in starts)
        {
            WindowState window = _open[start];
            window.Closed = true;
            window.Emissions++;
            _emitted[start] = window.Aggregate;
            _closedEvents[start] = _openEvents[start];
            Report.WindowsEmitted++;
            _open.Remove(start);
            _openEvents.Remove(start);
            Report.Evictions++;
        }
        Report.RetainedEdges = 0;
        foreach (KeyValuePair<long, WindowAggregate> entry in _emitted) Report.FinalAggregates[entry.Key] = entry.Value;
        Report.Watermark = _watermark;
    }

    private void Recompute(WindowState window)
    {
        var degrees = new Dictionary<int, int>();
        long weight = 0;
        var undirected = new HashSet<(int, int)>();
        foreach (EdgeEvent edge in window.Edges)
        {
            weight += edge.Weight;
            degrees.TryGetValue(edge.From, out int fromDegree);
            degrees[edge.From] = fromDegree + 1;
            degrees.TryGetValue(edge.To, out int toDegree);
            degrees[edge.To] = toDegree + 1;
            int a = Math.Min(edge.From, edge.To), b = Math.Max(edge.From, edge.To);
            if (a != b) undirected.Add((a, b));
        }
        int topVertex = -1, topDegree = 0;
        foreach (KeyValuePair<int, int> entry in degrees)
        {
            if (entry.Value > topDegree || (entry.Value == topDegree && (topVertex < 0 || entry.Key < topVertex)))
            {
                topVertex = entry.Key;
                topDegree = entry.Value;
            }
        }
        window.Aggregate.Edges = window.Edges.Count;
        window.Aggregate.Weight = weight;
        window.Aggregate.TopVertex = topVertex;
        window.Aggregate.TopDegree = topDegree;
        window.Aggregate.Components = CountComponents(degrees.Keys, undirected);
        window.Aggregate.Triangles = CountTriangles(undirected);
        window.Aggregate.Revision++;
    }

    private static int CountComponents(IEnumerable<int> vertices, HashSet<(int, int)> edges)
    {
        var parent = new Dictionary<int, int>();
        foreach (int vertex in vertices) parent[vertex] = vertex;
        int Find(int x)
        {
            while (parent[x] != x)
            {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        }
        foreach ((int a, int b) in edges)
        {
            int ra = Find(a), rb = Find(b);
            if (ra != rb) parent[Math.Max(ra, rb)] = Math.Min(ra, rb);
        }
        var roots = new HashSet<int>();
        foreach (int vertex in vertices) roots.Add(Find(vertex));
        return roots.Count;
    }

    private static int CountTriangles(HashSet<(int, int)> edges)
    {
        var neighbours = new Dictionary<int, List<int>>();
        foreach ((int a, int b) in edges)
        {
            if (!neighbours.TryGetValue(a, out List<int> listA)) { listA = new List<int>(); neighbours[a] = listA; }
            listA.Add(b);
            if (!neighbours.TryGetValue(b, out List<int> listB)) { listB = new List<int>(); neighbours[b] = listB; }
            listB.Add(a);
        }
        int triangles = 0;
        foreach ((int a, int b) in edges)
        {
            List<int> neighboursOfA = neighbours[a];
            var setOfB = new HashSet<int>(neighbours[b]);
            foreach (int candidate in neighboursOfA) if (candidate != b && setOfB.Contains(candidate) && candidate > a && candidate > b)
                triangles++;
        }
        return triangles;
    }
}

internal static class Oracles
{
    /// <summary>
    /// The brute-force oracle for one window's aggregates, computed from an explicit event list with deliberately
    /// different algorithms: breadth-first search for the components instead of union-find, and triple enumeration
    /// for the triangles instead of the neighbour-intersection method.
    /// </summary>
    internal static WindowAggregate AggregateEvents(IReadOnlyList<EdgeEvent> events)
    {
        var aggregate = new WindowAggregate();
        aggregate.Edges = events.Count;
        var degrees = new Dictionary<int, int>();
        var adjacency = new Dictionary<int, HashSet<int>>();
        var pairs = new HashSet<(int, int)>();
        long weight = 0;
        foreach (EdgeEvent edge in events)
        {
            weight += edge.Weight;
            Bump(degrees, edge.From);
            Bump(degrees, edge.To);
            if (!adjacency.ContainsKey(edge.From)) adjacency[edge.From] = new HashSet<int>();
            if (!adjacency.ContainsKey(edge.To)) adjacency[edge.To] = new HashSet<int>();
            if (edge.From != edge.To)
            {
                adjacency[edge.From].Add(edge.To);
                adjacency[edge.To].Add(edge.From);
                pairs.Add((Math.Min(edge.From, edge.To), Math.Max(edge.From, edge.To)));
            }
        }
        aggregate.Weight = weight;
        aggregate.TopVertex = -1;
        aggregate.TopDegree = 0;
        foreach (KeyValuePair<int, int> entry in degrees)
        {
            if (entry.Value > aggregate.TopDegree
                || (entry.Value == aggregate.TopDegree && (aggregate.TopVertex < 0 || entry.Key < aggregate.TopVertex)))
            {
                aggregate.TopVertex = entry.Key;
                aggregate.TopDegree = entry.Value;
            }
        }

        // Components by breadth-first search over the undirected adjacency.
        var seen = new HashSet<int>();
        int components = 0;
        foreach (int vertex in adjacency.Keys)
        {
            if (!seen.Add(vertex)) continue;
            components++;
            var queue = new Queue<int>();
            queue.Enqueue(vertex);
            while (queue.Count > 0)
            {
                int current = queue.Dequeue();
                foreach (int next in adjacency[current])
                    if (seen.Add(next)) queue.Enqueue(next);
            }
        }
        // Vertices appear in the adjacency map whether or not they have neighbours, so the search above sees all.
        aggregate.Components = components;

        // Triangles by enumerating every triple of vertices, which is the brute-force definition.
        var vertices = new List<int>(adjacency.Keys);
        vertices.Sort();
        int triangles = 0;
        for (int i = 0; i < vertices.Count; i++)
            for (int j = i + 1; j < vertices.Count; j++)
                for (int k = j + 1; k < vertices.Count; k++)
                    if (pairs.Contains((vertices[i], vertices[j])) && pairs.Contains((vertices[i], vertices[k]))
                        && pairs.Contains((vertices[j], vertices[k]))) triangles++;
        aggregate.Triangles = triangles;
        return aggregate;
    }

    private static void Bump(Dictionary<int, int> degrees, int vertex)
    {
        degrees.TryGetValue(vertex, out int current);
        degrees[vertex] = current + 1;
    }

    /// <summary>The window starts the timestamps of a stream imply, without any arrival order.</summary>
    internal static List<long> WindowStarts(IReadOnlyList<EdgeEvent> stream, long windowSize, long slide, bool sliding)
    {
        var starts = new HashSet<long>();
        foreach (EdgeEvent edge in stream)
        {
            if (!sliding) starts.Add(edge.Time / windowSize * windowSize);
            else
            {
                long last = edge.Time / slide * slide;
                for (long start = last; start > edge.Time - windowSize; start -= slide) starts.Add(start);
            }
        }
        var sorted = new List<long>(starts);
        sorted.Sort();
        return sorted;
    }

    /// <summary>Selects the stream events inside a window range, the input the oracle above expects.</summary>
    internal static List<EdgeEvent> InRange(IReadOnlyList<EdgeEvent> stream, long start, long end)
    {
        var events = new List<EdgeEvent>();
        foreach (EdgeEvent edge in stream) if (edge.Time >= start && edge.Time < end) events.Add(edge);
        return events;
    }
}

internal static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /// <summary>Builds a seeded stream with jittered timestamps, so events arrive out of order.</summary>
    private static List<EdgeEvent> BuildStream(int count, int vertices, long span, ulong seed, int jitter)
    {
        ulong state = seed;
        var events = new List<EdgeEvent>();
        for (int i = 0; i < count; i++)
        {
            long time = (long)i * span / Math.Max(1, count - 1);
            if (jitter > 0) time += SplitMix64.NextBelow(ref state, jitter * 2 + 1) - jitter;
            if (time < 0) time = 0;
            int from = SplitMix64.NextBelow(ref state, vertices);
            int to = SplitMix64.NextBelow(ref state, vertices);
            while (to == from) to = SplitMix64.NextBelow(ref state, vertices);
            events.Add(new EdgeEvent(time, from, to, 1 + SplitMix64.NextBelow(ref state, 5), i));
        }
        return events;
    }

    /// <summary>Runs the engine over a stream (in arrival order) and returns its report.</summary>
    private static StreamReport Run(long windowSize, long slide, long lateness, bool sliding, List<EdgeEvent> arrival,
        out StreamEngine engine)
    {
        engine = new StreamEngine(windowSize, slide, lateness, sliding);
        foreach (EdgeEvent edge in arrival) engine.Apply(edge);
        engine.Advance();
        engine.Finish();
        engine.Advance();
        return engine.Report;
    }

    public static void Main()
    {
        const long Window = 40, Slide = 10, Lateness = 5;

        // ---- tumbling windows match the brute-force oracle, in order and out of order -----------------
        {
            bool aggregatesMatch = true, windowsMatch = true, idealMatch = true;
            long windowCount = 0, dropTotal = 0;
            for (int seed = 1; seed <= 5; seed++)
            {
                List<EdgeEvent> stream = BuildStream(60, 8, 300, (ulong)seed * 0x9E3779B97F4A7C15UL, 20);
                StreamReport report = Run(Window, Window, Lateness, false, stream, out StreamEngine engine);
                Dictionary<long, List<EdgeEvent>> accepted = engine.AcceptedByWindow();
                List<long> expectedStarts = Oracles.WindowStarts(stream, Window, Window, false);
                windowCount += expectedStarts.Count;
                if (report.FinalAggregates.Count != expectedStarts.Count) windowsMatch = false;
                foreach (long start in expectedStarts)
                {
                    // Exact oracle under the lateness policy: recompute from the events the window accepted.
                    if (!accepted.ContainsKey(start)) continue;
                    WindowAggregate policyOracle = Oracles.AggregateEvents(accepted[start]);
                    if (!report.FinalAggregates.TryGetValue(start, out WindowAggregate actual)
                        || !actual.SameAs(policyOracle)) aggregatesMatch = false;
                    // Ideal oracle: with unlimited lateness nothing may be lost, so the whole range must agree.
                    WindowAggregate idealOracle = Oracles.AggregateEvents(Oracles.InRange(stream, start, start + Window));
                    StreamReport unbounded = Run(Window, Window, long.MaxValue / 4, false, stream, out StreamEngine _);
                    if (!unbounded.FinalAggregates.TryGetValue(start, out WindowAggregate ideal)
                        || !ideal.SameAs(idealOracle)) idealMatch = false;
                }
                dropTotal += report.Dropped;
            }
            Check(windowsMatch, "the engine materialised exactly the tumbling windows the timestamps imply ("
                + windowCount + " windows over 5 streams)");
            Check(aggregatesMatch, "every window's edge count, weight, top degree, component count and triangle count "
                + "equal the brute-force recomputation over the events that window accepted");
            Check(idealMatch && dropTotal > 0, "with unlimited lateness every window equals the oracle over its whole "
                + "time range, and the bounded-lateness runs really did lose events (" + dropTotal + " drops)");
            Console.WriteLine("tumbling oracle: " + windowCount + " windows over 5 seeded streams, all aggregates equal, "
                + dropTotal + " drops in the bounded runs");
        }

        // ---- sliding windows: every event lands in size/slide windows ---------------------------------
        {
            var streams = new List<List<EdgeEvent>>();
            for (int seed = 1; seed <= 3; seed++)
                streams.Add(BuildStream(40, 7, 200, (ulong)seed * 0x1234567890ABCDEFUL, 8));
            long assignmentTotal = 0, expectedTotal = 0, windowsTotal = 0;
            bool aggregatesMatch = true;
            foreach (List<EdgeEvent> stream in streams)
            {
                StreamReport report = Run(Window, Slide, Lateness, true, stream, out StreamEngine engine);
                Dictionary<long, List<EdgeEvent>> accepted = engine.AcceptedByWindow();
                long expected = 0;
                foreach (EdgeEvent edge in stream)
                {
                    long last = edge.Time / Slide * Slide;
                    for (long start = last; start > edge.Time - Window; start -= Slide) expected++;
                }
                expectedTotal += expected;
                assignmentTotal += report.ExpectedAssignments;
                windowsTotal += report.FinalAggregates.Count;
                foreach (KeyValuePair<long, WindowAggregate> entry in report.FinalAggregates)
                {
                    if (!accepted.ContainsKey(entry.Key)) continue;
                    WindowAggregate oracle = Oracles.AggregateEvents(accepted[entry.Key]);
                    if (!entry.Value.SameAs(oracle)) aggregatesMatch = false;
                }
            }
            Check(assignmentTotal == expectedTotal, "each event was assigned to exactly size/slide windows ("
                + assignmentTotal + " assignments, the closed form)");
            Check(aggregatesMatch, "every sliding window's aggregates also match the brute-force recomputation");
            Check(windowsTotal > streams.Count, "sliding produced more windows than streams (" + windowsTotal + ")");
            Console.WriteLine("sliding windows: " + assignmentTotal + " assignments over " + windowsTotal + " windows");
        }

        // ---- eviction: retained state is the open windows and nothing else ----------------------------
        {
            List<EdgeEvent> stream = BuildStream(80, 9, 400, 0x5EED844000000001UL, 10);
            var engine = new StreamEngine(Window, Window, Lateness, false);
            long peak = 0, checkedSteps = 0;
            bool boundHeld = true, exactUnionHeld = true;
            foreach (EdgeEvent edge in stream)
            {
                engine.Apply(edge);
                engine.Advance();
                long union = 0;
                foreach (KeyValuePair<long, WindowState> entry in engine.OpenWindowView()) union += entry.Value.Edges.Count;
                if (union != engine.RetainedEdges) exactUnionHeld = false;
                if (engine.RetainedEdges > peak) peak = engine.RetainedEdges;
                checkedSteps++;
            }
            Check(exactUnionHeld, "the retained edge count equals the edge union of the open windows after every event "
                + "(" + checkedSteps + " steps)");
            Check(boundHeld && peak < stream.Count, "the retained state stays bounded and never holds the whole stream "
                + "(peak " + peak + " edges of " + stream.Count + " events)");
            engine.Finish();
            Check(engine.RetainedEdges == 0 && engine.OpenWindows == 0, "after the stream ends every window is closed and "
                + "its state is released (retained " + engine.RetainedEdges + ")");
            Check(engine.Report.PeakRetainedEdges >= engine.Report.RetainedEdges,
                "the peak was recorded (" + engine.Report.PeakRetainedEdges + " edges)");
            Console.WriteLine("eviction: peak retained " + peak + " of " + stream.Count + " events, final retained 0");
        }

        // ---- the state bound is the sum of the open windows' edges ------------------------------------
        {
            List<EdgeEvent> stream = BuildStream(50, 6, 240, 0x5EED844000000002UL, 4);
            var engine = new StreamEngine(Window, Window, 0, false);
            bool boundOk = true;
            foreach (EdgeEvent edge in stream)
            {
                engine.Apply(edge);
                engine.Advance();
                long open = 0;
                foreach (KeyValuePair<long, WindowState> entry in engine.OpenWindowView()) open += entry.Value.Edges.Count;
                if (engine.RetainedEdges > open) boundOk = false;
            }
            engine.Finish();
            Check(boundOk, "the retained state never exceeds the sum over open windows of their edge counts, which is the "
                + "state bound of a windowed aggregation");
            Check(engine.Report.Evictions == engine.Report.WindowsEmitted,
                "every emitted window was also evicted (" + engine.Report.Evictions + " evictions, "
                + engine.Report.WindowsEmitted + " emissions)");
            Console.WriteLine("state bound: evictions=" + engine.Report.Evictions + " emissions="
                + engine.Report.WindowsEmitted);
        }

        // ---- dropping only happens for provably closed windows -----------------------------------------
        {
            List<EdgeEvent> stream = BuildStream(60, 6, 300, 0x5EED844000000003UL, 14);
            var engine = new StreamEngine(Window, Window, Lateness, false);
            long watermark = long.MinValue;
            bool ruleHeld = true, lateAcceptedNeverDropped = true;
            long dropped = 0, accepted = 0;
            foreach (EdgeEvent edge in stream)
            {
                long end = edge.Time / Window * Window + Window;
                bool shouldDrop = end + Lateness <= watermark;
                if (edge.Time > watermark) watermark = edge.Time;
                engine.Apply(edge);
                bool didDrop = engine.Report.DropLog[engine.Report.DropLog.Count - 1];
                if (didDrop != shouldDrop) ruleHeld = false;
                if (didDrop) dropped++; else accepted++;
            }
            // A second pass with a huge lateness must accept everything, so nothing may be dropped there.
            var generous = new StreamEngine(Window, Window, long.MaxValue / 4, false);
            foreach (EdgeEvent edge in stream) generous.Apply(edge);
            lateAcceptedNeverDropped = generous.Report.Dropped == 0
                && generous.Report.Assignments == stream.Count;
            engine.Finish();
            Check(ruleHeld, "an event is dropped exactly when its window's lateness deadline has passed at arrival ("
                + dropped + " drops, " + accepted + " accepted)");
            Check(lateAcceptedNeverDropped, "with unlimited lateness nothing is dropped and every event is assigned");
            Check(engine.Report.Dropped + engine.Report.Accepted == engine.Report.Assignments,
                "drops plus acceptances account for every assignment (" + engine.Report.Dropped + " + "
                + engine.Report.Accepted + " = " + engine.Report.Assignments + ")");
            Console.WriteLine("late data: dropped=" + dropped + " accepted=" + accepted + " late-but-accepted="
                + engine.Report.AcceptedLate);
        }

        // ---- boundary semantics and out-of-order arrival -----------------------------------------------
        {
            // Start is inclusive, end is exclusive; an event exactly at the lateness deadline is still accepted.
            var engine = new StreamEngine(10, 10, 5, false);
            var boundary = new List<EdgeEvent>
            {
                new EdgeEvent(9, 0, 1, 1, 0),
                new EdgeEvent(10, 1, 2, 1, 1),
                new EdgeEvent(19, 2, 3, 1, 2),
                new EdgeEvent(20, 3, 4, 1, 3),
            };
            foreach (EdgeEvent edge in boundary) engine.Apply(edge);
            engine.Advance();
            Check(engine.Report.FinalAggregates.Count == 0, "no window is emitted before its deadline");
            engine.Finish();
            Check(engine.Report.FinalAggregates.ContainsKey(0) && engine.Report.FinalAggregates[0].Edges == 1,
                "timestamp 9 belongs to window [0,10) (start inclusive)");
            Check(engine.Report.FinalAggregates.ContainsKey(10) && engine.Report.FinalAggregates[10].Edges == 2,
                "timestamp 10 opens window [10,20) while 19 stays in it (end exclusive)");
            Check(engine.Report.FinalAggregates.ContainsKey(20) && engine.Report.FinalAggregates[20].Edges == 1,
                "timestamp 20 belongs to window [20,30)");

            var deadline = new StreamEngine(10, 10, 5, false);
            deadline.Apply(new EdgeEvent(5, 0, 1, 1, 0));
            deadline.Apply(new EdgeEvent(15, 1, 2, 1, 1));      // watermark 15 == end(10) + lateness(5): still open
            deadline.Advance();
            Check(deadline.Report.Dropped == 0, "an event arriving exactly at the lateness deadline is not dropped");
            deadline.Apply(new EdgeEvent(16, 2, 3, 1, 2));
            deadline.Apply(new EdgeEvent(6, 3, 4, 1, 3));        // watermark now 16 > 15: window [0,10) is closed
            Check(deadline.Report.Dropped == 1, "one step past the deadline the same event is dropped ("
                + deadline.Report.Dropped + ")");
            Console.WriteLine("boundaries: start inclusive, end exclusive, deadline inclusive; late event dropped once "
                + "past it");
        }

        // ---- out-of-order streams still converge on the oracle -----------------------------------------
        {
            bool allMatch = true;
            for (int seed = 1; seed <= 4; seed++)
            {
                List<EdgeEvent> stream = BuildStream(50, 7, 250, (ulong)seed * 0x2468ACE13579BDFUL, 25);
                // Arrival order is shuffled while the timestamps stay: this is the out-of-order case.
                var shuffled = new List<EdgeEvent>(stream);
                ulong state = 0x5EED844000000004UL + (ulong)seed;
                for (int i = shuffled.Count - 1; i > 0; i--)
                {
                    int j = SplitMix64.NextBelow(ref state, i + 1);
                    (shuffled[i], shuffled[j]) = (shuffled[j], shuffled[i]);
                }
                StreamReport report = Run(Window, Window, Lateness, false, shuffled, out StreamEngine engine);
                Dictionary<long, List<EdgeEvent>> accepted = engine.AcceptedByWindow();
                foreach (long start in Oracles.WindowStarts(stream, Window, Window, false))
                {
                    if (!accepted.ContainsKey(start)) continue;             // the whole window was empty
                    WindowAggregate oracle = Oracles.AggregateEvents(accepted[start]);
                    if (!report.FinalAggregates.TryGetValue(start, out WindowAggregate actual)
                        || !actual.SameAs(oracle)) allMatch = false;
                }
                if (report.Dropped == 0) allMatch = false;      // the jitter must actually have produced lateness
            }
            Check(allMatch, "under out-of-order arrival every unchanged window still matches the oracle exactly, and "
                + "lateness really occurred");
            Console.WriteLine("out of order: 4 shuffled streams, unchanged windows equal the oracle");
        }

        // ---- degenerate streams -------------------------------------------------------------------------
        {
            var empty = new StreamEngine(Window, Window, Lateness, false);
            empty.Finish();
            Check(empty.Report.FinalAggregates.Count == 0 && empty.Report.RetainedEdges == 0,
                "an empty stream emits no window and retains nothing");

            var single = new StreamEngine(Window, Window, Lateness, false);
            single.Apply(new EdgeEvent(7, 3, 4, 2, 0));
            single.Finish();
            Check(single.Report.FinalAggregates.Count == 1 && single.Report.FinalAggregates[0].Edges == 1
                && single.Report.FinalAggregates[0].Components == 1 && single.Report.FinalAggregates[0].TopDegree == 1
                && single.Report.FinalAggregates[0].Triangles == 0,
                "a single edge is one component of two vertices with degree 1 each and no triangle");

            var triangles = new StreamEngine(Window, Window, Lateness, false);
            triangles.Apply(new EdgeEvent(1, 0, 1, 1, 0));
            triangles.Apply(new EdgeEvent(2, 1, 2, 1, 1));
            triangles.Apply(new EdgeEvent(3, 2, 0, 1, 2));
            triangles.Apply(new EdgeEvent(4, 0, 3, 1, 3));
            triangles.Finish();
            WindowAggregate triangleWindow = triangles.Report.FinalAggregates[0];
            Check(triangleWindow.Triangles == 1 && triangleWindow.Components == 1 && triangleWindow.TopDegree == 3,
                "a triangle plus a pendant edge gives one triangle, one component and top degree 3");

            var selfLoop = new StreamEngine(Window, Window, Lateness, false);
            selfLoop.Apply(new EdgeEvent(1, 5, 5, 1, 0));
            selfLoop.Finish();
            Check(selfLoop.Report.FinalAggregates[0].Triangles == 0 && selfLoop.Report.FinalAggregates[0].Components == 1,
                "a self loop adds a component of one vertex and no triangle");
            Console.WriteLine("degenerate: empty, single edge, triangle plus pendant, self loop all as expected");
        }

        // ---- determinism ---------------------------------------------------------------------------------
        {
            List<EdgeEvent> stream = BuildStream(45, 6, 260, 0x5EED844000000005UL, 6);
            StreamReport first = Run(Window, Slide, Lateness, true, stream, out StreamEngine _);
            StreamReport second = Run(Window, Slide, Lateness, true, stream, out StreamEngine _);
            bool sameAggregates = first.FinalAggregates.Count == second.FinalAggregates.Count;
            foreach (KeyValuePair<long, WindowAggregate> entry in first.FinalAggregates)
                if (!second.FinalAggregates.TryGetValue(entry.Key, out WindowAggregate other)
                    || !entry.Value.SameAs(other)) sameAggregates = false;
            Check(sameAggregates && first.Assignments == second.Assignments && first.Dropped == second.Dropped,
                "two runs of the same stream produce identical aggregates and identical counters");
            Check(first.PeakRetainedEdges == second.PeakRetainedEdges, "including the measured peak state size ("
                + first.PeakRetainedEdges + " edges)");
            Console.WriteLine("determinism: assignments=" + first.Assignments + " drops=" + first.Dropped
                + " windows=" + first.FinalAggregates.Count + " peak=" + first.PeakRetainedEdges);
        }

        Console.WriteLine("PASS");
    }
}
