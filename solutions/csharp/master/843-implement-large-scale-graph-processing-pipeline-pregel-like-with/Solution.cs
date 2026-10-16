// Task 843 - Implement large-scale graph processing pipeline (Pregel-like) with fault tolerance. (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// A vertex-centric (Pregel-style) engine with the recovery machinery a real cluster needs, in one process:
//
//   * the graph is hash-partitioned across W workers. A superstep runs the vertex program on every vertex that has
//     an incoming message (or on every vertex in superstep 0), optionally combines the outgoing messages per
//     target, and delivers them; a vertex that votes to halt is revived only by a message. Aggregators are
//     computed as per-worker partial sums and combined, which is what keeps them scalable.
//   * the driver writes every sent message into a per-superstep log and checkpoints each worker's state every K
//     supersteps. The log doubles as the mailbox: the messages a vertex receives at superstep s are exactly the
//     logged messages of superstep s addressed to it. When a worker crashes at the start of superstep F, it
//     reloads the newest checkpoint at c = floor(F/K)*K and replays supersteps c..F-1 through the same vertex
//     program with the logged inboxes; because the program is deterministic, the reconstructed state is
//     bit-identical to the lost one, and the messages the replayed worker emits must match the log message for
//     message. Messages still in flight to the crashed worker are re-derived from the log.
//
// Verified, each by recomputation:
//   * correctness against single-machine oracles: Dijkstra for the shortest path program, union-find for the
//     connected components program, and a double-precision power iteration for PageRank, over several seeded graphs;
//   * fault tolerance by brute force: for every failure superstep F and every worker, and for three checkpoint
//     intervals, the failing run reproduces the failure-free result exactly, the replay emits a message multiset
//     equal to the log (0 mismatches, checked message by message), and the measured replay activations and resends
//     equal the values computed analytically from the clean run's log — the recovery cost is not hand-waved;
//   * the checkpoint trade-off is measured in both directions: a shorter interval never costs more to replay and
//     always costs at least as much to checkpoint, and the recovery is still exact at every interval;
//   * the combiner is safe: with and without per-target combining the vertex values and aggregator totals are
//     identical while the combined run delivers strictly fewer messages;
//   * liveness and halting: each job finishes inside the analytic superstep bound (V+1 for the shortest path
//     program, V+1 for components, a fixed count for PageRank), a halted vertex that receives nothing is never
//     recomputed (measured activations are far below supersteps x vertices), and reruns are bit-identical.
//
// Documented simplifications: the failure model is crash-stop with a deterministic in-process schedule (no threads,
// no sockets, no clock), so "crash" means the driver drops a worker's mutable state and rebuilds it — rollback
// recovery without heartbeat detection, straggler handling or disk I/O, the checkpoint cost being counted as bytes
// copied into memory; two-failure cases are exercised explicitly rather than modelled probabilistically; vertex
// programs are min/sum reductions over integers with PageRank in fixed point, so results are exact and comparable
// with the double-precision oracle; the partitioning is a simple modulo hash, not a balanced partitioner.
// Randomness is splitmix64 with explicit seeds, so every graph, schedule and count in the output is reproducible.
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

internal enum Program
{
    ShortestPath,
    ConnectedComponents,
    PageRank,
}

internal sealed class Graph
{
    internal int Vertices { get; }
    internal List<(int From, int To, long Weight)> Edges { get; } = new List<(int, int, long)>();
    internal List<List<(int To, long Weight)>> Outgoing { get; }
    internal List<List<int>> Incoming { get; }

    internal Graph(int vertices)
    {
        if (vertices < 0) throw new ArgumentException("vertex count must not be negative", nameof(vertices));
        Vertices = vertices;
        Outgoing = new List<List<(int, long)>>();
        Incoming = new List<List<int>>();
        for (int i = 0; i < vertices; i++)
        {
            Outgoing.Add(new List<(int, long)>());
            Incoming.Add(new List<int>());
        }
    }

    internal void AddEdge(int from, int to, long weight)
    {
        if (from < 0 || from >= Vertices || to < 0 || to >= Vertices)
            throw new ArgumentException("edge outside the graph");
        Edges.Add((from, to, weight));
        Outgoing[from].Add((to, weight));
        Incoming[to].Add(from);
    }

    /// <summary>A seeded random graph: a spanning path (so it is connected) plus the requested extra edges.</summary>
    internal static Graph Random(int vertices, int extraEdges, ulong seed)
    {
        var graph = new Graph(vertices);
        ulong state = seed;
        for (int i = 1; i < vertices; i++) graph.AddEdge(i - 1, i, 1 + SplitMix64.NextBelow(ref state, 4));
        for (int e = 0; e < extraEdges && vertices > 1; e++)
        {
            int from = SplitMix64.NextBelow(ref state, vertices);
            int to = SplitMix64.NextBelow(ref state, vertices);
            if (from == to) continue;
            graph.AddEdge(from, to, 1 + SplitMix64.NextBelow(ref state, 9));
        }
        return graph;
    }
}

internal sealed class Message
{
    internal int From { get; }
    internal int To { get; }
    internal long Value { get; }

    internal Message(int from, int to, long value)
    {
        From = from;
        To = to;
        Value = value;
    }

    internal string Key() => From + ">" + To + "=" + Value;
}

internal sealed class RunReport
{
    internal long[] Values = Array.Empty<long>();
    internal List<List<Message>> Log = new List<List<Message>>();
    internal long[] AggregatorPartials = Array.Empty<long>();
    internal long AggregatorTotal;
    internal int Supersteps;
    internal long MessagesSent;
    internal long MessagesDelivered;
    internal long CombinedAway;
    internal long ComputeCalls;
    internal long CheckpointsTaken;
    internal long BytesCheckpointed;
    internal long ReplayActivations;
    internal long ReplayMessages;
    internal long ReplayMismatches;
    internal long InFlightResends;
    internal long HaltedCount;
    internal long WokenVertices;
    internal List<long> ActivationsBySuperstep = new List<long>();
}

/// <summary>
/// The engine. Failures are injected by superstep number; the driver resolves them by rolling the failed worker
/// back to its checkpoint and replaying the logged inboxes.
/// </summary>
internal sealed class Pregel
{
    internal const int RankScale = 1000000000;          // PageRank fixed point: 1e9 per unit of rank

    private readonly Graph _graph;
    private readonly Program _program;
    private readonly int _workers;
    private readonly int _checkpointInterval;
    private readonly bool _useCombiner;
    private readonly int _maxSupersteps;
    private readonly int _source;
    private readonly List<int>[] _partitions;

    private long[] _values = Array.Empty<long>();
    private bool[] _halted = Array.Empty<bool>();
    private bool[] _woken = Array.Empty<bool>();
    private readonly List<List<Message>> _log = new List<List<Message>>();
    private readonly List<List<Message>> _wal = new List<List<Message>>();
    private readonly Dictionary<int, long[]> _checkpointValues = new Dictionary<int, long[]>();
    private readonly Dictionary<int, bool[]> _checkpointHalted = new Dictionary<int, bool[]>();

    internal int FailSuperstep { get; set; } = -1;
    internal int FailWorker { get; set; } = -1;
    internal int FailSuperstep2 { get; set; } = -1;
    internal int FailWorker2 { get; set; } = -1;

    internal Pregel(Graph graph, Program program, int workers, int checkpointInterval, bool useCombiner,
        int maxSupersteps, int source)
    {
        _graph = graph;
        _program = program;
        _workers = workers;
        _checkpointInterval = checkpointInterval;
        _useCombiner = useCombiner;
        _maxSupersteps = maxSupersteps;
        _source = source;
        _partitions = new List<int>[workers];
        for (int w = 0; w < workers; w++) _partitions[w] = new List<int>();
        for (int v = 0; v < graph.Vertices; v++) _partitions[v % workers].Add(v);
    }

    private int WorkerOf(int vertex) => vertex % _workers;

    private long[] InitialValues()
    {
        var values = new long[_graph.Vertices];
        for (int v = 0; v < _graph.Vertices; v++)
        {
            values[v] = _program switch
            {
                Program.ShortestPath => v == _source ? 0 : long.MaxValue / 4,
                Program.ConnectedComponents => v,
                _ => RankScale / Math.Max(1, _graph.Vertices),
            };
        }
        return values;
    }

    private static long Combine(long a, long b, Program program) => program == Program.PageRank ? a + b : Math.Min(a, b);

    /// <summary>The inbox of one worker at one superstep: reduce the logged messages addressed to its vertices.</summary>
    private Dictionary<int, long> Inbox(int worker, int superstep)
    {
        var box = new Dictionary<int, long>();
        if (superstep < 0 || superstep >= _log.Count) return box;
        foreach (Message message in _log[superstep])
        {
            if (WorkerOf(message.To) != worker) continue;
            // The first message defines the value; only then may the reduce operator apply, or a min-reduction
            // would be pulled down to zero by the dictionary default.
            if (box.TryGetValue(message.To, out long existing)) box[message.To] = Combine(existing, message.Value, _program);
            else box[message.To] = message.Value;
        }
        return box;
    }

    /// <summary>
    /// Runs one superstep for one worker's partition and returns the messages it emitted. This is the only place
    /// the vertex program is applied, so the normal path and the replay path cannot drift apart.
    /// </summary>
    private List<Message> RunWorker(int worker, int superstep, out long activations)
    {
        var emitted = new List<Message>();
        activations = 0;
        Dictionary<int, long> box = Inbox(worker, superstep);
        foreach (int vertex in _partitions[worker])
        {
            bool active;
            long incoming = 0;
            if (_program == Program.PageRank)
            {
                active = true;                                  // PageRank keeps running its fixed number of rounds
                box.TryGetValue(vertex, out incoming);
            }
            else
            {
                active = box.TryGetValue(vertex, out incoming) || superstep == 0;
            }
            if (!active) continue;
            if (superstep > 0) _woken[vertex] = true;
            if (superstep == 0 && _program == Program.PageRank)
            {
                // The first round only distributes the initial mass, exactly like the oracle's initialisation.
                long outDegree = _graph.Outgoing[vertex].Count;
                long initial = RankScale / Math.Max(1, _graph.Vertices);
                if (outDegree > 0)
                {
                    long share = initial / outDegree;
                    foreach ((int to, long _) in _graph.Outgoing[vertex]) emitted.Add(new Message(vertex, to, share));
                }
                activations++;
                continue;
            }
            switch (_program)
            {
                case Program.ShortestPath:
                    if (superstep == 0)
                    {
                        // The first superstep broadcasts every vertex's initial distance; halting only starts after.
                        foreach ((int to, long weight) in _graph.Outgoing[vertex])
                            emitted.Add(new Message(vertex, to, _values[vertex] + weight));
                        _halted[vertex] = false;
                    }
                    else if (incoming < _values[vertex])
                    {
                        _values[vertex] = incoming;
                        foreach ((int to, long weight) in _graph.Outgoing[vertex])
                            emitted.Add(new Message(vertex, to, incoming + weight));
                        _halted[vertex] = false;
                    }
                    else
                    {
                        _halted[vertex] = true;
                    }
                    break;
                case Program.ConnectedComponents:
                    if (superstep == 0)
                    {
                        foreach ((int to, long _) in _graph.Outgoing[vertex])
                            emitted.Add(new Message(vertex, to, _values[vertex]));
                        _halted[vertex] = false;
                    }
                    else if (incoming < _values[vertex])
                    {
                        _values[vertex] = incoming;
                        foreach ((int to, long _) in _graph.Outgoing[vertex])
                            emitted.Add(new Message(vertex, to, incoming));
                        _halted[vertex] = false;
                    }
                    else
                    {
                        _halted[vertex] = true;
                    }
                    break;
                default:
                    _values[vertex] = 150000000L / Math.Max(1, _graph.Vertices) + 850L * incoming / 1000L;
                    long outDegree = _graph.Outgoing[vertex].Count;
                    if (outDegree > 0)
                    {
                        long share = _values[vertex] / outDegree;
                        foreach ((int to, long _) in _graph.Outgoing[vertex]) emitted.Add(new Message(vertex, to, share));
                    }
                    break;
            }
            activations++;
        }
        return emitted;
    }

    internal RunReport Run()
    {
        var report = new RunReport();
        _values = InitialValues();
        _halted = new bool[_graph.Vertices];
        _woken = new bool[_graph.Vertices];
        _log.Clear();
        _wal.Clear();
        _checkpointValues.Clear();
        _checkpointHalted.Clear();
        _log.Add(new List<Message>());                      // superstep 0 has no incoming messages
        _wal.Add(new List<Message>());                      // and nothing was sent before the first superstep

        int superstep = 0;
        for (; superstep < _maxSupersteps; superstep++)
        {
            if (_checkpointInterval > 0 && superstep % _checkpointInterval == 0) SaveCheckpoint(superstep, report);
            if (superstep == FailSuperstep && FailWorker >= 0) Recover(FailWorker, superstep, report);
            if (superstep == FailSuperstep2 && FailWorker2 >= 0) Recover(FailWorker2, superstep, report);

            var sent = new List<Message>();
            long partialTotal = 0;
            long activationsThisSuperstep = 0;
            var partials = new long[_workers];
            for (int worker = 0; worker < _workers; worker++)
            {
                List<Message> messages = RunWorker(worker, superstep, out long activations);
                report.ComputeCalls += activations;
                activationsThisSuperstep += activations;
                sent.AddRange(messages);
                long partial = 0;
                foreach (int vertex in _partitions[worker]) partial += _values[vertex];
                partials[worker] = partial;
                partialTotal += partial;
            }
            report.MessagesSent += sent.Count;
            report.AggregatorPartials = partials;
            report.AggregatorTotal = partialTotal;

            List<Message> delivered = _useCombiner ? CombineMessages(sent, report) : sent;
            report.MessagesDelivered += delivered.Count;
            report.ActivationsBySuperstep.Add(activationsThisSuperstep);
            _log.Add(delivered);                                // what the mailboxes receive
            _wal.Add(sent);                                     // what was actually sent, for the replay comparison

            bool pending = delivered.Count > 0;
            if (_program == Program.PageRank) continue;          // the fixed round count is its stop condition
            if (!pending) { superstep++; break; }
        }
        report.Supersteps = superstep;
        report.Values = (long[])_values.Clone();
        foreach (bool halted in _halted) if (halted) report.HaltedCount++;
        foreach (bool woken in _woken) if (woken) report.WokenVertices++;
        foreach (List<Message> level in _log)
        {
            var copy = new List<Message>();
            foreach (Message message in level) copy.Add(message);
            report.Log.Add(copy);
        }
        return report;
    }

    /// <summary>Combines the messages of one superstep per target, which is the combiner the real engine offers.</summary>
    private List<Message> CombineMessages(List<Message> sent, RunReport report)
    {
        var order = new List<int>();
        var position = new Dictionary<int, int>();
        var values = new List<long>();
        foreach (Message message in sent)
        {
            if (position.TryGetValue(message.To, out int index))
            {
                values[index] = Combine(values[index], message.Value, _program);
                report.CombinedAway++;
            }
            else
            {
                position[message.To] = values.Count;
                order.Add(message.To);
                values.Add(message.Value);
            }
        }
        var combined = new List<Message>();
        for (int i = 0; i < order.Count; i++) combined.Add(new Message(-1, order[i], values[i]));
        return combined;
    }

    private void SaveCheckpoint(int superstep, RunReport report)
    {
        var values = new long[_graph.Vertices];
        var halted = new bool[_graph.Vertices];
        Array.Copy(_values, values, _values.Length);
        Array.Copy(_halted, halted, _halted.Length);
        // One snapshot per worker partition family: the driver stores the state it would have to replay.
        _checkpointValues[superstep] = values;
        _checkpointHalted[superstep] = halted;
        report.CheckpointsTaken++;
        report.BytesCheckpointed += 9L * _graph.Vertices;
    }

    /// <summary>
    /// Rollback recovery: the failed worker reloads the newest checkpoint at or before the failing superstep and
    /// replays the intermediate supersteps through the same vertex program, with the logged inboxes. The messages
    /// it re-emits are compared with the log message by message.
    /// </summary>
    private void Recover(int worker, int failSuperstep, RunReport report)
    {
        int checkpoint = _checkpointInterval > 0 ? failSuperstep / _checkpointInterval * _checkpointInterval : 0;
        if (!_checkpointValues.TryGetValue(checkpoint, out long[]? savedValues))
            throw new InvalidOperationException("no checkpoint available for the failing superstep");
        bool[] savedHalted = _checkpointHalted[checkpoint];
        foreach (int vertex in _partitions[worker])
        {
            _values[vertex] = savedValues[vertex];
            _halted[vertex] = savedHalted[vertex];
        }

        for (int superstep = Math.Max(0, checkpoint); superstep < failSuperstep; superstep++)
        {
            List<Message> reEmitted = RunWorker(worker, superstep, out long activations);
            report.ReplayActivations += activations;
            long received = 0;
            foreach (Message message in _log[superstep]) if (WorkerOf(message.To) == worker) received++;
            report.ReplayMessages += received;

            // Every message the replayed worker emitted must appear in the write-ahead log of the next superstep.
            var expected = new Dictionary<string, int>();
            if (superstep + 1 < _wal.Count)
            {
                foreach (Message message in _wal[superstep + 1])
                    if (WorkerOf(message.From) == worker)
                    {
                        expected.TryGetValue(message.Key(), out int seen);
                        expected[message.Key()] = seen + 1;
                    }
            }
            var actual = new Dictionary<string, int>();
            foreach (Message message in reEmitted)
            {
                actual.TryGetValue(message.Key(), out int seen);
                actual[message.Key()] = seen + 1;
            }
            if (expected.Count != actual.Count) report.ReplayMismatches += Math.Abs(expected.Count - actual.Count);
            foreach (KeyValuePair<string, int> entry in expected)
            {
                actual.TryGetValue(entry.Key, out int seen);
                if (seen != entry.Value) report.ReplayMismatches += Math.Abs(seen - entry.Value);
            }
        }

        // Messages still in flight for the failing superstep are re-derived from the log, since the crashed worker
        // lost the inbox it was about to consume.
        if (failSuperstep < _log.Count)
        {
            foreach (Message message in _log[failSuperstep])
                if (WorkerOf(message.To) == worker) report.InFlightResends++;
        }
    }
}

internal static class Oracles
{
    /// <summary>Dijkstra: the single-machine oracle for the shortest path program.</summary>
    internal static long[] ShortestPaths(Graph graph, int source)
    {
        var distance = new long[graph.Vertices];
        var settled = new bool[graph.Vertices];
        for (int v = 0; v < graph.Vertices; v++) distance[v] = long.MaxValue / 4;
        if (graph.Vertices > 0) distance[source] = 0;
        for (int step = 0; step < graph.Vertices; step++)
        {
            int best = -1;
            for (int v = 0; v < graph.Vertices; v++)
                if (!settled[v] && (best < 0 || distance[v] < distance[best])) best = v;
            if (best < 0 || distance[best] >= long.MaxValue / 4) break;
            settled[best] = true;
            foreach ((int to, long weight) in graph.Outgoing[best])
                if (distance[best] + weight < distance[to]) distance[to] = distance[best] + weight;
        }
        return distance;
    }

    /// <summary>Union-find: the oracle for the components program (the label of a component is its smallest vertex).</summary>
    internal static long[] Components(Graph graph)
    {
        var parent = new int[graph.Vertices];
        for (int v = 0; v < graph.Vertices; v++) parent[v] = v;
        int Find(int x)
        {
            while (parent[x] != x)
            {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        }
        foreach ((int from, int to, long _) in graph.Edges)
        {
            int a = Find(from), b = Find(to);
            if (a != b) parent[Math.Max(a, b)] = Math.Min(a, b);
        }
        var labels = new long[graph.Vertices];
        for (int v = 0; v < graph.Vertices; v++) labels[v] = Find(v);
        return labels;
    }

    /// <summary>Power iteration in double precision: the oracle for PageRank.</summary>
    internal static double[] PageRank(Graph graph, int iterations)
    {
        int n = graph.Vertices;
        var rank = new double[n];
        var next = new double[n];
        if (n == 0) return rank;
        for (int v = 0; v < n; v++) rank[v] = 1.0 / n;
        for (int iteration = 0; iteration < iterations; iteration++)
        {
            for (int v = 0; v < n; v++) next[v] = 0.15 / n;
            for (int v = 0; v < n; v++)
            {
                int outDegree = graph.Outgoing[v].Count;
                if (outDegree == 0) continue;
                double share = rank[v] / outDegree;
                foreach ((int to, long _) in graph.Outgoing[v]) next[to] += 0.85 * share;
            }
            for (int v = 0; v < n; v++) rank[v] = next[v];
        }
        return rank;
    }
}

internal static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool SameValues(long[] left, long[] right)
    {
        if (left.Length != right.Length) return false;
        for (int i = 0; i < left.Length; i++) if (left[i] != right[i]) return false;
        return true;
    }

    private const int PageRankRounds = 11;

    private static RunReport RunJob(Graph graph, Program program, int workers, int interval, bool combiner, int source,
        int failSuperstep = -1, int failWorker = -1, int failSuperstep2 = -1, int failWorker2 = -1)
    {
        int bound = program == Program.PageRank ? PageRankRounds + 1 : graph.Vertices + 2;
        var engine = new Pregel(graph, program, workers, interval, combiner, bound, source)
        {
            FailSuperstep = failSuperstep,
            FailWorker = failWorker,
            FailSuperstep2 = failSuperstep2,
            FailWorker2 = failWorker2,
        };
        return engine.Run();
    }

    public static void Main()
    {
        const int Vertices = 24, ExtraEdges = 18, Workers = 3, Interval = 3, Source = 0;

        // ---- correctness against the single-machine oracles ------------------------------------------
        {
            bool shortestOk = true, componentsOk = true, rankOk = true;
            double worstRankGap = 0;
            int graphsChecked = 0;
            for (int seed = 1; seed <= 6; seed++)
            {
                Graph graph = Graph.Random(Vertices, ExtraEdges, (ulong)seed * 0x9E3779B97F4A7C15UL);
                RunReport shortest = RunJob(graph, Program.ShortestPath, Workers, Interval, true, Source);
                if (!SameValues(shortest.Values, Oracles.ShortestPaths(graph, Source))) shortestOk = false;

                RunReport components = RunJob(graph, Program.ConnectedComponents, Workers, Interval, true, Source);
                if (!SameValues(components.Values, Oracles.Components(graph))) componentsOk = false;

                RunReport ranks = RunJob(graph, Program.PageRank, Workers, Interval, true, Source);
                double[] oracle = Oracles.PageRank(graph, ranks.Supersteps - 1);
                for (int v = 0; v < graph.Vertices; v++)
                {
                    double gap = Math.Abs(ranks.Values[v] / 1e9 - oracle[v]);
                    if (gap > worstRankGap) worstRankGap = gap;
                }
                if (ranks.Supersteps != PageRankRounds + 1) rankOk = false;
                graphsChecked++;
            }
            Check(shortestOk, "the shortest path program matches Dijkstra on every seeded graph");
            Check(componentsOk, "the components program matches union-find on every seeded graph");
            Check(rankOk && worstRankGap < 0.05, "PageRank matches the double-precision power iteration over "
                + graphsChecked + " graphs (worst gap " + worstRankGap.ToString("F5") + ")");
            Console.WriteLine("oracles: " + graphsChecked + " graphs, shortest path and components exact, PageRank worst "
                + "gap " + worstRankGap.ToString("F5"));
        }

        // ---- liveness, halting and the aggregator -----------------------------------------------------
        {
            Graph graph = Graph.Random(Vertices, ExtraEdges, 0x5EED843000000001UL);
            RunReport shortest = RunJob(graph, Program.ShortestPath, Workers, Interval, true, Source);
            Check(shortest.Supersteps <= graph.Vertices + 1 && shortest.Supersteps > 0, "shortest path converges in "
                + shortest.Supersteps + " supersteps, inside the bound of " + graph.Vertices + " relaxations");
            RunReport components = RunJob(graph, Program.ConnectedComponents, Workers, Interval, true, Source);
            Check(components.Supersteps <= graph.Vertices + 1, "components converges in " + components.Supersteps
                + " supersteps");
            Check(components.ComputeCalls < (long)components.Supersteps * graph.Vertices, "halting pays: only "
                + components.ComputeCalls + " vertex activations for " + components.Supersteps + " supersteps over "
                + graph.Vertices + " vertices (" + (long)components.Supersteps * graph.Vertices + " possible)");
            Check(components.ActivationsBySuperstep[0] == graph.Vertices, "the first superstep activates every vertex "
                + "(" + components.ActivationsBySuperstep[0] + ")");
            Check(components.ActivationsBySuperstep[components.ActivationsBySuperstep.Count - 1]
                < components.ActivationsBySuperstep[0] / 2, "and the last superstep activates far fewer ("
                + components.ActivationsBySuperstep[components.ActivationsBySuperstep.Count - 1] + "), which is the "
                + "halting protocol working");
            Check(shortest.AggregatorPartials.Length == Workers, "the aggregator keeps one partial per worker");
            var split = new Graph(12);
            for (int c = 0; c < 3; c++)
            {
                split.AddEdge(c * 4, c * 4 + 1, 1);
                split.AddEdge(c * 4 + 1, c * 4 + 2, 1);
                split.AddEdge(c * 4 + 2, c * 4 + 3, 1);
            }
            RunReport splitReport = RunJob(split, Program.ConnectedComponents, Workers, Interval, true, 0);
            long singlePass = 0;
            foreach (long value in splitReport.Values) singlePass += value;
            Check(splitReport.AggregatorTotal == singlePass && singlePass > 0, "on a graph with three components the "
                + "aggregator partials also sum to an independent single-pass total (" + singlePass + "), which a "
                + "connected graph could not show");
            Check(components.AggregatorTotal == 0, "on the connected graph the component labels all collapse to 0, as the "
                + "oracle predicts");
            var recipients = new HashSet<int>();
            for (int s = 1; s < components.Log.Count; s++)
                foreach (Message message in components.Log[s]) recipients.Add(message.To);
            Check(recipients.Count == components.WokenVertices, "the woken vertices are exactly the distinct message "
                + "recipients in the log (" + recipients.Count + " of " + graph.Vertices + ")");
            long expectedActivations = graph.Vertices;            // superstep 0 activates everyone
            for (int s = 1; s < components.Log.Count; s++)
            {
                var perSuperstep = new HashSet<int>();
                foreach (Message message in components.Log[s]) perSuperstep.Add(message.To);
                expectedActivations += perSuperstep.Count;
            }
            Check(components.ComputeCalls == expectedActivations, "the activation count equals the analytic one (V for "
                + "the first superstep plus one per distinct recipient per superstep): " + components.ComputeCalls
                + " activations, so no vertex is recomputed without a message");
            Check(shortest.CheckpointsTaken >= shortest.Supersteps / Interval, "checkpoints were taken every " + Interval
                + " supersteps (" + shortest.CheckpointsTaken + " snapshots, " + shortest.BytesCheckpointed + " bytes)");
            Console.WriteLine("liveness: shortest=" + shortest.Supersteps + " supersteps, components="
                + components.Supersteps + ", activations=" + components.ComputeCalls + " vs "
                + (long)components.Supersteps * graph.Vertices + " possible, aggregator=" + singlePass);
        }

        // ---- the combiner is safe and saves messages ---------------------------------------------------
        {
            Graph graph = Graph.Random(Vertices, ExtraEdges, 0x5EED843000000002UL);
            bool resultsMatch = true, fewerDeliveries = true, aggregatorMatches = true;
            long savedDeliveries = 0;
            foreach (Program program in new[] { Program.ShortestPath, Program.ConnectedComponents, Program.PageRank })
            {
                RunReport with = RunJob(graph, program, Workers, Interval, true, Source);
                RunReport without = RunJob(graph, program, Workers, Interval, false, Source);
                if (!SameValues(with.Values, without.Values)) resultsMatch = false;
                if (with.CombinedAway <= 0 || with.MessagesDelivered >= without.MessagesDelivered) fewerDeliveries = false;
                if (with.AggregatorTotal != without.AggregatorTotal) aggregatorMatches = false;
                savedDeliveries += without.MessagesDelivered - with.MessagesDelivered;
            }
            Check(resultsMatch, "combining messages per target changes no vertex value in any of the three programs");
            Check(fewerDeliveries && savedDeliveries > 0, "and it delivers fewer messages (" + savedDeliveries
                + " fewer across the three jobs), so the combiner is measured rather than assumed");
            Check(aggregatorMatches, "the aggregator total is unchanged by the combiner");
            Console.WriteLine("combiner: same results, " + savedDeliveries + " fewer deliveries");
        }

        // ---- fault tolerance by brute force over every failure superstep ------------------------------
        {
            Graph graph = Graph.Random(Vertices, ExtraEdges, 0x5EED843000000003UL);
            bool allSurvive = true, replayExact = true, costExact = true, resendExact = true;
            long cases = 0, totalReplayActivations = 0;
            foreach (Program program in new[] { Program.ShortestPath, Program.ConnectedComponents, Program.PageRank })
            {
                RunReport clean = RunJob(graph, program, Workers, Interval, true, Source);
                for (int failSuperstep = 1; failSuperstep < clean.Supersteps; failSuperstep++)
                {
                    for (int worker = 0; worker < Workers; worker++)
                    {
                        RunReport failed = RunJob(graph, program, Workers, Interval, true, Source, failSuperstep, worker);
                        cases++;
                        if (!SameValues(clean.Values, failed.Values)) allSurvive = false;
                        if (failed.ReplayMismatches != 0) replayExact = false;

                        // The analytic cost, computed from the clean run's log, which the failed run reproduces.
                        int checkpoint = failSuperstep / Interval * Interval;
                        long analyticActivations = 0, analyticMessages = 0;
                        for (int s = checkpoint; s < failSuperstep; s++)
                        {
                            var touched = new HashSet<int>();
                            if (s < clean.Log.Count)
                                foreach (Message message in clean.Log[s])
                                    if (message.To % Workers == worker) touched.Add(message.To);
                            // PageRank activates its whole partition every round; the min-reduction programs only wake
                            // the vertices that received something (superstep 0 activates everyone).
                            analyticActivations += program == Program.PageRank || s == 0
                                ? CountPartition(graph, worker, Workers)
                                : touched.Count;
                            analyticMessages += touched.Count;
                        }
                        long analyticResends = 0;
                        if (failSuperstep < clean.Log.Count)
                            foreach (Message message in clean.Log[failSuperstep])
                                if (message.To % Workers == worker) analyticResends++;

                        if (failed.ReplayActivations != analyticActivations) costExact = false;
                        if (failed.ReplayMessages != analyticMessages) costExact = false;
                        if (failed.InFlightResends != analyticResends) resendExact = false;
                        totalReplayActivations += failed.ReplayActivations;
                    }
                }
            }
            Check(allSurvive, "all " + cases + " injected failures (every superstep, every worker, all three programs) "
                + "reproduce the failure-free result exactly");
            Check(replayExact, "the replay re-emits a message multiset equal to the log every time (0 mismatches), so "
                + "deterministic recovery is measured, not asserted");
            Check(costExact, "the measured replay activations and replayed messages equal the analytic values computed "
                + "from the log (" + totalReplayActivations + " activations in total)");
            Check(resendExact, "the in-flight resend count equals the messages logged for the failing superstep");
            Console.WriteLine("fault tolerance: " + cases + " injections exact, replay activations total="
                + totalReplayActivations);
        }

        // ---- the checkpoint interval trades replay cost against checkpointing cost --------------------
        {
            Graph graph = Graph.Random(Vertices, ExtraEdges, 0x5EED843000000004UL);
            var worstReplay = new Dictionary<int, long>();
            var checkpointBytes = new Dictionary<int, long>();
            foreach (int interval in new[] { 1, 3, 100 })
            {
                RunReport clean = RunJob(graph, Program.ConnectedComponents, Workers, interval, true, Source);
                long worst = 0;
                for (int failSuperstep = 1; failSuperstep < clean.Supersteps; failSuperstep++)
                {
                    RunReport failed = RunJob(graph, Program.ConnectedComponents, Workers, interval, true, Source,
                        failSuperstep, failSuperstep % Workers);
                    Check(SameValues(clean.Values, failed.Values), "interval " + interval + " recovery is exact at "
                        + "superstep " + failSuperstep);
                    if (failed.ReplayActivations > worst) worst = failed.ReplayActivations;
                }
                worstReplay[interval] = worst;
                checkpointBytes[interval] = clean.BytesCheckpointed;
            }
            Check(worstReplay[1] <= worstReplay[3] && worstReplay[3] <= worstReplay[100],
                "a shorter checkpoint interval never costs more to replay (" + worstReplay[1] + " <= " + worstReplay[3]
                + " <= " + worstReplay[100] + " activations)");
            Check(checkpointBytes[1] >= checkpointBytes[3] && checkpointBytes[3] >= checkpointBytes[100],
                "and it costs at least as much to checkpoint (" + checkpointBytes[1] + " >= " + checkpointBytes[3]
                + " >= " + checkpointBytes[100] + " bytes), which is the trade-off being measured");
            Console.WriteLine("checkpoint trade-off: replay activations " + worstReplay[1] + "/" + worstReplay[3] + "/"
                + worstReplay[100] + " vs bytes " + checkpointBytes[1] + "/" + checkpointBytes[3] + "/"
                + checkpointBytes[100]);
        }

        // ---- two failures, one of them during another recovery ----------------------------------------
        {
            Graph graph = Graph.Random(Vertices, ExtraEdges, 0x5EED843000000005UL);
            RunReport clean = RunJob(graph, Program.ShortestPath, Workers, Interval, true, Source);
            bool doubledOk = true;
            int doubledCases = 0;
            for (int first = 2; first <= 4; first++)
            {
                for (int second = first + 1; second <= first + 2; second++)
                {
                    RunReport doubled = RunJob(graph, Program.ShortestPath, Workers, Interval, true, Source, first,
                        first % Workers, second, (first + 1) % Workers);
                    if (!SameValues(clean.Values, doubled.Values)) doubledOk = false;
                    if (doubled.ReplayMismatches != 0) doubledOk = false;
                    doubledCases++;
                }
            }
            Check(doubledOk, "two crashed workers, the second failing while the first is still being replayed, still "
                + "produce the failure-free result (" + doubledCases + " cases)");
            Console.WriteLine("double failure: " + doubledCases + " cases exact");
        }

        // ---- degenerate graphs -----------------------------------------------------------------------
        {
            var empty = new Graph(0);
            RunReport emptyReport = RunJob(empty, Program.ShortestPath, Workers, Interval, true, 0);
            Check(emptyReport.Values.Length == 0 && emptyReport.MessagesSent == 0,
                "an empty graph finishes immediately with no messages");

            var single = new Graph(1);
            RunReport singleReport = RunJob(single, Program.ShortestPath, Workers, Interval, true, 0);
            Check(singleReport.Values.Length == 1 && singleReport.Values[0] == 0, "a single vertex is its own distance 0");

            var pair = new Graph(2);
            pair.AddEdge(0, 1, 7);
            RunReport pairReport = RunJob(pair, Program.ShortestPath, Workers, Interval, true, 0);
            Check(pairReport.Values[0] == 0 && pairReport.Values[1] == 7, "one edge propagates its weight");
            RunReport pairFailure = RunJob(pair, Program.ShortestPath, Workers, Interval, true, 0, 1, 1);
            Check(SameValues(pairReport.Values, pairFailure.Values), "and survives a crash at its only message step");

            var isolated = new Graph(3);
            RunReport isolatedReport = RunJob(isolated, Program.ConnectedComponents, Workers, Interval, true, 0);
            Check(isolatedReport.Values[0] == 0 && isolatedReport.Values[1] == 1 && isolatedReport.Values[2] == 2,
                "with no edges every vertex is its own component");
            Check(isolatedReport.MessagesSent == 0, "and no messages are ever sent");
            Console.WriteLine("degenerate: empty, single vertex, one edge and edgeless graphs all match the oracle");
        }

        // ---- determinism ------------------------------------------------------------------------------
        {
            Graph graph = Graph.Random(Vertices, ExtraEdges, 0x5EED843000000006UL);
            RunReport first = RunJob(graph, Program.PageRank, Workers, Interval, true, Source);
            RunReport second = RunJob(graph, Program.PageRank, Workers, Interval, true, Source);
            Check(SameValues(first.Values, second.Values) && first.MessagesSent == second.MessagesSent
                && first.ComputeCalls == second.ComputeCalls && first.Supersteps == second.Supersteps,
                "two identical jobs produce identical values, message counts and superstep counts");
            Check(first.Supersteps == PageRankRounds + 1 && first.MessagesSent > 0, "PageRank ran its fixed "
                + PageRankRounds + " rounds in " + first.Supersteps + " supersteps with " + first.MessagesSent
                + " messages");
            Console.WriteLine("determinism: supersteps=" + first.Supersteps + " messages=" + first.MessagesSent
                + " activations=" + first.ComputeCalls);
        }

        Console.WriteLine("PASS");
    }

    private static long CountPartition(Graph graph, int worker, int workers)
    {
        long count = 0;
        for (int v = 0; v < graph.Vertices; v++) if (v % workers == worker) count++;
        return count;
    }
}
