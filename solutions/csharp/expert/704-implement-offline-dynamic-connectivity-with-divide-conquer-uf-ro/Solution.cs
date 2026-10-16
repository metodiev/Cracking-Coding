// Task 704 - Implement offline dynamic connectivity with divide & conquer + UF rollback.
// (Tier: Expert, CTCI Ch 4 — Trees and Graphs)
//
// The offline dynamic connectivity problem asks, for a sequence of time steps, which vertex pairs
// are connected in the graph that is active at that step. Instead of replaying the graph once per
// step, every edge becomes a *time interval* and the intervals are distributed over a segment tree
// of the timeline: an interval is stored in the O(log T) nodes that exactly cover it, so an edge is
// in force along every root-to-leaf path that passes through one of those nodes. A DFS of that
// segment tree then walks time: on the way down it unions the interval edges of the node, at a leaf
// it answers the queries of that step, and on the way up it *rolls the unions back*.
//
// Rollback is what makes this work. The union-find structure uses union by size and deliberately no
// path compression, so a union rewrites a constant number of cells and can be undone; a mark/roll
// pair therefore restores the structure exactly. The invariant checked after every single node is
// "component count and history length are both back to their values on entry", and the whole DFS
// must end with an empty history and n singleton components.
//
// Verified against an independent oracle that replays the active edge set of each time step into a
// fresh union-find: a hand-checked 5-step fixture, 60 random steps in which edges appear, vanish and
// appear again, intervals of length one, empty intervals that must be ignored, self queries and
// rejected malformed input. The number of unions the offline structure performs is also compared
// with the naive replay cost.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

/// <summary>Union by size only: no path compression, so every union can be undone in O(1).</summary>
internal sealed class RollbackDsu
{
    private readonly int[] _parent;
    private readonly int[] _size;
    private readonly List<int[]> _history = new List<int[]>();

    internal RollbackDsu(int count)
    {
        _parent = new int[count];
        _size = new int[count];
        for (int i = 0; i < count; i++) { _parent[i] = i; _size[i] = 1; }
        Components = count;
    }

    internal int Components { get; private set; }
    internal int Unions { get; private set; }
    internal long HistoryLength => _history.Count;

    internal int Find(int x)
    {
        while (_parent[x] != x) x = _parent[x];
        return x;
    }

    internal bool Union(int a, int b)
    {
        int ra = Find(a), rb = Find(b);
        if (ra == rb) return false;
        if (_size[ra] < _size[rb]) { int swap = ra; ra = rb; rb = swap; }
        _history.Add(new[] { rb, ra, _size[ra] });       // (child, parent, parent size before)
        _parent[rb] = ra;
        _size[ra] += _size[rb];
        Components--;
        Unions++;
        return true;
    }

    internal int Mark() => _history.Count;

    internal void Rollback(int mark)
    {
        while (_history.Count > mark)
        {
            int[] entry = _history[_history.Count - 1];
            _history.RemoveAt(_history.Count - 1);
            _parent[entry[0]] = entry[0];
            _size[entry[1]] = entry[2];
            Components++;
        }
    }

    internal bool IsSingleton(int x) => _parent[x] == x && _size[x] == 1;
}

internal sealed class OfflineConnectivity
{
    private readonly int _steps, _leafCount;
    private readonly List<int[]>[] _bucket;
    private readonly List<int[]>[] _queries;
    private readonly RollbackDsu _dsu;
    private bool[] _answers = Array.Empty<bool>();

    internal OfflineConnectivity(int vertices, int steps)
    {
        if (vertices <= 0 || steps <= 0) throw new ArgumentOutOfRangeException(nameof(steps));
        _steps = steps;
        VertexCount = vertices;
        _leafCount = 1;
        while (_leafCount < steps) _leafCount <<= 1;
        _bucket = new List<int[]>[2 * _leafCount];
        _queries = new List<int[]>[steps];
        for (int i = 0; i < _bucket.Length; i++) _bucket[i] = new List<int[]>();
        for (int i = 0; i < steps; i++) _queries[i] = new List<int[]>();
        _dsu = new RollbackDsu(vertices);
    }

    internal int UnionCalls { get; private set; }
    internal int StoredIntervals { get; private set; }
    internal int QueryCount { get; private set; }
    internal long LeafVisits { get; private set; }
    internal bool RollbackBroken { get; private set; }
    internal bool[] ConnectedFlags => _answers;

    /// <summary>Registers an edge active on the closed time interval [from, to].</summary>
    internal void AddEdgeInterval(int u, int v, int from, int to)
    {
        if (u == v) throw new ArgumentOutOfRangeException(nameof(v), "a self loop is not an edge of a graph query");
        if (from < 0 || to >= _steps) throw new ArgumentOutOfRangeException(nameof(from), "interval outside the timeline");
        if (from > to) return;                      // empty interval: the edge never exists
        StoredIntervals++;
        int lo = from + _leafCount, hi = to + _leafCount + 1;
        while (lo < hi)
        {
            if ((lo & 1) == 1) _bucket[lo++].Add(new[] { u, v });
            if ((hi & 1) == 1) _bucket[--hi].Add(new[] { u, v });
            lo >>= 1;
            hi >>= 1;
        }
    }

    /// <summary>Registers a connectivity query and returns the slot the answer is written to.</summary>
    internal int AddQuery(int time, int u, int v)
    {
        if (time < 0 || time >= _steps) throw new ArgumentOutOfRangeException(nameof(time));
        int slot = QueryCount++;
        if (_answers.Length < QueryCount) Array.Resize(ref _answers, Math.Max(8, QueryCount * 2));
        _queries[time].Add(new[] { u, v, slot });
        return slot;
    }

    internal void Solve() => Run(1, 0, _leafCount - 1);

    /// <summary>True when the union-find is back to n singletons with an empty history.</summary>
    internal bool IsClean()
    {
        if (_dsu.HistoryLength != 0 || _dsu.Components != VertexCount) return false;
        for (int i = 0; i < VertexCount; i++) if (!_dsu.IsSingleton(i)) return false;
        return true;
    }

    internal int VertexCount { get; }

    private void Run(int node, int lo, int hi)
    {
        int mark = _dsu.Mark();
        int componentsBefore = _dsu.Components;
        foreach (int[] edge in _bucket[node])
        {
            UnionCalls++;
            _dsu.Union(edge[0], edge[1]);
        }
        if (lo == hi)
        {
            LeafVisits++;
            if (lo < _steps)
            {
                foreach (int[] q in _queries[lo]) _answers[q[2]] = _dsu.Find(q[0]) == _dsu.Find(q[1]);
            }
        }
        else
        {
            int mid = (lo + hi) / 2;
            Run(2 * node, lo, mid);
            Run(2 * node + 1, mid + 1, hi);
        }
        _dsu.Rollback(mark);
        // The rollback invariant: the node's unions left no trace at all.
        if (_dsu.Components != componentsBefore || _dsu.HistoryLength != mark) RollbackBroken = true;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /// <summary>Oracle: is u connected to v in the graph active at this step? (fresh DSU, no rollback)</summary>
    private static bool ConnectedAt(int vertices, List<int[]>[] activeByTime, int time, int u, int v)
    {
        var dsu = new RollbackDsu(vertices);
        foreach (int[] e in activeByTime[time]) dsu.Union(e[0], e[1]);
        return dsu.Find(u) == dsu.Find(v);
    }

    public static void Main()
    {
        // ---- a small hand-checkable timeline --------------------------------------------------
        {
            // vertices 0..3; edge 0-1 alive on [0,2], edge 2-3 alive on [1,3], edge 1-2 only at t=2.
            const int Steps = 5, Vertices = 4;
            var solver = new OfflineConnectivity(Vertices, Steps);
            var active = new List<int[]>[Steps];
            for (int t = 0; t < Steps; t++) active[t] = new List<int[]>();
            var schedule = new[]
            {
                (u: 0, v: 1, from: 0, to: 2),
                (u: 2, v: 3, from: 1, to: 3),
                (u: 1, v: 2, from: 2, to: 2),
            };
            foreach (var edge in schedule)
            {
                solver.AddEdgeInterval(edge.u, edge.v, edge.from, edge.to);
                for (int t = edge.from; t <= edge.to; t++) active[t].Add(new[] { edge.u, edge.v });
            }
            var pairs = new[] { (u: 0, v: 1), (u: 0, v: 3), (u: 1, v: 3) };
            var expected = new bool[Steps][];
            var slots = new int[Steps][];
            for (int t = 0; t < Steps; t++)
            {
                expected[t] = new bool[pairs.Length];
                slots[t] = new int[pairs.Length];
                for (int p = 0; p < pairs.Length; p++)
                {
                    slots[t][p] = solver.AddQuery(t, pairs[p].u, pairs[p].v);
                    expected[t][p] = ConnectedAt(Vertices, active, t, pairs[p].u, pairs[p].v);
                }
            }
            solver.Solve();
            Check(!solver.RollbackBroken, "the rollback invariant holds on every node of the tree");
            Check(solver.LeafVisits == 8, "a 5-step timeline gets 8 leaves (the tree is padded to a power of two)");

            // independently hand-checked table: t=0 only 0-1, t=1 adds 2-3, t=2 joins everything,
            // t=3 loses 0-1 (leaving only 2-3), t=4 loses 2-3 as well.
            var handChecked = new[]
            {
                new[] { true, false, false },
                new[] { true, false, false },
                new[] { true, true, true },
                new[] { false, false, false },
                new[] { false, false, false },
            };
            bool handOk = true, oracleOk = true;
            for (int t = 0; t < Steps; t++)
            {
                for (int p = 0; p < pairs.Length; p++)
                {
                    if (solver.ConnectedFlags[slots[t][p]] != handChecked[t][p]) handOk = false;
                    if (solver.ConnectedFlags[slots[t][p]] != expected[t][p]) oracleOk = false;
                }
            }
            Check(handOk, "the hand-checked timeline answers match");
            Check(oracleOk, "the fixture also agrees with the replay oracle");
            Console.WriteLine("fixtureSteps=" + Steps + " fixtureQueries=" + solver.QueryCount
                + " storedIntervals=" + solver.StoredIntervals);
        }

        // ---- boundary cases --------------------------------------------------------------------
        {
            var empty = new OfflineConnectivity(3, 4);
            int[] slots = new int[4];
            for (int t = 0; t < 4; t++) slots[t] = empty.AddQuery(t, 0, 1);
            empty.Solve();
            bool allDisconnected = true;
            for (int i = 0; i < 4; i++) if (empty.ConnectedFlags[slots[i]]) allDisconnected = false;
            Check(allDisconnected, "with no edges nothing is connected");
            Check(!empty.RollbackBroken, "an edge-free tree still restores perfectly");
            Check(empty.UnionCalls == 0, "no unions are performed without edges");

            var instant = new OfflineConnectivity(2, 1);
            instant.AddEdgeInterval(0, 1, 0, 0);
            int only = instant.AddQuery(0, 0, 1);
            instant.Solve();
            Check(instant.ConnectedFlags[only], "an edge alive for a single instant is seen at that instant");

            var ignored = new OfflineConnectivity(2, 3);
            ignored.AddEdgeInterval(0, 1, 2, 1);          // empty interval: dropped, not an error
            int none = ignored.AddQuery(0, 0, 1);
            ignored.Solve();
            Check(ignored.StoredIntervals == 0 && !ignored.ConnectedFlags[none],
                "an empty interval is ignored instead of creating an edge");

            bool rejected = false;
            try { ignored.AddEdgeInterval(0, 1, -1, 2); } catch (ArgumentOutOfRangeException) { rejected = true; }
            Check(rejected, "an interval outside the timeline is rejected");
            rejected = false;
            try { ignored.AddEdgeInterval(0, 1, 0, 3); } catch (ArgumentOutOfRangeException) { rejected = true; }
            Check(rejected, "an interval running past the last step is rejected");
            rejected = false;
            try { ignored.AddQuery(9, 0, 1); } catch (ArgumentOutOfRangeException) { rejected = true; }
            Check(rejected, "a query outside the timeline is rejected");
            rejected = false;
            try { ignored.AddEdgeInterval(2, 2, 0, 1); } catch (ArgumentOutOfRangeException) { rejected = true; }
            Check(rejected, "a self loop is rejected as a malformed edge");
        }

        // ---- randomized timeline against the replay oracle ------------------------------------
        {
            const int Vertices = 14, Steps = 60, EdgeCount = 28;
            var solver = new OfflineConnectivity(Vertices, Steps);
            var active = new List<int[]>[Steps];
            for (int t = 0; t < Steps; t++) active[t] = new List<int[]>();
            ulong state = 0x5EED_F00DUL;
            int intervals = 0;

            for (int e = 0; e < EdgeCount; e++)
            {
                int u = Rng.NextInt(ref state, Vertices);
                int v = Rng.NextInt(ref state, Vertices);
                if (u == v) v = (v + 1) % Vertices;
                int from = Rng.NextInt(ref state, Steps);
                int to = Math.Min(Steps - 1, from + Rng.NextInt(ref state, 20));
                solver.AddEdgeInterval(u, v, from, to);
                intervals++;
                for (int t = from; t <= to; t++) active[t].Add(new[] { u, v });
                if (Rng.NextInt(ref state, 3) == 0)     // a second, disjoint lifetime for the same edge
                {
                    int from2 = Math.Min(Steps - 1, to + 1 + Rng.NextInt(ref state, 5));
                    int to2 = Math.Min(Steps - 1, from2 + Rng.NextInt(ref state, 8));
                    if (from2 <= to2)
                    {
                        solver.AddEdgeInterval(u, v, from2, to2);
                        intervals++;
                        for (int t = from2; t <= to2; t++) active[t].Add(new[] { u, v });
                    }
                }
            }

            var queries = new List<int[]>[Steps];
            for (int t = 0; t < Steps; t++) queries[t] = new List<int[]>();
            var slots = new List<int>();
            for (int t = 0; t < Steps; t++)
            {
                for (int p = 0; p < 6; p++)
                {
                    int u = Rng.NextInt(ref state, Vertices);
                    int v = p == 0 ? u : Rng.NextInt(ref state, Vertices);      // includes self queries
                    slots.Add(solver.AddQuery(t, u, v));
                    queries[t].Add(new[] { u, v });
                }
            }
            solver.Solve();
            Check(!solver.RollbackBroken, "no node of the 60-step tree left residue behind");

            bool agree = true;
            string firstFailure = "none";
            long naiveUnions = 0;
            for (int t = 0, index = 0; t < Steps; t++)
            {
                var reference = new RollbackDsu(Vertices);
                foreach (int[] e in active[t]) { reference.Union(e[0], e[1]); naiveUnions++; }
                foreach (int[] q in queries[t])
                {
                    bool expected = reference.Find(q[0]) == reference.Find(q[1]);
                    if (solver.ConnectedFlags[slots[index]] != expected)
                    {
                        agree = false;
                        if (firstFailure == "none") firstFailure = "t=" + t + " pair (" + q[0] + "," + q[1] + ")";
                    }
                    index++;
                }
            }
            Check(agree, "all " + solver.QueryCount + " answers match the replay oracle (" + firstFailure + ")");
            Check(intervals == solver.StoredIntervals, "every interval was registered");
            Check(solver.UnionCalls < naiveUnions,
                "offline unions (" + solver.UnionCalls + ") beat naive replay unions (" + naiveUnions + ")");
            long intervalBudget = 2L * intervals * 2 * 6;    // <= 2*log2(64) nodes per interval
            Check(solver.UnionCalls <= intervalBudget,
                "each interval costs only O(log T) unions (" + solver.UnionCalls + " <= " + intervalBudget + ")");

            // after the DFS the structure must be back to singletons: the strongest rollback check
            Check(solver.IsClean(), "the union-find is back to singleton components with an empty history");
            Console.WriteLine("steps=" + Steps + " vertices=" + Vertices + " intervals=" + intervals
                + " queries=" + solver.QueryCount + " offlineUnions=" + solver.UnionCalls
                + " naiveUnions=" + naiveUnions);
        }

        Console.WriteLine("PASS");
    }
}
