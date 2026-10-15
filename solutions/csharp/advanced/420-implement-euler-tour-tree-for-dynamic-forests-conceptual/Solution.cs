// Task 420 - Implement Euler tour tree for dynamic forests (conceptual). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// An Euler tour tree stores one Euler circuit of each tree as a sequence of DIRECTED EDGE
// events (u -> v): travelling the circuit enters u and leaves towards v, so the sequence
// chains (each event's target is the next event's source, cyclically) and every directed
// edge of the tree appears exactly once. A real ETT keeps these sequences in balanced
// search trees and links/cuts by splitting/merging them in O(log n); this conceptual
// implementation uses plain lists, so each operation costs O(size) — the operations are
// identical, only the balance is simulated away (documented deviation).
//
//   * Connect(u, v): rotate u's tour to start leaving u and v's tour to start leaving v,
//     then splice:  tour(u) + [u->v] + tour(v) + [v->u]  — a valid circuit of the merged tree.
//   * Disconnect(u, v) (edge exists): rotate to start with [u->v], find the matching
//     [v->u]; the events strictly between them form the v-side circuit, the rest (after
//     [v->u]) forms the u-side circuit.
//   * Reroot(v): rotate v's circuit so it starts with an event leaving v.
//   * Connected / ComponentVertices: answered from the owner map (a list simulation of
//     the tree lookup a real ETT answers via the roots of its balanced trees).
using System;
using System.Collections.Generic;
using System.Linq;

internal class EulerTourForest
{
    public readonly int N;
    private readonly Dictionary<int, List<int[]>> tours = new Dictionary<int, List<int[]>>();
    private readonly int[] owner;
    private int nextId;

    public EulerTourForest(int n)
    {
        N = n;
        owner = new int[n + 1];
        for (int v = 1; v <= n; v++)
        {
            owner[v] = v;
            tours[v] = new List<int[]>();
        }
        nextId = n + 1;
    }

    public List<int[]> TourOf(int x) => tours[owner[x]];

    /// <summary>Rotate so the sequence starts with an event leaving x (no-op if isolated).</summary>
    private static List<int[]> Rotate(List<int[]> tour, int x)
    {
        for (int i = 0; i < tour.Count; i++)
        {
            if (tour[i][0] == x)
            {
                var outp = new List<int[]>(tour.GetRange(i, tour.Count - i));
                outp.AddRange(tour.GetRange(0, i));
                return outp;
            }
        }
        return new List<int[]>(tour);
    }

    private static HashSet<int> Endpoints(List<int[]> tour)
    {
        var verts = new HashSet<int>();
        foreach (var e in tour)
        {
            verts.Add(e[0]);
            verts.Add(e[1]);
        }
        return verts;
    }

    private void Reassign(int id, HashSet<int> members)
    {
        foreach (int vert in members) owner[vert] = id;
    }

    /// <summary>Link u and v (must be in different components); returns true when linked.</summary>
    public bool Connect(int u, int v)
    {
        if (owner[u] == owner[v]) return false;
        var merged = new List<int[]>(Rotate(TourOf(u), u));
        merged.Add(new[] { u, v });
        merged.AddRange(Rotate(TourOf(v), v));
        merged.Add(new[] { v, u });
        int oldU = owner[u], oldV = owner[v];
        int id = nextId++;
        tours.Remove(oldU);
        tours.Remove(oldV);
        tours[id] = merged;
        var members = Endpoints(merged);
        members.Add(u);
        members.Add(v);
        Reassign(id, members);
        return true;
    }

    /// <summary>Cut the edge {u, v} when it exists; returns true when cut.</summary>
    public bool Disconnect(int u, int v)
    {
        if (owner[u] != owner[v]) return false;
        var tour = TourOf(u);
        int forward = -1, reverse = -1;
        for (int i = 0; i < tour.Count; i++)
        {
            var e = tour[i];
            if (e[0] == u && e[1] == v) forward = i;
            if (e[0] == v && e[1] == u) reverse = i;
        }
        if (forward < 0 || reverse < 0) return false;
        var rot = new List<int[]>(tour.GetRange(forward, tour.Count - forward));
        rot.AddRange(tour.GetRange(0, forward));               // starts with u -> v
        int q = -1;
        for (int i = 0; i < rot.Count; i++)
        {
            var e = rot[i];
            if (e[0] == v && e[1] == u) { q = i; break; }
        }
        var sideV = rot.GetRange(1, q - 1);                    // closed walk at v
        var sideU = rot.GetRange(q + 1, rot.Count - q - 1);    // closed walk at u
        int oldId = owner[u];
        int idU = nextId++;
        int idV = nextId++;
        tours.Remove(oldId);
        tours[idU] = sideU;
        tours[idV] = sideV;
        var mu = Endpoints(sideU);
        mu.Add(u);
        Reassign(idU, mu);
        var mv = Endpoints(sideV);
        mv.Add(v);
        Reassign(idV, mv);
        return true;
    }

    public void Reroot(int x)
    {
        int id = owner[x];
        tours[id] = Rotate(tours[id], x);
    }

    public bool Connected(int u, int v) => owner[u] == owner[v];

    public int[] ComponentVertices(int x)
    {
        var members = Endpoints(tours[owner[x]]);
        members.Add(x);
        return members.OrderBy(v => v).ToArray();
    }

    public int ComponentSize(int x) => ComponentVertices(x).Length;
}

/// <summary>Adjacency oracle with BFS components.</summary>
internal class OracleForest
{
    private readonly List<SortedSet<int>> adj;

    public OracleForest(int n)
    {
        adj = new List<SortedSet<int>>();
        for (int i = 0; i <= n; i++) adj.Add(new SortedSet<int>());
    }

    public void Link(int u, int v)
    {
        adj[u].Add(v);
        adj[v].Add(u);
    }

    public void Cut(int u, int v)
    {
        adj[u].Remove(v);
        adj[v].Remove(u);
    }

    public bool HasEdge(int u, int v) => adj[u].Contains(v);

    public HashSet<int> Component(int x)
    {
        var seen = new HashSet<int> { x };
        var queue = new Queue<int>();
        queue.Enqueue(x);
        while (queue.Count > 0)
        {
            int cur = queue.Dequeue();
            foreach (int nxt in adj[cur])
            {
                if (seen.Add(nxt)) queue.Enqueue(nxt);
            }
        }
        return seen;
    }

    public bool Connected(int u, int v) => Component(u).Contains(v);
}

public static class Solution
{
    private static int seed;

    private static int NextRand()
    {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    /// <summary>Every component's tour must be a valid Euler circuit of the oracle tree.</summary>
    private static void Validate(EulerTourForest forest, OracleForest oracle, string label)
    {
        var seen = new HashSet<int>();
        for (int start = 1; start <= forest.N; start++)
        {
            if (seen.Contains(start)) continue;
            var members = oracle.Component(start);
            foreach (int v in members) seen.Add(v);
            var tour = forest.TourOf(start);
            if (members.Count == 1)
            {
                if (tour.Count != 0) throw new Exception("FAILED: singleton tour " + label + " " + start);
                continue;
            }
            if (tour.Count != 2 * (members.Count - 1)) throw new Exception("FAILED: tour size " + label);
            for (int i = 0; i < tour.Count; i++)
            {
                if (tour[i][1] != tour[(i + 1) % tour.Count][0])
                {
                    throw new Exception("FAILED: chaining " + label + " " + i);
                }
            }
            var seenEvents = new HashSet<string>();
            var covered = new HashSet<int>();
            foreach (var e in tour)
            {
                string key = e[0] + "->" + e[1];
                if (!seenEvents.Add(key)) throw new Exception("FAILED: duplicate event " + label);
                if (!oracle.HasEdge(e[0], e[1])) throw new Exception("FAILED: bogus edge " + label);
                covered.Add(e[0]);
                covered.Add(e[1]);
            }
            if (!covered.SetEquals(members)) throw new Exception("FAILED: coverage " + label + " " + start);
            var cv = forest.ComponentVertices(start);
            if (cv.Length != members.Count) throw new Exception("FAILED: vertices " + label);
            foreach (int v in cv)
            {
                if (!members.Contains(v)) throw new Exception("FAILED: stray vertex " + label);
            }
            if (forest.ComponentSize(start) != members.Count) throw new Exception("FAILED: size " + label);
        }
    }

    private static bool SameEvents(List<int[]> tour, int[][] want)
    {
        if (tour.Count != want.Length) return false;
        for (int i = 0; i < want.Length; i++)
        {
            if (tour[i][0] != want[i][0] || tour[i][1] != want[i][1]) return false;
        }
        return true;
    }

    public static void Main()
    {
        // hand trace on three vertices
        var forest = new EulerTourForest(3);
        Check(forest.Connected(1, 1) && !forest.Connected(1, 2), "isolated");
        var single = forest.ComponentVertices(1);
        Check(single.Length == 1 && single[0] == 1, "single");
        Check(forest.Connect(1, 2), "connect 1 2");
        Check(SameEvents(forest.TourOf(1), new[] { new[] { 1, 2 }, new[] { 2, 1 } }), "pair tour");
        Check(forest.Connect(2, 3), "connect 2 3");
        // tour of the walk 1 -> 2 -> 3 -> 2 -> 1, rotated to start leaving 2
        Check(SameEvents(forest.TourOf(2), new[]
        {
            new[] { 2, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 2 }
        }), "joined tour");
        Check(forest.Connected(1, 3) && forest.ComponentSize(1) == 3, "chain");
        Check(SameEvents(forest.ComponentVertices(1).Select(v => new[] { v, v }).ToList(),
            new[] { new[] { 1, 1 }, new[] { 2, 2 }, new[] { 3, 3 } }), "chain vertices");

        // reroot rotates the circuit so that it starts leaving the new root
        forest.Reroot(3);
        Check(SameEvents(forest.TourOf(3), new[]
        {
            new[] { 3, 2 }, new[] { 2, 1 }, new[] { 1, 2 }, new[] { 2, 3 }
        }), "reroot tour");

        // cut edge 2-3: the v-side is the singleton {3}, the rest keeps {1, 2}
        Check(forest.Disconnect(2, 3), "disconnect 2 3");
        Check(!forest.Connected(1, 3), "1 3 split");
        Check(forest.Connected(1, 2), "1 2 kept");
        Check(forest.TourOf(3).Count == 0, "3 singleton");
        var side = forest.TourOf(1);
        Check(side.Count == 2, "side size");
        foreach (var e in side)
        {
            Check((e[0] == 1 && e[1] == 2) || (e[0] == 2 && e[1] == 1), "side event");
        }
        Check(forest.ComponentVertices(3).Length == 1, "3 alone");

        // cutting again without the edge is a no-op
        Check(!forest.Disconnect(2, 3), "disconnect gone");
        Check(forest.Connected(1, 2), "1 2 still");

        // a non-adjacent pair cannot be disconnected even when connected
        forest.Connect(2, 3);
        Check(forest.Connected(1, 3), "reconnected");
        Check(!forest.Disconnect(1, 3), "no 1 3 edge");
        Check(forest.Connected(1, 3), "still connected");
        Check(forest.Disconnect(1, 2), "cut 1 2");
        Check(!forest.Connected(1, 2) && forest.Connected(2, 3), "final shape");

        // scripted operations against the union-find + tree oracle
        int n = 25;
        seed = 4200;
        forest = new EulerTourForest(n);
        var oracle = new OracleForest(n);
        for (int step = 0; step < 320; step++)
        {
            int kind = NextRand() % 4;
            int u = 1 + NextRand() % n;
            int v = 1 + NextRand() % n;
            if (kind == 0)
            {
                if (u != v && !oracle.Connected(u, v))
                {
                    Check(forest.Connect(u, v), "script connect " + step);
                    oracle.Link(u, v);
                    Check(forest.Connected(u, v), "script connect check " + step);
                }
            }
            else if (kind == 1)
            {
                if (u != v && oracle.HasEdge(u, v))
                {
                    Check(forest.Disconnect(u, v), "script cut " + step);
                    oracle.Cut(u, v);
                    Check(!forest.Connected(u, v), "script cut check " + step);
                }
            }
            else if (kind == 2)
            {
                Check(forest.Connected(u, v) == oracle.Connected(u, v), "script conn " + step);
                var cv = forest.ComponentVertices(u);
                var members = oracle.Component(u);
                Check(cv.Length == members.Count, "script vertices " + step);
                foreach (int w in cv) Check(members.Contains(w), "script member " + step);
                Check(forest.ComponentSize(u) == members.Count, "script size " + step);
            }
            else
            {
                forest.Reroot(u);
                Check(forest.Connected(u, v) == oracle.Connected(u, v), "script reroot " + step);
            }
            // structural validity after every single operation
            Validate(forest, oracle, "step " + step);
            // incremental spot checks on fresh random pairs
            for (int q = 0; q < 3; q++)
            {
                int a = 1 + NextRand() % n;
                int b = 1 + NextRand() % n;
                Check(forest.Connected(a, b) == oracle.Connected(a, b), "spot " + step);
            }
        }

        // final full audit: every pair, every component
        for (int u = 1; u <= n; u++)
        {
            for (int v = u + 1; v <= n; v++)
            {
                Check(forest.Connected(u, v) == oracle.Connected(u, v), "final conn " + u + " " + v);
            }
        }
        Validate(forest, oracle, "final");
        // final shape: the deterministic script leaves one giant component plus a few
        // cut-off singletons; assert the giant exists (exact matching is oracle-checked)
        int largest = 0;
        for (int v = 1; v <= n; v++) largest = Math.Max(largest, oracle.Component(v).Count);
        Check(largest >= n - 5, "giant component");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
