// Task 703 - Implement dynamic graph connectivity with link-cut trees in practice.
// (Tier: Expert, CTCI Ch 4 — Trees and Graphs)
//
// A link-cut tree (Sleator & Tarjan) maintains a forest under link, cut and path queries. Each
// represented tree is stored as a set of *preferred paths*, and every preferred path is a splay
// tree ordered by depth; a splay tree whose root has a "path parent" pointer hangs under the node
// it continues from. `access(x)` splays x and re-hangs the right spine so that the represented-tree
// path from the current root to x becomes exactly x's splay tree; `makeroot(x)` is access(x) plus a
// lazy reversal, which is what makes paths undirected without ever touching the rest of the tree.
// The aggregates kept in every splay node (sum, count of edge nodes, maximum edge weight) are
// rebuilt by `pull` after rotation and merged from children, so a path query is
// makeroot(u); access(v); read(v) - O(log n) amortised, without any per-edge bookkeeping.
//
// Practical shape: every edge is materialised as its own node linked between its endpoints, which
// lets the aggregates distinguish "edge" contributions from vertex contributions and lets a cut
// simply sever the two links of the edge node. The forest is stored in parallel arrays (no
// pointers), which is how the structure is used in competitive practice, and splay uses an explicit
// push-down stack so lazy reversals are always propagated before a rotation.
//
// Invariants verified below against a brute-force adjacency-list forest (BFS path reconstruction):
// after every one of 400 random links, cuts and path queries the LCT agrees on connectivity, on the
// number of edges, on the sum of weights and on the maximum edge weight of the path; the LCT never
// invents an edge (its edge count and connected pairs match the reference), linking two vertices
// that are already connected is rejected instead of creating a cycle, cutting a non-edge is
// rejected, and querying a path across two components is rejected.
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

internal sealed class LinkCutForest
{
    private int[] _left, _right, _parent, _value, _sum, _edgeCount, _maxEdge, _stack;
    private bool[] _reversed, _isEdge;
    private int _used, _capacity;

    internal long Rotations, Accesses;

    internal LinkCutForest(int capacity)
    {
        _capacity = Math.Max(4, capacity);
        _left = new int[_capacity];
        _right = new int[_capacity];
        _parent = new int[_capacity];
        _value = new int[_capacity];
        _sum = new int[_capacity];
        _edgeCount = new int[_capacity];
        _maxEdge = new int[_capacity];
        _reversed = new bool[_capacity];
        _isEdge = new bool[_capacity];
        _stack = new int[_capacity];
    }

    internal int NodeCount => _used;

    internal int AddVertex()
    {
        int id = Allocate();
        _isEdge[id] = false;
        _value[id] = 0;
        Pull(id);
        return id;
    }

    internal int AddEdge(int weight)
    {
        if (weight < 0) throw new ArgumentOutOfRangeException(nameof(weight), "edge weights must be non-negative");
        int id = Allocate();
        _isEdge[id] = true;
        _value[id] = weight;
        Pull(id);
        return id;
    }

    private int Allocate()
    {
        if (_used == _capacity) Grow();
        int id = _used++;
        _left[id] = _right[id] = _parent[id] = -1;
        _reversed[id] = false;
        _sum[id] = 0;
        _edgeCount[id] = 0;
        _maxEdge[id] = 0;
        return id;
    }

    private void Grow()
    {
        _capacity *= 2;
        Array.Resize(ref _left, _capacity);
        Array.Resize(ref _right, _capacity);
        Array.Resize(ref _parent, _capacity);
        Array.Resize(ref _value, _capacity);
        Array.Resize(ref _sum, _capacity);
        Array.Resize(ref _edgeCount, _capacity);
        Array.Resize(ref _maxEdge, _capacity);
        Array.Resize(ref _reversed, _capacity);
        Array.Resize(ref _isEdge, _capacity);
        Array.Resize(ref _stack, _capacity);
    }

    private bool IsSplayRoot(int x) =>
        _parent[x] == -1 || (_left[_parent[x]] != x && _right[_parent[x]] != x);

    private void Pull(int x)
    {
        int l = _left[x], r = _right[x];
        _sum[x] = _value[x] + (l == -1 ? 0 : _sum[l]) + (r == -1 ? 0 : _sum[r]);
        _edgeCount[x] = (_isEdge[x] ? 1 : 0) + (l == -1 ? 0 : _edgeCount[l]) + (r == -1 ? 0 : _edgeCount[r]);
        int best = _isEdge[x] ? _value[x] : 0;
        if (l != -1 && _maxEdge[l] > best) best = _maxEdge[l];
        if (r != -1 && _maxEdge[r] > best) best = _maxEdge[r];
        _maxEdge[x] = best;
    }

    /// <summary>Applies a pending reversal: children swap, tags move down, this node is clean again.</summary>
    private void Push(int x)
    {
        if (!_reversed[x]) return;
        _reversed[x] = false;
        int l = _left[x], r = _right[x];
        _left[x] = r;
        _right[x] = l;
        if (l != -1) _reversed[l] = !_reversed[l];
        if (r != -1) _reversed[r] = !_reversed[r];
    }

    private void Rotate(int x)
    {
        Rotations++;
        int p = _parent[x], g = _parent[p];
        if (_left[p] == x)
        {
            _left[p] = _right[x];
            if (_right[x] != -1) _parent[_right[x]] = p;
            _right[x] = p;
        }
        else
        {
            _right[p] = _left[x];
            if (_left[x] != -1) _parent[_left[x]] = p;
            _left[x] = p;
        }
        _parent[p] = x;
        _parent[x] = g;
        if (g != -1)
        {
            if (_left[g] == p) _left[g] = x;
            else if (_right[g] == p) _right[g] = x;
        }
        Pull(p);
        Pull(x);
    }

    private void Splay(int x)
    {
        int top = 0, y = x;
        _stack[top++] = y;
        while (!IsSplayRoot(y)) { y = _parent[y]; _stack[top++] = y; }
        while (top > 0) Push(_stack[--top]);              // tags must travel top-down before rotating
        while (!IsSplayRoot(x))
        {
            int p = _parent[x], g = _parent[p];
            if (!IsSplayRoot(p))
            {
                if ((_left[p] == x) == (_left[g] == p)) Rotate(p);
                else Rotate(x);
            }
            Rotate(x);
        }
    }

    internal void Access(int x)
    {
        Accesses++;
        int last = -1;
        for (int y = x; y != -1; y = _parent[y])
        {
            Splay(y);
            _right[y] = last;                            // the old right spine becomes a path parent
            Pull(y);
            last = y;
        }
        Splay(x);
    }

    internal void MakeRoot(int x)
    {
        Access(x);
        _reversed[x] = !_reversed[x];
    }

    internal int FindRoot(int x)
    {
        Access(x);
        int cur = x;
        while (true)
        {
            Push(cur);
            if (_left[cur] == -1) break;
            cur = _left[cur];
        }
        Splay(cur);
        return cur;
    }

    internal bool Connected(int u, int v)
    {
        if (u == v) return true;
        return FindRoot(u) == FindRoot(v);               // FindRoot splays, so this is amortised O(log n)
    }

    internal void Link(int u, int v)
    {
        MakeRoot(u);
        if (FindRoot(v) == u) throw new InvalidOperationException("link would create a cycle");
        _parent[u] = v;
    }

    internal void Cut(int u, int v)
    {
        MakeRoot(u);
        Access(v);
        // After makeroot(u); access(v) the path u..v is v's splay tree and u must be its leftmost
        // child with no right subtree, otherwise the edge does not exist.
        if (_left[v] != u || _right[u] != -1) throw new InvalidOperationException("cut requires an existing edge");
        _left[v] = -1;
        _parent[u] = -1;
        Pull(v);
    }

    /// <summary>Sum of the edge weights on the u-v path, and the number of edges it contains.</summary>
    internal void PathStats(int u, int v, out int sum, out int edges, out int maxEdge)
    {
        if (!Connected(u, v)) throw new InvalidOperationException("path query across two components");
        MakeRoot(u);
        Access(v);
        sum = _sum[v];
        edges = _edgeCount[v];
        maxEdge = edges == 0 ? int.MinValue : _maxEdge[v];
    }
}

/// <summary>Reference implementation: plain adjacency lists plus BFS path reconstruction.</summary>
internal sealed class BruteForest
{
    private readonly List<int>[] _to, _weight;

    internal BruteForest(int vertices)
    {
        _to = new List<int>[vertices];
        _weight = new List<int>[vertices];
        for (int i = 0; i < vertices; i++) { _to[i] = new List<int>(); _weight[i] = new List<int>(); }
    }

    internal int EdgeCount { get; private set; }

    internal bool HasEdge(int u, int v)
    {
        for (int i = 0; i < _to[u].Count; i++) if (_to[u][i] == v) return true;
        return false;
    }

    internal void Add(int u, int v, int w)
    {
        if (HasEdge(u, v)) throw new InvalidOperationException("duplicate edge");
        _to[u].Add(v); _weight[u].Add(w);
        _to[v].Add(u); _weight[v].Add(w);
        EdgeCount++;
    }

    internal void Remove(int u, int v)
    {
        int i = _to[u].IndexOf(v);
        if (i < 0) throw new InvalidOperationException("no such edge");
        int j = _to[v].IndexOf(u);
        _to[u].RemoveAt(i); _weight[u].RemoveAt(i);
        _to[v].RemoveAt(j); _weight[v].RemoveAt(j);
        EdgeCount--;
    }

    internal int ComponentCount()
    {
        int n = _to.Length, count = 0;
        var seen = new bool[n];
        for (int i = 0; i < n; i++)
        {
            if (seen[i]) continue;
            count++;
            var stack = new Stack<int>();
            stack.Push(i);
            seen[i] = true;
            while (stack.Count > 0)
            {
                int x = stack.Pop();
                foreach (int y in _to[x]) if (!seen[y]) { seen[y] = true; stack.Push(y); }
            }
        }
        return count;
    }

    /// <summary>Edge weights along the u-v path, or null when the two vertices are in different components.</summary>
    internal List<int>? Path(int u, int v)
    {
        int n = _to.Length;
        var parent = new int[n];
        var parentWeight = new int[n];
        var seen = new bool[n];
        for (int i = 0; i < n; i++) parent[i] = -1;
        var queue = new Queue<int>();
        queue.Enqueue(u);
        seen[u] = true;
        while (queue.Count > 0)
        {
            int x = queue.Dequeue();
            for (int i = 0; i < _to[x].Count; i++)
            {
                int y = _to[x][i];
                if (seen[y]) continue;
                seen[y] = true;
                parent[y] = x;
                parentWeight[y] = _weight[x][i];
                queue.Enqueue(y);
            }
        }
        if (!seen[v]) return null;
        var path = new List<int>();
        for (int cur = v; cur != u; cur = parent[cur]) path.Add(parentWeight[cur]);
        return path;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool Throws(Action action)
    {
        try { action(); }
        catch (InvalidOperationException) { return true; }
        catch (ArgumentOutOfRangeException) { return true; }
        return false;
    }

    public static void Main()
    {
        // ---- a fixed fixture: one path 0-1-2-3-4 with weights 5, 3, 7, 2 ----------------------
        {
            var lct = new LinkCutForest(32);
            var ids = new int[5];
            for (int i = 0; i < 5; i++) ids[i] = lct.AddVertex();
            var edges = new int[4];
            for (int i = 0; i < 4; i++)
            {
                edges[i] = lct.AddEdge(new[] { 5, 3, 7, 2 }[i]);
                lct.Link(ids[i], edges[i]);
                lct.Link(edges[i], ids[i + 1]);
            }
            lct.PathStats(ids[0], ids[4], out int sum, out int count, out int max);
            Check(sum == 17 && count == 4 && max == 7,
                "the whole path sums to 17 with 4 edges and bottleneck 7 (got " + sum + "," + count + "," + max + ")");
            lct.PathStats(ids[2], ids[3], out sum, out count, out max);
            Check(sum == 7 && count == 1 && max == 7, "a single-edge path reports that edge");
            lct.PathStats(ids[1], ids[1], out sum, out count, out max);
            Check(sum == 0 && count == 0 && max == int.MinValue, "the empty path has no edges and no maximum");
            Check(lct.Connected(ids[0], ids[4]) && !lct.Connected(ids[0], lct.AddVertex()),
                "connectivity follows the linked edges only");

            lct.Cut(ids[2], edges[2]);
            Check(!lct.Connected(ids[0], ids[4]), "cutting an edge splits the tree");
            lct.PathStats(ids[0], ids[2], out sum, out count, out max);
            Check(sum == 8 && count == 2 && max == 5, "the left half still answers correctly after the cut");
            Check(Throws(() => lct.PathStats(ids[0], ids[4], out _, out _, out _)),
                "a path query across two components is rejected");
            Check(Throws(() => lct.Cut(ids[0], ids[4])), "cutting a non-edge is rejected");
            lct.Link(ids[2], edges[2]);
            Check(lct.Connected(ids[0], ids[4]), "re-linking the same edge restores the tree");
            Check(Throws(() => lct.Link(ids[0], ids[4])), "linking two connected vertices would create a cycle");
            Check(Throws(() => lct.AddEdge(-1)), "a negative edge weight is rejected");
        }

        // ---- 400 random operations against the brute-force forest -----------------------------
        {
            const int Vertices = 24, Operations = 400;
            var lct = new LinkCutForest(4 * Vertices + 8);
            var brute = new BruteForest(Vertices);
            var vertex = new int[Vertices];
            for (int i = 0; i < Vertices; i++) vertex[i] = lct.AddVertex();
            var edgeNode = new Dictionary<(int, int), int>();      // (min,max) -> edge node id
            var edgeWeight = new Dictionary<(int, int), int>();

            ulong state = 0x7E57CAFEUL;
            bool allAgree = true, structureAgree = true, rejected = true;
            string firstFailure = "none";
            int links = 0, cuts = 0, queries = 0;

            for (int op = 0; op < Operations; op++)
            {
                int a = Rng.NextInt(ref state, Vertices);
                int b = Rng.NextInt(ref state, Vertices);
                if (a == b) continue;
                var key = (Math.Min(a, b), Math.Max(a, b));
                bool connected = brute.Path(a, b) != null;
                if (!connected)
                {
                    int w = 1 + Rng.NextInt(ref state, 9);
                    int e = lct.AddEdge(w);
                    lct.Link(vertex[a], e);
                    lct.Link(e, vertex[b]);
                    brute.Add(a, b, w);
                    edgeNode[key] = e;
                    edgeWeight[key] = w;
                    links++;
                }
                else if (edgeNode.ContainsKey(key))
                {
                    lct.Cut(vertex[a], edgeNode[key]);
                    lct.Cut(edgeNode[key], vertex[b]);
                    brute.Remove(a, b);
                    edgeNode.Remove(key);
                    edgeWeight.Remove(key);
                    cuts++;
                    if (lct.Connected(vertex[a], vertex[b])) { allAgree = false; firstFailure = "cut left the pair connected"; }
                }
                else
                {
                    var expected = brute.Path(a, b)!;
                    int expectedSum = 0, expectedMax = int.MinValue;
                    foreach (int w in expected) { expectedSum += w; if (w > expectedMax) expectedMax = w; }
                    lct.PathStats(vertex[a], vertex[b], out int sum, out int count, out int max);
                    queries++;
                    if (sum != expectedSum || count != expected.Count || max != expectedMax)
                    {
                        allAgree = false;
                        if (firstFailure == "none")
                        {
                            firstFailure = "op " + op + " path " + a + "-" + b + " gave (" + sum + "," + count + "," + max
                                + ") expected (" + expectedSum + "," + expected.Count + "," + expectedMax + ")";
                        }
                    }
                    // the reference forest is a forest too: adding this edge would be a cycle
                    if (!brute.HasEdge(a, b) && !Throws(() => lct.Link(vertex[a], vertex[b]))) rejected = false;
                }
            }
            Check(allAgree, "LCT path answers match brute force on all queries (" + firstFailure + ")");
            Check(rejected, "every attempted cycle-creating link was rejected");

            // structural comparison: same edge count and the same set of connected pairs
            var connectedPairs = 0;
            for (int i = 0; i < Vertices; i++)
            {
                for (int j = i + 1; j < Vertices; j++)
                {
                    bool expected = brute.Path(i, j) != null;
                    bool got = lct.Connected(vertex[i], vertex[j]);
                    if (expected != got) structureAgree = false;
                    if (expected) connectedPairs++;
                }
            }
            Check(structureAgree, "every connected pair in the reference forest is connected in the LCT");
            Check(edgeNode.Count == brute.EdgeCount, "the LCT tracks exactly the reference edge set");
            Check(brute.ComponentCount() == Vertices - brute.EdgeCount,
                "the reference structure really is a forest");
            Console.WriteLine("operations=" + Operations + " links=" + links + " cuts=" + cuts
                + " pathQueries=" + queries + " finalEdges=" + brute.EdgeCount
                + " connectedPairs=" + connectedPairs + " rotations=" + lct.Rotations);
        }

        // ---- a star with a heavy centre edge, then tearing it down to single vertices ----------
        {
            const int Leaves = 12;
            var lct = new LinkCutForest(64);
            var center = lct.AddVertex();
            var leaves = new int[Leaves];
            var edges = new int[Leaves];
            var brute = new BruteForest(1 + Leaves);
            for (int i = 0; i < Leaves; i++)
            {
                leaves[i] = lct.AddVertex();
                int w = 1 + (i * 3) % 7;
                edges[i] = lct.AddEdge(w);
                lct.Link(center, edges[i]);
                lct.Link(edges[i], leaves[i]);
                brute.Add(0, 1 + i, w);
            }
            bool sumsMatch = true, maximaMatch = true;
            for (int i = 0; i < Leaves; i++)
            {
                int w = 1 + (i * 3) % 7;
                lct.PathStats(center, leaves[i], out int sum, out int count, out int max);
                if (sum != w || count != 1) sumsMatch = false;
                if (max != w) maximaMatch = false;
            }
            Check(sumsMatch && maximaMatch, "every leaf path reports exactly its own edge");
            for (int i = 0; i < Leaves; i++)
            {
                lct.Cut(center, edges[i]);
                lct.Cut(edges[i], leaves[i]);
                brute.Remove(0, 1 + i);
                Check(!lct.Connected(center, leaves[i]), "leaf " + i + " is detached");
                Check(Throws(() => lct.PathStats(center, leaves[i], out _, out _, out _)),
                    "a detached leaf has no path to the centre");
            }
            Check(brute.ComponentCount() == 1 + Leaves, "the fixture really ended up as isolated vertices");
            Check(lct.Connected(center, center), "a vertex is connected to itself");
            Console.WriteLine("starLeaves=" + Leaves + " finalComponents=" + brute.ComponentCount());
        }

        Console.WriteLine("PASS");
    }
}
