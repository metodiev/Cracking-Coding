// Task 684 - Implement longest-prefix-match lookup optimized (binary trie). (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// A path-compressed (Patricia-style) binary trie over 32-bit IPv4 addresses. Every node owns
// the prefix bits of its own path (right-padded with zeros) and therefore knows its depth, so
// one "does the key agree with this node's whole path" test walks an entire chain of
// single-child bit levels at once instead of touching every level. A route (prefix, length,
// next hop) lives exactly at the node whose depth equals the route length, so a lookup only has
// to remember the deepest route met on the root-to-leaf walk.
//
// Invariants maintained by Insert and re-verified after every mutation:
//   * a node's route length always equals the node's depth;
//   * a node's path extends its parent's path and takes the parent's branch bit;
//   * depth strictly increases along a path, so a lookup touches at most 33 nodes no matter how
//     many routes are stored (the "optimized" part compared with a linear table scan).
//
// The trie is cross-checked against a brute-force linear scan of the route list on random
// tables, and against itself for insertion-order independence and route replacement.
using System;
using System.Collections.Generic;

internal static class Bits
{
    public static int Bit(uint value, int index) => (int)((value >> (31 - index)) & 1u);

    /// <summary>Keeps the top <paramref name="count"/> bits and zeroes the rest.</summary>
    public static uint TopBits(uint value, int count) => count == 0 ? 0u : (value >> (32 - count)) << (32 - count);

    public static string Ip(uint value) =>
        $"{(value >> 24) & 255}.{(value >> 16) & 255}.{(value >> 8) & 255}.{value & 255}";

    public static uint Parse(string dotted)
    {
        string[] parts = dotted.Split('.');
        uint value = 0;
        foreach (string part in parts) value = (value << 8) | uint.Parse(part);
        return value;
    }
}

internal sealed class Route
{
    public readonly uint Prefix;
    public readonly int Length;
    public readonly int NextHop;

    public Route(uint prefix, int length, int nextHop)
    {
        Prefix = Bits.TopBits(prefix, length);
        Length = length;
        NextHop = nextHop;
    }

    public override string ToString() => $"{Bits.Ip(Prefix)}/{Length}->{NextHop}";
}

internal sealed class PatriciaTrie
{
    private sealed class Node
    {
        public int Depth;          // number of path bits owned by this node
        public uint Path;          // those bits, low bits zeroed
        public Route? Route;       // non-null only when Route.Length == Depth
        public Node? Child0;       // subtree whose bit at Depth is 0
        public Node? Child1;
    }

    private readonly Node _root = new Node { Depth = 0, Path = 0 };
    private int _nodes = 1;
    private long _lookupHops;

    public int NodeCount => _nodes;

    /// <summary>Nodes visited by the most recent lookup - used to show O(prefix length) cost.</summary>
    public long LastLookupHops => _lookupHops;

    public int RouteCount => CountRoutes(_root);

    private static int CountRoutes(Node node)
    {
        int total = node.Route == null ? 0 : 1;
        if (node.Child0 != null) total += CountRoutes(node.Child0);
        if (node.Child1 != null) total += CountRoutes(node.Child1);
        return total;
    }

    /// <summary>True when <paramref name="key"/> copies <paramref name="node"/>'s path bits.</summary>
    private static bool AgreesOnPath(uint key, Node node) =>
        node.Depth == 0 || ((key ^ node.Path) >> (32 - node.Depth)) == 0;

    public void Insert(uint prefix, int length, int nextHop)
    {
        if (length < 0 || length > 32) throw new ArgumentOutOfRangeException(nameof(length));
        InsertCore(new Route(prefix, length, nextHop));
    }

    private void InsertCore(Route route)
    {
        Node node = _root;
        while (node.Depth < route.Length)                     // bits below the route length decide the branch
        {
            int bit = Bits.Bit(route.Prefix, node.Depth);
            Node? child = bit == 0 ? node.Child0 : node.Child1;
            if (child == null)
            {
                var fresh = new Node { Depth = route.Length, Path = Bits.TopBits(route.Prefix, route.Length), Route = route };
                _nodes++;
                if (bit == 0) node.Child0 = fresh; else node.Child1 = fresh;
                return;
            }
            if (!AgreesOnPath(route.Prefix, child))
            {
                // The compressed edge carries fewer bits than hoped: cut it at the first difference.
                int depth = FirstDifference(route.Prefix, child, node.Depth);
                Split(node, child, depth, depth == route.Length ? route : null);
                if (depth == route.Length) return;
                InsertCore(route);                            // the route still has bits below the new fork
                return;
            }
            if (child.Depth > route.Length)
            {
                // The new route is a strict prefix of the child's path, so it becomes a fork on that edge.
                Split(node, child, route.Length, route);
                return;
            }
            node = child;
        }
        node.Route = route;                                   // replace-on-same-prefix semantics
    }

    private static int FirstDifference(uint key, Node child, int parentDepth)
    {
        for (int i = parentDepth + 1; i < child.Depth; i++)
            if (Bits.Bit(key, i) != Bits.Bit(child.Path, i)) return i;
        throw new InvalidOperationException("compressed edge already agrees");
    }

    /// <summary>Replaces the edge parent->child by parent->mid->child, putting a route on mid.</summary>
    private void Split(Node parent, Node child, int depth, Route? route)
    {
        var mid = new Node { Depth = depth, Path = Bits.TopBits(child.Path, depth), Route = route };
        _nodes++;
        if (Bits.Bit(child.Path, parent.Depth) == 0) parent.Child0 = mid; else parent.Child1 = mid;
        if (Bits.Bit(child.Path, depth) == 0) mid.Child0 = child; else mid.Child1 = child;
    }

    /// <summary>Longest matching route for <paramref name="key"/>, or null when nothing matches.</summary>
    public Route? Lookup(uint key)
    {
        _lookupHops = 0;
        Node node = _root;
        Route? best = node.Route;                              // depth 0 == the default route
        while (node.Depth < 32)
        {
            _lookupHops++;
            Node? child = Bits.Bit(key, node.Depth) == 0 ? node.Child0 : node.Child1;
            if (child == null || !AgreesOnPath(key, child)) break;
            node = child;
            if (node.Route != null) best = node.Route;          // deeper route always wins
        }
        return best;
    }

    public bool Validate() => ValidateNode(_root, null, -1);

    private static bool ValidateNode(Node node, Node? parent, int branchBit)
    {
        if (node.Depth < 0 || node.Depth > 32) return false;
        if (node.Path != Bits.TopBits(node.Path, node.Depth)) return false;
        if (parent != null)
        {
            if (node.Depth <= parent.Depth) return false;
            if (Bits.TopBits(node.Path, parent.Depth) != parent.Path) return false;
            if (Bits.Bit(node.Path, parent.Depth) != branchBit) return false;
        }
        if (node.Route != null)
        {
            if (node.Route.Length != node.Depth) return false;
            if (Bits.TopBits(node.Route.Prefix, node.Depth) != node.Path) return false;
        }
        if (node.Child0 != null && !ValidateNode(node.Child0, node, 0)) return false;
        if (node.Child1 != null && !ValidateNode(node.Child1, node, 1)) return false;
        return true;
    }

    /// <summary>Depth of the deepest node, i.e. the largest number of forks on any path.</summary>
    public int MaxDepth() => MaxDepthOf(_root);

    private static int MaxDepthOf(Node node)
    {
        int best = node.Depth;
        if (node.Child0 != null) best = Math.Max(best, MaxDepthOf(node.Child0));
        if (node.Child1 != null) best = Math.Max(best, MaxDepthOf(node.Child1));
        return best;
    }
}

/// <summary>Deliberately naive oracle: scan every route and keep the longest match.</summary>
internal static class BruteForce
{
    public static int Lookup(List<Route> routes, uint key)
    {
        int bestLength = -1, bestHop = -1;
        foreach (Route route in routes)
        {
            if (route.Length <= bestLength) continue;
            if (Bits.TopBits(key, route.Length) != route.Prefix) continue;
            bestLength = route.Length;
            bestHop = route.NextHop;
        }
        return bestHop;
    }
}

internal static class Rng
{
    private static ulong _state = 0x9E3779B97F4A7C15UL;

    public static void Seed(ulong seed) => _state = seed;

    /// <summary>splitmix64 - fixed seed, so every run produces the same tables.</summary>
    public static ulong Next()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    public static int NextInt(int bound) => (int)(Next() % (ulong)bound);

    public static uint NextUInt() => (uint)Next();
}

public static class Solution
{
    public static void Main()
    {
        var trie = new PatriciaTrie();
        Check(trie.Lookup(Bits.Parse("10.0.0.1")) == null, "empty trie has no match");
        Check(trie.NodeCount == 1 && trie.RouteCount == 0, "empty trie is just the root");
        Check(trie.Validate(), "empty trie invariant");

        // A single host route: only the exact address matches.
        trie.Insert(Bits.Parse("192.168.1.7"), 32, 7);
        Check(trie.Lookup(Bits.Parse("192.168.1.7"))!.NextHop == 7, "/32 exact match");
        Check(trie.Lookup(Bits.Parse("192.168.1.6")) == null, "/32 rejects neighbour");
        Check(trie.Lookup(Bits.Parse("192.168.1.7")).Length == 32, "/32 reports full length");

        // Overlapping prefixes: the longest one wins, and the shorter ones stay reachable.
        var lpm = new PatriciaTrie();
        lpm.Insert(Bits.Parse("10.0.0.0"), 8, 8);
        lpm.Insert(Bits.Parse("10.1.0.0"), 16, 16);
        lpm.Insert(Bits.Parse("10.1.2.0"), 24, 24);
        Check(lpm.Lookup(Bits.Parse("10.1.2.9")).NextHop == 24, "/24 beats /16 and /8");
        Check(lpm.Lookup(Bits.Parse("10.1.9.9")).NextHop == 16, "/16 beats /8");
        Check(lpm.Lookup(Bits.Parse("10.9.9.9")).NextHop == 8, "/8 is the fallback inside 10/8");
        Check(lpm.Lookup(Bits.Parse("11.0.0.1")) == null, "outside 10/8 nothing matches");

        // Boundary lengths: /0 catches everything, /31 and /30 only their own pair/quad.
        var boundary = new PatriciaTrie();
        Check(boundary.Lookup(Bits.Parse("8.8.8.8")) == null, "/0 absent before insert");
        boundary.Insert(0u, 0, 99);
        Check(boundary.Lookup(0u).NextHop == 99 && boundary.Lookup(uint.MaxValue).NextHop == 99, "/0 matches all keys");
        boundary.Insert(Bits.Parse("172.16.0.0"), 31, 31);
        Check(boundary.Lookup(Bits.Parse("172.16.0.0")).NextHop == 31, "/31 low address");
        Check(boundary.Lookup(Bits.Parse("172.16.0.1")).NextHop == 31, "/31 high address");
        Check(boundary.Lookup(Bits.Parse("172.16.0.2")).NextHop == 99, "/31 stops after two addresses");
        boundary.Insert(Bits.Parse("172.16.0.2"), 30, 30);
        Check(boundary.Lookup(Bits.Parse("172.16.0.3")).NextHop == 30, "/30 covers its quad");

        // Low bits of the prefix argument are masked away, so both spellings are the same route.
        var masked = new PatriciaTrie();
        masked.Insert(Bits.Parse("10.1.2.3"), 24, 1);
        Check(masked.Lookup(Bits.Parse("10.1.2.200")).NextHop == 1, "non-zero host bits masked");
        Check(masked.Lookup(Bits.Parse("10.1.3.0")) == null, "masked route keeps its /24 boundary");

        // Replacement keeps the node count and the invariant intact.
        int nodesBefore = masked.NodeCount;
        masked.Insert(Bits.Parse("10.1.2.99"), 24, 2);
        Check(masked.Lookup(Bits.Parse("10.1.2.5")).NextHop == 2, "re-insert replaces next hop");
        Check(masked.NodeCount == nodesBefore && masked.RouteCount == 1, "re-insert does not grow the trie");
        Check(masked.Validate(), "replace keeps invariants");

        // Long chains of single-child levels must be compressed, not expanded.
        var compressed = new PatriciaTrie();
        var chainRoutes = new List<Route>();
        for (int i = 0; i < 24; i++)
        {
            uint prefix = 0x0A000000u | ((uint)i << 20);       // 10.i.0.0/20
            compressed.Insert(prefix, 20, 100 + i);
            chainRoutes.Add(new Route(prefix, 20, 100 + i));
        }
        Check(compressed.NodeCount < 24 * 20 / 4, "path compression keeps the node count small");
        Check(compressed.NodeCount >= compressed.RouteCount, "at least one node per route");
        Check(compressed.Validate(), "compressed trie invariant");
        for (int i = 0; i < 24; i++)
        {
            uint probe = 0x0A000000u | ((uint)i << 20) | 0x000ABC;
            Check(compressed.Lookup(probe).NextHop == 100 + i, "compressed chain member " + i);
        }

        // Randomized cross-check: 70 routes, 300 probes, brute force must agree everywhere.
        Rng.Seed(0x5EED684UL);
        var randomTrie = new PatriciaTrie();
        var randomRoutes = new List<Route>();
        for (int i = 0; i < 70; i++)
        {
            int length = new[] { 0, 4, 8, 12, 16, 20, 24, 28, 30, 31, 32 }[Rng.NextInt(11)];
            uint prefix = Rng.NextUInt();
            int hop = 1000 + i;
            var route = new Route(prefix, length, hop);
            randomTrie.Insert(prefix, length, hop);
            ReplaceEquivalent(randomRoutes, route);
        }
        Check(randomTrie.RouteCount == randomRoutes.Count, "random table size matches oracle");
        for (int i = 0; i < 300; i++)
        {
            uint key = (i % 3 == 0) ? Rng.NextUInt() : (Rng.NextUInt() & 0xFFFFFF00u);
            Route? hit = randomTrie.Lookup(key);
            int expected = BruteForce.Lookup(randomRoutes, key);
            int actual = hit == null ? -1 : hit.NextHop;
            Check(actual == expected, $"random probe {i} key={Bits.Ip(key)} got={actual} want={expected}");
        }
        Check(randomTrie.Validate(), "random trie invariant");

        // Lookup cost must depend on the address width, not on the table size.
        long maxHops = 0;
        for (int i = 0; i < 400; i++)
        {
            randomTrie.Lookup(Rng.NextUInt());
            maxHops = Math.Max(maxHops, randomTrie.LastLookupHops);
        }
        Check(maxHops <= 33, "lookup visits at most 33 nodes, saw " + maxHops);
        Check(randomTrie.MaxDepth() <= 32, "no path is deeper than 32 bits");

        // Insertion order must not change the outcome.
        var forward = new PatriciaTrie();
        var backward = new PatriciaTrie();
        for (int i = 0; i < randomRoutes.Count; i++) forward.Insert(randomRoutes[i].Prefix, randomRoutes[i].Length, randomRoutes[i].NextHop);
        for (int i = randomRoutes.Count - 1; i >= 0; i--) backward.Insert(randomRoutes[i].Prefix, randomRoutes[i].Length, randomRoutes[i].NextHop);
        Check(forward.RouteCount == backward.RouteCount, "order-independent route count");
        for (int i = 0; i < 200; i++)
        {
            uint key = Rng.NextUInt();
            Route? a = forward.Lookup(key);
            Route? b = backward.Lookup(key);
            Check((a == null) == (b == null), "order-independent hit/miss");
            if (a != null) Check(a.NextHop == b.NextHop && a.Length == b.Length, "order-independent match " + Bits.Ip(key));
        }

        // Rejecting an impossible prefix length is part of the contract.
        bool rejected = false;
        try { forward.Insert(1u, 33, 1); }
        catch (ArgumentOutOfRangeException) { rejected = true; }
        Check(rejected, "length 33 is rejected");

        Console.WriteLine($"routes={forward.RouteCount} nodes={forward.NodeCount} maxdepth={forward.MaxDepth()}");
        Console.WriteLine("PASS");
    }

    private static void ReplaceEquivalent(List<Route> routes, Route route)
    {
        for (int i = 0; i < routes.Count; i++)
        {
            if (routes[i].Length == route.Length && routes[i].Prefix == route.Prefix)
            {
                routes[i] = route;
                return;
            }
        }
        routes.Add(route);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
