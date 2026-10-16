// Task 683 - Implement compacted trie (Patricia trie) for IP routing table simulation.
// (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// A prefix table for IPv4 longest-prefix-match, as a path-compressed binary trie (Patricia / LC-trie
// style): every node carries a prefix, children are keyed by the single bit that follows the parent's
// prefix length, and chains of single-child nodes collapse into one long edge, so a table of P prefixes
// costs O(P) nodes instead of O(P * 32). Lookup walks down as long as the query matches the node's prefix,
// remembering the deepest node that carries a next hop — which is exactly longest-prefix match, because
// deeper nodes have strictly longer prefixes.
// Insert has the three cases a compressed trie needs: extend an existing branch (attach below the deepest
// matching node), insert a shorter prefix *above* a longer one, and split the edge when the new prefix
// diverges from an existing node before that node's own bit — the last case materialises a branching node
// with no route of its own, which is why the compaction invariant is "every node carries a route or has two
// children", not "every node carries a route". Delete clears a route and then re-compacts (a childless node
// disappears, a single-child node is spliced out).
// Verification is against brute force: after every mutation of every random table the whole route list is
// enumerated and compared with the inserted set, and every lookup is compared with a linear scan over all
// prefixes that takes the longest matching one. On top of that the structural invariants are re-checked
// after every operation (children extend their parent and are longer; every route is reachable; the node
// count stays within the O(P) bound; no node is left as a pointless pass-through), and the classic edge
// cases are pinned down: the default route, /32 host routes, duplicate inserts, deletion falling back to the
// covering prefix, and the no-route answer.
using System;
using System.Collections.Generic;

internal static class Rng
{
    /// splitmix64 — deterministic table and query generation.
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextBelow(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);

    internal static uint NextUInt(ref ulong state) => (uint)(Next(ref state) >> 32);
}

internal sealed class Node
{
    internal uint Bits;
    internal int Length;
    internal int NextHop = -1;
    internal Node Parent = null!;
    internal Node Left = null!;
    internal Node Right = null!;
}

internal sealed class Route
{
    internal uint Bits;
    internal int Length;
    internal int NextHop;
}

/// Bit helpers. Position 0 is the most significant bit, matching prefix notation: /8 tests bit 0..7.
internal static class Bits
{
    internal static int Bit(uint value, int position) => (int)((value >> (31 - position)) & 1u);

    internal static uint Mask(uint value, int length) =>
        length <= 0 ? 0u : length >= 32 ? value : value & (0xFFFFFFFFu << (32 - length));

    internal static bool Matches(uint address, Node node) =>
        (Bits.Mask(address, node.Length) == Bits.Mask(node.Bits, node.Length));

    internal static int FirstDifferingBit(uint a, uint b)
    {
        uint difference = a ^ b;
        if (difference == 0) return 32;
        int position = 0;
        while ((difference & 0x80000000u) == 0)
        {
            difference <<= 1;
            position++;
        }
        return position;
    }
}

internal sealed class PatriciaTrie
{
    internal Node Root = null!;
    internal int NodeCount;

    internal bool Insert(uint bits, int length, int nextHop)
    {
        bits = Bits.Mask(bits, length);
        if (Root == null)
        {
            Root = Make(bits, length, nextHop);
            return true;
        }

        Node parent = null!;
        Node current = Root;
        while (current != null && current.Length < length && Bits.Matches(bits, current))
        {
            int direction = Bits.Bit(bits, current.Length);
            parent = current;
            current = direction == 0 ? current.Left : current.Right;
        }

        if (current != null && current.Length == length && current.Bits == bits)
        {
            current.NextHop = nextHop;                 // the same prefix again: update the route
            return false;
        }

        Node node = new Node { Bits = bits, Length = length, NextHop = nextHop, Parent = parent };
        NodeCount++;

        if (current == null)
        {
            // No node at that branch: the new prefix simply extends the deepest match.
            Attach(parent, node);
            return true;
        }

        int diverging = Bits.FirstDifferingBit(bits, current.Bits);
        if (diverging >= length)
        {
            // The new prefix is a prefix of `current`, so it must sit between parent and current.
            Attach(node, current);
            ReplaceUnder(parent, current, node);
            return true;
        }

        // The prefixes diverge before the end of `current`: materialise the branching node. `parent` is the
        // node `current` currently hangs from, captured before the re-parenting below.
        var middle = new Node
        {
            Bits = Bits.Mask(bits, diverging),
            Length = diverging,
            NextHop = -1,
            Parent = parent
        };
        NodeCount++;
        Attach(middle, current);
        Attach(middle, node);
        ReplaceUnder(parent, current, middle);
        return true;
    }

    private Node Make(uint bits, int length, int nextHop)
    {
        NodeCount++;
        return new Node { Bits = bits, Length = length, NextHop = nextHop };
    }

    /// Attaches `child` in the slot selected by the bit that follows the parent's prefix.
    private void Attach(Node parent, Node child)
    {
        child.Parent = parent;
        if (parent == null)
        {
            Root = child;
            return;
        }
        if (Bits.Bit(child.Bits, parent.Length) == 0) parent.Left = child;
        else parent.Right = child;
    }

    private void ReplaceUnder(Node? parent, Node old, Node? replacement)
    {
        if (parent == null)
        {
            Root = replacement!;
            if (replacement != null) replacement.Parent = null!;
            return;
        }
        if (parent.Left == old) parent.Left = replacement!;
        else parent.Right = replacement!;
        if (replacement != null) replacement.Parent = parent;
    }

    /// Longest prefix match: the deepest node whose prefix matches the address and that carries a route.
    internal Node LookupNode(uint address)
    {
        Node best = null!;
        Node current = Root;
        while (current != null)
        {
            if (!Bits.Matches(address, current)) break;
            if (current.NextHop >= 0) best = current;
            current = Bits.Bit(address, current.Length) == 0 ? current.Left : current.Right;
        }
        return best;
    }

    internal int Lookup(uint address)
    {
        Node best = LookupNode(address);
        return best == null ? -1 : best.NextHop;
    }

    internal bool Remove(uint bits, int length)
    {
        bits = Bits.Mask(bits, length);
        Node current = Root;
        while (current != null)
        {
            if (!Bits.Matches(bits, current)) return false;
            if (current.Length == length && current.Bits == bits) break;
            if (current.Length >= length) return false;
            current = Bits.Bit(bits, current.Length) == 0 ? current.Left : current.Right;
        }
        if (current == null) return false;

        current.NextHop = -1;
        // Re-compact upwards. A node that carries a route, or that has two children, is a legitimate
        // stopping point; anything else (a route-less leaf, or a route-less pass-through with one child)
        // disappears and the walk continues from its parent.
        Node? node = current;
        while (node != null)
        {
            if (node.NextHop >= 0) break;
            int children = (node.Left == null ? 0 : 1) + (node.Right == null ? 0 : 1);
            if (children == 2) break;
            Node? parent = node.Parent;
            Node? replacement = children == 1 ? (node.Left ?? node.Right) : null;
            ReplaceUnder(parent, node, replacement);
            NodeCount--;
            node = parent;
        }
        return true;
    }

    internal List<Route> Enumerate()
    {
        var routes = new List<Route>();
        var stack = new Stack<Node>();
        if (Root != null) stack.Push(Root);
        while (stack.Count > 0)
        {
            Node node = stack.Pop();
            if (node.NextHop >= 0) routes.Add(new Route { Bits = node.Bits, Length = node.Length, NextHop = node.NextHop });
            if (node.Left != null) stack.Push(node.Left);
            if (node.Right != null) stack.Push(node.Right);
        }
        routes.Sort((a, b) => a.Length != b.Length ? a.Length - b.Length : a.Bits.CompareTo(b.Bits));
        return routes;
    }

    /// Structural invariants. Returns a description of the first problem found, or an empty string.
    internal string Validate()
    {
        if (Root == null) return NodeCount == 0 ? "" : "empty table still claims nodes";
        var seen = new HashSet<Node>();
        var stack = new Stack<Node>();
        stack.Push(Root);
        while (stack.Count > 0)
        {
            Node node = stack.Pop();
            if (!seen.Add(node)) return "the trie contains a cycle";
            if (node.NextHop < 0)
            {
                int children = (node.Left == null ? 0 : 1) + (node.Right == null ? 0 : 1);
                if (children != 2) return "a pass-through node has " + children + " children";
            }
            foreach (Node child in new[] { node.Left, node.Right })
            {
                if (child == null) continue;
                if (child.Parent != node) return "a child does not point back at its parent";
                if (child.Length <= node.Length) return "a child does not extend its parent's prefix";
                if (!Bits.Matches(child.Bits, node)) return "a child does not match its parent's prefix";
                int direction = Bits.Bit(child.Bits, node.Length);
                if (direction == 0 && node.Left != child) return "a zero-bit child sits in the one-bit slot";
                if (direction == 1 && node.Right != child) return "a one-bit child sits in the zero-bit slot";
                stack.Push(child);
            }
        }
        if (seen.Count != NodeCount) return "the node count " + NodeCount + " disagrees with the traversal " + seen.Count;
        return "";
    }

    /// The same table as a flat list, used as the brute-force reference.
    internal List<Route> Routes() => Enumerate();
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static int BruteForceLookup(List<Route> table, uint address)
    {
        int best = -1;
        int bestLength = -1;
        foreach (Route route in table)
        {
            if ((address & Bits.Mask(0xFFFFFFFFu, route.Length)) != (route.Bits & Bits.Mask(0xFFFFFFFFu, route.Length)))
            {
                continue;
            }
            if (route.Length > bestLength)
            {
                bestLength = route.Length;
                best = route.NextHop;
            }
        }
        return best;
    }

    private static string Describe(Route route)
    {
        return (route.Bits >> (32 - route.Length)) + "/" + route.Length + "->" + route.NextHop;
    }

    private static bool SameRoutes(List<Route> expected, List<Route> actual)
    {
        if (expected.Count != actual.Count) return false;
        for (int i = 0; i < expected.Count; i++)
        {
            if (expected[i].Bits != actual[i].Bits || expected[i].Length != actual[i].Length
                || expected[i].NextHop != actual[i].NextHop) return false;
        }
        return true;
    }

    public static void Main()
    {
        // ---- the classic edge cases ------------------------------------------------------------
        var edge = new PatriciaTrie();
        Check(edge.Lookup(0xC0A80001u) == -1, "an empty table answers no-route");
        Check(edge.Validate() == "", "and is structurally valid");
        edge.Insert(0x00000000u, 0, 1);
        Check(edge.Lookup(0xC0A80001u) == 1, "the default route 0.0.0.0/0 matches everything");
        Check(edge.Lookup(0xFFFFFFFFu) == 1, "including the all-ones address");
        edge.Insert(0xC0A80000u, 16, 2);
        Check(edge.Lookup(0xC0A80001u) == 2, "192.168.0.0/16 wins over the default route");
        Check(edge.Lookup(0xC0A90001u) == 1, "but a neighbouring block still falls back to the default");
        edge.Insert(0xC0A80001u, 32, 3);
        Check(edge.Lookup(0xC0A80001u) == 3, "a /32 host route wins over the /16");
        Check(edge.Lookup(0xC0A80002u) == 2, "while the rest of the /16 still uses the /16");
        Check(edge.Validate() == "", "the compressed trie is valid: " + edge.Validate());
        edge.Insert(0xC0A80001u, 32, 9);
        Check(edge.Lookup(0xC0A80001u) == 9, "re-inserting the same prefix updates its next hop");
        Check(edge.Enumerate().Count == 3, "and does not duplicate the route (got " + edge.Enumerate().Count + ")");
        Check(edge.NodeCount <= 2 * 3, "the table costs O(P) nodes: " + edge.NodeCount + " for 3 prefixes");
        Check(edge.Remove(0xC0A80001u, 32), "the host route can be deleted");
        Check(edge.Lookup(0xC0A80001u) == 2, "and the lookup falls back to the covering /16");
        Check(!edge.Remove(0xC0A80001u, 32), "deleting it twice reports failure the second time");
        Check(edge.Validate() == "", "the trie re-compacted correctly: " + edge.Validate());

        // ---- insert placed such that an edge has to be split -----------------------------------
        var split = new PatriciaTrie();
        split.Insert(0x00000000u, 0, 1);
        split.Insert(0x80000000u, 1, 2);
        split.Insert(0x40000000u, 2, 3);
        Check(split.Lookup(0x40000000u) == 3 && split.Lookup(0x80000000u) == 2 && split.Lookup(0x20000000u) == 1,
              "splitting a branch keeps every route reachable");
        Check(split.Validate() == "", "and leaves the structure valid: " + split.Validate());
        split.Insert(0x60000000u, 3, 4);
        Check(split.Lookup(0x60000000u) == 4 && split.Lookup(0x40000001u) == 3,
              "a longer prefix deepens the split branch");
        Check(split.Validate() == "", "with the structure still valid");

        // ---- randomised tables: insert, lookup, delete, re-validate -----------------------------
        ulong state = 683001UL;
        int tables = 20;
        long lookups = 0;
        int maxNodesPerPrefix = 0;
        for (int t = 0; t < tables; t++)
        {
            var trie = new PatriciaTrie();
            var table = new List<Route>();
            int prefixes = 25 + Rng.NextBelow(ref state, 25);
            for (int p = 0; p < prefixes; p++)
            {
                uint bits = Rng.NextUInt(ref state);
                int length = 1 + Rng.NextBelow(ref state, 32);
                bits = Bits.Mask(bits, length);
                int nextHop = 1 + Rng.NextBelow(ref state, 99);
                trie.Insert(bits, length, nextHop);
                bool replaced = false;
                foreach (Route existing in table)
                {
                    if (existing.Bits == bits && existing.Length == length)
                    {
                        existing.NextHop = nextHop;
                        replaced = true;
                    }
                }
                if (!replaced) table.Add(new Route { Bits = bits, Length = length, NextHop = nextHop });
                Check(trie.Validate() == "", "table " + t + ": structure valid after inserting " + Describe(table[table.Count - 1])
                      + " (" + trie.Validate() + ")");
                Check(trie.NodeCount <= 2 * table.Count + 1, "table " + t + ": node count stays linear ("
                      + trie.NodeCount + " nodes for " + table.Count + " prefixes)");
            }

            table.Sort((a, b) => a.Length != b.Length ? a.Length - b.Length : a.Bits.CompareTo(b.Bits));
            Check(SameRoutes(table, trie.Routes()), "table " + t + ": the trie enumerates exactly the inserted routes");
            if (trie.NodeCount > maxNodesPerPrefix * table.Count) maxNodesPerPrefix = trie.NodeCount / table.Count;

            for (int q = 0; q < 200; q++)
            {
                uint address = Rng.NextUInt(ref state);
                int expected = BruteForceLookup(table, address);
                int actual = trie.Lookup(address);
                Check(actual == expected, "table " + t + " query " + q + ": lookup " + address.ToString("X8")
                      + " returned " + actual + " but the longest match is " + expected);
                lookups++;
            }

            // Delete half the table and re-verify everything.
            for (int p = 0; p < table.Count; p += 2)
            {
                Route route = table[p];
                route.NextHop = -1;
                trie.Remove(route.Bits, route.Length);
            }
            var remaining = new List<Route>();
            foreach (Route route in table) if (route.NextHop >= 0) remaining.Add(route);
            Check(SameRoutes(remaining, trie.Routes()), "table " + t + ": deletion leaves exactly the remaining routes");
            Check(trie.Validate() == "", "table " + t + ": structure valid after deletions (" + trie.Validate() + ")");
            for (int q = 0; q < 100; q++)
            {
                uint address = Rng.NextUInt(ref state);
                Check(trie.Lookup(address) == BruteForceLookup(remaining, address), "table " + t + " post-delete query "
                      + q + " falls back correctly");
                lookups++;
            }
        }
        Check(lookups >= 5000, "the random tables answered " + lookups + " verified lookups");
        Check(maxNodesPerPrefix <= 2, "the compaction bound held: at most " + maxNodesPerPrefix
              + " nodes per prefix over every table");

        // ---- deletion down to an empty table ---------------------------------------------------
        var drain = new PatriciaTrie();
        drain.Insert(0x0A000000u, 8, 5);
        drain.Insert(0x0A000000u, 16, 6);
        drain.Insert(0x0A010000u, 16, 7);
        while (drain.Enumerate().Count > 0)
        {
            Route route = drain.Enumerate()[0];
            Check(drain.Remove(route.Bits, route.Length), "each route can be removed while draining");
            Check(drain.Validate() == "", "the trie stays valid while draining (" + drain.Validate() + ")");
        }
        Check(drain.NodeCount == 0 && drain.Lookup(0x0A000001u) == -1, "draining the table empties the trie");

        Console.WriteLine("edge cases: default route, /16, /32 host route, update, delete and fall back");
        Console.WriteLine("random tables: " + tables + ", " + lookups + " verified lookups, at most "
              + maxNodesPerPrefix + " nodes per prefix");
        Console.WriteLine("PASS");
    }
}
