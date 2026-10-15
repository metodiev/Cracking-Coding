// Task 470 - Implement consistent hashing ring (virtual nodes). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Consistent-hashing ring with VNODES = 64 virtual nodes per physical node. Ring
// positions come from the FIXED LCG hash (documented, deterministic): fold over
// bytes with h = (h*75 + 74 + ch) mod 65537, then two LCG avalanche steps
// h = (h*75 + 74) mod 65537 (the plain fold alone clusters badly - measured shares
// 0.02..0.48; the avalanched hash balances). Points are sorted by (position, node
// label); a key hashes with the same function and is owned by the first vnode
// clockwise at or after its position (wrap-around).
//
// Checks on 4 physical nodes x 64 vnodes and 4096 keys:
//   determinism (identical rebuild), bisect lookup equals a brute-force clockwise
//   scan on a key sample, balance within the documented [0.18, 0.32] band (measured
//   0.220..0.270), adding "node-e" moves exactly 716 keys (0.175 <= 0.30 documented)
//   with every moved key going to "node-e", and removing "node-b" moves exactly its
//   1034 keys (0.252 <= 0.30) with no other key changing owner.
using System;
using System.Collections.Generic;

internal class ConsistentHashRing
{
    public const int Vnodes = 64;
    public const int Modulus = 65537;

    private readonly List<(long Pos, string Node, int Idx)> points = new();

    public static int LcgHash(string text)
    {
        int h = 0;
        foreach (char ch in text) h = (h * 75 + 74 + ch) % Modulus;
        h = (h * 75 + 74) % Modulus;
        h = (h * 75 + 74) % Modulus;
        return h;
    }

    public ConsistentHashRing(string[] nodes)
    {
        foreach (string node in nodes)
            for (int i = 0; i < Vnodes; i++) points.Add((LcgHash(node + "#" + i), node, i));
        points.Sort((a, b) =>
        {
            if (a.Pos != b.Pos) return a.Pos.CompareTo(b.Pos);
            int cmp = string.CompareOrdinal(a.Node, b.Node);
            return cmp != 0 ? cmp : a.Idx.CompareTo(b.Idx);
        });
    }

    public int DistinctPositions()
    {
        var seen = new HashSet<long>();
        foreach (var point in points) seen.Add(point.Pos);
        return seen.Count;
    }

    public string Owner(string key)
    {
        long pos = LcgHash(key);
        int lo = 0, hi = points.Count;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (points[mid].Pos < pos) lo = mid + 1;
            else hi = mid;
        }
        if (lo == points.Count) lo = 0; // wrap around
        return points[lo].Node;
    }

    public string OwnerBrute(string key)
    {
        long pos = LcgHash(key);
        (long Pos, string Node, int Idx)? best = null;
        foreach (var point in points)
        {
            if (point.Pos >= pos && (best == null || Compare(point, best.Value) < 0)) best = point;
        }
        return (best ?? points[0]).Node;
    }

    private static int Compare((long Pos, string Node, int Idx) a, (long Pos, string Node, int Idx) b)
    {
        if (a.Pos != b.Pos) return a.Pos.CompareTo(b.Pos);
        int cmp = string.CompareOrdinal(a.Node, b.Node);
        return cmp != 0 ? cmp : a.Idx.CompareTo(b.Idx);
    }
}

public static class Solution
{
    public static void Main()
    {
        var nodes = new[] { "node-a", "node-b", "node-c", "node-d" };
        var keys = new string[4096];
        for (int i = 0; i < keys.Length; i++) keys[i] = "k" + i;

        var ring = new ConsistentHashRing(nodes);
        var first = Assign(ring, keys);
        var again = Assign(new ConsistentHashRing(nodes), keys);
        Check(again.Count == first.Count, "deterministic size");
        foreach (var entry in first) Check(again[entry.Key] == entry.Value, "deterministic " + entry.Key);
        Check(ring.DistinctPositions() > nodes.Length * ConsistentHashRing.Vnodes / 2, "positions spread");

        for (int i = 0; i < keys.Length; i += 79)
            Check(ring.Owner(keys[i]) == ring.OwnerBrute(keys[i]), "bisect equals brute " + i);

        var shares = new Dictionary<string, int>();
        foreach (string node in nodes) shares[node] = 0;
        foreach (string owner in first.Values) shares[owner]++;
        foreach (string node in nodes)
        {
            double share = shares[node] / (double)keys.Length;
            Check(share >= 0.18 && share <= 0.32, "share " + node + " " + share);
        }

        var afterAdd = Assign(new ConsistentHashRing(new[] { "node-a", "node-b", "node-c", "node-d", "node-e" }), keys);
        var moved = new List<string>();
        foreach (string key in keys) if (afterAdd[key] != first[key]) moved.Add(key);
        Check(moved.Count <= 0.30 * keys.Length, "add movement " + moved.Count);
        foreach (string key in moved) Check(afterAdd[key] == "node-e", "moved to node-e");

        var afterRemove = Assign(new ConsistentHashRing(new[] { "node-a", "node-c", "node-d" }), keys);
        var movedRemove = new List<string>();
        foreach (string key in keys) if (afterRemove[key] != first[key]) movedRemove.Add(key);
        Check(movedRemove.Count <= 0.30 * keys.Length, "remove movement " + movedRemove.Count);
        foreach (string key in keys)
            if (first[key] != "node-b") Check(afterRemove[key] == first[key], "untouched " + key);
        foreach (string key in movedRemove) Check(first[key] == "node-b", "only node-b keys move");

        Check(moved.Count == 716 && movedRemove.Count == 1034, "measured counts");
        Console.WriteLine("PASS");
    }

    private static Dictionary<string, string> Assign(ConsistentHashRing ring, string[] keys)
    {
        var owners = new Dictionary<string, string>();
        foreach (string key in keys) owners[key] = ring.Owner(key);
        return owners;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
