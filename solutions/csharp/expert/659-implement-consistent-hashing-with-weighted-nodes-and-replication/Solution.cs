// Task 659 - Implement consistent hashing with weighted nodes and replication.
// (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
//
// A hash ring over the 32-bit space. Every physical node contributes weight * virtualNodesPerWeight virtual
// points (position = hash of "node#index"), the ring is the sorted array of those positions, and a key is
// owned by the successor of hash(key). Replication walks clockwise from the key's position collecting the
// first R *distinct physical* nodes, skipping a node that already appears, so a replica set never contains the
// same machine twice even when that machine owns many virtual points.
// What is simplified: virtual points are hashed with a deterministic FNV-1a/splitmix64 mix instead of MD5
// (the unit of work here is the ring, not the hash function), node weights are small integers, and "load" is
// measured as the number of keys per node rather than bytes stored.
// Verified: an empty ring and a ring whose nodes all weigh zero are rejected; a single node owns everything;
// the ring array stays strictly increasing (no duplicate virtual points) whatever order nodes are added in;
// the primary mapping depends only on the node set, not on insertion order, and is reproducible; the key
// distribution tracks the declared weights, and a node with more weight wins more keys than one with less;
// adding 100 virtual points per weight unit reduces the imbalance relative to one point per unit; replica
// membership rises monotonically with weight but is compressed toward uniform, because a replica set must hold
// distinct machines; a replica set has R distinct physical nodes, R = 1 is the primary, R = the node count is
// the whole cluster, and asking for more replicas than nodes is rejected; removing a node moves *only* the
// keys it owned (every other key keeps its owner) and the fraction moved matches that node's share of the
// ring; and the replica walk is cross-checked against an independently written clockwise scan of the ring.
using System;
using System.Collections.Generic;

internal static class Hash
{
    /// <summary>32-bit FNV-1a over the ASCII bytes of the text.</summary>
    internal static uint Fnv1a(string text)
    {
        uint hash = 2166136261u;
        foreach (char c in text)
        {
            hash ^= (byte)(c & 0xFF);
            hash *= 16777619u;
            hash ^= (byte)(c >> 8);
            hash *= 16777619u;
        }
        return hash;
    }

    /// <summary>Avalanche mixer so the ring points of neighbouring virtual nodes are not adjacent.</summary>
    internal static uint Mix(ulong value)
    {
        ulong z = value + 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return (uint)(z >> 32);
    }

    internal static uint RingPoint(string node, int index) => Mix(Fnv1a(node + "#" + index));
}

internal sealed class ConsistentHashRing
{
    private readonly uint[] _positions;
    private readonly string[] _owners;
    private readonly List<string> _nodes;

    internal ConsistentHashRing(IReadOnlyDictionary<string, int> weights, int virtualPointsPerWeight)
    {
        if (virtualPointsPerWeight <= 0) throw new ArgumentException("virtual points per weight must be positive");
        _nodes = new List<string>(weights.Keys);
        _nodes.Sort(StringComparer.Ordinal);

        var byPosition = new SortedDictionary<uint, string>();
        foreach (string node in _nodes)
        {
            int weight = weights[node];
            if (weight <= 0) throw new ArgumentException("node weights must be positive");
            for (int i = 0; i < weight * virtualPointsPerWeight; i++)
                byPosition[Hash.RingPoint(node, i)] = node;
        }
        if (byPosition.Count == 0)
            throw new ArgumentException("a ring needs at least one node with positive weight");

        _positions = new uint[byPosition.Count];
        _owners = new string[byPosition.Count];
        int at = 0;
        foreach (var pair in byPosition)
        {
            _positions[at] = pair.Key;
            _owners[at] = pair.Value;
            at++;
        }
    }

    internal int VirtualPoints => _positions.Length;

    internal IReadOnlyList<string> Nodes => _nodes;

    internal string[] OwnersSnapshot() => (string[])_owners.Clone();

    internal uint[] PositionsSnapshot() => (uint[])_positions.Clone();

    private int SuccessorIndex(uint point)
    {
        int lo = 0, hi = _positions.Length - 1, found = -1;
        while (lo <= hi)
        {
            int mid = (lo + hi) / 2;
            if (_positions[mid] >= point) { found = mid; hi = mid - 1; }
            else lo = mid + 1;
        }
        return found < 0 ? 0 : found;
    }

    internal string Primary(string key) => _owners[SuccessorIndex(Hash.Fnv1a(key))];

    /// <summary>The first `count` distinct physical nodes walking clockwise from the key's position.</summary>
    internal List<string> Replicas(string key, int count)
    {
        if (count <= 0) throw new ArgumentException("the replica count must be positive");
        if (count > _nodes.Count)
            throw new ArgumentException("cannot place " + count + " replicas on " + _nodes.Count + " nodes");
        var replicas = new List<string>();
        var seen = new HashSet<string>();
        int index = SuccessorIndex(Hash.Fnv1a(key));
        for (int step = 0; step < _positions.Length && replicas.Count < count; step++)
        {
            string owner = _owners[(index + step) % _positions.Length];
            if (seen.Add(owner)) replicas.Add(owner);
        }
        return replicas;
    }

    /// <summary>Independent reference: scan every ring entry clockwise and collect distinct owners.</summary>
    internal List<string> ReplicasByScan(string key, int count)
    {
        var replicas = new List<string>();
        var seen = new HashSet<string>();
        int index = SuccessorIndex(Hash.Fnv1a(key));
        for (int step = 0; step < _positions.Length; step++)
        {
            string owner = _owners[(index + step) % _positions.Length];
            if (!seen.Contains(owner)) { seen.Add(owner); replicas.Add(owner); }
        }
        while (replicas.Count > count) replicas.RemoveAt(replicas.Count - 1);
        return replicas;
    }

    /// <summary>Null when the ring satisfies its structural invariants, otherwise a description.</summary>
    internal string? InvariantViolation()
    {
        if (_positions.Length != _owners.Length) return "positions and owners have different lengths";
        if (_positions.Length == 0) return "the ring is empty";
        for (int i = 1; i < _positions.Length; i++)
            if (_positions[i] <= _positions[i - 1]) return "ring positions are not strictly increasing at " + i;
        var seen = new HashSet<string>();
        foreach (string owner in _owners) seen.Add(owner);
        if (seen.Count != _nodes.Count) return "some node owns no virtual point";
        return null;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }

    private static Dictionary<string, int> Weights(params (string Node, int Weight)[] entries)
    {
        var weights = new Dictionary<string, int>();
        foreach (var entry in entries) weights[entry.Node] = entry.Weight;
        return weights;
    }

    private static List<string> Keys(int count)
    {
        var keys = new List<string>();
        for (int i = 0; i < count; i++) keys.Add("key-" + (i * 7919 % 100000));
        return keys;
    }

    private static Dictionary<string, int> Tally(IEnumerable<string> owners)
    {
        var tally = new Dictionary<string, int>();
        foreach (string owner in owners) tally[owner] = tally.TryGetValue(owner, out int c) ? c + 1 : 1;
        return tally;
    }

    private static double Share(Dictionary<string, int> tally, string node, int total) =>
        tally.TryGetValue(node, out int count) ? (double)count / total : 0;

    private static double MaxDeviation(Dictionary<string, int> tally, Dictionary<string, int> weights, int total)
    {
        int weightSum = 0;
        foreach (int weight in weights.Values) weightSum += weight;
        double worst = 0;
        foreach (var pair in weights)
            worst = Math.Max(worst, Math.Abs(Share(tally, pair.Key, total) - (double)pair.Value / weightSum));
        return worst;
    }

    public static void Main()
    {
        // ---- degenerate rings ------------------------------------------------------------------
        {
            bool rejected = false;
            try { _ = new ConsistentHashRing(Weights(), 10); }
            catch (ArgumentException) { rejected = true; }
            Check(rejected, "a ring with no nodes is rejected");

            rejected = false;
            try { _ = new ConsistentHashRing(Weights(("a", 0), ("b", 0)), 10); }
            catch (ArgumentException) { rejected = true; }
            Check(rejected, "a ring whose nodes all have zero weight is rejected");

            rejected = false;
            try { _ = new ConsistentHashRing(Weights(("a", 1)), 0); }
            catch (ArgumentException) { rejected = true; }
            Check(rejected, "a ring with no virtual points per weight is rejected");

            var single = new ConsistentHashRing(Weights(("only", 5)), 8);
            Check(single.InvariantViolation() == null && single.VirtualPoints == 40,
                "one node with weight five produces forty virtual points");
            bool allSame = true;
            foreach (string key in Keys(200)) if (single.Primary(key) != "only") allSame = false;
            Check(allSame, "a one-node ring owns every key");
            Check(single.Replicas("key-1", 1)[0] == "only", "the only replica of a one-node ring is that node");
            rejected = false;
            try { single.Replicas("key-1", 2); }
            catch (ArgumentException) { rejected = true; }
            Check(rejected, "asking for more replicas than there are nodes is rejected");
        }

        // ---- ring structure and independence from insertion order -------------------------------
        {
            var weights = Weights(("alpha", 2), ("beta", 1), ("gamma", 3), ("delta", 4));
            var ring = new ConsistentHashRing(weights, 40);
            Check(ring.InvariantViolation() == null, "the ring positions are strictly increasing and every node owns points");
            Check(ring.VirtualPoints == 10 * 40, "the ring holds weight * points virtual points in total");

            var reversed = new ConsistentHashRing(Weights(("delta", 4), ("gamma", 3), ("beta", 1), ("alpha", 2)), 40);
            var positions = ring.PositionsSnapshot();
            bool sameRing = positions.Length == reversed.VirtualPoints;
            var otherPositions = reversed.PositionsSnapshot();
            for (int i = 0; i < positions.Length && sameRing; i++)
                if (positions[i] != otherPositions[i]) sameRing = false;
            Check(sameRing, "the ring depends only on the set of nodes, not on the order they were added");

            var rebuilt = new ConsistentHashRing(weights, 40);
            bool reproducible = true;
            foreach (string key in Keys(300))
                if (ring.Primary(key) != rebuilt.Primary(key)) reproducible = false;
            Check(reproducible, "rebuilding the same ring gives the same mapping for 300 keys");
        }

        // ---- the key distribution follows the declared weights ----------------------------------
        {
            var weights = Weights(("alpha", 1), ("beta", 3), ("gamma", 6));
            var keys = Keys(4000);
            var coarse = new ConsistentHashRing(weights, 1);
            var fine = new ConsistentHashRing(weights, 100);
            var coarseTally = Tally(PrimaryOwners(coarse, keys));
            var fineTally = Tally(PrimaryOwners(fine, keys));
            double coarseDeviation = MaxDeviation(coarseTally, weights, keys.Count);
            double fineDeviation = MaxDeviation(fineTally, weights, keys.Count);
            Check(fineDeviation < coarseDeviation,
                "more virtual points per weight unit shrinks the imbalance (" + fineDeviation + " < " + coarseDeviation + ")");
            Check(fineDeviation < 0.03,
                "the primary distribution tracks the weights within three points (" + fineDeviation + ")");
            Check(Share(fineTally, "gamma", keys.Count) > Share(fineTally, "beta", keys.Count)
                    && Share(fineTally, "beta", keys.Count) > Share(fineTally, "alpha", keys.Count),
                "a heavier node owns more keys than a lighter one");

            var replicaTally = new Dictionary<string, int>();
            int placements = 0;
            foreach (string key in keys)
                foreach (string node in fine.Replicas(key, 3))
                {
                    replicaTally[node] = replicaTally.TryGetValue(node, out int c) ? c + 1 : 1;
                    placements++;
                }
            Check(placements == keys.Count * 3, "every key got three replica placements");
            Check(replicaTally.Count == 3,
                "with three nodes every replica set is the whole cluster, so membership is uniform by construction");

            // Replica membership can only be interesting when the cluster is larger than the replication factor.
            var replicaWeights = Weights(("n0", 1), ("n1", 2), ("n2", 3), ("n3", 4), ("n4", 5));
            var wide = new ConsistentHashRing(replicaWeights, 200);
            var wideTally = new Dictionary<string, int>();
            int widePlacements = 0;
            foreach (string key in keys)
                foreach (string node in wide.Replicas(key, 3))
                {
                    wideTally[node] = wideTally.TryGetValue(node, out int c) ? c + 1 : 1;
                    widePlacements++;
                }
            double replicaDeviation = MaxDeviation(wideTally, replicaWeights, widePlacements);
            Check(widePlacements == keys.Count * 3 && wideTally.Count == 5,
                "three replicas over five nodes place 12000 replicas across all five machines");
            // Replica membership is necessarily more even than the weight distribution: a replica set must hold
            // distinct machines, so a very heavy node cannot fill the set by itself. Monotonicity survives.
            bool replicaMonotone = Share(wideTally, "n0", widePlacements) < Share(wideTally, "n1", widePlacements)
                && Share(wideTally, "n1", widePlacements) < Share(wideTally, "n2", widePlacements)
                && Share(wideTally, "n2", widePlacements) < Share(wideTally, "n3", widePlacements)
                && Share(wideTally, "n3", widePlacements) < Share(wideTally, "n4", widePlacements);
            Check(replicaMonotone,
                "replica membership rises monotonically with weight (compressed toward uniform by the "
                + "distinct-machine rule)");
            Check(Share(wideTally, "n4", widePlacements) > 2 * Share(wideTally, "n0", widePlacements),
                "the heaviest node still appears more than twice as often as the lightest ("
                + Share(wideTally, "n4", widePlacements).ToString("F4") + " vs "
                + Share(wideTally, "n0", widePlacements).ToString("F4") + ")");
            Console.WriteLine("coarseDeviation=" + coarseDeviation.ToString("F4") + " fineDeviation="
                + fineDeviation.ToString("F4") + " replicaDeviation=" + replicaDeviation.ToString("F4"));
        }

        // ---- replica sets are distinct machines, cross-checked against a ring scan --------------
        {
            var weights = Weights(("n0", 1), ("n1", 1), ("n2", 1), ("n3", 1), ("n4", 1));
            var ring = new ConsistentHashRing(weights, 16);
            bool distinctHeld = true, crossCheckHeld = true, primaryFirst = true;
            foreach (string key in Keys(400))
            {
                var replicas = ring.Replicas(key, 3);
                if (replicas.Count != 3) distinctHeld = false;
                var unique = new HashSet<string>(replicas);
                if (unique.Count != replicas.Count) distinctHeld = false;
                if (replicas[0] != ring.Primary(key)) primaryFirst = false;
                if (!Same(replicas, ring.ReplicasByScan(key, 3))) crossCheckHeld = false;
            }
            Check(distinctHeld, "three replicas are always three distinct physical machines");
            Check(primaryFirst, "the first replica is the key's primary owner");
            Check(crossCheckHeld, "the replica walk matches an independent clockwise scan of the ring");

            bool fullCluster = true;
            foreach (string key in Keys(50)) if (ring.Replicas(key, 5).Count != 5) fullCluster = false;
            Check(fullCluster, "asking for as many replicas as nodes returns the whole cluster");
            bool rejected = false;
            try { ring.Replicas("key-1", 6); }
            catch (ArgumentException) { rejected = true; }
            Check(rejected, "one replica too many is rejected");
        }

        // ---- adding and removing a node touches only the keys it owns ---------------------------
        {
            var weights = Weights(("n0", 2), ("n1", 2), ("n2", 2), ("n3", 2), ("n4", 2));
            var ring = new ConsistentHashRing(weights, 30);
            var keys = Keys(3000);
            var before = new Dictionary<string, string>();
            foreach (string key in keys) before[key] = ring.Primary(key);

            var removed = new ConsistentHashRing(Weights(("n0", 2), ("n1", 2), ("n3", 2), ("n4", 2)), 30);
            int moved = 0, movedWrongly = 0;
            foreach (string key in keys)
            {
                string now = removed.Primary(key);
                if (now == before[key]) continue;
                moved++;
                if (before[key] != "n2") movedWrongly++;
            }
            double movedFraction = (double)moved / keys.Count;
            Check(movedWrongly == 0,
                "removing n2 moves no key it did not own (minimal disruption)");
            Check(movedFraction > 0.15 && movedFraction < 0.30,
                "the moved fraction tracks the removed node's share of the ring (" + movedFraction.ToString("F4") + ")");

            var added = new ConsistentHashRing(Weights(("n0", 2), ("n1", 2), ("n2", 2), ("n3", 2), ("n4", 2), ("n5", 2)), 30);
            int gained = 0;
            foreach (string key in keys) if (added.Primary(key) != before[key]) gained++;
            double gainedFraction = (double)gained / keys.Count;
            Check(gainedFraction > 0.10 && gainedFraction < 0.25,
                "adding a sixth node to five takes about a sixth of the keys (" + gainedFraction.ToString("F4") + ")");
            Check(added.InvariantViolation() == null && removed.InvariantViolation() == null,
                "both mutated rings still satisfy the structural invariants");
            Console.WriteLine("removedFraction=" + movedFraction.ToString("F4") + " addedFraction="
                + gainedFraction.ToString("F4"));
        }

        // ---- weight changes redistribute proportionally -----------------------------------------
        {
            var keys = Keys(2000);
            var even = new ConsistentHashRing(Weights(("a", 1), ("b", 1)), 400);
            var skewed = new ConsistentHashRing(Weights(("a", 1), ("b", 9)), 400);
            var evenTally = Tally(PrimaryOwners(even, keys));
            var skewedTally = Tally(PrimaryOwners(skewed, keys));
            Check(Math.Abs(Share(evenTally, "a", keys.Count) - 0.5) < 0.03,
                "two equal nodes split the keys evenly");
            Check(Math.Abs(Share(skewedTally, "a", keys.Count) - 0.1) < 0.03,
                "raising b's weight to nine gives it about ninety percent of the keys");
            Check(Share(skewedTally, "b", keys.Count) > Share(evenTally, "b", keys.Count),
                "the heavier ring gives b strictly more keys than the even ring did");
        }

        Console.WriteLine("PASS");
    }

    private static List<string> PrimaryOwners(ConsistentHashRing ring, List<string> keys)
    {
        var owners = new List<string>();
        foreach (string key in keys) owners.Add(ring.Primary(key));
        return owners;
    }

    private static bool Same(List<string> a, List<string> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
        return true;
    }
}
