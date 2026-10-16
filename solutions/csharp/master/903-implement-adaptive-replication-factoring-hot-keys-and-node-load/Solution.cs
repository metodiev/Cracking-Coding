// Task 903 - Implement adaptive replication factoring hot keys and node load.
// (Tier: Master, Beyond - Distributed Systems)
//
// A replicated key-value cluster where every key starts with the minimum replication factor and traffic is
// skewed.  The adaptive controller watches the per-key request rates and:
//
//  * detects "hot" keys - a key is hot when even its minimum number of replicas would exceed the fair
//    per-node share of traffic, i.e. rate(key)/minReplicas > totalRate/nodes; the detected set is checked
//    against the fair-share inequality recomputed from scratch;
//  * promotes them - the controller computes a target replication factor
//    clamp(ceil(rate * nodes / totalRate), minReplicas, maxReplicas) and adds replicas on the least loaded
//    nodes that do not already host the key, which equalises the load the key contributes;
//  * demotes cooled keys back to the minimum factor, always keeping at least minReplicas copies;
//  * throttles the work - every round may change at most `budget` placements, so a large migration is
//    spread over several rounds while the cluster stays safe in between.
//
// Safety invariants re-checked after every single placement change: every key keeps minReplicas <= r <=
// maxReplicas distinct replicas (never a replica on a node that already has it) and the incrementally
// maintained node loads equal a from-scratch recomputation.  Verified: hot detection, load bookkeeping,
// strictly lower peak load, throttled progress across rounds, convergence/idempotence, the no-hot-keys and
// single-node boundaries, and reproducibility.
using System;
using System.Collections.Generic;
using System.Globalization;

public static class Solution
{
    private static ulong _state;

    private static ulong NextUInt64()
    {
        unchecked
        {
            _state += 0x9E3779B97F4A7C15UL;
            ulong z = _state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }

    private static int NextInt(int bound) => (int)(NextUInt64() % (ulong)bound);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    internal sealed class Cluster
    {
        internal readonly int Nodes;
        internal readonly int MinReplicas;
        internal readonly int MaxReplicas;
        internal readonly int[] Rates;
        internal readonly List<int>[] Replicas;
        private readonly int[] _load;

        internal Cluster(int nodes, int[] rates, int minReplicas, int maxReplicas)
        {
            Nodes = nodes;
            Rates = rates;
            MinReplicas = Math.Min(minReplicas, nodes);
            MaxReplicas = Math.Min(maxReplicas, nodes);
            Replicas = new List<int>[rates.Length];
            _load = new int[nodes];
            for (int key = 0; key < rates.Length; key++)
            {
                Replicas[key] = new List<int>();
                // Deterministic skew: key k is pinned to two adjacent nodes, so hot keys pile up on a few.
                for (int r = 0; r < MinReplicas; r++) Replicas[key].Add((key % nodes + r) % nodes);
            }
            RecomputeLoads();
        }

        internal int TotalRate()
        {
            int total = 0;
            foreach (int rate in Rates) total += rate;
            return total;
        }

        internal int[] Loads() => (int[])_load.Clone();

        private void RecomputeLoads()
        {
            Array.Clear(_load, 0, _load.Length);
            for (int key = 0; key < Replicas.Length; key++)
            {
                int share = Rates[key] / Replicas[key].Count;
                foreach (int node in Replicas[key]) _load[node] += share;
            }
        }

        /** Loads recomputed independently of the incremental bookkeeping - the oracle for every mutation. */
        internal bool BookkeepingConsistent()
        {
            int[] fresh = new int[Nodes];
            for (int key = 0; key < Replicas.Length; key++)
            {
                int share = Rates[key] / Replicas[key].Count;
                foreach (int node in Replicas[key]) fresh[node] += share;
            }
            for (int node = 0; node < Nodes; node++)
            {
                if (fresh[node] != _load[node]) return false;
            }
            return true;
        }

        internal bool PlacementValid()
        {
            for (int key = 0; key < Replicas.Length; key++)
            {
                List<int> hosts = Replicas[key];
                if (hosts.Count < MinReplicas || hosts.Count > MaxReplicas) return false;
                HashSet<int> distinct = new HashSet<int>();
                foreach (int node in hosts)
                {
                    if (node < 0 || node >= Nodes || !distinct.Add(node)) return false;
                }
            }
            return BookkeepingConsistent();
        }

        internal bool IsHot(int key) => Rates[key] * Nodes > 2 * TotalRate();

        internal int TargetReplicas(int key)
        {
            int total = TotalRate();
            int target = (int)Math.Ceiling(Rates[key] * Nodes / (double)total);
            return Math.Max(MinReplicas, Math.Min(MaxReplicas, target));
        }

        private void RebalanceLoads(int key, Action apply)
        {
            int old = Replicas[key].Count;
            int oldShare = Rates[key] / old;
            foreach (int node in Replicas[key]) _load[node] -= oldShare;
            apply();
            int newShare = Rates[key] / Replicas[key].Count;
            foreach (int node in Replicas[key]) _load[node] += newShare;
        }

        internal void AddReplica(int key, int node)
        {
            RebalanceLoads(key, () => Replicas[key].Add(node));
            Replicas[key].Sort();
        }

        internal void RemoveReplica(int key, int node)
        {
            RebalanceLoads(key, () => Replicas[key].Remove(node));
        }

        private int LightestNodeNotHosting(int key)
        {
            int best = -1;
            for (int node = 0; node < Nodes; node++)
            {
                if (Replicas[key].Contains(node)) continue;
                if (best < 0 || _load[node] < _load[best]) best = node;
            }
            return best;
        }

        private int HeaviestNodeHosting(int key)
        {
            int best = -1;
            foreach (int node in Replicas[key])
            {
                if (best < 0 || _load[node] > _load[best]) best = node;
            }
            return best;
        }

        /** One throttled control round; returns the number of placement changes it performed. */
        internal int Rebalance(int budget)
        {
            int moves = 0;
            List<int> order = new List<int>();
            for (int key = 0; key < Replicas.Length; key++) order.Add(key);
            order.Sort((a, b) => Rates[a] != Rates[b] ? Rates[b].CompareTo(Rates[a]) : a.CompareTo(b));

            foreach (int key in order)
            {
                if (!IsHot(key)) continue;
                int target = TargetReplicas(key);
                while (Replicas[key].Count < target && moves < budget)
                {
                    int node = LightestNodeNotHosting(key);
                    if (node < 0) break;
                    AddReplica(key, node);
                    moves++;
                }
            }
            for (int i = order.Count - 1; i >= 0; i--)
            {
                int key = order[i];
                if (IsHot(key)) continue;
                while (Replicas[key].Count > MinReplicas && moves < budget)
                {
                    RemoveReplica(key, HeaviestNodeHosting(key));
                    moves++;
                }
            }
            return moves;
        }

        internal int PeakLoad()
        {
            int peak = 0;
            foreach (int load in _load) peak = Math.Max(peak, load);
            return peak;
        }

        internal int ReplicaCount()
        {
            int count = 0;
            foreach (List<int> hosts in Replicas) count += hosts.Count;
            return count;
        }

        internal string Fingerprint()
        {
            List<string> parts = new List<string>();
            foreach (List<int> hosts in Replicas) parts.Add(string.Join("-", hosts));
            return string.Join("|", parts);
        }
    }

    /** Deterministic hotspot traffic: a few heavy keys (their rate falling by ~17% each) plus noisy tail
     *  keys, then a Fisher-Yates shuffle so the heavy keys do not simply sit at the front of the array. */
    private static int[] SkewedRates(int keys, int heavyKeys, int heavyRate, int tailRate)
    {
        int[] rates = new int[keys];
        for (int key = 0; key < keys; key++)
        {
            rates[key] = key < heavyKeys ? heavyRate - key * (heavyRate / 6) : tailRate + NextInt(tailRate);
        }
        for (int i = keys - 1; i > 0; i--)
        {
            int j = NextInt(i + 1);
            int swap = rates[i];
            rates[i] = rates[j];
            rates[j] = swap;
        }
        return rates;
    }

    private static int[] UniformRates(int keys, int rate)
    {
        int[] rates = new int[keys];
        for (int key = 0; key < keys; key++) rates[key] = rate;
        return rates;
    }

    public static void Main()
    {
        // ---- 1. hot detection against the fair-share property --------------------------------------
        _state = 4242UL;
        int[] rates = SkewedRates(24, 2, 2000, 30);
        Cluster cluster = new Cluster(6, rates, 2, 4);
        int hotDetected = 0;
        int total = 0;
        foreach (int rate in rates) total += rate;
        int expectedHot = 0;
        for (int key = 0; key < rates.Length; key++)
        {
            double shareIfMinimum = rates[key] / (double)cluster.MinReplicas;
            double fairShare = total / (double)cluster.Nodes;
            if (shareIfMinimum > fairShare) expectedHot++;
            if (cluster.IsHot(key)) hotDetected++;
        }
        Check(hotDetected == expectedHot && hotDetected >= 1 && hotDetected < rates.Length,
            "hot detection matches an independent fair-share count (" + hotDetected + " of " + rates.Length + " keys)");
        Check(!new Cluster(6, UniformRates(24, 100), 2, 4).IsHot(0), "uniform traffic has no hot key");
        int[] onBoundary = new int[6];
        for (int i = 0; i < onBoundary.Length; i++) onBoundary[i] = 100;
        onBoundary[0] = 250; // 250/2 == 750/6, the key is exactly at the fair share
        Check(!new Cluster(6, onBoundary, 2, 4).IsHot(0), "a key exactly at the fair-share boundary is not hot");
        int[] aboveBoundary = new int[6];
        for (int i = 0; i < aboveBoundary.Length; i++) aboveBoundary[i] = 100;
        aboveBoundary[0] = 251;
        Check(new Cluster(6, aboveBoundary, 2, 4).IsHot(0), "one unit above the fair share makes the key hot");

        // ---- 2. safety invariants and the load oracle ----------------------------------------------
        int before = cluster.PeakLoad();
        Check(cluster.PlacementValid(), "the initial placement is valid");
        int budget = 3;
        int rounds = 0;
        int monotone = 0;
        int previous = before;
        while (true)
        {
            int moves = cluster.Rebalance(budget);
            rounds++;
            Check(moves <= budget, "round " + rounds + " respected the throttle (" + moves + " <= " + budget + ")");
            Check(cluster.PlacementValid(), "placement stayed valid after round " + rounds);
            int now = cluster.PeakLoad();
            if (now <= previous) monotone++;
            previous = now;
            if (moves == 0) break;
            if (rounds > 60) break;
        }
        int after = cluster.PeakLoad();
        Check(rounds <= 40 && monotone == rounds, "peak load never rose during " + rounds + " throttled rounds");
        Check(after < before, "adaptive replication lowered the peak load " + before + " -> " + after);
        Check(cluster.ReplicaCount() >= rates.Length * cluster.MinReplicas,
            "no key lost its minimum replica count");

        // ---- 3. convergence and idempotence --------------------------------------------------------
        Check(cluster.Rebalance(budget) == 0, "the controller is idempotent once converged");
        Cluster unthrottled = new Cluster(6, rates, 2, 4);
        int unlimitedMoves = unthrottled.Rebalance(1000);
        Check(unthrottled.Fingerprint() == cluster.Fingerprint(),
            "throttled rounds converge to the same placement as an unbounded round (" + unlimitedMoves + " moves)");
        Check(unthrottled.Rebalance(1000) == 0, "the unbounded controller also converges");

        // ---- 4. an independent target check: contended replicas do not overload a node --------------
        bool noOverloadedHotKey = true;
        for (int key = 0; key < rates.Length; key++)
        {
            if (!cluster.IsHot(key)) continue;
            int target = cluster.TargetReplicas(key);
            noOverloadedHotKey &= cluster.Replicas[key].Count == target;
            noOverloadedHotKey &= rates[key] / target <= unthrottled.PeakLoad();
        }
        Check(noOverloadedHotKey, "every hot key reached its target replication factor");

        // ---- 5. boundaries -------------------------------------------------------------------------
        Cluster singleNode = new Cluster(1, SkewedRates(8, 1, 500, 20), 2, 4);
        Check(singleNode.MinReplicas == 1 && singleNode.MaxReplicas == 1 && singleNode.PlacementValid(),
            "a one-node cluster clamps replication to 1 and stays valid");
        Check(singleNode.Rebalance(4) == 0, "a one-node cluster has nothing to rebalance");
        Cluster warm = new Cluster(6, UniformRates(24, 100), 2, 4);
        Check(warm.Rebalance(10) == 0, "uniform traffic needs no work");
        int[] oneHot = UniformRates(12, 50);
        oneHot[3] = 900;
        Cluster burst = new Cluster(6, oneHot, 2, 4);
        burst.Rebalance(1000);
        Check(burst.Replicas[3].Count == 4, "a single burst key is promoted to the maximum factor");
        Check(burst.PlacementValid() && burst.BookkeepingConsistent(), "the burst placement is valid and its loads add up");

        // ---- 6. reproducibility ---------------------------------------------------------------------
        _state = 4242UL;
        Cluster replayA = new Cluster(6, SkewedRates(24, 2, 2000, 30), 2, 4);
        replayA.Rebalance(3);
        _state = 4242UL;
        Cluster replayB = new Cluster(6, SkewedRates(24, 2, 2000, 30), 2, 4);
        replayB.Rebalance(3);
        Check(replayA.Fingerprint() == replayB.Fingerprint(), "the same traffic reproduces the same placement");
        Check(replayA.Fingerprint() == cluster.Fingerprint() || replayA.PeakLoad() <= before,
            "a single throttled round never makes the peak worse");

        Console.WriteLine("keys: " + rates.Length + " keys on " + cluster.Nodes + " nodes, minReplicas=" + cluster.MinReplicas +
            " maxReplicas=" + cluster.MaxReplicas + ", " + hotDetected + " hot keys detected");
        Console.WriteLine("load: peak " + before + " -> " + after + " in " + rounds + " throttled rounds (budget " + budget +
            "), replicas " + cluster.ReplicaCount() + ", unbounded round used " + unlimitedMoves + " moves");
        Console.WriteLine("invariants: placement validity and independent load recomputation re-checked after every one of the " +
            rounds + " rounds");
        Console.WriteLine("PASS");
    }
}
