// Task 660 - Implement quorum consensus read/write latency optimization tradeoffs.
// (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
//
// Quorum replication over N replicas with per-replica response latencies. A write is acknowledged by the W
// fastest replicas, a read contacts the R fastest and returns the highest version it sees (then repairs the
// stale replicas it touched). Choosing the R/W fastest replicas makes the quorum latency the maximum of the
// chosen latencies, i.e. the W-th (resp. R-th) order statistic, so the *tradeoff* is explicit: R + W > N is
// required for a read to overlap the most recent write, and among the valid pairs the cheapest is the one
// sitting exactly on the boundary R + W = N + 1.
// What is simplified: no network jitter, retries, timeouts or partial failures are modelled, replicas never
// fail, and "latency" is a fixed per-node response time rather than a distribution - the tradeoff structure,
// which is the point of the task, is intact, and the fastest-quorum choice is verified by exhaustive
// enumeration of all C(N, k) subsets instead of being assumed.
// Verified: the rule "quorums intersect iff R + W > N" is checked against an enumeration of every pair of
// quorums for small clusters, and the minimum intersection is confirmed to be max(0, R + W - N); a concrete
// disjoint read/write witness is built for R + W <= N and shown to serve a stale value, which is exactly the
// failure the constraint prevents; the order-statistic formula for quorum latency is cross-checked against a
// brute-force scan of all subsets; replicated writes followed by quorum reads return the newest version for
// every valid (R, W) with random quorum membership; read repair leaves the contacted replicas up to date;
// latency is non-decreasing in the quorum size; the latency-optimal configuration is found by exhaustive
// search and always lies on the boundary R + W = N + 1; and invalid quorum sizes are rejected.
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

internal sealed class Cluster
{
    internal int Nodes;
    internal double[] Latency = Array.Empty<double>();   // ascending

    internal static Cluster Create(int nodes, double baseLatency, double step)
    {
        var cluster = new Cluster { Nodes = nodes, Latency = new double[nodes] };
        for (int i = 0; i < nodes; i++) cluster.Latency[i] = baseLatency + step * i;
        return cluster;
    }

    internal static Cluster CreateShuffled(int nodes, ulong seed)
    {
        ulong state = seed;
        var cluster = new Cluster { Nodes = nodes, Latency = new double[nodes] };
        for (int i = 0; i < nodes; i++) cluster.Latency[i] = 1 + Rng.NextInt(ref state, 900) / 100.0;
        Array.Sort(cluster.Latency);
        return cluster;
    }

    /// <summary>Fastest `size` replicas, as replica indices ordered by ascending latency.</summary>
    internal List<int> Fastest(int size)
    {
        if (size <= 0 || size > Nodes) throw new ArgumentException("the quorum size must be between 1 and N");
        var chosen = new List<int>();
        for (int i = 0; i < size; i++) chosen.Add(i);
        return chosen;
    }

    internal double WriteLatency(int w) => Latency[w - 1];

    internal double ReadLatency(int r) => Latency[r - 1];

    internal double TotalLatency(int r, int w) => ReadLatency(r) + WriteLatency(w);

    internal static bool QuorumsIntersect(int r, int w, int n) => r + w > n;

    internal static int MinimumIntersection(int r, int w, int n) => Math.Max(0, r + w - n);

    /// <summary>Brute-force scan: the slowest replica inside the best `size`-subset, over all subsets.</summary>
    internal double BestQuorumLatencyByScan(int size)
    {
        double best = double.MaxValue;
        foreach (List<int> subset in Subsets(size))
        {
            double slowest = 0;
            foreach (int replica in subset) slowest = Math.Max(slowest, Latency[replica]);
            best = Math.Min(best, slowest);
        }
        return best;
    }

    internal List<List<int>> Subsets(int size)
    {
        var all = new List<List<int>>();
        int masks = 1 << Nodes;
        for (int mask = 1; mask < masks; mask++)
        {
            int bits = 0;
            for (int i = 0; i < Nodes; i++) if ((mask & (1 << i)) != 0) bits++;
            if (bits != size) continue;
            var subset = new List<int>();
            for (int i = 0; i < Nodes; i++) if ((mask & (1 << i)) != 0) subset.Add(i);
            all.Add(subset);
        }
        return all;
    }

    /// <summary>Exhaustive search over every valid (R, W) for the cheapest read + write round trip.</summary>
    internal (int R, int W, double Total) Best()
    {
        int bestR = 1, bestW = Nodes;
        double best = double.MaxValue;
        for (int r = 1; r <= Nodes; r++)
            for (int w = 1; w <= Nodes; w++)
            {
                if (!QuorumsIntersect(r, w, Nodes)) continue;
                double total = TotalLatency(r, w);
                if (total < best) { best = total; bestR = r; bestW = w; }
            }
        return (bestR, bestW, best);
    }
}

internal sealed class ReplicaStore
{
    internal long[] Version = Array.Empty<long>();
    internal string[] Payload = Array.Empty<string>();

    internal static ReplicaStore Create(int nodes)
    {
        var store = new ReplicaStore { Version = new long[nodes], Payload = new string[nodes] };
        for (int i = 0; i < nodes; i++) store.Payload[i] = "v0";
        return store;
    }

    internal void Write(IReadOnlyList<int> quorum, long version, string payload)
    {
        foreach (int replica in quorum) { Version[replica] = version; Payload[replica] = payload; }
    }

    /// <summary>Returns the newest version seen in the quorum and the number of replicas that answered.</summary>
    internal (string Payload, long Version, int Responders) Read(IReadOnlyList<int> quorum)
    {
        long newest = -1;
        string payload = "none";
        foreach (int replica in quorum)
            if (Version[replica] > newest) { newest = Version[replica]; payload = Payload[replica]; }
        return (payload, newest, quorum.Count);
    }

    /// <summary>Pushes the version the read observed back onto every replica it contacted.</summary>
    internal void Repair(IReadOnlyList<int> quorum, long version, string payload)
    {
        foreach (int replica in quorum)
            if (Version[replica] < version) { Version[replica] = version; Payload[replica] = payload; }
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }

    private static bool Intersects(List<int> a, List<int> b)
    {
        foreach (int value in a) if (b.Contains(value)) return true;
        return false;
    }

    public static void Main()
    {
        // ---- the intersection rule, checked against every pair of quorums -----------------------
        {
            const int N = 5;
            var cluster = Cluster.Create(N, 1, 1);
            bool ruleHolds = true, sizeHolds = true;
            int pairs = 0, disjointWitnesses = 0;
            for (int r = 1; r <= N; r++)
                for (int w = 1; w <= N; w++)
                {
                    bool anyDisjoint = false;
                    int smallestIntersection = int.MaxValue;
                    foreach (List<int> readQuorum in cluster.Subsets(r))
                        foreach (List<int> writeQuorum in cluster.Subsets(w))
                        {
                            int shared = 0;
                            foreach (int replica in readQuorum) if (writeQuorum.Contains(replica)) shared++;
                            smallestIntersection = Math.Min(smallestIntersection, shared);
                            if (shared == 0) anyDisjoint = true;
                            pairs++;
                        }
                    if (anyDisjoint == Cluster.QuorumsIntersect(r, w, N)) ruleHolds = false;
                    if (smallestIntersection != Cluster.MinimumIntersection(r, w, N)) sizeHolds = false;
                    if (!Cluster.QuorumsIntersect(r, w, N)) disjointWitnesses++;
                }
            Check(pairs > 0 && ruleHolds,
                "over all " + pairs + " quorum pairs the read and write sets intersect exactly when R + W > N");
            Check(sizeHolds, "the smallest possible intersection is max(0, R + W - N) for every pair of sizes");
            Check(disjointWitnesses == 10,
                "exactly the ten size combinations with R + W <= N admit a disjoint pair (got "
                + disjointWitnesses + ")");
        }

        // ---- a disjoint quorum really does serve a stale value ---------------------------------
        {
            const int N = 4;
            var cluster = Cluster.Create(N, 1, 1);
            var store = ReplicaStore.Create(N);
            List<int> writeQuorum = cluster.Fastest(2);          // {0, 1}
            var readQuorum = new List<int> { 2, 3 };             // disjoint from the write quorum
            Check(!Intersects(readQuorum, writeQuorum), "the witness read and write quorums are disjoint");
            store.Write(writeQuorum, 1, "first");
            store.Write(writeQuorum, 2, "second");
            var (stalePayload, staleVersion, _) = store.Read(readQuorum);
            Check(stalePayload == "v0" && staleVersion == 0,
                "with R + W <= N a read can miss the newest write entirely (" + stalePayload + ")");

            var overlappingRead = new List<int> { 1, 2 };
            Check(Intersects(overlappingRead, writeQuorum), "an overlapping read quorum intersects the write quorum");
            var (freshPayload, freshVersion, responders) = store.Read(overlappingRead);
            Check(freshPayload == "second" && freshVersion == 2 && responders == 2,
                "the overlapping read returns the newest version and its payload");
            store.Repair(overlappingRead, freshVersion, freshPayload);
            Check(store.Version[2] == 2 && store.Payload[2] == "second",
                "read repair fills in the replica that was behind");
        }

        // ---- every valid (R, W) serves the newest version under random quorum membership --------
        {
            const int N = 6;
            var cluster = Cluster.CreateShuffled(N, 0x3141592653589793UL);
            ulong state = 0x2718281828459045UL;
            bool consistent = true, monotone = true, repairHeld = true;
            string? failure = null;
            for (int r = 1; r <= N; r++)
                for (int w = 1; w <= N; w++)
                {
                    if (!Cluster.QuorumsIntersect(r, w, N)) continue;
                    var store = ReplicaStore.Create(N);
                    for (int step = 1; step <= 40; step++)
                    {
                        List<int> writeQuorum = cluster.Subsets(w)[Rng.NextInt(ref state, cluster.Subsets(w).Count)];
                        store.Write(writeQuorum, step, "value" + step);
                        List<int> readQuorum = cluster.Subsets(r)[Rng.NextInt(ref state, cluster.Subsets(r).Count)];
                        var (payload, version, _) = store.Read(readQuorum);
                        if (version != step || payload != "value" + step)
                        {
                            consistent = false;
                            failure ??= "R=" + r + " W=" + w + " step=" + step + " saw " + version;
                        }
                        store.Repair(readQuorum, version, payload);
                        foreach (int replica in readQuorum)
                            if (store.Version[replica] != step) repairHeld = false;
                    }
                }
            Check(consistent, "every valid (R, W) returns the newest write through random quorums: "
                + (failure ?? "ok"));
            Check(repairHeld, "after a read the contacted replicas all hold the newest version");
            for (int size = 1; size <= N; size++)
                if (cluster.Latency[size - 1] > cluster.Latency[Math.Min(size, N - 1)]) monotone = false;
            Check(monotone && cluster.Latency[0] <= cluster.Latency[N - 1],
                "the sorted latency table is non-decreasing, so a bigger quorum is never faster");
        }

        // ---- quorum latency: order statistic versus a brute-force subset scan -------------------
        {
            const int N = 5;
            var cluster = Cluster.CreateShuffled(N, 0x0F0E0D0C0B0A0908UL);
            bool scanAgrees = true;
            for (int size = 1; size <= N; size++)
            {
                double byScan = cluster.BestQuorumLatencyByScan(size);
                double byFormula = size == 1 ? cluster.Latency[0] : cluster.Latency[size - 1];
                if (Math.Abs(byScan - byFormula) > 1e-9) scanAgrees = false;
            }
            Check(scanAgrees, "the fastest-quorum latency equals the size-th order statistic of the latencies");

            var ladder = Cluster.Create(4, 1, 1);        // latencies 1, 2, 3, 4
            Check(Math.Abs(ladder.WriteLatency(2) - 2) < 1e-9 && Math.Abs(ladder.ReadLatency(2) - 2) < 1e-9,
                "on latencies 1,2,3,4 a two-replica quorum costs 2 in both directions");
            Check(Math.Abs(ladder.TotalLatency(1, 4) - (1 + 4)) < 1e-9
                    && Math.Abs(ladder.TotalLatency(4, 1) - (4 + 1)) < 1e-9,
                "the extreme configurations cost the same here: a fast write with a slow read and vice versa");
            Check(Math.Abs(ladder.TotalLatency(3, 2) - (3 + 2)) < 1e-9,
                "the boundary configuration R=3, W=2 costs 5");

            bool nonDecreasing = true;
            for (int w = 2; w <= 4; w++) if (ladder.WriteLatency(w) < ladder.WriteLatency(w - 1)) nonDecreasing = false;
            Check(nonDecreasing, "write latency never drops as the write quorum grows");
        }

        // ---- the cheapest configuration sits on the R + W = N + 1 boundary ---------------------
        {
            const int N = 7;
            var cluster = Cluster.CreateShuffled(N, 0x1122334455667788UL);
            var (bestR, bestW, bestTotal) = cluster.Best();
            Check(Cluster.QuorumsIntersect(bestR, bestW, N), "the chosen configuration satisfies R + W > N");
            Check(bestR + bestW == N + 1,
                "the cheapest configuration lies exactly on the boundary R + W = N + 1 (R=" + bestR + " W=" + bestW + ")");
            bool unbeaten = true;
            int validPairs = 0;
            for (int r = 1; r <= N; r++)
                for (int w = 1; w <= N; w++)
                {
                    if (!Cluster.QuorumsIntersect(r, w, N)) continue;
                    validPairs++;
                    if (cluster.TotalLatency(r, w) < bestTotal - 1e-9) unbeaten = false;
                }
            Check(unbeaten && validPairs == N * (N + 1) / 2,
                "exhaustive search over the " + validPairs + " valid pairs confirms the choice is optimal");
            Check(Math.Abs(bestTotal - (cluster.Latency[bestR - 1] + cluster.Latency[bestW - 1])) < 1e-9,
                "the reported total is the sum of the two order statistics");
            Check(cluster.Best() == (bestR, bestW, bestTotal), "the search is deterministic");

            var uniform = Cluster.Create(N, 4, 0);
            var (uniformR, uniformW, uniformTotal) = uniform.Best();
            Check(Math.Abs(uniformTotal - 8) < 1e-9 && uniformR + uniformW == N + 1,
                "with identical latencies every boundary pair ties at 8, and the search still stays on it ("
                + uniformR + "," + uniformW + ")");
            Console.WriteLine("latencies=" + string.Join(",", cluster.Latency)
                + " bestR=" + bestR + " bestW=" + bestW + " bestTotal=" + bestTotal.ToString("F2"));
        }

        // ---- rejected configurations -----------------------------------------------------------
        {
            var cluster = Cluster.Create(3, 1, 1);
            int rejected = 0;
            foreach (int bad in new[] { 0, -1, 4 })
            {
                try { cluster.Fastest(bad); }
                catch (ArgumentException) { rejected++; }
            }
            Check(rejected == 3, "quorum sizes of 0, -1 and N + 1 are rejected");
            Check(cluster.Fastest(3).Count == 3 && cluster.Fastest(1)[0] == 0,
                "a full quorum holds every replica and a quorum of one takes the fastest");
            bool rejected2 = false;
            try { _ = cluster.Latency[99]; }
            catch (IndexOutOfRangeException) { rejected2 = true; }
            Check(rejected2, "reading a latency outside the cluster is an error, not a silent zero");
        }

        Console.WriteLine("PASS");
    }
}
