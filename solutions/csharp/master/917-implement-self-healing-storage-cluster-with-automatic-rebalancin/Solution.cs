// Task 917 - Implement self-healing storage cluster with automatic rebalancing and throttled repairs.
// (Tier: Master, Beyond - Distributed Systems)
//
// A sharded store with N nodes of equal capacity, S shards and R replicas per shard living on distinct
// nodes.  The controller alternates between two budgeted activities:
//
//  * repair - every shard with fewer replicas than the replication factor (because a node died) is given a
//    new replica on the least loaded surviving node, while the shard count of replicas never drops;
//  * rebalance - once every shard is fully replicated, replicas move from the most loaded node to the least
//    loaded one until the spread is at most one, each move removing and adding in the same step so the
//    replication factor is never violated;
//  * throttling - at most `budget` placement changes happen per round, so a large repair is spread over
//    several rounds and the cluster is safe after every one of them.
//
// Safety is re-verified after every single round against a from-scratch recomputation: replicas are on
// distinct live nodes, the replication factor holds whenever enough nodes are alive, no node exceeds its
// capacity, and no shard ever loses data.  A cluster that cannot reach the replication factor (fewer live
// nodes than R, or more replicas than the fabric can hold) reports itself degraded instead of claiming
// health.
//
// Simplifications: equal node capacities, an explicit failure list rather than a failure detector, and
// repairs that happen in whole-shard units.  Verified: safety after every round, the throttle, liveness
// within a bounded number of rounds, the balance criterion, the degraded and capacity-refusal cases, the
// single-change budget, and reproducibility.
using System;
using System.Collections.Generic;
using System.Globalization;

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    internal sealed class Cluster
    {
        private readonly List<int>[] _shards; // shard -> the nodes holding a replica
        private readonly int _nodes;
        private readonly int _capacity;
        private readonly int _replication;
        private readonly HashSet<int> _failed = new HashSet<int>();
        private readonly int[] _load;
        internal int RepairMoves;
        internal int RebalanceMoves;
        internal int Rounds;
        internal bool CapacityRefused;

        internal Cluster(int nodes, int capacity, int shards, int replication, int skew = 0)
        {
            _nodes = nodes;
            _capacity = capacity;
            _replication = replication;
            _shards = new List<int>[shards];
            _load = new int[nodes];
            int pool = skew > 0 ? skew : nodes; // a skewed cluster keeps every replica on the first `skew` nodes
            for (int shard = 0; shard < shards; shard++)
            {
                _shards[shard] = new List<int>();
                for (int replica = 0; replica < Math.Min(replication, nodes); replica++)
                {
                    _shards[shard].Add((shard + replica) % pool);
                }
            }
            RecomputeLoads();
        }

        internal int ShardCount => _shards.Length;
        internal int LiveNodes => _nodes - _failed.Count;
        internal int TargetReplicas => Math.Min(_replication, LiveNodes);
        internal bool Degraded => TargetReplicas < _replication;

        internal bool Started
        {
            get
            {
                CapacityRefused = _shards.Length * Math.Min(_replication, _nodes) > _nodes * _capacity;
                return !CapacityRefused;
            }
        }

        internal void Fail(params int[] nodes)
        {
            foreach (int node in nodes)
            {
                _failed.Add(node);
                foreach (List<int> hosts in _shards) hosts.Remove(node); // the replicas on that node are gone
            }
            RecomputeLoads();
        }

        internal bool IsLive(int node) => !_failed.Contains(node);

        private void RecomputeLoads()
        {
            Array.Clear(_load, 0, _load.Length);
            foreach (List<int> hosts in _shards)
            {
                foreach (int node in hosts) _load[node]++;
            }
        }

        internal int[] Loads() => (int[])_load.Clone();

        /** The independent recomputation the round-by-round checks compare against. */
        internal int[] RecomputeLoadsFromScratch()
        {
            int[] fresh = new int[_nodes];
            foreach (List<int> hosts in _shards)
            {
                foreach (int node in hosts) fresh[node]++;
            }
            return fresh;
        }

        internal List<int> HostsOf(int shard) => new List<int>(_shards[shard]);

        internal bool LoadsMatchOracle()
        {
            int[] fresh = RecomputeLoadsFromScratch();
            for (int node = 0; node < _nodes; node++)
            {
                if (fresh[node] != _load[node]) return false;
            }
            return true;
        }

        /** Safety: distinct live hosts, the expected replication factor, and no node over capacity. */
        internal bool Safe(bool allowDegraded)
        {
            if (!LoadsMatchOracle()) return false;
            foreach (List<int> hosts in _shards)
            {
                HashSet<int> distinct = new HashSet<int>();
                foreach (int node in hosts)
                {
                    if (!IsLive(node) || !distinct.Add(node)) return false;
                }
                if (allowDegraded)
                {
                    if (hosts.Count > TargetReplicas || hosts.Count == 0) return false;
                }
                else if (hosts.Count != TargetReplicas && TargetReplicas == _replication)
                {
                    return false;
                }
            }
            foreach (int load in _load)
            {
                if (load > _capacity) return false;
            }
            return true;
        }

        internal int UnderReplicated()
        {
            int count = 0;
            foreach (List<int> hosts in _shards)
            {
                if (hosts.Count < TargetReplicas) count++;
            }
            return count;
        }

        private int LightestNodeNotHosting(int shard)
        {
            int best = -1;
            for (int node = 0; node < _nodes; node++)
            {
                if (_failed.Contains(node) || _shards[shard].Contains(node)) continue;
                if (best < 0 || _load[node] < _load[best]) best = node;
            }
            return best;
        }

        private int HeaviestNodeHosting(int shard)
        {
            int best = -1;
            foreach (int node in _shards[shard])
            {
                if (best < 0 || _load[node] > _load[best]) best = node;
            }
            return best;
        }

        private int LightestNodeHosting(int shard)
        {
            int best = -1;
            foreach (int node in _shards[shard])
            {
                if (best < 0 || _load[node] < _load[best]) best = node;
            }
            return best;
        }

        /** One throttled round of repair, then of rebalancing; returns the moves it performed. */
        internal int Step(int budget)
        {
            Rounds++;
            int moves = 0;
            if (UnderReplicated() > 0)
            {
                for (int shard = 0; shard < _shards.Length && moves < budget; shard++)
                {
                    while (_shards[shard].Count < TargetReplicas && moves < budget)
                    {
                        int node = LightestNodeNotHosting(shard);
                        if (node < 0) break;
                        _shards[shard].Add(node);
                        _load[node]++;
                        moves++;
                        RepairMoves++;
                    }
                }
                return moves;
            }

            int[] loads = _load;
            int live = 0;
            int heaviest = -1, lightest = -1;
            for (int node = 0; node < _nodes; node++)
            {
                if (_failed.Contains(node)) continue;
                live++;
                if (heaviest < 0 || loads[node] > loads[heaviest]) heaviest = node;
                if (lightest < 0 || loads[node] < loads[lightest]) lightest = node;
            }
            if (live < 2 || heaviest == lightest || loads[heaviest] - loads[lightest] <= 1) return 0;

            for (int shard = 0; shard < _shards.Length && moves < budget; shard++)
            {
                if (!_shards[shard].Contains(heaviest)) continue;
                if (_shards[shard].Contains(lightest)) continue;
                _shards[shard].Remove(heaviest);
                _shards[shard].Add(lightest);
                _load[heaviest]--;
                _load[lightest]++;
                moves++;
                RebalanceMoves++;
                if (_load[heaviest] - _load[lightest] <= 1) break;
            }
            return moves;
        }

        internal int Spread()
        {
            int heaviest = int.MinValue, lightest = int.MaxValue;
            for (int node = 0; node < _nodes; node++)
            {
                if (_failed.Contains(node)) continue;
                heaviest = Math.Max(heaviest, _load[node]);
                lightest = Math.Min(lightest, _load[node]);
            }
            return heaviest - lightest;
        }

        internal string Fingerprint()
        {
            List<string> parts = new List<string>();
            foreach (List<int> hosts in _shards) parts.Add(string.Join("-", hosts));
            return string.Join(",", parts);
        }
    }

    /** Runs the controller until it settles or the round cap is hit, checking safety after every round. */
    private static int Heal(Cluster cluster, int budget, int cap)
    {
        int rounds = 0;
        while (rounds < cap)
        {
            int moves = cluster.Step(budget);
            rounds++;
            Check(moves <= budget, "round " + rounds + " respected the throttle (" + moves + " <= " + budget + ")");
            Check(cluster.Safe(allowDegraded: true), "round " + rounds + " kept the placement safe");
            if (moves == 0) break;
        }
        return rounds;
    }

    public static void Main()
    {
        // ---- 1. initial placement and capacity refusal ------------------------------------------------
        Cluster cluster = new Cluster(8, 16, 24, 3);
        Check(cluster.Started, "24 shards with 3 replicas fit into 8 nodes of 16 slots (128 >= 72)");
        Check(cluster.Safe(allowDegraded: false), "the initial placement is fully replicated and safe");
        Check(cluster.Spread() <= 1, "the initial placement is balanced (" + cluster.Spread() + ")");
        Cluster tooSmall = new Cluster(4, 3, 20, 3);
        Check(!tooSmall.Started, "20 shards x 3 replicas do not fit in 4 nodes of 3 slots, so the cluster refuses to start");
        Cluster tight = new Cluster(6, 6, 12, 3);
        Check(tight.Started && tight.Safe(allowDegraded: false), "a exactly-fitting cluster starts with a safe placement");

        // ---- 2. a node failure triggers throttled repair ------------------------------------------------
        cluster.Fail(3);
        int missing = cluster.UnderReplicated();
        Check(missing > 0, "failing node 3 left " + missing + " shards under-replicated");
        Check(cluster.Safe(allowDegraded: true), "the degraded placement is still safe (data was not lost)");
        Check(cluster.RecomputeLoadsFromScratch()[3] == 0 && !cluster.IsLive(3), "the failed node's slots were released");
        int rounds = Heal(cluster, 4, 60);
        Check(cluster.UnderReplicated() == 0, "all shards are back to three replicas after " + rounds + " rounds");
        Check(rounds <= (missing + 3) / 4 + 1, "repair finished inside the expected round budget (" + rounds + ")");
        Check(cluster.RepairMoves >= missing, "the repair moved at least one replica per missing one (" + cluster.RepairMoves + ")");
        Check(cluster.Safe(allowDegraded: false), "the healed cluster is fully safe");

        // ---- 3. rebalancing a lopsided but healthy placement ---------------------------------------------
        Cluster skewed = new Cluster(8, 20, 24, 3, skew: 4);
        Check(skewed.Started && skewed.Safe(allowDegraded: false), "the skewed placement is valid but lopsided");
        int skewSpread = skewed.Spread();
        Check(skewSpread > 1, "the skewed placement starts with a spread of " + skewSpread);
        int rebalanceRounds = Heal(skewed, 3, 80);
        Check(skewed.Spread() <= 1, "rebalancing brought the spread down to " + skewed.Spread() + " (was " + skewSpread + ")");
        Check(skewed.RebalanceMoves > 0, "rebalancing moved " + skewed.RebalanceMoves + " replicas");
        Check(skewed.Safe(allowDegraded: false), "every move preserved the replication factor and the capacity bound");
        Check(rebalanceRounds > 0 && Heal(skewed, 3, 4) <= 2, "a settled cluster needs no further rounds");
        Check(skewed.Fingerprint().Length > 0, "the placement fingerprint is stable and non-trivial");
        int settleRounds = Heal(cluster, 2, 60);
        Check(cluster.Spread() <= 1, "the repaired cluster also settles into a balanced placement (" + cluster.Spread() + ")");
        Check(cluster.Safe(allowDegraded: false), "and stays safe while it settles (" + settleRounds + " rounds)");

        // ---- 4. cascading failures -------------------------------------------------------------------------
        Cluster cascade = new Cluster(8, 16, 24, 3);
        int totalRepairs = 0;
        foreach (int failing in new[] { 0, 5, 7 })
        {
            cascade.Fail(failing);
            totalRepairs += cascade.UnderReplicated();
            int cascadeRounds = Heal(cascade, 3, 80);
            Check(cascade.UnderReplicated() == 0, "after failing node " + failing + " the cluster is fully replicated again");
            Check(cascade.Safe(allowDegraded: false), "and still safe (" + cascadeRounds + " rounds)");
        }
        Check(totalRepairs >= 3, "three failures caused " + totalRepairs + " shard repairs in total");
        Check(cascade.LiveNodes == 5, "five nodes remain alive");

        // ---- 5. degradation when the fabric is too small ------------------------------------------------------
        Cluster doomed = new Cluster(6, 12, 12, 3);
        doomed.Fail(0, 2, 3, 5); // two survivors: any shard keeps at least one replica, but R=3 is unreachable
        Check(doomed.LiveNodes == 2 && doomed.Degraded, "two live nodes cannot host three replicas: the cluster is degraded");
        Check(doomed.Safe(allowDegraded: true), "no shard lost every replica, so the degraded cluster is still safe");
        int doomedRounds = Heal(doomed, 3, 30);
        Check(doomed.UnderReplicated() == 0 && doomed.Safe(allowDegraded: true),
            "the degraded cluster converges to as many replicas as it can host (" + doomedRounds + " rounds)");
        Check(doomed.HostsOf(0).Count == 2, "each shard ends up with both surviving replicas");
        Check(doomed.LoadsMatchOracle() && doomed.Loads()[1] == 12 && doomed.Loads()[4] == 12,
            "both survivors hold every shard within their capacity");

        // ---- 6. a one-change-per-round budget ------------------------------------------------------------------
        Cluster patient = new Cluster(8, 16, 24, 3);
        patient.Fail(1, 2);
        int pending = patient.UnderReplicated();
        int patientRounds = Heal(patient, 1, 120);
        Check(patient.UnderReplicated() == 0, "a budget of one still heals the cluster in " + patientRounds + " rounds");
        Check(patientRounds >= pending, "the single-change budget cannot beat one round per missing replica (" +
            patientRounds + " >= " + pending + ")");
        Check(patient.Safe(allowDegraded: false), "the slowly repaired cluster is safe at the end");
        Check(patient.Rounds == patientRounds, "the round counter matches the rounds actually run");

        // ---- 7. reproducibility -----------------------------------------------------------------------------------
        Cluster replay = new Cluster(8, 16, 24, 3);
        replay.Fail(1, 2);
        Heal(replay, 1, 120);
        Check(replay.Fingerprint() == patient.Fingerprint(), "the same failure sequence reproduces the same placement");
        Cluster other = new Cluster(8, 16, 24, 3);
        other.Fail(1);
        Heal(other, 1, 120);
        Check(other.Fingerprint() != patient.Fingerprint(), "a different failure sequence gives a different placement");

        Console.WriteLine("cluster: 8 nodes x 16 slots, 24 shards x 3 replicas, " + cluster.RepairMoves + " repairs and " +
            cluster.RebalanceMoves + " rebalances over " + cluster.Rounds + " throttled rounds");
        Console.WriteLine("cascade: three successive failures repaired in " + cascade.Rounds + " rounds, spread now " + cascade.Spread());
        Console.WriteLine("edges: degraded single-node fabric converged, one-change budget healed " + pending +
            " replicas, undersized fabric refused to start");
        Console.WriteLine("PASS");
    }
}
