// Task 695 - Implement anti-entropy repair scheduling with bandwidth bounds. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
//
// A replica repair simulator with a real Merkle comparison and a bandwidth bounded scheduler.
// Each node holds one replica of a range-partitioned data set (R ranges of RangeBytes each). A
// comparison walks two Merkle hash trees top down: equal subtree hashes prune the whole subtree
// with one comparison, unequal hashes descend, so the leaves reached are exactly the diverging
// ranges at a cost of O(d log R + 1) instead of the O(R) of comparing every range. The comparison
// returns both the differing ranges and the number of hash comparisons, which is what makes the
// "anti-entropy is incremental" claim measurable below.
//
// Repair is scheduled per round under two bounds: a per pair link budget and a per node budget
// (bytes a node may send plus receive per round). Candidate pairs are ordered either greedily by
// number of differing ranges (largest divergence first) or round robin by the most stale pair, and
// a range is moved in pieces while the budgets allow, so a range larger than the link budget still
// finishes after a few rounds instead of stalling forever. Every range carries a version and
// anti-entropy copies the higher version over the lower one, so a stale replica is repaired while
// a newer write propagates - both directions are checked.
//
// Documented simplifications: 64 bit FNV-1a digests in memory (a deployment would use a
// cryptographic tree over the same shape), transfers are instantaneous inside a round, the pair
// set is the complete graph (no topology discovery), and one round is one scheduling epoch. What is
// verified: the Merkle walk finds exactly the differing ranges and matches a leaf by leaf oracle,
// convergence of every replica to the reference and to the newest write, that no per node or per
// pair budget is ever exceeded, byte accounting equal to ranges repaired times range size, that a
// pair which already agrees moves nothing, that an oversized range is finished over several rounds,
// the starvation bound, the greedy ordering property, the analytic round lower bound, and
// determinism.
using System;
using System.Collections.Generic;

internal sealed class MerkleComparison
{
    public readonly List<int> DifferingRanges = new List<int>();
    public long HashComparisons;
}

internal static class MerkleComparer
{
    /// <summary>64 bit FNV-1a; deterministic, dependency free, good enough for divergence detection.</summary>
    public static ulong Hash(byte[] data, int offset, int length)
    {
        ulong hash = 14695981039346656037UL;
        for (int i = 0; i < length; i++)
        {
            hash ^= data[offset + i];
            hash *= 1099511628211UL;
        }
        return hash;
    }

    public static ulong Hash(byte[] data) => Hash(data, 0, data.Length);

    /// <summary>Builds the Merkle tree of a replica bottom up; level 0 holds the range digests.</summary>
    public static List<ulong[]> BuildTree(byte[][] replica)
    {
        var levels = new List<ulong[]>();
        var leaves = new ulong[replica.Length];
        for (int i = 0; i < replica.Length; i++) leaves[i] = Hash(replica[i]);
        levels.Add(leaves);
        while (levels[levels.Count - 1].Length > 1)
        {
            ulong[] below = levels[levels.Count - 1];
            var above = new ulong[(below.Length + 1) / 2];
            for (int i = 0; i < above.Length; i++)
            {
                ulong left = below[2 * i];
                ulong right = 2 * i + 1 < below.Length ? below[2 * i + 1] : 0UL;
                above[i] = (left * 1000003UL) ^ right ^ (left << 7);
            }
            levels.Add(above);
        }
        return levels;
    }

    public static MerkleComparison Compare(List<ulong[]> left, List<ulong[]> right)
    {
        var result = new MerkleComparison();
        Walk(left, right, left.Count - 1, 0, result);
        return result;
    }

    private static void Walk(List<ulong[]> left, List<ulong[]> right, int level, int index, MerkleComparison result)
    {
        result.HashComparisons++;
        if (left[level][index] == right[level][index]) return;      // whole subtree pruned
        if (level == 0)
        {
            result.DifferingRanges.Add(index);
            return;
        }
        Walk(left, right, level - 1, 2 * index, result);
        if (2 * index + 1 < left[level - 1].Length) Walk(left, right, level - 1, 2 * index + 1, result);
    }

    /// <summary>Oracle: compare every range digest directly, no tree at all.</summary>
    public static List<int> LeafByLeaf(byte[][] left, byte[][] right)
    {
        var differing = new List<int>();
        for (int i = 0; i < left.Length; i++) if (Hash(left[i]) != Hash(right[i])) differing.Add(i);
        return differing;
    }
}

internal sealed class AntiEntropyCluster
{
    private readonly int _nodes;
    private readonly int _ranges;
    private readonly int _rangeBytes;
    private readonly byte[][][] _data;
    private readonly long[][] _version;
    private readonly List<List<ulong[]>> _trees = new List<List<ulong[]>>();
    private readonly Dictionary<(int, int), long> _lastRepaired = new Dictionary<(int, int), long>();
    private readonly Dictionary<(int, int), long> _firstServed = new Dictionary<(int, int), long>();
    private readonly Dictionary<(int, int, int), int> _partialProgress = new Dictionary<(int, int, int), int>();

    public readonly int LinkBudget;
    public readonly int NodeBudget;

    public long Rounds;
    public long RangesRepaired;
    public long BytesTransferred;
    public long HashComparisons;
    public long MaxNodeUsage;
    public long MaxLinkUsage;

    public AntiEntropyCluster(int nodes, int ranges, int rangeBytes, int linkBudget, int nodeBudget)
    {
        if (nodes < 2 || ranges < 2 || rangeBytes <= 0) throw new ArgumentException("need >= 2 nodes and >= 2 ranges");
        if (linkBudget <= 0 || nodeBudget <= 0) throw new ArgumentException("bandwidth bounds must be positive");
        if ((ranges & (ranges - 1)) != 0) throw new ArgumentException("the range count must be a power of two");
        _nodes = nodes;
        _ranges = ranges;
        _rangeBytes = rangeBytes;
        LinkBudget = linkBudget;
        NodeBudget = nodeBudget;
        _data = new byte[nodes][][];
        _version = new long[nodes][];
        for (int node = 0; node < nodes; node++)
        {
            _data[node] = new byte[ranges][];
            _version[node] = new long[ranges];
            for (int range = 0; range < ranges; range++)
            {
                _data[node][range] = new byte[rangeBytes];
                for (int b = 0; b < rangeBytes; b++) _data[node][range][b] = (byte)(7 * range + b + 1);
            }
        }
        foreach ((int a, int b) in Pairs())
        {
            _lastRepaired[(a, b)] = 0;
            _firstServed[(a, b)] = -1;
        }
        RefreshTrees();
    }

    public int Nodes => _nodes;
    public int Ranges => _ranges;
    public int RangeBytes => _rangeBytes;

    public IEnumerable<(int, int)> Pairs()
    {
        for (int a = 0; a < _nodes; a++)
            for (int b = a + 1; b < _nodes; b++) yield return (a, b);
    }

    public byte[] RangeOf(int node, int range) => _data[node][range];

    public byte[][] ReplicaOf(int node) => _data[node];

    public long FirstServedRound(int a, int b) => _firstServed[(a, b)];

    public long VersionOf(int node, int range) => _version[node][range];

    /// <summary>A new write: this node wins, so anti-entropy propagates the value to everybody.</summary>
    public void Diverge(int node, int range, byte value)
    {
        if (range < 0 || range >= _ranges) throw new ArgumentOutOfRangeException(nameof(range));
        for (int b = 0; b < _rangeBytes; b++) _data[node][range][b] = (byte)(value + b);
        _version[node][range] = MaxVersion(range) + 1;
        RefreshTrees();
    }

    /// <summary>A stale replica: corrupt bytes and an older version, so repair undoes the damage.</summary>
    public void MakeStale(int node, int range)
    {
        if (range < 0 || range >= _ranges) throw new ArgumentOutOfRangeException(nameof(range));
        for (int b = 0; b < _rangeBytes; b++) _data[node][range][b] ^= 0xA5;
        _version[node][range] = MinVersion(range) - 1;
        RefreshTrees();
    }

    private long MaxVersion(int range)
    {
        long max = long.MinValue;
        for (int node = 0; node < _nodes; node++) max = Math.Max(max, _version[node][range]);
        return max;
    }

    private long MinVersion(int range)
    {
        long min = long.MaxValue;
        for (int node = 0; node < _nodes; node++) min = Math.Min(min, _version[node][range]);
        return min;
    }

    private void RefreshTrees()
    {
        _trees.Clear();
        for (int node = 0; node < _nodes; node++) _trees.Add(MerkleComparer.BuildTree(_data[node]));
    }

    public List<int> DifferingRanges(int a, int b) => MerkleComparer.Compare(_trees[a], _trees[b]).DifferingRanges;

    public long TotalDivergence()
    {
        long total = 0;
        foreach ((int a, int b) in Pairs()) total += DifferingRanges(a, b).Count;
        return total;
    }

    /// <summary>One scheduling round: digest exchange, ordering, then budgeted piecewise transfers.</summary>
    public void RunRound(bool greedy)
    {
        Rounds++;
        var candidates = new List<(int A, int B, List<int> Differing)>();
        foreach ((int a, int b) in Pairs())
        {
            MerkleComparison comparison = MerkleComparer.Compare(_trees[a], _trees[b]);
            HashComparisons += comparison.HashComparisons;
            if (comparison.DifferingRanges.Count > 0) candidates.Add((a, b, comparison.DifferingRanges));
        }

        candidates.Sort((x, y) =>
        {
            if (greedy && x.Differing.Count != y.Differing.Count) return y.Differing.Count.CompareTo(x.Differing.Count);
            int byAge = _lastRepaired[(x.A, x.B)].CompareTo(_lastRepaired[(y.A, y.B)]);
            if (byAge != 0) return byAge;
            if (x.A != y.A) return x.A.CompareTo(y.A);
            return x.B.CompareTo(y.B);
        });

        var nodeUsed = new long[_nodes];
        var linkUsed = new Dictionary<(int, int), long>();
        foreach ((int a, int b, List<int> differing) in candidates)
        {
            long moved = 0;
            foreach (int range in differing)
            {
                long linkFree = LinkBudget - linkUsed.GetValueOrDefault((a, b));
                long nodeFree = Math.Min(NodeBudget - nodeUsed[a], NodeBudget - nodeUsed[b]);
                (int, int, int) key = (a, b, range);
                int progress = _partialProgress.GetValueOrDefault(key);
                long needed = _rangeBytes - progress;
                long canMove = Math.Min(needed, Math.Min(linkFree, nodeFree));
                if (canMove <= 0) continue;                        // this range waits, other pairs may still fit
                int source = _version[a][range] >= _version[b][range] ? a : b;
                int target = source == a ? b : a;
                Array.Copy(_data[source][range], progress, _data[target][range], progress, (int)canMove);
                linkUsed[(a, b)] = linkUsed.GetValueOrDefault((a, b)) + canMove;
                nodeUsed[a] += canMove;
                nodeUsed[b] += canMove;
                moved += canMove;
                BytesTransferred += canMove;
                if (progress + canMove == _rangeBytes)
                {
                    _version[target][range] = _version[source][range];
                    _partialProgress.Remove(key);
                    RangesRepaired++;
                }
                else
                {
                    _partialProgress[key] = (int)(progress + canMove);
                }
            }
            if (moved > 0 && _firstServed[(a, b)] < 0) _firstServed[(a, b)] = Rounds;
            _lastRepaired[(a, b)] = Rounds;
        }

        long peakNode = 0;
        foreach (long used in nodeUsed) peakNode = Math.Max(peakNode, used);
        MaxNodeUsage = Math.Max(MaxNodeUsage, peakNode);
        foreach (var entry in linkUsed) MaxLinkUsage = Math.Max(MaxLinkUsage, entry.Value);
        RefreshTrees();
    }

    public void Run(int rounds, bool greedy)
    {
        for (int i = 0; i < rounds; i++) RunRound(greedy);
    }

    public bool AllAgree(out int worstNode)
    {
        worstNode = -1;
        for (int node = 1; node < _nodes; node++)
        {
            for (int range = 0; range < _ranges; range++)
                for (int b = 0; b < _rangeBytes; b++)
                    if (_data[node][range][b] != _data[0][range][b]) { worstNode = node; return false; }
        }
        return true;
    }

    public bool MatchesReference(byte[][][] reference)
    {
        for (int node = 0; node < _nodes; node++)
            for (int range = 0; range < _ranges; range++)
                for (int b = 0; b < _rangeBytes; b++)
                    if (_data[node][range][b] != reference[node][range][b]) return false;
        return true;
    }

    public byte[][][] Snapshot()
    {
        var copy = new byte[_nodes][][];
        for (int node = 0; node < _nodes; node++)
        {
            copy[node] = new byte[_ranges][];
            for (int range = 0; range < _ranges; range++) copy[node][range] = (byte[])_data[node][range].Clone();
        }
        return copy;
    }
}

internal static class Rng695
{
    private static ulong _state = 0x9E3779B97F4A7C15UL;

    public static void Seed(ulong seed) => _state = seed;

    public static ulong Next()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

public static class Solution
{
    public static void Main()
    {
        // --- The Merkle walk isolates the differing ranges at log cost. ------------------
        var cluster = new AntiEntropyCluster(nodes: 5, ranges: 8, rangeBytes: 64, linkBudget: 256, nodeBudget: 512);
        cluster.MakeStale(1, 3);
        List<int> differing = cluster.DifferingRanges(0, 1);
        Check(differing.Count == 1 && differing[0] == 3, "the Merkle walk isolated the single stale range");

        MerkleComparison oneRange = Measure(cluster, 0, 1);
        Check(oneRange.HashComparisons == 2 * 3 + 1,
            "one differing range costs 2 log2(R) + 1 comparisons, got " + oneRange.HashComparisons);
        MerkleComparison clean = Measure(cluster, 0, 2);
        Check(clean.HashComparisons == 1 && clean.DifferingRanges.Count == 0, "identical replicas cost one root comparison");

        var wide = new AntiEntropyCluster(nodes: 5, ranges: 8, rangeBytes: 64, linkBudget: 256, nodeBudget: 512);
        for (int range = 0; range < 8; range++) wide.MakeStale(1, range);
        MerkleComparison allRanges = Measure(wide, 0, 1);
        Check(allRanges.DifferingRanges.Count == 8, "all eight ranges are reported when the whole replica differs");
        Check(allRanges.HashComparisons == 15, "a full traversal costs 2R - 1 comparisons, got " + allRanges.HashComparisons);
        Check(allRanges.HashComparisons > oneRange.HashComparisons, "a wider divergence costs more comparisons");

        // Cross-check the walk against a leaf by leaf oracle on deterministic damage patterns.
        Rng695.Seed(0x695C0FFEEUL);
        int patterns = 0, oracleMismatches = 0;
        for (int trial = 0; trial < 60; trial++)
        {
            var probe = new AntiEntropyCluster(nodes: 3, ranges: 8, rangeBytes: 32, linkBudget: 128, nodeBudget: 256);
            for (int range = 0; range < 8; range++) if (Rng695.Next() % 3 == 0) probe.MakeStale(1, range);
            List<int> walk = probe.DifferingRanges(0, 1);
            List<int> oracle = MerkleComparer.LeafByLeaf(probe.ReplicaOf(0), probe.ReplicaOf(1));
            patterns++;
            if (string.Join(",", walk) != string.Join(",", oracle)) oracleMismatches++;
        }
        Check(patterns == 60 && oracleMismatches == 0, $"the walk matched the oracle on all {patterns} damage patterns");

        // --- Convergence: stale replicas are repaired, new writes win. ------------------
        var stale = new AntiEntropyCluster(nodes: 5, ranges: 8, rangeBytes: 64, linkBudget: 256, nodeBudget: 512);
        byte[][][] reference = stale.Snapshot();
        for (int range = 0; range < 8; range++) stale.MakeStale(2, range);
        stale.MakeStale(4, 5);
        Check(stale.TotalDivergence() > 0, "the cluster starts divergent, divergence=" + stale.TotalDivergence());
        int roundsToConverge = RoundsToConverge(stale, greedy: true);
        Check(stale.AllAgree(out _), $"all replicas agree after {roundsToConverge} rounds");
        Check(stale.MatchesReference(reference), "the stale copies were repaired back to the reference");
        Check(stale.TotalDivergence() == 0, "no pair disagrees once the repair is complete");

        var winner = new AntiEntropyCluster(nodes: 5, ranges: 8, rangeBytes: 64, linkBudget: 256, nodeBudget: 512);
        winner.Diverge(3, 1, 0x5A);
        RoundsToConverge(winner, greedy: true);
        Check(winner.AllAgree(out _), "a newer write also converges everybody");
        Check(winner.RangeOf(0, 1)[0] == 0x5A, "the newest version won, as anti-entropy promises");

        // --- Bandwidth bounds are never exceeded. ---------------------------------------
        var bounded = new AntiEntropyCluster(nodes: 6, ranges: 8, rangeBytes: 64, linkBudget: 192, nodeBudget: 256);
        for (int range = 0; range < 8; range++) bounded.MakeStale(1, range);
        for (int range = 0; range < 8; range += 2) bounded.MakeStale(3, range);
        RoundsToConverge(bounded, greedy: true);
        Check(bounded.MaxLinkUsage <= bounded.LinkBudget,
            $"no pair ever exceeded the link budget ({bounded.MaxLinkUsage} <= {bounded.LinkBudget})");
        Check(bounded.MaxNodeUsage <= bounded.NodeBudget,
            $"no node ever exceeded its budget ({bounded.MaxNodeUsage} <= {bounded.NodeBudget})");
        Check(bounded.BytesTransferred == bounded.RangesRepaired * bounded.RangeBytes,
            "every byte transferred belongs to exactly one repaired range");
        Check(bounded.AllAgree(out _), "the bounded cluster still converged");
        Check(bounded.HashComparisons > bounded.Rounds, "a digest comparison happened in every round");

        // --- A range bigger than the link budget is finished over several rounds. -------
        var slow = new AntiEntropyCluster(nodes: 2, ranges: 2, rangeBytes: 512, linkBudget: 256, nodeBudget: 1024);
        slow.MakeStale(1, 0);
        Check(!slow.AllAgree(out _), "the large range starts out of sync");
        RoundsToConverge(slow, greedy: true);
        Check(slow.RangesRepaired == 1 && slow.BytesTransferred == 512, "the oversized range was completed exactly once");
        Check(slow.MaxLinkUsage <= slow.LinkBudget, "the slow link respected its budget while waiting");
        Check(slow.Rounds > 1, "one range needed more than a single round, rounds=" + slow.Rounds);

        // --- Anti-entropy is incremental, not a full copy. ------------------------------
        var incremental = new AntiEntropyCluster(nodes: 4, ranges: 8, rangeBytes: 64, linkBudget: 512, nodeBudget: 1024);
        incremental.MakeStale(2, 6);
        incremental.Run(10, greedy: true);
        Check(incremental.BytesTransferred == 3L * 64,
            "only the three pairs that disagreed moved the bad range, bytes=" + incremental.BytesTransferred);
        Check(incremental.BytesTransferred < 8L * 64, "far less than the eight ranges of a whole replica");
        Check(incremental.AllAgree(out _), "the single divergence is healed");

        // --- Fair scheduling serves every divergent pair. -------------------------------
        var starved = new AntiEntropyCluster(nodes: 5, ranges: 4, rangeBytes: 64, linkBudget: 192, nodeBudget: 256);
        for (int node = 1; node < 5; node++) starved.MakeStale(node, node % 4);
        int fairRounds = RoundsToConverge(starved, greedy: false);
        Check(fairRounds <= 10, "the fair scheduler converged inside the pair count, rounds=" + fairRounds);
        for (int node = 1; node < 5; node++)
            Check(starved.FirstServedRound(0, node) >= 1, "pair 0-" + node + " was served at some round");

        // --- Greedy ordering: the widest divergence is served first. --------------------
        var greedy = new AntiEntropyCluster(nodes: 5, ranges: 8, rangeBytes: 64, linkBudget: 128, nodeBudget: 128);
        for (int range = 0; range < 8; range++) greedy.MakeStale(1, range);
        greedy.MakeStale(3, 0);
        greedy.RunRound(greedy: true);
        Check(greedy.FirstServedRound(0, 1) == 1, "the widest divergence was served in the first round");
        Check(greedy.FirstServedRound(0, 1) <= greedy.FirstServedRound(0, 3) || greedy.FirstServedRound(0, 3) < 0,
            "the widest divergence was not served after the narrow one");
        Check(greedy.MaxLinkUsage <= greedy.LinkBudget, "the greedy round stayed inside the budgets");

        // --- The analytic lower bound on rounds. ----------------------------------------
        var bound = new AntiEntropyCluster(nodes: 2, ranges: 8, rangeBytes: 64, linkBudget: 256, nodeBudget: 512);
        for (int range = 0; range < 8; range++) bound.MakeStale(1, range);
        long perRoundCapacity = Math.Min(bound.LinkBudget, bound.NodeBudget);
        long lowerBound = (8L * 64 + perRoundCapacity - 1) / perRoundCapacity;
        int actualRounds = RoundsToConverge(bound, greedy: true);
        Check(actualRounds >= lowerBound, $"no schedule can beat the capacity bound ({actualRounds} >= {lowerBound})");
        Check(actualRounds == lowerBound, $"the greedy schedule meets the bound exactly ({actualRounds} rounds)");
        Check(bound.BytesTransferred == 8L * 64, "the whole divergence moved exactly once");

        // --- Two policies, one shape: both converge and move the same bytes. ------------
        var policyA = new AntiEntropyCluster(nodes: 5, ranges: 8, rangeBytes: 64, linkBudget: 128, nodeBudget: 192);
        var policyB = new AntiEntropyCluster(nodes: 5, ranges: 8, rangeBytes: 64, linkBudget: 128, nodeBudget: 192);
        foreach (var policy in new[] { policyA, policyB })
        {
            for (int range = 0; range < 8; range++) policy.MakeStale(1, range);
            policy.MakeStale(4, 2);
        }
        int greedyRounds = RoundsToConverge(policyA, greedy: true);
        int roundRobinRounds = RoundsToConverge(policyB, greedy: false);
        Check(policyA.AllAgree(out _) && policyB.AllAgree(out _), "both policies converge");
        Check(policyA.BytesTransferred == policyB.BytesTransferred,
            $"both policies move the same total bytes ({policyA.BytesTransferred})");
        Check(greedyRounds > 0 && roundRobinRounds > 0, $"greedy={greedyRounds} roundRobin={roundRobinRounds} rounds");

        // --- Determinism and validation. -------------------------------------------------
        string first = Summary();
        string second = Summary();
        Check(first == second, "reruns are identical: " + first);
        Check(first.Contains("rounds=") && first.Contains("bytes="), "the summary carries the accounting");

        bool rejected = false;
        try { new AntiEntropyCluster(1, 8, 64, 256, 512); } catch (ArgumentException) { rejected = true; }
        Check(rejected, "a single node cluster is rejected");
        rejected = false;
        try { new AntiEntropyCluster(4, 7, 64, 256, 512); } catch (ArgumentException) { rejected = true; }
        Check(rejected, "a non power of two range count is rejected");
        rejected = false;
        try { new AntiEntropyCluster(4, 8, 64, 0, 512); } catch (ArgumentException) { rejected = true; }
        Check(rejected, "a zero bandwidth bound is rejected");
        rejected = false;
        try { cluster.MakeStale(0, 99); } catch (ArgumentOutOfRangeException) { rejected = true; }
        Check(rejected, "damaging a range that does not exist is rejected");

        Console.WriteLine($"merkleOne={oneRange.HashComparisons} merkleAll={allRanges.HashComparisons} " +
                          $"rounds={roundsToConverge} bytes={stale.BytesTransferred} boundRounds={actualRounds} " +
                          $"nodePeak={bounded.MaxNodeUsage} linkPeak={bounded.MaxLinkUsage}");
        Console.WriteLine("PASS");
    }

    private static MerkleComparison Measure(AntiEntropyCluster cluster, int a, int b)
    {
        var left = MerkleComparer.BuildTree(cluster.ReplicaOf(a));
        var right = MerkleComparer.BuildTree(cluster.ReplicaOf(b));
        return MerkleComparer.Compare(left, right);
    }

    private static int RoundsToConverge(AntiEntropyCluster cluster, bool greedy, int limit = 200)
    {
        for (int round = 1; round <= limit; round++)
        {
            cluster.RunRound(greedy);
            if (cluster.AllAgree(out _) && cluster.TotalDivergence() == 0) return round;
        }
        throw new Exception("FAILED: the cluster did not converge within " + limit + " rounds");
    }

    private static string Summary()
    {
        var cluster = new AntiEntropyCluster(nodes: 4, ranges: 8, rangeBytes: 64, linkBudget: 192, nodeBudget: 256);
        for (int range = 0; range < 8; range += 3) cluster.MakeStale(2, range);
        int rounds = RoundsToConverge(cluster, greedy: true);
        return $"rounds={rounds} ranges={cluster.RangesRepaired} bytes={cluster.BytesTransferred} " +
               $"nodePeak={cluster.MaxNodeUsage} linkPeak={cluster.MaxLinkUsage}";
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
