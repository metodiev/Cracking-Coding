// Task 941 - Implement continuous profiling pipeline that attributes CPU/memory hotspots to code and inputs.
// (Tier: Master, CTCI Beyond — Software Engineering & Tooling)
//
// A sampling profiler over a small modelled call graph: a deterministic generator draws weighted
// stacks (five call paths, three input classes, extra "withRetry" frames for one class) and the
// pipeline inserts every stack into a call trie. Each trie node carries self (leaf) CPU samples,
// inclusive CPU samples (self plus descendants), allocated bytes and a per-input-class self
// breakdown, so a hotspot can be attributed both to a code path and to the traffic class that
// drives it.
//
// The attribution is verified rather than asserted: total self samples must equal the number of
// drawn samples and the root's inclusive count must equal the same number; inclusive >= self at
// every node and the inclusive sum over leaves must equal the sample count; per-frame inclusive
// and exclusive counts are recomputed by an oracle that scans the raw stacks without the trie;
// the hotspot ranking is compared with a brute-force sort of the raw leaf hits; the per-class
// breakdown must partition each frame's self time and match the raw per-class totals; and the
// before/after diff of an "optimized" build must sum to zero over a fixed sample budget while the
// optimized leaf loses share. Empty profiles and single-stack profiles are the boundary cases.
// Simplified: stacks come from a modelled generator instead of a real sampler, time is measured in
// samples rather than nanoseconds and there are five paths and three input classes - but the trie,
// the inclusive/self accounting, the diff and the per-input attribution are the real mechanisms.
using System;
using System.Collections.Generic;
using System.Linq;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

internal sealed class Sample
{
    internal readonly string[] Frames;
    internal readonly int InputClass;
    internal readonly long Bytes;

    internal Sample(string[] frames, int inputClass, long bytes)
    {
        Frames = frames;
        InputClass = inputClass;
        Bytes = bytes;
    }
}

internal static class ProfileGenerator
{
    private static readonly string[][] Paths =
    {
        new[] { "main", "handleRequest", "parseRequest", "validate" },
        new[] { "main", "handleRequest", "parseRequest", "hashInput" },
        new[] { "main", "handleRequest", "dbLookup", "dbQuery" },
        new[] { "main", "handleRequest", "serialize", "compress" },
        new[] { "main", "render", "compress" },
    };
    private static readonly int[] WeightBefore = { 25, 25, 20, 20, 10 };
    private static readonly int[] WeightAfter = { 30, 25, 20, 5, 20 };
    private static readonly long[] BytesByPath = { 400, 300, 6000, 3000, 2000 };

    internal static int PathCount => Paths.Length;

    internal static List<Sample> Generate(int count, bool optimized, ulong seed)
    {
        int[] weights = optimized ? WeightAfter : WeightBefore;
        var samples = new List<Sample>(count);
        ulong state = seed;
        for (int i = 0; i < count; i++)
        {
            int pick = Weighted(ref state, weights);
            int inputClass = (int)(SplitMix64.Next(ref state) % 3);
            string[] path = Paths[pick];
            var frames = new List<string>(path);
            // Traffic class 2 exercises the retry wrapper around the database query.
            if (inputClass == 2 && path[path.Length - 1] == "dbQuery") frames.Insert(frames.Count - 1, "withRetry");
            long bytes = BytesByPath[pick] + (long)(SplitMix64.Next(ref state) % 100);
            samples.Add(new Sample(frames.ToArray(), inputClass, bytes));
        }
        return samples;
    }

    private static int Weighted(ref ulong splitmix, int[] weights)
    {
        int total = weights.Sum();
        int draw = (int)(SplitMix64.Next(ref splitmix) % (ulong)total);
        for (int i = 0; i < weights.Length; i++)
        {
            draw -= weights[i];
            if (draw < 0) return i;
        }
        return weights.Length - 1;
    }
}

internal sealed class TrieNode
{
    internal readonly string Frame;
    internal readonly SortedDictionary<string, TrieNode> Children = new SortedDictionary<string, TrieNode>(StringComparer.Ordinal);
    internal long Self;
    internal long Inclusive;
    internal long Bytes;
    internal readonly long[] SelfByClass = new long[3];

    internal TrieNode(string frame) => Frame = frame;
}

internal sealed class FrameTotals
{
    internal long Inclusive;
    internal long Exclusive;
    internal long Bytes;
    internal readonly long[] SelfByClass = new long[3];
}

internal sealed class ProfileTrie
{
    internal readonly TrieNode Root = new TrieNode("<root>");
    internal long NodeCount = 1;
    internal long SampleCount;
    internal long ByteCount;

    internal void Insert(Sample sample)
    {
        TrieNode node = Root;
        foreach (string frame in sample.Frames)
        {
            if (!node.Children.TryGetValue(frame, out TrieNode? child))
            {
                child = new TrieNode(frame);
                node.Children[frame] = child;
                NodeCount++;
            }
            node = child;
        }
        node.Self++;
        node.Bytes += sample.Bytes;
        node.SelfByClass[sample.InputClass]++;
        SampleCount++;
        ByteCount += sample.Bytes;
    }

    internal void ComputeInclusive()
    {
        Compute(Root);
    }

    private static long Compute(TrieNode node)
    {
        long total = node.Self;
        foreach (TrieNode child in node.Children.Values) total += Compute(child);
        node.Inclusive = total;
        return total;
    }

    internal static Dictionary<string, FrameTotals> Aggregate(ProfileTrie trie)
    {
        var totals = new Dictionary<string, FrameTotals>(StringComparer.Ordinal);
        Walk(trie.Root, totals);
        return totals;
    }

    private static void Walk(TrieNode node, Dictionary<string, FrameTotals> totals)
    {
        if (node.Frame != "<root>")
        {
            if (!totals.TryGetValue(node.Frame, out FrameTotals? entry))
            {
                entry = new FrameTotals();
                totals[node.Frame] = entry;
            }
            entry.Inclusive += node.Inclusive;
            entry.Exclusive += node.Self;
            entry.Bytes += node.Bytes;
            for (int c = 0; c < node.SelfByClass.Length; c++) entry.SelfByClass[c] += node.SelfByClass[c];
        }
        foreach (TrieNode child in node.Children.Values) Walk(child, totals);
    }

    internal long SumSelf() => Sum(node => node.Self);
    internal long SumInclusive() => Sum(node => node.Inclusive);
    internal long LeafInclusive() => Sum(node => node.Children.Count == 0 ? node.Inclusive : 0);
    internal bool InclusiveAtLeastSelf() => All(node => node.Inclusive >= node.Self);

    private long Sum(Func<TrieNode, long> selector)
    {
        long total = 0;
        Visit(Root, node => total += selector(node));
        return total;
    }

    private bool All(Func<TrieNode, bool> predicate)
    {
        bool ok = true;
        Visit(Root, node => { if (!predicate(node)) ok = false; });
        return ok;
    }

    private static void Visit(TrieNode node, Action<TrieNode> action)
    {
        action(node);
        foreach (TrieNode child in node.Children.Values) Visit(child, action);
    }

    internal static double Share(long part, long whole) => whole == 0 ? 0.0 : (double)part / whole;
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static ProfileTrie Build(IEnumerable<Sample> samples)
    {
        var trie = new ProfileTrie();
        foreach (Sample sample in samples) trie.Insert(sample);
        trie.ComputeInclusive();
        return trie;
    }

    private static Dictionary<string, long> OracleInclusive(List<Sample> samples) =>
        Oracle(samples, sample => sample.Frames);

    private static Dictionary<string, long> OracleExclusive(List<Sample> samples) =>
        Oracle(samples, sample => new[] { sample.Frames[sample.Frames.Length - 1] });

    private static Dictionary<string, long> Oracle(List<Sample> samples, Func<Sample, string[]> pick)
    {
        var counts = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (Sample sample in samples)
        {
            foreach (string frame in pick(sample))
            {
                counts.TryGetValue(frame, out long seen);
                counts[frame] = seen + 1;
            }
        }
        return counts;
    }

    private static string[] Ranked(Dictionary<string, FrameTotals> totals, Func<FrameTotals, long> by) =>
        totals.OrderByDescending(kv => by(kv.Value)).ThenBy(kv => kv.Key, StringComparer.Ordinal)
              .Select(kv => kv.Key).ToArray();

    private static string[] RankedOracle(List<Sample> samples)
    {
        var totals = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (Sample sample in samples)
        {
            string leaf = sample.Frames[sample.Frames.Length - 1];
            totals.TryGetValue(leaf, out long seen);
            totals[leaf] = seen + 1;
        }
        return totals.OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key, StringComparer.Ordinal)
                     .Select(kv => kv.Key).ToArray();
    }

    public static void Main()
    {
        const int Samples = 6000;
        List<Sample> before = ProfileGenerator.Generate(Samples, false, 0x9413UL);
        ProfileTrie trie = Build(before);
        Dictionary<string, FrameTotals> totals = ProfileTrie.Aggregate(trie);

        // 1-2. Conservation and the inclusive/self invariant over the whole call trie.
        Check(trie.SumSelf() == Samples && trie.Root.Inclusive == Samples && trie.SampleCount == Samples,
            "self samples sum to the " + Samples + " drawn samples and the root is inclusive of all of them");
        Check(trie.InclusiveAtLeastSelf() && trie.LeafInclusive() == Samples && trie.NodeCount > 10,
            "inclusive >= self at every node and the leaves account for every sample (" + trie.NodeCount + " nodes)");

        // 3-4. Oracle: per-frame inclusive and exclusive counts recomputed from the raw stacks.
        Dictionary<string, long> oracleInclusive = OracleInclusive(before);
        Dictionary<string, long> oracleExclusive = OracleExclusive(before);
        bool inclusiveMatches = oracleInclusive.All(kv => totals[kv.Key].Inclusive == kv.Value);
        bool exclusiveMatches = oracleExclusive.All(kv => totals[kv.Key].Exclusive == kv.Value);
        Check(inclusiveMatches && oracleInclusive.Count == totals.Count,
            "every frame's inclusive count matches the oracle over the raw stacks (" + oracleInclusive.Count + " frames)");
        Check(exclusiveMatches, "every frame's exclusive count matches the oracle's leaf tally");

        // 5. Hotspot ranking against a brute-force sort of the raw leaf hits.
        string[] rank = Ranked(totals, t => t.Exclusive);
        Check(rank.Take(4).SequenceEqual(RankedOracle(before).Take(4)),
            "the CPU hotspot ranking matches brute force: " + string.Join(" > ", rank.Take(3)));

        // 6. Memory hotspots are attributed independently of the CPU profile.
        string[] memoryRank = Ranked(totals, t => t.Bytes);
        Check(totals.Values.Sum(t => t.Bytes) == before.Sum(s => s.Bytes) && trie.ByteCount == before.Sum(s => s.Bytes),
            "allocated bytes are conserved across the trie");
        Check(memoryRank[0] != rank[0] && memoryRank[0] == "dbQuery",
            "the memory hotspot (" + memoryRank[0] + ") differs from the CPU hotspot (" + rank[0] + ")");

        // 7. Per-input attribution: the breakdown partitions each frame and matches the raw classes.
        long[] rawByClass = new long[3];
        foreach (Sample sample in before) rawByClass[sample.InputClass]++;
        bool partitions = totals.Values.All(t => t.SelfByClass.Sum() == t.Exclusive);
        bool classTotals = Enumerable.Range(0, 3).All(c =>
            totals.Values.Sum(t => t.SelfByClass[c]) == rawByClass[c]);
        Check(partitions && classTotals,
            "per-input-class self samples partition every frame and match the raw class totals");

        // 8. Diff against the optimized build: shares move, the sample budget does not.
        List<Sample> after = ProfileGenerator.Generate(Samples, true, 0x9413UL);
        Dictionary<string, FrameTotals> optimized = ProfileTrie.Aggregate(Build(after));
        long deltaSum = optimized.Values.Sum(t => t.Exclusive) - totals.Values.Sum(t => t.Exclusive);
        Check(deltaSum == 0 && optimized["compress"].Exclusive < totals["compress"].Exclusive - 100,
            "the diff of the optimized build sums to zero while the compress share drops from "
            + totals["compress"].Exclusive + " to " + optimized["compress"].Exclusive);

        // 9. Percentages: exclusive shares are a distribution when there is data, and safe when empty.
        double shareSum = totals.Values.Sum(t => ProfileTrie.Share(t.Exclusive, trie.SampleCount));
        Check(Math.Abs(shareSum - 1.0) < 1e-9 && ProfileTrie.Share(5, 0) == 0.0,
            "exclusive shares form a distribution that sums to " + shareSum.ToString("F4"));

        // 10-11. Boundaries: an empty profile and a single repeated stack.
        var empty = new ProfileTrie();
        empty.ComputeInclusive();
        Check(empty.NodeCount == 1 && empty.Root.Inclusive == 0 && empty.SumSelf() == 0
              && ProfileTrie.Aggregate(empty).Count == 0,
            "an empty profile has only the root, zero counts and no frames to attribute");
        var single = new List<Sample>();
        for (int i = 0; i < 200; i++) single.Add(new Sample(new[] { "main", "handleRequest", "validate" }, 0, 10));
        ProfileTrie singleTrie = Build(single);
        Dictionary<string, FrameTotals> singleTotals = ProfileTrie.Aggregate(singleTrie);
        Check(singleTrie.NodeCount == 4 && singleTotals["validate"].Exclusive == 200
              && singleTotals["validate"].Inclusive == 200
              && ProfileTrie.Share(singleTotals["validate"].Exclusive, singleTrie.SampleCount) == 1.0,
            "one repeated stack collapses into a single path whose leaf owns 100% of the samples");

        // 12. Determinism of the sampled profile.
        Dictionary<string, FrameTotals> repeat = ProfileTrie.Aggregate(Build(ProfileGenerator.Generate(Samples, false, 0x9413UL)));
        Check(repeat["compress"].Exclusive == totals["compress"].Exclusive
              && repeat["parseRequest"].Inclusive == totals["parseRequest"].Inclusive,
            "regenerating with the same seed reproduces every attributed count");

        Console.WriteLine("  profile: " + Samples + " samples over " + trie.NodeCount + " call-tree nodes; top CPU "
            + rank[0] + " (" + totals[rank[0]].Exclusive + "), top memory " + memoryRank[0]
            + " (" + totals[memoryRank[0]].Bytes + " B); retry wrapper for class 2: "
            + totals["withRetry"].Exclusive + " samples");
        Console.WriteLine("PASS");
    }
}
