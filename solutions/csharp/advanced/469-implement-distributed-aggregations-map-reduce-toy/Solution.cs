// Task 469 - Implement distributed aggregations map-reduce toy. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Deterministic word-count MapReduce with explicit phases:
//   split   - three input splits of LCG-generated lines over a 40-word vocabulary;
//   map     - each split emits (word, 1) pairs;
//   partition - partition_of(word) = polynomial hash (h = (h*31 + ord(ch)) mod
//             1000003) mod R with R = 4 reducers (documented rule);
//   shuffle - pairs grouped by (partition, word), partitions and words in sorted order;
//   reduce  - each (partition, word) group summed to one (word, count) pair.
//
// Checks: reduced counts equal a plain count of all words; every word's pairs land in
// exactly one partition equal to its hash; per-partition pair totals are conserved;
// every word is reduced exactly once; the pipeline is deterministic; and the
// hand-computed partition of "w00" is 3 (h("w00") = 115895, mod 4 = 3).
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    private const int Reducers = 4;
    private static int seed;

    private static int Lcg() => seed = (seed * 75 + 74) % 65537;

    private static string[][] MakeSplits(int startSeed)
    {
        seed = startSeed;
        var splits = new string[3][];
        for (int s = 0; s < 3; s++)
        {
            var words = new List<string>();
            for (int line = 0; line < 20; line++)
            {
                int length = Lcg() % 8 + 1;
                for (int k = 0; k < length; k++) words.Add("w" + (Lcg() % 40).ToString("D2"));
            }
            splits[s] = words.ToArray();
        }
        return splits;
    }

    private static int PartitionOf(string word)
    {
        int h = 0;
        foreach (char ch in word) h = (h * 31 + ch) % 1000003;
        return h % Reducers;
    }

    private static List<(int Split, string Word)> MapPhase(string[][] splits)
    {
        var pairs = new List<(int, string)>();
        for (int splitId = 0; splitId < splits.Length; splitId++)
            foreach (string word in splits[splitId]) pairs.Add((splitId, word));
        return pairs;
    }

    private static SortedDictionary<int, SortedDictionary<string, List<int>>> ShufflePhase(
        List<(int Split, string Word)> pairs)
    {
        var groups = new SortedDictionary<int, SortedDictionary<string, List<int>>>();
        foreach (var (_, word) in pairs)
        {
            int part = PartitionOf(word);
            if (!groups.TryGetValue(part, out var words))
            {
                words = new SortedDictionary<string, List<int>>();
                groups[part] = words;
            }
            if (!words.TryGetValue(word, out var counts))
            {
                counts = new List<int>();
                words[word] = counts;
            }
            counts.Add(1);
        }
        return groups;
    }

    private static SortedDictionary<int, SortedDictionary<string, int>> ReducePhase(
        SortedDictionary<int, SortedDictionary<string, List<int>>> groups)
    {
        var reduced = new SortedDictionary<int, SortedDictionary<string, int>>();
        foreach (var partEntry in groups)
        {
            var outWords = new SortedDictionary<string, int>();
            foreach (var wordEntry in partEntry.Value) outWords[wordEntry.Key] = wordEntry.Value.Sum();
            reduced[partEntry.Key] = outWords;
        }
        return reduced;
    }

    private static SortedDictionary<int, SortedDictionary<string, int>> RunPipeline(int startSeed) =>
        ReducePhase(ShufflePhase(MapPhase(MakeSplits(startSeed))));

    public static void Main()
    {
        Check(PartitionOf("w00") == 3, "hand partition w00"); // h("w00") = 115895
        Check(PartitionOf("w") == 3, "hand partition w"); // h("w") = 119

        var splits = MakeSplits(20260116);
        var pairs = MapPhase(splits);
        var reduced = RunPipeline(20260116);

        // Plain count oracle over all words.
        var counts = new Dictionary<string, int>();
        foreach (var split in splits)
            foreach (string word in split)
                counts[word] = counts.TryGetValue(word, out int c) ? c + 1 : 1;
        var flat = new Dictionary<string, int>();
        foreach (var part in reduced.Values)
            foreach (var entry in part)
            {
                Check(!flat.ContainsKey(entry.Key), "word reduced once");
                flat[entry.Key] = entry.Value;
            }
        Check(flat.Count == counts.Count, "word count");
        foreach (var entry in counts) Check(flat[entry.Key] == entry.Value, "counter oracle");

        // Routing: every word sits in the partition of its hash.
        foreach (var part in reduced)
            foreach (string word in part.Value.Keys)
                Check(part.Key == PartitionOf(word), "routing");

        // Conservation of pair totals per partition.
        var perPartitionIn = new int[Reducers];
        foreach (var (_, word) in pairs) perPartitionIn[PartitionOf(word)]++;
        var perPartitionOut = new int[Reducers];
        foreach (var part in reduced) perPartitionOut[part.Key] = part.Value.Values.Sum();
        Check(perPartitionIn.SequenceEqual(perPartitionOut), "partition conservation");
        Check(perPartitionIn.Sum() == pairs.Count, "pair total");

        // Determinism.
        var again = RunPipeline(20260116);
        Check(again.Count == reduced.Count, "deterministic partitions");
        foreach (var part in reduced)
            Check(again[part.Key].SequenceEqual(part.Value), "deterministic words");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
