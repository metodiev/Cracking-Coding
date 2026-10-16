// Task 607 - Implement external-memory suffix array construction (disk-based). (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// The suffix array is built by prefix doubling, but every round sorts its records *on disk*:
//   * a record is (key = (rank[i] + 1) << 32 | (rank[i + k] + 1), value = i), so sorting records by
//     key sorts the suffixes by their first 2k characters;
//   * the records of a round are cut into runs of at most runRecords, each run is sorted in a RAM
//     buffer and written out as fixed-size blocks of blockSize records;
//   * while more than one run remains, up to fanIn runs are merged per pass; a merge holds exactly one
//     block of every input run plus one output block, so peak memory is fanIn + 1 blocks regardless
//     of text length and the pass count grows logarithmically in the number of runs.
// Fresh ranks are then computed in RAM (the merged run is read back once) and the loop stops when all
// ranks are distinct - at that point the record order *is* the suffix array.
//
// Simplification (documented): the disk is *simulated in memory*. SimulatedDisk stores blocks as
// arrays, pads each block to blockSize with (long.MaxValue, -1) sentinels and rejects any block whose
// length differs from the block size, so the block-by-block I/O pattern, the run/pass structure and
// the bounded RAM buffer are real and measurable - only the file system is not.
//
// Verified: the suffix array against a naive comparison sort and against an independent in-memory
// doubling implementation for ten fixed texts and random texts up to 512 characters; I/O invariants
// (every block holds exactly blockSize records, one or more blocks per run, reads proportional to
// writes, peak RAM within fanIn + 1 blocks); multiple runs and merge passes really occur at 512
// characters; determinism (a repeated run gives identical suffix arrays and counters); boundary cases
// (empty text, single character, one-symbol text, single-run text, text shorter than a block).
using System;
using System.Collections.Generic;
using System.Text;

// Fixed-size block store with read/write counters; records are kept as two parallel arrays.
internal class SimulatedDisk
{
    private readonly List<long[]> blockKeys = new List<long[]>();
    private readonly List<int[]> blockValues = new List<int[]>();
    public SimulatedDisk(int blockSize) => BlockSize = blockSize;

    public int BlockSize { get; }
    public long Reads { get; private set; } public long Writes { get; private set; }
    public int Blocks => blockKeys.Count;

    public void Write(long[] keys, int[] values)
    {
        if (keys.Length != BlockSize || values.Length != BlockSize)
            throw new Exception("FAILED: a disk block must hold exactly " + BlockSize + " records");
        blockKeys.Add(keys);
        blockValues.Add(values);
        Writes++;
    }

    public void Read(int block, out long[] keys, out int[] values)
    {
        Reads++;
        keys = blockKeys[block];
        values = blockValues[block];
    }
}

// A run is a contiguous range of blocks holding `Records` real records (the tail block is padded).
internal class Run
{
    public int FirstBlock, Blocks, Records;
}

// External merge sort: sorted runs are created in RAM and then merged fanIn at a time via the disk.
internal class ExternalSorter
{
    private readonly SimulatedDisk disk;
    private readonly int runRecords, fanIn;

    public ExternalSorter(SimulatedDisk disk, int runRecords, int fanIn)
    {
        this.disk = disk; this.runRecords = runRecords; this.fanIn = fanIn;
    }

    public int MergePasses { get; private set; }
    public int PeakRamBlocks { get; private set; } public int RunsWritten { get; private set; }
    public void Sort(long[] keys, int[] values, out long[] sortedKeys, out int[] sortedValues)
    {
        int n = keys.Length;
        sortedKeys = Array.Empty<long>();
        sortedValues = Array.Empty<int>();
        if (n == 0) return;
        var runs = new List<Run>();
        for (int start = 0; start < n; start += runRecords)
        {
            int count = Math.Min(runRecords, n - start);
            var bufferKeys = new long[count]; var bufferValues = new int[count];
            Array.Copy(keys, start, bufferKeys, 0, count); Array.Copy(values, start, bufferValues, 0, count);
            Array.Sort(bufferKeys, bufferValues);                 // the RAM-resident part of the sort
            PeakRamBlocks = Math.Max(PeakRamBlocks, BlocksFor(count));
            runs.Add(WriteRun(bufferKeys, bufferValues, count));
        }
        while (runs.Count > 1)
        {
            MergePasses++;
            var merged = new List<Run>();
            for (int i = 0; i < runs.Count; i += fanIn)
            {
                int take = Math.Min(fanIn, runs.Count - i);
                merged.Add(take == 1 ? runs[i] : Merge(runs.GetRange(i, take)));
            }
            runs = merged;
        }
        ReadRun(runs[0], out sortedKeys, out sortedValues);
    }

    private int BlocksFor(int records) => (records + disk.BlockSize - 1) / disk.BlockSize;
    private Run WriteRun(long[] keys, int[] values, int count)
    {
        var run = new Run { FirstBlock = disk.Blocks, Blocks = BlocksFor(count), Records = count };
        for (int block = 0; block < run.Blocks; block++)
        {
            var blockKeys = new long[disk.BlockSize]; var blockValues = new int[disk.BlockSize];
            for (int i = 0; i < disk.BlockSize; i++)
            {
                int index = block * disk.BlockSize + i;
                blockKeys[i] = index < count ? keys[index] : long.MaxValue;
                blockValues[i] = index < count ? values[index] : -1;
            }
            disk.Write(blockKeys, blockValues);
        }
        RunsWritten++;
        return run;
    }

    private void ReadRun(Run run, out long[] keys, out int[] values)
    {
        keys = new long[run.Records]; values = new int[run.Records];
        for (int block = 0; block < run.Blocks; block++)
        {
            disk.Read(run.FirstBlock + block, out long[] blockKeys, out int[] blockValues);
            int offset = block * disk.BlockSize;
            int take = Math.Min(disk.BlockSize, run.Records - offset);
            Array.Copy(blockKeys, 0, keys, offset, take); Array.Copy(blockValues, 0, values, offset, take);
        }
    }

    // k-way merge with one input block per run and one output block: memory stays at fanIn + 1 blocks.
    private Run Merge(List<Run> group)
    {
        var inputKeys = new long[group.Count][];
        var inputValues = new int[group.Count][];
        var position = new int[group.Count];
        var blockIndex = new int[group.Count];
        var remaining = new int[group.Count];
        PeakRamBlocks = Math.Max(PeakRamBlocks, group.Count + 1);
        for (int r = 0; r < group.Count; r++)
        {
            remaining[r] = group[r].Records;
            disk.Read(group[r].FirstBlock, out inputKeys[r], out inputValues[r]);
        }
        var run = new Run { FirstBlock = disk.Blocks };
        var outKeys = new long[disk.BlockSize]; var outValues = new int[disk.BlockSize];
        int outCount = 0, produced = 0;
        while (true)
        {
            int best = -1;
            for (int r = 0; r < group.Count; r++)
                if (remaining[r] > 0 && (best == -1 || inputKeys[r][position[r]] < inputKeys[best][position[best]])) best = r;
            if (best == -1) break;
            int source = position[best];
            outKeys[outCount] = inputKeys[best][source];
            outValues[outCount] = inputValues[best][source];
            outCount++; produced++; remaining[best]--; position[best]++;
            if (remaining[best] > 0 && position[best] == disk.BlockSize)
            {
                disk.Read(group[best].FirstBlock + ++blockIndex[best], out inputKeys[best], out inputValues[best]);
                position[best] = 0;
            }
            if (outCount == disk.BlockSize) Flush(outKeys, outValues, ref outCount, run);
        }
        if (outCount > 0) Flush(outKeys, outValues, ref outCount, run);
        run.Records = produced;
        return run;
    }

    private void Flush(long[] outKeys, int[] outValues, ref int outCount, Run run)
    {
        var blockKeys = new long[disk.BlockSize]; var blockValues = new int[disk.BlockSize];
        Array.Copy(outKeys, blockKeys, disk.BlockSize);
        Array.Copy(outValues, blockValues, disk.BlockSize);
        disk.Write(blockKeys, blockValues);
        run.Blocks++;
        outCount = 0;
    }
}

internal class SimulationStats
{
    public int Rounds, BlocksWritten, MergePasses, PeakRamBlocks, RunsWritten; public long Reads, Writes;
}

internal class ExternalSuffixArray
{
    // Prefix doubling where each round's sort runs through the simulated disk.
    public static int[] Build(string text, int blockSize, int runRecords, int fanIn, out SimulationStats stats)
    {
        int n = text.Length;
        stats = new SimulationStats();
        if (n == 0) return Array.Empty<int>();
        var disk = new SimulatedDisk(blockSize);
        var sorter = new ExternalSorter(disk, runRecords, fanIn);
        var rank = new int[n];
        for (int i = 0; i < n; i++) rank[i] = text[i];
        var keys = new long[n];
        var values = new int[n];
        var order = new int[n];
        for (int k = 1; ; k <<= 1)
        {
            stats.Rounds++;
            for (int i = 0; i < n; i++)
            {
                keys[i] = ((long)(rank[i] + 1) << 32) | (uint)(i + k < n ? rank[i + k] + 1 : 0);
                values[i] = i;
            }
            sorter.Sort(keys, values, out long[] sortedKeys, out int[] sortedValues);
            Array.Copy(sortedValues, order, n);
            var next = new int[n];
            for (int i = 1; i < n; i++) next[order[i]] = next[order[i - 1]] + (sortedKeys[i] != sortedKeys[i - 1] ? 1 : 0);
            rank = next;
            if (rank[order[n - 1]] == n - 1) break;
        }
        stats.BlocksWritten = disk.Blocks;
        stats.Reads = disk.Reads;
        stats.Writes = disk.Writes;
        stats.MergePasses = sorter.MergePasses;
        stats.PeakRamBlocks = sorter.PeakRamBlocks;
        stats.RunsWritten = sorter.RunsWritten;
        return order;
    }
}

public static class Solution
{
    private static ulong seed = 0x1F83D9ABFB41BD6BUL;

    private static ulong NextRandom()
    {
        seed += 0x9E3779B97F4A7C15UL;
        ulong z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static string RandomText(int length, string alphabet)
    {
        var sb = new StringBuilder(length);
        for (int i = 0; i < length; i++) sb.Append(alphabet[(int)(NextRandom() % (ulong)alphabet.Length)]);
        return sb.ToString();
    }

    private static int Compare(string text, int i, int j)
    {
        while (i < text.Length && j < text.Length && text[i] == text[j]) { i++; j++; }
        if (i == text.Length || j == text.Length) return i == j ? 0 : (i == text.Length ? -1 : 1);
        return text[i] < text[j] ? -1 : 1;
    }

    private static int[] Naive(string text)
    {
        var order = new int[text.Length];
        for (int i = 0; i < text.Length; i++) order[i] = i;
        Array.Sort(order, (a, b) => Compare(text, a, b));
        return order;
    }

    // Independent oracle: classic in-memory doubling with comparison sorting.
    private static int[] InMemoryDoubling(string text)
    {
        int n = text.Length;
        if (n == 0) return Array.Empty<int>();
        var order = new int[n]; var rank = new int[n];
        for (int i = 0; i < n; i++) { order[i] = i; rank[i] = text[i]; }
        for (int k = 1; ; k <<= 1)
        {
            int step = k;
            Array.Sort(order, (a, b) =>
            {
                if (rank[a] != rank[b]) return rank[a] - rank[b];
                return (a + step < n ? rank[a + step] : -1) - (b + step < n ? rank[b + step] : -1);
            });
            var next = new int[n];
            for (int i = 1; i < n; i++)
            {
                int p = order[i - 1], q = order[i];
                int pa = p + step < n ? rank[p + step] : -1, qa = q + step < n ? rank[q + step] : -1;
                next[q] = next[p] + (rank[p] != rank[q] || pa != qa ? 1 : 0);
            }
            Array.Copy(next, rank, n);
            if (rank[order[n - 1]] == n - 1) break;
        }
        return order;
    }
    // Correctness against both oracles plus the I/O invariants for one text.
    private static SimulationStats Check(string text, int blockSize, int runRecords, int fanIn)
    {
        int n = text.Length;
        var external = ExternalSuffixArray.Build(text, blockSize, runRecords, fanIn, out SimulationStats stats);
        Check(string.Join(",", external) == string.Join(",", InMemoryDoubling(text)), "doubling agreement \"" + text + "\"");
        if (n > 0 && n <= 200) Check(string.Join(",", external) == string.Join(",", Naive(text)), "naive agreement");
        Check(stats.PeakRamBlocks <= fanIn + 1, "peak RAM bound");
        Check(stats.BlocksWritten == stats.Writes, "block/write accounting");
        if (n == 0) return stats;
        int runsForOneRound = 0;
        for (int records = n; records > 0; records -= runRecords) runsForOneRound++;
        Check(stats.RunsWritten >= runsForOneRound && stats.Writes >= stats.RunsWritten &&
            stats.Reads >= stats.Writes / 2, "run and I/O accounting");
        return stats;
    }

    public static void Main()
    {
        var empty = ExternalSuffixArray.Build("", 4, 4, 2, out SimulationStats emptyStats);
        Check(empty.Length == 0 && emptyStats.Rounds == 0 && emptyStats.Writes == 0, "empty text");
        var single = ExternalSuffixArray.Build("a", 8, 4, 2, out SimulationStats singleStats);
        Check(single.Length == 1 && single[0] == 0 && singleStats.Rounds == 1, "single character");
        foreach (string text in new[] { "a", "ab", "ba", "aaaa", "banana", "mississippi", "abracadabra", "abcabcab" })
            Check(text, 4, 4, 2);
        foreach (string alphabet in new[] { "ab", "abcdefghijklmnopqrstuvwxyz" })
        {
            for (int trial = 0; trial < 5; trial++) Check(RandomText(40 + trial * 7, alphabet), 8, 16, 4);
        }

        // 512 characters need several blocks per run and several merge passes.
        string large = RandomText(512, "abcd");
        var largeStats = Check(large, 16, 32, 4);
        var repeatRun = ExternalSuffixArray.Build(large, 16, 32, 4, out SimulationStats repeatStats);
        var firstRun = ExternalSuffixArray.Build(large, 16, 32, 4, out _);
        Check(string.Join(",", repeatRun) == string.Join(",", firstRun), "determinism");
        Check(repeatStats.Writes == largeStats.Writes && repeatStats.Reads == largeStats.Reads, "stable I/O counters");

        // Run shapes: a 20-character text is one run per round, 200 characters need merges.
        var oneRun = ExternalSuffixArray.Build(RandomText(20, "ab"), 16, 32, 4, out SimulationStats oneRunStats);
        Check(oneRun.Length == 20 && oneRunStats.RunsWritten == oneRunStats.Rounds && oneRunStats.MergePasses == 0, "single run");
        var manyRuns = ExternalSuffixArray.Build(RandomText(200, "ab"), 16, 16, 4, out SimulationStats manyStats);
        Check(manyStats.RunsWritten > manyStats.Rounds && manyStats.MergePasses >= 2, "multiple runs and passes");
        var equal = Check(new string('a', 100), 16, 16, 4);
        Check(equal.PeakRamBlocks == 5, "merge RAM is fanIn + 1 blocks");

        Console.WriteLine("512 chars: " + largeStats.Rounds + " rounds, " + largeStats.RunsWritten + " runs, " +
            largeStats.MergePasses + " merge passes, " + largeStats.Writes + " writes / " + largeStats.Reads +
            " reads, peak RAM " + largeStats.PeakRamBlocks + " blocks (fanIn 4 + 1) of 16 records");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
