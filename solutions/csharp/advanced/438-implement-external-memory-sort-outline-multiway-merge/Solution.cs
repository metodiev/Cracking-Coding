// Task 438 - Implement external memory sort outline (multiway merge). (Tier: Advanced, CTCI Ch 10)
//
// Simulated external sort with documented block parameters: BLOCK_SIZE = 4 integers per disk block,
// MEMORY_BLOCKS = 4 blocks of RAM (16 integers per initial run) and FAN_IN = 4 (each merge pass merges
// groups of up to 4 sorted runs). Phase 1 loads up to one memory worth of integers, sorts them in RAM
// and writes each initial run back to the simulated disk. Phase 2 keeps one block of every run in
// memory, picks the next output element with a linear min-scan over the at most FAN_IN = 4 buffers (a
// heap would take over for larger fan-in) and streams a single output block; every block read and
// every block written is counted by SimDisk. After the last pass the final run is read back. The
// counted block transfers are asserted against the documented bound 2 * ceil(n/BLOCK_SIZE) *
// (1 + passes) - one initial write pass, two transfers per block per merge pass, one final read pass -
// and the result is checked for equality against a naive in-memory sort (correctness only). The disk
// also keeps every intermediate run, so the accounting assert TotalItems == n * (passes + 1) verifies
// that no element was dropped or duplicated.
using System;
using System.Collections.Generic;

internal class SimDisk
{
    public readonly Dictionary<int, List<List<int>>> Runs = new Dictionary<int, List<List<int>>>();
    public int NextId;
    public int Reads;
    public int Writes;

    public int CreateRun()
    {
        int runId = NextId++;
        Runs[runId] = new List<List<int>>();
        return runId;
    }

    public List<int> ReadBlock(int runId, int index)
    {
        Reads++;
        return Runs[runId][index];
    }

    public void AppendBlock(int runId, List<int> block)
    {
        Writes++;
        Runs[runId].Add(new List<int>(block));
    }

    public int BlockCount(int runId) => Runs[runId].Count;

    public int IoOps() => Reads + Writes;

    public long TotalItems()
    {
        long total = 0;
        foreach (var run in Runs.Values)
        {
            foreach (var block in run) total += block.Count;
        }
        return total;
    }

    public bool BlocksSizedOk()
    {
        foreach (var run in Runs.Values)
        {
            foreach (var block in run)
            {
                if (block.Count < 1 || block.Count > ExternalSort.BlockSize) return false;
            }
        }
        return true;
    }
}

internal class SortResult
{
    public readonly List<int> Data;
    public readonly SimDisk Disk;
    public readonly int Passes;

    public SortResult(List<int> data, SimDisk disk, int passes)
    {
        Data = data;
        Disk = disk;
        Passes = passes;
    }
}

internal static class ExternalSort
{
    public const int BlockSize = 4;
    public const int MemoryBlocks = 4;
    public const int FanIn = 4;

    public static int IoBound(int itemCount, int passes)
    {
        int blocks = (itemCount + BlockSize - 1) / BlockSize;
        return 2 * blocks * (1 + passes);
    }

    public static SortResult Sort(List<int> data)
    {
        var disk = new SimDisk();
        int capacity = MemoryBlocks * BlockSize;
        var runs = new List<int>();
        for (int start = 0; start < data.Count; start += capacity)
        {
            int length = Math.Min(capacity, data.Count - start);
            var chunk = data.GetRange(start, length);
            chunk.Sort();
            int runId = disk.CreateRun();
            for (int index = 0; index < chunk.Count; index += BlockSize)
            {
                disk.AppendBlock(runId, chunk.GetRange(index, Math.Min(BlockSize, chunk.Count - index)));
            }
            runs.Add(runId);
        }
        int passes = 0;
        while (runs.Count > 1)
        {
            passes++;
            var merged = new List<int>();
            for (int start = 0; start < runs.Count; start += FanIn)
            {
                merged.Add(MergeGroup(disk, runs.GetRange(start, Math.Min(FanIn, runs.Count - start))));
            }
            runs = merged;
        }
        var result = new List<int>();
        if (runs.Count > 0)
        {
            int runId = runs[0];
            for (int index = 0; index < disk.BlockCount(runId); index++)
            {
                result.AddRange(disk.ReadBlock(runId, index));
            }
        }
        return new SortResult(result, disk, passes);
    }

    private static int MergeGroup(SimDisk disk, List<int> group)
    {
        int outId = disk.CreateRun();
        int slots = group.Count;
        var blocks = new List<List<int>>();
        foreach (int runId in group) blocks.Add(disk.ReadBlock(runId, 0));
        var positions = new int[slots];
        var indices = new int[slots];
        var outBlock = new List<int>();
        while (true)
        {
            int best = -1;
            for (int slot = 0; slot < slots; slot++)
            {
                if (indices[slot] < blocks[slot].Count)
                {
                    if (best == -1 || blocks[slot][indices[slot]] < blocks[best][indices[best]]) best = slot;
                }
            }
            if (best == -1) break;
            outBlock.Add(blocks[best][indices[best]]);
            indices[best]++;
            if (outBlock.Count == BlockSize)
            {
                disk.AppendBlock(outId, outBlock);
                outBlock = new List<int>();
            }
            if (indices[best] == blocks[best].Count)
            {
                positions[best]++;
                if (positions[best] < disk.BlockCount(group[best]))
                {
                    blocks[best] = disk.ReadBlock(group[best], positions[best]);
                    indices[best] = 0;
                }
            }
        }
        if (outBlock.Count > 0) disk.AppendBlock(outId, outBlock);
        return outId;
    }
}

public static class Solution
{
    public static void Main()
    {
        SortResult result = ExternalSort.Sort(new List<int>());
        Check(result.Data.Count == 0 && result.Passes == 0 && result.Disk.IoOps() == 0, "empty input");

        result = ExternalSort.Sort(new List<int> { 7 });
        Check(result.Data.Count == 1 && result.Data[0] == 7 && result.Passes == 0, "single item");
        Check(result.Disk.Reads == 1 && result.Disk.Writes == 1, "single item io");

        var data = new List<int> { 5, 1, 4, 2, 8, 3, 6, 7, 0, 9 };
        result = ExternalSort.Sort(data);
        Check(string.Join(",", result.Data) == "0,1,2,3,4,5,6,7,8,9", "hand sorted");
        Check(result.Passes == 0, "hand passes");
        Check(result.Disk.IoOps() <= ExternalSort.IoBound(data.Count, result.Passes), "hand io bound");
        Check(result.Disk.TotalItems() == (long)data.Count * (result.Passes + 1), "hand accounting");
        Check(result.Disk.BlocksSizedOk(), "hand blocks");

        var duplicates = new List<int> { -3, 5, 5, 1, 0, -7, 3, 3, 3, -7, 9, 0, -1, 2, 2, 8, -5 };
        result = ExternalSort.Sort(duplicates);
        var sortedDuplicates = new List<int>(duplicates);
        sortedDuplicates.Sort();
        Check(string.Join(",", result.Data) == string.Join(",", sortedDuplicates), "duplicates sorted");
        Check(result.Passes == 1, "duplicates passes " + result.Passes);
        Check(result.Disk.IoOps() <= ExternalSort.IoBound(duplicates.Count, result.Passes), "duplicates io bound");
        Check(result.Disk.TotalItems() == (long)duplicates.Count * (result.Passes + 1), "duplicates accounting");
        Check(result.Disk.BlocksSizedOk(), "duplicates blocks");

        var big = new List<int>();
        int seed = 4380;
        for (int i = 0; i < 2000; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            big.Add(seed % 1000);
        }
        result = ExternalSort.Sort(big);
        var naive = new List<int>(big);
        naive.Sort();
        Check(string.Join(",", result.Data) == string.Join(",", naive), "big sorted");
        Check(result.Passes == 4, "big passes " + result.Passes);
        Check(result.Disk.IoOps() <= ExternalSort.IoBound(big.Count, result.Passes), "big io bound");
        Check(result.Disk.TotalItems() == (long)big.Count * (result.Passes + 1), "big accounting");
        Check(result.Disk.BlocksSizedOk(), "big blocks");
        int maxBlock = 0;
        foreach (var run in result.Disk.Runs.Values)
        {
            foreach (var block in run) maxBlock = Math.Max(maxBlock, block.Count);
        }
        Check(maxBlock == ExternalSort.BlockSize, "full blocks present");

        var reversed = new List<int>();
        for (int i = 0; i < 200; i++) reversed.Add(200 - i);
        result = ExternalSort.Sort(reversed);
        var reversedSorted = new List<int>(reversed);
        reversedSorted.Sort();
        Check(string.Join(",", result.Data) == string.Join(",", reversedSorted), "reversed sorted");
        Check(result.Disk.IoOps() <= ExternalSort.IoBound(reversed.Count, result.Passes), "reversed io bound");
        Check(result.Passes > 0, "reversed passes");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
