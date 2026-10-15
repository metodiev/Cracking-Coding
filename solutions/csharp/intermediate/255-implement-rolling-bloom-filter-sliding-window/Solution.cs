// Task 255 - Implement rolling bloom filter sliding window. (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;

public static class Solution
{
    private const int Slices = 4;
    private const int BitsPerSlice = 64;
    private const int Hashes = 3;
    // Deterministic expectation for the "loaded" scenario in Main (cross-language check).
    private const int ExpectedHits = 4;

    private static uint Fnv1a(string text)
    {
        uint h = 2166136261u;
        foreach (char ch in text)
        {
            h ^= ch;
            h *= 16777619u;
        }
        return h;
    }

    private static uint Fmix32(uint x)
    {
        x ^= x >> 16;
        x *= 0x85EBCA6Bu;
        x ^= x >> 13;
        x *= 0xC2B2AE35u;
        x ^= x >> 16;
        return x;
    }

    private static int[] Positions(string item, int bits, int hashes)
    {
        long h1 = Fmix32(Fnv1a(item));
        long h2 = Fmix32(Fnv1a(item + "#")) | 1L;
        var outArr = new int[hashes];
        for (int i = 0; i < hashes; i++) outArr[i] = (int)((h1 + i * h2) % bits);
        return outArr;
    }

    // Every insert writes into ALL slices (OR query); every step clears one slice.
    private sealed class RollingBloom
    {
        private readonly bool[][] _slices;
        private readonly int _bits;
        private readonly int _hashes;
        private int _current;

        public RollingBloom(int sliceCount, int bits, int hashes)
        {
            _slices = new bool[sliceCount][];
            for (int i = 0; i < sliceCount; i++) _slices[i] = new bool[bits];
            _bits = bits;
            _hashes = hashes;
        }

        public void Insert(string item)
        {
            int[] probes = Positions(item, _bits, _hashes);
            foreach (bool[] slice in _slices)
            {
                foreach (int p in probes) slice[p] = true;
            }
        }

        public void Step()
        {
            _current = (_current + 1) % _slices.Length;
            _slices[_current] = new bool[_bits];
        }

        public bool Contains(string item)
        {
            int[] probes = Positions(item, _bits, _hashes);
            foreach (bool[] slice in _slices)
            {
                bool all = true;
                foreach (int p in probes)
                {
                    if (!slice[p]) { all = false; break; }
                }
                if (all) return true;
            }
            return false;
        }

        public int Popcount(int index)
        {
            int count = 0;
            foreach (bool b in _slices[index]) if (b) count++;
            return count;
        }
    }

    public static void Main()
    {
        var rb = new RollingBloom(Slices, BitsPerSlice, Hashes);
        Check(!rb.Contains("alpha"), "empty window: guaranteed miss");

        rb.Insert("alpha");
        Check(rb.Contains("alpha"), "just inserted");
        for (int i = 0; i < Slices; i++) Check(rb.Popcount(i) == Hashes, "3 distinct probes per slice");

        rb.Step();
        Check(rb.Contains("alpha"), "age 1: still inside the window");
        int zeroed = 0;
        for (int i = 0; i < Slices; i++) if (rb.Popcount(i) == 0) zeroed++;
        Check(zeroed == 1, "exactly one slice rotated");

        var ghost = new RollingBloom(Slices, BitsPerSlice, Hashes);
        ghost.Insert("solo");
        for (int i = 0; i < Slices; i++) ghost.Step();
        Check(!ghost.Contains("solo"), "window passed: expired");
        for (int i = 0; i < Slices; i++) Check(ghost.Popcount(i) == 0, "all slices cleared");

        var keep = new RollingBloom(Slices, BitsPerSlice, Hashes);
        keep.Insert("keep");
        for (int i = 0; i < Slices - 1; i++) keep.Step();
        Check(keep.Contains("keep"), "age 3: still inside the window");
        keep.Step();
        Check(!keep.Contains("keep"), "age 4: every slice was cleared");

        var timeline = new RollingBloom(Slices, BitsPerSlice, Hashes);
        var history = new string[9];
        var insertedAt = new int[9];
        int tracked = 0;
        int stepNo = 0;

        history[tracked] = "alpha";
        insertedAt[tracked] = stepNo;
        tracked++;
        timeline.Insert("alpha");
        for (int i = 1; i <= 8; i++)
        {
            timeline.Step();
            stepNo++;
            string item = "w" + i;
            history[tracked] = item;
            insertedAt[tracked] = stepNo;
            tracked++;
            timeline.Insert(item);
            for (int j = 0; j < tracked; j++)                 // never a false negative
            {
                if (stepNo - insertedAt[j] <= Slices - 1)
                {
                    Check(timeline.Contains(history[j]), "in-window miss for " + history[j]);
                }
            }
        }
        Check(!timeline.Contains("alpha"), "8 steps old: fully evicted");

        var loaded = new RollingBloom(Slices, BitsPerSlice, Hashes);
        for (int i = 0; i < 8; i++) loaded.Insert("member-" + i);
        for (int i = 0; i < 8; i++) Check(loaded.Contains("member-" + i), "loaded member " + i);
        int hits = 0;
        for (int i = 0; i < 100; i++) if (loaded.Contains("q" + i)) hits++;
        Check(hits == ExpectedHits, "loaded false positives agree across languages");
        Check(hits >= 1, "false positives do happen: expected, never false negatives");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
