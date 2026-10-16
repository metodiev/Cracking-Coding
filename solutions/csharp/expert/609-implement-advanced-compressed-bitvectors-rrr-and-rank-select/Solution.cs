// Task 609 - Implement advanced compressed bitvectors (RRR) and rank/select. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// RRR (Raman-Raman-Rao 2002) compressed bitvector. The vector is cut into blocks of b = 15 bits
// and every block is described by its *class* (c, o): c = number of 1-bits, o = number of leading
// 0-bits (o = b for the all-zero block). A class (c, o) contains exactly C(b - o - 1, c - 1)
// distinct blocks, so the pattern needs only ceil(log2 C(b-o-1, c-1)) <= 12 bits instead of the 15
// raw bits and is recovered from a decoder table built once by enumerating all 2^15 words and
// bucketing them by class in increasing numeric order; a block is stored as (class code,
// in-class pointer). Padding past `length` is stored as zeros and masked out of the zero counts.
//
// Rank/select ride on O(1) block decoding: `sample` keeps the 1-count at every 8th block boundary,
// so rank1 is one sample lookup plus at most 8 block popcounts; select binary-searches the samples
// (on the requested bit kind, so 0-selects work too) and then walks at most SampleBlocks blocks.
// Access decodes the single block holding the bit.
//
// Documented simplification: the class code of each block is stored verbatim (8 bits), so this is
// not the asymptotic n*H0 + o(n) space bound. What is entropy-bounded - and asserted below - is
// the payload (the in-class pointer, <= 12 bits per block), plus total encoded size below the raw
// n bits for low-density vectors. Class decomposition, the enumeration decoder and the rank/select
// index are the real RRR machinery, verified bit-for-bit against brute force, including every one
// of the 32 five-bit vectors.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;         // deterministic stand-in for Random
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

internal sealed class RrrBitVector
{
    internal const int BlockBits = 15;
    internal const int SampleBlocks = 8;
    internal const int ClassStride = 16;                 // class code = c * ClassStride + o
    internal const int ClassCodeBits = 8;

    private static readonly int[][] ClassPatterns = BuildClassTable();
    private static readonly int[] ClassWidths = BuildWidths();

    private readonly int length;
    private readonly int blockCount;
    private readonly int totalOnes;
    private readonly int[] blockClass;
    private readonly int[] blockPointer;
    private readonly int[] blockPop;
    private readonly int[] sample;                       // ones before block 0, 8, 16, ... plus total

    internal RrrBitVector(IReadOnlyList<int> bits)
    {
        length = bits.Count;
        blockCount = (length + BlockBits - 1) / BlockBits;
        blockClass = new int[blockCount];
        blockPointer = new int[blockCount];
        blockPop = new int[blockCount];
        int ones = 0;
        for (int k = 0; k < blockCount; k++)
        {
            int start = k * BlockBits;
            int stop = Math.Min(length, start + BlockBits);
            int word = 0;
            for (int i = start; i < stop; i++)
            {
                if (bits[i] == 1) word |= 1 << (i - start);
            }
            int cls = ClassId(word);
            blockClass[k] = cls;
            blockPop[k] = PopCount(word);
            ones += blockPop[k];
            int lo = 0;                                  // in-class pointer: patterns are sorted
            int hi = ClassPatterns[cls].Length - 1;
            while (lo < hi)
            {
                int mid = (lo + hi) / 2;
                if (ClassPatterns[cls][mid] < word) lo = mid + 1;
                else hi = mid;
            }
            blockPointer[k] = lo;
        }
        totalOnes = ones;
        var samples = new List<int>();
        int running = 0;
        for (int k = 0; k < blockCount; k++)
        {
            if (k % SampleBlocks == 0) samples.Add(running);
            running += blockPop[k];
        }
        samples.Add(running);
        sample = samples.ToArray();
    }

    internal int Length => length;
    internal int Ones => totalOnes;
    internal int Zeros => length - totalOnes;
    internal int BlockCount => blockCount;

    internal int Access(int index)
    {
        if (index < 0 || index >= length) return -1;
        return (DecodeBlock(index / BlockBits) >> (index % BlockBits)) & 1;
    }

    /** Number of 1-bits in [0, index). */
    internal int Rank1(int index)
    {
        if (index <= 0) return 0;
        if (index >= length) return totalOnes;
        int block = index / BlockBits;
        int group = block / SampleBlocks;
        int rank = sample[group];
        for (int j = group * SampleBlocks; j < block; j++) rank += blockPop[j];
        int within = index % BlockBits;
        if (within > 0) rank += PopCount(DecodeBlock(block) & ((1 << within) - 1));
        return rank;
    }

    internal int Rank0(int index) => index - Rank1(index);

    internal int Select1(int rank) => Select(rank, true);
    internal int Select0(int rank) => Select(rank, false);

    /** 0-based position of the rank-th bit of the requested kind, or -1 when out of range. */
    private int Select(int rank, bool one)
    {
        if (rank < 1) return -1;
        int total = one ? totalOnes : length - totalOnes;
        if (rank > total) return -1;
        int lo = 0;
        int hi = sample.Length - 2;
        int group = 0;
        while (lo <= hi)
        {
            int mid = (lo + hi) / 2;
            if (CountBeforeGroup(mid, one) < rank)
            {
                group = mid;
                lo = mid + 1;
            }
            else hi = mid - 1;
        }
        int remaining = rank - CountBeforeGroup(group, one);
        for (int k = group * SampleBlocks; k < blockCount; k++)
        {
            int count = one ? blockPop[k] : ValidBits(k) - blockPop[k];
            if (remaining <= count) return SelectInBlock(k, remaining, one);
            remaining -= count;
        }
        return -1;
    }

    /** Bits of the requested kind before the first block of sample group g. */
    private int CountBeforeGroup(int group, bool one)
    {
        int blockStart = group * SampleBlocks;
        return one ? sample[group] : blockStart * BlockBits - sample[group];
    }

    private int SelectInBlock(int block, int rank, bool one)
    {
        int word = DecodeBlock(block);
        int seen = 0;
        for (int i = 0; i < ValidBits(block); i++)
        {
            if (((word >> i) & 1) == (one ? 1 : 0) && ++seen == rank) return block * BlockBits + i;
        }
        return -1;
    }

    private int ValidBits(int block) => Math.Min(BlockBits, length - block * BlockBits);
    private int DecodeBlock(int block) => ClassPatterns[blockClass[block]][blockPointer[block]];

    internal int PayloadBits()
    {
        int bits = 0;
        for (int k = 0; k < blockCount; k++) bits += ClassWidths[blockClass[k]];
        return bits;
    }

    internal int TotalEncodedBits() =>
        PayloadBits() + blockCount * ClassCodeBits + sample.Length * 32;

    internal int MaxPayloadBits()
    {
        int max = 0;
        for (int k = 0; k < blockCount; k++) max = Math.Max(max, ClassWidths[blockClass[k]]);
        return max;
    }

    internal static int MaxClassWidth()
    {
        int max = 0;
        foreach (int w in ClassWidths) max = Math.Max(max, w);
        return max;
    }

    internal static int ClassSize(int cls) => ClassPatterns[cls].Length;

    internal static int ClassId(int word)
    {
        int ones = PopCount(word);
        if (ones == 0) return BlockBits;
        int leading = 0;
        int probe = BlockBits - 1;
        while (((word >> probe) & 1) == 0)
        {
            leading++;
            probe--;
        }
        return ones * ClassStride + leading;
    }

    internal static int PopCount(int value)
    {
        int x = value - ((value >> 1) & 0x55555555);
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
        x = (x + (x >> 4)) & 0x0F0F0F0F;
        return (x * 0x01010101) >> 24;
    }

    private static int[][] BuildClassTable()
    {
        var buckets = new List<int>[ClassStride * (BlockBits + 1)];
        for (int i = 0; i < buckets.Length; i++) buckets[i] = new List<int>();
        for (int word = 0; word < (1 << BlockBits); word++) buckets[ClassId(word)].Add(word);
        var table = new int[buckets.Length][];
        for (int i = 0; i < buckets.Length; i++) table[i] = buckets[i].ToArray();
        return table;
    }

    private static int[] BuildWidths()
    {
        var widths = new int[ClassPatterns.Length];
        for (int i = 0; i < widths.Length; i++)
        {
            int w = 0;
            while ((1 << w) < ClassPatterns[i].Length) w++;
            widths[i] = w;
        }
        return widths;
    }

    internal string Describe() => "n=" + length + " ones=" + totalOnes + " blocks=" + blockCount
        + " payloadBits<=" + MaxPayloadBits() + " encodedBits=" + TotalEncodedBits();
}

public static class Solution
{
    private static ulong state = 0xC0FFEE1234567UL;

    private static int[] Bits(string text)
    {
        var bits = new int[text.Length];
        for (int i = 0; i < text.Length; i++) bits[i] = text[i] == '1' ? 1 : 0;
        return bits;
    }

    private static int[] Fill(int n, int value)
    {
        var bits = new int[n];
        for (int i = 0; i < n; i++) bits[i] = value;
        return bits;
    }

    private static int[] RandomBits(int length, int numerator, int denominator)
    {
        var bits = new int[length];
        for (int i = 0; i < length; i++)
        {
            bits[i] = (int)(SplitMix64.Next(ref state) % (ulong)denominator) < numerator ? 1 : 0;
        }
        return bits;
    }

    /** Brute force: all three query families are re-derived from the raw bits. */
    private static bool Matches(RrrBitVector vector, IReadOnlyList<int> bits)
    {
        int ones = 0;
        int zeros = 0;
        if (vector.Rank1(0) != 0 || vector.Rank0(0) != 0) return false;
        for (int i = 0; i < bits.Count; i++)
        {
            ones += bits[i];
            zeros += 1 - bits[i];
            if (vector.Access(i) != bits[i]) return false;
            if (vector.Rank1(i + 1) != ones || vector.Rank0(i + 1) != i + 1 - ones) return false;
            if (bits[i] == 1 && vector.Select1(ones) != i) return false;
            if (bits[i] == 0 && vector.Select0(zeros) != i) return false;
        }
        return vector.Select1(ones + 1) == -1 && vector.Select0(zeros + 1) == -1
            && vector.Select0(0) == -1 && vector.Access(bits.Count) == -1;
    }

    public static void Main()
    {
        var five = new RrrBitVector(Bits("10110"));
        Check(five.Length == 5 && five.Ones == 3, "length/ones");
        Check(five.Access(0) == 1 && five.Access(1) == 0 && five.Access(4) == 0, "access 10110");
        Check(five.Rank1(3) == 2 && five.Rank1(5) == 3 && five.Rank0(5) == 2, "rank 10110");
        Check(five.Select1(1) == 0 && five.Select1(3) == 3 && five.Select0(2) == 4, "select 10110");
        Check(five.Select1(4) == -1 && five.Select1(0) == -1 && five.Access(5) == -1, "out of range");

        var empty = new RrrBitVector(Array.Empty<int>());
        Check(empty.Length == 0 && empty.Rank1(0) == 0 && empty.Select1(1) == -1, "empty vector");

        var single1 = new RrrBitVector(Bits("1"));
        var single0 = new RrrBitVector(Bits("0"));
        Check(single1.Select1(1) == 0 && single1.Select0(1) == -1, "single one");
        Check(single0.Select0(1) == 0 && single0.Select1(1) == -1, "single zero");

        var allOnes = new RrrBitVector(Fill(64, 1));
        var allZeros = new RrrBitVector(Fill(64, 0));
        Check(allOnes.Ones == 64 && allZeros.Ones == 0, "uniform density");
        Check(Matches(allOnes, Fill(64, 1)) && Matches(allZeros, Fill(64, 0)), "uniform queries");

        foreach (string pattern in new[] { "1010101010", "0101010101", "1111100000", "0000011111", "1", "0" })
        {
            var bits = Bits(pattern);
            Check(Matches(new RrrBitVector(bits), bits), "pattern " + pattern);
        }

        // Exhaustive: every 5-bit vector, checked on access, rank and select.
        for (int mask = 0; mask < 32; mask++)
        {
            var bits = new int[5];
            for (int i = 0; i < 5; i++) bits[i] = (mask >> i) & 1;
            Check(Matches(new RrrBitVector(bits), bits), "exhaustive " + mask);
        }

        // Sample/word boundaries: exactly one block, one block plus one, and 8/9 block shapes.
        foreach (int n in new[] { 15, 16, 17, 119, 120, 121, 255, 256 })
        {
            var bits = new int[n];
            for (int i = 0; i < n; i++) bits[i] = (i * 7 + 3) % 11 == 0 ? 1 : 0;
            var vector = new RrrBitVector(bits);
            Check(Matches(vector, bits), "boundary rank n=" + n);
            Check(vector.Rank1(n) == vector.Ones && vector.Rank0(n) + vector.Ones == n,
                "boundary totals n=" + n);
        }

        foreach (var spec in new[] { (1024, 1, 2), (2048, 1, 8), (4096, 1, 32), (4096, 3, 4), (1000, 1, 3) })
        {
            var bits = RandomBits(spec.Item1, spec.Item2, spec.Item3);
            Check(Matches(new RrrBitVector(bits), bits), "random " + spec.Item2 + "/" + spec.Item3);
        }

        var dense = new RrrBitVector(RandomBits(4096, 1, 2));
        var sparseBits = RandomBits(4096, 1, 32);
        var sparse = new RrrBitVector(sparseBits);
        Check(RrrBitVector.MaxClassWidth() <= 12, "class payload width bound");
        Check(RrrBitVector.ClassSize(RrrBitVector.ClassId(0x7F80)) == 3432, "max class size C(14,7)");
        Check(RrrBitVector.ClassSize(RrrBitVector.ClassId(0)) == 1, "zero class is a singleton");
        Check(RrrBitVector.ClassSize(RrrBitVector.ClassId(1 << 14)) == 1, "leaf class is a singleton");
        Check(dense.MaxPayloadBits() <= 12 && sparse.MaxPayloadBits() <= 12, "per-block payload bound");
        Check(sparse.TotalEncodedBits() < sparse.Length, "sparse vector compresses");
        Check(dense.TotalEncodedBits() > sparse.TotalEncodedBits(), "dense metadata exceeds sparse");

        // Reproducibility: the same bits always produce the same stream and the same answers.
        var again = new RrrBitVector(sparseBits);
        Check(again.TotalEncodedBits() == sparse.TotalEncodedBits(), "reproducible encoding");
        Check(again.Select1(500) == sparse.Select1(500) && again.Rank1(3000) == sparse.Rank1(3000),
            "reproducible queries");
        Check(sparse.Select1(sparse.Rank1(4000)) >= 0 && sparse.Select0(sparse.Rank0(4000)) >= 0,
            "last occurrence selectable");

        Console.WriteLine("  " + dense.Describe());
        Console.WriteLine("  " + sparse.Describe());
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
