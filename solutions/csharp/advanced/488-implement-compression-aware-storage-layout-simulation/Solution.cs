// Task 488 - Implement compression-aware storage layout simulation. (Tier: Advanced, CTCI Ch 14)
//
// Value blocks (32 non-negative 16-bit ints each; the last block may be shorter) are encoded with
// the cheaper of two schemes, chosen by exact byte count with a documented tie rule (RLE first):
// - RLE     : header 2 bytes (mode=0, run count) + 4 bytes per run (value, run length).
// - BITPACK : header 2 bytes (mode=1, bit width w) + ceil(count * w / 8) bytes, LSB-first packing,
//             where w = max(1, max bit length of the block's values). For values < 2^31 bit packing
//             with w <= 31 bits is never worse than a raw 32-bit copy, so raw mode is not needed.
//
// "Compression-aware layout": blocks are grouped by similarity - a stable sort by bit width puts
// similar blocks next to each other, and consecutive blocks sharing a width cost one 2-byte group
// header for the whole run instead of per-block overhead. StoredBytes counts group headers plus
// per-block payloads only.
//
// The test asserts exact round-trips on skewed, mixed and hand blocks, checks the mode actually
// chosen per block, and bounds stored bytes on skewed data: for 1024 ints in {0..3} the layout
// stores 322 bytes vs 4096 raw (<= raw/3), matching the pinned exact count.
using System;
using System.Collections.Generic;

internal class BlockInfo
{
    public readonly int Mode;
    public readonly int Width; // -1 for RLE
    public readonly byte[] Payload;
    public readonly int Runs; // -1 for BITPACK

    public BlockInfo(int mode, int width, byte[] payload, int runs)
    {
        Mode = mode;
        Width = width;
        Payload = payload;
        Runs = runs;
    }
}

internal class LayoutEntry
{
    public readonly int Index;
    public readonly int Group;
    public readonly BlockInfo Info;

    public LayoutEntry(int index, int group, BlockInfo info)
    {
        Index = index;
        Group = group;
        Info = info;
    }
}

internal class Layout
{
    public readonly List<LayoutEntry> Entries;
    public readonly int StoredBytes;
    public readonly int Groups;

    public Layout(List<LayoutEntry> entries, int storedBytes, int groups)
    {
        Entries = entries;
        StoredBytes = storedBytes;
        Groups = groups;
    }
}

public static class Solution
{
    private const int ModeRle = 0;
    private const int ModeBitpack = 1;
    private const int Header = 2;

    private static int BitLength(int value)
    {
        int bits = 0;
        while (value > 0)
        {
            bits++;
            value >>= 1;
        }
        return bits;
    }

    private static BlockInfo EncodeBlock(int[] values)
    {
        var runs = new List<int[]>();
        foreach (int value in values)
        {
            if (runs.Count > 0 && runs[runs.Count - 1][0] == value) runs[runs.Count - 1][1]++;
            else runs.Add(new[] { value, 1 });
        }
        int rleSize = Header + 4 * runs.Count;
        int width = 1;
        foreach (int value in values) width = Math.Max(width, BitLength(value));
        int bitpackSize = Header + (values.Length * width + 7) / 8;
        if (rleSize <= bitpackSize)
        {
            var payload = new byte[rleSize];
            payload[0] = ModeRle;
            payload[1] = (byte)runs.Count;
            int at = Header;
            foreach (int[] run in runs)
            {
                payload[at++] = (byte)(run[0] & 0xFF);
                payload[at++] = (byte)((run[0] >> 8) & 0xFF);
                payload[at++] = (byte)(run[1] & 0xFF);
                payload[at++] = (byte)((run[1] >> 8) & 0xFF);
            }
            return new BlockInfo(ModeRle, -1, payload, runs.Count);
        }
        var packed = new byte[bitpackSize];
        packed[0] = ModeBitpack;
        packed[1] = (byte)width;
        int acc = 0, bits = 0, offset = Header;
        foreach (int value in values)
        {
            acc |= value << bits;
            bits += width;
            while (bits >= 8)
            {
                packed[offset++] = (byte)(acc & 0xFF);
                acc = (int)((uint)acc >> 8);
                bits -= 8;
            }
        }
        if (bits > 0) packed[offset++] = (byte)(acc & 0xFF);
        Check(offset == bitpackSize, "bitpack size " + offset + " " + bitpackSize);
        return new BlockInfo(ModeBitpack, width, packed, -1);
    }

    private static int[] DecodeBlock(BlockInfo info, int count)
    {
        byte[] payload = info.Payload;
        if (payload[0] == ModeRle)
        {
            var values = new List<int>();
            int at = Header;
            for (int i = 0; i < payload[1]; i++)
            {
                int value = payload[at] | (payload[at + 1] << 8);
                int run = payload[at + 2] | (payload[at + 3] << 8);
                at += 4;
                for (int j = 0; j < run; j++) values.Add(value);
            }
            Check(values.Count == count, "rle count");
            return values.ToArray();
        }
        int width = payload[1];
        var decoded = new int[count];
        int accumulator = 0, bits = 0, offset = Header;
        for (int i = 0; i < count; i++)
        {
            while (bits < width)
            {
                accumulator |= payload[offset] << bits;
                offset++;
                bits += 8;
            }
            decoded[i] = accumulator & ((1 << width) - 1);
            accumulator = (int)((uint)accumulator >> width);
            bits -= width;
        }
        return decoded;
    }

    private static Layout BuildLayout(List<int[]> blocks)
    {
        var infos = new List<BlockInfo>();
        foreach (int[] values in blocks) infos.Add(EncodeBlock(values));
        var order = new List<int>();
        for (int i = 0; i < blocks.Count; i++) order.Add(i);
        order.Sort((a, b) =>
        {
            int cmp = infos[a].Width.CompareTo(infos[b].Width);
            return cmp != 0 ? cmp : a.CompareTo(b);
        });
        var entries = new List<LayoutEntry>();
        int stored = 0, groups = 0, previousWidth = int.MinValue;
        foreach (int index in order)
        {
            BlockInfo info = infos[index];
            if (info.Width != previousWidth)
            {
                groups++;
                stored += Header;
                previousWidth = info.Width;
            }
            stored += info.Payload.Length;
            entries.Add(new LayoutEntry(index, groups - 1, info));
        }
        return new Layout(entries, stored, groups);
    }

    private static List<int[]> DecodeLayout(Layout layout, int[] lengths)
    {
        var blocks = new int[lengths.Length][];
        foreach (LayoutEntry entry in layout.Entries)
        {
            Check(blocks[entry.Index] == null, "duplicate index");
            blocks[entry.Index] = DecodeBlock(entry.Info, lengths[entry.Index]);
        }
        return new List<int[]>(blocks);
    }

    private static List<int[]> Chunk(int[] values, int size)
    {
        var blocks = new List<int[]>();
        for (int i = 0; i < values.Length; i += size)
        {
            int length = Math.Min(size, values.Length - i);
            var block = new int[length];
            Array.Copy(values, i, block, 0, length);
            blocks.Add(block);
        }
        return blocks;
    }

    private static int seed = 488;

    private static int[] LcgValues(int count, int limit)
    {
        var values = new int[count];
        for (int i = 0; i < count; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            values[i] = seed % limit;
        }
        return values;
    }

    private static bool SameBlocks(List<int[]> a, List<int[]> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++)
        {
            if (a[i].Length != b[i].Length) return false;
            for (int j = 0; j < a[i].Length; j++) if (a[i][j] != b[i][j]) return false;
        }
        return true;
    }

    private static int[] Lengths(List<int[]> blocks)
    {
        var lengths = new int[blocks.Count];
        for (int i = 0; i < blocks.Count; i++) lengths[i] = blocks[i].Length;
        return lengths;
    }

    public static void Main()
    {
        var constantValues = new int[32];
        for (int i = 0; i < 32; i++) constantValues[i] = 7;
        BlockInfo constant = EncodeBlock(constantValues);
        Check(constant.Mode == ModeRle && constant.Runs == 1, "constant rle");
        Check(constant.Payload.Length == 6, "constant size");
        Check(SameBlocks(new List<int[]> { DecodeBlock(constant, 32) }, new List<int[]> { constantValues }), "constant round trip");

        BlockInfo zeros = EncodeBlock(new int[32]);
        Check(zeros.Mode == ModeRle && zeros.Payload.Length == 6, "zeros rle");
        Check(SameBlocks(new List<int[]> { DecodeBlock(zeros, 32) }, new List<int[]> { new int[32] }), "zeros round trip");

        var narrowValues = new int[32];
        for (int i = 0; i < 32; i++) narrowValues[i] = i % 4;
        BlockInfo narrow = EncodeBlock(narrowValues);
        Check(narrow.Mode == ModeBitpack && narrow.Width == 2, "narrow bitpack");
        Check(narrow.Payload.Length == 2 + 8, "narrow size");
        Check(SameBlocks(new List<int[]> { DecodeBlock(narrow, 32) }, new List<int[]> { narrowValues }), "narrow round trip");

        var wideValues = new int[32];
        for (int i = 0; i < 32; i++) wideValues[i] = new[] { 65535, 0, 12345, 65534 }[i % 4];
        BlockInfo wide = EncodeBlock(wideValues);
        Check(wide.Mode == ModeBitpack && wide.Width == 16, "wide bitpack");
        Check(wide.Payload.Length == 2 + 64, "wide size");
        Check(SameBlocks(new List<int[]> { DecodeBlock(wide, 32) }, new List<int[]> { wideValues }), "wide round trip");

        var countingValues = new int[32];
        for (int i = 0; i < 32; i++) countingValues[i] = i + 1;
        BlockInfo counting = EncodeBlock(countingValues);
        Check(counting.Mode == ModeBitpack && counting.Width == 6, "counting bitpack");
        Check(SameBlocks(new List<int[]> { DecodeBlock(counting, 32) }, new List<int[]> { countingValues }), "counting round trip");

        seed = 488;
        int[] skewed = LcgValues(1024, 4);
        List<int[]> blocks = Chunk(skewed, 32);
        Check(blocks.Count == 32, "block count");
        Layout layout = BuildLayout(blocks);
        Check(layout.Groups == 1, "skewed groups");
        Check(layout.StoredBytes == 322, "skewed stored " + layout.StoredBytes);
        int raw = 4 * skewed.Length;
        Check(layout.StoredBytes <= raw / 3, "skewed bound");
        Check(SameBlocks(DecodeLayout(layout, Lengths(blocks)), blocks), "skewed round trip");

        seed = 489;
        int[] firstHalf = LcgValues(512, 4);
        seed = 490;
        int[] secondHalf = LcgValues(512, 65536);
        var mixedValues = new int[1024];
        Array.Copy(firstHalf, 0, mixedValues, 0, 512);
        Array.Copy(secondHalf, 0, mixedValues, 512, 512);
        List<int[]> mixed = Chunk(mixedValues, 32);
        Check(mixed.Count == 32, "mixed block count");
        layout = BuildLayout(mixed);
        Check(layout.Groups == 2, "mixed groups " + layout.Groups);
        Check(layout.StoredBytes == 16 * 10 + 16 * 66 + 2 * 2, "mixed stored " + layout.StoredBytes);
        Check(layout.StoredBytes <= (4 * mixedValues.Length) / 3, "mixed bound");
        Check(SameBlocks(DecodeLayout(layout, Lengths(mixed)), mixed), "mixed round trip");
        int previousWidth = int.MinValue, changes = 0, groupCount = 0;
        foreach (LayoutEntry entry in layout.Entries)
        {
            if (entry.Info.Width != previousWidth)
            {
                groupCount++;
                if (previousWidth != int.MinValue) changes++;
                previousWidth = entry.Info.Width;
            }
        }
        Check(changes == groupCount - 1 && groupCount == layout.Groups, "grouping contiguous");

        var shortValues = new int[33];
        for (int i = 0; i < 33; i++) shortValues[i] = 5;
        List<int[]> shorts = Chunk(shortValues, 32);
        Check(shorts.Count == 2 && shorts[0].Length == 32 && shorts[1].Length == 1, "short blocks");
        layout = BuildLayout(shorts);
        Check(SameBlocks(DecodeLayout(layout, new[] { 32, 1 }), shorts), "short round trip");
        Check(layout.Groups == 2, "short groups");
        Check(layout.StoredBytes == 6 + 3 + 2 + 2, "short stored " + layout.StoredBytes);
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
