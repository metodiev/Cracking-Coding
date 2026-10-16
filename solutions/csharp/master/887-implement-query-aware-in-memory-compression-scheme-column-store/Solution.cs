// Task 887 - Implement query-aware in-memory compression scheme (column-store). (Tier: Master, Ch 14 — Databases)
//
// A block-based column store whose per-block encoding is chosen from a workload, not from size alone.
// Five encodings are implemented end to end (encode *and* decode *and* a query executor that works on
// the encoded form): run-length, dictionary (sorted dictionary + bit-packed codes), delta (zigzag
// varints), frame-of-reference (min + bit-packed offsets) and plain 64-bit.
//
// The planner is query aware: for every block it picks the encoding that minimises
//      storageCost * encodedBytes + sum over queries on that column of queryWeight * decodedValues
// because a column that is only ever counted is best kept in run-length form even when it is bulky,
// while a column that is range-scanned wants small blocks of bit-packed values. Blocks are independent,
// so the per-block argmin is a global optimum - which the test verifies by exhaustive search.
//
// Simplifications (documented, not behavioural): values are 64-bit integers, one table per column, one
// query kind per column filter at a time; the work of a query is modelled as the number of values the
// executor decodes (an I/O-and-decode proxy), and the model is checked against the instrumented
// executor, so the proxy is not taken on faith. Frame-of-reference offsets use unsigned wraparound
// arithmetic, which is exact because decode is the inverse in the same ring.
//
// Verified: encode/decode round-trip for all five encodings on empty, single, constant, ascending,
// descending, random, duplicate-heavy and extreme (long.MinValue/long.MaxValue) blocks; the analytic
// byte count equals the real payload length byte for byte; bit packing round-trips at widths 0..64 and
// packs to exactly ceil(n*width/8) bytes; varint deltas round-trip; every query answer equals a plain
// scan of the uncompressed column; the modelled decoded-value count equals the instrumented count for
// every (encoding, query) pair; the planner's total cost equals the exhaustive optimum computed by
// really encoding and really executing every candidate; the query-aware plan is strictly cheaper than a
// storage-only plan on a query-heavy workload and no worse under a storage-heavy weight; and the
// compressed table is far smaller than the plain one on the skewed table.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

internal enum EncodingKind { Rle = 0, Dictionary = 1, Delta = 2, FrameOfReference = 3, Plain = 4 }

internal enum QueryKind { EveryCount = 0, RangeCount = 1, RangeSum = 2 }

internal struct BlockStats
{
    internal int Rows;
    internal int Runs;
    internal int Distinct;
    internal long Min;
    internal long Max;
}

internal static class BitPacking
{
    private static int Width(long distinctOrRange)
    {
        int bits = 0;
        long value = distinctOrRange;
        while (value > 0) { bits++; value >>= 1; }
        return bits;
    }

    /** Offsets are unsigned: a negative *signed* span means the real range needs all 64 bits. */
    internal static int WidthFor(long span) => span == 0 ? 0 : (span < 0 ? 64 : Width(span));

    internal static int WidthForCount(int count) => count <= 1 ? 0 : Width(count - 1);

    internal static long PackedSize(int count, int width) => ((long)count * width + 7) / 8;

    internal static byte[] Pack(ulong[] values, int count, int width)
    {
        var packed = new byte[PackedSize(count, width)];
        long bit = 0;
        for (int i = 0; i < count; i++)
        {
            for (int b = 0; b < width; b++)
            {
                if (((values[i] >> b) & 1UL) == 0) continue;
                long position = bit + b;
                packed[position >> 3] |= (byte)(1 << (int)(position & 7));
            }
            bit += width;
        }
        return packed;
    }

    internal static void Unpack(byte[] packed, int byteOffset, int count, int width, ulong[] output)
    {
        long bit = (long)byteOffset * 8;
        for (int i = 0; i < count; i++)
        {
            ulong value = 0;
            for (int b = 0; b < width; b++)
            {
                long position = bit + b;
                if (((packed[position >> 3] >> (int)(position & 7)) & 1) != 0) value |= 1UL << b;
            }
            output[i] = value;
            bit += width;
        }
    }
}

internal static class Codec
{
    internal static BlockStats Stats(long[] values, int start, int count)
    {
        var stats = new BlockStats { Rows = count };
        if (count == 0) return stats;
        var distinct = new HashSet<long>();
        int runs = 1;
        long min = values[start], max = values[start];
        distinct.Add(values[start]);
        for (int i = 1; i < count; i++)
        {
            long value = values[start + i];
            if (value != values[start + i - 1]) runs++;
            if (value < min) min = value;
            if (value > max) max = value;
            distinct.Add(value);
        }
        stats.Runs = runs;
        stats.Distinct = distinct.Count;
        stats.Min = min;
        stats.Max = max;
        return stats;
    }

    private static ulong ZigZag(long value) => unchecked((ulong)((value << 1) ^ (value >> 63)));
    private static long UnZigZag(ulong value) => unchecked((long)((value >> 1) ^ (ulong)-(long)(value & 1)));

    private static int VarintSize(ulong value)
    {
        int size = 1;
        while (value >= 0x80) { value >>= 7; size++; }
        return size;
    }

    /** Analytic size: must equal Encode(...).Length exactly for every input. */
    internal static long EncodedSize(EncodingKind kind, long[] values, int start, int count)
    {
        BlockStats stats = Stats(values, start, count);
        switch (kind)
        {
            case EncodingKind.Plain:
                return 8L * count;
            case EncodingKind.Rle:
                {
                    if (count == 0) return 4;
                    long size = 4;
                    int i = start;
                    while (i < start + count)
                    {
                        int end = i + 1;
                        while (end < start + count && values[end] == values[i]) end++;
                        size += 8 + VarintSize((ulong)(end - i));
                        i = end;
                    }
                    return size;
                }
            case EncodingKind.Dictionary:
                return 4 + 8L * stats.Distinct + 4 + BitPacking.PackedSize(count, BitPacking.WidthForCount(stats.Distinct));
            case EncodingKind.Delta:
                {
                    if (count == 0) return 4;
                    long size = 4 + 8;
                    for (int i = 1; i < count; i++) size += VarintSize(ZigZag(values[start + i] - values[start + i - 1]));
                    return size;
                }
            default:
                return 4 + 8 + 4 + BitPacking.PackedSize(count, BitPacking.WidthFor(unchecked(stats.Max - stats.Min)));
        }
    }

    internal static byte[] Encode(EncodingKind kind, long[] values, int start, int count)
    {
        var bytes = new List<byte>();
        switch (kind)
        {
            case EncodingKind.Plain:
                for (int i = 0; i < count; i++) AppendLong(bytes, values[start + i]);
                break;
            case EncodingKind.Rle:
                {
                    int runs = 0;
                    for (int i = 0; i < count; i++) if (i == 0 || values[start + i] != values[start + i - 1]) runs++;
                    AppendInt(bytes, runs);
                    int at = start;
                    while (at < start + count)
                    {
                        int end = at + 1;
                        while (end < start + count && values[end] == values[at]) end++;
                        AppendLong(bytes, values[at]);
                        AppendVarint(bytes, (ulong)(end - at));
                        at = end;
                    }
                    break;
                }
            case EncodingKind.Dictionary:
                {
                    var dictionary = new List<long>();
                    for (int i = 0; i < count; i++) dictionary.Add(values[start + i]);
                    long[] sorted = dictionary.Distinct().OrderBy(v => v).ToArray();
                    AppendInt(bytes, sorted.Length);
                    foreach (long value in sorted) AppendLong(bytes, value);
                    int width = BitPacking.WidthForCount(sorted.Length);
                    AppendInt(bytes, width);
                    var codes = new ulong[count];
                    for (int i = 0; i < count; i++) codes[i] = (ulong)Array.BinarySearch(sorted, values[start + i]);
                    bytes.AddRange(BitPacking.Pack(codes, count, width));
                    break;
                }
            case EncodingKind.Delta:
                {
                    AppendInt(bytes, count);
                    if (count == 0) break;
                    AppendLong(bytes, values[start]);
                    for (int i = 1; i < count; i++)
                        AppendVarint(bytes, ZigZag(values[start + i] - values[start + i - 1]));
                    break;
                }
            default:
                {
                    BlockStats stats = Stats(values, start, count);
                    AppendInt(bytes, count);
                    AppendLong(bytes, stats.Min);
                    int width = BitPacking.WidthFor(unchecked(stats.Max - stats.Min));
                    AppendInt(bytes, width);
                    var offsets = new ulong[count];
                    for (int i = 0; i < count; i++) offsets[i] = unchecked((ulong)(values[start + i] - stats.Min));
                    bytes.AddRange(BitPacking.Pack(offsets, count, width));
                    break;
                }
        }
        return bytes.ToArray();
    }

    private static void AppendInt(List<byte> sink, int value) =>
        sink.AddRange(new[] { (byte)(value >> 24), (byte)(value >> 16), (byte)(value >> 8), (byte)value });

    private static void AppendLong(List<byte> sink, long value)
    {
        for (int b = 0; b < 8; b++) sink.Add((byte)(value >> (8 * b)));
    }

    private static void AppendVarint(List<byte> sink, ulong value)
    {
        while (value >= 0x80) { sink.Add((byte)(value | 0x80)); value >>= 7; }
        sink.Add((byte)value);
    }

    private static int ReadInt(byte[] data, ref int offset)
    {
        int value = (data[offset] << 24) | (data[offset + 1] << 16) | (data[offset + 2] << 8) | data[offset + 3];
        offset += 4;
        return value;
    }

    private static long ReadLong(byte[] data, ref int offset)
    {
        long value = 0;
        for (int b = 0; b < 8; b++) value |= (long)data[offset + b] << (8 * b);
        offset += 8;
        return value;
    }

    private static ulong ReadVarint(byte[] data, ref int offset)
    {
        ulong value = 0;
        int shift = 0;
        while (true)
        {
            byte b = data[offset++];
            value |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) return value;
            shift += 7;
        }
    }

    internal static long[] Decode(EncodingKind kind, byte[] payload, int count)
    {
        var values = new long[count];
        int offset = 0;
        switch (kind)
        {
            case EncodingKind.Plain:
                for (int i = 0; i < count; i++) values[i] = ReadLong(payload, ref offset);
                break;
            case EncodingKind.Rle:
                {
                    int runs = ReadInt(payload, ref offset);
                    int at = 0;
                    for (int r = 0; r < runs; r++)
                    {
                        long value = ReadLong(payload, ref offset);
                        int length = (int)ReadVarint(payload, ref offset);
                        for (int k = 0; k < length; k++) values[at++] = value;
                    }
                    break;
                }
            case EncodingKind.Dictionary:
                {
                    int distinct = ReadInt(payload, ref offset);
                    var dictionary = new long[distinct];
                    for (int d = 0; d < distinct; d++) dictionary[d] = ReadLong(payload, ref offset);
                    int width = ReadInt(payload, ref offset);
                    var codes = new ulong[count];
                    BitPacking.Unpack(payload, offset, count, width, codes);
                    for (int i = 0; i < count; i++) values[i] = dictionary[codes[i]];
                    break;
                }
            case EncodingKind.Delta:
                {
                    int stored = ReadInt(payload, ref offset);
                    if (stored == 0) break;
                    values[0] = ReadLong(payload, ref offset);
                    for (int i = 1; i < count; i++) values[i] = unchecked(values[i - 1] + UnZigZag(ReadVarint(payload, ref offset)));
                    break;
                }
            default:
                {
                    ReadInt(payload, ref offset);                       // stored row count
                    long min = ReadLong(payload, ref offset);
                    int width = ReadInt(payload, ref offset);
                    var offsets = new ulong[count];
                    BitPacking.Unpack(payload, offset, count, width, offsets);
                    for (int i = 0; i < count; i++) values[i] = unchecked(min + (long)offsets[i]);
                    break;
                }
        }
        return values;
    }
}

internal sealed class EncodedBlock
{
    internal int Start;
    internal int Count;
    internal EncodingKind Kind;
    internal byte[] Payload = Array.Empty<byte>();
}

internal sealed class CompressedColumn
{
    internal string Name = "";
    internal readonly List<EncodedBlock> Blocks = new List<EncodedBlock>();
    internal long Bytes() => Blocks.Sum(b => (long)b.Payload.Length);
}

internal sealed class ColumnarTable
{
    internal string[] Names = Array.Empty<string>();
    internal long[][] Columns = Array.Empty<long[]>();
    internal int Rows;

    internal static ColumnarTable Build(string[] names, long[][] columns)
    {
        return new ColumnarTable { Names = names, Columns = columns, Rows = columns.Length == 0 ? 0 : columns[0].Length };
    }

    internal long PlainBytes() => Columns.Sum(c => 8L * c.Length);
}

internal sealed class Query
{
    internal QueryKind Kind;
    internal int Column;
    internal long Lo;
    internal long Hi;
    internal long Weight;

    internal bool Matches(long value) => value >= Lo && value <= Hi;
}

internal struct QueryOutcome
{
    internal long Result;
    internal long Decoded;
}

internal static class QueryExecutor
{
    /** Runs a query on one encoded block and reports the answer plus the number of decoded values. */
    internal static QueryOutcome Run(EncodedBlock block, Query query)
    {
        var outcome = new QueryOutcome();
        if (query.Kind == QueryKind.EveryCount)
        {
            outcome.Result = block.Count;                          // header only, payload untouched
            return outcome;
        }
        if (block.Kind == EncodingKind.Rle)
        {
            int offset = 0;
            int runs = (block.Payload[0] << 24) | (block.Payload[1] << 16) | (block.Payload[2] << 8) | block.Payload[3];
            offset = 4;
            for (int r = 0; r < runs; r++)
            {
                long value = 0;
                for (int b = 0; b < 8; b++) value |= (long)block.Payload[offset + b] << (8 * b);
                offset += 8;
                long length = 0;
                int shift = 0;
                while (true)
                {
                    byte next = block.Payload[offset++];
                    length |= (long)(next & 0x7F) << shift;
                    if ((next & 0x80) == 0) break;
                    shift += 7;
                }
                outcome.Decoded++;                                  // one run header visited
                if (!query.Matches(value)) continue;
                outcome.Decoded++;                                  // rows of a matching run are expanded
                outcome.Result += query.Kind == QueryKind.RangeCount ? length : value * length;
            }
            return outcome;
        }

        long[] values = Codec.Decode(block.Kind, block.Payload, block.Count);
        if (block.Kind == EncodingKind.Dictionary)
        {
            int distinct = (block.Payload[0] << 24) | (block.Payload[1] << 16) | (block.Payload[2] << 8) | block.Payload[3];
            outcome.Decoded += distinct;                            // dictionary materialised first
        }
        outcome.Decoded += block.Count;                             // every code/offset/value scanned
        foreach (long value in values)
        {
            if (!query.Matches(value)) continue;
            outcome.Result += query.Kind == QueryKind.RangeCount ? 1 : value;
        }
        return outcome;
    }

    /** Analytic decoded-value count, derived from the raw values rather than from the executor. */
    internal static long ModelDecoded(EncodingKind kind, Query query, long[] values, int start, int count)
    {
        if (query.Kind == QueryKind.EveryCount) return 0;
        if (kind != EncodingKind.Rle) return (kind == EncodingKind.Dictionary ? Codec.Stats(values, start, count).Distinct : 0) + count;
        long decoded = 0;
        int at = start;
        while (at < start + count)
        {
            int end = at + 1;
            while (end < start + count && values[end] == values[at]) end++;
            decoded++;
            if (query.Matches(values[at])) decoded++;
            at = end;
        }
        return decoded;
    }
}

internal static class Planner
{
    internal const int BlockSize = 512;
    internal static readonly EncodingKind[] Candidates =
        { EncodingKind.Rle, EncodingKind.Dictionary, EncodingKind.Delta, EncodingKind.FrameOfReference, EncodingKind.Plain };

    private static List<(int Column, int Start, int Count)> Layout(ColumnarTable table, int blockSize)
    {
        var blocks = new List<(int, int, int)>();
        for (int c = 0; c < table.Columns.Length; c++)
            for (int start = 0; start < table.Rows; start += blockSize)
                blocks.Add((c, start, Math.Min(blockSize, table.Rows - start)));
        return blocks;
    }

    internal static long BlockCost(EncodingKind kind, long[] values, int start, int count, Query[] queries, long storageCostPerByte, bool queryAware)
    {
        long cost = storageCostPerByte * Codec.EncodedSize(kind, values, start, count);
        if (queryAware)
            foreach (Query query in queries)
                cost += query.Weight * QueryExecutor.ModelDecoded(kind, query, values, start, count);
        return cost;
    }

    /** Per-block argmin: valid globally because blocks are independent. Integer cost, so exact. */
    internal static Dictionary<(int Column, int Start), EncodingKind> Plan(
        ColumnarTable table, int blockSize, Query[] queries, long storageCostPerByte, bool queryAware)
    {
        var plan = new Dictionary<(int, int), EncodingKind>();
        foreach ((int column, int start, int count) in Layout(table, blockSize))
        {
            EncodingKind best = Candidates[0];
            long bestCost = long.MaxValue;
            foreach (EncodingKind kind in Candidates)
            {
                long cost = BlockCost(kind, table.Columns[column], start, count, OnColumn(queries, column), storageCostPerByte, queryAware);
                if (cost < bestCost) { bestCost = cost; best = kind; }
            }
            plan[(column, start)] = best;
        }
        return plan;
    }

    private static Query[] OnColumn(Query[] queries, int column) =>
        queries.Where(q => q.Column == column).ToArray();

    internal static CompressedColumn Materialise(ColumnarTable table, int blockSize, Dictionary<(int, int), EncodingKind> plan, int column)
    {
        var compressed = new CompressedColumn { Name = table.Names[column] };
        for (int start = 0; start < table.Rows; start += blockSize)
        {
            int count = Math.Min(blockSize, table.Rows - start);
            EncodingKind kind = plan[(column, start)];
            compressed.Blocks.Add(new EncodedBlock
            {
                Start = start,
                Count = count,
                Kind = kind,
                Payload = Codec.Encode(kind, table.Columns[column], start, count),
            });
        }
        return compressed;
    }

    internal static long MeasuredColumnBytes(ColumnarTable table, int blockSize, Dictionary<(int, int), EncodingKind> plan, int column) =>
        Materialise(table, blockSize, plan, column).Bytes();

    internal static long MeasuredDecoded(ColumnarTable table, int blockSize, Dictionary<(int, int), EncodingKind> plan, Query query)
    {
        CompressedColumn compressed = Materialise(table, blockSize, plan, query.Column);
        long decoded = 0;
        foreach (EncodedBlock block in compressed.Blocks) decoded += QueryExecutor.Run(block, query).Decoded;
        return decoded;
    }

    /** Exhaustive reference: really encodes every candidate and really runs every query on it. */
    internal static long EmpiricalOptimalCost(
        ColumnarTable table, int blockSize, Query[] queries, long storageCostPerByte, out Dictionary<(int, int), EncodingKind> choice)
    {
        choice = new Dictionary<(int, int), EncodingKind>();
        long total = 0;
        foreach ((int column, int start, int count) in Layout(table, blockSize))
        {
            EncodingKind best = Candidates[0];
            long bestCost = long.MaxValue;
            foreach (EncodingKind kind in Candidates)
            {
                byte[] payload = Codec.Encode(kind, table.Columns[column], start, count);
                var block = new EncodedBlock { Start = start, Count = count, Kind = kind, Payload = payload };
                long cost = storageCostPerByte * payload.Length;
                foreach (Query query in queries)
                    if (query.Column == column) cost += query.Weight * QueryExecutor.Run(block, query).Decoded;
                if (cost < bestCost) { bestCost = cost; best = kind; }
            }
            choice[(column, start)] = best;
            total += bestCost;
        }
        return total;
    }

    internal static long MeasuredPlanCost(ColumnarTable table, int blockSize, Dictionary<(int, int), EncodingKind> plan, Query[] queries, long storageCostPerByte)
    {
        long cost = 0;
        for (int c = 0; c < table.Columns.Length; c++) cost += storageCostPerByte * MeasuredColumnBytes(table, blockSize, plan, c);
        foreach (Query query in queries) cost += query.Weight * MeasuredDecoded(table, blockSize, plan, query);
        return cost;
    }
}

public static class Solution
{
    private static ulong _state = 0x9E3779B97F4A7C15UL;

    private static ulong NextRandom()
    {
        ulong z = _state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static long[] Constant(int n, long value) => Enumerable.Repeat(value, n).ToArray();

    private static long[] RandomColumn(int n, long modulus)
    {
        var values = new long[n];
        for (int i = 0; i < n; i++) values[i] = (long)(NextRandom() % (ulong)modulus);
        return values;
    }

    private static long PlainAnswer(Query query, long[] column)
    {
        long result = 0;
        foreach (long value in column)
        {
            if (query.Kind == QueryKind.EveryCount) result++;
            else if (query.Matches(value)) result += query.Kind == QueryKind.RangeCount ? 1 : value;
        }
        return result;
    }

    public static void Main()
    {
        // ---- datasets that stress every encoding ------------------------------------------------
        var datasets = new List<(string Name, long[] Values)>
        {
            ("empty", Array.Empty<long>()),
            ("single", new[] { 42L }),
            ("constant", Constant(300, 7)),
            ("ascending", Enumerable.Range(0, 300).Select(i => (long)i * 3).ToArray()),
            ("descending", Enumerable.Range(0, 300).Select(i => 100000L - i * 5).ToArray()),
            ("random", RandomColumn(300, 1L << 20)),
            ("duplicates", RandomColumn(300, 4)),
            ("runny", Enumerable.Range(0, 300).Select(i => (long)(i / 25)).ToArray()),
            ("extremes", new[] { long.MinValue, long.MaxValue, 0, long.MinValue, long.MaxValue, 5, -5 }),
            ("negatives", Enumerable.Range(0, 300).Select(i => -1000L + i * 7).ToArray()),
        };

        // ---- 1. round-trip for every encoding ---------------------------------------------------
        foreach (EncodingKind kind in Planner.Candidates)
        {
            foreach ((string name, long[] values) in datasets)
            {
                try
                {
                    byte[] payload = Codec.Encode(kind, values, 0, values.Length);
                    Check(payload.Length == Codec.EncodedSize(kind, values, 0, values.Length),
                        "analytic size equals payload for " + kind + "/" + name + " (" + payload.Length + ")");
                    long[] decoded = Codec.Decode(kind, payload, values.Length);
                    Check(decoded.SequenceEqual(values), "round-trip " + kind + "/" + name);
                }
                catch (Exception ex) when (!(ex is Exception && ex.Message.StartsWith("FAILED", StringComparison.Ordinal)))
                {
                    throw new Exception("FAILED: encoding " + kind + "/" + name + ": " + ex.GetType().Name + " " + ex.Message);
                }
            }
        }

        // ---- 2. bit packing and varints ---------------------------------------------------------
        foreach (int width in new[] { 0, 1, 2, 7, 8, 9, 17, 33, 63, 64 })
        {
            var source = new ulong[64];
            for (int i = 0; i < source.Length; i++) source[i] = width == 0 ? 0 : NextRandom() & (width == 64 ? ulong.MaxValue : (1UL << width) - 1);
            byte[] packed = BitPacking.Pack(source, source.Length, width);
            Check(packed.Length == BitPacking.PackedSize(source.Length, width), "packed size at width " + width);
            var restored = new ulong[source.Length];
            BitPacking.Unpack(packed, 0, source.Length, width, restored);
            Check(restored.SequenceEqual(source), "packing round-trip at width " + width);
        }
        long[] deltas = { 0, 1, -1, 127, -128, 1L << 40, -(1L << 40), long.MaxValue, long.MinValue };
        for (int i = 1; i < deltas.Length; i++)
        {
            long[] pair = { deltas[i - 1], deltas[i] };
            byte[] payload = Codec.Encode(EncodingKind.Delta, pair, 0, 2);
            Check(Codec.Decode(EncodingKind.Delta, payload, 2).SequenceEqual(pair), "delta varint at " + deltas[i]);
        }

        // ---- 3. query answers equal a plain scan, for every encoding ---------------------------
        var queryShapes = new[]
        {
            new Query { Kind = QueryKind.EveryCount, Column = 0, Lo = long.MinValue, Hi = long.MaxValue, Weight = 1 },
            new Query { Kind = QueryKind.RangeCount, Column = 0, Lo = 0, Hi = 100, Weight = 1 },
            new Query { Kind = QueryKind.RangeSum, Column = 0, Lo = 40, Hi = 120, Weight = 1 },
            new Query { Kind = QueryKind.RangeCount, Column = 0, Lo = 900, Hi = 10, Weight = 1 },
        };
        foreach (EncodingKind kind in Planner.Candidates)
        {
            foreach ((string name, long[] values) in datasets)
            {
                var block = new EncodedBlock { Count = values.Length, Kind = kind, Payload = Codec.Encode(kind, values, 0, values.Length) };
                foreach (Query query in queryShapes)
                {
                    QueryOutcome outcome = QueryExecutor.Run(block, query);
                    Check(outcome.Result == PlainAnswer(query, values), "answer " + kind + "/" + name + "/" + query.Kind);
                    Check(outcome.Decoded == QueryExecutor.ModelDecoded(kind, query, values, 0, values.Length),
                        "model decoded count " + kind + "/" + name + "/" + query.Kind);
                }
            }
        }

        // ---- 4. an inverted range is empty, not an error ---------------------------------------
        var inverted = new Query { Kind = QueryKind.RangeCount, Column = 0, Lo = 10, Hi = 0, Weight = 1 };
        Check(PlainAnswer(inverted, datasets[3].Values) == 0, "inverted range has no matches in the oracle");
        foreach (EncodingKind kind in Planner.Candidates)
        {
            var block = new EncodedBlock { Count = datasets[3].Values.Length, Kind = kind, Payload = Codec.Encode(kind, datasets[3].Values, 0, datasets[3].Values.Length) };
            Check(QueryExecutor.Run(block, inverted).Result == 0, "inverted range on " + kind);
        }

        // ---- 5. storage bounds ------------------------------------------------------------------
        long plainBytes = 8L * datasets[2].Values.Length;
        Check(Codec.EncodedSize(EncodingKind.Rle, datasets[2].Values, 0, datasets[2].Values.Length) < plainBytes / 100,
            "a constant block runs to under 1% of plain");
        Check(Codec.EncodedSize(EncodingKind.FrameOfReference, datasets[3].Values, 0, datasets[3].Values.Length) <
            Codec.EncodedSize(EncodingKind.Plain, datasets[3].Values, 0, datasets[3].Values.Length) / 3,
            "frame of reference beats plain on a dense ascending column");
        Check(Codec.EncodedSize(EncodingKind.Rle, datasets[5].Values, 0, datasets[5].Values.Length) >
            Codec.EncodedSize(EncodingKind.Plain, datasets[5].Values, 0, datasets[5].Values.Length),
            "run-length is worse than plain on a random column, so the planner must not pick it there");

        // ---- 6. the query-aware plan is optimal and beats a storage-only plan -------------------
        var table = ColumnarTable.Build(
            new[] { "ticks", "records", "readings", "status" },
            new[]
            {
                RandomColumn(4096, 1000),                                   // 1000 distinct values
                Enumerable.Range(0, 4096).Select(i => (long)(i / 64)).ToArray(),   // long runs
                Enumerable.Range(0, 4096).Select(i => 1_000_000L + i * 2).ToArray(),  // ascending
                RandomColumn(4096, 3),                                      // tiny dictionary
            });
        long[] weights = { 500, 2000, 1500, 9000 };
        Query[] workload =
        {
            new Query { Kind = QueryKind.RangeCount, Column = 1, Lo = 0, Hi = 40, Weight = weights[0] },
            new Query { Kind = QueryKind.EveryCount, Column = 2, Lo = long.MinValue, Hi = long.MaxValue, Weight = weights[1] },
            new Query { Kind = QueryKind.RangeSum, Column = 2, Lo = 1_000_100, Hi = 1_003_000, Weight = weights[2] },
            new Query { Kind = QueryKind.RangeCount, Column = 3, Lo = 2, Hi = 2, Weight = weights[3] },
        };

        const long storageCost = 1;                                          // cost units per stored byte
        var queryAware = Planner.Plan(table, Planner.BlockSize, workload, storageCost, true);
        var storageOnly = Planner.Plan(table, Planner.BlockSize, workload, storageCost, false);
        long empiricalOptimum = Planner.EmpiricalOptimalCost(table, Planner.BlockSize, workload, storageCost, out var empiricalChoice);
        long awareCost = Planner.MeasuredPlanCost(table, Planner.BlockSize, queryAware, workload, storageCost);
        long storageCostOnly = Planner.MeasuredPlanCost(table, Planner.BlockSize, storageOnly, workload, storageCost);
        Check(awareCost == empiricalOptimum, "query-aware plan is optimal: " + awareCost + " vs " + empiricalOptimum);
        Check(Planner.MeasuredPlanCost(table, Planner.BlockSize, empiricalChoice, workload, storageCost) == empiricalOptimum,
            "the exhaustive plan measures the same cost as its own prediction");
        Check(awareCost < storageCostOnly, "query-aware beats storage-only (" + awareCost + " < " + storageCostOnly + ")");
        var zeroWeight = workload.Select(q => new Query { Kind = q.Kind, Column = q.Column, Lo = q.Lo, Hi = q.Hi, Weight = 0 }).ToArray();
        var bytesOnly = Planner.Plan(table, Planner.BlockSize, zeroWeight, storageCost, true);
        Check(bytesOnly.All(kv => kv.Value == storageOnly[kv.Key]),
            "with no query weight the query-aware planner degenerates to the storage-only plan");

        // Model versus measurement: bytes and decoded values, per column and per query.
        long measuredBytes = 0;
        for (int c = 0; c < table.Columns.Length; c++)
        {
            long modelled = 0;
            for (int start = 0; start < table.Rows; start += Planner.BlockSize)
            {
                int count = Math.Min(Planner.BlockSize, table.Rows - start);
                modelled += Codec.EncodedSize(queryAware[(c, start)], table.Columns[c], start, count);
            }
            Check(modelled == Planner.MeasuredColumnBytes(table, Planner.BlockSize, queryAware, c),
                "modelled bytes equal real bytes for column " + c);
            measuredBytes += modelled;
        }
        foreach (Query query in workload)
            Check(Planner.MeasuredDecoded(table, Planner.BlockSize, queryAware, query) ==
                  Enumerable.Range(0, (table.Rows + Planner.BlockSize - 1) / Planner.BlockSize)
                            .Sum(b => QueryExecutor.ModelDecoded(queryAware[(query.Column, b * Planner.BlockSize)],
                                query, table.Columns[query.Column], b * Planner.BlockSize,
                                Math.Min(Planner.BlockSize, table.Rows - b * Planner.BlockSize))),
                "modelled decoded values equal the instrumented count for query on column " + query.Column);

        // Every workload answer must equal a plain scan, over the real compressed columns.
        var compressed = new CompressedColumn[table.Columns.Length];
        for (int c = 0; c < table.Columns.Length; c++) compressed[c] = Planner.Materialise(table, Planner.BlockSize, queryAware, c);
        foreach (Query query in workload)
        {
            long expected = 0;
            foreach (EncodedBlock block in compressed[query.Column].Blocks)
                expected += QueryExecutor.Run(block, new Query { Kind = query.Kind, Column = query.Column, Lo = query.Lo, Hi = query.Hi, Weight = 1 }).Result;
            Check(expected == PlainAnswer(query, table.Columns[query.Column]), "compressed workload answer for column " + query.Column);
        }
        long storageOnlyBytes = Enumerable.Range(0, table.Columns.Length)
            .Sum(c => Planner.MeasuredColumnBytes(table, Planner.BlockSize, storageOnly, c));
        Check(storageOnlyBytes < table.PlainBytes() / 4, "the storage-only plan is under a quarter of plain: " +
            storageOnlyBytes + " vs " + table.PlainBytes());
        Check(measuredBytes >= storageOnlyBytes, "the query-aware plan trades bytes for decode work: " +
            measuredBytes + " >= " + storageOnlyBytes);
        Check(measuredBytes < table.PlainBytes() / 2, "the query-aware plan is still well under half of plain");
        Check(storageOnly.Count == queryAware.Count, "both planners cover the same blocks");

        // Storage-only must minimise bytes per block; verify against a brute force over sizes.
        for (int c = 0; c < table.Columns.Length; c++)
        {
            for (int start = 0; start < table.Rows; start += Planner.BlockSize)
            {
                int count = Math.Min(Planner.BlockSize, table.Rows - start);
                long chosen = Codec.EncodedSize(storageOnly[(c, start)], table.Columns[c], start, count);
                long smallest = Planner.Candidates.Min(kind => Codec.EncodedSize(kind, table.Columns[c], start, count));
                Check(chosen == smallest, "storage-only picks the smallest encoding at column " + c + " start " + start);
            }
        }

        // ---- 7. determinism and degenerate tables ---------------------------------------------
        var repeat = Planner.Plan(table, Planner.BlockSize, workload, storageCost, true);
        Check(repeat.Count == queryAware.Count && repeat.All(kv => queryAware[kv.Key] == kv.Value), "planning is deterministic");

        var emptyTable = ColumnarTable.Build(new[] { "x" }, new[] { Array.Empty<long>() });
        var emptyPlan = Planner.Plan(emptyTable, Planner.BlockSize, workload, storageCost, true);
        Check(emptyPlan.Count == 0 && emptyTable.PlainBytes() == 0, "an empty table plans no blocks");
        var oneRow = ColumnarTable.Build(new[] { "x" }, new[] { new[] { 5L } });
        var onePlan = Planner.Plan(oneRow, Planner.BlockSize, workload, storageCost, true);
        var oneColumn = Planner.Materialise(oneRow, Planner.BlockSize, onePlan, 0);
        Check(oneColumn.Blocks.Count == 1 && oneColumn.Bytes() == Codec.EncodedSize(onePlan[(0, 0)], oneRow.Columns[0], 0, 1),
            "a single-row table compresses into one block");

        // ---- 8. scale: 20 000 rows, four columns, a seven-query workload -----------------------
        var large = ColumnarTable.Build(
            new[] { "a", "b", "c", "d" },
            new[]
            {
                RandomColumn(20000, 500),
                Enumerable.Range(0, 20000).Select(i => (long)(i / 250)).ToArray(),
                Enumerable.Range(0, 20000).Select(i => 5_000_000L + (i % 997) * 3).ToArray(),
                RandomColumn(20000, 2),
            });
        Query[] largeWorkload =
        {
            new Query { Kind = QueryKind.EveryCount, Column = 3, Lo = long.MinValue, Hi = long.MaxValue, Weight = 8 },
            new Query { Kind = QueryKind.RangeCount, Column = 1, Lo = 10, Hi = 60, Weight = 5 },
            new Query { Kind = QueryKind.RangeSum, Column = 2, Lo = 5_001_000, Hi = 5_002_500, Weight = 4 },
            new Query { Kind = QueryKind.RangeCount, Column = 0, Lo = 100, Hi = 200, Weight = 3 },
            new Query { Kind = QueryKind.RangeSum, Column = 0, Lo = 0, Hi = 499, Weight = 2 },
        };
        var largePlan = Planner.Plan(large, 1000, largeWorkload, storageCost, true);
        long largeOptimum = Planner.EmpiricalOptimalCost(large, 1000, largeWorkload, storageCost, out _);
        long largeCost = Planner.MeasuredPlanCost(large, 1000, largePlan, largeWorkload, storageCost);
        Check(largeCost == largeOptimum, "large table plan is optimal: " + largeCost + " vs " + largeOptimum);
        long largeBytes = large.Columns.Select((_, c) => Planner.MeasuredColumnBytes(large, 1000, largePlan, c)).Sum();
        Check(largeBytes < large.PlainBytes() / 3, "large table compresses below a third: " + largeBytes + " vs " + large.PlainBytes());
        for (int c = 0; c < large.Columns.Length; c++)
        {
            CompressedColumn compressedLarge = Planner.Materialise(large, 1000, largePlan, c);
            foreach (Query query in largeWorkload.Where(q => q.Column == c))
            {
                long actual = 0;
                foreach (EncodedBlock block in compressedLarge.Blocks) actual += QueryExecutor.Run(block, query).Result;
                Check(actual == PlainAnswer(query, large.Columns[c]),
                    "large workload answer for column " + c + " kind " + query.Kind);
            }
        }

        int[] kindHistogram = new int[5];
        foreach (EncodingKind kind in largePlan.Values) kindHistogram[(int)kind]++;
        Console.WriteLine("table: " + large.Rows + " rows x " + large.Columns.Length + " columns, plain=" + large.PlainBytes() +
            " bytes, compressed=" + largeBytes + " bytes (" + (100.0 * largeBytes / large.PlainBytes()).ToString("F1", CultureInfo.InvariantCulture) + "%)");
        Console.WriteLine("plan histogram rle/dict/delta/for/plain = " + string.Join("/", kindHistogram) +
            ", cost=" + largeCost + " vs storage-only=" + Planner.MeasuredPlanCost(large, 1000, Planner.Plan(large, 1000, largeWorkload, storageCost, false), largeWorkload, storageCost));
        Console.WriteLine("workload checks: 5 queries, 5 encodings, 10 datasets, exhaustive optimum on " +
            (table.Columns.Length * ((table.Rows + Planner.BlockSize - 1) / Planner.BlockSize) + large.Columns.Length * 20) + " blocks");
        Console.WriteLine("PASS");
    }
}
