// Task 647 - Implement column-store compressed page layout and predicate pushdown simulation. (Tier: Expert, CTCI Ch 14 - Databases)
//
// A column is a list of fixed-size pages, and every page is encoded with whichever of three layouts is
// smallest for that page - the layout is chosen by encoding the page three times and comparing bytes,
// so the choice is data driven rather than hard-coded:
//   * Plain      - 32 bits per value (the uncompressed baseline),
//   * Dictionary - 32-bit dictionary count, the distinct values, then one packed index per value with
//                  ceil(log2(distinct)) bits,
//   * Rle        - run-length encoded (value, run length) pairs, 8 bytes per run.
// Every page also carries a zone map (min/max) of the values it holds. A scan decodes a page only when
// the zone map overlaps the predicate's range, which is the predicate pushdown; the soundness
// requirement is that a skipped page can never contain a matching value.
//
// Bit packing is a hand-written LSB-first bit writer/reader, so the byte accounting is exact and can be
// checked by hand. Simplified: single-column scans with a closed range predicate, pages of equal size,
// no nulls / no delete vectors / no run-merging across pages, no SIMD (task 648 covers vectorisation)
// and the "rows examined" figure counts decoded values rather than real bytes touched in a cache.
//
// Verified: bit packing round-trips for every width, each layout decodes back to the exact original
// values, the chosen layout is the smallest of the three (and matches hand-computed byte counts), the
// pushdown answer equals a brute-force scan for many predicates, no skipped page ever contained a
// match, pushdown decodes strictly fewer pages for a selective predicate, and late materialisation of a
// second column touches only the pages that hold the requested rows.
using System;
using System.Collections.Generic;
using System.Linq;

internal enum PageEncoding
{
    Plain,
    Dictionary,
    Rle,
}

internal sealed class BitWriter
{
    private readonly List<byte> _bytes = new List<byte>();
    private int _bits;

    public int BitLength => _bits;
    public int ByteLength => _bytes.Count;

    public void Write(int value, int width)
    {
        for (int i = 0; i < width; i++)
        {
            int bit = (value >> i) & 1;
            if (_bits % 8 == 0) _bytes.Add(0);
            if (bit != 0) _bytes[_bytes.Count - 1] |= (byte)(1 << (_bits % 8));
            _bits++;
        }
    }

    public byte[] ToArray() => _bytes.ToArray();
}

internal sealed class BitReader
{
    private readonly byte[] _bytes;
    private int _bits;

    public BitReader(byte[] bytes) => _bytes = bytes;

    public int Read(int width)
    {
        int value = 0;
        for (int i = 0; i < width; i++)
        {
            int bit = (_bytes[_bits / 8] >> (_bits % 8)) & 1;
            value |= bit << i;
            _bits++;
        }
        return value;
    }
}

internal sealed class ColumnPage
{
    public readonly int RowStart;
    public readonly int Count;
    public readonly PageEncoding Encoding;
    public readonly byte[] Payload;
    public readonly int Min;
    public readonly int Max;

    public int Decodes;

    public ColumnPage(int rowStart, PageEncoding encoding, byte[] payload, int count, int min, int max)
    {
        RowStart = rowStart;
        Encoding = encoding;
        Payload = payload;
        Count = count;
        Min = min;
        Max = max;
    }

    public int ByteLength => Payload.Length;

    public bool Overlaps(int low, int high) => Max >= low && Min <= high;

    public int[] Decode()
    {
        Decodes++;
        var reader = new BitReader(Payload);
        var values = new int[Count];
        switch (Encoding)
        {
            case PageEncoding.Plain:
                for (int i = 0; i < Count; i++) values[i] = reader.Read(32);
                break;
            case PageEncoding.Dictionary:
            {
                int distinct = reader.Read(32);
                var dictionary = new int[distinct];
                for (int i = 0; i < distinct; i++) dictionary[i] = reader.Read(32);
                int bits = ColumnCodec.BitsNeeded(Math.Max(1, distinct - 1));
                for (int i = 0; i < Count; i++) values[i] = dictionary[reader.Read(bits)];
                break;
            }
            default:
            {
                int runs = reader.Read(32);
                int position = 0;
                for (int r = 0; r < runs; r++)
                {
                    int value = reader.Read(32);
                    int length = reader.Read(32);
                    for (int i = 0; i < length; i++) values[position++] = value;
                }
                break;
            }
        }
        return values;
    }
}

internal static class ColumnCodec
{
    public static int BitsNeeded(int maxValue)
    {
        int bits = 0;
        while ((1 << bits) <= maxValue && bits < 32) bits++;
        return Math.Max(1, bits);
    }

    public static ColumnPage EncodePage(int[] values, int rowStart)
    {
        int min = values.Min();
        int max = values.Max();

        Dictionary<int, int> map = new Dictionary<int, int>();
        foreach (int value in values)
        {
            if (!map.ContainsKey(value)) map[value] = map.Count;
        }
        int[] dictionary = map.Keys.OrderBy(v => v).ToArray();
        var indexOf = new Dictionary<int, int>();
        for (int i = 0; i < dictionary.Length; i++) indexOf[dictionary[i]] = i;

        var runs = new List<KeyValuePair<int, int>>();
        foreach (int value in values)
        {
            if (runs.Count > 0 && runs[runs.Count - 1].Key == value)
            {
                KeyValuePair<int, int> last = runs[runs.Count - 1];
                runs[runs.Count - 1] = new KeyValuePair<int, int>(last.Key, last.Value + 1);
            }
            else
            {
                runs.Add(new KeyValuePair<int, int>(value, 1));
            }
        }

        int plainBytes = values.Length * 4;
        int indexBits = BitsNeeded(Math.Max(1, dictionary.Length - 1));
        int dictionaryBytes = 4 + dictionary.Length * 4 + (values.Length * indexBits + 7) / 8;
        int rleBytes = 4 + runs.Count * 8;

        int smallest = Math.Min(plainBytes, Math.Min(dictionaryBytes, rleBytes));
        if (plainBytes == smallest) return new ColumnPage(rowStart, PageEncoding.Plain, EncodePlain(values), values.Length, min, max);
        if (dictionaryBytes == smallest) return new ColumnPage(rowStart, PageEncoding.Dictionary, EncodeDictionary(values, dictionary, indexOf), values.Length, min, max);
        return new ColumnPage(rowStart, PageEncoding.Rle, EncodeRle(runs), values.Length, min, max);
    }

    public static int EncodedSize(IReadOnlyList<int> values)
    {
        int[] buffer = values.ToArray();
        int distinct = buffer.Distinct().Count();
        int indexBits = BitsNeeded(Math.Max(1, distinct - 1));
        int runs = 1;
        for (int i = 1; i < buffer.Length; i++)
        {
            if (buffer[i] != buffer[i - 1]) runs++;
        }
        return Math.Min(buffer.Length * 4, Math.Min(4 + distinct * 4 + (buffer.Length * indexBits + 7) / 8, 4 + runs * 8));
    }

    private static byte[] EncodePlain(int[] values)
    {
        var writer = new BitWriter();
        foreach (int value in values) writer.Write(value, 32);
        return writer.ToArray();
    }

    private static byte[] EncodeDictionary(int[] values, int[] dictionary, Dictionary<int, int> indexOf)
    {
        var writer = new BitWriter();
        writer.Write(dictionary.Length, 32);
        foreach (int value in dictionary) writer.Write(value, 32);
        int bits = BitsNeeded(Math.Max(1, dictionary.Length - 1));
        foreach (int value in values) writer.Write(indexOf[value], bits);
        return writer.ToArray();
    }

    private static byte[] EncodeRle(List<KeyValuePair<int, int>> runs)
    {
        var writer = new BitWriter();
        writer.Write(runs.Count, 32);
        foreach (KeyValuePair<int, int> run in runs)
        {
            writer.Write(run.Key, 32);
            writer.Write(run.Value, 32);
        }
        return writer.ToArray();
    }
}

internal sealed class Column
{
    public readonly string Name;
    public readonly int RowCount;
    public readonly List<ColumnPage> Pages = new List<ColumnPage>();
    private readonly int[] _values;

    public Column(string name, int[] values, int pageSize)
    {
        Name = name;
        _values = values;
        RowCount = values.Length;
        for (int start = 0; start < values.Length; start += pageSize)
        {
            int count = Math.Min(pageSize, values.Length - start);
            Pages.Add(ColumnCodec.EncodePage(values.Skip(start).Take(count).ToArray(), start));
        }
    }

    public int OriginalBytes => RowCount * 4;
    public int CompressedBytes => Pages.Sum(p => p.ByteLength) + Pages.Count * 8;
    public int PagesDecoded => Pages.Sum(p => p.Decodes);
    public void ResetCounters() { foreach (ColumnPage page in Pages) page.Decodes = 0; }

    public List<int> RangeScan(int low, int high, bool pushdown)
    {
        var matches = new List<int>();
        foreach (ColumnPage page in Pages)
        {
            if (pushdown && !page.Overlaps(low, high)) continue;
            int[] values = page.Decode();
            for (int i = 0; i < values.Length; i++)
            {
                if (values[i] >= low && values[i] <= high) matches.Add(page.RowStart + i);
            }
        }
        return matches;
    }

    // Late materialisation: only the pages that hold a requested row are decoded, and only the requested
    // values are copied out.
    public int[] Fetch(IEnumerable<int> rowIds, out int pagesTouched, out int valuesCopied)
    {
        var wanted = new SortedDictionary<int, List<int>>();
        foreach (int row in rowIds)
        {
            if (!wanted.TryGetValue(row / Pages[0].Count, out List<int>? list))
            {
                list = new List<int>();
                wanted[row / Pages[0].Count] = list;
            }
            list.Add(row);
        }
        var result = new List<int>();
        pagesTouched = 0;
        valuesCopied = 0;
        foreach (KeyValuePair<int, List<int>> entry in wanted)
        {
            ColumnPage page = Pages[entry.Key];
            int[] values = page.Decode();
            pagesTouched++;
            foreach (int row in entry.Value)
            {
                result.Add(values[row - page.RowStart]);
                valuesCopied++;
            }
        }
        return result.ToArray();
    }
}

public static class Solution
{
    private static ulong _seed = 0x2545F4914F6CDD1DUL;

    private static ulong NextRand()
    {
        _seed += 0x9E3779B97F4A7C15UL;
        ulong z = _seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static int NextInt(int bound) => (int)(NextRand() % (ulong)bound);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static int[] RandomColumn(int n, int range)
    {
        var values = new int[n];
        for (int i = 0; i < n; i++) values[i] = NextInt(range);
        return values;
    }

    private static int[] LowCardinality(int n, int distinct)
    {
        var values = new int[n];
        for (int i = 0; i < n; i++) values[i] = NextInt(distinct) * 10;
        return values;
    }

    private static int[] RunColumn(int n, int runLength, int distinct)
    {
        var values = new int[n];
        for (int i = 0; i < n; i++) values[i] = (i / runLength % distinct) * 5;
        return values;
    }

    private static void BitPacking()
    {
        for (int width = 1; width <= 16; width++)
        {
            var writer = new BitWriter();
            var expected = new List<int>();
            int limit = Math.Min(1 << width, 64);
            for (int value = 0; value < limit; value++)
            {
                writer.Write(value, width);
                expected.Add(value);
            }
            var reader = new BitReader(writer.ToArray());
            for (int i = 0; i < expected.Count; i++)
            {
                Check(reader.Read(width) == expected[i], "bit packing round trip width " + width + " value " + i);
            }
        }
        Check(ColumnCodec.BitsNeeded(0) == 1, "one bit is needed to store the single value 0");
        Check(ColumnCodec.BitsNeeded(1) == 1 && ColumnCodec.BitsNeeded(2) == 2, "bits needed follows ceil(log2(max+1))");
        Check(ColumnCodec.BitsNeeded(255) == 8 && ColumnCodec.BitsNeeded(256) == 9, "bits needed at byte boundaries");
    }

    private static void LayoutChoiceIsSmallest()
    {
        // Hand-computed: 8 values, 2 distinct -> 4 + 2 * 4 + ceil(8 * 1 / 8) = 13 bytes for the
        // dictionary layout, versus 32 plain bytes and 4 + 8 * 8 = 68 RLE bytes.
        var handMade = new[] { 1, 1, 2, 2, 1, 2, 1, 2 };
        ColumnPage handPage = ColumnCodec.EncodePage(handMade, 0);
        Check(handPage.Encoding == PageEncoding.Dictionary, "the tiny hand-made page picks dictionary encoding");
        Check(handPage.ByteLength == 13, "hand-computed dictionary size is 13 bytes, got " + handPage.ByteLength);
        Check(handPage.Min == 1 && handPage.Max == 2, "the zone map records min and max");
        Check(handPage.Decode().SequenceEqual(handMade), "the hand-made page round-trips");

        var low = new Column("low", LowCardinality(200, 8), 50);
        Check(low.Pages.All(p => p.Encoding == PageEncoding.Dictionary), "low cardinality picks dictionary encoding");

        // A page of 100 rows holding two 50-row runs costs 4 + 2 * 8 = 20 bytes as runs, versus
        // 4 + 2 * 4 + ceil(100 * 1 / 8) = 25 bytes as a dictionary, so RLE is the smallest layout.
        var runs = new Column("runs", RunColumn(200, 50, 4), 100);
        Check(runs.Pages.All(p => p.Encoding == PageEncoding.Rle), "long runs pick run-length encoding");
        Check(runs.Pages[0].ByteLength == 20, "hand-computed two-run page size is 20 bytes, got " + runs.Pages[0].ByteLength);

        var wide = new Column("wide", RandomColumn(200, 100000), 50);
        Check(wide.Pages.All(p => p.Encoding == PageEncoding.Plain), "high cardinality falls back to plain encoding");

        var constant = new Column("constant", Enumerable.Repeat(7, 120).ToArray(), 40);
        Check(constant.Pages.All(p => p.Encoding == PageEncoding.Rle), "a constant column is a single run per page");
        Check(constant.Pages[0].ByteLength == 4 + 8, "a single-run page costs 4 + 8 bytes, got " + constant.Pages[0].ByteLength);

        foreach (Column column in new[] { low, runs, constant })
        {
            Check(column.CompressedBytes < column.OriginalBytes,
                column.Name + ": compressed " + column.CompressedBytes + " < original " + column.OriginalBytes);
        }
        // Random high-cardinality pages are incompressible: they must fall back to the plain layout and
        // only pay the per-page zone map overhead on top of the raw values.
        Check(wide.CompressedBytes <= wide.OriginalBytes + wide.Pages.Count * 8,
            "incompressible data is stored as plain values plus zone maps");

        foreach (Column column in new[] { low, runs, wide, constant })
        {
            foreach (ColumnPage page in column.Pages)
            {
                int[] decoded = page.Decode();
                Check(decoded.Length == page.Count, column.Name + ": decoded page length");
                Check(decoded.Min() == page.Min && decoded.Max() == page.Max, column.Name + ": zone map matches the decoded values");
            }
        }
    }

    private static void PushdownCorrectnessAndSoundness()
    {
        var columns = new[]
        {
            new Column("random", RandomColumn(500, 1000), 64),
            new Column("low", LowCardinality(500, 6), 64),
            new Column("runs", RunColumn(500, 25, 8), 64),
        };

        int selective = 0;
        foreach (Column column in columns)
        {
            for (int trial = 0; trial < 20; trial++)
            {
                int low = NextInt(1000);
                int high = low + NextInt(50);
                column.ResetCounters();
                List<int> pushed = column.RangeScan(low, high, true);
                int pushedPages = column.PagesDecoded;
                column.ResetCounters();
                List<int> full = column.RangeScan(low, high, false);
                int fullPages = column.PagesDecoded;

                Check(pushed.SequenceEqual(full), column.Name + ": pushdown matches the brute-force scan");

                // Soundness: a page that was skipped must contain no matching value at all.
                var skipped = new HashSet<int>();
                for (int p = 0; p < column.Pages.Count; p++)
                {
                    ColumnPage page = column.Pages[p];
                    if (!page.Overlaps(low, high)) skipped.Add(p);
                }
                foreach (int row in pushed)
                {
                    Check(!skipped.Contains(row / column.Pages[0].Count), column.Name + ": a skipped page held a match");
                }

                Check(pushedPages <= fullPages, column.Name + ": pushdown decodes no more pages than a full scan");
                if (low > 0 && high < 100) selective++;
            }
        }
        Check(selective > 0, "the sweep really produced selective predicates");

        // Zone maps prune by page range, so the payoff shows up on clustered data: a column whose values
        // grow with the row id leaves each page with a narrow range, and a predicate over the top of the
        // domain then touches only the last page.
        var clustered = new Column("clustered", Enumerable.Range(0, 1000).Select(i => i * 100).ToArray(), 100);
        clustered.ResetCounters();
        List<int> top = clustered.RangeScan(90000, 100000, true);
        int pushedDecoded = clustered.PagesDecoded;
        clustered.ResetCounters();
        List<int> topFull = clustered.RangeScan(90000, 100000, false);
        int fullDecoded = clustered.PagesDecoded;
        Check(top.SequenceEqual(topFull), "pushdown on clustered data matches the brute-force scan");
        Check(top.Count == 100, "the predicate matches the top 100 values");
        Check(pushedDecoded == 1, "only the last page overlaps the predicate, decoded " + pushedDecoded);
        Check(pushedDecoded < fullDecoded, "a selective predicate decodes fewer pages ("
            + pushedDecoded + " < " + fullDecoded + ")");
        Console.WriteLine("pushdown: selective scan decoded " + pushedDecoded + " of " + fullDecoded + " pages");
    }

    private static void Boundaries()
    {
        var empty = new Column("empty", Array.Empty<int>(), 16);
        Check(empty.Pages.Count == 0, "an empty column has no pages");
        Check(empty.RangeScan(0, 100, true).Count == 0, "scanning an empty column yields nothing");
        Check(empty.CompressedBytes == 0, "an empty column occupies no bytes");

        var single = new Column("single", new[] { 42 }, 16);
        Check(single.Pages.Count == 1 && single.Pages[0].Count == 1, "a one-value column has one page");
        Check(single.RangeScan(42, 42, true).Count == 1, "an exact predicate finds the single value");
        Check(single.RangeScan(43, 50, true).Count == 0, "a non-matching predicate decodes and finds nothing");

        var matchAll = new Column("all", RandomColumn(100, 500), 25);
        matchAll.ResetCounters();
        Check(matchAll.RangeScan(0, 1000, true).Count == 100, "a predicate covering everything matches every row");
        Check(matchAll.PagesDecoded == matchAll.Pages.Count, "a predicate covering everything still decodes every page");
        Console.WriteLine("boundaries: empty, single-value, match-all and match-none predicates");

        var none = new Column("none", RandomColumn(100, 500), 25);
        none.ResetCounters();
        Check(none.RangeScan(100000, 100050, true).Count == 0, "a disjoint predicate matches nothing");
        Check(none.PagesDecoded == 0, "a disjoint predicate decodes no page at all");
    }

    private static void LateMaterialisation()
    {
        int[] payloadValues = RandomColumn(400, 100000);
        var payload = new Column("payload", payloadValues, 50);
        var key = new Column("key", Enumerable.Range(0, 400).Select(i => i / 4).ToArray(), 50);
        List<int> rows = key.RangeScan(0, 24, true);
        Check(rows.Count == 100, "the key column matches the first quarter of the rows");

        payload.ResetCounters();
        int[] fetched = payload.Fetch(rows, out int pagesTouched, out int valuesCopied);
        Check(fetched.Length == 100, "late materialisation fetches one value per matched row");
        Check(valuesCopied == 100, "late materialisation copies only the requested values");
        Check(pagesTouched == 2, "only the two pages holding rows 0..99 are touched, got " + pagesTouched);
        Check(pagesTouched < payload.Pages.Count, "late materialisation decodes fewer pages than the whole column");
        Check(payload.PagesDecoded == 2, "the payload column decoded exactly two pages");
        for (int i = 0; i < rows.Count; i++)
        {
            Check(fetched[i] == payloadValues[rows[i]], "materialised value matches the source row " + rows[i]);
        }

        payload.ResetCounters();
        int[] everyRow = payload.Fetch(Enumerable.Range(0, 400), out int allPages, out int allValues);
        Check(allValues == 400 && allPages == payload.Pages.Count, "fetching every row touches every page");
        Check(everyRow.SequenceEqual(payloadValues), "fetching every row reproduces the column");
        Console.WriteLine("late materialisation: 100 of 400 rows touched " + pagesTouched + " of " + payload.Pages.Count + " pages");
    }

    private static void Deterministic()
    {
        int[] data = RandomColumn(300, 700);
        var first = new Column("det", data, 40);
        var second = new Column("det2", data, 40);
        Check(first.Pages.Count == second.Pages.Count, "the same data produces the same page count");
        for (int i = 0; i < first.Pages.Count; i++)
        {
            Check(first.Pages[i].Encoding == second.Pages[i].Encoding, "the same data picks the same encoding");
            Check(first.Pages[i].Payload.SequenceEqual(second.Pages[i].Payload), "the same data produces identical payloads");
        }
        List<int> once = first.RangeScan(100, 200, true);
        List<int> twice = first.RangeScan(100, 200, true);
        Check(once.SequenceEqual(twice), "repeated scans return the same rows in the same order");
    }

    public static void Main()
    {
        BitPacking();
        LayoutChoiceIsSmallest();
        PushdownCorrectnessAndSoundness();
        Boundaries();
        LateMaterialisation();
        Deterministic();
        Console.WriteLine("PASS");
    }
}
