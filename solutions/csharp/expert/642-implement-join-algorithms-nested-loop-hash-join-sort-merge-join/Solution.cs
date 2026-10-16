// Task 642 - Implement join algorithms: nested-loop, hash join, sort-merge join. (Tier: Expert, CTCI Ch 14 - Databases)
//
// Four physical join operators over in-memory integer rows, all of them real implementations rather
// than models of one:
//   * NestedLoopJoin      - for every left row, scan the whole right relation;
//   * BlockNestedLoopJoin - the same, but the right relation is re-scanned only once per *block* of
//                           left rows, so the number of inner scans is ceil(|left| / blockSize) and
//                           the page traffic is |outer pages| + innerScans * |inner pages|
//                           (the classic |R| + ceil(|R|/b) * |S| block-nested-loop formula);
//   * HashJoin            - build a chained hash table on the right relation and probe it, keeping
//                           duplicate keys grouped so no match is lost;
//   * SortMergeJoin       - sort both inputs by the join key and merge them with duplicate groups.
//
// None of them deduplicates: a join over duplicates returns the full cross product per key, which is
// the invariant every cross-check below relies on. Simplified: the page/IO accounting is a formula
// over tuples-per-page constants instead of a real buffer manager, there is no spill to disk, no
// grace/hybrid hash partitioning and no parallel join.
//
// Verified: all four operators return the same multiset of joined rows on random inputs with heavy
// duplicate keys, the block-nested-loop scan counter matches the closed-form page formula, a bigger
// buffer never increases the page traffic, duplicate keys multiply, and the empty / no-match cases
// are exact.
using System;
using System.Collections.Generic;
using System.Linq;

internal sealed class Row
{
    public readonly int Key;
    public readonly int Payload;

    public Row(int key, int payload)
    {
        Key = key;
        Payload = payload;
    }
}

internal sealed class JoinedRow
{
    public readonly int Key;
    public readonly int LeftPayload;
    public readonly int RightPayload;

    public JoinedRow(int key, int leftPayload, int rightPayload)
    {
        Key = key;
        LeftPayload = leftPayload;
        RightPayload = rightPayload;
    }

    public string Canonical => Key + ":" + LeftPayload + ":" + RightPayload;
}

internal sealed class JoinStats
{
    public int InnerScans;
    public int PageReads;
    public long Comparisons;
}

internal static class Joins
{
    public const int RowsPerPage = 10;

    public static int Pages(int rows) => (rows + RowsPerPage - 1) / RowsPerPage;

    public static List<JoinedRow> NestedLoop(List<Row> left, List<Row> right, JoinStats? stats = null)
    {
        var output = new List<JoinedRow>();
        foreach (Row outer in left)
        {
            foreach (Row inner in right)
            {
                if (stats != null) stats.Comparisons++;
                if (outer.Key == inner.Key) output.Add(new JoinedRow(outer.Key, outer.Payload, inner.Payload));
            }
        }
        return output;
    }

    // The right relation is scanned once per block of left rows; both the scan count and the page
    // traffic are computed as the loops run, so the closed-form formula can be cross-checked.
    public static List<JoinedRow> BlockNestedLoop(List<Row> left, List<Row> right, int blockSize, JoinStats stats)
    {
        var output = new List<JoinedRow>();
        stats.InnerScans = 0;
        stats.PageReads = Pages(left.Count);
        for (int start = 0; start < left.Count; start += blockSize)
        {
            stats.InnerScans++;
            stats.PageReads += Pages(right.Count);
            for (int i = start; i < Math.Min(start + blockSize, left.Count); i++)
            {
                foreach (Row inner in right)
                {
                    stats.Comparisons++;
                    if (left[i].Key == inner.Key)
                    {
                        output.Add(new JoinedRow(left[i].Key, left[i].Payload, inner.Payload));
                    }
                }
            }
        }
        return output;
    }

    public static List<JoinedRow> HashJoin(List<Row> left, List<Row> right, JoinStats? stats = null)
    {
        var table = new Dictionary<int, List<Row>>();
        foreach (Row row in right)
        {
            if (!table.TryGetValue(row.Key, out List<Row> bucket))
            {
                bucket = new List<Row>();
                table[row.Key] = bucket;
            }
            bucket.Add(row);
        }

        var output = new List<JoinedRow>();
        foreach (Row probe in left)
        {
            if (stats != null) stats.Comparisons++;
            if (!table.TryGetValue(probe.Key, out List<Row> bucket)) continue;
            foreach (Row build in bucket)
            {
                output.Add(new JoinedRow(probe.Key, probe.Payload, build.Payload));
            }
        }
        return output;
    }

    public static List<JoinedRow> SortMerge(List<Row> left, List<Row> right, JoinStats? stats = null)
    {
        List<Row> sortedLeft = SortByKey(left);
        List<Row> sortedRight = SortByKey(right);
        var output = new List<JoinedRow>();
        int i = 0, j = 0;
        while (i < sortedLeft.Count && j < sortedRight.Count)
        {
            if (stats != null) stats.Comparisons++;
            int leftKey = sortedLeft[i].Key;
            int rightKey = sortedRight[j].Key;
            if (leftKey < rightKey)
            {
                i++;
                continue;
            }
            if (rightKey < leftKey)
            {
                j++;
                continue;
            }

            int leftEnd = i;
            while (leftEnd < sortedLeft.Count && sortedLeft[leftEnd].Key == leftKey) leftEnd++;
            int rightEnd = j;
            while (rightEnd < sortedRight.Count && sortedRight[rightEnd].Key == rightKey) rightEnd++;
            for (int a = i; a < leftEnd; a++)
            {
                for (int b = j; b < rightEnd; b++)
                {
                    output.Add(new JoinedRow(leftKey, sortedLeft[a].Payload, sortedRight[b].Payload));
                }
            }
            i = leftEnd;
            j = rightEnd;
        }
        return output;
    }

    // Deterministic, stable sort by key only (payload order inside a key is preserved).
    public static List<Row> SortByKey(List<Row> rows) => rows.OrderBy(r => r.Key).ToList();

    public static bool IsNonDecreasing(List<Row> rows)
    {
        for (int i = 1; i < rows.Count; i++)
        {
            if (rows[i - 1].Key > rows[i].Key) return false;
        }
        return true;
    }

    // Multiset comparison: sorting the canonical form makes the four operators comparable even though
    // hash join and sort-merge emit rows in different orders.
    public static string Canonical(List<JoinedRow> rows)
        => string.Join(";", rows.Select(r => r.Canonical).OrderBy(s => s, StringComparer.Ordinal));

    public static List<JoinedRow> Reference(List<Row> left, List<Row> right)
    {
        var counts = new Dictionary<int, int>();
        foreach (Row row in right)
        {
            counts.TryGetValue(row.Key, out int seen);
            counts[row.Key] = seen + 1;
        }
        var output = new List<JoinedRow>();
        foreach (Row outer in left)
        {
            if (counts.TryGetValue(outer.Key, out int matches) && matches > 0)
            {
                foreach (Row inner in right.Where(r => r.Key == outer.Key))
                {
                    output.Add(new JoinedRow(outer.Key, outer.Payload, inner.Payload));
                }
            }
        }
        return output;
    }
}

public static class Solution
{
    private static ulong _seed = 0x243F6A8885A308D3UL;

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

    private static List<Row> Build(int rows, int keySpace)
    {
        var result = new List<Row>();
        for (int i = 0; i < rows; i++) result.Add(new Row(NextInt(keySpace), i));
        return result;
    }

    private static void AllOperatorsAgree()
    {
        var stats = new JoinStats();
        for (int trial = 0; trial < 25; trial++)
        {
            List<Row> left = Build(40 + NextInt(40), 12);
            List<Row> right = Build(40 + NextInt(40), 12);
            string expected = Joins.Canonical(Joins.NestedLoop(left, right));
            Check(Joins.Canonical(Joins.HashJoin(left, right)) == expected, "hash join agrees with nested loop (trial " + trial + ")");
            Check(Joins.Canonical(Joins.SortMerge(left, right)) == expected, "sort-merge agrees with nested loop (trial " + trial + ")");
            Check(Joins.Canonical(Joins.BlockNestedLoop(left, right, 8, stats)) == expected, "block nested loop agrees (trial " + trial + ")");
            Check(Joins.Canonical(Joins.Reference(left, right)) == expected, "count-based reference agrees (trial " + trial + ")");
            Check(Joins.Canonical(Joins.NestedLoop(left, right)) == expected, "nested loop is deterministic (trial " + trial + ")");
        }
        Console.WriteLine("cross-check: nested loop, block nested loop, hash join, sort-merge and the count reference agree");
    }

    private static void DuplicatesMultiply()
    {
        var left = new List<Row>();
        for (int i = 0; i < 3; i++) left.Add(new Row(5, 100 + i));
        var right = new List<Row>();
        for (int i = 0; i < 2; i++) right.Add(new Row(5, 200 + i));
        right.Add(new Row(9, 999));

        int expected = 3 * 2;
        Check(Joins.NestedLoop(left, right).Count == expected, "duplicates: nested loop emits |L5| * |R5| rows");
        Check(Joins.HashJoin(left, right).Count == expected, "duplicates: hash join keeps every pair");
        Check(Joins.SortMerge(left, right).Count == expected, "duplicates: sort-merge expands the duplicate groups");
        Check(Joins.HashJoin(left, right).All(r => r.Key == 5), "duplicates: the unmatched right row is not joined");

        var skewed = new List<Row>();
        var skewedRight = new List<Row>();
        for (int i = 0; i < 30; i++) skewed.Add(new Row(1, i));
        for (int i = 0; i < 40; i++) skewedRight.Add(new Row(1, i));
        Check(Joins.HashJoin(skewed, skewedRight).Count == 1200, "skew: one hot key produces 30 * 40 rows");
        Check(Joins.SortMerge(skewed, skewedRight).Count == 1200, "skew: sort-merge handles the hot key");
    }

    private static void BlockNestedLoopAccounting()
    {
        var stats = new JoinStats();
        List<Row> left = Build(100, 5);
        List<Row> right = Build(70, 5);

        Joins.BlockNestedLoop(left, right, 25, stats);
        int expectedScans = (100 + 25 - 1) / 25;
        Check(stats.InnerScans == expectedScans, "block nested loop: ceil(100/25) = 4 inner scans, got " + stats.InnerScans);
        int innerPages = Joins.Pages(right.Count);
        Check(stats.PageReads == Joins.Pages(left.Count) + expectedScans * innerPages,
            "block nested loop: page traffic follows |R| + ceil(|R|/b) * |S|");

        int previousPages = int.MaxValue;
        foreach (int blockSize in new[] { 5, 10, 25, 50, 100, 200 })
        {
            var probe = new JoinStats();
            Joins.BlockNestedLoop(left, right, blockSize, probe);
            Check(probe.InnerScans == (100 + blockSize - 1) / blockSize, "inner scans for block size " + blockSize);
            Check(probe.PageReads <= previousPages, "a bigger buffer never increases page traffic (block " + blockSize + ")");
            previousPages = probe.PageReads;
        }
        Check(previousPages == Joins.Pages(left.Count) + Joins.Pages(right.Count), "one block covering everything reads each relation once");

        var whole = new JoinStats();
        Joins.BlockNestedLoop(left, right, 1000, whole);
        var single = new JoinStats();
        Joins.NestedLoop(left, right, single);
        Check(whole.Comparisons == single.Comparisons, "blocking does not change the number of tuple comparisons");
        Console.WriteLine("block nested loop: 4 inner scans at block 25, " + stats.PageReads + " page reads vs "
            + whole.PageReads + " with a large buffer");
    }

    private static void SortMergeInvariant()
    {
        List<Row> left = Build(60, 8);
        List<Row> right = Build(60, 8);
        List<Row> sortedLeft = Joins.SortByKey(left);
        List<Row> sortedRight = Joins.SortByKey(right);
        Check(Joins.IsNonDecreasing(sortedLeft) && Joins.IsNonDecreasing(sortedRight), "sort-merge sorts both inputs by the join key");
        Check(sortedLeft.Count == left.Count, "sorting preserves the row count");

        var payloads = left.OrderBy(r => r.Payload).Select(r => r.Payload).ToList();
        var sortedPayloads = sortedLeft.OrderBy(r => r.Payload).Select(r => r.Payload).ToList();
        Check(payloads.SequenceEqual(sortedPayloads), "sorting only reorders rows, it never loses or duplicates one");

        List<Row> alreadySorted = Joins.SortByKey(sortedLeft);
        Check(alreadySorted.Select(r => r.Payload).SequenceEqual(sortedLeft.Select(r => r.Payload)),
            "sorting an already sorted input is stable and idempotent");
    }

    private static void Boundaries()
    {
        var empty = new List<Row>();
        List<Row> rows = Build(20, 4);
        var stats = new JoinStats();

        Check(Joins.NestedLoop(empty, rows).Count == 0 && Joins.NestedLoop(rows, empty).Count == 0, "empty left or right yields no rows");
        Check(Joins.HashJoin(empty, rows).Count == 0 && Joins.HashJoin(rows, empty).Count == 0, "hash join handles empty inputs");
        Check(Joins.SortMerge(empty, empty).Count == 0, "sort-merge handles two empty inputs");
        Check(Joins.BlockNestedLoop(empty, rows, 8, stats).Count == 0, "block nested loop handles an empty left side");
        Check(stats.InnerScans == 0 && stats.PageReads == 0, "no inner scan happens for an empty left relation");

        var leftKeys = new List<Row> { new Row(1, 1), new Row(2, 2) };
        var rightKeys = new List<Row> { new Row(3, 3), new Row(4, 4) };
        Check(Joins.NestedLoop(leftKeys, rightKeys).Count == 0, "disjoint key sets yield no rows");
        Check(Joins.HashJoin(leftKeys, rightKeys).Count == 0, "hash join on disjoint keys yields no rows");
        Check(Joins.SortMerge(leftKeys, rightKeys).Count == 0, "sort-merge on disjoint keys yields no rows");

        var leftSingle = new List<Row> { new Row(7, 1) };
        var rightSingle = new List<Row> { new Row(7, 2) };
        Check(Joins.HashJoin(leftSingle, rightSingle).Count == 1, "single matching pair joins");
        Check(Joins.SortMerge(leftSingle, rightSingle)[0].LeftPayload == 1 && Joins.SortMerge(leftSingle, rightSingle)[0].RightPayload == 2,
            "the joined row carries both payloads from the correct sides");
    }

    private static void LargerScaleAgreement()
    {
        List<Row> left = Build(200, 30);
        List<Row> right = Build(200, 30);
        var stats = new JoinStats();
        List<JoinedRow> nested = Joins.NestedLoop(left, right);
        int hashCount = Joins.HashJoin(left, right).Count;
        int mergeCount = Joins.SortMerge(left, right).Count;
        int blockCount = Joins.BlockNestedLoop(left, right, 16, stats).Count;

        // Independent count: sum over keys of |L(k)| * |R(k)|.
        var leftCounts = new Dictionary<int, int>();
        var rightCounts = new Dictionary<int, int>();
        foreach (Row row in left)
        {
            leftCounts.TryGetValue(row.Key, out int seen);
            leftCounts[row.Key] = seen + 1;
        }
        foreach (Row row in right)
        {
            rightCounts.TryGetValue(row.Key, out int seen);
            rightCounts[row.Key] = seen + 1;
        }
        long expected = 0;
        foreach (KeyValuePair<int, int> entry in leftCounts)
        {
            if (rightCounts.TryGetValue(entry.Key, out int matches)) expected += (long)entry.Value * matches;
        }

        Check(nested.Count == expected, "nested loop produced every pair exactly once");
        Check(hashCount == expected && mergeCount == expected && blockCount == expected, "all operators agree with the counting formula");
        Console.WriteLine("200 x 200 join: " + expected + " result rows, page reads with block 16 = " + stats.PageReads);
    }

    public static void Main()
    {
        AllOperatorsAgree();
        DuplicatesMultiply();
        BlockNestedLoopAccounting();
        SortMergeInvariant();
        Boundaries();
        LargerScaleAgreement();
        Console.WriteLine("PASS");
    }
}
