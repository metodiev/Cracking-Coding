// Task 188 - Implement stable sort wrapper around existing unstable sort. (Tier: Beginner, CTCI Beyond — Software Engineering & Tooling)
using System;
using System.Collections.Generic;

public static class Solution
{
    private class Record
    {
        public int Key;
        public string Tag;
        public int Order;     // original position, -1 when not decorated

        public Record(int key, string tag, int order = -1)
        {
            Key = key;
            Tag = tag;
            Order = order;
        }
    }

    /// <summary>
    /// Array.Sort is an introsort: it sorts by key but gives no stability guarantee, which
    /// makes it exactly the kind of "unstable black box" this task wraps.
    /// </summary>
    private static List<Record> UnstableSort(List<Record> records)
    {
        var items = new Record[records.Count];
        records.CopyTo(items);
        Array.Sort(items, (a, b) => a.Key.CompareTo(b.Key));
        return new List<Record>(items);
    }

    /// <summary>Stable wrapper: decorate with the original index, sort (key, order) pairs, undecorate.</summary>
    private static List<Record> StableSort(List<Record> records)
    {
        var items = new Record[records.Count];
        for (int i = 0; i < records.Count; i++)
        {
            items[i] = new Record(records[i].Key, records[i].Tag, i);
        }
        Array.Sort(items, (a, b) => a.Key != b.Key ? a.Key.CompareTo(b.Key) : a.Order.CompareTo(b.Order));
        var result = new List<Record>();
        foreach (var item in items) result.Add(new Record(item.Key, item.Tag));
        return result;
    }

    private static string Tags(List<Record> records)
    {
        var parts = new List<string>();
        foreach (var record in records) parts.Add(record.Tag);
        return string.Join(",", parts);
    }

    private static string GroupTags(List<Record> records, int key)
    {
        var parts = new List<string>();
        foreach (var record in records)
        {
            if (record.Key == key) parts.Add(record.Tag);
        }
        return string.Join(",", parts);
    }

    private static bool KeysSorted(List<Record> records)
    {
        for (int i = 1; i < records.Count; i++)
        {
            if (records[i - 1].Key > records[i].Key) return false;
        }
        return true;
    }

    private static bool IsPermutation(List<Record> left, List<Record> right)
    {
        var a = new List<string>();
        var b = new List<string>();
        foreach (var record in left) a.Add(record.Key + ":" + record.Tag);
        foreach (var record in right) b.Add(record.Key + ":" + record.Tag);
        a.Sort();
        b.Sort();
        return string.Join("|", a) == string.Join("|", b);
    }

    public static void Main()
    {
        var records = new List<Record>
        {
            new Record(3, "a"), new Record(1, "x"), new Record(3, "b"), new Record(2, "y"),
            new Record(3, "c"), new Record(1, "w"), new Record(2, "z"),
        };

        Check(Tags(StableSort(records)) == "x,w,y,z,a,b,c", "ties keep original order");
        Check(Tags(records) == "a,x,b,y,c,w,z", "input untouched");

        // the black box itself is only guaranteed to order by key, not to keep ties in order
        var raw = UnstableSort(records);
        Check(KeysSorted(raw), "raw sort ordered by key");
        Check(IsPermutation(raw, records), "raw sort is a permutation");

        Check(Tags(StableSort(new List<Record> { new Record(1, "p"), new Record(1, "q"),
                new Record(1, "r"), new Record(1, "s") })) == "p,q,r,s", "all keys equal");
        Check(StableSort(new List<Record>()).Count == 0, "empty input");
        Check(Tags(StableSort(new List<Record> { new Record(9, "only") })) == "only", "single record");
        Check(Tags(StableSort(new List<Record> { new Record(0, "p"), new Record(-1, "q"),
                new Record(0, "r"), new Record(-1, "s") })) == "q,s,p,r", "negative keys");

        // larger batch: equal keys must stay in their original order
        var rng = new Random(41);
        var batch = new List<Record>();
        for (int i = 0; i < 120; i++) batch.Add(new Record(rng.Next(5), "tag-" + i.ToString("000")));
        var ordered = StableSort(batch);
        Check(KeysSorted(ordered), "batch keys sorted");
        Check(IsPermutation(ordered, batch), "batch is a permutation");
        for (int key = 0; key < 5; key++)
        {
            Check(GroupTags(ordered, key) == GroupTags(batch, key), "batch group " + key);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
