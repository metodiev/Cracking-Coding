// Task 345 - RandomizedSet: insert/delete/getRandom in O(1). (Tier: Intermediate, CTCI Ch 7)
using System;
using System.Collections.Generic;

public class RandomizedSet
{
    private readonly List<int> items = new List<int>();
    private readonly Dictionary<int, int> index = new Dictionary<int, int>();
    private long rng;

    public RandomizedSet(long seed)
    {
        rng = seed;
    }

    public bool Insert(int value)
    {
        if (index.ContainsKey(value)) return false;
        index[value] = items.Count;
        items.Add(value);
        return true;
    }

    public bool Remove(int value)
    {
        if (!index.TryGetValue(value, out int at)) return false;
        int last = items[items.Count - 1];
        items[at] = last;                 // move the last element into the hole
        index[last] = at;
        items.RemoveAt(items.Count - 1);
        index.Remove(value);
        return true;
    }

    public int GetRandom()
    {
        if (items.Count == 0) throw new InvalidOperationException("getRandom on an empty set");
        rng = (rng * 75 + 74) % 65537;
        return items[(int)(rng % items.Count)];
    }

    public int Size => items.Count;

    public List<int> ToList() => new List<int>(items);

    public bool Validate()
    {
        if (items.Count != index.Count) return false;
        foreach (KeyValuePair<int, int> entry in index)
        {
            if (items[entry.Value] != entry.Key) return false;
        }
        return true;
    }
}

public static class Solution
{
    public static void Main()
    {
        var set = new RandomizedSet(2024);
        Check(set.Size == 0, "starts empty");

        int[] values = { 5, 3, 9, 1, 7, 5 };
        bool[] expected = { true, true, true, true, true, false };
        for (int i = 0; i < values.Length; i++)
        {
            Check(set.Insert(values[i]) == expected[i], "insert " + values[i]);
        }
        Check(set.Remove(3), "remove present value");
        Check(!set.Remove(42), "remove missing value returns false");
        Check(set.Insert(3), "re-insert after removal");
        Check(string.Join(",", set.ToList()) == "5,7,9,1,3", "swap-remove layout");
        Check(set.Size == 5, "size 5");
        Check(set.Validate(), "array and map agree");

        // deterministic LCG draws - identical in every language
        int[] draws = { 5, 1, 3, 3, 3, 1, 1, 1 };
        for (int i = 0; i < draws.Length; i++)
        {
            Check(set.GetRandom() == draws[i], "draw " + i + " expected " + draws[i]);
        }

        var empty = new RandomizedSet(1);
        try
        {
            empty.GetRandom();
            throw new Exception("FAILED: expected InvalidOperationException");
        }
        catch (InvalidOperationException)
        {
            // documented behaviour
        }

        var bulk = new RandomizedSet(98765);
        for (int v = 1; v <= 50; v++) Check(bulk.Insert(v), "bulk insert " + v);
        for (int v = 1; v <= 50; v++) Check(!bulk.Insert(v), "bulk duplicate " + v);
        for (int v = 2; v <= 50; v += 2) Check(bulk.Remove(v), "bulk remove " + v);
        for (int v = 2; v <= 50; v += 2) Check(!bulk.Remove(v), "bulk re-remove " + v);
        Check(bulk.Size == 25, "25 odds remain");
        var remaining = bulk.ToList();
        remaining.Sort();
        var odds = new List<int>();
        for (int v = 1; v <= 50; v += 2) odds.Add(v);
        Check(string.Join(",", remaining) == string.Join(",", odds), "only odds remain");
        Check(bulk.Validate(), "bulk layout consistent");
        var oddsSet = new HashSet<int>(odds);
        var drawn = new HashSet<int>();
        for (int i = 0; i < 500; i++)
        {
            int value = bulk.GetRandom();
            Check(oddsSet.Contains(value), "draw is one of the odds");
            drawn.Add(value);
        }
        Check(drawn.Count == oddsSet.Count, "500 deterministic draws cover all odds");

        var small = new RandomizedSet(7);
        small.Insert(10);
        small.Insert(20);
        small.Insert(30);
        Check(small.Remove(10), "swap victim removed");
        Check(string.Join(",", small.ToList()) == "30,20", "layout after victim removal");
        Check(small.Remove(20), "last element removed");
        Check(string.Join(",", small.ToList()) == "30", "layout after last removal");
        Check(small.Remove(30), "final removal");
        Check(small.ToList().Count == 0, "empty again");
        Check(small.Validate(), "small layout consistent");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
