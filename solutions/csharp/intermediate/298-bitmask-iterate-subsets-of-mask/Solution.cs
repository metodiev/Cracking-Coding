// Task 298 - Bitmask: iterate subsets of a mask. (Tier: Intermediate, CTCI Ch 5)
using System;
using System.Collections.Generic;

public static class Solution
{
    /// <summary>All submasks of mask in descending order, including 0 (mask == 0 -> [0]).</summary>
    public static int[] EnumerateSubmasks(int mask)
    {
        var result = new List<int>();
        int sub = mask;
        while (true)
        {
            result.Add(sub);
            if (sub == 0) break;
            sub = (sub - 1) & mask;
        }
        return result.ToArray();
    }

    private static int PopCount(int value)
    {
        int count = 0;
        while (value != 0)
        {
            count += value & 1;
            value >>= 1;
        }
        return count;
    }

    private static int seed = 9090;

    public static void Main()
    {
        Check(Same(EnumerateSubmasks(0), new[] { 0 }), "mask 0");
        Check(Same(EnumerateSubmasks(1), new[] { 1, 0 }), "mask 1");
        Check(Same(EnumerateSubmasks(0b1011), new[] { 11, 10, 9, 8, 3, 2, 1, 0 }), "mask 0b1011");

        var all = EnumerateSubmasks(0b11111);
        Check(all.Length == 32 && all[0] == 31 && all[31] == 0, "mask 0b11111 bounds");
        bool descending = true;
        for (int i = 0; i + 1 < all.Length; i++) if (all[i] <= all[i + 1]) descending = false;
        Check(descending, "mask 0b11111 descending");

        var masks = new List<int> { 0, 1, 0b1011, 0b11111, 0b1000, 0b101010 };
        for (int i = 0; i < 40; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            masks.Add(seed % 4096);
        }

        foreach (int mask in masks)
        {
            var submasks = EnumerateSubmasks(mask);
            Check(submasks.Length == (1 << PopCount(mask)), "count for " + mask);
            var seen = new HashSet<int>();
            bool valid = true;
            for (int i = 0; i < submasks.Length; i++)
            {
                int sub = submasks[i];
                if (!seen.Add(sub)) valid = false;
                if ((sub & ~mask) != 0) valid = false;
                if (i + 1 < submasks.Length && submasks[i] <= submasks[i + 1]) valid = false;
            }
            if (submasks[0] != mask || submasks[submasks.Length - 1] != 0) valid = false;
            Check(valid, "properties for " + mask);
            // independent oracle: descending sweep over the integers below mask
            int expected = 0;
            for (int value = mask; value >= 0; value--) if ((value & ~mask) == 0) expected++;
            Check(submasks.Length == expected, "oracle for " + mask);
        }
        Console.WriteLine("PASS");
    }

    private static bool Same(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
