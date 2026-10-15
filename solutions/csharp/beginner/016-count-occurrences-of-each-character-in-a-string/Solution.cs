// Task 016 - Count occurrences of each character in a string. (Tier: Beginner, CTCI Ch 1)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static Dictionary<char, int> CharCounts(string s)
    {
        var counts = new Dictionary<char, int>();
        foreach (var c in s)
        {
            counts.TryGetValue(c, out var n);
            counts[c] = n + 1;
        }
        return counts;
    }

    public static void Main()
    {
        var c = CharCounts("aabbbc");
        Check(c['a'] == 2 && c['b'] == 3 && c['c'] == 1, "aabbbc");
        Check(CharCounts("").Count == 0, "empty");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
